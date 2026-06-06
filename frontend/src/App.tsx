import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type {
  EnvironmentConfig,
  EpisodeResponse,
  EpisodeSummary,
  RaceInclude,
  SystemStatus,
} from "./api";
import { ReplayCanvas } from "./components/ReplayCanvas";
import { GuidedSteps } from "./components/GuidedSteps";
import { ExplainerCard } from "./components/ExplainerCard";
import { ImprovementLab, type ScoreEntry } from "./components/ImprovementLab";
import { SensorPanel } from "./components/SensorPanel";

type Diff = "easy" | "medium" | "hard";

type EnvForm = {
  difficulty: number;
  room_width: number;
  room_height: number;
  num_obstacles: number;
  wind_strength: number;
  sensor_noise: number;
  seed: number | null;
  max_steps: number;
  model_name: string;
};

const DEFAULT_FORM: EnvForm = {
  difficulty: 1,
  room_width: 10,
  room_height: 10,
  num_obstacles: 3,
  wind_strength: 0,
  sensor_noise: 0,
  seed: null,
  max_steps: 250,
  model_name: "drone_policy_v1",
};

const DIFF_PRESETS: Record<Diff, Partial<EnvForm>> = {
  easy: { difficulty: 1, num_obstacles: 3, room_width: 10, room_height: 10 },
  medium: { difficulty: 3, num_obstacles: 5, room_width: 10, room_height: 10 },
  hard: { difficulty: 6, num_obstacles: 8, room_width: 12, room_height: 12 },
};

export function App() {
  const [diff, setDiff] = useState<Diff>("easy");
  const [form, setForm] = useState<EnvForm>(DEFAULT_FORM);
  const [env, setEnv] = useState<EnvironmentConfig | null>(null);
  const [episode, setEpisode] = useState<EpisodeResponse | null>(null);
  const [overlay, setOverlay] = useState<EpisodeResponse | null>(null);
  const [overlayLLM, setOverlayLLM] = useState<EpisodeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<{ msg: string; ok: boolean }[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [episodeList, setEpisodeList] = useState<EpisodeSummary[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [datasetName] = useState("imitation_v1");
  const [scoreLog, setScoreLog] = useState<ScoreEntry[]>([]);
  const [iterCount, setIterCount] = useState(0);
  const [baselineRate, setBaselineRate] = useState<number | null>(null);
  const [baselineDiff, setBaselineDiff] = useState<number | null>(null);
  const [includeLLM, setIncludeLLM] = useState(true);
  const busyRef = useRef(false);

  const setBusyState = (b: boolean) => {
    busyRef.current = b;
    setBusy(b);
  };

  const append = useCallback((msg: string, ok = true) => {
    setLog((l) => [...l.slice(-30), { msg, ok }]);
  }, []);

  const refreshEpisodes = useCallback(async () => {
    try {
      setEpisodeList(await api.listEpisodes());
    } catch (e: any) {
      append(`list err: ${e.message}`, false);
    }
  }, [append]);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.systemStatus());
    } catch {}
  }, []);

  useEffect(() => {
    refreshEpisodes();
    refreshStatus();
    setForm((f) => ({ ...f, ...DIFF_PRESETS.easy }));
  }, [refreshEpisodes, refreshStatus]);

  useEffect(() => {
    const id = setInterval(refreshStatus, busy ? 1000 : 5000);
    return () => clearInterval(id);
  }, [busy, refreshStatus]);

  const generateEnvRequest = (overrides: Partial<EnvForm> = {}) => ({
    difficulty: overrides.difficulty ?? form.difficulty,
    room_width: overrides.room_width ?? form.room_width,
    room_height: overrides.room_height ?? form.room_height,
    num_obstacles: overrides.num_obstacles ?? form.num_obstacles,
    wind_strength: overrides.wind_strength ?? form.wind_strength,
    sensor_noise: overrides.sensor_noise ?? form.sensor_noise,
    seed: overrides.seed ?? form.seed,
  });

  const ensureEnv = async (): Promise<EnvironmentConfig | null> => {
    if (env) return env;
    const e = await api.generateEnv(generateEnvRequest(DIFF_PRESETS[diff]));
    setEnv(e);
    return e;
  };

  const clearOverlays = () => {
    setOverlay(null);
    setOverlayLLM(null);
  };

  const onBuildWorld = async () => {
    setBusyState(true);
    try {
      const e = await api.generateEnv(generateEnvRequest(DIFF_PRESETS[diff]));
      setEnv(e);
      setEpisode(null);
      clearOverlays();
      append(`world built (${diff})`);
    } catch (e: any) {
      append(`env err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onDifficultyChange = (d: Diff) => {
    setDiff(d);
    setForm((f) => ({ ...f, ...DIFF_PRESETS[d] }));
  };

  const onRunRuleBased = async () => {
    const useEnv = await ensureEnv();
    if (!useEnv) return;
    setBusyState(true);
    clearOverlays();
    try {
      const ep = await api.runEpisode(useEnv, "heuristic", form.max_steps, form.seed);
      setEpisode(ep);
      append(
        `teacher: ${ep.success ? "reached target" : ep.collision ? "crashed" : "ran out of time"} in ${ep.steps} steps`,
      );
      await refreshEpisodes();
    } catch (e: any) {
      append(`run err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onCollectFlights = async () => {
    setBusyState(true);
    try {
      const r = await api.generateDataset({
        num_episodes: 200,
        difficulty: form.difficulty,
        room_width: form.room_width,
        room_height: form.room_height,
        num_obstacles: form.num_obstacles,
        max_steps: form.max_steps,
        seed: 42,
        dataset_name: datasetName,
      });
      append(
        `collected ${r.num_samples.toLocaleString()} examples (${(r.success_rate * 100).toFixed(0)}% of demo flights succeeded)`,
      );
      await refreshStatus();
    } catch (e: any) {
      append(`dataset err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onTrainNetwork = async () => {
    setBusyState(true);
    try {
      const r = await api.trainPolicy({
        dataset_name: datasetName,
        epochs: 10,
        batch_size: 64,
        learning_rate: 1e-3,
        hidden_size: 64,
        model_name: form.model_name,
      });
      append(
        `network trained: ${(r.test_accuracy * 100).toFixed(1)}% accuracy matching the teacher`,
      );
      await refreshStatus();
    } catch (e: any) {
      append(`train err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  // === Race (multi-pilot on shared world) =========================
  const onRace = async () => {
    const useEnv = await ensureEnv();
    if (!useEnv) return;
    setBusyState(true);
    try {
      const include: RaceInclude[] = ["heuristic", "trained"];
      if (includeLLM && status?.llm_available) include.push("llm");
      const r = await api.race(
        useEnv,
        form.model_name,
        include,
        form.max_steps,
        60, // cap LLM steps — each step is an API call
        form.seed,
      );
      setEpisode(r.heuristic);
      setOverlay(r.trained);
      setOverlayLLM(r.llm);
      const parts: string[] = [];
      if (r.heuristic) parts.push(`teacher ${r.heuristic.success ? "✓" : "✗"} ${r.heuristic.steps}s`);
      if (r.trained) parts.push(`network ${r.trained.success ? "✓" : "✗"} ${r.trained.steps}s`);
      if (r.llm) parts.push(`llm ${r.llm.success ? "✓" : "✗"} ${r.llm.steps}s`);
      append(`race: ${parts.join(" · ")}`);
      await refreshEpisodes();
    } catch (e: any) {
      append(`race err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  // === Improvement lab handlers ===================================
  const BC_ROOT = "drone_policy_v1";
  const PRACTICED = "drone_policy_v1_practiced";
  const formDifficulty = form.difficulty;

  const recordScore = (label: string, success_rate: number, n: number) => {
    setScoreLog((xs) => [
      ...xs,
      { label, success_rate, difficulty: formDifficulty, n, at: Date.now() },
    ]);
  };

  const autoRaceWithModel = async (modelName: string) => {
    const useEnv = env;
    if (!useEnv) return;
    try {
      const include: RaceInclude[] = ["heuristic", "trained"];
      if (includeLLM && status?.llm_available) include.push("llm");
      const r = await api.race(useEnv, modelName, include, form.max_steps, 60, form.seed);
      setEpisode(r.heuristic);
      setOverlay(r.trained);
      setOverlayLLM(r.llm);
    } catch (e: any) {
      append(`auto-race err: ${e.message}`, false);
    }
  };

  const onScoreCurrent = async () => {
    setBusyState(true);
    try {
      if (baselineRate == null || baselineDiff !== formDifficulty) {
        const base = await api.evaluateBaseline("heuristic", 20, formDifficulty);
        setBaselineRate(base.success_rate);
        setBaselineDiff(formDifficulty);
        append(
          `teacher on 20 ${diff} worlds: ${(base.success_rate * 100).toFixed(0)}% (the BC ceiling)`,
        );
      }
      const r = await api.evaluateModel(form.model_name, 20, formDifficulty);
      const label =
        scoreLog.length === 0 ? "Baseline (after step 4)" : `After try #${iterCount}`;
      recordScore(label, r.success_rate, r.num_episodes);
      append(`scored ${form.model_name}: ${(r.success_rate * 100).toFixed(0)}%`);
      await refreshStatus();
    } catch (e: any) {
      append(`score err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onMoreDemos = async () => {
    setBusyState(true);
    try {
      const more = await api.generateDataset({
        num_episodes: 400,
        difficulty: form.difficulty,
        room_width: form.room_width,
        room_height: form.room_height,
        num_obstacles: form.num_obstacles,
        max_steps: form.max_steps,
        seed: 100 + iterCount,
        dataset_name: datasetName,
        append: true,
      });
      append(`dataset now ${more.num_samples.toLocaleString()} examples`);
      const tr = await api.trainPolicy({
        dataset_name: datasetName,
        epochs: 10,
        batch_size: 64,
        learning_rate: 1e-3,
        hidden_size: 64,
        model_name: BC_ROOT,
      });
      append(`retrained ${BC_ROOT}: ${(tr.test_accuracy * 100).toFixed(1)}% demo accuracy`);
      setForm((f) => ({ ...f, model_name: BC_ROOT }));
      setIterCount((i) => i + 1);
      await refreshStatus();
      const r = await api.evaluateModel(BC_ROOT, 20, formDifficulty);
      recordScore(`+more demos #${iterCount + 1}`, r.success_rate, r.num_episodes);
      await autoRaceWithModel(BC_ROOT);
    } catch (e: any) {
      append(`more-demos err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onPractice = async () => {
    setBusyState(true);
    try {
      const practicedExists = (status?.models ?? []).some((m) => m.name === PRACTICED);
      const warmFrom = practicedExists ? PRACTICED : BC_ROOT;
      const r = await api.trainRL({
        episodes: 500,
        learning_rate: 3e-4,
        gamma: 0.97,
        hidden_size: 64,
        difficulty: form.difficulty,
        room_width: form.room_width,
        room_height: form.room_height,
        num_obstacles: form.num_obstacles,
        max_steps: form.max_steps,
        model_name: PRACTICED,
        warm_start_from: warmFrom,
        seed: iterCount,
      });
      append(
        `practiced 500 episodes (from ${warmFrom}) → ${PRACTICED}: rolling succ ${(r.final_success_rate * 100).toFixed(0)}%`,
      );
      setForm((f) => ({ ...f, model_name: PRACTICED }));
      setIterCount((i) => i + 1);
      await refreshStatus();
      const ev = await api.evaluateModel(PRACTICED, 20, formDifficulty);
      recordScore(`+practice #${iterCount + 1}`, ev.success_rate, ev.num_episodes);
      await autoRaceWithModel(PRACTICED);
    } catch (e: any) {
      append(`practice err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onCurriculumBC = async () => {
    setBusyState(true);
    try {
      const r1 = await api.generateDataset({
        num_episodes: 300,
        difficulties: [1, 2, 3],
        dataset_name: datasetName,
        append: false,
      });
      append(
        `mixed-difficulty demos · ${r1.num_samples.toLocaleString()} examples · ${(r1.success_rate * 100).toFixed(0)}% demo-success`,
      );
      const tr = await api.trainPolicy({
        dataset_name: datasetName,
        epochs: 10,
        batch_size: 64,
        learning_rate: 1e-3,
        hidden_size: 64,
        model_name: BC_ROOT,
      });
      append(`retrained ${BC_ROOT}: ${(tr.test_accuracy * 100).toFixed(1)}% demo accuracy`);
      setForm((f) => ({ ...f, model_name: BC_ROOT }));
      setIterCount((i) => i + 1);
      await refreshStatus();
      const ev = await api.evaluateModel(BC_ROOT, 20, formDifficulty);
      recordScore(`+curriculum #${iterCount + 1}`, ev.success_rate, ev.num_episodes);
      await autoRaceWithModel(BC_ROOT);
    } catch (e: any) {
      append(`curriculum err: ${e.message}`, false);
    } finally {
      setBusyState(false);
    }
  };

  const onResetExperiment = () => {
    setScoreLog([]);
    setIterCount(0);
    setBaselineRate(null);
    setBaselineDiff(null);
  };

  const taskName = status?.current_task?.name;
  const TASK_LABELS: Record<string, string> = {
    generate_dataset: "Collecting demo flights",
    train_policy: "Training the network",
    train_rl: "Reinforcement training",
    run_episode: "Flying",
    compare: "Racing pilots",
    race: "Racing pilots",
    evaluate_model: "Evaluating in the sim",
    evaluate_baseline: "Evaluating teacher",
  };

  const hasEnv = env != null;
  const hasHeuristicRun =
    episode?.summary.agent_type === "heuristic" ||
    episodeList.some((e) => e.agent_type === "heuristic");
  const hasDataset = (status?.datasets.length ?? 0) > 0;
  const hasTrainedModel = (status?.models ?? []).some(
    (m) => m.method === "behavior_cloning",
  );
  const hasComparison = overlay != null || overlayLLM != null;

  return (
    <div className="app simple">
      <div className="header">
        <h1>DroneGym</h1>
        <span className="tag">an introductory tour of building a sim + ML loop</span>
        <span style={{ flex: 1 }} />
        {taskName ? (
          <span style={{ fontSize: 12 }}>
            <span style={{ color: "var(--warn)" }}>⏳ {TASK_LABELS[taskName] ?? taskName}</span>{" "}
            <span style={{ color: "var(--muted)" }}>{status?.current_task?.detail}</span>{" "}
            <span style={{ color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
              {status?.current_task?.elapsed.toFixed(1)}s
            </span>
          </span>
        ) : (
          <span style={{ fontSize: 12, color: "var(--muted)" }}>● idle</span>
        )}
      </div>

      <div className="sidebar">
        <GuidedSteps
          difficulty={diff}
          busy={busy}
          hasEnv={hasEnv}
          hasHeuristicRun={hasHeuristicRun}
          hasDataset={hasDataset}
          hasTrainedModel={hasTrainedModel}
          hasComparison={hasComparison}
          episode={episode}
          status={status}
          onDifficultyChange={onDifficultyChange}
          onBuildWorld={onBuildWorld}
          onRunRuleBased={onRunRuleBased}
          onCollectFlights={onCollectFlights}
          onTrainNetwork={onTrainNetwork}
          onCompare={onRace}
          llmAvailable={status?.llm_available ?? false}
          llmModel={status?.llm_model ?? null}
          includeLLM={includeLLM}
          onToggleLLM={setIncludeLLM}
          laboratory={
            <ImprovementLab
              busy={busy}
              scores={scoreLog}
              baselineRate={baselineRate}
              hasTrainedModel={hasTrainedModel}
              activeModel={form.model_name}
              isPracticed={form.model_name.endsWith("_practiced")}
              onScoreCurrent={onScoreCurrent}
              onMoreDemos={onMoreDemos}
              onPractice={onPractice}
              onCurriculumBC={onCurriculumBC}
              onResetExperiment={onResetExperiment}
            />
          }
        />
        <div className="section">
          <h2>Log</h2>
          <div className="log">
            {log
              .slice()
              .reverse()
              .map((l, i) => (
                <div key={i} className={l.ok ? "ok" : "err"}>
                  {l.msg}
                </div>
              ))}
          </div>
        </div>
      </div>
      <div className="main">
        <ReplayCanvas
          episode={episode}
          env={env}
          overlay={overlay}
          overlayLLM={overlayLLM}
          onStepChange={setCurrentStep}
          showSensors={!overlay && !overlayLLM}
        />
      </div>
      <div className="right">
        <ExplainerCard
          episode={episode}
          overlay={overlay}
          overlayLLM={overlayLLM}
          hasEnv={hasEnv}
          hasTrainedModel={hasTrainedModel}
        />
        {episode && !overlay && !overlayLLM && (
          <SensorPanel episode={episode} step={currentStep} />
        )}
        {episode && (
          <div className="section">
            <h2>This flight</h2>
            <div className="metrics-grid">
              <span className="label">Outcome</span>
              <span className="val">
                {episode.success
                  ? "reached target"
                  : episode.collision
                    ? "crashed"
                    : "ran out of time"}
              </span>
              <span className="label">Steps</span>
              <span className="val">{episode.steps}</span>
              <span className="label">Final distance</span>
              <span className="val">{episode.summary.final_distance_to_target.toFixed(1)}</span>
              <span className="label">Path length</span>
              <span className="val">{episode.summary.path_length.toFixed(1)}</span>
            </div>
          </div>
        )}
        {hasTrainedModel && (
          <div className="section">
            <h2>Network's progress</h2>
            {status?.models
              .filter((m) => m.method === "behavior_cloning")
              .slice(0, 1)
              .map((m) => (
                <div key={m.name} style={{ fontSize: 12, lineHeight: 1.6 }}>
                  <div>
                    Accuracy on held-out demos:{" "}
                    <strong>{((m.test_accuracy ?? 0) * 100).toFixed(1)}%</strong>
                  </div>
                  {m.sim_eval_success != null && (
                    <div>
                      Success in {m.sim_eval_n} fresh rooms:{" "}
                      <strong>{((m.sim_eval_success ?? 0) * 100).toFixed(0)}%</strong>
                    </div>
                  )}
                  {baselineRate != null && m.sim_eval_success != null && (
                    <div
                      style={{
                        marginTop: 6,
                        padding: 6,
                        borderRadius: 4,
                        background:
                          m.sim_eval_success >= baselineRate
                            ? "rgba(89,224,139,0.10)"
                            : "rgba(255,209,102,0.10)",
                        border: `1px solid ${m.sim_eval_success >= baselineRate ? "rgba(89,224,139,0.4)" : "rgba(255,209,102,0.4)"}`,
                      }}
                    >
                      <div>
                        Rule-based pilot on same difficulty:{" "}
                        <strong>{(baselineRate * 100).toFixed(0)}%</strong>
                      </div>
                      <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 2 }}>
                        {m.sim_eval_success >= baselineRate
                          ? "Network ≥ teacher. BC's ceiling reached — practice can push further."
                          : `Network is ${Math.round((baselineRate - m.sim_eval_success) * 100)} points below teacher.`}
                      </div>
                    </div>
                  )}
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
