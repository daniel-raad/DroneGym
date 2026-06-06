import { useMemo, useState } from "react";
import { api } from "../api";
import type { SystemStatus } from "../api";
import { upsertCustomPilot, slugify } from "../arcade/customPilots";

// Studio for promoting a trained model into a named pilot the arcade can race.
// Five sequenced steps: name → demos → train (BC) → score → practice (RL) →
// graduate. Each step is gated on the previous so the user always knows the
// next action. Existing dashboard panels still work; this is just the "happy
// path" condensed into one card.

type Props = {
  status: SystemStatus | null;
  onStatusChange: () => Promise<void> | void;
  onGraduated: (pilotId: string) => void;
};

const FACES = ["🦊", "🐱", "🐰", "🐻", "🦉", "🐯", "🐼", "🦅", "🐙", "🐝", "🦋", "🐢"];
const HUES = [
  { label: "cyan", value: 195 },
  { label: "pink", value: 320 },
  { label: "gold", value: 45 },
  { label: "lime", value: 130 },
  { label: "violet", value: 270 },
  { label: "coral", value: 15 },
];

type Stage = "identity" | "demos" | "train" | "score" | "practice" | "graduate" | "done";

// Default training params chosen to match the dashboard's "easy preset" so
// the rookie has a reasonable shot at beating the heuristic in 30-60s.
const ROOM = { difficulty: 1, room_width: 10, room_height: 10, num_obstacles: 3, max_steps: 250 };
const DEMO_EPISODES = 200;
const EVAL_EPISODES = 20;
const RL_EPISODES = 500;

type RayPreset = { rays: number; label: string; sub: string };
const RAY_PRESETS: RayPreset[] = [
  { rays: 3, label: "3", sub: "cheap drone" },
  { rays: 8, label: "8", sub: "lidar lite" },
  { rays: 16, label: "16", sub: "Skydio-class" },
  { rays: 32, label: "32", sub: "dense lidar" },
];
const DEFAULT_RAYS = 16;

export function PilotStudio({ status, onStatusChange, onGraduated }: Props) {
  const [name, setName] = useState("");
  const [face, setFace] = useState(FACES[0]);
  const [hue, setHue] = useState(HUES[0].value);
  const [stage, setStage] = useState<Stage>("identity");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [demoSamples, setDemoSamples] = useState<number | null>(null);
  const [bcAccuracy, setBcAccuracy] = useState<number | null>(null);
  const [evalSuccess, setEvalSuccess] = useState<number | null>(null);
  const [practiced, setPracticed] = useState(false);
  const [numRays, setNumRays] = useState<number>(DEFAULT_RAYS);

  const modelName = useMemo(() => (name ? slugify(name) : ""), [name]);
  const datasetName = useMemo(() => (modelName ? `${modelName}_dataset` : ""), [modelName]);

  const task = status?.current_task;

  const guard = (next: Stage, fn: () => Promise<void>) => async () => {
    if (busy) return;
    setErr(null);
    setBusy(true);
    try {
      await fn();
      setStage(next);
    } catch (e: any) {
      setErr(e.message ?? "something went wrong");
    } finally {
      setBusy(false);
    }
  };

  const onName = () => {
    if (!modelName) return;
    setStage("demos");
  };

  const onCollect = guard("train", async () => {
    const r = await api.generateDataset({
      num_episodes: DEMO_EPISODES,
      difficulty: ROOM.difficulty,
      room_width: ROOM.room_width,
      room_height: ROOM.room_height,
      num_obstacles: ROOM.num_obstacles,
      max_steps: ROOM.max_steps,
      seed: 42,
      dataset_name: datasetName,
      append: false,
      num_rays: numRays,
    });
    setDemoSamples(r.num_samples);
    await onStatusChange();
  });

  const onTrain = guard("score", async () => {
    const r = await api.trainPolicy({
      dataset_name: datasetName,
      epochs: 10,
      batch_size: 64,
      learning_rate: 1e-3,
      hidden_size: 64,
      model_name: modelName,
      num_rays: numRays,
    });
    setBcAccuracy(r.test_accuracy);
    await onStatusChange();
  });

  const onScore = guard("practice", async () => {
    const r = await api.evaluateModel(modelName, EVAL_EPISODES, ROOM.difficulty);
    setEvalSuccess(r.success_rate);
    await onStatusChange();
  });

  const onPractice = guard("graduate", async () => {
    const r = await api.trainRL({
      episodes: RL_EPISODES,
      learning_rate: 3e-4,
      gamma: 0.97,
      hidden_size: 64,
      difficulty: ROOM.difficulty,
      room_width: ROOM.room_width,
      room_height: ROOM.room_height,
      num_obstacles: ROOM.num_obstacles,
      max_steps: ROOM.max_steps,
      model_name: modelName,
      warm_start_from: modelName,
      seed: 7,
      num_rays: numRays,
    });
    // Score the practiced model so the pilot card reflects the new reality.
    const ev = await api.evaluateModel(modelName, EVAL_EPISODES, ROOM.difficulty);
    setEvalSuccess(ev.success_rate);
    setPracticed(true);
    await onStatusChange();
    void r;
  });

  const onSkipPractice = () => setStage("graduate");

  const onGraduate = () => {
    if (!modelName) return;
    const id = `custom:${modelName}`;
    upsertCustomPilot({
      id,
      modelName,
      name: name.trim() || modelName,
      face,
      hue,
      createdAt: Date.now(),
      practiced,
      bcAccuracy,
      evalSuccess,
    });
    setStage("done");
    onGraduated(id);
  };

  const reset = () => {
    setStage("identity");
    setDemoSamples(null);
    setBcAccuracy(null);
    setEvalSuccess(null);
    setPracticed(false);
    setNumRays(DEFAULT_RAYS);
    setErr(null);
  };

  return (
    <div className="section studio">
      <div className="studio-head">
        <h2 style={{ margin: 0 }}>Pilot Studio</h2>
        <span className="studio-sub">train a rookie → send to the arcade</span>
      </div>

      {/* Identity ----------------------------------------------------- */}
      <Step
        n={1}
        title="Name your pilot"
        active={stage === "identity"}
        done={stage !== "identity"}
        summary={stage !== "identity" ? `${face} ${name}` : null}
      >
        <div className="studio-identity">
          <input
            className="studio-input"
            placeholder="e.g. Zoom"
            value={name}
            onChange={(e) => setName(e.target.value.slice(0, 22))}
            maxLength={22}
          />
          <div className="studio-pickers">
            <div className="studio-picker">
              <div className="studio-picker-label">FACE</div>
              <div className="studio-face-row">
                {FACES.map((f) => (
                  <button
                    key={f}
                    className={`studio-face-btn ${face === f ? "on" : ""}`}
                    onClick={() => setFace(f)}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>
            <div className="studio-picker">
              <div className="studio-picker-label">COLOR</div>
              <div className="studio-hue-row">
                {HUES.map((h) => (
                  <button
                    key={h.value}
                    className={`studio-hue-btn ${hue === h.value ? "on" : ""}`}
                    style={{ background: `hsl(${h.value}, 70%, 60%)` }}
                    onClick={() => setHue(h.value)}
                    title={h.label}
                  />
                ))}
              </div>
            </div>
          </div>
          <button className="studio-cta" disabled={!modelName} onClick={onName}>
            Begin training →
          </button>
          {modelName && (
            <div className="studio-meta">
              model file: <code>{modelName}.pt</code>
            </div>
          )}
        </div>
      </Step>

      {/* Demos -------------------------------------------------------- */}
      <Step
        n={2}
        title="Collect demo flights"
        active={stage === "demos"}
        done={["train", "score", "practice", "graduate", "done"].includes(stage)}
        summary={
          demoSamples != null
            ? `${demoSamples.toLocaleString()} samples · ${numRays}-ray sensor`
            : null
        }
      >
        <p className="studio-blurb">
          Sgt. Heuristic flies {DEMO_EPISODES} random rooms; the successful runs become the
          training data for your rookie.
        </p>
        <div className="studio-picker" style={{ marginBottom: 12 }}>
          <div className="studio-picker-label">SENSOR</div>
          <div className="studio-ray-row">
            {RAY_PRESETS.map((p) => (
              <button
                key={p.rays}
                className={`studio-ray-btn ${numRays === p.rays ? "on" : ""}`}
                onClick={() => setNumRays(p.rays)}
                title={p.sub}
                disabled={busy || stage !== "demos"}
              >
                <div className="studio-ray-n">{p.label}</div>
                <div className="studio-ray-sub">{p.sub}</div>
              </button>
            ))}
          </div>
          <div className="studio-meta" style={{ marginTop: 4 }}>
            {numRays === 3
              ? "front / left / right only — diagonal obstacles are invisible"
              : `${numRays} rays around the drone — lidar-style scan`}
          </div>
        </div>
        <button className="studio-cta" disabled={busy || stage !== "demos"} onClick={onCollect}>
          {busy && stage === "demos"
            ? task
              ? `Collecting… ${task.detail || ""}`
              : "Collecting…"
            : "Collect demos"}
        </button>
      </Step>

      {/* Train -------------------------------------------------------- */}
      <Step
        n={3}
        title="Train the network (behavior cloning)"
        active={stage === "train"}
        done={["score", "practice", "graduate", "done"].includes(stage)}
        summary={
          bcAccuracy != null
            ? `${(bcAccuracy * 100).toFixed(1)}% accuracy matching Sgt`
            : null
        }
      >
        <p className="studio-blurb">
          Fits a 64-unit MLP to predict Sgt's action from the {3 + numRays}-number observation
          ({numRays} ray distances + heading & battery). Takes a few seconds.
        </p>
        <button className="studio-cta" disabled={busy || stage !== "train"} onClick={onTrain}>
          {busy && stage === "train" ? "Training…" : "Train network"}
        </button>
      </Step>

      {/* Score -------------------------------------------------------- */}
      <Step
        n={4}
        title="Score on fresh rooms"
        active={stage === "score"}
        done={["practice", "graduate", "done"].includes(stage)}
        summary={
          evalSuccess != null && !practiced
            ? `${(evalSuccess * 100).toFixed(0)}% success across ${EVAL_EPISODES} rooms`
            : null
        }
      >
        <p className="studio-blurb">
          Flies {EVAL_EPISODES} rooms it's never seen. The ceiling here is roughly Sgt's
          success rate — behavior cloning can match the teacher but not exceed them.
        </p>
        <button className="studio-cta" disabled={busy || stage !== "score"} onClick={onScore}>
          {busy && stage === "score" ? "Scoring…" : "Score"}
        </button>
      </Step>

      {/* Practice (RL, optional) ------------------------------------- */}
      <Step
        n={5}
        title="Practice (reinforcement learning) — optional"
        active={stage === "practice"}
        done={practiced || ["graduate", "done"].includes(stage)}
        summary={
          practiced && evalSuccess != null
            ? `${(evalSuccess * 100).toFixed(0)}% after RL practice`
            : null
        }
      >
        <p className="studio-blurb">
          Runs {RL_EPISODES} self-play episodes warm-started from your BC weights. This is the
          only path that can push past Sgt's ceiling — but it's slower (~30-60s).
        </p>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            className="studio-cta"
            disabled={busy || stage !== "practice"}
            onClick={onPractice}
          >
            {busy && stage === "practice"
              ? task
                ? `Practicing… ep ${task.extra?.episode ?? "?"}/${RL_EPISODES}`
                : "Practicing…"
              : "Practice (RL)"}
          </button>
          <button
            className="studio-ghost"
            disabled={busy || stage !== "practice"}
            onClick={onSkipPractice}
          >
            Skip
          </button>
        </div>
      </Step>

      {/* Graduate ----------------------------------------------------- */}
      <Step
        n={6}
        title="Graduate to the Arcade"
        active={stage === "graduate"}
        done={stage === "done"}
        summary={stage === "done" ? "Pilot saved — ready to race." : null}
      >
        <div className="studio-graduation">
          <div className="studio-pilot-preview" style={{ ['--hue' as any]: hue }}>
            <div className="studio-preview-face">{face}</div>
            <div>
              <div className="studio-preview-name">{name || "unnamed"}</div>
              <div className="studio-preview-meta">
                {practiced ? "BC + RL practiced" : "behavior-cloned"} ·{" "}
                {evalSuccess != null ? `${(evalSuccess * 100).toFixed(0)}% sim` : "untested"}
              </div>
            </div>
          </div>
          {stage !== "done" ? (
            <button className="studio-cta primary" disabled={busy} onClick={onGraduate}>
              Send {name || "pilot"} to the Arcade →
            </button>
          ) : (
            <div style={{ display: "flex", gap: 8 }}>
              <button className="studio-cta primary" onClick={() => onGraduated(`custom:${modelName}`)}>
                Go race {name} →
              </button>
              <button className="studio-ghost" onClick={reset}>
                Train another
              </button>
            </div>
          )}
        </div>
      </Step>

      {err && <div className="studio-err">⚠ {err}</div>}
    </div>
  );
}

function Step({
  n,
  title,
  active,
  done,
  summary,
  children,
}: {
  n: number;
  title: string;
  active: boolean;
  done: boolean;
  summary?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className={`studio-step ${active ? "active" : ""} ${done ? "done" : ""}`}>
      <div className="studio-step-head">
        <div className="studio-step-n">{done ? "✓" : n}</div>
        <div className="studio-step-title">{title}</div>
        {summary && <div className="studio-step-summary">{summary}</div>}
      </div>
      {active && <div className="studio-step-body">{children}</div>}
    </div>
  );
}
