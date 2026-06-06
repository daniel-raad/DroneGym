import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type {
  Course,
  CourseResult,
  CurrentTask,
  ModelInfo,
  RolloutSnapshot,
  SystemStatus,
} from "../api";
import { slugify, upsertCustomPilot } from "../arcade/customPilots";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { DonePanel } from "./DonePanel";
import { LivePanel } from "./LivePanel";
import { PlanComposer } from "./PlanComposer";
import { RecipePicker } from "./RecipePicker";
import { TrainingStrip } from "./TrainingStrip";
import { diagnose } from "./diagnostics";
import { summarizePlan } from "./knobMeta";
import { RECIPE_META, recipe, type RecipeName, type Stage } from "./plan";
import { runPlan } from "./runPlan";
import "./gym.css";

type Props = {
  onGraduated?: (pilotId: string) => void;
  onOpenArcade?: () => void;
};

const FACES = ["🦊", "🐱", "🐰", "🐻", "🦉", "🐯", "🐼", "🦅", "🐙", "🐝", "🦋", "🐢"];
const HUES = [195, 320, 45, 130, 270, 15];

type View = "idle" | "training" | "done" | "customize";

// The workshop is a tiny state machine. Each `view` shows exactly one job:
//   idle      → pick a recipe + name your pilot, press Train
//   training  → live rollout dominates, stages shown as a compact strip
//   done      → headline result + Send to Arcade or Try harder
//   customize → power user view (PlanComposer w/ all knobs)
export function Workshop({ onGraduated, onOpenArcade }: Props) {
  const [view, setView] = useState<View>("idle");
  const [name, setName] = useState("");
  const [face, setFace] = useState(FACES[0]);
  const [hue, setHue] = useState(HUES[0]);
  const [recipeName, setRecipeName] = useState<RecipeName>("minimum-viable");
  const [stages, setStages] = useState<Stage[]>(() => recipe("minimum-viable"));

  // Stage progress (filled by runPlan events)
  const [activeStageId, setActiveStageId] = useState<string | null>(null);
  const [stageSummaries, setStageSummaries] = useState<Map<string, string>>(new Map());
  const [stageErrors, setStageErrors] = useState<Map<string, string>>(new Map());

  // Status + courses
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [courses, setCourses] = useState<Course[]>([]);
  const [courseResults, setCourseResults] = useState<Map<string, CourseResult>>(new Map());
  const [raceReady, setRaceReady] = useState(false);
  const lastRollout = useRef<RolloutSnapshot | null>(null);

  const modelName = useMemo(() => (name ? slugify(name) : ""), [name]);
  const datasetName = useMemo(
    () => (modelName ? `${modelName}_dataset` : ""),
    [modelName],
  );

  useEffect(() => {
    api.listCourses().then(setCourses).catch(() => {});
    refreshStatus();
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await api.systemStatus();
      setStatus(s);
      const r = s.current_task?.extra?.rollout;
      if (r) lastRollout.current = r;
    } catch {}
  }, []);

  useEffect(() => {
    const id = setInterval(refreshStatus, view === "training" ? 800 : 4000);
    return () => clearInterval(id);
  }, [view, refreshStatus]);

  const myModel: ModelInfo | null = useMemo(() => {
    if (!modelName || !status) return null;
    return status.models.find((m) => m.name === modelName) ?? null;
  }, [modelName, status]);

  useEffect(() => {
    if (myModel?.course_results) {
      const m = new Map<string, CourseResult>();
      for (const k of Object.keys(myModel.course_results)) {
        m.set(k, myModel.course_results[k] as CourseResult);
      }
      setCourseResults(m);
      setRaceReady(!!myModel.race_ready);
    }
  }, [myModel?.name, myModel?.course_results, myModel?.race_ready]);

  const diagnostics = useMemo(
    () => diagnose(myModel, Array.from(courseResults.values())),
    [myModel, courseResults],
  );

  // Selecting a recipe in idle/done view also resets the plan and clears
  // stale stage state so the user always starts from a clean slate.
  const onSelectRecipe = (name: RecipeName) => {
    if (view === "training") return;
    setRecipeName(name);
    setStages(recipe(name));
    setStageSummaries(new Map());
    setStageErrors(new Map());
  };

  const runStages = async (planStages: Stage[]) => {
    setView("training");
    setStageSummaries(new Map());
    setStageErrors(new Map());
    setActiveStageId(null);
    const ctx = {
      modelName,
      datasetName,
      base: { room_width: 10, room_height: 10, num_obstacles: 3, max_steps: 250 },
    };
    try {
      for await (const ev of runPlan(planStages, ctx)) {
        if (ev.type === "stage-start") setActiveStageId(ev.stageId);
        else if (ev.type === "stage-end") {
          setStageSummaries((m) => {
            const next = new Map(m);
            next.set(ev.stageId, ev.summary);
            return next;
          });
          setActiveStageId(null);
        } else if (ev.type === "stage-error") {
          setStageErrors((m) => {
            const next = new Map(m);
            next.set(ev.stageId, ev.message);
            return next;
          });
          setActiveStageId(null);
        } else if (ev.type === "courses") {
          const m = new Map<string, CourseResult>();
          for (const r of ev.results) m.set(r.course_id, r);
          setCourseResults(m);
          setRaceReady(ev.raceReady);
        }
        await refreshStatus();
      }
    } finally {
      setActiveStageId(null);
      await refreshStatus();
      setView("done");
    }
  };

  const onTrain = () => {
    if (!modelName) return;
    runStages(stages);
  };

  const onSendToArcade = () => {
    if (!modelName) return;
    const id = `custom:${modelName}`;
    upsertCustomPilot({
      id,
      modelName,
      name: name.trim() || modelName,
      face,
      hue,
      createdAt: Date.now(),
      practiced: stages.some((s) => s.kind === "rl"),
      bcAccuracy: myModel?.test_accuracy ?? null,
      evalSuccess: bestCourse(courseResults),
    });
    onGraduated?.(id);
  };

  const onTryHarder = (next: RecipeName) => {
    setRecipeName(next);
    setStages(recipe(next));
    setStageSummaries(new Map());
    setStageErrors(new Map());
    setView("idle");
  };

  const onRetry = () => {
    setStageSummaries(new Map());
    setStageErrors(new Map());
    setView("idle");
  };

  return (
    <div className="gym2">
      <Header
        view={view}
        onOpenArcade={onOpenArcade}
        onCustomize={() => setView(view === "customize" ? "idle" : "customize")}
        results={courseResults}
        courses={courses}
        raceReady={raceReady}
      />

      {view === "idle" && (
        <IdleView
          name={name}
          face={face}
          hue={hue}
          recipeName={recipeName}
          onName={setName}
          onFace={setFace}
          onHue={setHue}
          onSelectRecipe={onSelectRecipe}
          canTrain={!!modelName}
          onTrain={onTrain}
        />
      )}

      {view === "training" && (
        <TrainingView
          stages={stages}
          activeStageId={activeStageId}
          stageSummaries={stageSummaries}
          stageErrors={stageErrors}
          task={(status?.current_task as CurrentTask | null) ?? null}
          fallback={lastRollout.current}
          smoothedReward={myModel?.smoothed_reward ?? []}
          smoothedSuccess={myModel?.smoothed_success ?? []}
        />
      )}

      {view === "done" && (
        <DoneView
          task={(status?.current_task as CurrentTask | null) ?? null}
          fallback={lastRollout.current}
          smoothedReward={myModel?.smoothed_reward ?? []}
          smoothedSuccess={myModel?.smoothed_success ?? []}
          raceReady={raceReady}
          results={courseResults}
          currentRecipe={recipeName}
          diagnostics={diagnostics}
          onSendToArcade={onSendToArcade}
          onTryHarder={onTryHarder}
          onRetry={onRetry}
        />
      )}

      {view === "customize" && (
        <CustomizeView
          stages={stages}
          onChange={setStages}
          name={name}
          face={face}
          hue={hue}
          onName={setName}
          onFace={setFace}
          onHue={setHue}
          canTrain={!!modelName}
          onTrain={onTrain}
        />
      )}
    </div>
  );
}

function bestCourse(results: Map<string, CourseResult>): number | null {
  let best: number | null = null;
  for (const r of results.values()) {
    if (best == null || r.success_rate > best) best = r.success_rate;
  }
  return best;
}

// === Header ================================================================

function Header({
  view,
  onOpenArcade,
  onCustomize,
  results,
  courses,
  raceReady,
}: {
  view: View;
  onOpenArcade?: () => void;
  onCustomize: () => void;
  results: Map<string, CourseResult>;
  courses: Course[];
  raceReady: boolean;
}) {
  return (
    <header className="gym2-header">
      <div className="gym2-brand">
        <h1>DroneGym</h1>
        <span className="gym2-tag">build a pilot that can clear the arcade</span>
      </div>
      <ScorecardCompact courses={courses} results={results} raceReady={raceReady} />
      <div className="gym2-header-actions">
        <button className="gym2-link" onClick={onCustomize} title="Power mode: tune every knob">
          {view === "customize" ? "← Recipes" : "Customize ⚙"}
        </button>
        <button className="gym2-link" onClick={onOpenArcade}>
          ← Arcade
        </button>
      </div>
    </header>
  );
}

function ScorecardCompact({
  courses,
  results,
  raceReady,
}: {
  courses: Course[];
  results: Map<string, CourseResult>;
  raceReady: boolean;
}) {
  return (
    <div className={`gym2-score ${raceReady ? "ready" : ""}`}>
      <span className="gym2-score-label">Readiness</span>
      {courses.map((c) => {
        const r = results.get(c.id);
        const pct = r ? r.success_rate : 0;
        return (
          <div key={c.id} className="gym2-score-pill" title={c.label}>
            <span className="gym2-score-name">{c.label}</span>
            <div className="gym2-score-bar">
              <div
                className="gym2-score-fill"
                style={{ width: `${Math.min(pct, 1) * 100}%` }}
              />
            </div>
            <span className="gym2-score-pct">
              {r ? `${(pct * 100).toFixed(0)}%` : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// === Idle view =============================================================

function IdleView({
  name,
  face,
  hue,
  recipeName,
  onName,
  onFace,
  onHue,
  onSelectRecipe,
  canTrain,
  onTrain,
}: {
  name: string;
  face: string;
  hue: number;
  recipeName: RecipeName;
  onName: (v: string) => void;
  onFace: (v: string) => void;
  onHue: (v: number) => void;
  onSelectRecipe: (n: RecipeName) => void;
  canTrain: boolean;
  onTrain: () => void;
}) {
  return (
    <main className="gym2-idle">
      <div className="gym2-hero">
        <h2>1. Pick a training recipe</h2>
        <p className="gym2-hero-sub">
          Each recipe is a fully-formed plan. Pick one and press Train — that's it.
          Want to tweak the details? Hit <em>Customize</em> in the top-right.
        </p>
      </div>

      <RecipePicker selected={recipeName} onSelect={onSelectRecipe} />

      <div className="gym2-launch">
        <div className="gym2-launch-name">
          <h2 className="gym2-launch-step">2. Name your pilot</h2>
          <input
            className="gym2-name"
            placeholder="e.g. Zoom"
            value={name}
            maxLength={22}
            onChange={(e) => onName(e.target.value)}
          />
          <div className="gym2-launch-mini">
            <div className="gym2-faces">
              {FACES.map((f) => (
                <button
                  key={f}
                  className={`gym2-face ${face === f ? "on" : ""}`}
                  onClick={() => onFace(f)}
                  type="button"
                >
                  {f}
                </button>
              ))}
            </div>
            <div className="gym2-hues">
              {HUES.map((h) => (
                <button
                  key={h}
                  className={`gym2-hue ${hue === h ? "on" : ""}`}
                  style={{ background: `hsl(${h},70%,60%)` }}
                  onClick={() => onHue(h)}
                  type="button"
                />
              ))}
            </div>
          </div>
        </div>
        <div className="gym2-launch-train">
          <h2 className="gym2-launch-step">3. Train</h2>
          <button
            className="gym2-train-big"
            disabled={!canTrain}
            onClick={onTrain}
            title={canTrain ? "Run the selected recipe" : "Name your pilot first"}
          >
            ▶ Train pilot
          </button>
          <div className="gym2-launch-note">
            {canTrain
              ? `Will run "${RECIPE_META[recipeName].name}" (${RECIPE_META[recipeName].time}).`
              : "Enter a name first."}
          </div>
        </div>
      </div>
    </main>
  );
}

// === Training view =========================================================

function TrainingView({
  stages,
  activeStageId,
  stageSummaries,
  stageErrors,
  task,
  fallback,
  smoothedReward,
  smoothedSuccess,
}: {
  stages: Stage[];
  activeStageId: string | null;
  stageSummaries: Map<string, string>;
  stageErrors: Map<string, string>;
  task: CurrentTask | null;
  fallback: RolloutSnapshot | null;
  smoothedReward: number[];
  smoothedSuccess: number[];
}) {
  return (
    <main className="gym2-training">
      <div className="gym2-live-wrap">
        <LivePanel
          task={task}
          fallback={fallback}
          history={{ smoothedReward, smoothedSuccess }}
        />
      </div>
      <TrainingStrip
        stages={stages}
        activeStageId={activeStageId}
        stageSummaries={stageSummaries}
        stageErrors={stageErrors}
      />
    </main>
  );
}

// === Done view =============================================================

function DoneView({
  task,
  fallback,
  smoothedReward,
  smoothedSuccess,
  raceReady,
  results,
  currentRecipe,
  diagnostics,
  onSendToArcade,
  onTryHarder,
  onRetry,
}: {
  task: CurrentTask | null;
  fallback: RolloutSnapshot | null;
  smoothedReward: number[];
  smoothedSuccess: number[];
  raceReady: boolean;
  results: Map<string, CourseResult>;
  currentRecipe: RecipeName;
  diagnostics: ReturnType<typeof diagnose>;
  onSendToArcade: () => void;
  onTryHarder: (next: RecipeName) => void;
  onRetry: () => void;
}) {
  return (
    <main className="gym2-done">
      <div className="gym2-done-left">
        <LivePanel
          task={task}
          fallback={fallback}
          history={{ smoothedReward, smoothedSuccess }}
        />
      </div>
      <div className="gym2-done-right">
        <DonePanel
          raceReady={raceReady}
          results={results}
          currentRecipe={currentRecipe}
          onSendToArcade={onSendToArcade}
          onTryHarder={onTryHarder}
          onRetry={onRetry}
        />
        <DiagnosticsPanel items={diagnostics} />
      </div>
    </main>
  );
}

// === Customize view (power mode) ===========================================

function CustomizeView({
  stages,
  onChange,
  name,
  face,
  hue,
  onName,
  onFace,
  onHue,
  canTrain,
  onTrain,
}: {
  stages: Stage[];
  onChange: (next: Stage[]) => void;
  name: string;
  face: string;
  hue: number;
  onName: (v: string) => void;
  onFace: (v: string) => void;
  onHue: (v: number) => void;
  canTrain: boolean;
  onTrain: () => void;
}) {
  const summary = summarizePlan(stages);
  return (
    <main className="gym2-custom">
      <div className="gym2-custom-head">
        <div>
          <h2>Customize</h2>
          <p className="gym2-custom-sub">
            Every knob has an explanation of what it changes and how it affects
            your pilot. Use the Advanced toggle on each stage for less common knobs.
          </p>
        </div>
        <div className="gym2-custom-name">
          <input
            className="gym2-name"
            placeholder="Pilot name"
            value={name}
            maxLength={22}
            onChange={(e) => onName(e.target.value)}
          />
          <div className="gym2-faces small">
            {FACES.map((f) => (
              <button
                key={f}
                className={`gym2-face ${face === f ? "on" : ""}`}
                onClick={() => onFace(f)}
                type="button"
              >
                {f}
              </button>
            ))}
          </div>
          <div className="gym2-hues">
            {HUES.map((h) => (
              <button
                key={h}
                className={`gym2-hue ${hue === h ? "on" : ""}`}
                style={{ background: `hsl(${h},70%,60%)` }}
                onClick={() => onHue(h)}
                type="button"
              />
            ))}
          </div>
          <button
            className="gym2-train-big"
            disabled={!canTrain}
            onClick={onTrain}
          >
            ▶ Train
          </button>
        </div>
      </div>
      <div className="gym2-summary">
        <div className="gym2-summary-label">This plan will…</div>
        <div className="gym2-summary-text">{summary}</div>
      </div>
      <PlanComposer
        stages={stages}
        running={false}
        activeStageId={null}
        stageSummaries={new Map()}
        stageErrors={new Map()}
        onChange={onChange}
      />
    </main>
  );
}
