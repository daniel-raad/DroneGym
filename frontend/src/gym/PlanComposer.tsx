import {
  STAGE_LABELS,
  mkBC,
  mkCollect,
  mkDAgger,
  mkEval,
  mkRL,
  type Stage,
} from "./plan";

type Props = {
  stages: Stage[];
  running: boolean;
  activeStageId: string | null;
  stageSummaries: Map<string, string>;
  stageErrors: Map<string, string>;
  onChange: (next: Stage[]) => void;
};

// Right-rail vertical stack. Each stage is a card with its own knobs.
// Stages are independent — re-order, clone, delete freely. The runner walks
// the list top-to-bottom.
export function PlanComposer({
  stages,
  running,
  activeStageId,
  stageSummaries,
  stageErrors,
  onChange,
}: Props) {
  const update = (id: string, patch: Partial<Stage>) => {
    onChange(stages.map((s) => (s.id === id ? ({ ...s, ...patch } as Stage) : s)));
  };
  const remove = (id: string) => onChange(stages.filter((s) => s.id !== id));
  const move = (id: string, dir: -1 | 1) => {
    const idx = stages.findIndex((s) => s.id === id);
    if (idx < 0) return;
    const next = [...stages];
    const swap = idx + dir;
    if (swap < 0 || swap >= next.length) return;
    [next[idx], next[swap]] = [next[swap], next[idx]];
    onChange(next);
  };
  const add = (kind: Stage["kind"]) => {
    let s: Stage;
    if (kind === "collect") s = mkCollect();
    else if (kind === "bc") s = mkBC();
    else if (kind === "dagger") s = mkDAgger();
    else if (kind === "rl") s = mkRL();
    else s = mkEval();
    onChange([...stages, s]);
  };

  return (
    <div className="gym-plan">
      <div className="gym-plan-stages">
        {stages.map((s, i) => {
          const active = s.id === activeStageId;
          const summary = stageSummaries.get(s.id);
          const error = stageErrors.get(s.id);
          const done = !!summary && !active;
          return (
            <div
              key={s.id}
              className={`gym-stage ${active ? "active" : ""} ${done ? "done" : ""} ${error ? "error" : ""}`}
            >
              <div className="gym-stage-head">
                <span className="gym-stage-n">{done ? "✓" : active ? "▶" : i + 1}</span>
                <span className="gym-stage-title">{STAGE_LABELS[s.kind]}</span>
                <span style={{ flex: 1 }} />
                <button
                  className="gym-stage-arrow"
                  disabled={running}
                  onClick={() => move(s.id, -1)}
                  title="move up"
                >
                  ↑
                </button>
                <button
                  className="gym-stage-arrow"
                  disabled={running}
                  onClick={() => move(s.id, 1)}
                  title="move down"
                >
                  ↓
                </button>
                <button
                  className="gym-stage-x"
                  disabled={running}
                  onClick={() => remove(s.id)}
                >
                  ✕
                </button>
              </div>
              <StageBody stage={s} disabled={running} onPatch={(p) => update(s.id, p)} />
              {summary && <div className="gym-stage-summary">{summary}</div>}
              {error && <div className="gym-stage-err">⚠ {error}</div>}
            </div>
          );
        })}
      </div>
      <div className="gym-plan-add">
        <span>Add stage:</span>
        {(["collect", "bc", "dagger", "rl", "eval"] as const).map((k) => (
          <button key={k} disabled={running} onClick={() => add(k)}>
            + {STAGE_LABELS[k]}
          </button>
        ))}
      </div>
    </div>
  );
}

function StageBody({
  stage,
  disabled,
  onPatch,
}: {
  stage: Stage;
  disabled: boolean;
  onPatch: (p: Partial<Stage>) => void;
}) {
  if (stage.kind === "collect") {
    return (
      <div className="gym-stage-knobs">
        <Knob
          label="episodes"
          value={stage.num_episodes}
          onChange={(v) => onPatch({ num_episodes: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <KnobText
          label="difficulty mix"
          value={stage.difficulties.join(",")}
          onChange={(v) =>
            onPatch({
              difficulties: v
                .split(",")
                .map((s) => parseInt(s.trim(), 10))
                .filter((n) => !isNaN(n)),
            } as Partial<Stage>)
          }
          disabled={disabled}
        />
        <KnobSelect
          label="rays"
          value={String(stage.num_rays)}
          options={["3", "8", "16", "32"]}
          onChange={(v) => onPatch({ num_rays: parseInt(v, 10) } as Partial<Stage>)}
          disabled={disabled}
        />
      </div>
    );
  }
  if (stage.kind === "bc") {
    return (
      <div className="gym-stage-knobs">
        <Knob
          label="epochs"
          value={stage.epochs}
          onChange={(v) => onPatch({ epochs: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="hidden"
          value={stage.hidden_size}
          onChange={(v) => onPatch({ hidden_size: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="layers"
          value={stage.num_layers}
          onChange={(v) => onPatch({ num_layers: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="lr ×1e-4"
          value={Math.round(stage.learning_rate * 1e4)}
          onChange={(v) =>
            onPatch({ learning_rate: Math.max(1, v) * 1e-4 } as Partial<Stage>)
          }
          disabled={disabled}
        />
      </div>
    );
  }
  if (stage.kind === "dagger") {
    return (
      <div className="gym-stage-knobs">
        <Knob
          label="rounds"
          value={stage.rounds}
          onChange={(v) => onPatch({ rounds: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="eps/round"
          value={stage.episodes_per_round}
          onChange={(v) => onPatch({ episodes_per_round: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="refit epochs"
          value={stage.epochs}
          onChange={(v) => onPatch({ epochs: v } as Partial<Stage>)}
          disabled={disabled}
        />
      </div>
    );
  }
  if (stage.kind === "rl") {
    return (
      <div className="gym-stage-knobs">
        <KnobSelect
          label="algo"
          value={stage.algorithm}
          options={["a2c", "ppo"]}
          onChange={(v) => onPatch({ algorithm: v as "a2c" | "ppo" } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="episodes"
          value={stage.episodes}
          onChange={(v) => onPatch({ episodes: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="entropy ×100"
          value={Math.round(stage.entropy_coef * 100)}
          onChange={(v) =>
            onPatch({ entropy_coef: Math.max(0, v) / 100 } as Partial<Stage>)
          }
          disabled={disabled}
        />
        <Knob
          label="hidden"
          value={stage.hidden_size}
          onChange={(v) => onPatch({ hidden_size: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <Knob
          label="layers"
          value={stage.num_layers}
          onChange={(v) => onPatch({ num_layers: v } as Partial<Stage>)}
          disabled={disabled}
        />
        <KnobText
          label="curriculum"
          value={stage.curriculum_schedule.join(",")}
          onChange={(v) =>
            onPatch({
              curriculum_schedule: v
                .split(",")
                .map((s) => parseInt(s.trim(), 10))
                .filter((n) => !isNaN(n)),
            } as Partial<Stage>)
          }
          disabled={disabled}
        />
        <Knob
          label="rand wind ×100"
          value={Math.round(stage.randomize_wind * 100)}
          onChange={(v) =>
            onPatch({ randomize_wind: Math.max(0, v) / 100 } as Partial<Stage>)
          }
          disabled={disabled}
        />
        <Knob
          label="rand noise ×100"
          value={Math.round(stage.randomize_noise * 100)}
          onChange={(v) =>
            onPatch({ randomize_noise: Math.max(0, v) / 100 } as Partial<Stage>)
          }
          disabled={disabled}
        />
        <Knob
          label="±obstacles"
          value={stage.randomize_obstacles}
          onChange={(v) =>
            onPatch({ randomize_obstacles: Math.max(0, v) } as Partial<Stage>)
          }
          disabled={disabled}
        />
        <KnobBool
          label="warm start"
          value={stage.warm_start}
          onChange={(v) => onPatch({ warm_start: v } as Partial<Stage>)}
          disabled={disabled}
        />
      </div>
    );
  }
  // eval
  return (
    <div className="gym-stage-knobs">
      <Knob
        label="runs/course"
        value={stage.runs_per_course}
        onChange={(v) => onPatch({ runs_per_course: v } as Partial<Stage>)}
        disabled={disabled}
      />
    </div>
  );
}

function Knob({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className="gym-knob">
      <span className="gym-knob-label">{label}</span>
      <input
        type="number"
        className="gym-knob-input"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseInt(e.target.value || "0", 10))}
      />
    </label>
  );
}

function KnobText({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="gym-knob">
      <span className="gym-knob-label">{label}</span>
      <input
        type="text"
        className="gym-knob-input"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function KnobSelect({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="gym-knob">
      <span className="gym-knob-label">{label}</span>
      <select
        className="gym-knob-input"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function KnobBool({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="gym-knob gym-knob-bool">
      <span className="gym-knob-label">{label}</span>
      <input
        type="checkbox"
        checked={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
