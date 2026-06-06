import { useState } from "react";
import {
  STAGE_LABELS,
  mkBC,
  mkCollect,
  mkDAgger,
  mkEval,
  mkRL,
  type Stage,
} from "./plan";
import {
  STAGE_META,
  formatValue,
  parseValue,
  type KnobMeta,
  type KnobUI,
} from "./knobMeta";

type Props = {
  stages: Stage[];
  running: boolean;
  activeStageId: string | null;
  stageSummaries: Map<string, string>;
  stageErrors: Map<string, string>;
  onChange: (next: Stage[]) => void;
};

// Editable list of stages. Each stage card shows what it does, then a small
// set of "primary" knobs each with a one-line "what this changes + impact on
// pilot" line, then an Advanced expander with the rest.
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
    <div className="pc-root">
      <div className="pc-stages">
        {stages.map((s, i) => {
          const active = s.id === activeStageId;
          const summary = stageSummaries.get(s.id);
          const error = stageErrors.get(s.id);
          const done = !!summary && !active;
          return (
            <StageCard
              key={s.id}
              stage={s}
              index={i}
              active={active}
              done={done}
              error={error}
              summary={summary}
              disabled={running}
              onPatch={(p) => update(s.id, p)}
              onMoveUp={() => move(s.id, -1)}
              onMoveDown={() => move(s.id, 1)}
              onRemove={() => remove(s.id)}
            />
          );
        })}
      </div>
      <div className="pc-add">
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

function StageCard({
  stage,
  index,
  active,
  done,
  error,
  summary,
  disabled,
  onPatch,
  onMoveUp,
  onMoveDown,
  onRemove,
}: {
  stage: Stage;
  index: number;
  active: boolean;
  done: boolean;
  error: string | undefined;
  summary: string | undefined;
  disabled: boolean;
  onPatch: (p: Partial<Stage>) => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onRemove: () => void;
}) {
  const meta = STAGE_META[stage.kind];
  const [showAdvanced, setShowAdvanced] = useState(false);
  const primary = meta.knobs.filter((k) => k.tone === "primary");
  const advanced = meta.knobs.filter((k) => k.tone === "advanced");

  return (
    <div
      className={`pc-stage ${active ? "active" : ""} ${done ? "done" : ""} ${error ? "error" : ""}`}
    >
      <div className="pc-stage-head">
        <span className="pc-stage-n">{done ? "✓" : active ? "▶" : index + 1}</span>
        <span className="pc-stage-title">{meta.title}</span>
        <span style={{ flex: 1 }} />
        <button
          className="pc-iconbtn"
          disabled={disabled}
          onClick={onMoveUp}
          title="move up"
        >
          ↑
        </button>
        <button
          className="pc-iconbtn"
          disabled={disabled}
          onClick={onMoveDown}
          title="move down"
        >
          ↓
        </button>
        <button
          className="pc-iconbtn"
          disabled={disabled}
          onClick={onRemove}
          title="remove"
        >
          ✕
        </button>
      </div>
      <div className="pc-stage-what">{meta.what}</div>

      <div className="pc-knobs">
        {primary.map((knob) => (
          <KnobRow
            key={knob.key}
            knob={knob}
            value={(stage as any)[knob.key]}
            disabled={disabled}
            onChange={(v) => onPatch({ [knob.key]: v } as Partial<Stage>)}
          />
        ))}
      </div>

      {advanced.length > 0 && (
        <div className="pc-advanced">
          <button
            type="button"
            className="pc-advanced-toggle"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? "▾" : "▸"} Advanced ({advanced.length})
          </button>
          {showAdvanced && (
            <div className="pc-knobs">
              {advanced.map((knob) => (
                <KnobRow
                  key={knob.key}
                  knob={knob}
                  value={(stage as any)[knob.key]}
                  disabled={disabled}
                  onChange={(v) => onPatch({ [knob.key]: v } as Partial<Stage>)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {summary && <div className="pc-summary">{summary}</div>}
      {error && <div className="pc-err">⚠ {error}</div>}
    </div>
  );
}

function KnobRow({
  knob,
  value,
  disabled,
  onChange,
}: {
  knob: KnobMeta;
  value: any;
  disabled: boolean;
  onChange: (v: any) => void;
}) {
  return (
    <div className="pc-knob">
      <div className="pc-knob-row">
        <span className="pc-knob-label">{knob.label}</span>
        <KnobInput
          ui={knob.ui}
          value={value}
          disabled={disabled}
          onChange={onChange}
        />
      </div>
      <div className="pc-knob-effect">{knob.effect}</div>
    </div>
  );
}

function KnobInput({
  ui,
  value,
  disabled,
  onChange,
}: {
  ui: KnobUI;
  value: any;
  disabled: boolean;
  onChange: (v: any) => void;
}) {
  if (ui.kind === "select-int") {
    return (
      <select
        className="pc-input"
        value={String(value)}
        disabled={disabled}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
      >
        {ui.options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  }
  if (ui.kind === "select-str") {
    return (
      <select
        className="pc-input wide"
        value={String(value)}
        disabled={disabled}
        onChange={(e) => onChange(isFinite(Number(e.target.value)) && e.target.value.match(/^-?\d/) ? Number(e.target.value) : e.target.value)}
      >
        {ui.options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    );
  }
  if (ui.kind === "bool") {
    return (
      <input
        type="checkbox"
        className="pc-checkbox"
        checked={!!value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  }
  if (ui.kind === "csv-int") {
    return (
      <input
        type="text"
        className="pc-input wide"
        value={formatValue(ui, value)}
        disabled={disabled}
        onChange={(e) => onChange(parseValue(ui, e.target.value, value))}
        placeholder="1,2,3"
      />
    );
  }
  if (ui.kind === "pct") {
    return (
      <div className="pc-pct">
        <input
          type="number"
          className="pc-input"
          value={formatValue(ui, value)}
          disabled={disabled}
          min={0}
          max={Math.round((ui.max ?? 1) * 100)}
          step={1}
          onChange={(e) => onChange(parseValue(ui, e.target.value, value))}
        />
        <span className="pc-suffix">%</span>
      </div>
    );
  }
  if (ui.kind === "float") {
    return (
      <input
        type="number"
        className="pc-input"
        value={formatValue(ui, value)}
        disabled={disabled}
        min={ui.min}
        max={ui.max}
        step={ui.step ?? 0.01}
        onChange={(e) => onChange(parseValue(ui, e.target.value, value))}
      />
    );
  }
  // int
  return (
    <input
      type="number"
      className="pc-input"
      value={value}
      disabled={disabled}
      min={ui.min}
      max={ui.max}
      step={ui.step ?? 1}
      onChange={(e) => onChange(parseValue(ui, e.target.value, value))}
    />
  );
}
