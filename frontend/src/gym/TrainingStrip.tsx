import type { Stage } from "./plan";
import { STAGE_LABELS } from "./plan";

type Props = {
  stages: Stage[];
  activeStageId: string | null;
  stageSummaries: Map<string, string>;
  stageErrors: Map<string, string>;
};

// Compact horizontal pipeline that replaces the plan composer during training.
// Each stage is a bullet that lights up as the runner reaches it.
export function TrainingStrip({
  stages,
  activeStageId,
  stageSummaries,
  stageErrors,
}: Props) {
  return (
    <div className="ts-row">
      {stages.map((s, i) => {
        const active = s.id === activeStageId;
        const summary = stageSummaries.get(s.id);
        const error = stageErrors.get(s.id);
        const done = !!summary && !active;
        return (
          <div
            key={s.id}
            className={`ts-step ${active ? "active" : ""} ${done ? "done" : ""} ${error ? "error" : ""}`}
          >
            <div className="ts-dot">
              {done ? "✓" : active ? <span className="ts-spin">●</span> : i + 1}
            </div>
            <div className="ts-text">
              <div className="ts-label">{STAGE_LABELS[s.kind]}</div>
              {summary && <div className="ts-summary">{summary}</div>}
              {error && <div className="ts-error">⚠ {error}</div>}
            </div>
            {i < stages.length - 1 && <div className="ts-connector" />}
          </div>
        );
      })}
    </div>
  );
}
