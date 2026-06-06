import type { Diagnostic } from "./diagnostics";

type Props = { items: Diagnostic[] };

export function DiagnosticsPanel({ items }: Props) {
  return (
    <div className="gym-diag">
      <div className="gym-diag-head">
        <strong>Diagnostics</strong>
        <span className="gym-diag-count">
          {items.length === 0 ? "no issues detected" : `${items.length} issue${items.length === 1 ? "" : "s"}`}
        </span>
      </div>
      {items.length === 0 ? (
        <div className="gym-diag-empty">
          Train a pilot to see actionable warnings — the diagnostic engine reads
          training curves and per-course results and suggests fixes.
        </div>
      ) : (
        <div className="gym-diag-list">
          {items.map((d) => (
            <div key={d.id} className={`gym-diag-item ${d.severity}`}>
              <div className="gym-diag-title">
                <span className="gym-diag-sev">
                  {d.severity === "error" ? "✗" : d.severity === "warn" ? "⚠" : "·"}
                </span>
                {d.title}
              </div>
              <div className="gym-diag-detail">{d.detail}</div>
              <div className="gym-diag-fix">
                <strong>Fix:</strong> {d.fix}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
