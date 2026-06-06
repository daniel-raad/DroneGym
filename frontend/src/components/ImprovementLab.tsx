import React from "react";

export type ScoreEntry = {
  label: string;
  success_rate: number;
  difficulty: number;
  n: number;
  at: number; // ms timestamp
};

type Props = {
  busy: boolean;
  scores: ScoreEntry[];
  baselineRate: number | null;
  hasTrainedModel: boolean;
  activeModel: string;
  isPracticed: boolean;
  onScoreCurrent: () => Promise<void>;
  onMoreDemos: () => Promise<void>;
  onPractice: () => Promise<void>;
  onCurriculumBC: () => Promise<void>;
  onResetExperiment: () => void;
};

const IMPROVEMENTS = [
  {
    key: "demos",
    title: "Show it more examples",
    blurb:
      "Collect 400 more demo flights and retrain. Bigger dataset = clearer pattern, but the network still inherits any quirks of the rule-based teacher.",
    cost: "~30s",
    color: "#5db8ff",
  },
  {
    key: "practice",
    title: "Let it practice on its own",
    blurb:
      "The network flies for real in the simulator. It gets +100 reward for reaching the target, −100 for crashing, and updates its own weights from the result. Starts from the current network (not from scratch).",
    cost: "~60s",
    color: "#59e08b",
  },
  {
    key: "curriculum",
    title: "Train on easy worlds first",
    blurb:
      "Collect demos at difficulty 1, 2, and 3 — mixed together — then retrain. Easier rooms produce more successful demos, which gives a cleaner signal before tackling the hard rooms.",
    cost: "~45s",
    color: "#ffd166",
  },
];

export function ImprovementLab(p: Props) {
  if (!p.hasTrainedModel) return null;

  const best = p.scores.length
    ? Math.max(...p.scores.map((s) => s.success_rate))
    : 0;
  const baseline = p.scores[0]?.success_rate;
  const latest = p.scores[p.scores.length - 1];

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: 12,
        marginBottom: 8,
        background: "var(--panel-2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 12,
            fontWeight: 700,
            background: "var(--accent-2)",
            color: "#fff",
            flexShrink: 0,
          }}
        >
          6
        </div>
        <strong style={{ fontSize: 13 }}>Improve the network</strong>
      </div>
      <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, marginBottom: 8 }}>
        Pick a strategy, run it, then re-score. The goal: get the success-rate bar to rise.
      </div>

      <div
        style={{
          fontSize: 11,
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 6,
          padding: "6px 8px",
          marginBottom: 8,
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
          alignItems: "center",
        }}
      >
        <span style={{ color: "var(--muted)" }}>Active network</span>
        <span style={{ fontFamily: "ui-monospace, monospace" }}>
          {p.activeModel}{" "}
          <span className={`badge ${p.isPracticed ? "warn" : "muted"}`}>
            {p.isPracticed ? "RL-trained" : "BC-trained"}
          </span>
        </span>
      </div>

      <button disabled={p.busy} onClick={p.onScoreCurrent} style={{ marginBottom: 8 }}>
        {p.scores.length === 0
          ? "Score the network on 20 fresh worlds"
          : "Re-score on 20 fresh worlds"}
      </button>

      {p.scores.length > 0 && (
        <div
          style={{
            background: "var(--panel)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            padding: 8,
            marginBottom: 10,
          }}
        >
          <div
            style={{
              fontSize: 11,
              color: "var(--muted)",
              textTransform: "uppercase",
              letterSpacing: 1,
              marginBottom: 4,
            }}
          >
            Experiment log
          </div>
          {p.baselineRate != null && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 60px 80px",
                gap: 6,
                alignItems: "center",
                fontSize: 11,
                paddingBottom: 4,
                borderBottom: "1px dashed var(--border)",
                marginBottom: 4,
              }}
            >
              <span style={{ color: "var(--muted)" }}>Rule-based pilot (target)</span>
              <div
                style={{
                  height: 6,
                  background: "var(--panel-2)",
                  borderRadius: 3,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    width: `${p.baselineRate * 100}%`,
                    height: "100%",
                    background: "var(--muted)",
                  }}
                />
              </div>
              <span
                style={{
                  fontVariantNumeric: "tabular-nums",
                  textAlign: "right",
                  color: "var(--muted)",
                }}
              >
                {(p.baselineRate * 100).toFixed(0)}%
              </span>
            </div>
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {p.scores.map((s, i) => {
              const isBest = s.success_rate === best && best > 0;
              const delta =
                baseline != null && i > 0 ? s.success_rate - baseline : null;
              return (
                <div
                  key={i}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 60px 80px",
                    gap: 6,
                    alignItems: "center",
                    fontSize: 11,
                  }}
                >
                  <span style={{ color: isBest ? "var(--success)" : "var(--text)" }}>
                    {isBest && "★ "}
                    {s.label}
                  </span>
                  <div
                    style={{
                      height: 6,
                      background: "var(--panel-2)",
                      borderRadius: 3,
                      overflow: "hidden",
                      position: "relative",
                    }}
                  >
                    <div
                      style={{
                        width: `${s.success_rate * 100}%`,
                        height: "100%",
                        background: isBest ? "var(--success)" : "var(--accent)",
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontVariantNumeric: "tabular-nums",
                      textAlign: "right",
                      color: "var(--muted)",
                    }}
                  >
                    {(s.success_rate * 100).toFixed(0)}%
                    {delta != null && (
                      <span
                        style={{
                          marginLeft: 4,
                          color:
                            delta > 0
                              ? "var(--success)"
                              : delta < 0
                                ? "var(--danger)"
                                : "var(--muted)",
                        }}
                      >
                        ({delta > 0 ? "+" : ""}
                        {(delta * 100).toFixed(0)})
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
          {latest && p.scores.length > 0 && (
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6, lineHeight: 1.5 }}>
              {p.baselineRate != null ? (
                latest.success_rate >= p.baselineRate ? (
                  <>
                    The network is matching or beating the rule-based pilot ({(p.baselineRate * 100).toFixed(0)}%).
                    Behavior cloning's ceiling is the teacher — to reliably exceed it you'd need <b>Practice</b>.
                  </>
                ) : (
                  <>
                    Still {((p.baselineRate - latest.success_rate) * 100).toFixed(0)} points below the rule-based pilot.
                    More demos / curriculum push you toward the teacher's level; practice can push past it.
                  </>
                )
              ) : p.scores.length > 1 && baseline != null && latest.success_rate > baseline ? (
                "Up from baseline — keep iterating."
              ) : p.scores.length > 1 ? (
                "Same or worse than baseline. Try a different strategy or run it for longer."
              ) : null}
            </div>
          )}
        </div>
      )}

      <div
        style={{
          fontSize: 10,
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 4,
        }}
      >
        Strategies to try
      </div>

      {IMPROVEMENTS.map((imp) => (
        <div
          key={imp.key}
          style={{
            border: "1px solid var(--border)",
            borderLeft: `3px solid ${imp.color}`,
            borderRadius: 6,
            padding: 8,
            marginBottom: 6,
            background: "var(--panel)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 4,
            }}
          >
            <strong style={{ fontSize: 12, color: imp.color }}>{imp.title}</strong>
            <span style={{ fontSize: 10, color: "var(--muted)" }}>{imp.cost}</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.45, marginBottom: 6 }}>
            {imp.blurb}
          </div>
          <button
            className="secondary"
            disabled={p.busy}
            onClick={
              imp.key === "demos"
                ? p.onMoreDemos
                : imp.key === "practice"
                  ? p.onPractice
                  : p.onCurriculumBC
            }
            style={{ margin: 0, padding: "5px 8px", fontSize: 11 }}
          >
            Run this
          </button>
        </div>
      ))}

      {p.scores.length > 0 && (
        <button
          className="secondary"
          onClick={p.onResetExperiment}
          style={{ marginTop: 8, padding: "4px 8px", fontSize: 10 }}
        >
          Reset experiment log
        </button>
      )}
    </div>
  );
}
