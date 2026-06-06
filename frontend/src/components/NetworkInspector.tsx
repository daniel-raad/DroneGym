import React from "react";
import type { PolicyInspectResponse } from "../api";

type Props = {
  inspect: PolicyInspectResponse | null;
  loading: boolean;
  step: number;
  takenAction?: string;
  flightLabel: string; // "this flight" or "teacher's flight"
};

export function NetworkInspector({
  inspect,
  loading,
  step,
  takenAction,
  flightLabel,
}: Props) {
  if (loading) {
    return (
      <div className="section">
        <h2>What the network would do</h2>
        <div style={{ color: "var(--muted)", fontSize: 12 }}>computing…</div>
      </div>
    );
  }
  if (!inspect || inspect.probs.length === 0) return null;

  const stepIdx = Math.max(0, Math.min(step, inspect.probs.length - 1));
  const row = inspect.probs[stepIdx] ?? [];
  const argmax = inspect.argmax[stepIdx];

  // Disagreement count over the whole flight
  let disagreement = 0;
  if (takenAction !== undefined) {
    const fl = Math.min(inspect.argmax.length, inspect.probs.length);
    for (let i = 0; i < fl; i++) {
      // We only count against takenAction when we know it; here we approximate by
      // assuming the caller passes per-step taken actions via flightLabel context.
      // For a single-step disagreement, see below.
    }
  }

  return (
    <div className="section">
      <h2>What the network would do</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6, lineHeight: 1.5 }}>
        For the sensor readings at step {stepIdx} of {flightLabel}, the network
        ({inspect.model_name}) outputs these probabilities. The bar with the
        outline is its top pick. {takenAction && (
          <>
            The action actually <i>taken</i> at this step is shaded.
          </>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {inspect.actions.map((a, i) => {
          const p = row[i] ?? 0;
          const isArgmax = a === argmax;
          const isTaken = takenAction && a === takenAction;
          return (
            <div
              key={a}
              style={{
                display: "grid",
                gridTemplateColumns: "100px 1fr 50px",
                gap: 8,
                alignItems: "center",
                fontSize: 11,
              }}
            >
              <span
                style={{
                  color: isArgmax ? "var(--accent)" : "var(--muted)",
                  fontWeight: isArgmax ? 700 : 400,
                  fontFamily: "ui-monospace, monospace",
                }}
              >
                {a}
              </span>
              <div
                style={{
                  height: 10,
                  background: isTaken ? "rgba(255,209,102,0.12)" : "var(--panel)",
                  border: `1px solid ${isArgmax ? "var(--accent)" : "var(--border)"}`,
                  borderRadius: 3,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    width: `${Math.max(p * 100, 0.5)}%`,
                    height: "100%",
                    background: isArgmax ? "var(--accent)" : "var(--muted)",
                    opacity: isArgmax ? 0.95 : 0.6,
                  }}
                />
              </div>
              <span
                style={{
                  fontVariantNumeric: "tabular-nums",
                  textAlign: "right",
                  color: isArgmax ? "var(--accent)" : "var(--muted)",
                }}
              >
                {(p * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
      {takenAction && argmax && takenAction !== argmax && (
        <div
          style={{
            fontSize: 11,
            color: "var(--warn)",
            marginTop: 8,
            padding: 6,
            background: "rgba(255,209,102,0.08)",
            borderRadius: 4,
            border: "1px solid rgba(255,209,102,0.3)",
          }}
        >
          Disagreement at this step: taken <b>{takenAction}</b>, network would pick <b>{argmax}</b>.
        </div>
      )}
    </div>
  );
}
