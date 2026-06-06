import React from "react";

type Props = {
  active: boolean;
  verdict: "success" | "collision" | "landed_off" | "timeout" | null;
  steps: number;
  lastAction: string | null;
  onActivate: () => void;
  onReset: () => void;
  onExit: () => void;
};

const KEYS: { key: string; label: string }[] = [
  { key: "↑", label: "move_forward" },
  { key: "↓", label: "move_back" },
  { key: "←", label: "strafe_left" },
  { key: "→", label: "strafe_right" },
  { key: "Q", label: "turn_left" },
  { key: "E", label: "turn_right" },
  { key: "H", label: "hover" },
  { key: "Space", label: "land" },
];

export function ManualFlightHud(p: Props) {
  if (!p.active) {
    return (
      <div
        style={{
          padding: 10,
          borderTop: "1px solid var(--border)",
          background: "var(--panel)",
          fontSize: 12,
          color: "var(--muted)",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <span>
          You're watching a replay. Click <b>Take the controls</b> to fly the drone
          yourself with the keyboard.
        </span>
        <span style={{ flex: 1 }} />
        <button
          onClick={p.onActivate}
          style={{ width: "auto", margin: 0, padding: "4px 10px", fontSize: 11 }}
        >
          Take the controls
        </button>
      </div>
    );
  }
  return (
    <div
      style={{
        padding: 10,
        borderTop: "1px solid var(--accent)",
        background: "rgba(93,184,255,0.06)",
        fontSize: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
        <strong style={{ color: "var(--accent)" }}>YOU'RE FLYING</strong>
        <span style={{ color: "var(--muted)" }}>
          step {p.steps}
          {p.lastAction && <> · last: <code>{p.lastAction}</code></>}
        </span>
        <span style={{ flex: 1 }} />
        <button
          onClick={p.onReset}
          className="secondary"
          style={{ width: "auto", margin: 0, padding: "4px 10px", fontSize: 11 }}
        >
          Restart
        </button>
        <button
          onClick={p.onExit}
          className="secondary"
          style={{ width: "auto", margin: 0, padding: "4px 10px", fontSize: 11 }}
        >
          Exit pilot mode
        </button>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {KEYS.map((k) => (
          <span
            key={k.key}
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: 11,
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "2px 6px",
            }}
          >
            <strong style={{ color: "var(--accent)" }}>{k.key}</strong>{" "}
            <span style={{ color: "var(--muted)" }}>{k.label}</span>
          </span>
        ))}
      </div>
      {p.verdict && (
        <div
          style={{
            marginTop: 8,
            padding: 8,
            borderRadius: 4,
            background:
              p.verdict === "success"
                ? "rgba(89,224,139,0.12)"
                : "rgba(255,107,107,0.10)",
            border: `1px solid ${p.verdict === "success" ? "var(--success)" : "var(--danger)"}`,
          }}
        >
          {p.verdict === "success"
            ? "✓ You reached the target! Now imagine writing rules to do this every time — that's the teacher."
            : p.verdict === "collision"
              ? "✗ Crashed. Notice how the 3 raycasts don't catch obstacles to the diagonals."
              : p.verdict === "landed_off"
                ? "Landed too far from the target. Try again, or use Space only when you're close."
                : "⏱ Out of time. The drone has a finite step budget; the rule-based pilot hits this too."}{" "}
          <button
            onClick={p.onReset}
            className="secondary"
            style={{ width: "auto", margin: "6px 0 0", padding: "3px 8px", fontSize: 11 }}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
