import React from "react";
import type { EpisodeResponse } from "../api";

type Props = {
  episode: EpisodeResponse | null;
  step: number;
};

function bearing(deg: number): string {
  if (Math.abs(deg) < 5) return "straight ahead";
  if (deg > 0) return `${Math.round(Math.abs(deg))}° to its left`;
  return `${Math.round(Math.abs(deg))}° to its right`;
}

function describeDistance(d: number, max = 6): string {
  if (d >= max - 0.05) return "clear (no obstacle in range)";
  if (d < 0.5) return "BLOCKED";
  if (d < 1.0) return "very close";
  return `clear ${d.toFixed(1)} cells`;
}

export function SensorPanel({ episode, step }: Props) {
  if (!episode || episode.observations.length === 0) return null;
  const obs = episode.observations[Math.min(step, episode.observations.length - 1)];
  const action = episode.actions[Math.min(step, episode.actions.length - 1)];

  return (
    <div className="section">
      <h2>What the drone sees (step {obs.step})</h2>
      <div style={{ fontSize: 12, lineHeight: 1.6 }}>
        <div>
          <span style={{ color: "var(--muted)" }}>Distance to target:</span>{" "}
          <strong>{obs.distance_to_target.toFixed(1)} cells</strong>
        </div>
        <div>
          <span style={{ color: "var(--muted)" }}>Bearing to target:</span>{" "}
          <strong>{obs.target_angle_deg.toFixed(0)}°</strong>{" "}
          <span style={{ color: "var(--muted)" }}>({bearing(obs.target_angle_deg)})</span>
        </div>
        <div style={{ marginTop: 6 }}>
          <span style={{ color: "#59e08b" }}>Front raycast:</span>{" "}
          <strong>{obs.front_distance.toFixed(2)}</strong>{" "}
          <span style={{ color: "var(--muted)" }}>— {describeDistance(obs.front_distance)}</span>
        </div>
        <div>
          <span style={{ color: "#5db8ff" }}>Left raycast (heading +90°):</span>{" "}
          <strong>{obs.left_distance.toFixed(2)}</strong>{" "}
          <span style={{ color: "var(--muted)" }}>— {describeDistance(obs.left_distance)}</span>
        </div>
        <div>
          <span style={{ color: "#ffd166" }}>Right raycast (heading −90°):</span>{" "}
          <strong>{obs.right_distance.toFixed(2)}</strong>{" "}
          <span style={{ color: "var(--muted)" }}>— {describeDistance(obs.right_distance)}</span>
        </div>
        <div style={{ marginTop: 6 }}>
          <span style={{ color: "var(--muted)" }}>Battery:</span>{" "}
          <strong>{obs.battery.toFixed(0)}%</strong>
        </div>
        {action && (
          <div style={{ marginTop: 6 }}>
            <span style={{ color: "var(--muted)" }}>Chose action:</span>{" "}
            <strong style={{ color: "var(--accent)" }}>{action}</strong>
          </div>
        )}
        <div
          style={{
            color: "var(--muted)",
            marginTop: 8,
            fontSize: 11,
            background: "var(--panel)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            padding: 6,
            lineHeight: 1.5,
          }}
        >
          These are the only 6 numbers the network ever sees. The teacher decides what
          to do from them with hand-written rules; the network learns the mapping by
          watching ~1,500 of these (obs → action) pairs. <b>Diagonals are invisible</b> —
          if an obstacle is at 45° off the heading, none of the rays catch it.
        </div>
      </div>
    </div>
  );
}
