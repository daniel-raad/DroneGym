import React from "react";
import type { EpisodeResponse } from "../api";

type Props = {
  episode: EpisodeResponse | null;
  overlay: EpisodeResponse | null;
  overlayLLM?: EpisodeResponse | null;
  hasEnv: boolean;
  hasTrainedModel: boolean;
};

export function ExplainerCard({
  episode,
  overlay,
  overlayLLM,
  hasEnv,
  hasTrainedModel,
}: Props) {
  let title = "Welcome";
  let body: React.ReactNode = (
    <>
      DroneGym is a tiny lab for teaching a neural network to fly a 2D drone. Walk
      through the steps on the left — each one builds on the last. The big square
      is the practice room.
    </>
  );

  if (hasEnv && !episode) {
    title = "World ready";
    body = (
      <>
        The <span style={{ color: "#5db8ff" }}>blue dot</span> is where the drone
        starts. The <span style={{ color: "#59e08b" }}>green circle</span> is the
        target it has to reach. <span style={{ color: "#b65dff" }}>Purple discs</span>{" "}
        are obstacles — flying into them ends the run.
        {hasTrainedModel && (
          <>
            <br />
            <br />
            <span style={{ color: "var(--muted)" }}>
              Your network is still trained from before. You can <b>race them</b> on this room
              (step 5) or <b>fly the teacher first</b> using the button on step 2.
            </span>
          </>
        )}
      </>
    );
  }

  if (episode && !overlay) {
    const verdict = episode.success
      ? "reached the target"
      : episode.collision
        ? "crashed into something"
        : "ran out of time";
    const isBC =
      episode.summary.agent_type.startsWith("trained:") ||
      episode.summary.agent_type === "trained";
    title = isBC ? "Neural network flight" : "Rule-based pilot flight";
    body = (
      <>
        The drone {verdict} in {episode.steps} steps along the{" "}
        <span style={{ color: "#5db8ff" }}>blue path</span>. Each step the drone
        reads 6 numbers from its sensors (distance + angle to the target, three
        raycasts, battery) and picks one of 8 actions.{" "}
        {isBC ? (
          <>
            This pilot is the trained neural network — it has no idea what an
            obstacle <i>is</i>, only what the sensor numbers look like.
          </>
        ) : (
          <>
            This pilot follows hand-written rules and is the "teacher" that the
            neural network will learn to copy.
          </>
        )}
        {episode.timeout && (
          <>
            <br />
            <br />
            <span style={{ color: "var(--muted)" }}>
              <b>Why it got stuck:</b> the drone only has 3 raycasts — straight ahead,
              hard left, hard right. <b>Diagonals are invisible.</b> Scrub the timeline
              and watch the three dashed rays from the drone — when obstacles sit
              between them at 30–60° off heading, the rules see "clear" and the drone
              oscillates between turning and strafing without making progress.
            </span>
          </>
        )}
      </>
    );
  }

  if (overlay || overlayLLM) {
    title = "Race result";
    const lineFor = (label: string, ep: EpisodeResponse | null | undefined, color: string) =>
      ep ? (
        <li>
          <span style={{ color }}>●</span> {label} —{" "}
          {ep.success ? "✓ reached target" : ep.collision ? "✗ crashed" : "ran out of time"} ·{" "}
          {ep.steps} steps
        </li>
      ) : null;
    body = (
      <>
        Same room, same start, same target. Each pilot only sees the 6 sensor numbers.
        <ul style={{ margin: "8px 0 0 16px", padding: 0, lineHeight: 1.6 }}>
          {lineFor("rule-based pilot", episode, "#5db8ff")}
          {lineFor("neural network (BC/RL)", overlay, "#ffd166")}
          {lineFor("LLM pilot", overlayLLM, "#b65dff")}
        </ul>
        {overlayLLM && (
          <div style={{ color: "var(--muted)", marginTop: 8, fontSize: 11, lineHeight: 1.5 }}>
            The LLM hasn't been trained on this task — it just reasons about the JSON
            observation each step. Watch how its trajectory differs from the trained
            network's: usually more "thoughtful" looking, sometimes worse, sometimes
            unexpectedly creative.
          </div>
        )}
      </>
    );
  }

  return (
    <div
      className="section"
      style={{ borderColor: "var(--accent)", background: "rgba(93,184,255,0.04)" }}
    >
      <h2 style={{ color: "var(--accent)" }}>{title}</h2>
      <div style={{ fontSize: 13, lineHeight: 1.55 }}>{body}</div>
    </div>
  );
}
