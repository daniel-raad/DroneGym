import React from "react";
import type { EpisodeResponse, PolicyInspectResponse } from "../api";
import type { Pilot } from "./pilots";

type Props = {
  pilot: Pilot;
  episode: EpisodeResponse | null;
  step: number;
  inspect: PolicyInspectResponse | null;
};

// The thing that makes this product different from every other AI dashboard:
// you can SEE what the pilot is thinking, step by step.
export function MindStream({ pilot, episode, step, inspect }: Props) {
  return (
    <aside className="arc-mind" style={{ ['--hue' as any]: pilot.hue }}>
      <div className="arc-mind-head">
        <div className="arc-mind-face">{pilot.face}</div>
        <div>
          <div className="arc-mind-name">{pilot.name}</div>
          <div className="arc-mind-sub">mind stream</div>
        </div>
      </div>

      {!episode ? (
        <div className="arc-mind-empty">
          <div className="arc-quote">“{pilot.voiceLine}”</div>
          <p>{pilot.bio}</p>
        </div>
      ) : pilot.id === "heuristic" ? (
        <HeuristicMind episode={episode} step={step} />
      ) : pilot.backend.kind === "trained" ? (
        <TrainedMind episode={episode} step={step} inspect={inspect} />
      ) : pilot.id === "llm" ? (
        <LLMMind episode={episode} step={step} />
      ) : (
        <RandomMind episode={episode} step={step} />
      )}
    </aside>
  );
}

function currentBits(episode: EpisodeResponse, step: number) {
  const obs = episode.observations[Math.min(step, episode.observations.length - 1)];
  const action = episode.actions[Math.min(step, episode.actions.length - 1)];
  return { obs, action };
}

function ActionPill({ action }: { action: string | undefined }) {
  if (!action) return null;
  return <span className="arc-action-pill">{action.replace(/_/g, " ")}</span>;
}

function RayBars({
  front,
  left,
  right,
}: {
  front: number;
  left: number;
  right: number;
}) {
  const SAFE = 0.9;
  const bar = (val: number) => Math.min(1, val / 3);
  const cls = (val: number) => (val < SAFE ? "danger" : "clear");
  return (
    <div className="arc-rays">
      <div className="arc-ray-row">
        <span className="arc-ray-label">L</span>
        <div className={`arc-ray-bar ${cls(left)}`}>
          <div className="arc-ray-fill" style={{ width: `${bar(left) * 100}%` }} />
        </div>
        <span className="arc-ray-val">{left.toFixed(2)}</span>
      </div>
      <div className="arc-ray-row">
        <span className="arc-ray-label">F</span>
        <div className={`arc-ray-bar ${cls(front)}`}>
          <div className="arc-ray-fill" style={{ width: `${bar(front) * 100}%` }} />
        </div>
        <span className="arc-ray-val">{front.toFixed(2)}</span>
      </div>
      <div className="arc-ray-row">
        <span className="arc-ray-label">R</span>
        <div className={`arc-ray-bar ${cls(right)}`}>
          <div className="arc-ray-fill" style={{ width: `${bar(right) * 100}%` }} />
        </div>
        <span className="arc-ray-val">{right.toFixed(2)}</span>
      </div>
    </div>
  );
}

function HeuristicMind({ episode, step }: { episode: EpisodeResponse; step: number }) {
  const { obs, action } = currentBits(episode, step);
  if (!obs) return null;

  // Mirror the rule cascade in heuristic_agent.py so the user sees WHICH
  // rule fired at this step.
  let rule = "aim & advance";
  if (obs.distance_to_target < 0.7) rule = "land — target under nose";
  else if (obs.front_distance < 1.2)
    rule =
      obs.left_distance >= obs.right_distance
        ? "blocked ahead → sidestep left"
        : "blocked ahead → sidestep right";
  else if (obs.left_distance < 0.5) rule = "obstacle on left → sidestep right";
  else if (obs.right_distance < 0.5) rule = "obstacle on right → sidestep left";
  else if (obs.target_angle_deg > 18) rule = "target far left → turn left";
  else if (obs.target_angle_deg < -18) rule = "target far right → turn right";
  else rule = "lane clear → move forward";

  return (
    <div className="arc-mind-body">
      <div className="arc-mind-row">
        <div className="arc-mind-k">range</div>
        <div className="arc-mind-v">{obs.distance_to_target.toFixed(2)} cells</div>
      </div>
      <div className="arc-mind-row">
        <div className="arc-mind-k">bearing</div>
        <div className="arc-mind-v">{obs.target_angle_deg.toFixed(0)}°</div>
      </div>
      <RayBars front={obs.front_distance} left={obs.left_distance} right={obs.right_distance} />
      <div className="arc-mind-rule">
        <div className="arc-mind-rule-label">rule fired</div>
        <div className="arc-mind-rule-text">{rule}</div>
      </div>
      <div className="arc-mind-action">
        <span>action</span>
        <ActionPill action={action} />
      </div>
    </div>
  );
}

function TrainedMind({
  episode,
  step,
  inspect,
}: {
  episode: EpisodeResponse;
  step: number;
  inspect: PolicyInspectResponse | null;
}) {
  const { action } = currentBits(episode, step);
  if (!inspect) {
    return (
      <div className="arc-mind-body">
        <div className="arc-mind-empty small">peering inside the network…</div>
        <div className="arc-mind-action">
          <span>action</span>
          <ActionPill action={action} />
        </div>
      </div>
    );
  }
  const i = Math.min(step, inspect.probs.length - 1);
  const probs = inspect.probs[i] ?? [];
  const argmax = inspect.argmax[i];
  const ranked = inspect.actions
    .map((a, k) => ({ a, p: probs[k] ?? 0 }))
    .sort((x, y) => y.p - x.p);

  return (
    <div className="arc-mind-body">
      <div className="arc-mind-sublabel">neuron output · softmax over actions</div>
      <div className="arc-prob-list">
        {ranked.map(({ a, p }) => (
          <div key={a} className={`arc-prob-row ${a === argmax ? "top" : ""}`}>
            <div className="arc-prob-name">{a.replace(/_/g, " ")}</div>
            <div className="arc-prob-track">
              <div className="arc-prob-fill" style={{ width: `${Math.max(2, p * 100)}%` }} />
            </div>
            <div className="arc-prob-val">{(p * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>
      <div className="arc-mind-action">
        <span>chose</span>
        <ActionPill action={action} />
      </div>
    </div>
  );
}

function LLMMind({ episode, step }: { episode: EpisodeResponse; step: number }) {
  const { action } = currentBits(episode, step);
  const recent = episode.actions
    .slice(Math.max(0, step - 5), step + 1)
    .map((a, i, arr) => ({ a, n: step - (arr.length - 1) + i }));

  return (
    <div className="arc-mind-body">
      <div className="arc-mind-sublabel">claude · per-step reasoning · last 6 steps</div>
      <div className="arc-llm-feed">
        {recent.map((r) => (
          <div key={r.n} className={`arc-llm-line ${r.n === step ? "now" : ""}`}>
            <span className="arc-llm-step">t{r.n.toString().padStart(2, "0")}</span>
            <span className="arc-llm-action">{r.a.replace(/_/g, " ")}</span>
          </div>
        ))}
      </div>
      <div className="arc-mind-empty small">
        each line is one LLM call — slow, but you can almost see it think.
      </div>
      <div className="arc-mind-action">
        <span>now</span>
        <ActionPill action={action} />
      </div>
    </div>
  );
}

function RandomMind({ episode, step }: { episode: EpisodeResponse; step: number }) {
  const { action } = currentBits(episode, step);
  const chaos = Math.abs(Math.sin(step * 13.37)) * 100;
  return (
    <div className="arc-mind-body">
      <div className="arc-mind-sublabel">no plan · no model · just vibes</div>
      <div className="arc-chaos-meter">
        <div className="arc-chaos-label">CHAOS</div>
        <div className="arc-chaos-track">
          <div className="arc-chaos-fill" style={{ width: `${chaos}%` }} />
        </div>
      </div>
      <div className="arc-mind-empty small">
        Static rolls a die every step. Whether it lands counts as art.
      </div>
      <div className="arc-mind-action">
        <span>rolled</span>
        <ActionPill action={action} />
      </div>
    </div>
  );
}
