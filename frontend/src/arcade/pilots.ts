// Pilot roster — the "brains" you can pick from. Each pilot maps to a backend
// agent type. Stats are cosmetic for now; voice lines and mind-style drive the
// race-screen personality.

export type PilotId = "heuristic" | "trained" | "llm" | "random" | string;

export type Pilot = {
  id: PilotId;
  name: string;
  face: string; // emoji or short symbol used as the avatar
  tagline: string;
  voiceLine: string;
  bio: string;
  hue: number; // 0-360 for the neon aura color
  // 0..10 cosmetic stat bars
  stats: { reaction: number; planning: number; reliability: number; chaos: number };
  // Backend mapping. Built-ins map to a fixed agent type or to the canonical
  // trained model. User-created pilots are "trained" with an explicit
  // `modelName` so the race uses *their* network, not the default one.
  backend:
    | { kind: "episode"; agentType: "heuristic" | "random" | "llm" }
    | { kind: "trained"; modelName?: string };
  custom?: boolean; // user-created (sourced from localStorage)
};

export const PILOTS: Pilot[] = [
  {
    id: "heuristic",
    name: "Sgt. Heuristic",
    face: "🪖",
    tagline: "Old-school rule-bot. Never improvises.",
    voiceLine: "Path's clear. Moving.",
    bio:
      "Hand-coded since 1987. Reads three raycasts, picks the open lane, lands when the target's under his nose. Predictable to a fault — and that's the point.",
    hue: 145,
    stats: { reaction: 7, planning: 6, reliability: 9, chaos: 1 },
    backend: { kind: "episode", agentType: "heuristic" },
  },
  {
    id: "trained",
    name: "Neon",
    face: "🧠",
    tagline: "Trained MLP. Feels its way through.",
    voiceLine: "…feels right.",
    bio:
      "A 64-neuron network that watched Sgt. Heuristic fly a thousand rooms and decided it could do better. Smooth on familiar tracks, weird on novel ones.",
    hue: 280,
    stats: { reaction: 9, planning: 6, reliability: 6, chaos: 4 },
    backend: { kind: "trained" },
  },
  {
    id: "llm",
    name: "Oracle",
    face: "👁",
    tagline: "Large language model. Thinks out loud.",
    voiceLine: "Permit me to consider…",
    bio:
      "Reads the observation as JSON, narrates a reason, returns an action. Slow per step, occasionally brilliant, occasionally takes the scenic route.",
    hue: 45,
    stats: { reaction: 4, planning: 9, reliability: 6, chaos: 5 },
    backend: { kind: "episode", agentType: "llm" },
  },
  {
    id: "random",
    name: "Static",
    face: "👾",
    tagline: "Pure chaos. Baseline gremlin.",
    voiceLine: "WHEEEEE",
    bio:
      "Flips a weighted coin every step. Exists so the rest of the roster has something to beat. Occasionally wins by accident, which is funnier than it sounds.",
    hue: 0,
    stats: { reaction: 5, planning: 1, reliability: 1, chaos: 10 },
    backend: { kind: "episode", agentType: "random" },
  },
];

export function pilotById(id: PilotId, customs: Pilot[] = []): Pilot {
  return (
    PILOTS.find((p) => p.id === id) ??
    customs.find((p) => p.id === id) ??
    PILOTS[0]
  );
}

// Build a Pilot card from a custom-pilot localStorage record so it slots
// straight into PilotSelect alongside the built-ins.
export function pilotFromCustom(c: {
  id: string;
  modelName: string;
  name: string;
  face: string;
  hue: number;
  practiced: boolean;
  evalSuccess: number | null;
}): Pilot {
  const successPct = c.evalSuccess != null ? `${Math.round(c.evalSuccess * 100)}%` : "untested";
  return {
    id: c.id,
    name: c.name,
    face: c.face,
    tagline: c.practiced
      ? `Trained + practiced rookie · ${successPct}`
      : `Trained rookie · ${successPct}`,
    voiceLine: "Let's see what I learned.",
    bio: c.practiced
      ? `Your own pilot — cloned from Sgt then practiced with reinforcement learning. Behavior shaped by the rooms you trained on.`
      : `Your own pilot — a behavior-cloned MLP. Mimics Sgt's style on rooms like the ones it was trained on.`,
    hue: c.hue,
    stats: {
      reaction: c.practiced ? 8 : 7,
      planning: c.practiced ? 7 : 6,
      reliability: c.evalSuccess != null ? Math.max(3, Math.round(c.evalSuccess * 10)) : 5,
      chaos: c.practiced ? 3 : 4,
    },
    backend: { kind: "trained", modelName: c.modelName },
    custom: true,
  };
}
