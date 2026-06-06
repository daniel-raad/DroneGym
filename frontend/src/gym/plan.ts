// The "training plan" is a vertical sequence of stages the user assembles.
// The runner executes them one after another, calling existing endpoints.
// Adding a new stage type = (a) extend Stage union (b) handle it in runner.

export type StageBase = { id: string };

export type CollectStage = StageBase & {
  kind: "collect";
  num_episodes: number;
  difficulties: number[];     // mix
  num_rays: number;
};

export type BCStage = StageBase & {
  kind: "bc";
  epochs: number;
  batch_size: number;
  learning_rate: number;
  hidden_size: number;
  num_layers: number;
};

export type DAggerStage = StageBase & {
  kind: "dagger";
  rounds: number;
  episodes_per_round: number;
  epochs: number;
};

export type RLStage = StageBase & {
  kind: "rl";
  algorithm: "a2c" | "ppo";
  episodes: number;
  learning_rate: number;
  gamma: number;
  entropy_coef: number;
  hidden_size: number;
  num_layers: number;
  batch_episodes: number;
  ppo_clip: number;
  ppo_epochs: number;
  curriculum_schedule: number[];
  randomize_wind: number;
  randomize_noise: number;
  randomize_obstacles: number;
  warm_start: boolean;
};

export type EvalStage = StageBase & {
  kind: "eval";
  runs_per_course: number;
};

export type Stage = CollectStage | BCStage | DAggerStage | RLStage | EvalStage;

let nextId = 1;
export const newId = () => `s${nextId++}`;

export const STAGE_LABELS: Record<Stage["kind"], string> = {
  collect: "Collect demos",
  bc: "BC fit",
  dagger: "DAgger",
  rl: "RL practice",
  eval: "Evaluate (3 courses)",
};

// ---- Recipes ---------------------------------------------------------------
// Each recipe returns a fresh plan. New ids per call so the UI's keys behave.

export type RecipeName =
  | "clone-the-teacher"
  | "beat-the-teacher"
  | "robust-pilot"
  | "minimum-viable";

export type RecipeMeta = {
  id: RecipeName;
  name: string;
  blurb: string;     // one-line "what this builds"
  time: string;      // human estimate
  steps: string[];   // ordered short names for the progress strip
  expected: { courseId: "breeze" | "cruise" | "tight"; label: string }[];
};

export const RECIPE_META: Record<RecipeName, RecipeMeta> = {
  "clone-the-teacher": {
    id: "clone-the-teacher",
    name: "Clone the teacher",
    blurb:
      "Watch Sgt. Heuristic fly 400 rooms and copy them. Fastest path to a working pilot — but caps at the teacher's skill.",
    time: "~30s",
    steps: ["Collect demos", "BC fit", "Evaluate"],
    expected: [
      { courseId: "breeze", label: "Breeze ✓" },
      { courseId: "cruise", label: "Cruise · likely" },
      { courseId: "tight", label: "Tight · won't pass" },
    ],
  },
  "minimum-viable": {
    id: "minimum-viable",
    name: "Minimum viable",
    blurb:
      "Clone the teacher, then practice with 800 RL episodes. First recipe that can beat the teacher.",
    time: "~1 min",
    steps: ["Collect demos", "BC fit", "RL practice", "Evaluate"],
    expected: [
      { courseId: "breeze", label: "Breeze ✓" },
      { courseId: "cruise", label: "Cruise ✓" },
      { courseId: "tight", label: "Tight · maybe" },
    ],
  },
  "beat-the-teacher": {
    id: "beat-the-teacher",
    name: "Beat the teacher",
    blurb:
      "Clone + DAgger refinement + PPO with a light curriculum. Aims to clear all three courses.",
    time: "~2 min",
    steps: ["Collect demos", "BC fit", "DAgger", "PPO practice", "Evaluate"],
    expected: [
      { courseId: "breeze", label: "Breeze ✓" },
      { courseId: "cruise", label: "Cruise ✓" },
      { courseId: "tight", label: "Tight · likely" },
    ],
  },
  "robust-pilot": {
    id: "robust-pilot",
    name: "Robust pilot",
    blurb:
      "Heavier net + full curriculum (D1→D3) + wind/noise/obstacle randomization. Slowest but most reliable.",
    time: "~4 min",
    steps: ["Collect demos", "BC fit", "DAgger", "PPO + randomization", "Evaluate"],
    expected: [
      { courseId: "breeze", label: "Breeze ✓" },
      { courseId: "cruise", label: "Cruise ✓" },
      { courseId: "tight", label: "Tight ✓" },
    ],
  },
};

export function recipe(name: RecipeName): Stage[] {
  if (name === "clone-the-teacher") {
    return [
      mkCollect({ num_episodes: 400, difficulties: [1], num_rays: 16 }),
      mkBC({ epochs: 15 }),
      mkEval(),
    ];
  }
  if (name === "minimum-viable") {
    return [
      mkCollect({ num_episodes: 200, difficulties: [1], num_rays: 16 }),
      mkBC({ epochs: 10 }),
      mkRL({ episodes: 800, warm_start: true }),
      mkEval(),
    ];
  }
  if (name === "beat-the-teacher") {
    return [
      mkCollect({ num_episodes: 400, difficulties: [1, 2], num_rays: 16 }),
      mkBC({ epochs: 15 }),
      mkDAgger({ rounds: 3, episodes_per_round: 60 }),
      mkRL({
        algorithm: "ppo",
        episodes: 1500,
        curriculum_schedule: [1, 1, 2],
        entropy_coef: 0.04,
        warm_start: true,
      }),
      mkEval(),
    ];
  }
  // robust-pilot
  return [
    mkCollect({ num_episodes: 500, difficulties: [1, 2, 3], num_rays: 16 }),
    mkBC({ epochs: 20 }),
    mkDAgger({ rounds: 3, episodes_per_round: 80 }),
    mkRL({
      algorithm: "ppo",
      episodes: 2500,
      curriculum_schedule: [1, 1, 2, 2, 3],
      entropy_coef: 0.05,
      randomize_wind: 0.05,
      randomize_noise: 0.05,
      randomize_obstacles: 1,
      hidden_size: 128,
      num_layers: 3,
      warm_start: true,
    }),
    mkEval(),
  ];
}

// ---- Constructors with sensible defaults -----------------------------------

export function mkCollect(p: Partial<CollectStage> = {}): CollectStage {
  return {
    id: newId(),
    kind: "collect",
    num_episodes: 300,
    difficulties: [1],
    num_rays: 16,
    ...p,
  };
}

export function mkBC(p: Partial<BCStage> = {}): BCStage {
  return {
    id: newId(),
    kind: "bc",
    epochs: 10,
    batch_size: 64,
    learning_rate: 1e-3,
    hidden_size: 64,
    num_layers: 2,
    ...p,
  };
}

export function mkDAgger(p: Partial<DAggerStage> = {}): DAggerStage {
  return {
    id: newId(),
    kind: "dagger",
    rounds: 3,
    episodes_per_round: 50,
    epochs: 5,
    ...p,
  };
}

export function mkRL(p: Partial<RLStage> = {}): RLStage {
  return {
    id: newId(),
    kind: "rl",
    algorithm: "a2c",
    episodes: 1000,
    learning_rate: 3e-4,
    gamma: 0.97,
    entropy_coef: 0.02,
    hidden_size: 64,
    num_layers: 2,
    batch_episodes: 8,
    ppo_clip: 0.2,
    ppo_epochs: 4,
    curriculum_schedule: [1],
    randomize_wind: 0,
    randomize_noise: 0,
    randomize_obstacles: 0,
    warm_start: true,
    ...p,
  };
}

export function mkEval(p: Partial<EvalStage> = {}): EvalStage {
  return { id: newId(), kind: "eval", runs_per_course: 20, ...p };
}
