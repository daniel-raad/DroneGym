// Knob metadata — how the Customize UI renders each stage's fields.
//
// Each KnobMeta tells the renderer: what kind of input, how to format the
// stored value for display, how to parse user input back, and — most
// importantly — what changing this knob ACTUALLY DOES to the pilot you're
// training. The description is shown inline under the input. It should
// answer: "what does this control, and how does it change the final pilot?".

import type { Stage } from "./plan";

export type Tone = "primary" | "advanced";

export type KnobUI =
  | { kind: "int"; min?: number; max?: number; step?: number }
  | { kind: "float"; min?: number; max?: number; step?: number; decimals?: number }
  | { kind: "select-int"; options: number[] }
  | { kind: "select-str"; options: { value: string; label: string }[] }
  | { kind: "pct"; max?: number } // store 0..1, display 0..100%
  | { kind: "csv-int" }            // store number[], display "1,2,3"
  | { kind: "bool" };

export type KnobMeta = {
  key: string;
  label: string;
  effect: string; // one short sentence: what the knob changes + how that lands on the pilot
  ui: KnobUI;
  tone: Tone;
};

export type StageMeta = {
  title: string;
  what: string; // one sentence: what this stage does in the pipeline
  knobs: KnobMeta[];
};

export const STAGE_META: Record<Stage["kind"], StageMeta> = {
  collect: {
    title: "Collect demos",
    what: "Sgt. Heuristic flies N rooms; the successful flights become your pilot's training data.",
    knobs: [
      {
        key: "num_episodes",
        label: "rooms to fly",
        effect:
          "How many demo flights Sgt does. More flights = your pilot sees more situations = handles novel rooms better. Diminishing returns past ~1500.",
        ui: { kind: "int", min: 50, max: 5000, step: 50 },
        tone: "primary",
      },
      {
        key: "difficulties",
        label: "difficulty mix",
        effect:
          "Which rooms the teacher flies. [1] = easy only — pilot fails on harder courses. [1,2,3] = mixed — pilot handles variety but takes longer to learn.",
        ui: { kind: "csv-int" },
        tone: "primary",
      },
      {
        key: "num_rays",
        label: "sensor density",
        effect:
          "The drone's 'eyes'. 3 = only front/left/right — blind to diagonal obstacles, will crash into them. 16+ = lidar sweep, sees everything.",
        ui: { kind: "select-int", options: [3, 8, 16, 32] },
        tone: "primary",
      },
    ],
  },
  bc: {
    title: "BC fit",
    what: "The neural network watches the demos and learns to mimic Sgt's actions in each situation.",
    knobs: [
      {
        key: "epochs",
        label: "training passes",
        effect:
          "How many times the network reviews every demo. Too few = pilot half-trained, mimics teacher poorly. Too many = pilot memorizes specific rooms, fails on new ones (overfitting). 10–20 is the sweet spot.",
        ui: { kind: "int", min: 1, max: 100, step: 1 },
        tone: "primary",
      },
      {
        key: "hidden_size",
        label: "brain width",
        effect:
          "Neurons per layer. Bigger = pilot can handle more nuanced situations, but needs more demos to avoid overfitting. 64 is fine for easy courses; 128+ for Tight.",
        ui: { kind: "select-int", options: [32, 64, 128, 256] },
        tone: "advanced",
      },
      {
        key: "num_layers",
        label: "brain depth",
        effect:
          "Number of hidden layers. Deeper = can learn more complex decisions but harder to train without more data. 2 is right for easy, 3+ when you add curriculum.",
        ui: { kind: "select-int", options: [1, 2, 3, 4] },
        tone: "advanced",
      },
      {
        key: "batch_size",
        label: "batch size",
        effect:
          "How many demos the network looks at per update. Bigger = smoother gradient, less noisy training. Smaller = faster per-step but jitterier convergence.",
        ui: { kind: "select-int", options: [32, 64, 128, 256] },
        tone: "advanced",
      },
      {
        key: "learning_rate",
        label: "learning rate",
        effect:
          "How big a step the network takes per update. Too high = pilot's behavior gets erratic, may diverge. Too low = pilot never finishes learning. 0.001 is a safe default.",
        ui: {
          kind: "select-str",
          options: [
            { value: "0.0001", label: "0.0001 (slow, steady)" },
            { value: "0.0003", label: "0.0003" },
            { value: "0.001", label: "0.001 (default)" },
            { value: "0.003", label: "0.003" },
            { value: "0.01", label: "0.01 (fast, risky)" },
          ],
        },
        tone: "advanced",
      },
    ],
  },
  dagger: {
    title: "DAgger refinement",
    what: "Your pilot flies for real, makes mistakes, and the teacher relabels what it SHOULD have done. Fixes BC's blind spots.",
    knobs: [
      {
        key: "rounds",
        label: "refinement rounds",
        effect:
          "How many cycles of (fly → relabel → retrain). More rounds = pilot's mistakes get progressively cleaned up. 2–4 is usually enough.",
        ui: { kind: "int", min: 1, max: 10, step: 1 },
        tone: "primary",
      },
      {
        key: "episodes_per_round",
        label: "rooms per round",
        effect:
          "How many rooms the pilot flies before the teacher relabels. More = covers more failure cases per round, but slower.",
        ui: { kind: "int", min: 10, max: 500, step: 10 },
        tone: "primary",
      },
      {
        key: "epochs",
        label: "refit epochs",
        effect:
          "How many passes through the (now bigger) dataset after each round. Lower than initial BC since you're just incorporating new examples.",
        ui: { kind: "int", min: 1, max: 30, step: 1 },
        tone: "advanced",
      },
    ],
  },
  rl: {
    title: "RL practice",
    what: "The pilot trains itself through trial and error in fresh rooms. Slower than BC, but the only path to BEAT the teacher.",
    knobs: [
      {
        key: "algorithm",
        label: "algorithm",
        effect:
          "A2C is simpler and faster per-step but noisier. PPO is more stable and usually produces a better final pilot — recommended once you've got the basics working.",
        ui: {
          kind: "select-str",
          options: [
            { value: "a2c", label: "A2C (simple, fast)" },
            { value: "ppo", label: "PPO (stable, recommended)" },
          ],
        },
        tone: "primary",
      },
      {
        key: "episodes",
        label: "practice episodes",
        effect:
          "How many practice rooms the pilot flies. More = more refined behavior. 500 = quick polish, 2000+ = pilot can meaningfully outperform the teacher.",
        ui: { kind: "int", min: 100, max: 10000, step: 100 },
        tone: "primary",
      },
      {
        key: "curriculum_schedule",
        label: "difficulty curriculum",
        effect:
          "Sequence of difficulties cycled per episode. [1] = pilot only sees easy rooms — fails on Tight. [1,1,2,2,3] = pilot gets harder rooms gradually, generalizes better.",
        ui: { kind: "csv-int" },
        tone: "primary",
      },
      {
        key: "entropy_coef",
        label: "exploration",
        effect:
          "How much the pilot tries random actions during RL. Higher = explores weirder strategies, escapes plateaus. Lower = commits to what it knows. Raise to 0.05+ if training stalls.",
        ui: { kind: "float", min: 0, max: 0.2, step: 0.005, decimals: 3 },
        tone: "advanced",
      },
      {
        key: "warm_start",
        label: "start from BC weights",
        effect:
          "ON = RL builds on what BC already learned. Much faster, much better final pilot. OFF = RL starts random — usually fails to converge in reasonable time.",
        ui: { kind: "bool" },
        tone: "advanced",
      },
      {
        key: "hidden_size",
        label: "brain width",
        effect:
          "Same as BC's brain width. Should match the BC stage if warm-starting. Bigger nets need more episodes to train.",
        ui: { kind: "select-int", options: [32, 64, 128, 256] },
        tone: "advanced",
      },
      {
        key: "num_layers",
        label: "brain depth",
        effect:
          "Same as BC's brain depth. Should match the BC stage if warm-starting.",
        ui: { kind: "select-int", options: [1, 2, 3, 4] },
        tone: "advanced",
      },
      {
        key: "randomize_wind",
        label: "wind variation",
        effect:
          "Random wind strength per episode. 0% = pilot expects perfect conditions, fails when conditions vary. 5–10% = pilot learns to compensate for drift, more robust.",
        ui: { kind: "pct", max: 0.2 },
        tone: "advanced",
      },
      {
        key: "randomize_noise",
        label: "sensor noise",
        effect:
          "Random Gaussian noise on sensor readings. 0 = perfect sensors. 5%+ = pilot learns to handle imperfect readings, transfers better to messy worlds.",
        ui: { kind: "pct", max: 0.2 },
        tone: "advanced",
      },
      {
        key: "randomize_obstacles",
        label: "±obstacles",
        effect:
          "How much the obstacle count varies. 0 = always exact N. 1 = sometimes N±1 — pilot adapts to room density, doesn't lock onto one layout.",
        ui: { kind: "int", min: 0, max: 5, step: 1 },
        tone: "advanced",
      },
      {
        key: "batch_episodes",
        label: "episodes per update",
        effect:
          "How many episodes the pilot completes before updating the network. Higher = more stable gradient but slower per-minute progress.",
        ui: { kind: "select-int", options: [4, 8, 16, 32] },
        tone: "advanced",
      },
      {
        key: "gamma",
        label: "future reward weight",
        effect:
          "How much the pilot cares about long-term reward vs. immediate. Higher = plans further ahead. 0.97–0.99 is the normal range.",
        ui: { kind: "float", min: 0.8, max: 0.999, step: 0.01, decimals: 3 },
        tone: "advanced",
      },
      {
        key: "learning_rate",
        label: "learning rate",
        effect:
          "Same idea as BC's learning rate but RL is much more sensitive — too high will destroy the BC weights. 0.0003 is the safe default.",
        ui: {
          kind: "select-str",
          options: [
            { value: "0.0001", label: "0.0001 (very slow)" },
            { value: "0.0003", label: "0.0003 (default)" },
            { value: "0.001", label: "0.001 (risky)" },
          ],
        },
        tone: "advanced",
      },
    ],
  },
  eval: {
    title: "Evaluate (3 courses)",
    what: "Score the pilot against Breeze, Cruise, and Tight. This sets the readiness meters at the top.",
    knobs: [
      {
        key: "runs_per_course",
        label: "runs per course",
        effect:
          "How many attempts on each course. More = the success rate you see is statistically meaningful, not luck. 20 is reasonable; 50 if you're comparing two pilots.",
        ui: { kind: "int", min: 5, max: 100, step: 5 },
        tone: "primary",
      },
    ],
  },
};

// ---- Value formatting + parsing ------------------------------------------
// One pair per UI kind. Renderer calls format(value) for display and
// parse(input) when the user types.

export function formatValue(ui: KnobUI, v: any): string {
  if (ui.kind === "csv-int") return Array.isArray(v) ? v.join(",") : "";
  if (ui.kind === "pct") return `${Math.round((Number(v) || 0) * 100)}`;
  if (ui.kind === "float") {
    const d = ui.decimals ?? 3;
    return Number(v ?? 0).toFixed(d);
  }
  if (ui.kind === "select-str") return String(v ?? "");
  if (ui.kind === "bool") return v ? "on" : "off";
  return String(v ?? "");
}

export function parseValue(ui: KnobUI, raw: string, prev: any): any {
  if (ui.kind === "csv-int") {
    return raw
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n));
  }
  if (ui.kind === "pct") {
    const n = parseFloat(raw);
    if (isNaN(n)) return prev;
    return Math.max(0, Math.min(1, n / 100));
  }
  if (ui.kind === "float") {
    const n = parseFloat(raw);
    return isNaN(n) ? prev : n;
  }
  if (ui.kind === "int" || ui.kind === "select-int") {
    const n = parseInt(raw, 10);
    return isNaN(n) ? prev : n;
  }
  if (ui.kind === "bool") {
    return raw === "true" || raw === "on";
  }
  return raw;
}

// ---- Plan summary --------------------------------------------------------
// Turn the assembled plan into a paragraph of English so the user can read
// what they're about to train BEFORE pressing the button. Updates live.

export function summarizePlan(stages: Stage[]): string {
  if (stages.length === 0) return "Empty plan — add a stage to get started.";
  const parts: string[] = [];
  for (const s of stages) {
    parts.push(summarizeStage(s));
  }
  return parts.join(" Then ").replace(/^./, (c) => c.toUpperCase()) + ".";
}

function summarizeStage(stage: Stage): string {
  if (stage.kind === "collect") {
    const diffs = stage.difficulties.length === 1
      ? `difficulty ${stage.difficulties[0]}`
      : `mixed difficulties ${stage.difficulties.join("/")}`;
    return `collect ${stage.num_episodes} demo flights with ${stage.num_rays}-ray sensors on ${diffs}`;
  }
  if (stage.kind === "bc") {
    return `train a ${stage.num_layers}×${stage.hidden_size} MLP for ${stage.epochs} epochs to mimic the teacher`;
  }
  if (stage.kind === "dagger") {
    return `run ${stage.rounds} DAgger refinement rounds (${stage.episodes_per_round} rooms each), refitting the network after each round`;
  }
  if (stage.kind === "rl") {
    const c = stage.curriculum_schedule;
    const curr = c.length <= 1
      ? `difficulty ${c[0] ?? 1}`
      : `a ${c.join("/")} curriculum`;
    const rand: string[] = [];
    if (stage.randomize_wind > 0) rand.push(`wind ±${Math.round(stage.randomize_wind * 100)}%`);
    if (stage.randomize_noise > 0) rand.push(`sensor noise ±${Math.round(stage.randomize_noise * 100)}%`);
    if (stage.randomize_obstacles > 0) rand.push(`±${stage.randomize_obstacles} obstacles`);
    const randPart = rand.length ? ` with random ${rand.join(", ")}` : "";
    const warm = stage.warm_start ? "" : " from random weights";
    return `practice with ${stage.algorithm.toUpperCase()} for ${stage.episodes} episodes on ${curr}${randPart}${warm}`;
  }
  if (stage.kind === "eval") {
    return `evaluate against all three arcade courses (${stage.runs_per_course} runs each)`;
  }
  return "";
}
