import React from "react";
import type { EpisodeResponse, SystemStatus } from "../api";

type Diff = "easy" | "medium" | "hard";

export type GuidedState = {
  difficulty: Diff;
  busy: boolean;
  hasEnv: boolean;
  hasHeuristicRun: boolean;
  hasDataset: boolean;
  hasTrainedModel: boolean;
  hasComparison: boolean;
  episode: EpisodeResponse | null;
  status: SystemStatus | null;
};

type Props = GuidedState & {
  onDifficultyChange: (d: Diff) => void;
  onBuildWorld: () => Promise<void>;
  onRunRuleBased: () => Promise<void>;
  onCollectFlights: () => Promise<void>;
  onTrainNetwork: () => Promise<void>;
  onCompare: () => Promise<void>;
  onManualFlight: () => Promise<void>;
  manualActive: boolean;
  tookManualFlight: boolean;
  llmAvailable: boolean;
  llmModel: string | null;
  includeLLM: boolean;
  onToggleLLM: (b: boolean) => void;
  laboratory?: React.ReactNode;
};

const STEPS_META = [
  {
    n: 1,
    title: "Build a practice world",
    blurb:
      "Drones need a place to fly. We'll generate a small 2D room with random obstacles, a start point, and a target the drone has to reach.",
    cta: "Generate a world",
  },
  {
    n: 2,
    title: "Watch a rule-based pilot",
    blurb:
      "This pilot follows simple, hand-written rules: aim at the target, dodge obstacles in front, land when close. It's our 'teacher' — the neural network will later learn by copying what it does.",
    cta: "Run the rule-based pilot",
  },
  {
    n: 3,
    title: "Collect training examples",
    blurb:
      "We let the rule-based pilot fly ~200 random worlds and save every successful flight as a list of (what the drone sees → what it did). These pairs are the training data.",
    cta: "Collect 200 demo flights",
  },
  {
    n: 4,
    title: "Train a tiny neural network",
    blurb:
      "A small network (6 sensor numbers in → 8 possible actions out) reads the demos and learns the mapping. It doesn't know about obstacles, targets, or physics — only what to do given a set of sensor readings. Takes ~10 seconds.",
    cta: "Train the neural net",
  },
  {
    n: 5,
    title: "Race the pilots",
    blurb:
      "Fly all three pilots on the same room: hand-written rules, the trained neural network, and an LLM. The network has never seen this exact room. The LLM has never been trained — it just reads the 6 sensor numbers each step and decides from a prompt.",
    cta: "Race rule-based vs network vs LLM",
  },
];

function StepBox({
  step,
  active,
  done,
  children,
  doneChildren,
  onActivate,
}: {
  step: (typeof STEPS_META)[number];
  active: boolean;
  done: boolean;
  children?: React.ReactNode;
  doneChildren?: React.ReactNode;
  onActivate?: () => void;
}) {
  return (
    <div
      onClick={!active && !done ? onActivate : undefined}
      style={{
        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
        borderRadius: 8,
        padding: 12,
        marginBottom: 8,
        background: active ? "rgba(93,184,255,0.06)" : "var(--panel-2)",
        opacity: done ? 0.75 : active ? 1 : 0.55,
        cursor: !active && !done ? "pointer" : "default",
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
            background: done
              ? "var(--success)"
              : active
                ? "var(--accent)"
                : "var(--panel)",
            color: done || active ? "#001022" : "var(--muted)",
            flexShrink: 0,
          }}
        >
          {done ? "✓" : step.n}
        </div>
        <strong style={{ fontSize: 13 }}>{step.title}</strong>
      </div>
      {(active || done) && (
        <div
          style={{
            fontSize: 12,
            color: "var(--muted)",
            lineHeight: 1.5,
            marginBottom: active ? 10 : 0,
          }}
        >
          {step.blurb}
        </div>
      )}
      {active && children}
      {done && doneChildren}
    </div>
  );
}

export function GuidedSteps(p: Props) {
  const stepDone = [
    p.hasEnv,
    p.hasHeuristicRun,
    p.hasDataset,
    p.hasTrainedModel,
    p.hasComparison,
  ];
  // Active = first not-done step
  const activeIdx = stepDone.findIndex((d) => !d);
  const active = activeIdx === -1 ? 4 : activeIdx;

  return (
    <div>
      <StepBox
        step={STEPS_META[0]}
        active={active === 0}
        done={stepDone[0]}
        doneChildren={
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 4 }}>
            <button
              className="secondary"
              disabled={p.busy}
              onClick={p.onBuildWorld}
              style={{ padding: "4px 8px", fontSize: 11, margin: 0 }}
            >
              Try a different room ({p.difficulty})
            </button>
            <button
              className="secondary"
              disabled={p.busy}
              onClick={p.onManualFlight}
              style={{
                padding: "4px 8px",
                fontSize: 11,
                margin: 0,
                background: p.tookManualFlight ? undefined : "rgba(93,184,255,0.15)",
                color: p.tookManualFlight ? undefined : "var(--accent)",
                fontWeight: p.tookManualFlight ? 400 : 600,
              }}
            >
              ✈ {p.tookManualFlight ? "Pilot again" : "Try flying it yourself"}
            </button>
          </div>
        }
      >
        <label style={{ marginTop: 0 }}>Difficulty</label>
        <div style={{ display: "flex", gap: 4 }}>
          {(["easy", "medium", "hard"] as Diff[]).map((d) => (
            <button
              key={d}
              className={p.difficulty === d ? "" : "secondary"}
              style={{ margin: 0, padding: "6px 8px", fontSize: 12 }}
              onClick={() => p.onDifficultyChange(d)}
            >
              {d}
            </button>
          ))}
        </div>
        <button disabled={p.busy} onClick={p.onBuildWorld} style={{ marginTop: 10 }}>
          {STEPS_META[0].cta}
        </button>
      </StepBox>

      <StepBox
        step={STEPS_META[1]}
        active={active === 1}
        done={stepDone[1]}
        doneChildren={
          <button
            className="secondary"
            disabled={p.busy || !p.hasEnv}
            onClick={p.onRunRuleBased}
            style={{ marginTop: 4, padding: "4px 8px", fontSize: 11 }}
          >
            Fly teacher on this room
          </button>
        }
      >
        <button disabled={p.busy || !p.hasEnv} onClick={p.onRunRuleBased}>
          {STEPS_META[1].cta}
        </button>
        {p.episode && p.episode.summary.agent_type === "heuristic" && (
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>
            Last flight: {p.episode.success ? "✓ reached target" : p.episode.collision ? "✗ crashed" : "ran out of time"}{" "}
            in {p.episode.steps} steps.
          </div>
        )}
      </StepBox>

      <StepBox step={STEPS_META[2]} active={active === 2} done={stepDone[2]}>
        <button disabled={p.busy} onClick={p.onCollectFlights}>
          {STEPS_META[2].cta}
        </button>
        {p.status?.datasets[0] && (
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>
            Have {p.status.datasets[0].num_samples.toLocaleString()} examples saved.
          </div>
        )}
      </StepBox>

      <StepBox step={STEPS_META[3]} active={active === 3} done={stepDone[3]}>
        <button disabled={p.busy || !p.hasDataset} onClick={p.onTrainNetwork}>
          {STEPS_META[3].cta}
        </button>
        {p.status?.models.find((m) => m.method === "behavior_cloning") && (
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>
            Trained — the network matches the teacher on{" "}
            {(
              (p.status.models.find((m) => m.method === "behavior_cloning")?.test_accuracy ?? 0) * 100
            ).toFixed(0)}
            % of held-out demos.
          </div>
        )}
      </StepBox>

      <StepBox
        step={STEPS_META[4]}
        active={active === 4}
        done={stepDone[4]}
        doneChildren={
          <button
            className="secondary"
            disabled={p.busy}
            onClick={p.onCompare}
            style={{ marginTop: 4, padding: "4px 8px", fontSize: 11 }}
          >
            Race again
          </button>
        }
      >
        <button disabled={p.busy || !p.hasTrainedModel} onClick={p.onCompare}>
          {STEPS_META[4].cta}
        </button>
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 8, lineHeight: 1.5 }}>
          <span style={{ color: "#5db8ff" }}>━━</span> teacher ·{" "}
          <span style={{ color: "#ffd166" }}>┄┄</span> neural net ·{" "}
          <span style={{ color: "#b65dff" }}>···</span> LLM
        </div>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginTop: 6,
            fontSize: 11,
            color: p.llmAvailable ? "var(--text)" : "var(--muted)",
          }}
        >
          <input
            type="checkbox"
            checked={p.includeLLM && p.llmAvailable}
            disabled={!p.llmAvailable}
            onChange={(e) => p.onToggleLLM(e.target.checked)}
            style={{ width: "auto", margin: 0 }}
          />
          Include LLM pilot
          {p.llmAvailable ? (
            <span style={{ color: "var(--muted)" }}>({p.llmModel}, ~60 steps cap)</span>
          ) : (
            <span style={{ color: "var(--danger)" }}>
              (set ANTHROPIC_API_KEY to enable)
            </span>
          )}
        </label>
      </StepBox>

      {p.laboratory}
    </div>
  );
}
