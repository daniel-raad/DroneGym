import { useEffect, useRef, useState } from "react";
import { api, streamEpisode } from "../api";
import type {
  AgentType,
  EnvironmentConfig,
  EpisodeResponse,
  PolicyInspectResponse,
  SystemStatus,
} from "../api";
import { ReplayCanvas } from "../components/ReplayCanvas";
import type { Pilot } from "./pilots";
import type { Drone } from "./drones";
import { MindStream } from "./MindStream";

type Props = {
  pilot: Pilot;
  drone: Drone;
  status: SystemStatus | null;
  onBack: () => void;
  onChangePilot: () => void;
};

type Phase = "loading-track" | "countdown" | "flying" | "done" | "error";

type Course = "breeze" | "cruise" | "tight";

// The arcade default has to feel winnable. "breeze" is the same as the
// dashboard's easy preset (where the teacher hits ~80% success), shifted one
// step lower on obstacle count so a first race almost always lands cleanly.
const COURSES: Record<Course, { difficulty: number; obstacles: number; label: string }> = {
  breeze: { difficulty: 1, obstacles: 2, label: "Breeze" },
  cruise: { difficulty: 1, obstacles: 3, label: "Cruise" },
  tight: { difficulty: 3, obstacles: 5, label: "Tight" },
};

// Pick the best trained model name available. Practiced wins if it exists,
// else the base BC model.
function preferredModelName(status: SystemStatus | null): string {
  const models = status?.models ?? [];
  const practiced = models.find((m) => m.name.endsWith("_practiced"));
  if (practiced) return practiced.name;
  const bc = models.find((m) => m.method === "behavior_cloning");
  if (bc) return bc.name;
  return models[0]?.name ?? "drone_policy_v1";
}

export function RaceScreen({ pilot, drone, status, onBack, onChangePilot }: Props) {
  const [phase, setPhase] = useState<Phase>("loading-track");
  const [count, setCount] = useState(3);
  const [env, setEnv] = useState<EnvironmentConfig | null>(null);
  const [episode, setEpisode] = useState<EpisodeResponse | null>(null);
  const [inspect, setInspect] = useState<PolicyInspectResponse | null>(null);
  const [step, setStep] = useState(0);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [trackSeed, setTrackSeed] = useState<number>(() => Math.floor(Math.random() * 10000));
  const [course, setCourse] = useState<Course>("breeze");
  const runId = useRef(0);
  // The streaming fetch lives across effect re-runs — we can't attach the
  // AbortController to a useEffect cleanup because setPhase("flying") below
  // re-triggers the same effect, which would then immediately abort the
  // stream we just started. Instead we hold the controller in a ref and only
  // abort it when a *new* race starts or the component unmounts.
  const streamAbort = useRef<AbortController | null>(null);
  useEffect(() => () => streamAbort.current?.abort(), []);

  // Drive the race. Generates a track, runs the pilot, then plays back.
  useEffect(() => {
    const myRun = ++runId.current;
    // A fresh race invalidates any in-flight stream from the previous race.
    streamAbort.current?.abort();
    setPhase("loading-track");
    setEpisode(null);
    setInspect(null);
    setCount(3);

    (async () => {
      try {
        const c = COURSES[course];
        const e = await api.generateEnv({
          difficulty: c.difficulty,
          room_width: 10,
          room_height: 10,
          num_obstacles: c.obstacles,
          wind_strength: 0,
          sensor_noise: 0,
          seed: trackSeed,
        });
        if (runId.current !== myRun) return;
        setEnv(e);
        setPhase("countdown");
      } catch (err: any) {
        if (runId.current !== myRun) return;
        setErrMsg(err.message || "track generator offline");
        setPhase("error");
      }
    })();

    return () => {
      runId.current++; // invalidate
    };
  }, [trackSeed, pilot.id, drone.id, course]);

  // Countdown 3 → 2 → 1 → GO, then dispatch the race.
  useEffect(() => {
    if (phase !== "countdown" || !env) return;
    if (count > 0) {
      const t = setTimeout(() => setCount((c) => c - 1), 650);
      return () => clearTimeout(t);
    }
    // count == 0 → fire the race
    const myRun = runId.current;
    // Cancel any previous in-flight stream (e.g. user spammed Next Race).
    streamAbort.current?.abort();
    const ctrl = new AbortController();
    streamAbort.current = ctrl;
    setPhase("flying");
    (async () => {
      try {
        // Custom pilots ship their own modelName; the built-in Neon falls
        // back to whatever the Gym last trained under the canonical name.
        const trainedModelName =
          pilot.backend.kind === "trained"
            ? pilot.backend.modelName ?? preferredModelName(status)
            : null;

        let agentType: AgentType;
        if (trainedModelName) agentType = "trained";
        else if (pilot.backend.kind === "episode") agentType = pilot.backend.agentType;
        else throw new Error("unsupported pilot backend");

        // LLM is slow per step — cap aggressively. Others get the env's full budget.
        const maxSteps = agentType === "llm" ? Math.min(60, env.max_steps) : env.max_steps;

        let finalEp: EpisodeResponse | null = null;
        for await (const frame of streamEpisode(
          env,
          agentType,
          maxSteps,
          trackSeed,
          trainedModelName,
          ctrl.signal,
        )) {
          if (runId.current !== myRun) return;
          if (frame.type === "start") {
            setEpisode({
              episode_id: frame.episode_id,
              success: false,
              collision: false,
              landed: false,
              timeout: false,
              steps: 0,
              score: 0,
              environment: frame.env,
              trajectory: [frame.initial.trajectory],
              actions: [],
              observations: [frame.initial.observation],
              events: [],
              summary: {
                episode_id: frame.episode_id,
                success: false,
                collision: false,
                landed: false,
                timeout: false,
                steps: 0,
                score: 0,
                final_distance_to_target: 0,
                path_length: 0,
                avg_obstacle_clearance: 0,
                action_counts: {},
                agent_type: agentType,
                difficulty: frame.env.difficulty,
              },
            });
          } else if (frame.type === "step") {
            setEpisode((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                trajectory: [...prev.trajectory, frame.trajectory],
                actions: [...prev.actions, frame.action],
                observations: [...prev.observations, frame.observation],
                events: frame.events.length ? [...prev.events, ...frame.events] : prev.events,
                steps: frame.step,
              };
            });
          } else if (frame.type === "end") {
            finalEp = frame.episode;
            setEpisode(frame.episode);
          }
        }
        if (runId.current !== myRun) return;
        setPhase("done");

        // For any trained pilot, peek inside the network so the mind-stream
        // can show per-step action probabilities.
        if (trainedModelName && finalEp && finalEp.observations.length) {
          api
            .inspectPolicy(trainedModelName, finalEp.observations)
            .then((r) => runId.current === myRun && setInspect(r))
            .catch(() => {});
        }
      } catch (err: any) {
        // AbortError is expected when the user navigates away mid-stream.
        if (err?.name === "AbortError") return;
        if (runId.current !== myRun) return;
        setErrMsg(err.message || "the pilot crashed before takeoff");
        setPhase("error");
      }
    })();
    // No cleanup here — see comment on streamAbort above.
  }, [phase, count, env, pilot, status, trackSeed]);

  // `Next race` rerolls the env seed. With the seed fixed, deterministic
  // pilots (Sgt, Neon) produce the exact same trajectory — that's a real
  // property of the policy, not a UI bug — so giving them a fresh world is
  // the only way races stop feeling like replays.
  const nextRace = () => setTrackSeed(Math.floor(Math.random() * 10000));
  // `Replay this track` keeps the seed — useful for "show me exactly how
  // that just happened" without resetting state.
  const replay = () => {
    setEpisode(null);
    setInspect(null);
    setCount(3);
    setPhase("countdown");
  };

  const outcome = episode
    ? episode.success
      ? "REACHED TARGET"
      : episode.collision
        ? "CRASHED"
        : "TIMED OUT"
    : "";
  const outcomeClass = episode
    ? episode.success
      ? "ok"
      : episode.collision
        ? "bad"
        : "warn"
    : "";

  return (
    <div className="arc-race" style={{ ['--hue' as any]: pilot.hue, ['--accent' as any]: drone.color }}>
      <header className="arc-race-head">
        <button className="arc-back" onClick={onBack}>
          ← HANGAR
        </button>
        <div className="arc-race-vs">
          <span className="arc-race-pill drone">
            <span className="emoji">{drone.emoji}</span>
            {drone.name}
          </span>
          <span className="arc-race-x">×</span>
          <span className="arc-race-pill pilot">
            <span className="emoji">{pilot.face}</span>
            {pilot.name}
          </span>
        </div>
        <div className="arc-race-course">
          {(Object.keys(COURSES) as Course[]).map((c) => (
            <button
              key={c}
              className={`arc-course-pill ${course === c ? "active" : ""}`}
              onClick={() => setCourse(c)}
              disabled={phase === "flying"}
            >
              {COURSES[c].label}
            </button>
          ))}
        </div>
        <div className="arc-race-track-code">
          #{trackSeed.toString().padStart(4, "0")}
        </div>
      </header>

      <div className="arc-race-stage">
        <div className="arc-canvas-wrap">
          <ReplayCanvas
            episode={episode}
            env={env}
            overlay={null}
            overlayLLM={null}
            onStepChange={setStep}
            showSensors={pilot.id === "heuristic"}
            controlsMode={phase === "flying" ? "live" : "full"}
          />

          {phase === "loading-track" && (
            <Overlay>
              <div className="arc-overlay-eyebrow">GENERATING TRACK</div>
              <div className="arc-overlay-big">…</div>
            </Overlay>
          )}

          {phase === "countdown" && (
            <Overlay>
              <div className="arc-overlay-eyebrow">{drone.name} × {pilot.name}</div>
              <div key={count} className="arc-countdown">
                {count === 0 ? "GO" : count}
              </div>
              <div className="arc-overlay-quote">“{pilot.voiceLine}”</div>
            </Overlay>
          )}

          {phase === "flying" && !episode && (
            <Overlay subtle>
              <div className="arc-overlay-eyebrow pulse">PREPARING PILOT…</div>
              <div className="arc-overlay-mini">
                {pilot.id === "llm" ? "Oracle is warming up the LLM." : "warming up"}
              </div>
            </Overlay>
          )}
          {phase === "flying" && episode && pilot.id === "llm" && (
            <div className="arc-live-chip">
              <span className="arc-live-dot" />
              LIVE · step {episode.steps}
            </div>
          )}

          {phase === "done" && episode && (
            <div className={`arc-result-banner ${outcomeClass}`}>
              <div className="arc-result-outcome">{outcome}</div>
              <div className="arc-result-stats">
                <span>
                  STEPS&nbsp;<b>{episode.steps}</b>
                </span>
                <span>
                  SCORE&nbsp;<b>{episode.score.toFixed(0)}</b>
                </span>
                <span>
                  DIST&nbsp;<b>{episode.summary.final_distance_to_target.toFixed(1)}</b>
                </span>
              </div>
            </div>
          )}

          {phase === "error" && (
            <Overlay>
              <div className="arc-overlay-eyebrow bad">SOMETHING WENT WRONG</div>
              <div className="arc-overlay-mini">{errMsg}</div>
              <button className="arc-cta small" onClick={nextRace}>
                TRY ANOTHER TRACK
              </button>
            </Overlay>
          )}
        </div>

        <MindStream pilot={pilot} episode={episode} step={step} inspect={inspect} />
      </div>

      <footer className="arc-race-foot">
        <button className="arc-ghost" onClick={onChangePilot}>
          change pilot
        </button>
        <button className="arc-ghost" onClick={replay} disabled={phase === "flying"} title="Same world, same pilot — useful for inspecting how that just happened">
          replay this track
        </button>
        <button className="arc-cta small" onClick={nextRace} disabled={phase === "flying"}>
          Next race&nbsp;→
        </button>
      </footer>
    </div>
  );
}

function Overlay({
  children,
  subtle,
}: {
  children: React.ReactNode;
  subtle?: boolean;
}) {
  return <div className={`arc-overlay ${subtle ? "subtle" : ""}`}>{children}</div>;
}
