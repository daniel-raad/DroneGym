import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  EnvironmentConfig,
  EpisodeResponse,
  EventEntry,
  Observation,
  TrajectoryPoint,
} from "./api";
import {
  applyAction,
  makeObservation,
  outOfBounds,
  pointInObstacle,
} from "./simulator";

type Verdict = "success" | "collision" | "landed_off" | "timeout" | null;

const KEY_ACTION: Record<string, string> = {
  ArrowUp: "move_forward",
  ArrowDown: "move_back",
  ArrowLeft: "move_left",
  ArrowRight: "move_right",
  q: "turn_left",
  Q: "turn_left",
  e: "turn_right",
  E: "turn_right",
  h: "hover",
  H: "hover",
  " ": "land",
};

export function useManualFlight(env: EnvironmentConfig | null, active: boolean) {
  const [drone, setDrone] = useState<{
    x: number;
    y: number;
    heading: number;
    battery: number;
    step: number;
  } | null>(null);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [actions, setActions] = useState<string[]>([]);
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [verdict, setVerdict] = useState<Verdict>(null);
  const [lastKeyAction, setLastKeyAction] = useState<string | null>(null);

  const done = verdict !== null;

  const reset = useCallback(() => {
    if (!env) return;
    const x = env.start[0];
    const y = env.start[1];
    const heading =
      (Math.atan2(env.target[1] - y, env.target[0] - x) * 180) / Math.PI;
    const battery = 100;
    setDrone({ x, y, heading, battery, step: 0 });
    setTrajectory([{ step: 0, x, y, heading, battery }]);
    setObservations([makeObservation(env, x, y, heading, battery, 0)]);
    setActions([]);
    setEvents([]);
    setVerdict(null);
    setLastKeyAction(null);
  }, [env]);

  // Reset whenever env or active flag flips on
  useEffect(() => {
    if (active && env) reset();
  }, [active, env, reset]);

  // Step function uses a ref to drone so handlers don't see stale state
  const droneRef = useRef(drone);
  droneRef.current = drone;

  const step = useCallback(
    (action: string) => {
      const cur = droneRef.current;
      if (!env || !cur || verdict !== null) return;

      let [nx, ny, nheading] = applyAction(cur.x, cur.y, cur.heading, action);
      const newEvents: EventEntry[] = [];
      let newVerdict: Verdict = null;
      let landedNow = false;

      if (action === "land") {
        const dist = Math.hypot(env.target[0] - cur.x, env.target[1] - cur.y);
        landedNow = true;
        nx = cur.x;
        ny = cur.y;
        nheading = cur.heading;
        if (dist <= env.target_radius * 2) {
          newVerdict = "success";
          newEvents.push({
            step: cur.step,
            type: "land_success",
            detail: `landed at dist ${dist.toFixed(2)}`,
          });
        } else {
          newVerdict = "landed_off";
          newEvents.push({
            step: cur.step,
            type: "land_fail",
            detail: `landed too far: ${dist.toFixed(2)}`,
          });
        }
      } else if (outOfBounds(nx, ny, env.room_width, env.room_height, env.drone_radius)) {
        newVerdict = "collision";
        newEvents.push({
          step: cur.step,
          type: "collision_wall",
          detail: `at (${nx.toFixed(2)},${ny.toFixed(2)})`,
        });
      } else if (pointInObstacle(nx, ny, env.obstacles, env.drone_radius)) {
        newVerdict = "collision";
        newEvents.push({
          step: cur.step,
          type: "collision_obstacle",
          detail: `at (${nx.toFixed(2)},${ny.toFixed(2)})`,
        });
      } else {
        const dist = Math.hypot(env.target[0] - nx, env.target[1] - ny);
        if (dist <= env.target_radius) {
          newVerdict = "success";
          newEvents.push({
            step: cur.step + 1,
            type: "reached_target",
            detail: `dist ${dist.toFixed(2)}`,
          });
        }
      }

      const newStep = cur.step + 1;
      const newBattery = cur.battery - 0.5;
      if (newVerdict === null && (newBattery <= 0 || newStep >= env.max_steps)) {
        newVerdict = "timeout";
        newEvents.push({
          step: newStep,
          type: "timeout",
          detail: `steps=${newStep}, batt=${newBattery.toFixed(1)}`,
        });
      }

      const newObs = makeObservation(env, nx, ny, nheading, newBattery, newStep);

      setDrone({ x: nx, y: ny, heading: nheading, battery: newBattery, step: newStep });
      setTrajectory((t) => [...t, { step: newStep, x: nx, y: ny, heading: nheading, battery: newBattery }]);
      setObservations((o) => [...o, newObs]);
      setActions((a) => [...a, action]);
      setEvents((ev) => [...ev, ...newEvents]);
      setLastKeyAction(action);
      if (newVerdict) setVerdict(newVerdict);
      if (landedNow && newVerdict === null) {
        // shouldn't happen — land always sets verdict — but guard
        setVerdict("landed_off");
      }
    },
    [env, verdict],
  );

  // Keyboard listener — only attached while active and not done
  useEffect(() => {
    if (!active || done || !env) return;
    const handler = (e: KeyboardEvent) => {
      // ignore when typing in a form field
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const action = KEY_ACTION[e.key];
      if (!action) return;
      e.preventDefault();
      step(action);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [active, done, env, step]);

  // Synthesize an Episode-shaped object so ReplayCanvas can render it directly.
  const episode = useMemo<EpisodeResponse | null>(() => {
    if (!env || !drone) return null;
    const success = verdict === "success";
    const collision = verdict === "collision";
    const timeout = verdict === "timeout";
    const landed = actions.includes("land") || verdict === "landed_off" || verdict === "success";
    return {
      episode_id: "manual",
      success,
      collision,
      landed,
      timeout,
      steps: drone.step,
      score: 0,
      environment: env,
      trajectory,
      actions,
      observations,
      events,
      summary: {
        episode_id: "manual",
        success,
        collision,
        landed,
        timeout,
        steps: drone.step,
        score: 0,
        final_distance_to_target: Math.hypot(drone.x - env.target[0], drone.y - env.target[1]),
        path_length: 0,
        avg_obstacle_clearance: 0,
        action_counts: actions.reduce<Record<string, number>>((acc, a) => {
          acc[a] = (acc[a] ?? 0) + 1;
          return acc;
        }, {}),
        agent_type: "manual",
        difficulty: env.difficulty,
      },
    };
  }, [env, drone, trajectory, observations, actions, events, verdict]);

  return {
    episode,
    drone,
    verdict,
    done,
    actions,
    lastAction: lastKeyAction,
    reset,
    step,
  };
}
