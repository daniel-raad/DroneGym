/**
 * Frontend port of the backend physics for the *manual flight* mode.
 *
 * The authoritative simulator lives in backend/app/simulator/. We re-implement
 * the small subset needed for interactive arrow-key control here so that each
 * keystroke advances the drone instantly without a network round-trip. Keep
 * the constants here in sync with `backend/app/simulator/physics.py`.
 */
import type { EnvironmentConfig, Observation, Obstacle } from "./api";

export const STEP_SIZE = 0.3;
export const TURN_DELTA_DEG = 15;
export const RAY_MAX = 6.0;
export const RAY_STEP = 0.05;

export function normalizeAngle(deg: number): number {
  let d = ((deg + 180) % 360) + 360;
  return (d % 360) - 180;
}

export function applyAction(
  x: number,
  y: number,
  heading: number,
  action: string,
): [number, number, number] {
  const rad = (heading * Math.PI) / 180;
  switch (action) {
    case "move_forward":
      return [x + Math.cos(rad) * STEP_SIZE, y + Math.sin(rad) * STEP_SIZE, heading];
    case "move_back":
      return [x - Math.cos(rad) * STEP_SIZE, y - Math.sin(rad) * STEP_SIZE, heading];
    case "move_left":
      return [
        x + Math.cos(rad + Math.PI / 2) * STEP_SIZE,
        y + Math.sin(rad + Math.PI / 2) * STEP_SIZE,
        heading,
      ];
    case "move_right":
      return [
        x + Math.cos(rad - Math.PI / 2) * STEP_SIZE,
        y + Math.sin(rad - Math.PI / 2) * STEP_SIZE,
        heading,
      ];
    case "turn_left":
      return [x, y, normalizeAngle(heading - TURN_DELTA_DEG)];
    case "turn_right":
      return [x, y, normalizeAngle(heading + TURN_DELTA_DEG)];
    default:
      return [x, y, heading];
  }
}

export function outOfBounds(x: number, y: number, w: number, h: number, droneRadius: number): boolean {
  return (
    x < droneRadius ||
    x > w - droneRadius ||
    y < droneRadius ||
    y > h - droneRadius
  );
}

export function pointInObstacle(
  x: number,
  y: number,
  obstacles: Obstacle[],
  droneRadius: number,
): boolean {
  for (const ob of obstacles) {
    if (Math.hypot(x - ob.x, y - ob.y) <= ob.radius + droneRadius) return true;
  }
  return false;
}

export function raycast(
  x: number,
  y: number,
  angleDeg: number,
  w: number,
  h: number,
  obstacles: Obstacle[],
): number {
  const rad = (angleDeg * Math.PI) / 180;
  const dx = Math.cos(rad);
  const dy = Math.sin(rad);
  let d = 0;
  while (d < RAY_MAX) {
    d += RAY_STEP;
    const px = x + dx * d;
    const py = y + dy * d;
    if (px <= 0 || px >= w || py <= 0 || py >= h) return d;
    for (const ob of obstacles) {
      if (Math.hypot(px - ob.x, py - ob.y) <= ob.radius) return d;
    }
  }
  return RAY_MAX;
}

export function makeObservation(
  env: EnvironmentConfig,
  x: number,
  y: number,
  heading: number,
  battery: number,
  step: number,
): Observation {
  const dx = env.target[0] - x;
  const dy = env.target[1] - y;
  const dist = Math.hypot(dx, dy);
  const worldAngle = (Math.atan2(dy, dx) * 180) / Math.PI;
  const targetAngle = normalizeAngle(worldAngle - heading);
  return {
    distance_to_target: Math.round(dist * 1000) / 1000,
    target_angle_deg: Math.round(targetAngle * 100) / 100,
    front_distance: raycast(x, y, heading, env.room_width, env.room_height, env.obstacles),
    left_distance: raycast(x, y, heading + 90, env.room_width, env.room_height, env.obstacles),
    right_distance: raycast(x, y, heading - 90, env.room_width, env.room_height, env.obstacles),
    battery: Math.round(battery * 100) / 100,
    step,
  };
}
