"""Environment generator and core stepping loop."""
from __future__ import annotations

import math
import random
from typing import Optional

from ..models import (
    EnvironmentConfig,
    EventEntry,
    GenerateEnvRequest,
    Obstacle,
    Observation,
    TrajectoryPoint,
)
from . import physics
from .observations import build_observation


def generate_environment(req: GenerateEnvRequest) -> EnvironmentConfig:
    seed = req.seed if req.seed is not None else random.randint(0, 1_000_000)
    rng = random.Random(seed)

    # Scale parameters by difficulty (1..10)
    difficulty = max(1, min(req.difficulty, 10))
    num_obstacles = req.num_obstacles + (difficulty - 1)
    # smaller obstacles at low difficulty, bigger at high
    min_r = 0.4 + 0.05 * difficulty
    max_r = 0.7 + 0.10 * difficulty

    w, h = req.room_width, req.room_height
    drone_radius = 0.25

    # Place start and target at opposite corners with jitter
    start = (1.0 + rng.uniform(0, 0.5), 1.0 + rng.uniform(0, 0.5))
    target = (w - 1.0 - rng.uniform(0, 0.5), h - 1.0 - rng.uniform(0, 0.5))

    obstacles: list[Obstacle] = []
    attempts = 0
    while len(obstacles) < num_obstacles and attempts < num_obstacles * 50:
        attempts += 1
        r = rng.uniform(min_r, max_r)
        ox = rng.uniform(r + 0.3, w - r - 0.3)
        oy = rng.uniform(r + 0.3, h - r - 0.3)
        # keep clear of start/target
        if math.hypot(ox - start[0], oy - start[1]) < r + 1.0:
            continue
        if math.hypot(ox - target[0], oy - target[1]) < r + 1.0:
            continue
        # avoid overlap with other obstacles
        ok = True
        for ob in obstacles:
            if math.hypot(ox - ob.x, oy - ob.y) < r + ob.radius + 0.3:
                ok = False
                break
        if not ok:
            continue
        obstacles.append(Obstacle(x=round(ox, 3), y=round(oy, 3), radius=round(r, 3)))

    max_steps = 200 + (difficulty - 1) * 30

    return EnvironmentConfig(
        difficulty=difficulty,
        room_width=w,
        room_height=h,
        num_obstacles=len(obstacles),
        wind_strength=req.wind_strength,
        sensor_noise=req.sensor_noise,
        obstacles=obstacles,
        start=start,
        target=target,
        target_radius=0.6,
        drone_radius=drone_radius,
        max_steps=max_steps,
        seed=seed,
    )


class Simulator:
    """Stateless-ish simulator: drives one episode."""

    def __init__(self, env: EnvironmentConfig, seed: Optional[int] = None, num_rays: int = 3):
        self.env = env
        self.rng = random.Random(seed if seed is not None else env.seed or 0)
        self.x, self.y = env.start
        # Initial heading: aim at target
        tx, ty = env.target
        self.heading = math.degrees(math.atan2(ty - self.y, tx - self.x))
        self.battery = 100.0
        self.step_idx = 0
        self.done = False
        self.collided = False
        self.landed = False
        self.success = False
        self.timeout = False
        self.num_rays = num_rays

    def _build_obs(self) -> Observation:
        return build_observation(
            self.env,
            self.x,
            self.y,
            self.heading,
            self.battery,
            self.step_idx,
            self.rng,
            num_rays=self.num_rays,
        )

    def initial_observation(self) -> Observation:
        return self._build_obs()

    def step(self, action: str) -> tuple[Observation, float, bool, list[EventEntry]]:
        events: list[EventEntry] = []
        prev_x, prev_y = self.x, self.y
        prev_dist = math.hypot(self.env.target[0] - prev_x, self.env.target[1] - prev_y)

        new_x, new_y, new_heading = physics.apply_action(self.x, self.y, self.heading, action)

        # Apply wind drift
        if self.env.wind_strength > 0:
            # constant wind toward +x
            new_x += self.env.wind_strength * 0.1

        reward = -1.0  # step penalty

        # Check land action
        if action == "land":
            tx, ty = self.env.target
            dist_to_target = math.hypot(self.x - tx, self.y - ty)
            self.landed = True
            self.done = True
            if dist_to_target <= self.env.target_radius * 2.0:
                self.success = True
                reward += 100.0
                events.append(EventEntry(step=self.step_idx, type="land_success", detail=f"landed at dist {dist_to_target:.2f}"))
            else:
                reward -= 50.0
                events.append(EventEntry(step=self.step_idx, type="land_fail", detail=f"landed too far: {dist_to_target:.2f}"))
            self.step_idx += 1
            obs = self._build_obs()
            return obs, reward, self.done, events

        # Collision detection
        if physics.out_of_bounds(new_x, new_y, self.env.room_width, self.env.room_height, self.env.drone_radius):
            self.collided = True
            self.done = True
            reward -= 100.0
            events.append(EventEntry(step=self.step_idx, type="collision_wall", detail=f"at ({new_x:.2f},{new_y:.2f})"))
            # clamp so visualization still shows last legal point near hit
            self.x = max(self.env.drone_radius, min(self.env.room_width - self.env.drone_radius, new_x))
            self.y = max(self.env.drone_radius, min(self.env.room_height - self.env.drone_radius, new_y))
            self.heading = new_heading
            self.step_idx += 1
            obs = self._build_obs()
            return obs, reward, self.done, events

        if physics.point_in_obstacle(new_x, new_y, self.env.obstacles, self.env.drone_radius):
            self.collided = True
            self.done = True
            reward -= 100.0
            events.append(EventEntry(step=self.step_idx, type="collision_obstacle", detail=f"at ({new_x:.2f},{new_y:.2f})"))
            self.x, self.y = new_x, new_y
            self.heading = new_heading
            self.step_idx += 1
            obs = self._build_obs()
            return obs, reward, self.done, events

        # Commit move
        self.x, self.y = new_x, new_y
        self.heading = new_heading

        # Check goal reached (passive success)
        tx, ty = self.env.target
        dist_to_target = math.hypot(self.x - tx, self.y - ty)
        if dist_to_target <= self.env.target_radius:
            self.success = True
            self.done = True
            reward += 100.0
            events.append(EventEntry(step=self.step_idx, type="reached_target", detail=f"dist {dist_to_target:.2f}"))

        # Shaping: reward distance reduction
        reward += (prev_dist - dist_to_target) * 5.0

        # Penalty for being close to obstacle in front
        front = physics.raycast(self.x, self.y, self.heading, self.env.room_width, self.env.room_height, self.env.obstacles, max_dist=2.0)
        if front < 0.5 and action == "move_forward":
            reward -= 2.0

        # Battery / time
        self.battery -= 0.5
        self.step_idx += 1
        if self.battery <= 0 or self.step_idx >= self.env.max_steps:
            if not self.done:
                self.timeout = True
                self.done = True
                events.append(EventEntry(step=self.step_idx, type="timeout", detail=f"steps={self.step_idx}, batt={self.battery:.1f}"))

        obs = self._build_obs()
        return obs, reward, self.done, events

    def current_trajectory_point(self) -> TrajectoryPoint:
        return TrajectoryPoint(
            step=self.step_idx,
            x=round(self.x, 4),
            y=round(self.y, 4),
            heading=round(self.heading, 2),
            battery=round(self.battery, 2),
        )
