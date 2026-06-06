"""Build observations from drone state and environment."""
from __future__ import annotations

import math
import random

from ..models import EnvironmentConfig, Observation
from . import physics


def build_observation(
    env: EnvironmentConfig,
    x: float,
    y: float,
    heading_deg: float,
    battery: float,
    step: int,
    rng: random.Random | None = None,
    num_rays: int = 3,
) -> Observation:
    tx, ty = env.target
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    target_world_angle = math.degrees(math.atan2(dy, dx))
    target_angle_deg = physics.normalize_angle_deg(target_world_angle - heading_deg)

    # Named convenience rays (always populated — the heuristic and LLM agents
    # use these by name regardless of how many extra rays the active model sees).
    front = physics.raycast(x, y, heading_deg, env.room_width, env.room_height, env.obstacles)
    left = physics.raycast(x, y, heading_deg + 90.0, env.room_width, env.room_height, env.obstacles)
    right = physics.raycast(x, y, heading_deg - 90.0, env.room_width, env.room_height, env.obstacles)

    # Lidar-style sweep: N rays evenly spaced 360°, ray[0] aligned with heading.
    # For num_rays == 3 we keep [front, left, right] so legacy checkpoints get
    # the exact feature vector they were trained on.
    rays: list[float]
    if num_rays <= 3:
        rays = [front, left, right][:max(num_rays, 0)]
    else:
        step_deg = 360.0 / num_rays
        rays = [
            physics.raycast(
                x, y, heading_deg + i * step_deg, env.room_width, env.room_height, env.obstacles
            )
            for i in range(num_rays)
        ]

    if env.sensor_noise > 0 and rng is not None:
        noise = env.sensor_noise
        dist = max(0.0, dist + rng.gauss(0, noise))
        target_angle_deg = physics.normalize_angle_deg(target_angle_deg + rng.gauss(0, noise * 10))
        front = max(0.0, front + rng.gauss(0, noise))
        left = max(0.0, left + rng.gauss(0, noise))
        right = max(0.0, right + rng.gauss(0, noise))
        rays = [max(0.0, r + rng.gauss(0, noise)) for r in rays]

    return Observation(
        distance_to_target=round(dist, 3),
        target_angle_deg=round(target_angle_deg, 2),
        front_distance=round(front, 3),
        left_distance=round(left, 3),
        right_distance=round(right, 3),
        battery=round(battery, 2),
        step=step,
        rays=[round(r, 3) for r in rays],
    )
