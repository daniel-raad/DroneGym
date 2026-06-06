"""Episode runner that ties simulator and agent together."""
from __future__ import annotations

import math
import uuid
from typing import Optional

from ..agents.base import DroneAgent
from ..models import (
    EnvironmentConfig,
    EpisodeResponse,
    EpisodeSummary,
    EventEntry,
    Observation,
    TrajectoryPoint,
    ACTIONS,
)
from . import physics
from .env import Simulator


def run_episode(
    env: EnvironmentConfig,
    agent: DroneAgent,
    max_steps: int = 200,
    seed: Optional[int] = None,
    agent_label: str = "heuristic",
    episode_id: Optional[str] = None,
) -> EpisodeResponse:
    num_rays = int(getattr(agent, "num_rays", 3))
    sim = Simulator(env, seed=seed, num_rays=num_rays)
    if max_steps and max_steps != env.max_steps:
        sim.env = env.model_copy(update={"max_steps": max_steps})

    trajectory: list[TrajectoryPoint] = [sim.current_trajectory_point()]
    observations: list[Observation] = []
    actions: list[str] = []
    events: list[EventEntry] = []

    obs = sim.initial_observation()
    observations.append(obs)
    total_reward = 0.0
    clearance_sum = 0.0
    clearance_count = 0

    while not sim.done:
        action = agent.act(obs)
        if action not in ACTIONS:
            action = "hover"
        actions.append(action)

        obs, reward, done, evs = sim.step(action)
        observations.append(obs)
        events.extend(evs)
        total_reward += reward

        trajectory.append(sim.current_trajectory_point())

        clearance_sum += physics.min_obstacle_clearance(
            sim.x, sim.y, env.obstacles, env.room_width, env.room_height
        )
        clearance_count += 1

    eid = episode_id or f"run_{uuid.uuid4().hex[:8]}"
    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += math.hypot(
            trajectory[i].x - trajectory[i - 1].x,
            trajectory[i].y - trajectory[i - 1].y,
        )
    final_dist = math.hypot(
        env.target[0] - trajectory[-1].x, env.target[1] - trajectory[-1].y
    )
    action_counts: dict[str, int] = {a: 0 for a in ACTIONS}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    avg_clearance = clearance_sum / max(clearance_count, 1)

    summary = EpisodeSummary(
        episode_id=eid,
        success=sim.success,
        collision=sim.collided,
        landed=sim.landed,
        timeout=sim.timeout,
        steps=sim.step_idx,
        score=round(total_reward, 2),
        final_distance_to_target=round(final_dist, 3),
        path_length=round(path_length, 3),
        avg_obstacle_clearance=round(avg_clearance, 3),
        action_counts=action_counts,
        agent_type=agent_label,
        difficulty=env.difficulty,
    )

    return EpisodeResponse(
        episode_id=eid,
        success=sim.success,
        collision=sim.collided,
        landed=sim.landed,
        timeout=sim.timeout,
        steps=sim.step_idx,
        score=round(total_reward, 2),
        environment=env,
        trajectory=trajectory,
        actions=actions,
        observations=observations,
        events=events,
        summary=summary,
    )
