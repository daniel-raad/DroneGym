"""Generate an imitation dataset by running the heuristic agent."""
from __future__ import annotations

import random
from typing import Iterator

from ..agents.heuristic_agent import HeuristicAgent
from ..models import GenerateDatasetRequest, GenerateEnvRequest, Observation
from ..simulator.env import Simulator, generate_environment
from ..simulator.observations import build_observation


def _obs_to_features(obs: Observation) -> list[float]:
    return [
        obs.distance_to_target,
        obs.target_angle_deg,
        obs.front_distance,
        obs.left_distance,
        obs.right_distance,
        obs.battery,
    ]


def generate_imitation_samples(req: GenerateDatasetRequest) -> tuple[list[dict], dict]:
    rng = random.Random(req.seed)
    samples: list[dict] = []
    n_success = 0

    for ep_i in range(req.num_episodes):
        diff_choices = req.difficulties or [req.difficulty]
        chosen_difficulty = rng.choice(diff_choices)
        env = generate_environment(
            GenerateEnvRequest(
                difficulty=chosen_difficulty,
                room_width=req.room_width,
                room_height=req.room_height,
                num_obstacles=req.num_obstacles,
                seed=rng.randint(0, 10_000_000),
            )
        )
        sim = Simulator(env, seed=rng.randint(0, 10_000_000))
        agent = HeuristicAgent(seed=rng.randint(0, 10_000_000))

        obs = sim.initial_observation()
        steps = 0
        ep_samples: list[dict] = []
        while not sim.done and steps < req.max_steps:
            action = agent.act(obs)
            ep_samples.append(
                {
                    "observation": {
                        "distance_to_target": obs.distance_to_target,
                        "target_angle_deg": obs.target_angle_deg,
                        "front_distance": obs.front_distance,
                        "left_distance": obs.left_distance,
                        "right_distance": obs.right_distance,
                        "battery": obs.battery,
                    },
                    "action": action,
                }
            )
            obs, _r, _done, _evs = sim.step(action)
            steps += 1

        # Only keep successful trajectories as imitation examples
        if sim.success:
            n_success += 1
            samples.extend(ep_samples)

    stats = {
        "num_episodes": req.num_episodes,
        "num_success": n_success,
        "success_rate": round(n_success / max(req.num_episodes, 1), 3),
        "num_samples": len(samples),
    }
    return samples, stats
