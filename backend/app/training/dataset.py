"""Generate an imitation dataset by running the heuristic agent."""
from __future__ import annotations

import json
import random

from ..agents.heuristic_agent import HeuristicAgent
from ..models import GenerateDatasetRequest, GenerateEnvRequest, Observation
from ..simulator.env import Simulator, generate_environment
from ..storage import run_store


def obs_to_features(obs: Observation, num_rays: int) -> list[float]:
    """Feature vector the policy network consumes.

    For num_rays == 3 we use the legacy [dist, angle, front, left, right, battery]
    layout so old checkpoints stay valid. For num_rays != 3 we use the lidar-style
    [dist, angle, *rays, battery] layout.
    """
    if num_rays == 3 and len(obs.rays) >= 3:
        return [
            obs.distance_to_target,
            obs.target_angle_deg,
            obs.rays[0],
            obs.rays[1],
            obs.rays[2],
            obs.battery,
        ]
    if num_rays == 3:
        # Legacy observation (no rays array) — keep prior 6-feature layout.
        return [
            obs.distance_to_target,
            obs.target_angle_deg,
            obs.front_distance,
            obs.left_distance,
            obs.right_distance,
            obs.battery,
        ]
    if len(obs.rays) != num_rays:
        raise ValueError(
            f"observation has {len(obs.rays)} rays but policy expects {num_rays}"
        )
    return [obs.distance_to_target, obs.target_angle_deg, *obs.rays, obs.battery]


def generate_imitation_samples(req: GenerateDatasetRequest) -> tuple[list[dict], dict]:
    rng = random.Random(req.seed)
    samples: list[dict] = []
    n_success = 0
    num_rays = max(req.num_rays, 3)

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
        sim = Simulator(env, seed=rng.randint(0, 10_000_000), num_rays=num_rays)
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
                        "rays": list(obs.rays),
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
        "num_rays": num_rays,
    }
    return samples, stats


def write_dataset_meta(dataset_name: str, meta: dict) -> None:
    """Sidecar JSON next to {name}.jsonl recording how the dataset was built."""
    path = run_store.DATASETS_DIR / f"{dataset_name}.meta.json"
    path.write_text(json.dumps(meta, indent=2))


def read_dataset_meta(dataset_name: str) -> dict:
    path = run_store.DATASETS_DIR / f"{dataset_name}.meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
