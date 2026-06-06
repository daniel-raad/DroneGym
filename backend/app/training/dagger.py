"""DAgger: roll out the current policy, ask the teacher what *it* would have done
in each visited state, append (state, teacher_action) to the dataset, refit BC.

This is the cleanest fix for behavior-cloning's covariate-shift problem: the
student gets to see — and learn correct behavior on — the states it actually
visits, not just the states the teacher would have visited.
"""
from __future__ import annotations

import json
import random
import time

import torch

from ..agents.heuristic_agent import HeuristicAgent
from ..models import (
    ACTIONS,
    DAggerRequest,
    DAggerResponse,
    GenerateEnvRequest,
    TrainPolicyRequest,
)
from ..runtime_state import STATE
from ..simulator.env import Simulator, generate_environment
from ..storage import run_store
from .dataset import read_dataset_meta, write_dataset_meta
from .evaluate_policy import TrainedPolicyAgent
from .policy_model import input_size_for
from .train_policy import load_policy, train_policy


def _obs_to_dict(obs) -> dict:
    return {
        "distance_to_target": obs.distance_to_target,
        "target_angle_deg": obs.target_angle_deg,
        "front_distance": obs.front_distance,
        "left_distance": obs.left_distance,
        "right_distance": obs.right_distance,
        "battery": obs.battery,
        "rays": list(obs.rays),
    }


def run_dagger(req: DAggerRequest) -> DAggerResponse:
    rng = random.Random(req.seed)

    # Resolve num_rays from existing dataset / model meta if unset.
    if req.num_rays is not None:
        num_rays = req.num_rays
    else:
        meta = read_dataset_meta(req.dataset_name)
        num_rays = int(meta.get("num_rays") or 3)

    STATE.start("dagger", detail=f"{req.rounds} rounds × {req.episodes_per_round} eps")
    teacher = HeuristicAgent()
    per_round_acc: list[float] = []
    final_test_acc = 0.0
    total_added = 0

    try:
        for round_i in range(req.rounds):
            STATE.update(
                detail=f"round {round_i + 1}/{req.rounds}: rolling out student",
                progress=round_i / max(req.rounds, 1),
            )
            # Load the latest student
            model = load_policy(req.model_name)
            student = TrainedPolicyAgent(model, num_rays=num_rays)

            samples: list[dict] = []
            for _ in range(req.episodes_per_round):
                env = generate_environment(
                    GenerateEnvRequest(
                        difficulty=req.difficulty,
                        room_width=req.room_width,
                        room_height=req.room_height,
                        num_obstacles=req.num_obstacles,
                        seed=rng.randint(0, 10_000_000),
                    )
                )
                sim = Simulator(env, seed=rng.randint(0, 10_000_000), num_rays=num_rays)
                obs = sim.initial_observation()
                steps = 0
                while not sim.done and steps < req.max_steps:
                    teacher_action = teacher.act(obs)
                    samples.append(
                        {"observation": _obs_to_dict(obs), "action": teacher_action}
                    )
                    student_action = student.act(obs)
                    if student_action not in ACTIONS:
                        student_action = "hover"
                    obs, _r, _done, _ev = sim.step(student_action)
                    steps += 1

            STATE.update(
                detail=f"round {round_i + 1}/{req.rounds}: appended {len(samples)} samples · refitting"
            )
            path, total = run_store.write_dataset(req.dataset_name, samples, append=True)
            total_added += len(samples)
            write_dataset_meta(
                req.dataset_name,
                {"num_rays": num_rays, "num_samples": total, "difficulty": req.difficulty},
            )

            # Refit BC on the now-expanded dataset
            bc_res = train_policy(
                TrainPolicyRequest(
                    dataset_name=req.dataset_name,
                    epochs=req.epochs,
                    batch_size=req.batch_size,
                    learning_rate=req.learning_rate,
                    hidden_size=req.hidden_size,
                    num_layers=req.num_layers,
                    model_name=req.model_name,
                    num_rays=num_rays,
                )
            )
            per_round_acc.append(round(bc_res.test_accuracy, 4))
            final_test_acc = bc_res.test_accuracy

    finally:
        STATE.finish()

    # Annotate model meta so the gym can show "dagger refined" badge.
    meta_path = run_store.models_dir() / f"{req.model_name}.meta.json"
    if meta_path.exists():
        try:
            mm = json.loads(meta_path.read_text())
            mm["dagger_rounds"] = (mm.get("dagger_rounds") or 0) + req.rounds
            mm["dagger_per_round_accuracy"] = per_round_acc
            mm["method"] = "dagger"
            meta_path.write_text(json.dumps(mm, indent=2))
        except Exception:
            pass

    return DAggerResponse(
        model_path=str(run_store.models_dir() / f"{req.model_name}.pt"),
        rounds=req.rounds,
        dataset_size=total_added,
        final_test_accuracy=round(final_test_acc, 4),
        per_round_accuracy=per_round_acc,
    )
