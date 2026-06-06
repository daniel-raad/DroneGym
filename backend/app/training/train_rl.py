"""Actor-critic (A2C) policy-gradient training in the simulator.

Compared to vanilla REINFORCE this trainer adds:
- a learned value baseline (critic) to reduce gradient variance,
- an entropy bonus to keep the policy from collapsing to one action,
- batched updates over multiple episodes,
- advantage normalization across the whole batch (not per-episode),
- gradient clipping,
- a moving success-rate trace for visual monitoring.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass

import torch
from torch import nn

from ..models import ACTIONS, Observation
from ..runtime_state import STATE
from ..simulator.env import Simulator, generate_environment
from ..models import GenerateEnvRequest
from ..storage import run_store
from .policy_model import ActorCritic, DronePolicy, INPUT_SIZE, NUM_ACTIONS


MAX_EPISODES = 20_000  # hard cap; UI clamps to 2000 but server cap is generous


@dataclass
class TrainRLRequest:
    episodes: int = 400
    learning_rate: float = 3e-4
    gamma: float = 0.97
    hidden_size: int = 64
    difficulty: int = 1
    room_width: float = 10.0
    room_height: float = 10.0
    num_obstacles: int = 3
    max_steps: int = 200
    model_name: str = "drone_policy_rl"
    warm_start_from: str | None = None
    seed: int = 0
    batch_episodes: int = 8
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0


@dataclass
class TrainRLResponse:
    model_path: str
    episodes: int
    final_success_rate: float
    avg_reward_last20: float
    reward_history: list[float]
    success_history: list[int]
    smoothed_reward: list[float]
    smoothed_success: list[float]


def _obs_to_tensor(obs: Observation) -> torch.Tensor:
    return torch.tensor(
        [
            obs.distance_to_target,
            obs.target_angle_deg,
            obs.front_distance,
            obs.left_distance,
            obs.right_distance,
            obs.battery,
        ],
        dtype=torch.float32,
    )


def _moving_average(xs: list[float], window: int = 20) -> list[float]:
    if not xs:
        return []
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in xs:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(round(s / len(q), 3))
    return out


def _try_load_bc(name: str, hidden_size: int) -> DronePolicy | None:
    path = run_store.models_dir() / f"{name}.pt"
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload["state_dict"]
        # Remap legacy "net.*" keys onto trunk+policy_head if needed
        if any(k.startswith("net.") for k in sd):
            new_sd: dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                if k.startswith("net.0."):
                    new_sd["trunk.0." + k.split(".", 2)[2]] = v
                elif k.startswith("net.2."):
                    new_sd["trunk.2." + k.split(".", 2)[2]] = v
                elif k.startswith("net.4."):
                    new_sd["policy_head." + k.split(".", 2)[2]] = v
            sd = new_sd
        policy = DronePolicy(hidden_size=payload.get("hidden_size", hidden_size))
        policy.load_state_dict(sd, strict=False)
        return policy
    except Exception:
        return None


def train_rl(req: TrainRLRequest) -> TrainRLResponse:
    episodes = max(1, min(req.episodes, MAX_EPISODES))
    rng = random.Random(req.seed)
    torch.manual_seed(req.seed)

    ac = ActorCritic(hidden_size=req.hidden_size)
    if req.warm_start_from:
        bc = _try_load_bc(req.warm_start_from, req.hidden_size)
        if bc is not None:
            ac.load_bc_weights(bc)

    opt = torch.optim.Adam(ac.parameters(), lr=req.learning_rate)

    reward_history: list[float] = []
    success_history: list[int] = []
    STATE.start(
        "train_rl",
        detail=f"A2C · {episodes} episodes · D{req.difficulty}",
    )

    # Rolling buffers for batched updates
    batch_log_probs: list[torch.Tensor] = []
    batch_values: list[torch.Tensor] = []
    batch_entropies: list[torch.Tensor] = []
    batch_returns: list[float] = []
    batch_ep_count = 0

    try:
        for ep_i in range(episodes):
            env = generate_environment(
                GenerateEnvRequest(
                    difficulty=req.difficulty,
                    room_width=req.room_width,
                    room_height=req.room_height,
                    num_obstacles=req.num_obstacles,
                    seed=rng.randint(0, 10_000_000),
                )
            )
            sim = Simulator(env, seed=rng.randint(0, 10_000_000))

            ep_log_probs: list[torch.Tensor] = []
            ep_values: list[torch.Tensor] = []
            ep_entropies: list[torch.Tensor] = []
            ep_rewards: list[float] = []

            obs = sim.initial_observation()
            done = False
            steps = 0
            while not done and steps < req.max_steps:
                x = _obs_to_tensor(obs).unsqueeze(0)
                logits, value = ac(x)
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                ep_log_probs.append(dist.log_prob(action_idx).squeeze(0))
                ep_values.append(value.squeeze(0))
                ep_entropies.append(dist.entropy().squeeze(0))
                action = ACTIONS[int(action_idx.item())]
                obs, r, done, _ev = sim.step(action)
                ep_rewards.append(r)
                steps += 1

            # Discounted returns per episode
            returns: list[float] = []
            G = 0.0
            for r in reversed(ep_rewards):
                G = r + req.gamma * G
                returns.insert(0, G)

            batch_log_probs.extend(ep_log_probs)
            batch_values.extend(ep_values)
            batch_entropies.extend(ep_entropies)
            batch_returns.extend(returns)
            batch_ep_count += 1

            total_r = sum(ep_rewards)
            reward_history.append(round(total_r, 2))
            success_history.append(1 if sim.success else 0)

            # Train every `batch_episodes` episodes
            if batch_ep_count >= req.batch_episodes:
                log_probs_t = torch.stack(batch_log_probs)
                values_t = torch.stack(batch_values)
                entropies_t = torch.stack(batch_entropies)
                returns_t = torch.tensor(batch_returns, dtype=torch.float32)

                # Advantages: returns minus critic baseline, normalized across batch
                advantages = returns_t - values_t.detach()
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

                policy_loss = -(log_probs_t * advantages).mean()
                value_loss = nn.functional.mse_loss(values_t, returns_t)
                entropy_loss = -entropies_t.mean()
                loss = (
                    policy_loss
                    + req.value_coef * value_loss
                    + req.entropy_coef * entropy_loss
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), req.max_grad_norm)
                opt.step()

                batch_log_probs.clear()
                batch_values.clear()
                batch_entropies.clear()
                batch_returns.clear()
                batch_ep_count = 0

            if (ep_i + 1) % max(1, req.batch_episodes) == 0 or ep_i == episodes - 1:
                window = success_history[-50:]
                sr = sum(window) / max(len(window), 1)
                avg_r = sum(reward_history[-50:]) / max(len(reward_history[-50:]), 1)
                STATE.update(
                    detail=f"ep {ep_i + 1}/{episodes} · reward(50)={avg_r:.1f} · succ(50)={sr:.0%}",
                    progress=(ep_i + 1) / episodes,
                )
    finally:
        STATE.finish()

    model_dir = run_store.models_dir()
    model_path = model_dir / f"{req.model_name}.pt"
    torch.save(
        {
            "state_dict": ac.state_dict(),
            "model_type": "actor_critic",
            "hidden_size": req.hidden_size,
            "input_size": INPUT_SIZE,
            "num_actions": NUM_ACTIONS,
            "actions": ACTIONS,
        },
        model_path,
    )

    smoothed_reward = _moving_average(reward_history, window=20)
    smoothed_success = _moving_average([float(s) for s in success_history], window=20)
    final_succ = sum(success_history[-50:]) / max(len(success_history[-50:]), 1)
    avg_last = sum(reward_history[-50:]) / max(len(reward_history[-50:]), 1)

    meta = {
        "model_name": req.model_name,
        "trained_at": time.time(),
        "method": "a2c",
        "episodes": episodes,
        "batch_episodes": req.batch_episodes,
        "learning_rate": req.learning_rate,
        "gamma": req.gamma,
        "entropy_coef": req.entropy_coef,
        "value_coef": req.value_coef,
        "hidden_size": req.hidden_size,
        "difficulty": req.difficulty,
        "warm_start_from": req.warm_start_from,
        "final_success_rate": round(final_succ, 4),
        "avg_reward_last20": round(avg_last, 2),
        "reward_history": reward_history,
        "success_history": success_history,
        "smoothed_reward": smoothed_reward,
        "smoothed_success": smoothed_success,
    }
    (model_dir / f"{req.model_name}.meta.json").write_text(json.dumps(meta, indent=2))

    return TrainRLResponse(
        model_path=str(model_path),
        episodes=episodes,
        final_success_rate=round(final_succ, 4),
        avg_reward_last20=round(avg_last, 2),
        reward_history=reward_history,
        success_history=success_history,
        smoothed_reward=smoothed_reward,
        smoothed_success=smoothed_success,
    )
