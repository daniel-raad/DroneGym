"""Policy-gradient training (A2C + PPO) in the simulator.

Knobs surfaced to the gym UI:
- algorithm: a2c (default) or ppo
- curriculum_schedule: cycle env difficulty per episode → robust pilots
- randomize_wind/noise/obstacles: domain randomization per episode
- entropy_coef / value_coef / batch_episodes / num_layers / hidden_size

Live rollout snapshots are pushed into STATE.extra["rollout"] every
ROLLOUT_EVERY episodes so the gym UI can render the current policy on a fixed
preview world.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from ..models import ACTIONS, Observation
from ..runtime_state import STATE
from ..simulator.env import Simulator, generate_environment
from ..models import GenerateEnvRequest
from ..storage import run_store
from .policy_model import ActorCritic, DronePolicy, NUM_ACTIONS, input_size_for


MAX_EPISODES = 20_000  # hard cap; UI clamps lower but server cap is generous
ROLLOUT_EVERY = 50     # publish a live rollout snapshot every N episodes


@dataclass
class TrainRLRequest:
    episodes: int = 400
    learning_rate: float = 3e-4
    gamma: float = 0.97
    hidden_size: int = 64
    num_layers: int = 2
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
    num_rays: int | None = None  # None → inherit from warm_start meta, fall back to 3
    algorithm: str = "a2c"  # "a2c" | "ppo"
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    curriculum_schedule: list[int] | None = None
    randomize_wind: float = 0.0
    randomize_noise: float = 0.0
    randomize_obstacles: int = 0


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


def _obs_to_tensor(obs: Observation, num_rays: int) -> torch.Tensor:
    if num_rays == 3:
        if len(obs.rays) >= 3:
            feats = [
                obs.distance_to_target,
                obs.target_angle_deg,
                obs.rays[0],
                obs.rays[1],
                obs.rays[2],
                obs.battery,
            ]
        else:
            feats = [
                obs.distance_to_target,
                obs.target_angle_deg,
                obs.front_distance,
                obs.left_distance,
                obs.right_distance,
                obs.battery,
            ]
    else:
        feats = [obs.distance_to_target, obs.target_angle_deg, *obs.rays, obs.battery]
    return torch.tensor(feats, dtype=torch.float32)


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


def _try_load_bc(name: str, hidden_size: int, input_size: int, num_layers: int) -> DronePolicy | None:
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
        bc_layers = payload.get("num_layers")
        if bc_layers is None:
            trunk_linears = sum(1 for k in sd if k.startswith("trunk.") and k.endswith(".weight"))
            bc_layers = max(trunk_linears, 1)
        policy = DronePolicy(
            hidden_size=payload.get("hidden_size", hidden_size),
            input_size=payload.get("input_size", input_size),
            num_layers=bc_layers,
        )
        policy.load_state_dict(sd, strict=False)
        return policy
    except Exception:
        return None


def _bc_meta_num_rays(name: str) -> int | None:
    path = run_store.models_dir() / f"{name}.meta.json"
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
        v = meta.get("num_rays")
        return int(v) if v is not None else None
    except Exception:
        return None


def _make_env_for_episode(
    req: TrainRLRequest,
    ep_i: int,
    rng: random.Random,
) -> tuple[GenerateEnvRequest, int]:
    """Build the env request for this episode (curriculum + randomization applied)."""
    if req.curriculum_schedule:
        difficulty = req.curriculum_schedule[ep_i % len(req.curriculum_schedule)]
    else:
        difficulty = req.difficulty

    wind = rng.uniform(0.0, req.randomize_wind) if req.randomize_wind > 0 else 0.0
    noise = rng.uniform(0.0, req.randomize_noise) if req.randomize_noise > 0 else 0.0
    obstacles = req.num_obstacles
    if req.randomize_obstacles > 0:
        obstacles = max(0, obstacles + rng.randint(-req.randomize_obstacles, req.randomize_obstacles))
    return (
        GenerateEnvRequest(
            difficulty=difficulty,
            room_width=req.room_width,
            room_height=req.room_height,
            num_obstacles=obstacles,
            wind_strength=wind,
            sensor_noise=noise,
            seed=rng.randint(0, 10_000_000),
        ),
        difficulty,
    )


def _rollout_snapshot(ac: ActorCritic, req: TrainRLRequest, num_rays: int) -> dict:
    """Deterministic rollout on a fixed preview world — what the UI shows live."""
    preview_seed = 1234
    env = generate_environment(
        GenerateEnvRequest(
            difficulty=req.difficulty,
            room_width=req.room_width,
            room_height=req.room_height,
            num_obstacles=req.num_obstacles,
            seed=preview_seed,
        )
    )
    sim = Simulator(env, seed=preview_seed, num_rays=num_rays)
    obs = sim.initial_observation()
    traj = [(sim.x, sim.y)]
    actions: list[str] = []
    steps = 0
    ac.eval()
    with torch.no_grad():
        while not sim.done and steps < req.max_steps:
            x = _obs_to_tensor(obs, num_rays).unsqueeze(0)
            logits, _ = ac(x)
            action_idx = int(torch.argmax(logits, dim=1).item())
            action = ACTIONS[action_idx]
            actions.append(action)
            obs, _r, _done, _ev = sim.step(action)
            traj.append((sim.x, sim.y))
            steps += 1
    ac.train()
    return {
        "env": env.model_dump(),
        "trajectory": [{"x": round(x, 3), "y": round(y, 3)} for x, y in traj],
        "actions": actions,
        "success": sim.success,
        "collision": sim.collided,
        "steps": sim.step_idx,
    }


def _ppo_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    old_logp_t: torch.Tensor,
    returns_t: torch.Tensor,
    req: TrainRLRequest,
) -> None:
    advantages = returns_t - ac(obs_t)[1].detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    for _ in range(req.ppo_epochs):
        logits, values = ac(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        new_logp = dist.log_prob(act_t)
        entropy = dist.entropy().mean()
        ratio = (new_logp - old_logp_t).exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - req.ppo_clip, 1.0 + req.ppo_clip) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = nn.functional.mse_loss(values, returns_t)
        loss = policy_loss + req.value_coef * value_loss - req.entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), req.max_grad_norm)
        opt.step()


def train_rl(req: TrainRLRequest) -> TrainRLResponse:
    episodes = max(1, min(req.episodes, MAX_EPISODES))
    rng = random.Random(req.seed)
    torch.manual_seed(req.seed)

    # Resolve num_rays: explicit > warm-start meta > 3.
    if req.num_rays is not None:
        num_rays = req.num_rays
    elif req.warm_start_from and (m := _bc_meta_num_rays(req.warm_start_from)) is not None:
        num_rays = m
    else:
        num_rays = 3
    input_size = input_size_for(num_rays)

    ac = ActorCritic(hidden_size=req.hidden_size, input_size=input_size, num_layers=req.num_layers)
    if req.warm_start_from:
        bc = _try_load_bc(req.warm_start_from, req.hidden_size, input_size, req.num_layers)
        if bc is not None:
            ac.load_bc_weights(bc)

    opt = torch.optim.Adam(ac.parameters(), lr=req.learning_rate)

    reward_history: list[float] = []
    success_history: list[int] = []
    STATE.start(
        "train_rl",
        detail=f"{req.algorithm.upper()} · {episodes} eps · D{req.difficulty}",
        algorithm=req.algorithm,
    )

    # Per-step buffers for batched updates
    batch_obs: list[torch.Tensor] = []
    batch_actions: list[int] = []
    batch_log_probs: list[torch.Tensor] = []
    batch_values: list[torch.Tensor] = []
    batch_entropies: list[torch.Tensor] = []
    batch_returns: list[float] = []
    batch_ep_count = 0

    try:
        for ep_i in range(episodes):
            env_req, _diff = _make_env_for_episode(req, ep_i, rng)
            env = generate_environment(env_req)
            sim = Simulator(env, seed=rng.randint(0, 10_000_000), num_rays=num_rays)

            ep_obs: list[torch.Tensor] = []
            ep_actions: list[int] = []
            ep_log_probs: list[torch.Tensor] = []
            ep_values: list[torch.Tensor] = []
            ep_entropies: list[torch.Tensor] = []
            ep_rewards: list[float] = []

            obs = sim.initial_observation()
            done = False
            steps = 0
            while not done and steps < req.max_steps:
                x = _obs_to_tensor(obs, num_rays).unsqueeze(0)
                logits, value = ac(x)
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                ep_obs.append(x.squeeze(0))
                ep_actions.append(int(action_idx.item()))
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

            batch_obs.extend(ep_obs)
            batch_actions.extend(ep_actions)
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
                if req.algorithm == "ppo":
                    obs_t = torch.stack(batch_obs)
                    act_t = torch.tensor(batch_actions, dtype=torch.long)
                    old_logp_t = torch.stack(batch_log_probs).detach()
                    returns_t = torch.tensor(batch_returns, dtype=torch.float32)
                    _ppo_update(ac, opt, obs_t, act_t, old_logp_t, returns_t, req)
                else:
                    log_probs_t = torch.stack(batch_log_probs)
                    values_t = torch.stack(batch_values)
                    entropies_t = torch.stack(batch_entropies)
                    returns_t = torch.tensor(batch_returns, dtype=torch.float32)

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

                batch_obs.clear()
                batch_actions.clear()
                batch_log_probs.clear()
                batch_values.clear()
                batch_entropies.clear()
                batch_returns.clear()
                batch_ep_count = 0

            # Live rollout snapshot for the UI
            if (ep_i + 1) % ROLLOUT_EVERY == 0 or ep_i == episodes - 1:
                try:
                    snap = _rollout_snapshot(ac, req, num_rays)
                    STATE.update(rollout=snap)
                except Exception:
                    pass  # snapshots are best-effort; never fail training over them

            if (ep_i + 1) % max(1, req.batch_episodes) == 0 or ep_i == episodes - 1:
                window = success_history[-50:]
                sr = sum(window) / max(len(window), 1)
                avg_r = sum(reward_history[-50:]) / max(len(reward_history[-50:]), 1)
                STATE.update(
                    detail=f"ep {ep_i + 1}/{episodes} · reward(50)={avg_r:.1f} · succ(50)={sr:.0%}",
                    progress=(ep_i + 1) / episodes,
                    smoothed_reward=_moving_average(reward_history, window=20),
                    smoothed_success=_moving_average(
                        [float(s) for s in success_history], window=20
                    ),
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
            "num_layers": req.num_layers,
            "input_size": input_size,
            "num_rays": num_rays,
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
        "method": req.algorithm,  # "a2c" or "ppo"
        "algorithm": req.algorithm,
        "episodes": episodes,
        "batch_episodes": req.batch_episodes,
        "learning_rate": req.learning_rate,
        "gamma": req.gamma,
        "entropy_coef": req.entropy_coef,
        "value_coef": req.value_coef,
        "hidden_size": req.hidden_size,
        "num_layers": req.num_layers,
        "num_rays": num_rays,
        "difficulty": req.difficulty,
        "curriculum_schedule": req.curriculum_schedule,
        "randomize_wind": req.randomize_wind,
        "randomize_noise": req.randomize_noise,
        "randomize_obstacles": req.randomize_obstacles,
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
