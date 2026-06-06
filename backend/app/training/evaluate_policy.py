"""Use a trained policy (BC or actor-critic) as a DroneAgent."""
from __future__ import annotations

import torch

from ..models import ACTIONS, Observation
from .policy_model import ActorCritic, DronePolicy


class TrainedPolicyAgent:
    """Acts greedily by argmax over policy logits.

    Handles both DronePolicy (behavior-cloning) and ActorCritic (A2C) checkpoints.
    Carries the model's expected num_rays so the runner can size observations
    correctly and act() can build the right feature vector.
    """

    def __init__(self, model, num_rays: int = 3):
        self.model = model
        self.is_actor_critic = isinstance(model, ActorCritic)
        self.num_rays = num_rays
        self.model.eval()

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_actor_critic:
            return self.model.policy_logits(x)
        return self.model(x)

    def _features(self, obs: Observation) -> list[float]:
        if self.num_rays == 3:
            if len(obs.rays) >= 3:
                return [
                    obs.distance_to_target,
                    obs.target_angle_deg,
                    obs.rays[0],
                    obs.rays[1],
                    obs.rays[2],
                    obs.battery,
                ]
            return [
                obs.distance_to_target,
                obs.target_angle_deg,
                obs.front_distance,
                obs.left_distance,
                obs.right_distance,
                obs.battery,
            ]
        return [obs.distance_to_target, obs.target_angle_deg, *obs.rays, obs.battery]

    def act(self, obs: Observation) -> str:
        x = torch.tensor([self._features(obs)], dtype=torch.float32)
        with torch.no_grad():
            logits = self._logits(x)
            idx = int(torch.argmax(logits, dim=1).item())
        return ACTIONS[idx]
