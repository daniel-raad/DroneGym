"""Use a trained policy (BC or actor-critic) as a DroneAgent."""
from __future__ import annotations

import torch

from ..models import ACTIONS, Observation
from .policy_model import ActorCritic, DronePolicy


class TrainedPolicyAgent:
    """Acts greedily by argmax over policy logits.

    Handles both DronePolicy (behavior-cloning) and ActorCritic (A2C) checkpoints.
    """

    def __init__(self, model):
        self.model = model
        self.is_actor_critic = isinstance(model, ActorCritic)
        self.model.eval()

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_actor_critic:
            return self.model.policy_logits(x)
        return self.model(x)

    def act(self, obs: Observation) -> str:
        x = torch.tensor(
            [
                [
                    obs.distance_to_target,
                    obs.target_angle_deg,
                    obs.front_distance,
                    obs.left_distance,
                    obs.right_distance,
                    obs.battery,
                ]
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = self._logits(x)
            idx = int(torch.argmax(logits, dim=1).item())
        return ACTIONS[idx]
