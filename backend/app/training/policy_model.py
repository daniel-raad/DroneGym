"""Policy and actor-critic networks."""
from __future__ import annotations

import torch
from torch import nn

from ..models import ACTIONS


INPUT_SIZE = 6  # legacy: [dist, angle, front, left, right, battery]
NUM_ACTIONS = len(ACTIONS)


def input_size_for(num_rays: int) -> int:
    """Feature vector is [distance_to_target, target_angle_deg, *rays, battery]."""
    return 3 + max(num_rays, 0)


def _build_trunk(input_size: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    """MLP trunk of `num_layers` hidden layers (>=1). Layer 0 maps input → hidden."""
    layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    for _ in range(max(num_layers - 1, 0)):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
    return nn.Sequential(*layers)


class DronePolicy(nn.Module):
    """Behavior-cloning MLP: input → (hidden ReLU) × num_layers → NUM_ACTIONS."""

    def __init__(
        self,
        hidden_size: int = 64,
        input_size: int = INPUT_SIZE,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = max(num_layers, 1)
        self.trunk = _build_trunk(input_size, hidden_size, self.num_layers)
        self.policy_head = nn.Linear(hidden_size, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(x))


class ActorCritic(nn.Module):
    """Shared trunk → (policy_head, value_head) for A2C / PPO."""

    def __init__(
        self,
        hidden_size: int = 64,
        input_size: int = INPUT_SIZE,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = max(num_layers, 1)
        self.trunk = _build_trunk(input_size, hidden_size, self.num_layers)
        self.policy_head = nn.Linear(hidden_size, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(x))

    def load_bc_weights(self, bc: DronePolicy) -> None:
        """Copy the policy MLP from a BC checkpoint; value head stays random.

        Tolerant of depth mismatches: copies only the prefix that lines up.
        """
        self_trunk = list(self.trunk.children())
        bc_trunk = list(bc.trunk.children())
        for sm, bm in zip(self_trunk, bc_trunk):
            if isinstance(sm, nn.Linear) and isinstance(bm, nn.Linear):
                if sm.weight.shape == bm.weight.shape:
                    sm.load_state_dict(bm.state_dict())
        if self.policy_head.weight.shape == bc.policy_head.weight.shape:
            self.policy_head.load_state_dict(bc.policy_head.state_dict())
