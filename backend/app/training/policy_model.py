"""Policy and actor-critic networks."""
from __future__ import annotations

import torch
from torch import nn

from ..models import ACTIONS


INPUT_SIZE = 6
NUM_ACTIONS = len(ACTIONS)


class DronePolicy(nn.Module):
    """Behavior-cloning MLP: 6 → hidden → hidden → 8."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(x))


class ActorCritic(nn.Module):
    """Shared trunk → (policy_head, value_head) for A2C."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(x))

    def load_bc_weights(self, bc: DronePolicy) -> None:
        """Copy the policy MLP from a behavior-cloning checkpoint; value head stays random."""
        self.trunk.load_state_dict(bc.trunk.state_dict())
        self.policy_head.load_state_dict(bc.policy_head.state_dict())
