"""Agent interfaces."""
from __future__ import annotations

from typing import Protocol

from ..models import Observation


class DroneAgent(Protocol):
    def act(self, observation: Observation) -> str: ...
