"""Hand-coded drone agent — turns toward target, avoids obstacles, lands when close."""
from __future__ import annotations

import random

from ..models import Observation


class HeuristicAgent:
    """Greedy navigation: aim, strafe around obstacles, land near target."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._avoid_dir: str | None = None  # 'move_left' / 'move_right'
        self._avoid_steps_left = 0

    # Safe move clearance: drone_radius (0.25) + step_size (0.3) + buffer
    SAFE = 0.7

    def act(self, obs: Observation) -> str:
        if obs.distance_to_target < 0.7:
            return "land"

        angle = obs.target_angle_deg
        front_blocked = obs.front_distance < self.SAFE + 0.3
        left_safe = obs.left_distance > self.SAFE
        right_safe = obs.right_distance > self.SAFE

        # Continue an in-progress sidestep around an obstacle
        if self._avoid_steps_left > 0 and self._avoid_dir is not None:
            self._avoid_steps_left -= 1
            if not front_blocked:
                self._avoid_dir = None
                self._avoid_steps_left = 0
            else:
                if self._avoid_dir == "move_left" and not left_safe:
                    self._avoid_dir = "turn_right"
                elif self._avoid_dir == "move_right" and not right_safe:
                    self._avoid_dir = "turn_left"
                return self._avoid_dir

        if front_blocked:
            if left_safe and (not right_safe or obs.left_distance >= obs.right_distance):
                self._avoid_dir = "move_left"
                self._avoid_steps_left = 4
                return self._avoid_dir
            if right_safe:
                self._avoid_dir = "move_right"
                self._avoid_steps_left = 4
                return self._avoid_dir
            # No safe sidestep: rotate toward more open side
            return "turn_left" if obs.left_distance >= obs.right_distance else "turn_right"

        # If something is very close to one side, sidestep AWAY from it before going forward
        if obs.left_distance < 0.5 and right_safe:
            return "move_right"
        if obs.right_distance < 0.5 and left_safe:
            return "move_left"

        # Strong misalignment: turn
        if angle > 18:
            return "turn_left"
        if angle < -18:
            return "turn_right"

        # Mild misalignment with safe sides: strafe to correct
        if 6 < angle <= 18 and left_safe:
            return "move_left"
        if -18 <= angle < -6 and right_safe:
            return "move_right"

        return "move_forward"


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.actions = [
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_right",
            "move_left",
            "move_right",
            "hover",
        ]

    def act(self, obs: Observation) -> str:
        if obs.distance_to_target < 0.7:
            return "land"
        return self.rng.choice(self.actions)
