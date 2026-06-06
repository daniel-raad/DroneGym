"""LLM-driven drone agent.

Uses Anthropic's Claude (Haiku by default for speed) to pick an action each step
from the 6-number observation. The system prompt is held constant across steps
so prompt caching reduces input cost on repeated calls.

Falls back to `hover` if no API key is configured or the model returns invalid JSON.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from ..models import ACTIONS, Observation


SYSTEM_PROMPT = """You are flying a 2D drone in a top-down room. Each step you receive a JSON
observation and must reply with a single JSON object — nothing else.

OBSERVATION FIELDS
  distance_to_target  — straight-line distance from drone to target (room cells)
  target_angle_deg    — bearing of the target relative to your heading.
                        0 means dead ahead, +90 is 90° to your left, -90 is 90° to your right.
  front_distance      — raycast straight ahead. <0.7 means imminent collision.
  left_distance       — raycast 90° to your left (along heading+90°).
  right_distance      — raycast 90° to your right (along heading-90°).
                        These three rays are your ONLY perception of obstacles.
                        You CANNOT see diagonals — an obstacle at 45° off heading
                        will not show up in any ray.
  battery             — remaining battery in arbitrary units (drops 0.5/step).
  step                — step index in this episode.

LEGAL ACTIONS  (you must pick exactly one):
  move_forward, move_back, move_left, move_right,
  turn_left  (heading -= 15°),
  turn_right (heading += 15°),
  hover,
  land       (ends the episode; success only if very close to target)

STRATEGY HINTS
- Land only when distance_to_target < 0.8.
- If front_distance < 0.9, do not move_forward — turn or strafe to the side with more clearance.
- If target_angle_deg > 15, turn_left until it shrinks. If < -15, turn_right.
- Strafing (move_left / move_right) corrects small bearing errors without re-aiming.

REPLY FORMAT (strict JSON, no prose, no markdown fence):
{"action": "<one of the legal actions>", "reason": "<short reason, <=12 words>"}
"""


_ACTION_RE = re.compile(r'"action"\s*:\s*"([a-z_]+)"')


def _parse_action(text: str) -> Optional[str]:
    """Try strict JSON first, then a regex fallback for messy outputs."""
    text = text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        action = data.get("action")
        if isinstance(action, str) and action in ACTIONS:
            return action
    except Exception:
        pass
    m = _ACTION_RE.search(text)
    if m and m.group(1) in ACTIONS:
        return m.group(1)
    return None


class LLMDroneAgent:
    """Calls Claude per step to choose an action."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 80,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("DRONEGYM_LLM_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.last_reason: str = ""
        self._client = None
        if api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def act(self, obs: Observation) -> str:
        if self._client is None:
            return "hover"
        user_msg = json.dumps(
            {
                "distance_to_target": round(obs.distance_to_target, 2),
                "target_angle_deg": round(obs.target_angle_deg, 1),
                "front_distance": round(obs.front_distance, 2),
                "left_distance": round(obs.left_distance, 2),
                "right_distance": round(obs.right_distance, 2),
                "battery": round(obs.battery, 0),
                "step": obs.step,
            }
        )
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            # First content block is text; defend against tool/other blocks.
            text = ""
            for block in resp.content:
                if getattr(block, "type", "text") == "text":
                    text = block.text
                    break
            action = _parse_action(text)
            if action is None:
                return "hover"
            # Capture reason for telemetry/debug
            try:
                self.last_reason = json.loads(text).get("reason", "")
            except Exception:
                self.last_reason = ""
            return action
        except Exception:
            return "hover"
