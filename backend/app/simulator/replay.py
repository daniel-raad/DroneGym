"""Replay serialization helpers."""
from __future__ import annotations

from ..models import EpisodeResponse


def episode_to_dict(ep: EpisodeResponse) -> dict:
    return ep.model_dump()
