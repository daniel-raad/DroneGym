"""Physics, collision detection, and raycasting helpers."""
from __future__ import annotations

import math
from typing import Iterable

from ..models import Obstacle


STEP_SIZE = 0.3
TURN_DELTA_DEG = 15.0
RAY_MAX = 6.0
RAY_STEP = 0.05


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def normalize_angle_deg(deg: float) -> float:
    """Normalize to [-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return d


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def point_in_obstacle(x: float, y: float, obstacles: Iterable[Obstacle], drone_radius: float) -> bool:
    for ob in obstacles:
        if distance(x, y, ob.x, ob.y) <= ob.radius + drone_radius:
            return True
    return False


def out_of_bounds(x: float, y: float, w: float, h: float, drone_radius: float) -> bool:
    return (
        x < drone_radius
        or x > w - drone_radius
        or y < drone_radius
        or y > h - drone_radius
    )


def raycast(
    x: float,
    y: float,
    angle_deg: float,
    room_w: float,
    room_h: float,
    obstacles: Iterable[Obstacle],
    max_dist: float = RAY_MAX,
) -> float:
    """Approximate ray marching. Returns distance to nearest hit."""
    rad = deg_to_rad(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)
    d = 0.0
    while d < max_dist:
        d += RAY_STEP
        px = x + dx * d
        py = y + dy * d
        if px <= 0 or px >= room_w or py <= 0 or py >= room_h:
            return d
        for ob in obstacles:
            if distance(px, py, ob.x, ob.y) <= ob.radius:
                return d
    return max_dist


def apply_action(
    x: float,
    y: float,
    heading_deg: float,
    action: str,
    step_size: float = STEP_SIZE,
) -> tuple[float, float, float]:
    """Returns new (x, y, heading) after action."""
    h_rad = deg_to_rad(heading_deg)
    if action == "move_forward":
        x += math.cos(h_rad) * step_size
        y += math.sin(h_rad) * step_size
    elif action == "move_back":
        x -= math.cos(h_rad) * step_size
        y -= math.sin(h_rad) * step_size
    elif action == "move_left":
        # strafe left: heading + 90
        x += math.cos(h_rad + math.pi / 2) * step_size
        y += math.sin(h_rad + math.pi / 2) * step_size
    elif action == "move_right":
        x += math.cos(h_rad - math.pi / 2) * step_size
        y += math.sin(h_rad - math.pi / 2) * step_size
    elif action == "turn_left":
        heading_deg = normalize_angle_deg(heading_deg - TURN_DELTA_DEG)
    elif action == "turn_right":
        heading_deg = normalize_angle_deg(heading_deg + TURN_DELTA_DEG)
    # hover / land: no motion
    return x, y, heading_deg


def min_obstacle_clearance(
    x: float, y: float, obstacles: Iterable[Obstacle], room_w: float, room_h: float
) -> float:
    """Distance to nearest obstacle surface or wall."""
    best = min(x, y, room_w - x, room_h - y)
    for ob in obstacles:
        d = distance(x, y, ob.x, ob.y) - ob.radius
        if d < best:
            best = d
    return max(best, 0.0)
