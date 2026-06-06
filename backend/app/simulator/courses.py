"""Canonical race courses the arcade uses.

The gym's success criterion is "can this pilot clear these courses?", so the
course definitions live here (not in the frontend) and the trainer + evaluator
import them directly. Keep names + params in sync with
`frontend/src/arcade/RaceScreen.tsx`.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import GenerateEnvRequest


@dataclass(frozen=True)
class Course:
    id: str
    label: str
    difficulty: int
    num_obstacles: int
    room_width: float = 10.0
    room_height: float = 10.0
    wind_strength: float = 0.0
    sensor_noise: float = 0.0
    # A fixed eval seed pool — same N rooms every time so two trained pilots
    # are compared on the same exact rooms (essential for reading deltas).
    eval_seeds: tuple[int, ...] = tuple(range(1000, 1020))

    def env_request(self, seed: int) -> GenerateEnvRequest:
        return GenerateEnvRequest(
            difficulty=self.difficulty,
            room_width=self.room_width,
            room_height=self.room_height,
            num_obstacles=self.num_obstacles,
            wind_strength=self.wind_strength,
            sensor_noise=self.sensor_noise,
            seed=seed,
        )


COURSES: dict[str, Course] = {
    "breeze": Course(id="breeze", label="Breeze", difficulty=1, num_obstacles=2),
    "cruise": Course(id="cruise", label="Cruise", difficulty=1, num_obstacles=3),
    "tight": Course(id="tight", label="Tight", difficulty=3, num_obstacles=5),
}


def all_courses() -> list[Course]:
    return list(COURSES.values())


def get_course(course_id: str) -> Course:
    if course_id not in COURSES:
        raise KeyError(f"Unknown course: {course_id}")
    return COURSES[course_id]
