"""In-memory tracker of what the backend is currently doing."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CurrentTask:
    name: str
    detail: str = ""
    started_at: float = field(default_factory=time.time)
    progress: float = 0.0  # 0..1
    extra: dict = field(default_factory=dict)


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._task: Optional[CurrentTask] = None

    def start(self, name: str, detail: str = "", **extra) -> None:
        with self._lock:
            self._task = CurrentTask(name=name, detail=detail, extra=extra)

    def update(self, detail: str | None = None, progress: float | None = None, **extra) -> None:
        with self._lock:
            if self._task is None:
                return
            if detail is not None:
                self._task.detail = detail
            if progress is not None:
                self._task.progress = max(0.0, min(1.0, progress))
            if extra:
                self._task.extra.update(extra)

    def finish(self) -> None:
        with self._lock:
            self._task = None

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            if self._task is None:
                return None
            return {
                "name": self._task.name,
                "detail": self._task.detail,
                "started_at": self._task.started_at,
                "elapsed": round(time.time() - self._task.started_at, 2),
                "progress": round(self._task.progress, 3),
                "extra": dict(self._task.extra),
            }


STATE = RuntimeState()
