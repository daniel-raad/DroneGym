"""Disk-backed JSON storage for episodes and datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models import EpisodeResponse, EpisodeSummary


# Resolve relative to this file → backend/app/storage → repo/data
BASE = Path(__file__).resolve().parents[3] / "data"
RUNS_DIR = BASE / "runs"
DATASETS_DIR = BASE / "datasets"


def _ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def save_episode(ep: EpisodeResponse) -> Path:
    _ensure_dirs()
    out = RUNS_DIR / ep.episode_id
    out.mkdir(parents=True, exist_ok=True)

    (out / "environment.json").write_text(ep.environment.model_dump_json(indent=2))
    with (out / "trajectory.jsonl").open("w") as f:
        for p in ep.trajectory:
            f.write(p.model_dump_json() + "\n")
    with (out / "observations.jsonl").open("w") as f:
        for o in ep.observations:
            f.write(o.model_dump_json() + "\n")
    with (out / "actions.jsonl").open("w") as f:
        for a in ep.actions:
            f.write(json.dumps({"action": a}) + "\n")
    with (out / "events.jsonl").open("w") as f:
        for e in ep.events:
            f.write(e.model_dump_json() + "\n")
    (out / "summary.json").write_text(ep.summary.model_dump_json(indent=2))
    (out / "episode.json").write_text(ep.model_dump_json())
    return out


def load_episode(episode_id: str) -> EpisodeResponse | None:
    path = RUNS_DIR / episode_id / "episode.json"
    if not path.exists():
        return None
    return EpisodeResponse.model_validate_json(path.read_text())


def list_episode_summaries() -> list[EpisodeSummary]:
    _ensure_dirs()
    out: list[EpisodeSummary] = []
    if not RUNS_DIR.exists():
        return out
    for sub in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        sf = sub / "summary.json"
        if sf.exists():
            try:
                out.append(EpisodeSummary.model_validate_json(sf.read_text()))
            except Exception:
                continue
    return out


def dataset_path(name: str) -> Path:
    _ensure_dirs()
    return DATASETS_DIR / f"{name}.jsonl"


def write_dataset(name: str, samples: Iterable[dict], append: bool = False) -> tuple[Path, int]:
    _ensure_dirs()
    p = dataset_path(name)
    mode = "a" if append and p.exists() else "w"
    n = 0
    with p.open(mode) as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            n += 1
    # If appended, return the TOTAL line count
    if mode == "a":
        with p.open() as f:
            total = sum(1 for _ in f)
        return p, total
    return p, n


def read_dataset(name: str) -> list[dict]:
    p = dataset_path(name)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def models_dir() -> Path:
    d = Path(__file__).resolve().parents[3] / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
