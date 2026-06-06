"""Train the PyTorch MLP from a JSONL imitation dataset."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from ..models import ACTIONS, TrainPolicyRequest, TrainPolicyResponse
from ..runtime_state import STATE
from ..storage import run_store
from .policy_model import DronePolicy, INPUT_SIZE, NUM_ACTIONS


ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def _samples_to_tensors(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for s in samples:
        o = s["observation"]
        xs.append(
            [
                o["distance_to_target"],
                o["target_angle_deg"],
                o["front_distance"],
                o["left_distance"],
                o["right_distance"],
                o["battery"],
            ]
        )
        ys.append(ACTION_TO_IDX[s["action"]])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_policy(req: TrainPolicyRequest) -> TrainPolicyResponse:
    rows = run_store.read_dataset(req.dataset_name)
    if not rows:
        raise ValueError(f"Dataset '{req.dataset_name}' is empty or missing")

    rng = random.Random(0)
    rng.shuffle(rows)
    split = int(len(rows) * 0.8)
    train_rows = rows[:split]
    test_rows = rows[split:] or rows[-1:]

    X_train, y_train = _samples_to_tensors(train_rows)
    X_test, y_test = _samples_to_tensors(test_rows)

    model = DronePolicy(hidden_size=req.hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=req.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_train)
    loss_history: list[float] = []
    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    STATE.start("train_policy", detail=f"dataset {req.dataset_name} ({len(rows)} samples)")
    last_loss = 0.0
    try:
        for epoch in range(req.epochs):
            model.train()
            idx = torch.randperm(n)
            running = 0.0
            for start in range(0, n, req.batch_size):
                batch_idx = idx[start : start + req.batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(batch_idx)
            last_loss = running / max(n, 1)
            loss_history.append(round(last_loss, 4))

            model.eval()
            with torch.no_grad():
                tr_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
                te_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            train_acc_history.append(round(tr_acc, 4))
            test_acc_history.append(round(te_acc, 4))

            STATE.update(
                detail=f"epoch {epoch + 1}/{req.epochs} loss={last_loss:.3f} test_acc={te_acc:.2%}",
                progress=(epoch + 1) / req.epochs,
                loss_history=loss_history,
                test_acc_history=test_acc_history,
            )
    finally:
        STATE.finish()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).argmax(dim=1)
        test_pred = model(X_test).argmax(dim=1)
        train_acc = (train_pred == y_train).float().mean().item()
        test_acc = (test_pred == y_test).float().mean().item()

    model_dir = run_store.models_dir()
    model_path = model_dir / f"{req.model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "drone_policy",
            "hidden_size": req.hidden_size,
            "input_size": INPUT_SIZE,
            "num_actions": NUM_ACTIONS,
            "actions": ACTIONS,
        },
        model_path,
    )
    meta = {
        "model_name": req.model_name,
        "trained_at": time.time(),
        "method": "behavior_cloning",
        "dataset_name": req.dataset_name,
        "num_samples": len(rows),
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "hidden_size": req.hidden_size,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "final_loss": round(last_loss, 4),
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "test_acc_history": test_acc_history,
    }
    (model_dir / f"{req.model_name}.meta.json").write_text(json.dumps(meta, indent=2))

    return TrainPolicyResponse(
        model_path=str(model_path),
        train_accuracy=round(train_acc, 4),
        test_accuracy=round(test_acc, 4),
        final_loss=round(last_loss, 4),
        epochs=req.epochs,
        num_samples=len(rows),
    )


def load_policy(model_name: str):
    """Load a saved checkpoint. Returns DronePolicy or ActorCritic depending on save type."""
    from .policy_model import ActorCritic

    path = run_store.models_dir() / f"{model_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = payload.get("hidden_size", 64)
    sd = payload["state_dict"]

    # Actor-critic checkpoints have a "value_head.*" key
    if payload.get("model_type") == "actor_critic" or any(k.startswith("value_head") for k in sd):
        ac = ActorCritic(hidden_size=hidden)
        ac.load_state_dict(sd)
        ac.eval()
        return ac

    # Legacy BC checkpoints used Sequential keys "net.0/2/4" — remap to trunk + policy_head
    if any(k.startswith("net.") for k in sd):
        new_sd: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("net.0."):
                new_sd["trunk.0." + k.split(".", 2)[2]] = v
            elif k.startswith("net.2."):
                new_sd["trunk.2." + k.split(".", 2)[2]] = v
            elif k.startswith("net.4."):
                new_sd["policy_head." + k.split(".", 2)[2]] = v
        sd = new_sd

    model = DronePolicy(hidden_size=hidden)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_model_meta(model_name: str) -> dict | None:
    path = run_store.models_dir() / f"{model_name}.meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
