"""FastAPI app entrypoint."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Eagerly import torch and dynamo on the main thread. PyTorch 2.12 has a
# lazy-init path in torch.optim that imports torch._dynamo only on first use;
# triggering it from a worker thread (as FastAPI does) hits a partial-enum-
# initialization bug. Forcing the import here avoids the worker-thread race.
import torch  # noqa: F401
try:  # noqa: SIM105
    import torch._dynamo  # noqa: F401
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router


app = FastAPI(title="DroneGym", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
