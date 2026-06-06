# DroneGym

A curriculum-based 2D drone navigation simulator. A drone agent learns to
navigate a top-down room with obstacles toward a target; every trajectory is
logged as synthetic embodied data and replayed in a polished SVG dashboard. A
small PyTorch MLP can be trained on heuristic-agent trajectories to imitate
the controller, closing the loop on the generative-simulation idea.

## Why this is interesting

DroneGym is a tiny embodied-AI sandbox. The same surface — observation → action
loop, environment generator, replay, reward — is what underlies things like
SimplerEnv, Habitat, Procgen, or any synthetic-data generator for robot
policies. By treating environments as procedurally-generated content and
agents as pluggable strategies (heuristic, random, learned, LLM), the
simulator lets you experiment with curriculum learning, imitation learning
and LLM-driven planners against the same physics, scoring, and dashboard.

## Architecture

```
┌───────────────────────────────┐         ┌───────────────────────────────┐
│  React + TypeScript (Vite)    │  HTTP   │  FastAPI + Pydantic           │
│                               │ ─────▶  │                               │
│  EnvironmentConfigPanel       │         │  /api/environments/generate   │
│  RunControls                  │         │  /api/episodes/run            │
│  ReplayCanvas (SVG)           │         │  /api/episodes/{id}, list     │
│  RunSummary / MetricsPanel    │         │  /api/curriculum/next         │
│  EpisodeTimeline              │         │  /api/training/generate-..    │
│  CurriculumPanel              │         │  /api/training/train-policy   │
└───────────────────────────────┘         └─────────────┬─────────────────┘
                                                        │
                                            ┌───────────┴────────────┐
                                            ▼                        ▼
                                  ┌────────────────────┐   ┌──────────────────┐
                                  │  Simulator         │   │  Storage         │
                                  │  - physics         │   │  data/runs/*     │
                                  │  - observations    │   │  data/datasets/* │
                                  │  - evaluator       │   │  models/*.pt     │
                                  └─────────┬──────────┘   └──────────────────┘
                                            │
                                ┌───────────┴───────────┐
                                ▼                       ▼
                       ┌──────────────────┐   ┌────────────────────┐
                       │  Agents          │   │  Training (PyTorch)│
                       │  - heuristic     │   │  - dataset (JSONL) │
                       │  - random        │   │  - MLP policy      │
                       │  - LLM stub      │   │  - imitation train │
                       │  - trained MLP   │   │  - evaluate        │
                       └──────────────────┘   └────────────────────┘
```

## Quick start

```bash
# Terminal 1 — backend
./run-backend.sh
# → http://localhost:8000  (docs at /docs)

# Terminal 2 — frontend

# → http://localhost:5173  (proxies /api → :8000)
```

Python 3.11+ and Node 18+ are required.

## Demo flow (end-to-end)

1. **Generate environment** — adjust difficulty / room size / obstacles in
   the side panel and click `Generate env`.
2. **Run episode** — pick `Heuristic` and click `Run episode`. The drone
   trajectory animates in SVG. Scrub the timeline to step through.
3. **Replay & inspect** — see success/collision/timeout badges, score,
   path length, action distribution, step-by-step timeline.
4. **Run curriculum batch** — runs N episodes, after each one the backend
   curriculum agent adjusts difficulty up/down based on success and
   collision rates.
5. **Generate dataset** — collects observations + actions from successful
   heuristic-agent runs into `data/datasets/imitation_v1.jsonl`.
6. **Train policy** — fits a small PyTorch MLP (6 → 64 → 64 → 8 actions)
   for behavior cloning. Reports train/test accuracy.
7. **Run trained policy** — replays an episode using the trained network.
   Compare it visually and numerically to the heuristic.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/environments/generate` | Procedurally generate a config |
| `POST /api/episodes/run` | Run one episode with chosen agent |
| `POST /api/episodes/run-trained-policy` | Run an episode with the trained MLP |
| `GET  /api/episodes` | List saved episode summaries |
| `GET  /api/episodes/{id}` | Load a saved episode |
| `POST /api/curriculum/next` | Suggest the next environment given history |
| `POST /api/training/generate-dataset` | Build imitation dataset from heuristic runs |
| `POST /api/training/train-policy` | Train the MLP policy |

Open `http://localhost:8000/docs` for live Swagger docs.

## Simulator

- 2D continuous top-down room, `room_width × room_height`.
- Drone state: `x, y, heading (deg), battery, step`.
- Action set: `move_forward, move_back, move_left, move_right, turn_left,
  turn_right, hover, land`.
- Observation (6-dim model input + step): distance + bearing to target,
  three approximate raycasts (front / left / right), battery, step.
- Termination: reached target, collision (wall or obstacle), successful
  land near target, step budget / battery exhausted.
- Score: +100 reach/land, −100 collision, −1 per step, dense shaping for
  distance reduction, small penalty for nosing into close obstacles.
- Optional wind drift and Gaussian sensor noise.

Replay artifacts per episode in `data/runs/{episode_id}/`:
`environment.json`, `trajectory.jsonl`, `observations.jsonl`,
`actions.jsonl`, `events.jsonl`, `summary.json`, `episode.json`.

## Training

`backend/app/training/`:

- `dataset.py` — runs the heuristic agent across N random environments,
  keeps only successful trajectories, writes JSONL samples
  `{observation, action}`.
- `policy_model.py` — small MLP: 6 → 64 → 64 → 8 (cross-entropy).
- `train_policy.py` — 80/20 split, Adam, configurable epochs / batch
  size; saves to `models/{model_name}.pt`.
- `evaluate_policy.py` — `TrainedPolicyAgent` exposes the network as a
  drop-in `DroneAgent` so it runs in the same simulator/replay loop.

## Agents

| Agent | File | Notes |
| --- | --- | --- |
| Heuristic | `app/agents/heuristic_agent.py` | Aims at target, sidesteps obstacles. |
| Random | same | Baseline for comparison. |
| LLM | `app/agents/llm_agent.py` | Stub: hovers unless a completion fn is wired. JSON-only protocol enforced. |
| Curriculum | `app/agents/curriculum_agent.py` | Rule-based difficulty controller. |

Curriculum rule (deterministic v1):

```
success_rate > 0.8 → difficulty += 1
success_rate < 0.4 → difficulty -= 1
collision_rate > 0.5 → fewer obstacles
timeout_rate   > 0.5 → fewer obstacles
```

## Next steps

- Wire `LLMDroneAgent` to a real provider (e.g. Anthropic via the AI SDK)
  and let it propose actions step-by-step.
- Add `LLMCurriculumAgent` that proposes harder environments from metrics.
- Move from JSON files to SQLite for episode storage.
- Replace behavior cloning with DAgger or REINFORCE.
- Train a sequence model on full trajectories instead of per-step state.

## Layout

```
backend/app/
  main.py
  models.py
  api/routes.py
  simulator/{env, physics, observations, evaluator, replay}.py
  agents/{base, heuristic_agent, llm_agent, curriculum_agent}.py
  training/{dataset, policy_model, train_policy, evaluate_policy}.py
  storage/run_store.py
frontend/src/
  App.tsx, api.ts, main.tsx, styles.css
  components/{EnvironmentConfigPanel, RunControls, ReplayCanvas,
              RunSummary, MetricsPanel, EpisodeTimeline,
              CurriculumPanel}.tsx
data/
  runs/, datasets/
models/
```
