"""FastAPI routes."""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from ..agents.heuristic_agent import HeuristicAgent, RandomAgent
from ..agents.llm_agent import LLMDroneAgent
from ..models import (
    CompareRequest,
    CompareResponse,
    CourseResult,
    DAggerRequest,
    DAggerResponse,
    EvaluateCoursesRequest,
    EvaluateCoursesResponse,
    ThreeWayRequest,
    ThreeWayResponse,
    DatasetInfo,
    EnvironmentConfig,
    EpisodeResponse,
    EpisodeRunRequest,
    EpisodeSummary,
    EvaluateBaselineRequest,
    EvaluateModelRequest,
    EvaluateModelResponse,
    GenerateDatasetRequest,
    GenerateDatasetResponse,
    GenerateEnvRequest,
    ModelInfo,
    SystemStatusResponse,
    TrainPolicyRequest,
    TrainPolicyResponse,
    TrainRLApiRequest,
    TrainRLApiResponse,
)
from ..simulator.courses import all_courses, COURSES, get_course
from ..runtime_state import STATE
from ..simulator import physics
from ..simulator.env import Simulator, generate_environment
from ..simulator.evaluator import run_episode
from ..storage import run_store
from ..training.dataset import generate_imitation_samples, write_dataset_meta
from ..training.evaluate_policy import TrainedPolicyAgent
from ..training.train_policy import load_model_meta, load_policy, train_policy
from ..training.train_rl import TrainRLRequest, train_rl
from ..training.dagger import run_dagger
from ..training.policy_model import ActorCritic
from ..models import (
    ACTIONS,
    PolicyInspectRequest,
    PolicyInspectResponse,
)


router = APIRouter(prefix="/api")


@router.post("/environments/generate", response_model=EnvironmentConfig)
def generate_env(req: GenerateEnvRequest) -> EnvironmentConfig:
    return generate_environment(req)


def _trained_agent_num_rays(model_name: str) -> int:
    """Pull num_rays from a trained checkpoint's .pt payload (sidecar fallback)."""
    import torch as _t

    path = run_store.models_dir() / f"{model_name}.pt"
    if path.exists():
        try:
            payload = _t.load(path, map_location="cpu", weights_only=False)
            v = payload.get("num_rays")
            if v is not None:
                return int(v)
        except Exception:
            pass
    meta = load_model_meta(model_name) or {}
    return int(meta.get("num_rays") or 3)


def _make_agent(agent_type: str, model_name: str | None = None):
    if agent_type == "heuristic":
        return HeuristicAgent()
    if agent_type == "random":
        return RandomAgent()
    if agent_type == "llm":
        agent = LLMDroneAgent()
        if not agent.available:
            raise HTTPException(
                400,
                "LLM agent is unavailable: set ANTHROPIC_API_KEY in the backend environment.",
            )
        return agent
    if agent_type == "trained":
        name = model_name or "drone_policy_v1"
        model = load_policy(name)
        nr = _trained_agent_num_rays(name)
        return TrainedPolicyAgent(model, num_rays=nr)
    raise HTTPException(400, f"Unknown agent_type: {agent_type}")


@router.post("/episodes/stream")
async def stream_one(req: EpisodeRunRequest, request: Request) -> StreamingResponse:
    """Stream an episode step-by-step as NDJSON.

    Frames:
      {"type": "start", "env": ..., "episode_id": ..., "initial": {...}}
      {"type": "step", "step": N, "action": ..., "trajectory": {...}, "observation": {...}, "events": [...], "reason": "..."}
      {"type": "end", "episode": EpisodeResponse}
    """
    try:
        agent = _make_agent(req.agent_type, req.model_name)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    eid = f"run_{uuid.uuid4().hex[:8]}"

    # Minimum wall-clock per streamed step. Heuristic/trained policies decide
    # in microseconds, so without pacing the whole episode would arrive in one
    # chunk and the canvas would jump straight to the final frame. Oracle's
    # per-step LLM call already takes ~1s so this sleep is dwarfed and adds no
    # delay there.
    TARGET_STEP_SEC = 0.05

    async def gen():
        STATE.start("run_episode", detail=f"stream:{req.agent_type}")
        try:
            num_rays = int(getattr(agent, "num_rays", 3))
            sim = Simulator(req.environment, seed=req.seed, num_rays=num_rays)
            if req.max_steps and req.max_steps != req.environment.max_steps:
                sim.env = req.environment.model_copy(update={"max_steps": req.max_steps})

            trajectory = [sim.current_trajectory_point()]
            observations = [sim.initial_observation()]
            actions: list[str] = []
            events: list = []
            total_reward = 0.0
            clearance_sum = 0.0
            clearance_count = 0

            yield json.dumps({
                "type": "start",
                "episode_id": eid,
                "env": req.environment.model_dump(),
                "initial": {
                    "trajectory": trajectory[0].model_dump(),
                    "observation": observations[0].model_dump(),
                },
            }) + "\n"

            obs = observations[0]
            tick = time.monotonic()
            while not sim.done:
                if await request.is_disconnected():
                    return
                # LLM agent makes a blocking HTTP call — push it off the event loop.
                action = await asyncio.to_thread(agent.act, obs)
                if action not in ACTIONS:
                    action = "hover"
                actions.append(action)
                reason = getattr(agent, "last_reason", "") or ""

                obs, reward, _done, evs = sim.step(action)
                observations.append(obs)
                events.extend(evs)
                total_reward += reward
                traj = sim.current_trajectory_point()
                trajectory.append(traj)
                clearance_sum += physics.min_obstacle_clearance(
                    sim.x, sim.y, sim.env.obstacles, sim.env.room_width, sim.env.room_height
                )
                clearance_count += 1

                yield json.dumps({
                    "type": "step",
                    "step": sim.step_idx,
                    "action": action,
                    "reason": reason,
                    "trajectory": traj.model_dump(),
                    "observation": obs.model_dump(),
                    "events": [e.model_dump() for e in evs],
                    "reward": round(reward, 3),
                    "done": sim.done,
                }) + "\n"

                # Pace the stream. Sleep the remainder of the per-step budget.
                elapsed = time.monotonic() - tick
                if elapsed < TARGET_STEP_SEC:
                    await asyncio.sleep(TARGET_STEP_SEC - elapsed)
                tick = time.monotonic()

            path_length = 0.0
            for i in range(1, len(trajectory)):
                path_length += math.hypot(
                    trajectory[i].x - trajectory[i - 1].x,
                    trajectory[i].y - trajectory[i - 1].y,
                )
            final_dist = math.hypot(
                sim.env.target[0] - trajectory[-1].x,
                sim.env.target[1] - trajectory[-1].y,
            )
            action_counts = {a: 0 for a in ACTIONS}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            from ..models import EpisodeSummary as _S
            summary = _S(
                episode_id=eid,
                success=sim.success,
                collision=sim.collided,
                landed=sim.landed,
                timeout=sim.timeout,
                steps=sim.step_idx,
                score=round(total_reward, 2),
                final_distance_to_target=round(final_dist, 3),
                path_length=round(path_length, 3),
                avg_obstacle_clearance=round(clearance_sum / max(clearance_count, 1), 3),
                action_counts=action_counts,
                agent_type=req.agent_type,
                difficulty=sim.env.difficulty,
            )
            episode = EpisodeResponse(
                episode_id=eid,
                success=sim.success,
                collision=sim.collided,
                landed=sim.landed,
                timeout=sim.timeout,
                steps=sim.step_idx,
                score=round(total_reward, 2),
                environment=sim.env,
                trajectory=trajectory,
                actions=actions,
                observations=observations,
                events=events,
                summary=summary,
            )
            run_store.save_episode(episode)
            yield json.dumps({"type": "end", "episode": episode.model_dump()}) + "\n"
        finally:
            STATE.finish()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.post("/episodes/run", response_model=EpisodeResponse)
def run_one(req: EpisodeRunRequest) -> EpisodeResponse:
    STATE.start("run_episode", detail=f"{req.agent_type} on D{req.environment.difficulty}")
    try:
        agent = _make_agent(req.agent_type)
        ep = run_episode(
            env=req.environment,
            agent=agent,
            max_steps=req.max_steps,
            seed=req.seed,
            agent_label=req.agent_type,
        )
        run_store.save_episode(ep)
        return ep
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    finally:
        STATE.finish()


class RunTrainedRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    environment: EnvironmentConfig
    model_name: str = "drone_policy_v1"
    max_steps: int = 200
    seed: int | None = None


@router.post("/episodes/run-trained-policy", response_model=EpisodeResponse)
def run_trained(req: RunTrainedRequest) -> EpisodeResponse:
    STATE.start("run_episode", detail=f"trained:{req.model_name}")
    try:
        agent = _make_agent("trained", req.model_name)
        ep = run_episode(
            env=req.environment,
            agent=agent,
            max_steps=req.max_steps,
            seed=req.seed,
            agent_label=f"trained:{req.model_name}",
        )
        run_store.save_episode(ep)
        return ep
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    finally:
        STATE.finish()


@router.post("/episodes/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    STATE.start("compare", detail=f"heuristic vs trained:{req.model_name}")
    try:
        try:
            trained_agent = _make_agent("trained", req.model_name)
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        h_ep = run_episode(
            env=req.environment,
            agent=HeuristicAgent(),
            max_steps=req.max_steps,
            seed=req.seed,
            agent_label="heuristic",
        )
        t_ep = run_episode(
            env=req.environment,
            agent=trained_agent,
            max_steps=req.max_steps,
            seed=req.seed,
            agent_label=f"trained:{req.model_name}",
        )
        run_store.save_episode(h_ep)
        run_store.save_episode(t_ep)
        return CompareResponse(heuristic=h_ep, trained=t_ep)
    finally:
        STATE.finish()


@router.post("/episodes/race", response_model=ThreeWayResponse)
def race(req: ThreeWayRequest) -> ThreeWayResponse:
    """Run any subset of {heuristic, trained, llm} on a single shared world."""
    STATE.start("race", detail=f"{', '.join(req.include)}")
    out = ThreeWayResponse()
    try:
        if "heuristic" in req.include:
            ep = run_episode(
                env=req.environment,
                agent=HeuristicAgent(),
                max_steps=req.max_steps,
                seed=req.seed,
                agent_label="heuristic",
            )
            run_store.save_episode(ep)
            out.heuristic = ep
        if "trained" in req.include:
            try:
                agent = _make_agent("trained", req.model_name)
                ep = run_episode(
                    env=req.environment,
                    agent=agent,
                    max_steps=req.max_steps,
                    seed=req.seed,
                    agent_label=f"trained:{req.model_name}",
                )
                run_store.save_episode(ep)
                out.trained = ep
            except (FileNotFoundError, HTTPException):
                # If model missing, skip — caller already sees absence in response.
                pass
        if "llm" in req.include:
            try:
                agent = _make_agent("llm")
                ep = run_episode(
                    env=req.environment,
                    agent=agent,
                    max_steps=req.llm_max_steps,
                    seed=req.seed,
                    agent_label="llm",
                )
                run_store.save_episode(ep)
                out.llm = ep
            except HTTPException as e:
                # LLM unavailable — return what we have, frontend will show a note.
                if e.status_code != 400:
                    raise
        return out
    finally:
        STATE.finish()


@router.get("/episodes", response_model=list[EpisodeSummary])
def list_episodes() -> list[EpisodeSummary]:
    return run_store.list_episode_summaries()


@router.get("/episodes/{episode_id}", response_model=EpisodeResponse)
def get_episode(episode_id: str) -> EpisodeResponse:
    ep = run_store.load_episode(episode_id)
    if ep is None:
        raise HTTPException(404, f"Episode '{episode_id}' not found")
    return ep


@router.post("/training/generate-dataset", response_model=GenerateDatasetResponse)
def generate_dataset(req: GenerateDatasetRequest) -> GenerateDatasetResponse:
    STATE.start(
        "generate_dataset",
        detail=f"{req.num_episodes} episodes · D{req.difficulty}",
    )
    try:
        samples, stats = generate_imitation_samples(req)
        if not samples:
            raise HTTPException(400, "No successful episodes — cannot build imitation dataset")
        path, n = run_store.write_dataset(req.dataset_name, samples, append=req.append)
        write_dataset_meta(
            req.dataset_name,
            {
                "num_rays": stats["num_rays"],
                "num_samples": n,
                "difficulty": req.difficulty,
            },
        )
        return GenerateDatasetResponse(
            dataset_path=str(path),
            num_samples=n,
            num_episodes=stats["num_episodes"],
            success_rate=stats["success_rate"],
        )
    finally:
        STATE.finish()


@router.post("/training/train-policy", response_model=TrainPolicyResponse)
def train(req: TrainPolicyRequest) -> TrainPolicyResponse:
    try:
        return train_policy(req)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/training/train-rl", response_model=TrainRLApiResponse)
def train_reinforce(req: TrainRLApiRequest) -> TrainRLApiResponse:
    r = train_rl(
        TrainRLRequest(
            episodes=req.episodes,
            learning_rate=req.learning_rate,
            gamma=req.gamma,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            difficulty=req.difficulty,
            room_width=req.room_width,
            room_height=req.room_height,
            num_obstacles=req.num_obstacles,
            max_steps=req.max_steps,
            model_name=req.model_name,
            warm_start_from=req.warm_start_from,
            seed=req.seed,
            num_rays=req.num_rays,
            batch_episodes=req.batch_episodes,
            entropy_coef=req.entropy_coef,
            value_coef=req.value_coef,
            algorithm=req.algorithm,
            ppo_clip=req.ppo_clip,
            ppo_epochs=req.ppo_epochs,
            curriculum_schedule=req.curriculum_schedule,
            randomize_wind=req.randomize_wind,
            randomize_noise=req.randomize_noise,
            randomize_obstacles=req.randomize_obstacles,
        )
    )
    return TrainRLApiResponse(
        model_path=r.model_path,
        episodes=r.episodes,
        final_success_rate=r.final_success_rate,
        avg_reward_last20=r.avg_reward_last20,
        reward_history=r.reward_history,
        success_history=r.success_history,
    )


@router.post("/training/dagger", response_model=DAggerResponse)
def dagger(req: DAggerRequest) -> DAggerResponse:
    try:
        return run_dagger(req)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))


# A pilot earns "race-ready" if it clears any one course at this rate.
RACE_READY_THRESHOLD = 0.5


@router.get("/courses", response_model=list[dict])
def list_courses() -> list[dict]:
    return [
        {
            "id": c.id,
            "label": c.label,
            "difficulty": c.difficulty,
            "num_obstacles": c.num_obstacles,
            "room_width": c.room_width,
            "room_height": c.room_height,
        }
        for c in all_courses()
    ]


@router.post("/training/evaluate-courses", response_model=EvaluateCoursesResponse)
def evaluate_courses(req: EvaluateCoursesRequest) -> EvaluateCoursesResponse:
    """Evaluate a trained model against every arcade course in turn.

    Uses each course's fixed eval_seeds so two pilots are graded on the same rooms.
    Persists per-course results in the model's meta sidecar.
    """
    STATE.start("evaluate_courses", detail=f"{req.model_name} · {req.runs_per_course}/course")
    try:
        try:
            agent_factory = lambda: _make_agent("trained", req.model_name)  # noqa: E731
            agent_factory()
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))

        results: list[CourseResult] = []
        for course in all_courses():
            STATE.update(detail=f"{req.model_name} on {course.label}")
            n_s = n_c = n_t = 0
            steps_sum = 0
            for i in range(req.runs_per_course):
                seed = course.eval_seeds[i % len(course.eval_seeds)] + (i // len(course.eval_seeds)) * 7919
                env = generate_environment(course.env_request(seed=seed))
                ep = run_episode(
                    env=env,
                    agent=agent_factory(),
                    max_steps=env.max_steps,
                    seed=seed,
                    agent_label=f"trained:{req.model_name}",
                )
                if ep.success:
                    n_s += 1
                if ep.collision:
                    n_c += 1
                if ep.timeout:
                    n_t += 1
                steps_sum += ep.steps
            n = req.runs_per_course
            results.append(
                CourseResult(
                    course_id=course.id,
                    label=course.label,
                    runs=n,
                    success_rate=round(n_s / n, 4),
                    collision_rate=round(n_c / n, 4),
                    timeout_rate=round(n_t / n, 4),
                    avg_steps=round(steps_sum / n, 2),
                )
            )

        race_ready = any(r.success_rate >= RACE_READY_THRESHOLD for r in results)

        # Persist into model meta so the arcade can read readiness without re-evaluating.
        meta_path = run_store.models_dir() / f"{req.model_name}.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        else:
            meta = {"model_name": req.model_name, "trained_at": time.time()}
        meta["course_results"] = {r.course_id: r.model_dump() for r in results}
        meta["race_ready"] = race_ready
        meta["course_eval_at"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2))

        return EvaluateCoursesResponse(
            model_name=req.model_name,
            results=results,
            race_ready=race_ready,
        )
    finally:
        STATE.finish()


@router.post("/training/evaluate", response_model=EvaluateModelResponse)
def evaluate_model(req: EvaluateModelRequest) -> EvaluateModelResponse:
    STATE.start("evaluate_model", detail=f"{req.model_name} · {req.num_episodes} episodes")
    try:
        try:
            agent_factory = lambda: _make_agent("trained", req.model_name)  # noqa: E731
            agent_factory()
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))

        import random as _r

        rng = _r.Random(req.seed)
        n_s = n_c = n_t = 0
        steps_sum = 0
        score_sum = 0.0
        for i in range(req.num_episodes):
            env = generate_environment(
                GenerateEnvRequest(difficulty=req.difficulty, seed=rng.randint(0, 10_000_000))
            )
            ep = run_episode(
                env=env,
                agent=agent_factory(),
                max_steps=env.max_steps,
                seed=rng.randint(0, 10_000_000),
                agent_label=f"trained:{req.model_name}",
            )
            if ep.success:
                n_s += 1
            if ep.collision:
                n_c += 1
            if ep.timeout:
                n_t += 1
            steps_sum += ep.steps
            score_sum += ep.score

        # Persist eval stats next to model meta
        meta_path = run_store.models_dir() / f"{req.model_name}.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        else:
            meta = {"model_name": req.model_name, "trained_at": time.time()}
        meta["sim_eval_success"] = round(n_s / req.num_episodes, 4)
        meta["sim_eval_n"] = req.num_episodes
        meta["sim_eval_difficulty"] = req.difficulty
        meta_path.write_text(json.dumps(meta, indent=2))

        return EvaluateModelResponse(
            model_name=req.model_name,
            num_episodes=req.num_episodes,
            success_rate=round(n_s / req.num_episodes, 4),
            collision_rate=round(n_c / req.num_episodes, 4),
            timeout_rate=round(n_t / req.num_episodes, 4),
            avg_steps=round(steps_sum / req.num_episodes, 2),
            avg_score=round(score_sum / req.num_episodes, 2),
        )
    finally:
        STATE.finish()


@router.post("/policy/inspect", response_model=PolicyInspectResponse)
def policy_inspect(req: PolicyInspectRequest) -> PolicyInspectResponse:
    """Return per-step softmax over actions for a saved policy on the given observations."""
    import torch

    try:
        model = load_policy(req.model_name)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))

    nr = _trained_agent_num_rays(req.model_name)

    def _feats(o):
        if nr == 3:
            if o.rays and len(o.rays) >= 3:
                return [o.distance_to_target, o.target_angle_deg, o.rays[0], o.rays[1], o.rays[2], o.battery]
            return [
                o.distance_to_target,
                o.target_angle_deg,
                o.front_distance,
                o.left_distance,
                o.right_distance,
                o.battery,
            ]
        return [o.distance_to_target, o.target_angle_deg, *o.rays, o.battery]

    feats = [_feats(o) for o in req.observations]
    if not feats:
        return PolicyInspectResponse(model_name=req.model_name, actions=ACTIONS, probs=[], argmax=[])
    x = torch.tensor(feats, dtype=torch.float32)
    with torch.no_grad():
        if isinstance(model, ActorCritic):
            logits = model.policy_logits(x)
        else:
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        am = torch.argmax(probs, dim=1)
    probs_list = [[round(float(v), 4) for v in row] for row in probs.tolist()]
    argmax_list = [ACTIONS[int(i)] for i in am.tolist()]
    return PolicyInspectResponse(
        model_name=req.model_name,
        actions=list(ACTIONS),
        probs=probs_list,
        argmax=argmax_list,
    )


@router.post("/training/evaluate-baseline", response_model=EvaluateModelResponse)
def evaluate_baseline(req: EvaluateBaselineRequest) -> EvaluateModelResponse:
    """Evaluate the hand-coded heuristic (or random) pilot across N fresh worlds."""
    STATE.start("evaluate_baseline", detail=f"{req.agent_type} · {req.num_episodes} eps")
    try:
        import random as _r

        rng = _r.Random(req.seed)
        n_s = n_c = n_t = 0
        steps_sum = 0
        score_sum = 0.0
        for _ in range(req.num_episodes):
            env = generate_environment(
                GenerateEnvRequest(difficulty=req.difficulty, seed=rng.randint(0, 10_000_000))
            )
            agent = _make_agent(req.agent_type)
            ep = run_episode(
                env=env,
                agent=agent,
                max_steps=env.max_steps,
                seed=rng.randint(0, 10_000_000),
                agent_label=req.agent_type,
            )
            if ep.success:
                n_s += 1
            if ep.collision:
                n_c += 1
            if ep.timeout:
                n_t += 1
            steps_sum += ep.steps
            score_sum += ep.score
        return EvaluateModelResponse(
            model_name=req.agent_type,
            num_episodes=req.num_episodes,
            success_rate=round(n_s / req.num_episodes, 4),
            collision_rate=round(n_c / req.num_episodes, 4),
            timeout_rate=round(n_t / req.num_episodes, 4),
            avg_steps=round(steps_sum / req.num_episodes, 2),
            avg_score=round(score_sum / req.num_episodes, 2),
        )
    finally:
        STATE.finish()


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    datasets: list[DatasetInfo] = []
    if run_store.DATASETS_DIR.exists():
        for p in sorted(run_store.DATASETS_DIR.glob("*.jsonl")):
            try:
                with p.open() as f:
                    n = sum(1 for _ in f)
                datasets.append(
                    DatasetInfo(
                        name=p.stem,
                        path=str(p),
                        num_samples=n,
                        mtime=p.stat().st_mtime,
                    )
                )
            except Exception:
                continue

    models: list[ModelInfo] = []
    md = run_store.models_dir()
    if md.exists():
        for p in sorted(md.glob("*.pt")):
            meta = load_model_meta(p.stem) or {}
            models.append(
                ModelInfo(
                    name=p.stem,
                    path=str(p),
                    mtime=p.stat().st_mtime,
                    method=meta.get("method"),
                    train_accuracy=meta.get("train_accuracy"),
                    test_accuracy=meta.get("test_accuracy"),
                    final_loss=meta.get("final_loss"),
                    epochs=meta.get("epochs"),
                    num_samples=meta.get("num_samples"),
                    final_success_rate=meta.get("final_success_rate"),
                    avg_reward_last20=meta.get("avg_reward_last20"),
                    loss_history=meta.get("loss_history", []),
                    test_acc_history=meta.get("test_acc_history", []),
                    reward_history=meta.get("reward_history", []),
                    success_history=meta.get("success_history", []),
                    smoothed_reward=meta.get("smoothed_reward", []),
                    smoothed_success=meta.get("smoothed_success", []),
                    sim_eval_success=meta.get("sim_eval_success"),
                    sim_eval_n=meta.get("sim_eval_n"),
                    num_rays=meta.get("num_rays"),
                    hidden_size=meta.get("hidden_size"),
                    num_layers=meta.get("num_layers"),
                    algorithm=meta.get("algorithm") or meta.get("method"),
                    course_results=meta.get("course_results"),
                    race_ready=meta.get("race_ready"),
                )
            )
    llm_probe = LLMDroneAgent()
    return SystemStatusResponse(
        datasets=datasets,
        models=models,
        current_task=STATE.snapshot(),
        llm_available=llm_probe.available,
        llm_model=llm_probe.model if llm_probe.available else None,
    )
