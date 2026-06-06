"""FastAPI routes."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from ..agents.heuristic_agent import HeuristicAgent, RandomAgent
from ..agents.llm_agent import LLMDroneAgent
from ..models import (
    CompareRequest,
    CompareResponse,
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
from ..runtime_state import STATE
from ..simulator.env import generate_environment
from ..simulator.evaluator import run_episode
from ..storage import run_store
from ..training.dataset import generate_imitation_samples
from ..training.evaluate_policy import TrainedPolicyAgent
from ..training.train_policy import load_model_meta, load_policy, train_policy
from ..training.train_rl import TrainRLRequest, train_rl


router = APIRouter(prefix="/api")


@router.post("/environments/generate", response_model=EnvironmentConfig)
def generate_env(req: GenerateEnvRequest) -> EnvironmentConfig:
    return generate_environment(req)


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
        return TrainedPolicyAgent(model)
    raise HTTPException(400, f"Unknown agent_type: {agent_type}")


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
            difficulty=req.difficulty,
            room_width=req.room_width,
            room_height=req.room_height,
            num_obstacles=req.num_obstacles,
            max_steps=req.max_steps,
            model_name=req.model_name,
            warm_start_from=req.warm_start_from,
            seed=req.seed,
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
