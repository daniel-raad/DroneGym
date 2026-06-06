"""Pydantic models for DroneGym."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class _BaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


ActionName = Literal[
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "hover",
    "land",
]

ACTIONS: list[str] = [
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "hover",
    "land",
]


class Obstacle(BaseModel):
    x: float
    y: float
    radius: float


class EnvironmentConfig(BaseModel):
    difficulty: int = 1
    room_width: float = 10.0
    room_height: float = 10.0
    num_obstacles: int = 3
    wind_strength: float = 0.0
    sensor_noise: float = 0.0
    obstacles: list[Obstacle] = Field(default_factory=list)
    start: tuple[float, float] = (1.0, 1.0)
    target: tuple[float, float] = (9.0, 9.0)
    target_radius: float = 0.6
    drone_radius: float = 0.25
    max_steps: int = 200
    seed: Optional[int] = None


class GenerateEnvRequest(BaseModel):
    difficulty: int = 1
    room_width: float = 10.0
    room_height: float = 10.0
    num_obstacles: int = 3
    wind_strength: float = 0.0
    sensor_noise: float = 0.0
    seed: Optional[int] = None


class Observation(BaseModel):
    distance_to_target: float
    target_angle_deg: float
    front_distance: float
    left_distance: float
    right_distance: float
    battery: float
    step: int
    # N evenly-spaced 360° rays starting at heading (ray[0] == front_distance).
    # Empty for legacy episodes; populated when the active agent requests more rays.
    rays: list[float] = Field(default_factory=list)


class DroneState(BaseModel):
    x: float
    y: float
    heading: float
    battery: float
    step: int


class TrajectoryPoint(BaseModel):
    step: int
    x: float
    y: float
    heading: float
    battery: float


class EventEntry(BaseModel):
    step: int
    type: str
    detail: str = ""


class EpisodeRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    environment: EnvironmentConfig
    agent_type: Literal["heuristic", "random", "trained", "llm"] = "heuristic"
    max_steps: int = 200
    seed: Optional[int] = None
    model_name: Optional[str] = None  # used when agent_type == "trained"


class EpisodeSummary(BaseModel):
    episode_id: str
    success: bool
    collision: bool
    landed: bool
    timeout: bool
    steps: int
    score: float
    final_distance_to_target: float
    path_length: float
    avg_obstacle_clearance: float
    action_counts: dict[str, int]
    agent_type: str
    difficulty: int


class EpisodeResponse(BaseModel):
    episode_id: str
    success: bool
    collision: bool
    landed: bool
    timeout: bool
    steps: int
    score: float
    environment: EnvironmentConfig
    trajectory: list[TrajectoryPoint]
    actions: list[str]
    observations: list[Observation]
    events: list[EventEntry]
    summary: EpisodeSummary


class GenerateDatasetRequest(BaseModel):
    num_episodes: int = 200
    difficulty: int = 1
    room_width: float = 10.0
    room_height: float = 10.0
    num_obstacles: int = 3
    max_steps: int = 200
    seed: Optional[int] = 42
    dataset_name: str = "imitation_v1"
    difficulties: Optional[list[int]] = None  # if set, sample each episode's difficulty from this list
    append: bool = False  # if True, append to existing dataset rather than overwrite
    num_rays: int = 3  # 3 = legacy front/left/right; 8/16/32 = lidar-style scan


class GenerateDatasetResponse(BaseModel):
    dataset_path: str
    num_samples: int
    num_episodes: int
    success_rate: float


class TrainPolicyRequest(_BaseModel):
    dataset_name: str = "imitation_v1"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    model_name: str = "drone_policy_v1"
    # If unset, BC trainer reads num_rays from the dataset sidecar (default 3).
    num_rays: Optional[int] = None


RLAlgorithm = Literal["a2c", "ppo"]


class TrainRLApiRequest(_BaseModel):
    episodes: int = 200
    learning_rate: float = 3e-3
    gamma: float = 0.95
    hidden_size: int = 64
    num_layers: int = 2
    difficulty: int = 1
    room_width: float = 10.0
    room_height: float = 10.0
    num_obstacles: int = 3
    max_steps: int = 200
    model_name: str = "drone_policy_rl"
    warm_start_from: str | None = None
    seed: int = 0
    # If unset, RL trainer inherits num_rays from warm-start meta (default 3).
    num_rays: Optional[int] = None
    # Optimization knobs
    batch_episodes: int = 8
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    # Algorithm: a2c (default) or ppo
    algorithm: RLAlgorithm = "a2c"
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    # Curriculum + randomization
    curriculum_schedule: Optional[list[int]] = None  # cycle through these difficulties per episode
    randomize_wind: float = 0.0  # max wind sampled uniformly per episode
    randomize_noise: float = 0.0  # max sensor noise sampled uniformly per episode
    randomize_obstacles: int = 0  # +/- N obstacles around base num_obstacles


class TrainRLApiResponse(_BaseModel):
    model_path: str
    episodes: int
    final_success_rate: float
    avg_reward_last20: float
    reward_history: list[float]
    success_history: list[int]


class DAggerRequest(_BaseModel):
    """One round of DAgger: roll out the current model in fresh envs, relabel each
    visited state with the heuristic teacher's action, append to the dataset, refit."""
    model_name: str
    dataset_name: str
    rounds: int = 3
    episodes_per_round: int = 50
    difficulty: int = 1
    room_width: float = 10.0
    room_height: float = 10.0
    num_obstacles: int = 3
    max_steps: int = 200
    # BC refit knobs
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    num_rays: Optional[int] = None
    seed: int = 0


class DAggerResponse(_BaseModel):
    model_path: str
    rounds: int
    dataset_size: int
    final_test_accuracy: float
    per_round_accuracy: list[float]


class EvaluateCoursesRequest(_BaseModel):
    model_name: str
    runs_per_course: int = 20  # uses each course's eval_seeds pool, cycles if needed


class CourseResult(_BaseModel):
    course_id: str
    label: str
    runs: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    avg_steps: float


class EvaluateCoursesResponse(_BaseModel):
    model_name: str
    results: list[CourseResult]
    # A pilot is race-ready if at least one course is >= READY_THRESHOLD.
    race_ready: bool


class DatasetInfo(BaseModel):
    name: str
    path: str
    num_samples: int
    mtime: float


class ModelInfo(_BaseModel):
    name: str
    path: str
    mtime: float
    method: str | None = None
    train_accuracy: float | None = None
    test_accuracy: float | None = None
    final_loss: float | None = None
    epochs: int | None = None
    num_samples: int | None = None
    final_success_rate: float | None = None
    avg_reward_last20: float | None = None
    loss_history: list[float] = Field(default_factory=list)
    test_acc_history: list[float] = Field(default_factory=list)
    reward_history: list[float] = Field(default_factory=list)
    success_history: list[int] = Field(default_factory=list)
    smoothed_reward: list[float] = Field(default_factory=list)
    smoothed_success: list[float] = Field(default_factory=list)
    sim_eval_success: float | None = None
    sim_eval_n: int | None = None
    num_rays: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    algorithm: str | None = None
    # Per-course readiness (set by /api/training/evaluate-courses)
    course_results: dict | None = None  # {course_id: {success_rate, runs, ...}}
    race_ready: bool | None = None


class SystemStatusResponse(BaseModel):
    datasets: list[DatasetInfo]
    models: list[ModelInfo]
    current_task: dict | None = None
    llm_available: bool = False
    llm_model: str | None = None


class CompareRequest(_BaseModel):
    environment: EnvironmentConfig
    model_name: str = "drone_policy_v1"
    max_steps: int = 250
    seed: Optional[int] = None


class CompareResponse(BaseModel):
    heuristic: "EpisodeResponse"
    trained: "EpisodeResponse"


class ThreeWayRequest(_BaseModel):
    environment: EnvironmentConfig
    model_name: str = "drone_policy_v1"
    max_steps: int = 250
    llm_max_steps: int = 60  # LLM is per-step API calls — cap conservatively
    seed: Optional[int] = None
    include: list[Literal["heuristic", "trained", "llm"]] = Field(
        default_factory=lambda: ["heuristic", "trained", "llm"]
    )


class ThreeWayResponse(BaseModel):
    heuristic: Optional["EpisodeResponse"] = None
    trained: Optional["EpisodeResponse"] = None
    llm: Optional["EpisodeResponse"] = None


class EvaluateModelRequest(_BaseModel):
    model_name: str
    num_episodes: int = 20
    difficulty: int = 1
    seed: int = 1000


class EvaluateModelResponse(BaseModel):
    model_name: str
    num_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    avg_steps: float
    avg_score: float


class EvaluateBaselineRequest(_BaseModel):
    agent_type: Literal["heuristic", "random"] = "heuristic"
    num_episodes: int = 20
    difficulty: int = 1
    seed: int = 2000


class PolicyInspectRequest(_BaseModel):
    model_name: str = "drone_policy_v1"
    observations: list[Observation]


class PolicyInspectResponse(_BaseModel):
    model_name: str
    actions: list[str]  # the action list (in order) the probs columns refer to
    probs: list[list[float]]  # [n_steps][n_actions]
    argmax: list[str]  # per step


class TrainPolicyResponse(_BaseModel):
    model_path: str
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    epochs: int
    num_samples: int
