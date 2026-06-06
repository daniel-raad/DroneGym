export type Obstacle = { x: number; y: number; radius: number };

export type EnvironmentConfig = {
  difficulty: number;
  room_width: number;
  room_height: number;
  num_obstacles: number;
  wind_strength: number;
  sensor_noise: number;
  obstacles: Obstacle[];
  start: [number, number];
  target: [number, number];
  target_radius: number;
  drone_radius: number;
  max_steps: number;
  seed?: number | null;
};

export type GenerateEnvRequest = {
  difficulty: number;
  room_width: number;
  room_height: number;
  num_obstacles: number;
  wind_strength: number;
  sensor_noise: number;
  seed?: number | null;
};

export type Observation = {
  distance_to_target: number;
  target_angle_deg: number;
  front_distance: number;
  left_distance: number;
  right_distance: number;
  battery: number;
  step: number;
};

export type TrajectoryPoint = {
  step: number;
  x: number;
  y: number;
  heading: number;
  battery: number;
};

export type EventEntry = { step: number; type: string; detail: string };

export type EpisodeSummary = {
  episode_id: string;
  success: boolean;
  collision: boolean;
  landed: boolean;
  timeout: boolean;
  steps: number;
  score: number;
  final_distance_to_target: number;
  path_length: number;
  avg_obstacle_clearance: number;
  action_counts: Record<string, number>;
  agent_type: string;
  difficulty: number;
};

export type EpisodeResponse = {
  episode_id: string;
  success: boolean;
  collision: boolean;
  landed: boolean;
  timeout: boolean;
  steps: number;
  score: number;
  environment: EnvironmentConfig;
  trajectory: TrajectoryPoint[];
  actions: string[];
  observations: Observation[];
  events: EventEntry[];
  summary: EpisodeSummary;
};

export type DatasetInfo = {
  name: string;
  path: string;
  num_samples: number;
  mtime: number;
};

export type ModelInfo = {
  name: string;
  path: string;
  mtime: number;
  method: string | null;
  train_accuracy: number | null;
  test_accuracy: number | null;
  final_loss: number | null;
  epochs: number | null;
  num_samples: number | null;
  final_success_rate: number | null;
  avg_reward_last20: number | null;
  loss_history: number[];
  test_acc_history: number[];
  reward_history: number[];
  success_history: number[];
  smoothed_reward: number[];
  smoothed_success: number[];
  sim_eval_success: number | null;
  sim_eval_n: number | null;
  num_rays?: number | null;
  hidden_size?: number | null;
  num_layers?: number | null;
  algorithm?: string | null;
  course_results?: Record<string, CourseResult> | null;
  race_ready?: boolean | null;
};

export type Course = {
  id: string;
  label: string;
  difficulty: number;
  num_obstacles: number;
  room_width: number;
  room_height: number;
};

export type CourseResult = {
  course_id: string;
  label: string;
  runs: number;
  success_rate: number;
  collision_rate: number;
  timeout_rate: number;
  avg_steps: number;
};

export type EvaluateCoursesResponse = {
  model_name: string;
  results: CourseResult[];
  race_ready: boolean;
};

export type RolloutSnapshot = {
  env: EnvironmentConfig;
  trajectory: { x: number; y: number }[];
  actions: string[];
  success: boolean;
  collision: boolean;
  steps: number;
};

export type TrainPolicyReq = {
  dataset_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  hidden_size: number;
  num_layers?: number;
  model_name: string;
  num_rays?: number;
};

export type TrainRLReq = {
  episodes: number;
  learning_rate: number;
  gamma: number;
  hidden_size: number;
  num_layers?: number;
  difficulty: number;
  room_width: number;
  room_height: number;
  num_obstacles: number;
  max_steps: number;
  model_name: string;
  warm_start_from?: string | null;
  seed: number;
  num_rays?: number;
  batch_episodes?: number;
  entropy_coef?: number;
  value_coef?: number;
  algorithm?: "a2c" | "ppo";
  ppo_clip?: number;
  ppo_epochs?: number;
  curriculum_schedule?: number[] | null;
  randomize_wind?: number;
  randomize_noise?: number;
  randomize_obstacles?: number;
};

export type DAggerReq = {
  model_name: string;
  dataset_name: string;
  rounds: number;
  episodes_per_round: number;
  difficulty: number;
  room_width: number;
  room_height: number;
  num_obstacles: number;
  max_steps: number;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  hidden_size?: number;
  num_layers?: number;
  num_rays?: number;
  seed?: number;
};

export type DAggerResponse = {
  model_path: string;
  rounds: number;
  dataset_size: number;
  final_test_accuracy: number;
  per_round_accuracy: number[];
};

export type CurrentTask = {
  name: string;
  detail: string;
  started_at: number;
  elapsed: number;
  progress: number;
  extra: {
    rollout?: RolloutSnapshot;
    smoothed_reward?: number[];
    smoothed_success?: number[];
    loss_history?: number[];
    test_acc_history?: number[];
    algorithm?: string;
    [k: string]: any;
  };
};

export type SystemStatus = {
  datasets: DatasetInfo[];
  models: ModelInfo[];
  current_task: CurrentTask | null;
  llm_available: boolean;
  llm_model: string | null;
};

export type CompareResponse = {
  heuristic: EpisodeResponse;
  trained: EpisodeResponse;
};

export type RaceResponse = {
  heuristic: EpisodeResponse | null;
  trained: EpisodeResponse | null;
  llm: EpisodeResponse | null;
};

export type RaceInclude = "heuristic" | "trained" | "llm";

export type EvalResponse = {
  model_name: string;
  num_episodes: number;
  success_rate: number;
  collision_rate: number;
  timeout_rate: number;
  avg_steps: number;
  avg_score: number;
};

export type PolicyInspectResponse = {
  model_name: string;
  actions: string[];
  probs: number[][];
  argmax: string[];
};

export type AgentType = "heuristic" | "random" | "trained" | "llm";

export type StreamStart = {
  type: "start";
  episode_id: string;
  env: EnvironmentConfig;
  initial: { trajectory: TrajectoryPoint; observation: Observation };
};
export type StreamStep = {
  type: "step";
  step: number;
  action: string;
  reason: string;
  trajectory: TrajectoryPoint;
  observation: Observation;
  events: EventEntry[];
  reward: number;
  done: boolean;
};
export type StreamEnd = { type: "end"; episode: EpisodeResponse };
export type StreamFrame = StreamStart | StreamStep | StreamEnd;

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${r.status} ${url}: ${txt}`);
  }
  return r.json();
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export const api = {
  generateEnv: (req: GenerateEnvRequest) =>
    post<EnvironmentConfig>("/api/environments/generate", req),
  runEpisode: (
    environment: EnvironmentConfig,
    agent_type: AgentType,
    max_steps: number,
    seed?: number | null,
  ) =>
    post<EpisodeResponse>("/api/episodes/run", {
      environment,
      agent_type,
      max_steps,
      seed,
    }),
  runTrained: (
    environment: EnvironmentConfig,
    model_name: string,
    max_steps: number,
    seed?: number | null,
  ) =>
    post<EpisodeResponse>("/api/episodes/run-trained-policy", {
      environment,
      model_name,
      max_steps,
      seed,
    }),
  compare: (
    environment: EnvironmentConfig,
    model_name: string,
    max_steps: number,
    seed?: number | null,
  ) =>
    post<CompareResponse>("/api/episodes/compare", {
      environment,
      model_name,
      max_steps,
      seed,
    }),
  race: (
    environment: EnvironmentConfig,
    model_name: string,
    include: RaceInclude[],
    max_steps: number,
    llm_max_steps: number,
    seed?: number | null,
  ) =>
    post<RaceResponse>("/api/episodes/race", {
      environment,
      model_name,
      include,
      max_steps,
      llm_max_steps,
      seed,
    }),
  listEpisodes: () => get<EpisodeSummary[]>("/api/episodes"),
  getEpisode: (id: string) => get<EpisodeResponse>(`/api/episodes/${id}`),
  generateDataset: (req: {
    num_episodes: number;
    difficulty?: number;
    difficulties?: number[];
    room_width?: number;
    room_height?: number;
    num_obstacles?: number;
    max_steps?: number;
    seed?: number;
    dataset_name?: string;
    append?: boolean;
    num_rays?: number;
  }) =>
    post<{
      dataset_path: string;
      num_samples: number;
      num_episodes: number;
      success_rate: number;
    }>("/api/training/generate-dataset", req),
  trainPolicy: (req: TrainPolicyReq) =>
    post<{
      model_path: string;
      train_accuracy: number;
      test_accuracy: number;
      final_loss: number;
      epochs: number;
      num_samples: number;
    }>("/api/training/train-policy", req),
  trainRL: (req: TrainRLReq) =>
    post<{
      model_path: string;
      episodes: number;
      final_success_rate: number;
      avg_reward_last20: number;
      reward_history: number[];
      success_history: number[];
    }>("/api/training/train-rl", req),
  dagger: (req: DAggerReq) => post<DAggerResponse>("/api/training/dagger", req),
  listCourses: () => get<Course[]>("/api/courses"),
  evaluateCourses: (model_name: string, runs_per_course = 20) =>
    post<EvaluateCoursesResponse>("/api/training/evaluate-courses", {
      model_name,
      runs_per_course,
    }),
  evaluateModel: (model_name: string, num_episodes: number, difficulty: number) =>
    post<EvalResponse>("/api/training/evaluate", {
      model_name,
      num_episodes,
      difficulty,
    }),
  evaluateBaseline: (
    agent_type: "heuristic" | "random",
    num_episodes: number,
    difficulty: number,
  ) =>
    post<EvalResponse>("/api/training/evaluate-baseline", {
      agent_type,
      num_episodes,
      difficulty,
    }),
  systemStatus: () => get<SystemStatus>("/api/system/status"),
  inspectPolicy: (model_name: string, observations: Observation[]) =>
    post<PolicyInspectResponse>("/api/policy/inspect", { model_name, observations }),
};

// Stream an episode step-by-step from the backend as NDJSON. The async iterator
// yields each frame as it arrives. Pass an AbortSignal to cancel mid-episode —
// the backend's request.is_disconnected() check will stop work (including any
// in-flight LLM call).
// Vite's dev proxy buffers NDJSON responses (collects the full body before
// forwarding), which defeats the point of streaming. In dev we talk to the
// backend directly on :8000 — CORS is wide open server-side. In prod the
// reverse proxy is expected to be streaming-friendly, so we use a relative URL.
// Detecting dev via the port (5173 = Vite default) avoids depending on Vite
// type augmentation for import.meta.env.
const STREAM_BASE =
  typeof window !== "undefined" && window.location.port === "5173"
    ? "http://localhost:8000"
    : "";

export async function* streamEpisode(
  environment: EnvironmentConfig,
  agent_type: AgentType,
  max_steps: number,
  seed?: number | null,
  model_name?: string | null,
  signal?: AbortSignal,
): AsyncGenerator<StreamFrame> {
  const res = await fetch(`${STREAM_BASE}/api/episodes/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ environment, agent_type, max_steps, seed, model_name }),
    signal,
  });
  if (!res.ok || !res.body) {
    const txt = await res.text().catch(() => "");
    throw new Error(`${res.status} /api/episodes/stream: ${txt}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line) as StreamFrame;
    }
  }
  const tail = buf.trim();
  if (tail) yield JSON.parse(tail) as StreamFrame;
}
