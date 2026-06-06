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
};

export type CurrentTask = {
  name: string;
  detail: string;
  started_at: number;
  elapsed: number;
  progress: number;
  extra: Record<string, any>;
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

export type AgentType = "heuristic" | "random" | "trained" | "llm";

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
  }) =>
    post<{
      dataset_path: string;
      num_samples: number;
      num_episodes: number;
      success_rate: number;
    }>("/api/training/generate-dataset", req),
  trainPolicy: (req: any) =>
    post<{
      model_path: string;
      train_accuracy: number;
      test_accuracy: number;
      final_loss: number;
      epochs: number;
      num_samples: number;
    }>("/api/training/train-policy", req),
  trainRL: (req: any) =>
    post<{
      model_path: string;
      episodes: number;
      final_success_rate: number;
      avg_reward_last20: number;
      reward_history: number[];
      success_history: number[];
    }>("/api/training/train-rl", req),
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
};
