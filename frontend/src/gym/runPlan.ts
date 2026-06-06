import { api } from "../api";
import type { CourseResult } from "../api";
import type { Stage } from "./plan";

export type StageRunEvent =
  | { type: "stage-start"; stageId: string }
  | { type: "stage-end"; stageId: string; summary: string }
  | { type: "stage-error"; stageId: string; message: string }
  | { type: "courses"; stageId: string; results: CourseResult[]; raceReady: boolean }
  | { type: "plan-end" };

export type PlanCtx = {
  modelName: string;
  datasetName: string;
  // Defaults for env params used by stages that don't override
  base: {
    room_width: number;
    room_height: number;
    num_obstacles: number;
    max_steps: number;
  };
};

/** Run the plan, awaiting each stage. Yields events the UI uses to render
 * stage status and to invalidate cached model state after each stage. */
export async function* runPlan(
  stages: Stage[],
  ctx: PlanCtx,
  signal?: AbortSignal,
): AsyncGenerator<StageRunEvent> {
  let lastNumRays = 16;

  for (const stage of stages) {
    if (signal?.aborted) return;
    yield { type: "stage-start", stageId: stage.id };
    try {
      if (stage.kind === "collect") {
        const r = await api.generateDataset({
          num_episodes: stage.num_episodes,
          difficulties: stage.difficulties,
          difficulty: stage.difficulties[0],
          room_width: ctx.base.room_width,
          room_height: ctx.base.room_height,
          num_obstacles: ctx.base.num_obstacles,
          max_steps: ctx.base.max_steps,
          seed: 42,
          dataset_name: ctx.datasetName,
          append: false,
          num_rays: stage.num_rays,
        });
        lastNumRays = stage.num_rays;
        yield {
          type: "stage-end",
          stageId: stage.id,
          summary: `${r.num_samples.toLocaleString()} samples · teacher ${(r.success_rate * 100).toFixed(0)}%`,
        };
      } else if (stage.kind === "bc") {
        const r = await api.trainPolicy({
          dataset_name: ctx.datasetName,
          epochs: stage.epochs,
          batch_size: stage.batch_size,
          learning_rate: stage.learning_rate,
          hidden_size: stage.hidden_size,
          num_layers: stage.num_layers,
          model_name: ctx.modelName,
          num_rays: lastNumRays,
        });
        yield {
          type: "stage-end",
          stageId: stage.id,
          summary: `demo accuracy ${(r.test_accuracy * 100).toFixed(1)}%`,
        };
      } else if (stage.kind === "dagger") {
        const r = await api.dagger({
          model_name: ctx.modelName,
          dataset_name: ctx.datasetName,
          rounds: stage.rounds,
          episodes_per_round: stage.episodes_per_round,
          difficulty: 1,
          room_width: ctx.base.room_width,
          room_height: ctx.base.room_height,
          num_obstacles: ctx.base.num_obstacles,
          max_steps: ctx.base.max_steps,
          epochs: stage.epochs,
          num_rays: lastNumRays,
        });
        yield {
          type: "stage-end",
          stageId: stage.id,
          summary: `${stage.rounds} rounds · final demo acc ${(r.final_test_accuracy * 100).toFixed(1)}%`,
        };
      } else if (stage.kind === "rl") {
        const r = await api.trainRL({
          episodes: stage.episodes,
          learning_rate: stage.learning_rate,
          gamma: stage.gamma,
          hidden_size: stage.hidden_size,
          num_layers: stage.num_layers,
          difficulty: stage.curriculum_schedule[0] ?? 1,
          room_width: ctx.base.room_width,
          room_height: ctx.base.room_height,
          num_obstacles: ctx.base.num_obstacles,
          max_steps: ctx.base.max_steps,
          model_name: ctx.modelName,
          warm_start_from: stage.warm_start ? ctx.modelName : null,
          seed: 7,
          num_rays: lastNumRays,
          batch_episodes: stage.batch_episodes,
          entropy_coef: stage.entropy_coef,
          algorithm: stage.algorithm,
          ppo_clip: stage.ppo_clip,
          ppo_epochs: stage.ppo_epochs,
          curriculum_schedule: stage.curriculum_schedule,
          randomize_wind: stage.randomize_wind,
          randomize_noise: stage.randomize_noise,
          randomize_obstacles: stage.randomize_obstacles,
        });
        yield {
          type: "stage-end",
          stageId: stage.id,
          summary: `${stage.algorithm.toUpperCase()} · final succ ${(r.final_success_rate * 100).toFixed(0)}%`,
        };
      } else if (stage.kind === "eval") {
        const r = await api.evaluateCourses(ctx.modelName, stage.runs_per_course);
        yield {
          type: "courses",
          stageId: stage.id,
          results: r.results,
          raceReady: r.race_ready,
        };
        yield {
          type: "stage-end",
          stageId: stage.id,
          summary: r.results
            .map((c) => `${c.label}: ${(c.success_rate * 100).toFixed(0)}%`)
            .join(" · "),
        };
      }
    } catch (e: any) {
      yield {
        type: "stage-error",
        stageId: stage.id,
        message: e?.message || "stage failed",
      };
      return;
    }
  }
  yield { type: "plan-end" };
}
