// Pure rules that read training history + course-eval results and emit
// actionable cards. New rules go here — UI is dumb, this module is the brain.

import type { CourseResult, ModelInfo } from "../api";

export type DiagnosticSeverity = "info" | "warn" | "error";

export type Diagnostic = {
  id: string;
  severity: DiagnosticSeverity;
  title: string;
  detail: string;
  fix: string;
};

const RACE_READY = 0.5;

export function diagnose(
  model: ModelInfo | null,
  courseResults: CourseResult[] | null,
): Diagnostic[] {
  if (!model) return [];
  const out: Diagnostic[] = [];

  // BC: high demo accuracy, low sim success → covariate shift
  if (
    model.method === "behavior_cloning" &&
    model.test_accuracy != null &&
    model.sim_eval_success != null &&
    model.test_accuracy - model.sim_eval_success > 0.25
  ) {
    out.push({
      id: "bc-covariate-shift",
      severity: "warn",
      title: "Behavior cloning fits demos but fails the sim",
      detail: `Demo accuracy ${(model.test_accuracy * 100).toFixed(0)}% but only ${(model.sim_eval_success * 100).toFixed(0)}% success in fresh rooms. Classic BC covariate shift — the pilot has never seen the states it visits when it makes mistakes.`,
      fix: "Add a DAgger stage (teacher relabels states the student visits) or follow up with an RL practice stage.",
    });
  }

  // RL flatline: smoothed success barely moves in the last quarter
  const ss = model.smoothed_success || [];
  if (ss.length >= 80) {
    const tail = ss.slice(-Math.floor(ss.length / 4));
    const head = ss.slice(-Math.floor(ss.length / 2), -Math.floor(ss.length / 4));
    const tailAvg = tail.reduce((a, b) => a + b, 0) / tail.length;
    const headAvg = head.reduce((a, b) => a + b, 0) / head.length;
    if (tailAvg < 0.5 && Math.abs(tailAvg - headAvg) < 0.03) {
      out.push({
        id: "rl-plateau",
        severity: "warn",
        title: "RL training plateaued",
        detail: `Smoothed success stuck around ${(tailAvg * 100).toFixed(0)}% for the last ~${tail.length} episodes.`,
        fix: "Raise the entropy coefficient (0.05+), enable a curriculum schedule (D1→D3), or switch to PPO.",
      });
    }
  }

  // Course readiness
  if (courseResults && courseResults.length) {
    const passed = courseResults.filter((c) => c.success_rate >= RACE_READY);
    const failed = courseResults.filter((c) => c.success_rate < RACE_READY);

    if (passed.length === 0) {
      out.push({
        id: "no-course-ready",
        severity: "error",
        title: "Not yet race-ready",
        detail: `Pilot is below ${RACE_READY * 100}% on every course (${courseResults
          .map((c) => `${c.label} ${(c.success_rate * 100).toFixed(0)}%`)
          .join(" · ")}).`,
        fix: "If you've only run BC, add an RL practice stage — that's the only thing that pushes past the teacher. If you've already practiced, add a curriculum stage covering D1–D3 and randomize wind/noise.",
      });
    } else if (failed.length) {
      const worst = failed.reduce((a, b) =>
        a.success_rate < b.success_rate ? a : b,
      );
      out.push({
        id: "course-gap",
        severity: "warn",
        title: `Strong on ${passed.map((c) => c.label).join("/")} — weak on ${failed.map((c) => c.label).join("/")}`,
        detail: `${worst.label} success ${(worst.success_rate * 100).toFixed(0)}%. The pilot overfit the easy courses.`,
        fix: `Add an RL stage with curriculum_schedule covering the harder difficulty (e.g. [1,2,3]). Also enable randomize_obstacles ≥ 1.`,
      });
    }
  }

  return out;
}
