import { useMemo } from "react";
import type { CurrentTask, RolloutSnapshot } from "../api";

type Props = {
  task: CurrentTask | null;
  // Static snapshot to show when nothing is training (last rollout, or null).
  fallback: RolloutSnapshot | null;
  history: { smoothedReward: number[]; smoothedSuccess: number[] };
};

const VIEW = 520;
const TASK_LABELS: Record<string, string> = {
  generate_dataset: "Collecting demos",
  train_policy: "Behavior cloning",
  dagger: "DAgger refining",
  train_rl: "RL practicing",
  evaluate_model: "Evaluating",
  evaluate_courses: "Scoring courses",
};

// Center panel of the gym. While training, it streams the latest rollout
// snapshot + smoothed success/reward curves. Idle, it shows the last snapshot.
export function LivePanel({ task, fallback, history }: Props) {
  const rollout = task?.extra?.rollout ?? fallback;
  const progress = task?.progress ?? 0;
  const taskLabel = task ? TASK_LABELS[task.name] ?? task.name : null;

  return (
    <div className="gym-live">
      <div className="gym-live-head">
        <div className="gym-live-status">
          {task ? (
            <>
              <span className="gym-live-dot active" />
              <strong>{taskLabel}</strong>
              <span className="gym-live-detail">{task.detail}</span>
            </>
          ) : (
            <>
              <span className="gym-live-dot" />
              <span className="gym-live-detail">Idle — assemble a plan and press Train</span>
            </>
          )}
        </div>
        {task && (
          <div className="gym-live-progress">
            <div
              className="gym-live-progress-fill"
              style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }}
            />
          </div>
        )}
      </div>
      <RolloutSvg rollout={rollout} />
      <Curves
        smoothedSuccess={
          task?.extra?.smoothed_success ?? history.smoothedSuccess
        }
        smoothedReward={
          task?.extra?.smoothed_reward ?? history.smoothedReward
        }
      />
    </div>
  );
}

function RolloutSvg({ rollout }: { rollout: RolloutSnapshot | null }) {
  if (!rollout) {
    return (
      <div className="gym-live-empty">
        <div>Live rollout will appear here as training runs.</div>
        <div className="gym-live-empty-sub">
          Snapshots come from a fixed preview world — same room every time so you can
          watch the pilot improve.
        </div>
      </div>
    );
  }
  const env = rollout.env;
  const scale = VIEW / Math.max(env.room_width, env.room_height);
  const xs = (x: number) => x * scale;
  const ys = (y: number) => VIEW - y * scale;
  const pathD = rollout.trajectory
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xs(p.x).toFixed(1)} ${ys(p.y).toFixed(1)}`)
    .join(" ");
  const head = rollout.trajectory[rollout.trajectory.length - 1];
  return (
    <svg
      className="gym-live-svg"
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x={0} y={0} width={VIEW} height={VIEW} className="gym-live-room" />
      {env.obstacles.map((o, i) => (
        <circle
          key={i}
          cx={xs(o.x)}
          cy={ys(o.y)}
          r={o.radius * scale}
          className="gym-live-obstacle"
        />
      ))}
      <circle
        cx={xs(env.start[0])}
        cy={ys(env.start[1])}
        r={env.drone_radius * scale}
        className="gym-live-start"
      />
      <circle
        cx={xs(env.target[0])}
        cy={ys(env.target[1])}
        r={env.target_radius * scale}
        className={`gym-live-target ${rollout.success ? "hit" : ""}`}
      />
      <path d={pathD} className="gym-live-path" />
      {head && (
        <circle
          cx={xs(head.x)}
          cy={ys(head.y)}
          r={env.drone_radius * scale}
          className={`gym-live-drone ${
            rollout.success ? "success" : rollout.collision ? "crash" : ""
          }`}
        />
      )}
      <text x={12} y={20} className="gym-live-stamp">
        {rollout.success
          ? `✓ landed in ${rollout.steps} steps`
          : rollout.collision
            ? `✗ crashed at step ${rollout.steps}`
            : `… ${rollout.steps} steps`}
      </text>
    </svg>
  );
}

function Curves({
  smoothedSuccess,
  smoothedReward,
}: {
  smoothedSuccess: number[];
  smoothedReward: number[];
}) {
  const W = 520;
  const H = 80;
  return (
    <div className="gym-live-curves">
      <CurveSvg label="success (smoothed)" data={smoothedSuccess} W={W} H={H} yRange={[0, 1]} />
      <CurveSvg label="reward (smoothed)" data={smoothedReward} W={W} H={H} />
    </div>
  );
}

function CurveSvg({
  label,
  data,
  W,
  H,
  yRange,
}: {
  label: string;
  data: number[];
  W: number;
  H: number;
  yRange?: [number, number];
}) {
  const path = useMemo(() => {
    if (!data || data.length < 2) return null;
    const lo = yRange ? yRange[0] : Math.min(...data);
    const hi = yRange ? yRange[1] : Math.max(...data);
    const span = hi - lo || 1;
    const xs = (i: number) => (i / (data.length - 1)) * W;
    const ys = (v: number) => H - ((v - lo) / span) * H;
    return data.map((v, i) => `${i === 0 ? "M" : "L"} ${xs(i).toFixed(1)} ${ys(v).toFixed(1)}`).join(" ");
  }, [data, W, H, yRange?.[0], yRange?.[1]]);
  return (
    <div className="gym-live-curve">
      <div className="gym-live-curve-label">
        <span>{label}</span>
        {data && data.length > 0 && (
          <span className="gym-live-curve-val">
            {data[data.length - 1].toFixed(yRange ? 2 : 1)}
          </span>
        )}
      </div>
      <svg className="gym-live-curve-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {path ? <path d={path} className="gym-live-curve-path" /> : null}
      </svg>
    </div>
  );
}
