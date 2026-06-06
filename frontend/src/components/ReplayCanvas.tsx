import React, { useEffect, useMemo, useRef, useState } from "react";
import type { EpisodeResponse, EnvironmentConfig } from "../api";

type Props = {
  episode: EpisodeResponse | null;
  env: EnvironmentConfig | null;
  overlay?: EpisodeResponse | null;
  overlayLLM?: EpisodeResponse | null;
  onStepChange?: (step: number) => void;
  showSensors?: boolean;
  controlsMode?: "full" | "live";
};

const VIEW = 720;

export function ReplayCanvas({
  episode,
  env,
  overlay,
  overlayLLM,
  onStepChange,
  showSensors = false,
  controlsMode = "full",
}: Props) {
  const room = episode?.environment ?? overlay?.environment ?? overlayLLM?.environment ?? env;
  const traj = episode?.trajectory ?? [];
  const trajB = overlay?.trajectory ?? [];
  const trajC = overlayLLM?.trajectory ?? [];
  const totalSteps =
    Math.max(traj.length, trajB.length, trajC.length) > 0
      ? Math.max(traj.length, trajB.length, trajC.length) - 1
      : 0;

  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [episode?.episode_id, overlay?.episode_id, overlayLLM?.episode_id]);

  // In live mode (manual flight), pin step to the end so the canvas always
  // shows the latest position the user has flown to.
  useEffect(() => {
    if (controlsMode === "live" && totalSteps !== step) {
      setStep(totalSteps);
    }
  }, [controlsMode, totalSteps, step]);

  useEffect(() => {
    onStepChange?.(step);
  }, [step, onStepChange]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (t: number) => {
      const dt = t - last;
      if (dt > 60) {
        last = t;
        setStep((s) => {
          if (s >= totalSteps) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, totalSteps]);

  const scale = useMemo(() => {
    if (!room) return 1;
    return VIEW / Math.max(room.room_width, room.room_height);
  }, [room]);

  if (!room)
    return (
      <div className="canvas-wrap">
        <div style={{ color: "var(--muted)" }}>
          Generate an environment to begin.
        </div>
      </div>
    );

  const w = room.room_width * scale;
  const h = room.room_height * scale;
  const tx = (x: number) => x * scale;
  const ty = (y: number) => h - y * scale;

  const visibleA = traj.slice(0, step + 1);
  const visibleB = trajB.slice(0, step + 1);
  const visibleC = trajC.slice(0, step + 1);
  const currentA = traj[Math.min(step, traj.length - 1)];
  const currentB = trajB[Math.min(step, trajB.length - 1)];
  const currentC = trajC[Math.min(step, trajC.length - 1)];

  const collisionA = episode?.events.find((e) => e.type.startsWith("collision"));
  const collisionB = overlay?.events.find((e) => e.type.startsWith("collision"));
  const collisionC = overlayLLM?.events.find((e) => e.type.startsWith("collision"));
  const successA = episode?.events.find(
    (e) => e.type === "reached_target" || e.type === "land_success",
  );
  const successB = overlay?.events.find(
    (e) => e.type === "reached_target" || e.type === "land_success",
  );
  const successC = overlayLLM?.events.find(
    (e) => e.type === "reached_target" || e.type === "land_success",
  );

  return (
    <>
      {(overlay || overlayLLM) && (
        <div
          style={{
            display: "flex",
            gap: 16,
            padding: "8px 12px",
            fontSize: 12,
            borderBottom: "1px solid var(--border)",
            background: "var(--panel)",
            flexWrap: "wrap",
          }}
        >
          {episode && (
            <span>
              <span style={{ color: "#5db8ff" }}>●</span> {episode.summary.agent_type} ·{" "}
              {episode.success ? "✓" : episode.collision ? "✗" : "…"} steps {episode.steps}
            </span>
          )}
          {overlay && (
            <span>
              <span style={{ color: "#ffd166" }}>●</span> {overlay.summary.agent_type} ·{" "}
              {overlay.success ? "✓" : overlay.collision ? "✗" : "…"} steps {overlay.steps}
            </span>
          )}
          {overlayLLM && (
            <span>
              <span style={{ color: "#b65dff" }}>●</span> {overlayLLM.summary.agent_type} ·{" "}
              {overlayLLM.success ? "✓" : overlayLLM.collision ? "✗" : "…"} steps {overlayLLM.steps}
            </span>
          )}
        </div>
      )}
      <div className="canvas-wrap">
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#5db8ff" />
            </marker>
            <marker
              id="arrow2"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#ffd166" />
            </marker>
            <marker
              id="arrow3"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#b65dff" />
            </marker>
          </defs>
          <rect x={0} y={0} width={w} height={h} fill="#07081a" stroke="#2a335c" />
          {Array.from({ length: Math.ceil(room.room_width) + 1 }).map((_, i) => (
            <line
              key={`vx${i}`}
              x1={tx(i)}
              y1={0}
              x2={tx(i)}
              y2={h}
              stroke="#141a36"
              strokeWidth={1}
            />
          ))}
          {Array.from({ length: Math.ceil(room.room_height) + 1 }).map((_, i) => (
            <line
              key={`hy${i}`}
              x1={0}
              y1={ty(i)}
              x2={w}
              y2={ty(i)}
              stroke="#141a36"
              strokeWidth={1}
            />
          ))}

          {room.obstacles.map((ob, i) => (
            <circle
              key={i}
              cx={tx(ob.x)}
              cy={ty(ob.y)}
              r={ob.radius * scale}
              fill="rgba(182, 93, 255, 0.25)"
              stroke="#b65dff"
              strokeWidth={1.5}
            />
          ))}

          <circle
            cx={tx(room.target[0])}
            cy={ty(room.target[1])}
            r={room.target_radius * scale}
            fill="rgba(89, 224, 139, 0.18)"
            stroke="#59e08b"
            strokeWidth={2}
          />
          <circle
            cx={tx(room.target[0])}
            cy={ty(room.target[1])}
            r={4}
            fill="#59e08b"
          />

          <circle
            cx={tx(room.start[0])}
            cy={ty(room.start[1])}
            r={6}
            fill="#5db8ff"
            opacity={0.8}
          />

          {visibleA.length > 1 && (
            <polyline
              points={visibleA.map((p) => `${tx(p.x)},${ty(p.y)}`).join(" ")}
              fill="none"
              stroke="#5db8ff"
              strokeWidth={2}
              strokeOpacity={0.9}
            />
          )}
          {overlay && visibleB.length > 1 && (
            <polyline
              points={visibleB.map((p) => `${tx(p.x)},${ty(p.y)}`).join(" ")}
              fill="none"
              stroke="#ffd166"
              strokeWidth={2}
              strokeDasharray="4 4"
              strokeOpacity={0.95}
            />
          )}
          {overlayLLM && visibleC.length > 1 && (
            <polyline
              points={visibleC.map((p) => `${tx(p.x)},${ty(p.y)}`).join(" ")}
              fill="none"
              stroke="#b65dff"
              strokeWidth={2}
              strokeDasharray="2 4"
              strokeOpacity={0.95}
            />
          )}

          {currentA && showSensors && !overlay && episode && episode.observations[step] && (
            <g
              transform={`translate(${tx(currentA.x)} ${ty(currentA.y)}) rotate(${-currentA.heading})`}
              opacity={0.85}
            >
              {/* heading is along +x in local frame; left = +90°, right = −90° */}
              <line
                x1={0}
                y1={0}
                x2={episode.observations[step].front_distance * scale}
                y2={0}
                stroke="#59e08b"
                strokeWidth={1.5}
                strokeDasharray="3 3"
              />
              <line
                x1={0}
                y1={0}
                x2={0}
                y2={-episode.observations[step].left_distance * scale}
                stroke="#5db8ff"
                strokeWidth={1.5}
                strokeDasharray="3 3"
              />
              <line
                x1={0}
                y1={0}
                x2={0}
                y2={episode.observations[step].right_distance * scale}
                stroke="#ffd166"
                strokeWidth={1.5}
                strokeDasharray="3 3"
              />
            </g>
          )}
          {currentA && (
            <g
              transform={`translate(${tx(currentA.x)} ${ty(currentA.y)}) rotate(${-currentA.heading})`}
            >
              <circle
                r={room.drone_radius * scale}
                fill="rgba(93,184,255,0.3)"
                stroke="#5db8ff"
                strokeWidth={2}
              />
              <line
                x1={0}
                y1={0}
                x2={room.drone_radius * scale * 2.2}
                y2={0}
                stroke="#5db8ff"
                strokeWidth={2}
                markerEnd="url(#arrow)"
              />
            </g>
          )}
          {overlay && currentB && (
            <g
              transform={`translate(${tx(currentB.x)} ${ty(currentB.y)}) rotate(${-currentB.heading})`}
            >
              <circle
                r={room.drone_radius * scale}
                fill="rgba(255,209,102,0.28)"
                stroke="#ffd166"
                strokeWidth={2}
              />
              <line
                x1={0}
                y1={0}
                x2={room.drone_radius * scale * 2.2}
                y2={0}
                stroke="#ffd166"
                strokeWidth={2}
                markerEnd="url(#arrow2)"
              />
            </g>
          )}
          {overlayLLM && currentC && (
            <g
              transform={`translate(${tx(currentC.x)} ${ty(currentC.y)}) rotate(${-currentC.heading})`}
            >
              <circle
                r={room.drone_radius * scale}
                fill="rgba(182,93,255,0.28)"
                stroke="#b65dff"
                strokeWidth={2}
              />
              <line
                x1={0}
                y1={0}
                x2={room.drone_radius * scale * 2.2}
                y2={0}
                stroke="#b65dff"
                strokeWidth={2}
                markerEnd="url(#arrow3)"
              />
            </g>
          )}
          {overlayLLM && collisionC && step >= collisionC.step && currentC && (
            <g transform={`translate(${tx(currentC.x)} ${ty(currentC.y)})`}>
              <circle r={14} fill="none" stroke="#ff6b6b" strokeWidth={2.5} />
              <line x1={-10} y1={-10} x2={10} y2={10} stroke="#ff6b6b" strokeWidth={2.5} />
              <line x1={10} y1={-10} x2={-10} y2={10} stroke="#ff6b6b" strokeWidth={2.5} />
            </g>
          )}
          {overlayLLM && successC && (
            <g transform={`translate(${tx(room.target[0])} ${ty(room.target[1])})`}>
              <circle r={24} fill="none" stroke="#b65dff" strokeWidth={2} strokeDasharray="3 3" />
            </g>
          )}

          {collisionA && step >= collisionA.step && currentA && (
            <g transform={`translate(${tx(currentA.x)} ${ty(currentA.y)})`}>
              <circle r={14} fill="none" stroke="#ff6b6b" strokeWidth={2.5} />
              <line x1={-10} y1={-10} x2={10} y2={10} stroke="#ff6b6b" strokeWidth={2.5} />
              <line x1={10} y1={-10} x2={-10} y2={10} stroke="#ff6b6b" strokeWidth={2.5} />
            </g>
          )}
          {overlay && collisionB && step >= collisionB.step && currentB && (
            <g transform={`translate(${tx(currentB.x)} ${ty(currentB.y)})`}>
              <circle r={14} fill="none" stroke="#ff6b6b" strokeWidth={2.5} />
              <line x1={-10} y1={-10} x2={10} y2={10} stroke="#ff6b6b" strokeWidth={2.5} />
              <line x1={10} y1={-10} x2={-10} y2={10} stroke="#ff6b6b" strokeWidth={2.5} />
            </g>
          )}

          {(successA || (overlay && successB)) && (
            <g transform={`translate(${tx(room.target[0])} ${ty(room.target[1])})`}>
              <circle r={20} fill="none" stroke="#59e08b" strokeWidth={2.5} />
            </g>
          )}
        </svg>
      </div>
      {controlsMode === "full" && (
      <div className="controls">
        <button
          onClick={() => {
            if (step >= totalSteps) setStep(0);
            setPlaying((p) => !p);
          }}
          disabled={totalSteps === 0}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          className="secondary"
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={totalSteps === 0}
        >
          ◀
        </button>
        <button
          className="secondary"
          onClick={() => setStep(Math.min(totalSteps, step + 1))}
          disabled={totalSteps === 0}
        >
          ▶
        </button>
        <input
          type="range"
          min={0}
          max={totalSteps}
          value={step}
          onChange={(e) => {
            setPlaying(false);
            setStep(Number(e.target.value));
          }}
        />
        <span className="step-readout">
          step {step} / {totalSteps}
        </span>
      </div>
      )}
    </>
  );
}
