import { useEffect, useMemo, useState } from "react";
import { PILOTS, type Pilot, type PilotId } from "./pilots";
import type { SystemStatus } from "../api";

type Props = {
  status: SystemStatus | null;
  customs?: Pilot[];
  initial?: PilotId;
  onPick: (id: PilotId) => void;
  onBack: () => void;
  onOpenGym: () => void;
};

// Character select. Built-in pilots first, then any user-created pilots
// trained in the Gym. Locked pilots show why (need trained model, need LLM
// key). User-created pilots are never locked — if they exist, you can fly them.
export function PilotSelect({ status, customs = [], initial, onPick, onBack, onOpenGym }: Props) {
  const roster = useMemo<Pilot[]>(() => [...PILOTS, ...customs], [customs]);
  const [idx, setIdx] = useState(() => {
    if (!initial) return 0;
    const i = roster.findIndex((p) => p.id === initial);
    return i < 0 ? 0 : i;
  });

  const hasTrained = (status?.models ?? []).some(
    (m) => m.method === "behavior_cloning" || m.method === "a2c" || m.method === "reinforce" || m.method === "ppo" || m.method === "dagger",
  );
  const llmReady = status?.llm_available ?? false;

  const isRaceReady = (p: Pilot): boolean => {
    if (!p.custom || p.backend.kind !== "trained") return false;
    const modelName = p.backend.modelName;
    if (!modelName) return false;
    const m = (status?.models ?? []).find((mi) => mi.name === modelName);
    return !!m?.race_ready;
  };

  const lockedReason = (p: Pilot): string | null => {
    if (p.id === "trained" && !hasTrained) return "no rookie trained yet";
    if (p.id === "llm" && !llmReady) return "no ANTHROPIC_API_KEY set";
    return null;
  };

  const move = (delta: number) =>
    setIdx((i) => (i + delta + roster.length) % roster.length);

  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") move(-1);
      else if (e.key === "ArrowRight") move(1);
      else if (e.key === "Enter") {
        const p = roster[idx];
        if (!lockedReason(p)) onPick(p.id);
      } else if (e.key === "Escape") onBack();
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [idx, onPick, onBack, hasTrained, llmReady, roster]);

  const active = roster[Math.min(idx, roster.length - 1)];
  const locked = lockedReason(active);

  return (
    <div className="arc-select" style={{ ['--hue' as any]: active.hue }}>
      <header className="arc-select-head">
        <button className="arc-back" onClick={onBack}>
          ← BACK
        </button>
        <div className="arc-select-title">SELECT&nbsp;YOUR&nbsp;PILOT</div>
        <button className="arc-gym-door inline" onClick={onOpenGym}>
          GYM →
        </button>
      </header>

      <div className="arc-stage">
        <button className="arc-arrow left" onClick={() => move(-1)} aria-label="prev">
          ‹
        </button>
        <div className="arc-roster">
          {roster.map((p, i) => {
            const dist = ((i - idx + roster.length) % roster.length);
            const relative = dist > roster.length / 2 ? dist - roster.length : dist;
            const isCenter = relative === 0;
            const lock = lockedReason(p);
            return (
              <button
                key={p.id}
                className={`arc-pilot-card ${isCenter ? "center" : ""} ${lock ? "locked" : ""} ${p.custom ? "custom" : ""}`}
                style={{
                  ['--hue' as any]: p.hue,
                  transform: `translateX(${relative * 220}px) scale(${isCenter ? 1 : 0.78}) translateZ(0)`,
                  opacity: Math.abs(relative) > 1 ? 0 : isCenter ? 1 : 0.55,
                  pointerEvents: Math.abs(relative) > 1 ? "none" : "auto",
                  zIndex: 10 - Math.abs(relative),
                }}
                onClick={() => setIdx(i)}
              >
                <div className="arc-pilot-aura" />
                {p.custom && (
                  <div className="arc-pilot-badge">
                    {isRaceReady(p) ? "★ RACE-READY" : "YOUR PILOT"}
                  </div>
                )}
                <div className="arc-pilot-face">{p.face}</div>
                <div className="arc-pilot-name">{p.name}</div>
                <div className="arc-pilot-tag">{p.tagline}</div>
                {lock && <div className="arc-lock-chip">⌧ {lock}</div>}
              </button>
            );
          })}
        </div>
        <button className="arc-arrow right" onClick={() => move(1)} aria-label="next">
          ›
        </button>
      </div>

      <div className="arc-pilot-detail">
        <div className="arc-pilot-bio">
          <div className="arc-quote">“{active.voiceLine}”</div>
          <p>{active.bio}</p>
          {active.id === "trained" && (
            <p className="arc-trainable">
              Neon is the only pilot that can <em>learn</em>. Improve them with
              more demos or RL practice in the{" "}
              <button className="arc-link" onClick={onOpenGym}>
                Gym
              </button>
              .
            </p>
          )}
          {active.id === "heuristic" && (
            <p className="arc-trainable subtle">
              Sgt is hand-coded — won't get better, but won't get worse either.
              He's the teacher Neon learns from.
            </p>
          )}
        </div>
        <div className="arc-stats">
          <StatBar label="REACTION" value={active.stats.reaction} hue={active.hue} />
          <StatBar label="PLANNING" value={active.stats.planning} hue={active.hue} />
          <StatBar label="RELIABILITY" value={active.stats.reliability} hue={active.hue} />
          <StatBar label="CHAOS" value={active.stats.chaos} hue={active.hue} />
        </div>
      </div>

      <div className="arc-confirm-bar">
        {locked ? (
          <button className="arc-cta locked" disabled>
            LOCKED — {locked}
          </button>
        ) : (
          <button className="arc-cta" onClick={() => onPick(active.id)}>
            SELECT&nbsp;{active.name.toUpperCase()}
            <span className="arc-cta-glow" />
          </button>
        )}
        <div className="arc-hint center">←/→ to browse · ↵ to select · esc to back out</div>
      </div>
    </div>
  );
}

function StatBar({ label, value, hue }: { label: string; value: number; hue: number }) {
  return (
    <div className="arc-stat">
      <div className="arc-stat-label">{label}</div>
      <div className="arc-stat-track">
        {Array.from({ length: 10 }, (_, i) => (
          <span
            key={i}
            className={`arc-stat-pip ${i < value ? "on" : ""}`}
            style={{ ['--hue' as any]: hue }}
          />
        ))}
      </div>
    </div>
  );
}
