import { useEffect, useState } from "react";
import { DRONES, type DroneId } from "./drones";

type Props = {
  initial?: DroneId;
  onPick: (id: DroneId) => void;
  onBack: () => void;
};

// Drone chassis select. Same shape as PilotSelect but tighter — speed-pick,
// not a soul-search. Stats are cosmetic v0; chassis tints the in-race color.
export function DroneSelect({ initial, onPick, onBack }: Props) {
  const [idx, setIdx] = useState(() => {
    if (!initial) return 0;
    const i = DRONES.findIndex((d) => d.id === initial);
    return i < 0 ? 0 : i;
  });
  const move = (delta: number) =>
    setIdx((i) => (i + delta + DRONES.length) % DRONES.length);

  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") move(-1);
      else if (e.key === "ArrowRight") move(1);
      else if (e.key === "Enter") onPick(DRONES[idx].id);
      else if (e.key === "Escape") onBack();
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [idx, onPick, onBack]);

  const active = DRONES[idx];

  return (
    <div className="arc-select drone" style={{ ['--accent' as any]: active.color }}>
      <header className="arc-select-head">
        <button className="arc-back" onClick={onBack}>
          ← BACK
        </button>
        <div className="arc-select-title">SELECT&nbsp;YOUR&nbsp;DRONE</div>
        <span style={{ width: 80 }} />
      </header>

      <div className="arc-drone-grid">
        {DRONES.map((d, i) => (
          <button
            key={d.id}
            className={`arc-drone-card ${i === idx ? "selected" : ""}`}
            style={{ ['--accent' as any]: d.color }}
            onClick={() => setIdx(i)}
          >
            <div className="arc-drone-emoji">{d.emoji}</div>
            <div className="arc-drone-name">{d.name}</div>
            <div className="arc-drone-tag">{d.tagline}</div>
            <div className="arc-stats compact">
              <Mini label="SPD" v={d.stats.speed} />
              <Mini label="AGI" v={d.stats.agility} />
              <Mini label="BAT" v={d.stats.battery} />
              <Mini label="SEN" v={d.stats.sensors} />
            </div>
          </button>
        ))}
      </div>

      <div className="arc-confirm-bar">
        <button className="arc-cta" onClick={() => onPick(active.id)}>
          FLY&nbsp;{active.name.toUpperCase()}
          <span className="arc-cta-glow" />
        </button>
        <div className="arc-hint center">←/→ · ↵ to confirm · esc to back out</div>
      </div>
    </div>
  );
}

function Mini({ label, v }: { label: string; v: number }) {
  return (
    <div className="arc-stat mini">
      <div className="arc-stat-label">{label}</div>
      <div className="arc-stat-track">
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className={`arc-stat-pip ${i < v ? "on" : ""}`} />
        ))}
      </div>
    </div>
  );
}
