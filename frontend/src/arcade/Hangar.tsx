import { useEffect } from "react";

type Props = {
  onEnter: () => void;
  onOpenGym: () => void;
};

// Cold open. Calm and confident — one headline, one CTA, one quiet link to
// the Gym. Everything else has been deliberately removed.
export function Hangar({ onEnter, onOpenGym }: Props) {
  return (
    <div className="arc-hangar">
      <div className="arc-hangar-bg" />

      <button className="arc-gym-door" onClick={onOpenGym} title="Open the Gym (training dashboard)">
        GYM&nbsp;→
      </button>

      <div className="arc-hangar-stage">
        <div className="arc-eyebrow">An arcade for AI pilots</div>
        <h1 className="arc-title">
          Pick a drone. Pick a brain. <em>Watch it fly.</em>
        </h1>
        <p className="arc-sub">
          Every pilot in here is a real algorithm. A hand-coded controller, a small neural
          network, a language model. They share the same drone, the same room, the same
          three rays of perception — and they fly very differently.
        </p>
        <button className="arc-cta" onClick={onEnter}>
          Start &rarr;
        </button>
        <div className="arc-hint">
          <kbd>↵</kbd> to start &nbsp;·&nbsp; want to train your own pilot? open the{" "}
          <button className="arc-link" onClick={onOpenGym}>
            Gym
          </button>
        </div>
      </div>

      <KeyToEnter onEnter={onEnter} />
    </div>
  );
}

function KeyToEnter({ onEnter }: { onEnter: () => void }) {
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") onEnter();
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [onEnter]);
  return null;
}
