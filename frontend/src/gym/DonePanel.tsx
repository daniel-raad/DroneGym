import type { CourseResult } from "../api";
import type { RecipeName } from "./plan";
import { RECIPE_META } from "./plan";

type Props = {
  raceReady: boolean;
  results: Map<string, CourseResult>;
  currentRecipe: RecipeName;
  onSendToArcade: () => void;
  onTryHarder: (next: RecipeName) => void;
  onRetry: () => void;
};

const PROGRESSION: RecipeName[] = [
  "clone-the-teacher",
  "minimum-viable",
  "beat-the-teacher",
  "robust-pilot",
];

// Shown after a training plan finishes. Frames the result as either
// "you're done, race it" or "here's the next thing to try".
export function DonePanel({
  raceReady,
  results,
  currentRecipe,
  onSendToArcade,
  onTryHarder,
  onRetry,
}: Props) {
  const totalSuccess =
    results.size > 0
      ? Array.from(results.values()).reduce((a, b) => a + b.success_rate, 0) /
        results.size
      : 0;
  const idx = PROGRESSION.indexOf(currentRecipe);
  const next = idx >= 0 && idx < PROGRESSION.length - 1 ? PROGRESSION[idx + 1] : null;

  return (
    <div className={`dp-card ${raceReady ? "ready" : "incomplete"}`}>
      <div className="dp-headline">
        {raceReady ? (
          <>
            <span className="dp-tick">★</span>
            <span>Race-ready</span>
          </>
        ) : (
          <>
            <span className="dp-warn">…</span>
            <span>Not quite there yet</span>
          </>
        )}
        <span className="dp-avg">
          avg {(totalSuccess * 100).toFixed(0)}% across {results.size} course
          {results.size === 1 ? "" : "s"}
        </span>
      </div>
      <div className="dp-actions">
        {raceReady ? (
          <button className="dp-primary" onClick={onSendToArcade}>
            Send to Arcade →
          </button>
        ) : next ? (
          <button className="dp-primary" onClick={() => onTryHarder(next)}>
            Try harder recipe: {RECIPE_META[next].name} →
          </button>
        ) : null}
        <button className="dp-ghost" onClick={onRetry}>
          {raceReady ? "Train another" : "Retry this recipe"}
        </button>
      </div>
    </div>
  );
}
