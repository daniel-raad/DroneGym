import type { RecipeMeta, RecipeName } from "./plan";
import { RECIPE_META } from "./plan";

type Props = {
  selected: RecipeName;
  onSelect: (name: RecipeName) => void;
  disabled?: boolean;
};

const ORDER: RecipeName[] = [
  "clone-the-teacher",
  "minimum-viable",
  "beat-the-teacher",
  "robust-pilot",
];

// Big hero cards. One click = recipe loaded as the plan. No knobs visible.
export function RecipePicker({ selected, onSelect, disabled }: Props) {
  return (
    <div className="rp-grid">
      {ORDER.map((id) => (
        <RecipeCard
          key={id}
          meta={RECIPE_META[id]}
          selected={id === selected}
          disabled={disabled}
          onSelect={() => onSelect(id)}
        />
      ))}
    </div>
  );
}

function RecipeCard({
  meta,
  selected,
  disabled,
  onSelect,
}: {
  meta: RecipeMeta;
  selected: boolean;
  disabled?: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      className={`rp-card ${selected ? "selected" : ""}`}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="rp-card-head">
        <span className="rp-card-name">{meta.name}</span>
        <span className="rp-card-time">{meta.time}</span>
      </div>
      <div className="rp-card-blurb">{meta.blurb}</div>
      <div className="rp-card-steps">
        {meta.steps.map((s, i) => (
          <span key={s} className="rp-card-step">
            {i > 0 && <span className="rp-card-arrow">→</span>}
            {s}
          </span>
        ))}
      </div>
      <div className="rp-card-expected">
        <div className="rp-card-expected-label">Expected outcome</div>
        <div className="rp-card-expected-list">
          {meta.expected.map((e) => {
            const tone = e.label.includes("✓")
              ? "good"
              : e.label.includes("likely") || e.label.includes("maybe")
                ? "maybe"
                : "bad";
            return (
              <span key={e.courseId} className={`rp-card-expected-row ${tone}`}>
                {e.label}
              </span>
            );
          })}
        </div>
      </div>
      {selected && <div className="rp-card-check">✓</div>}
    </button>
  );
}
