import type { Course, CourseResult } from "../api";

const READY = 0.5;

type Props = {
  courses: Course[];
  results: Map<string, CourseResult>;
  raceReady: boolean;
  canSend: boolean;
  onSendToArcade: () => void;
};

// The big readiness header. The three meters answer the question "is this
// pilot worth racing yet?" — which is the gym's whole job.
export function CourseScorecard({
  courses,
  results,
  raceReady,
  canSend,
  onSendToArcade,
}: Props) {
  return (
    <div className="gym-scorecard">
      <div className="gym-scorecard-label">
        <div className="gym-scorecard-title">Course readiness</div>
        <div className="gym-scorecard-sub">
          {raceReady ? "Pilot is race-ready" : `Hit ≥${READY * 100}% on any course to race`}
        </div>
      </div>
      <div className="gym-scorecard-courses">
        {courses.map((c) => {
          const r = results.get(c.id);
          const pct = r ? r.success_rate : 0;
          const ready = pct >= READY;
          return (
            <div key={c.id} className={`gym-course-meter ${ready ? "ready" : ""}`}>
              <div className="gym-course-meter-label">{c.label}</div>
              <div className="gym-course-meter-bar">
                <div
                  className="gym-course-meter-fill"
                  style={{ width: `${Math.min(pct, 1) * 100}%` }}
                />
                <div className="gym-course-meter-tick" style={{ left: `${READY * 100}%` }} />
              </div>
              <div className="gym-course-meter-pct">
                {r ? `${(pct * 100).toFixed(0)}%` : "—"}
                {r ? <span className="gym-course-meter-n"> / {r.runs}</span> : null}
              </div>
            </div>
          );
        })}
      </div>
      <button
        className="gym-send-arcade"
        disabled={!canSend || !raceReady}
        onClick={onSendToArcade}
        title={
          !canSend
            ? "Name and train a pilot first"
            : raceReady
              ? "Save this pilot and open the arcade"
              : "Pilot is not race-ready"
        }
      >
        Send to Arcade →
      </button>
    </div>
  );
}
