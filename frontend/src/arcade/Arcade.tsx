import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { SystemStatus } from "../api";
import { Hangar } from "./Hangar";
import { PilotSelect } from "./PilotSelect";
import { DroneSelect } from "./DroneSelect";
import { RaceScreen } from "./RaceScreen";
import { pilotById, pilotFromCustom, type Pilot, type PilotId } from "./pilots";
import { droneById, type DroneId } from "./drones";
import { loadCustomPilots, onCustomPilotsChange } from "./customPilots";
import "./arcade.css";

type Mode = "hangar" | "pilot" | "drone" | "race";

type Props = {
  onOpenGym: () => void;
  initialPilotId?: PilotId | null; // deep-link from Gym "Go race X" CTA
};

// Top-level shell for the arcade experience. Owns the route between the four
// arcade screens, plus the system status fetch that the pilot-select uses to
// lock/unlock pilots and the custom-pilot roster loaded from localStorage.
export function Arcade({ onOpenGym, initialPilotId }: Props) {
  const [mode, setMode] = useState<Mode>(initialPilotId ? "drone" : "hangar");
  const [pilotId, setPilotId] = useState<PilotId | null>(initialPilotId ?? null);
  const [droneId, setDroneId] = useState<DroneId | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [customRev, setCustomRev] = useState(0);

  const customs: Pilot[] = useMemo(
    () => loadCustomPilots().map(pilotFromCustom),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [customRev],
  );

  useEffect(() => onCustomPilotsChange(() => setCustomRev((r) => r + 1)), []);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.systemStatus());
    } catch {
      /* status is best-effort — Arcade still works without it */
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    const id = setInterval(refreshStatus, 5000);
    return () => clearInterval(id);
  }, [refreshStatus]);

  if (mode === "hangar") {
    return <Hangar onEnter={() => setMode("pilot")} onOpenGym={onOpenGym} />;
  }
  if (mode === "pilot") {
    return (
      <PilotSelect
        status={status}
        customs={customs}
        initial={pilotId ?? undefined}
        onBack={() => setMode("hangar")}
        onOpenGym={onOpenGym}
        onPick={(id) => {
          setPilotId(id);
          setMode("drone");
        }}
      />
    );
  }
  if (mode === "drone") {
    return (
      <DroneSelect
        initial={droneId ?? undefined}
        onBack={() => setMode("pilot")}
        onPick={(id) => {
          setDroneId(id);
          setMode("race");
        }}
      />
    );
  }
  // mode === "race"
  if (!pilotId || !droneId) {
    // Defensive — if state got nuked, ricochet back to pilot select
    setMode("pilot");
    return null;
  }
  return (
    <RaceScreen
      pilot={pilotById(pilotId, customs)}
      drone={droneById(droneId)}
      status={status}
      onBack={() => setMode("hangar")}
      onChangePilot={() => setMode("pilot")}
    />
  );
}
