// Persists user-created pilots in localStorage. The "character" is just a
// model file on the backend plus a small bag of presentation metadata — name,
// face, hue — that the arcade reads to render the card and the mind-stream.

const STORAGE_KEY = "dronegym.customPilots.v1";

export type CustomPilot = {
  id: string; // "custom:<slug>"
  modelName: string; // backend model filename (no extension)
  name: string;
  face: string; // emoji
  hue: number; // 0-360
  createdAt: number;
  practiced: boolean;
  bcAccuracy: number | null; // 0..1
  evalSuccess: number | null; // 0..1, last eval-sim success rate
  notes?: string;
};

function safeParse(raw: string | null): CustomPilot[] {
  if (!raw) return [];
  try {
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr.filter((p) => p && typeof p.id === "string" && typeof p.modelName === "string");
  } catch {
    return [];
  }
}

export function loadCustomPilots(): CustomPilot[] {
  if (typeof window === "undefined") return [];
  return safeParse(window.localStorage.getItem(STORAGE_KEY));
}

export function saveCustomPilots(pilots: CustomPilot[]): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(pilots));
  window.dispatchEvent(new CustomEvent("dronegym:custom-pilots-changed"));
}

export function upsertCustomPilot(pilot: CustomPilot): void {
  const all = loadCustomPilots();
  const i = all.findIndex((p) => p.id === pilot.id);
  if (i >= 0) all[i] = pilot;
  else all.unshift(pilot); // newest first
  saveCustomPilots(all);
}

export function removeCustomPilot(id: string): void {
  saveCustomPilots(loadCustomPilots().filter((p) => p.id !== id));
}

export function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 24) || "pilot"
  );
}

// Subscribe to changes (same-tab + cross-tab via storage event).
export function onCustomPilotsChange(fn: () => void): () => void {
  const local = () => fn();
  const remote = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) fn();
  };
  window.addEventListener("dronegym:custom-pilots-changed", local);
  window.addEventListener("storage", remote);
  return () => {
    window.removeEventListener("dronegym:custom-pilots-changed", local);
    window.removeEventListener("storage", remote);
  };
}
