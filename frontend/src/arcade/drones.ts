// Drone chassis roster. For v0 these are cosmetic — the backend has one
// physics model — but the chosen chassis tints the drone color in the replay
// and sets the flavor of the race screen.

export type DroneId = "hummingbird" | "tank" | "phantom" | "switchblade";

export type Drone = {
  id: DroneId;
  name: string;
  emoji: string;
  tagline: string;
  color: string; // primary neon color for trail / sprite
  stats: { speed: number; agility: number; battery: number; sensors: number };
};

export const DRONES: Drone[] = [
  {
    id: "switchblade",
    name: "Switchblade",
    emoji: "🛩",
    tagline: "Balanced all-rounder. The default cool kid.",
    color: "#5db8ff",
    stats: { speed: 7, agility: 7, battery: 7, sensors: 7 },
  },
  {
    id: "hummingbird",
    name: "Hummingbird",
    emoji: "🐦",
    tagline: "Twitchy and fast. Snaps turns nobody else can.",
    color: "#ff5dc9",
    stats: { speed: 9, agility: 9, battery: 4, sensors: 5 },
  },
  {
    id: "tank",
    name: "Tank",
    emoji: "🛡",
    tagline: "Slow, armored, ignores wind. Bring a snack.",
    color: "#ffd166",
    stats: { speed: 4, agility: 3, battery: 9, sensors: 6 },
  },
  {
    id: "phantom",
    name: "Phantom",
    emoji: "👻",
    tagline: "Long-range sensors, thin battery. Plans ahead.",
    color: "#9d5dff",
    stats: { speed: 7, agility: 6, battery: 5, sensors: 9 },
  },
];

export function droneById(id: DroneId): Drone {
  return DRONES.find((d) => d.id === id) ?? DRONES[0];
}
