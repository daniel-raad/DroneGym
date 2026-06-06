import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { Arcade } from "./arcade/Arcade";
import { Workshop } from "./gym/Workshop";
import "./styles.css";

type Route = { mode: "arcade" | "gym"; pilot?: string };

// Hash routes used:
//   #/            → Arcade
//   #/gym         → Gym (existing dashboard)
//   #/?pilot=ID   → Arcade with a specific pilot preselected (drone-select step).
//                  Used by PilotStudio after graduation so the new pilot lands
//                  directly in the race flow.
function parseHash(): Route {
  const h = window.location.hash.replace(/^#/, "");
  if (h.startsWith("/gym")) return { mode: "gym" };
  // Parse a pilot param from anything resembling `/?pilot=...` or `/?pilot=...&...`
  const qIdx = h.indexOf("?");
  if (qIdx >= 0) {
    const params = new URLSearchParams(h.slice(qIdx + 1));
    const pilot = params.get("pilot") ?? undefined;
    return { mode: "arcade", pilot };
  }
  return { mode: "arcade" };
}

function Root() {
  const [route, setRoute] = useState<Route>(parseHash);
  useEffect(() => {
    const fn = () => setRoute(parseHash());
    window.addEventListener("hashchange", fn);
    return () => window.removeEventListener("hashchange", fn);
  }, []);

  const openGym = () => {
    window.location.hash = "/gym";
  };
  const openArcade = () => {
    window.location.hash = "/";
  };
  const openArcadeWithPilot = (pilotId: string) => {
    window.location.hash = `/?pilot=${encodeURIComponent(pilotId)}`;
  };

  if (route.mode === "gym") {
    return (
      <Workshop
        onGraduated={openArcadeWithPilot}
        onOpenArcade={openArcade}
      />
    );
  }
  return <Arcade onOpenGym={openGym} initialPilotId={route.pilot ?? null} />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
