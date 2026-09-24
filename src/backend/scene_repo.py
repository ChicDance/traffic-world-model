"""Laedt kuratierte Szenen aus data/scenes/*.json (siehe scripts/build_scenes.py).

Falls (noch) keine extrahierten Szenen vorliegen -- z.B. weil die Pipeline noch
nicht gelaufen ist -- erzeugt dieses Modul ein paar synthetische Platzhalterszenen,
damit Backend und UI trotzdem sofort end-to-end lauffaehig sind.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.datapipeline.schema import AgentClass, AgentState, AgentTrack, SceneContext


def _dict_to_scene(d: dict) -> SceneContext:
    agents = [
        AgentTrack(
            agent_id=a["agent_id"],
            agent_class=AgentClass(a["agent_class"]),
            history=[AgentState(**s) for s in a["history"]],
            future=[AgentState(**s) for s in a["future"]],
        )
        for a in d["agents"]
    ]
    return SceneContext(
        scene_id=d["scene_id"],
        dt=d["dt"],
        horizon_steps=d["horizon_steps"],
        agents=agents,
        map_bounds=d["map_bounds"],
        lane_polygons=[[tuple(p) for p in poly] for poly in d.get("lane_polygons", [])],
        source=d.get("source", "unknown"),
    )


def _synthetic_scenes() -> list[SceneContext]:
    """Kleine handgebaute Szenen (Kfz + Fussgaenger kreuzen sich), nur als
    Fallback bevor scripts/build_scenes.py gegen echte Daten gelaufen ist.
    """
    dt = 0.25
    n_obs, n_fut = 8, 12
    scenes = []

    # Szene 1: Fahrzeug faehrt geradeaus, Fussgaenger quert von der Seite
    car_hist = [AgentState(x=-20 + i * 2.0, y=0.0, vx=8.0, vy=0.0) for i in range(n_obs)]
    car_fut = [AgentState(x=-20 + (n_obs + i) * 2.0, y=0.0, vx=8.0, vy=0.0) for i in range(n_fut)]
    ped_hist = [AgentState(x=0.0, y=-8 + i * 0.9, vx=0.0, vy=1.4) for i in range(n_obs)]
    ped_fut = [AgentState(x=0.0, y=-8 + (n_obs + i) * 0.9, vx=0.0, vy=1.4) for i in range(n_fut)]
    scenes.append(
        SceneContext(
            scene_id="demo-car-pedestrian",
            dt=dt,
            horizon_steps=n_fut,
            agents=[
                AgentTrack("car-1", AgentClass.VEHICLE, car_hist, car_fut),
                AgentTrack("ped-1", AgentClass.PEDESTRIAN, ped_hist, ped_fut),
            ],
            map_bounds={"xmin": -25, "xmax": 25, "ymin": -12, "ymax": 12},
            lane_polygons=[[(-25, -3.5), (25, -3.5), (25, 3.5), (-25, 3.5)]],
            source="synthetic-fallback",
        )
    )

    # Szene 2: Kreuzung mit Fahrzeug, Radfahrer, Fussgaenger
    car_hist = [AgentState(x=-18 + i * 1.8, y=-1.5, vx=7.2, vy=0.0) for i in range(n_obs)]
    car_fut = [AgentState(x=-18 + (n_obs + i) * 1.8, y=-1.5, vx=7.2, vy=0.0) for i in range(n_fut)]
    bike_hist = [AgentState(x=1.5, y=-16 + i * 1.6, vx=0.0, vy=3.2) for i in range(n_obs)]
    bike_fut = [AgentState(x=1.5, y=-16 + (n_obs + i) * 1.6, vx=0.0, vy=3.2) for i in range(n_fut)]
    ped_hist = [AgentState(x=-6 + i * 0.6, y=6.0, vx=1.0, vy=0.0) for i in range(n_obs)]
    ped_fut = [AgentState(x=-6 + (n_obs + i) * 0.6, y=6.0, vx=1.0, vy=0.0) for i in range(n_fut)]
    scenes.append(
        SceneContext(
            scene_id="demo-intersection",
            dt=dt,
            horizon_steps=n_fut,
            agents=[
                AgentTrack("car-2", AgentClass.VEHICLE, car_hist, car_fut),
                AgentTrack("bike-1", AgentClass.BICYCLE, bike_hist, bike_fut),
                AgentTrack("ped-2", AgentClass.PEDESTRIAN, ped_hist, ped_fut),
            ],
            map_bounds={"xmin": -20, "xmax": 20, "ymin": -20, "ymax": 20},
            lane_polygons=[
                [(-20, -4), (20, -4), (20, 4), (-20, 4)],
                [(-2, -20), (2, -20), (2, 20), (-2, 20)],
            ],
            source="synthetic-fallback",
        )
    )

    return scenes


class SceneRepository:
    def __init__(self, scenes_dir: str | Path):
        self.scenes_dir = Path(scenes_dir)
        self._scenes: dict[str, SceneContext] = {}
        self._load()

    def _load(self) -> None:
        files = sorted(self.scenes_dir.glob("[!_]*.json")) if self.scenes_dir.exists() else []
        if files:
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
                scene = _dict_to_scene(d)
                self._scenes[scene.scene_id] = scene
        else:
            for scene in _synthetic_scenes():
                self._scenes[scene.scene_id] = scene

    def list_summaries(self) -> list[dict]:
        return [
            {
                "scene_id": s.scene_id,
                "n_agents": len(s.agents),
                "classes": sorted({a.agent_class.value for a in s.agents}),
                "source": s.source,
            }
            for s in self._scenes.values()
        ]

    def get(self, scene_id: str) -> SceneContext | None:
        return self._scenes.get(scene_id)
