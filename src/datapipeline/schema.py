"""Gemeinsame Datenstrukturen fuer Szenen, Agenten-Tracks und generierte Varianten.

Diese Strukturen sind die Schnittstelle zwischen Datenpipeline, Modell/Generator
und FastAPI-Backend. Sie sind bewusst unabhaengig von TASI/pandas gehalten, damit
Mock- und Trained-Generator exakt dieselbe Form zurueckgeben.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AgentClass(str, Enum):
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    OTHER = "other"


@dataclass
class AgentState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float | None = None


@dataclass
class AgentTrack:
    agent_id: str
    agent_class: AgentClass
    history: list[AgentState]
    future: list[AgentState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_class": self.agent_class.value,
            "history": [vars(s) for s in self.history],
            "future": [vars(s) for s in self.future],
        }


@dataclass
class SceneContext:
    scene_id: str
    dt: float
    horizon_steps: int
    agents: list[AgentTrack]
    map_bounds: dict[str, float]
    lane_polygons: list[list[tuple[float, float]]] = field(default_factory=list)
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "dt": self.dt,
            "horizon_steps": self.horizon_steps,
            "agents": [a.to_dict() for a in self.agents],
            "map_bounds": self.map_bounds,
            "lane_polygons": self.lane_polygons,
            "source": self.source,
        }


@dataclass
class Trajectory:
    """Eine generierte Variante: pro Agent eine Zukunftssequenz."""

    variant_id: str
    agent_futures: dict[str, list[AgentState]]

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "agent_futures": {
                aid: [vars(s) for s in states] for aid, states in self.agent_futures.items()
            },
        }
