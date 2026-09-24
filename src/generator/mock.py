"""Phase A: erzeugt plausible Dummy-Varianten ohne trainiertes Modell.

Strategie: Falls eine reale Zukunft (ground truth) in der Szene vorliegt, wird
sie als Basis genommen und mit wachsendem, aber physikalisch plausiblem Rauschen
(korrelierte Random-Walk-Abweichung, keine Sprünge) pro Variante leicht verändert.
Liegt keine reale Zukunft vor, wird aus der letzten beobachteten Geschwindigkeit
linear extrapoliert und danach genauso verrauscht. Rückgabestruktur ist exakt
identisch zu TrainedModelVariantGenerator, damit die UI keinen Unterschied sieht.
"""

from __future__ import annotations

import random

from src.datapipeline.schema import AgentState, SceneContext, Trajectory
from src.generator.base import VariantGenerator


class MockVariantGenerator(VariantGenerator):
    def __init__(self, noise_scale: float = 0.35, seed: int | None = None) -> None:
        self.noise_scale = noise_scale
        self._rng = random.Random(seed)

    def generate(self, scene_context: SceneContext, n_variants: int) -> list[Trajectory]:
        variants: list[Trajectory] = []
        for v in range(n_variants):
            agent_futures: dict[str, list[AgentState]] = {}
            for agent in scene_context.agents:
                base = agent.future or self._extrapolate(agent.history, scene_context.horizon_steps, scene_context.dt)
                agent_futures[agent.agent_id] = self._perturb(base)
            variants.append(Trajectory(variant_id=f"{scene_context.scene_id}-mock-{v}", agent_futures=agent_futures))
        return variants

    def _extrapolate(self, history: list[AgentState], horizon_steps: int, dt: float) -> list[AgentState]:
        if not history:
            return []
        last = history[-1]
        out = []
        x, y = last.x, last.y
        for _ in range(horizon_steps):
            x += last.vx * dt
            y += last.vy * dt
            out.append(AgentState(x=x, y=y, vx=last.vx, vy=last.vy, heading=last.heading))
        return out

    def _perturb(self, base: list[AgentState]) -> list[AgentState]:
        if not base:
            return []
        out: list[AgentState] = []
        drift_x = 0.0
        drift_y = 0.0
        for i, s in enumerate(base):
            growth = self.noise_scale * ((i + 1) / max(len(base), 1)) ** 0.5
            drift_x += self._rng.gauss(0, growth * 0.15)
            drift_y += self._rng.gauss(0, growth * 0.15)
            out.append(
                AgentState(
                    x=s.x + drift_x,
                    y=s.y + drift_y,
                    vx=s.vx,
                    vy=s.vy,
                    heading=s.heading,
                )
            )
        return out
