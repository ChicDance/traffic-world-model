"""Austauschbare Inferenz-Schnittstelle (siehe plan.md, Abschnitt 5).

Backend und UI rufen ausschliesslich VariantGenerator.generate() auf. Welche
Implementierung dahinter steckt (Mock in Phase A, echtes Modell in Phase B),
wird ueber die Umgebungsvariable GENERATOR_MODE gesteuert (siehe config.py) und
ist fuer Backend/UI nicht unterscheidbar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.datapipeline.schema import SceneContext, Trajectory


class VariantGenerator(ABC):
    @abstractmethod
    def generate(self, scene_context: SceneContext, n_variants: int) -> list[Trajectory]:
        """Erzeugt n_variants plausible Fortsetzungen der Szene.

        Rueckgabe: Liste von Trajectory, eine je Variante, jede mit einer
        Zukunftssequenz pro Agent aus scene_context.agents.
        """
        raise NotImplementedError
