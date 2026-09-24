"""Zentrale Konfiguration inkl. der einen austauschbaren Zeile aus plan.md
Abschnitt 5: GENERATOR_MODE steuert, ob Mock- oder trainiertes Modell verwendet
wird. Backend und Frontend bleiben beim Wechsel unveraendert.
"""

from __future__ import annotations

import os

from src.generator.base import VariantGenerator
from src.generator.mock import MockVariantGenerator

GENERATOR_MODE = os.environ.get("GENERATOR_MODE", "mock")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/model.pt")
SCENES_DIR = os.environ.get("SCENES_DIR", "data/scenes")
DEVICE = os.environ.get("DEVICE", "cpu")


def build_generator() -> VariantGenerator:
    if GENERATOR_MODE == "mock":
        return MockVariantGenerator(seed=0)
    if GENERATOR_MODE == "trained":
        # Phase B: Checkpoint muss vorher per src/model/train.py auf der GPU-VM
        # erzeugt worden sein (siehe README "VM-Anleitung").
        from src.generator.trained import TrainedModelVariantGenerator

        return TrainedModelVariantGenerator(CHECKPOINT_PATH, device=DEVICE)
    raise ValueError(f"Unbekannter GENERATOR_MODE: {GENERATOR_MODE!r} (erwartet 'mock' oder 'trained')")
