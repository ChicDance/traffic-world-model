"""Torch-Dataset fuer Szenen im schema.SceneContext-Format (siehe datapipeline/schema.py).

Liest die von scripts/build_scenes.py erzeugten JSON-Szenen aus data/scenes/ und
wandelt sie in feste, gepolsterte Tensoren fuer das Diffusionsmodell um.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.datapipeline.schema import AgentClass, SceneContext

CLASS_TO_IDX = {c.value: i for i, c in enumerate(AgentClass)}


class SceneDataset(Dataset):
    def __init__(self, scenes_dir: str | Path, max_agents: int = 12, max_hist_len: int = 20, horizon_steps: int = 30):
        self.scenes_dir = Path(scenes_dir)
        self.files = sorted(self.scenes_dir.glob("*.json"))
        self.max_agents = max_agents
        self.max_hist_len = max_hist_len
        self.horizon_steps = horizon_steps

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with open(self.files[idx], encoding="utf-8") as f:
            scene = json.load(f)
        return scene_to_tensors(scene, self.max_agents, self.max_hist_len, self.horizon_steps)


def scene_to_tensors(scene: dict, max_agents: int, max_hist_len: int, horizon_steps: int) -> dict:
    agents = scene["agents"][:max_agents]
    n_agents = len(agents)

    history = torch.zeros(max_agents, max_hist_len, 4)
    hist_pad = torch.ones(max_agents, max_hist_len, dtype=torch.bool)
    agent_class = torch.zeros(max_agents, dtype=torch.long)
    agent_pad = torch.ones(max_agents, dtype=torch.bool)
    future_deltas = torch.zeros(max_agents, horizon_steps, 2)
    future_pad = torch.ones(max_agents, horizon_steps, dtype=torch.bool)

    for i, agent in enumerate(agents):
        agent_pad[i] = False
        agent_class[i] = CLASS_TO_IDX.get(agent["agent_class"], CLASS_TO_IDX["other"])

        hist = agent["history"][-max_hist_len:]
        for j, s in enumerate(hist):
            history[i, j] = torch.tensor([s["x"], s["y"], s["vx"], s["vy"]])
            hist_pad[i, j] = False

        fut = agent["future"][:horizon_steps]
        if fut:
            prev_x, prev_y = (hist[-1]["x"], hist[-1]["y"]) if hist else (fut[0]["x"], fut[0]["y"])
            for j, s in enumerate(fut):
                future_deltas[i, j] = torch.tensor([s["x"] - prev_x, s["y"] - prev_y])
                prev_x, prev_y = s["x"], s["y"]
                future_pad[i, j] = False

    return {
        "history": history,
        "hist_key_padding_mask": hist_pad,
        "agent_class": agent_class,
        "agent_padding_mask": agent_pad,
        "future_deltas": future_deltas,
        "future_padding_mask": future_pad,
        "n_agents": n_agents,
    }


def scene_context_to_inference_tensors(
    scene: SceneContext, max_agents: int, max_hist_len: int
) -> dict:
    """Wie scene_to_tensors, aber direkt aus einer SceneContext-Instanz (Live-Inferenz,
    keine future-Werte noetig) und mit Batch-Dimension 1 fuer model.sample().
    """
    history = torch.zeros(1, max_agents, max_hist_len, 4)
    hist_pad = torch.ones(1, max_agents, max_hist_len, dtype=torch.bool)
    agent_class = torch.zeros(1, max_agents, dtype=torch.long)
    agent_pad = torch.ones(1, max_agents, dtype=torch.bool)

    agents = scene.agents[:max_agents]
    for i, agent in enumerate(agents):
        agent_pad[0, i] = False
        agent_class[0, i] = CLASS_TO_IDX.get(agent.agent_class.value, CLASS_TO_IDX["other"])
        hist = agent.history[-max_hist_len:]
        for j, s in enumerate(hist):
            history[0, i, j] = torch.tensor([s.x, s.y, s.vx, s.vy])
            hist_pad[0, i, j] = False

    return {
        "history": history,
        "hist_key_padding_mask": hist_pad,
        "agent_class": agent_class,
        "agent_padding_mask": agent_pad,
        "agent_order": [a.agent_id for a in agents],
    }


class DummySceneDataset(Dataset):
    """Winzige synthetische Szenen fuer den Sanity-Check (keine echten Daten,
    keine GPU noetig, laeuft in Sekunden). Siehe plan.md Abschnitt 6, Tag 2.
    """

    def __init__(self, n_scenes: int = 4, max_agents: int = 3, max_hist_len: int = 8, horizon_steps: int = 6):
        self.n_scenes = n_scenes
        self.max_agents = max_agents
        self.max_hist_len = max_hist_len
        self.horizon_steps = horizon_steps

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(idx)
        n_agents = max(1, self.max_agents - (idx % self.max_agents))

        history = torch.zeros(self.max_agents, self.max_hist_len, 4)
        hist_pad = torch.ones(self.max_agents, self.max_hist_len, dtype=torch.bool)
        agent_class = torch.zeros(self.max_agents, dtype=torch.long)
        agent_pad = torch.ones(self.max_agents, dtype=torch.bool)
        future_deltas = torch.zeros(self.max_agents, self.horizon_steps, 2)
        future_pad = torch.ones(self.max_agents, self.horizon_steps, dtype=torch.bool)

        for i in range(n_agents):
            agent_pad[i] = False
            agent_class[i] = i % len(CLASS_TO_IDX)
            history[i] = torch.randn(self.max_hist_len, 4, generator=g)
            hist_pad[i] = False
            future_deltas[i] = torch.randn(self.horizon_steps, 2, generator=g) * 0.1
            future_pad[i] = False

        return {
            "history": history,
            "hist_key_padding_mask": hist_pad,
            "agent_class": agent_class,
            "agent_padding_mask": agent_pad,
            "future_deltas": future_deltas,
            "future_padding_mask": future_pad,
            "n_agents": n_agents,
        }
