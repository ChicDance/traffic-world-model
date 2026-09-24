"""Phase B: laedt den trainierten Checkpoint und sampelt echte Varianten.

Wird erst nutzbar, sobald auf der GPU-VM (Phase B) ein Checkpoint via
src/model/train.py erzeugt wurde. In Phase A auf dem Laptop existiert kein
Checkpoint -- dieser Generator ist geschrieben, aber unbenutzt (GENERATOR_MODE=mock).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.datapipeline.schema import AgentState, SceneContext, Trajectory
from src.generator.base import VariantGenerator
from src.model.dataset import scene_context_to_inference_tensors
from src.model.diffusion import SceneDiffusionModel


class TrainedModelVariantGenerator(VariantGenerator):
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", max_agents: int = 12):
        self.device = device
        self.max_agents = max_agents

        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = checkpoint["config"]
        self.max_hist_len = cfg.get("max_hist_len", 20)
        self.model = SceneDiffusionModel(
            d_model=cfg.get("d_model", 128),
            max_hist_len=self.max_hist_len,
            horizon_steps=cfg.get("horizon_steps", 30),
            n_diffusion_steps=cfg.get("n_diffusion_steps", 100),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def generate(self, scene_context: SceneContext, n_variants: int) -> list[Trajectory]:
        tensors = scene_context_to_inference_tensors(scene_context, self.max_agents, self.max_hist_len)
        agent_order = tensors.pop("agent_order")

        samples = self.model.sample(
            history=tensors["history"].to(self.device),
            agent_class=tensors["agent_class"].to(self.device),
            hist_key_padding_mask=tensors["hist_key_padding_mask"].to(self.device),
            agent_padding_mask=tensors["agent_padding_mask"].to(self.device),
            n_samples=n_variants,
        )  # (n_variants, 1, A, T_pred, 2) Deltas

        agents_by_id = {a.agent_id: a for a in scene_context.agents}
        variants: list[Trajectory] = []
        for v in range(n_variants):
            agent_futures: dict[str, list[AgentState]] = {}
            for i, agent_id in enumerate(agent_order):
                agent = agents_by_id[agent_id]
                last = agent.history[-1] if agent.history else AgentState(x=0.0, y=0.0)
                x, y = last.x, last.y
                states = []
                for step in range(samples.shape[3]):
                    dx, dy = samples[v, 0, i, step].tolist()
                    x, y = x + dx, y + dy
                    states.append(AgentState(x=x, y=y, vx=dx / scene_context.dt, vy=dy / scene_context.dt))
                agent_futures[agent_id] = states
            variants.append(Trajectory(variant_id=f"{scene_context.scene_id}-trained-{v}", agent_futures=agent_futures))
        return variants
