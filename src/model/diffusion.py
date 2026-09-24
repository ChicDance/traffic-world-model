"""Trajektorien-Diffusionsmodell fuer die gemeinsame Fortsetzung einer Verkehrsszene.

Methodisches Vorbild: NVlabs/CTG (Guided Conditional Diffusion for Controllable
Traffic Simulation), Trajectron++/Trajeglish (siehe plan.md Abschnitt 2). Eigene,
schlanke Implementierung, zugeschnitten auf das DLR-UT-Szenenformat.

Kernidee: Das Modell sampelt nicht pro Agent isoliert, sondern die zukuenftigen
Positionsdeltas ALLER Agenten einer Szene gemeinsam -- Konsistenz zwischen den
Agenten entsteht durch Cross-Agent-Attention sowohl im History-Encoder als auch
im Denoising-Netzwerk selbst.

Eingaben sind auf feste Tensor-Shapes gepolstert (Padding + Attention-Mask), damit
Szenen mit unterschiedlich vielen Agenten batchbar sind.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

N_AGENT_CLASSES = 4  # vehicle, bicycle, pedestrian, other (siehe schema.AgentClass)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard DDPM-Sinus-Embedding fuer den Diffusions-Zeitschritt."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class AgentHistoryEncoder(nn.Module):
    """Kodiert die Beobachtungshistorie jedes Agenten zu einem Kontextvektor
    und laesst Agenten anschliessend per Self-Attention voneinander "wissen"
    (Szene-Kontext), bevor daraus die Konditionierung fuer den Denoiser wird.
    """

    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, max_hist_len: int = 50):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(4, d_model)  # x, y, vx, vy
        self.class_embed = nn.Embedding(N_AGENT_CLASSES, d_model)
        self.pos_embed = nn.Parameter(torch.randn(max_hist_len, d_model) * 0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=n_layers)

        scene_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(scene_layer, num_layers=n_layers)

    def forward(
        self,
        history: torch.Tensor,  # (B, A, T_obs, 4)
        agent_class: torch.Tensor,  # (B, A) long
        hist_key_padding_mask: torch.Tensor,  # (B, A, T_obs) bool, True = padded/ignore
        agent_padding_mask: torch.Tensor,  # (B, A) bool, True = padded agent slot
    ) -> torch.Tensor:
        b, a, t, _ = history.shape
        x = self.input_proj(history) + self.pos_embed[:t][None, None, :, :]
        x = x + self.class_embed(agent_class)[:, :, None, :]

        x = x.reshape(b * a, t, self.d_model)
        pad = hist_key_padding_mask.reshape(b * a, t)
        # Agenten ohne jegliche Historie wuerden eine komplett maskierte Sequenz
        # ergeben (NaN in Attention) -- fuer diese eine Position freigeben.
        all_masked = pad.all(dim=1)
        pad = pad.clone()
        pad[all_masked, 0] = False

        encoded = self.temporal_encoder(x, src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        pooled = pooled.reshape(b, a, self.d_model)

        scene_ctx = self.scene_encoder(pooled, src_key_padding_mask=agent_padding_mask)
        return scene_ctx  # (B, A, d_model)


class SceneDenoiser(nn.Module):
    """Sagt den Diffusions-Rauschterm fuer die zukuenftigen Positionsdeltas
    aller Agenten gemeinsam voraus, konditioniert auf Szenen-Kontext + Zeitschritt.
    """

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, horizon_steps: int = 30):
        super().__init__()
        self.d_model = d_model
        self.horizon_steps = horizon_steps

        self.delta_proj = nn.Linear(2, d_model)
        self.time_pos_embed = nn.Parameter(torch.randn(horizon_steps, d_model) * 0.02)
        self.timestep_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=n_layers)

        cross_agent_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.cross_agent_encoder = nn.TransformerEncoder(cross_agent_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        noisy_deltas: torch.Tensor,  # (B, A, T_pred, 2)
        scene_ctx: torch.Tensor,  # (B, A, d_model) aus AgentHistoryEncoder
        t: torch.Tensor,  # (B,) Diffusions-Zeitschritt
        agent_padding_mask: torch.Tensor,  # (B, A) bool, True = padded agent
    ) -> torch.Tensor:
        b, a, tp, _ = noisy_deltas.shape
        t_emb = self.timestep_mlp(sinusoidal_timestep_embedding(t, self.d_model))  # (B, d_model)

        x = self.delta_proj(noisy_deltas) + self.time_pos_embed[:tp][None, None, :, :]
        x = x + scene_ctx[:, :, None, :] + t_emb[:, None, None, :]

        # Zeitliche Selbstaufmerksamkeit je Agent
        x_t = x.reshape(b * a, tp, self.d_model)
        agent_mask_rep = agent_padding_mask.reshape(b * a)
        x_t_safe = x_t.clone()
        x_t = self.temporal_encoder(x_t)
        x_t = torch.where(agent_mask_rep[:, None, None], x_t_safe, x_t)
        x = x_t.reshape(b, a, tp, self.d_model)

        # Cross-Agent-Attention je Zeitschritt -> gemeinsame, konsistente Szene
        x_a = x.permute(0, 2, 1, 3).reshape(b * tp, a, self.d_model)
        agent_mask_expanded = agent_padding_mask.unsqueeze(1).expand(b, tp, a).reshape(b * tp, a)
        x_a = self.cross_agent_encoder(x_a, src_key_padding_mask=agent_mask_expanded)
        x = x_a.reshape(b, tp, a, self.d_model).permute(0, 2, 1, 3)

        return self.out_proj(x)  # (B, A, T_pred, 2) predicted noise


class SceneDiffusionModel(nn.Module):
    """Fasst History-Encoder + Denoiser zusammen und kapselt den DDPM-Rauschplan."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_hist_len: int = 50,
        horizon_steps: int = 30,
        n_diffusion_steps: int = 100,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.n_diffusion_steps = n_diffusion_steps

        self.encoder = AgentHistoryEncoder(d_model, n_layers, n_heads, max_hist_len)
        self.denoiser = SceneDenoiser(d_model, n_layers + 1, n_heads, horizon_steps)

        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1m_ac = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def forward(
        self,
        history: torch.Tensor,
        agent_class: torch.Tensor,
        hist_key_padding_mask: torch.Tensor,
        agent_padding_mask: torch.Tensor,
        noisy_future_deltas: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        scene_ctx = self.encoder(history, agent_class, hist_key_padding_mask, agent_padding_mask)
        return self.denoiser(noisy_future_deltas, scene_ctx, t, agent_padding_mask)

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        agent_class: torch.Tensor,
        hist_key_padding_mask: torch.Tensor,
        agent_padding_mask: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """DDPM-Sampling: gibt (n_samples, A, T_pred, 2) Positionsdeltas zurueck."""
        device = history.device
        b, a, _, _ = history.shape
        scene_ctx = self.encoder(history, agent_class, hist_key_padding_mask, agent_padding_mask)

        outputs = []
        for _ in range(n_samples):
            x = torch.randn(b, a, self.horizon_steps, 2, device=device)
            for step in reversed(range(self.n_diffusion_steps)):
                t = torch.full((b,), step, device=device, dtype=torch.long)
                eps_pred = self.denoiser(x, scene_ctx, t, agent_padding_mask)
                alpha = self.alphas[step]
                alpha_cumprod = self.alphas_cumprod[step]
                beta = self.betas[step]
                mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_cumprod).sqrt()) * eps_pred)
                if step > 0:
                    noise = torch.randn_like(x)
                    x = mean + beta.sqrt() * noise
                else:
                    x = mean
            outputs.append(x)
        return torch.stack(outputs, dim=0)
