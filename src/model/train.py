"""Trainingsskript fuer das SceneDiffusionModel.

WICHTIG (siehe plan.md): Auf dem Laptop (Phase A, kein GPU) darf dieses Skript
NUR mit --sanity-check ausgefuehrt werden (winzige Dummy-Daten, 1-2 Iterationen,
CPU, Sekunden). Ein echtes mehrepochiges Training auf den realen DLR-UT-Daten
gehoert ausschliesslich in Phase B auf die GPU-VM:

    python -m src.model.train --scenes-dir data/scenes --epochs 50 --device cuda

Sanity-Check (Phase A, erlaubt):
    python -m src.model.train --sanity-check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.dataset import DummySceneDataset, SceneDataset
from src.model.diffusion import SceneDiffusionModel


def collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) if k != "n_agents" else [b[k] for b in batch] for k in keys}


def train_loop(
    model: SceneDiffusionModel,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    max_steps: int | None = None,
) -> list[float]:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            history = batch["history"].to(device)
            hist_pad = batch["hist_key_padding_mask"].to(device)
            agent_class = batch["agent_class"].to(device)
            agent_pad = batch["agent_padding_mask"].to(device)
            future_deltas = batch["future_deltas"].to(device)

            b = history.shape[0]
            t = torch.randint(0, model.n_diffusion_steps, (b,), device=device)
            noise = torch.randn_like(future_deltas)
            noisy = model.q_sample(future_deltas, t, noise)

            pred_noise = model(history, agent_class, hist_pad, agent_pad, noisy, t)

            valid = (~agent_pad).float()[:, :, None, None]
            loss = (((pred_noise - noise) ** 2) * valid).sum() / valid.sum().clamp(min=1.0) / future_deltas.shape[-1]

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            step += 1
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            if max_steps is not None and step >= max_steps:
                return losses
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Nur winzige Dummy-Daten, 1-2 Iterationen, CPU.")
    parser.add_argument("--scenes-dir", type=str, default="data/scenes")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--max-hist-len", type=int, default=20)
    parser.add_argument("--n-diffusion-steps", type=int, default=100)
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/model.pt")
    args = parser.parse_args()

    if args.sanity_check:
        print("=== SANITY CHECK (Phase A): Dummy-Daten, 2 Iterationen, CPU, kein echtes Training ===")
        dataset = DummySceneDataset(n_scenes=4, max_agents=3, max_hist_len=8, horizon_steps=6)
        model = SceneDiffusionModel(
            d_model=32, n_layers=1, n_heads=2, max_hist_len=8, horizon_steps=6, n_diffusion_steps=10
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
        losses = train_loop(model, loader, device="cpu", epochs=1, lr=args.lr, max_steps=2)
        assert len(losses) == 2, "Sanity-Check sollte genau 2 Trainingsschritte ausfuehren"
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), "Loss ist nicht endlich -- Bug im Modell"
        print(f"Sanity-Check OK. Losses: {losses}")
        return

    print(
        "WARNUNG: Echtes Training angefordert. Dieses Skript soll ausschliesslich in "
        "Phase B auf der GPU-VM mit den vollstaendigen echten Daten laufen (siehe plan.md). "
        "Auf dem Laptop ohne GPU bitte nur --sanity-check verwenden."
    )
    if not torch.cuda.is_available():
        print(
            "Kein CUDA-Geraet gefunden. Breche ab, um versehentliches CPU-Training auf "
            "echten Daten zu verhindern (das wuerde Tage statt Stunden dauern)."
        )
        return

    scenes_dir = Path(args.scenes_dir)
    dataset = SceneDataset(scenes_dir, max_agents=12, max_hist_len=args.max_hist_len, horizon_steps=args.horizon_steps)
    model = SceneDiffusionModel(
        d_model=args.d_model,
        max_hist_len=args.max_hist_len,
        horizon_steps=args.horizon_steps,
        n_diffusion_steps=args.n_diffusion_steps,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    train_loop(model, loader, device=args.device, epochs=args.epochs, lr=args.lr)

    out_path = Path(args.checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, out_path)
    print(f"Checkpoint gespeichert: {out_path}")


if __name__ == "__main__":
    main()
