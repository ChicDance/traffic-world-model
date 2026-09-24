# DLR-UT World-Model Demo

Interactive browser demo: from a briefly observed real traffic situation at the
AIM research intersection in Braunschweig (DLR-UT dataset), a generative
trajectory world model produces several plausible continuations of the scene,
shown as a bird's-eye-view map animation ("Real" vs. "Generated variants").
Details and background: [`plan.md`](./plan.md) (in German — the original
project planning document).

This repo is deliberately split into two phases on two machines:

- **Phase A (laptop, no GPU):** complete code, mock inference, app runs in the browser.
- **Phase B (VM, GPU):** real training, then `GENERATOR_MODE=trained`.

## Phase A — run locally (no GPU needed)

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git-Bash; on cmd/PowerShell: .venv\Scripts\activate
pip install -r requirements-cpu.txt

# Optional: extract real DLR-UT scenes (otherwise synthetic demo scenes are used)
python scripts/build_scenes.py

# Optional: load real lane geometry from the OpenDRIVE map of Braunschweig's
# inner ring road (separate dataset, see scripts/extract_opendrive_lanes.py)
python -m pip install pyxodr
python scripts/extract_opendrive_lanes.py

# Generate the map background (uses the OpenDRIVE geometry if present,
# otherwise falls back to lane/path density from real trajectories)
python -m scripts.build_map_background

# Sanity-check the training script (dummy data, seconds, no real training)
python -m src.model.train --sanity-check

# Tests
pytest

# Start the app
GENERATOR_MODE=mock uvicorn src.backend.app:app --reload --port 8000
```

Then open in your browser: **http://127.0.0.1:8000**

- **Play**: replay the real scene (observation + real continuation)
- **Generate variants**: calls the `MockVariantGenerator`, shows N plausible
  alternative continuations as dashed, colored trails next to the real trajectory
- Scene picker top-left; legend at the bottom of the stage area

## Architecture

- `src/datapipeline/schema.py` — shared data structures (SceneContext, AgentTrack, Trajectory)
- `scripts/build_scenes.py` — extracts interaction scenes from raw DLR-UT CSVs
- `scripts/extract_opendrive_lanes.py` — loads real lane polygons from the separate OpenDRIVE map of Braunschweig's inner ring road (Zenodo DOI 10.5281/zenodo.4043193), cropped to the scenes' map extent
- `scripts/build_map_background.py` — renders `frontend/map_background.png`: prefers the real OpenDRIVE geometry, otherwise falls back to trajectory density
- `src/model/diffusion.py` — trajectory diffusion model (methodology after NVlabs/CTG, see plan.md)
- `src/model/train.py` — training script (`--sanity-check` for Phase A, real training only in Phase B)
- `src/generator/` — swappable `VariantGenerator` interface: `MockVariantGenerator` (Phase A) and `TrainedModelVariantGenerator` (Phase B)
- `src/backend/app.py` — FastAPI service, injects the generator via `GENERATOR_MODE`
- `frontend/` — static BEV UI (vanilla JS, no build step needed)

## VM instructions (Phase B, GPU VM with 8x V100)

```bash
git clone <repo-url> && cd traffic-world-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt

# Full scene extraction over the entire dataset (optionally extend with --all-files)
python scripts/build_scenes.py --max-scenes 200

# Real training on one of the 8 V100s (see plan.md section 3, ~2-6h)
python -m src.model.train --scenes-dir data/scenes --epochs 50 --device cuda \
    --checkpoint-out checkpoints/model.pt

# Start the app with the trained model
GENERATOR_MODE=trained CHECKPOINT_PATH=checkpoints/model.pt DEVICE=cuda \
    uvicorn src.backend.app:app --port 8000
```

Backend, frontend, and all endpoints stay unchanged when switching from `mock`
to `trained` — only the `GENERATOR_MODE` environment variable changes (see
`src/backend/config.py`, plan.md section 5).

## Phase A status (definition of done, see plan.md section 6)

- [x] App runs locally in the browser, Play/Generate work with `MockVariantGenerator`
- [x] Training script written, dummy sanity check passed, **not actually trained**
- [x] `requirements-gpu.txt` prepared, not installed/tested on the laptop
- [x] This README contains the VM instructions for Phase B
