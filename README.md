# DLR-UT World-Model-Demo

Interaktive Browser-Demo: Aus einer kurz beobachteten realen Verkehrssituation
an der AIM-Forschungskreuzung Braunschweig (DLR-UT-Datensatz) erzeugt ein
generatives Trajektorien-World-Model mehrere plausible Fortsetzungen der Szene,
dargestellt als Bird's-Eye-View-Kartenanimation ("Real" vs. "Generierte
Varianten"). Details und Hintergrund: [`plan.md`](./plan.md).

Dieses Repo ist bewusst in zwei Phasen auf zwei Maschinen aufgeteilt:

- **Phase A (Laptop, kein GPU):** kompletter Code, Mock-Inferenz, App laeuft im Browser.
- **Phase B (VM, GPU):** echtes Training, danach GENERATOR_MODE=trained.

## Phase A — lokal starten (kein GPU noetig)

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git-Bash; unter cmd/PowerShell: .venv\Scripts\activate
pip install -r requirements-cpu.txt

# Optional: echte DLR-UT-Szenen extrahieren (sonst laufen synthetische Demo-Szenen)
python scripts/build_scenes.py

# Optional: echte Fahrspurgeometrie aus der OpenDRIVE-Karte der Braunschweiger
# Innenstadtring-Strassen laden (separater Datensatz, siehe scripts/extract_opendrive_lanes.py)
python -m pip install pyxodr
python scripts/extract_opendrive_lanes.py

# Kartenhintergrund erzeugen (nutzt die OpenDRIVE-Geometrie falls vorhanden,
# sonst Fahrspur-/Wege-Dichte aus echten Trajektorien als Fallback)
python -m scripts.build_map_background

# Sanity-Check des Trainingsskripts (Dummy-Daten, Sekunden, kein echtes Training)
python -m src.model.train --sanity-check

# Tests
pytest

# App starten
GENERATOR_MODE=mock uvicorn src.backend.app:app --reload --port 8000
```

Danach im Browser: **http://127.0.0.1:8000**

- **Play**: reale Szene abspielen (Beobachtung + reale Fortsetzung)
- **Generiere Varianten**: ruft den `MockVariantGenerator` auf, zeigt N plausible
  Alternativ-Fortsetzungen gestrichelt/farbig neben der realen Trajektorie
- Szenen-Auswahl oben links; Legende unten im Stage-Bereich

## Architektur

- `src/datapipeline/schema.py` — gemeinsame Datenstrukturen (SceneContext, AgentTrack, Trajectory)
- `scripts/build_scenes.py` — extrahiert Interaktionsszenen aus rohen DLR-UT-CSVs
- `scripts/extract_opendrive_lanes.py` — laedt echte Fahrspur-Polygone aus der separaten OpenDRIVE-Karte der Braunschweiger Innenstadtring-Strassen (Zenodo DOI 10.5281/zenodo.4043193), beschraenkt auf den Szenen-Kartenausschnitt
- `scripts/build_map_background.py` — rendert `frontend/map_background.png`: bevorzugt aus der echten OpenDRIVE-Geometrie, sonst als Fallback aus der Trajektoriendichte
- `src/model/diffusion.py` — Trajektorien-Diffusionsmodell (Methodik nach NVlabs/CTG, siehe plan.md)
- `src/model/train.py` — Trainingsskript (`--sanity-check` fuer Phase A, echtes Training nur Phase B)
- `src/generator/` — austauschbare `VariantGenerator`-Schnittstelle: `MockVariantGenerator` (Phase A) und `TrainedModelVariantGenerator` (Phase B)
- `src/backend/app.py` — FastAPI-Service, injiziert den Generator ueber `GENERATOR_MODE`
- `frontend/` — statische BEV-UI (Vanilla JS, kein Build-Schritt noetig)

## VM-Anleitung (Phase B, GPU-VM mit 8x V100)

```bash
git clone <repo-url> && cd traffic-world-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt

# Volle Szenen-Extraktion ueber den gesamten Datensatz (optional --all-files erweitern)
python scripts/build_scenes.py --max-scenes 200

# Echtes Training auf einer der 8 V100 (siehe plan.md Abschnitt 3, ca. 2-6h)
python -m src.model.train --scenes-dir data/scenes --epochs 50 --device cuda \
    --checkpoint-out checkpoints/model.pt

# App mit trainiertem Modell starten
GENERATOR_MODE=trained CHECKPOINT_PATH=checkpoints/model.pt DEVICE=cuda \
    uvicorn src.backend.app:app --port 8000
```

Backend, Frontend und alle Endpunkte bleiben beim Wechsel von `mock` zu `trained`
unveraendert — es wird ausschliesslich die Umgebungsvariable `GENERATOR_MODE`
umgestellt (siehe `src/backend/config.py`, plan.md Abschnitt 5).

## Status Phase A (Definition of Done, siehe plan.md Abschnitt 6)

- [x] App laeuft lokal im Browser, Play/Generate funktionieren mit `MockVariantGenerator`
- [x] Trainingsskript geschrieben, Dummy-Sanity-Check bestanden, **nicht real trainiert**
- [x] `requirements-gpu.txt` vorbereitet, auf dem Laptop nicht installiert/getestet
- [x] Diese README enthaelt die VM-Anleitung fuer Phase B
