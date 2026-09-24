"""Extrahiert kurze Interaktionsszenen aus rohen DLR-UT-Trajektorien-CSVs und
speichert sie im schema.SceneContext-JSON-Format nach data/scenes/.

Laedt bewusst nur ausgewaehlte 15-Minuten-Rohdateien (nicht den ganzen, mehrere
GB grossen Datensatz), das reicht fuer eine Handvoll kuratierter Demo-Szenen
(siehe plan.md, Tag 1 + Tag 4). Fuer echtes Training in Phase B auf der VM kann
dasselbe Skript mit --all-files ueber den vollstaendigen Datensatz laufen.

Versucht zuerst TASI (DLR-TS/TASI) fuer den CSV-Import zu nutzen (Referenz laut
plan.md Abschnitt 4); falls TASI/geopandas auf der jeweiligen Maschine nicht
sauber installierbar ist, faellt das Skript auf direktes pandas-Parsing des
verifizierten DLR-UT-Schemas zurueck (Spalten siehe unten), ohne Ergebnis-Unterschied
fuer die hier benoetigten Felder (Position, Geschwindigkeit, Klasse, Zeit).

Verifiziertes Rohschema (Phase A, Tag 1):
    timestamp, id, center_easting, center_northing, velocity_easting,
    velocity_northing, velocity_magnitude, acceleration_easting,
    acceleration_northing, acceleration_magnitude, yaw,
    dimension_length, dimension_width, dimension_height,
    classifications_pedestrian, classifications_bicycle, classifications_motorbike,
    classifications_car, classifications_van, classifications_truck
Koordinatensystem: UTM (easting/northing, Meter). Sample-Rate: ~20 Hz (dt ~= 0.05s).
Klassifikation liegt als Wahrscheinlichkeitsverteilung ueber 6 Klassen vor (nicht
als hartes Label) -- wir nehmen argmax und mappen auf schema.AgentClass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/extracted/DLR-UT_v1-0-0/trajectories")
OUT_DIR = Path("data/scenes")

CLASS_COLS = {
    "classifications_pedestrian": "pedestrian",
    "classifications_bicycle": "bicycle",
    "classifications_motorbike": "vehicle",
    "classifications_car": "vehicle",
    "classifications_van": "vehicle",
    "classifications_truck": "vehicle",
}

RAW_HZ = 20.0  # ~20 Hz native Abtastrate im DLR-UT-Rohformat (siehe Docstring)


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Laedt eine rohe DLR-UT-Trajektorien-CSV. Versucht zuerst TASI, faellt bei
    Problemen (z.B. fehlende geopandas-Systemabhaengigkeiten) auf pandas zurueck.
    """
    try:
        from tasi.dlr import DLRTrajectoryDataset  # noqa: F401

        ds = DLRTrajectoryDataset.from_csv(str(path))
        df = ds.data if hasattr(ds, "data") else ds
        if isinstance(df, pd.DataFrame) and "center_easting" in df.columns:
            print(f"  (geladen via TASI: {path.name})")
            return df
        raise RuntimeError("Unerwartetes TASI-Rueckgabeformat, falle auf pandas zurueck")
    except Exception as exc:  # noqa: BLE001 - bewusst breiter Fallback, siehe Docstring
        print(f"  (TASI nicht nutzbar ({exc.__class__.__name__}: {exc}), lade direkt mit pandas: {path.name})")
        return pd.read_csv(path, parse_dates=["timestamp"])


def classify(row: pd.Series) -> str:
    probs = {label: 0.0 for label in set(CLASS_COLS.values())}
    for col, label in CLASS_COLS.items():
        probs[label] = probs.get(label, 0.0) + float(row.get(col, 0.0))
    return max(probs, key=probs.get)


def extract_scenes(
    df: pd.DataFrame,
    obs_seconds: float = 2.0,
    pred_seconds: float = 3.0,
    downsample_hz: float = 4.0,
    stride_seconds: float = 5.0,
    min_agents: int = 3,
    max_scenes: int = 6,
) -> list[dict]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["agent_class"] = df.apply(classify, axis=1)

    step_every = max(1, round(RAW_HZ / downsample_hz))
    dt = step_every / RAW_HZ

    t0 = df["timestamp"].min()
    df["t_sec"] = (df["timestamp"] - t0).dt.total_seconds()

    window_len = obs_seconds + pred_seconds
    t_max = df["t_sec"].max()

    scenes: list[dict] = []
    t = obs_seconds
    while t + pred_seconds <= t_max and len(scenes) < max_scenes * 8:
        window = df[(df["t_sec"] >= t - obs_seconds) & (df["t_sec"] < t + pred_seconds)]
        n_agents = window["id"].nunique()
        if n_agents >= min_agents:
            classes_present = set(window.groupby("id")["agent_class"].first())
            scene = build_scene(df, t, obs_seconds, pred_seconds, step_every, dt, classes_present)
            if scene is not None:
                scenes.append(scene)
        t += stride_seconds

    scenes.sort(key=lambda s: -len(s["agents"]))
    return scenes[:max_scenes]


def build_scene(
    df: pd.DataFrame,
    t_center: float,
    obs_seconds: float,
    pred_seconds: float,
    step_every: int,
    dt: float,
    classes_present: set[str],
) -> dict | None:
    agents = []
    window = df[(df["t_sec"] >= t_center - obs_seconds) & (df["t_sec"] < t_center + pred_seconds)]

    for agent_id, g in window.groupby("id"):
        g = g.sort_values("t_sec").iloc[::step_every]
        obs = g[g["t_sec"] < t_center]
        fut = g[g["t_sec"] >= t_center]
        if len(obs) < 3 or len(fut) < 3:
            continue

        history = [
            {"x": float(r.center_easting), "y": float(r.center_northing), "vx": float(r.velocity_easting), "vy": float(r.velocity_northing), "heading": float(r.yaw)}
            for r in obs.itertuples()
        ]
        future = [
            {"x": float(r.center_easting), "y": float(r.center_northing), "vx": float(r.velocity_easting), "vy": float(r.velocity_northing), "heading": float(r.yaw)}
            for r in fut.itertuples()
        ]
        agents.append(
            {
                "agent_id": str(agent_id),
                "agent_class": g["agent_class"].iloc[0],
                "history": history,
                "future": future,
            }
        )

    if len(agents) < 2:
        return None

    all_x = [s["x"] for a in agents for s in a["history"] + a["future"]]
    all_y = [s["y"] for a in agents for s in a["history"] + a["future"]]
    map_bounds = {"xmin": min(all_x) - 5, "xmax": max(all_x) + 5, "ymin": min(all_y) - 5, "ymax": max(all_y) + 5}

    return {
        "scene_id": f"scene-{t_center:.1f}",
        "dt": dt,
        "horizon_steps": min(len(a["future"]) for a in agents),
        "agents": agents,
        "map_bounds": map_bounds,
        # Fahrspuren/Wege werden nicht mehr als grobe Konvexe-Huelle pro Szene
        # rekonstruiert, sondern als Dichte-Rasterkarte ueber den ganzen
        # Kreuzungsbereich gerendert (siehe scripts/build_map_background.py)
        # und vom Frontend als Hintergrundbild eingeblendet.
        "lane_polygons": [],
        "source": "DLR-UT_v1-0-0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=str, default="trajectories_230924-080000_230924-081500.csv")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--min-agents", type=int, default=3)
    args = parser.parse_args()

    raw_path = RAW_DIR / args.raw_file
    print(f"Lade Rohdaten: {raw_path}")
    df = load_raw_csv(raw_path)
    print(f"  {len(df)} Zeilen, {df['id'].nunique()} Objekte")

    scenes = extract_scenes(df, min_agents=args.min_agents, max_scenes=args.max_scenes)
    print(f"{len(scenes)} Interaktionsszenen extrahiert")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        out_path = out_dir / f"{scene['scene_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scene, f)
        print(f"  gespeichert: {out_path} ({len(scene['agents'])} Agenten)")


if __name__ == "__main__":
    main()
