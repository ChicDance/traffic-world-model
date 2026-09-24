"""Rendert die Kartenhintergrund-Rasterkarte (PNG) fuer die BEV-Ansicht.

Zwei Quellen, in Prioritaetsreihenfolge:

1. **Echte OpenDRIVE-Fahrspurgeometrie** (bevorzugt), sofern
   `scripts/extract_opendrive_lanes.py` vorher gelaufen ist und
   `data/scenes/_opendrive_lanes.json` existiert. Quelle: M. Scholz,
   "OpenDRIVE dataset of the inner ring road in Brunswick", DLR, 2020,
   Zenodo DOI 10.5281/zenodo.4043193 (CC BY 4.0) -- referenziert in der
   DLR-UT-Doku Abschnitt 9.2, aber NICHT Teil des DLR-UT-Downloads selbst.
2. **Fallback: Punktdichte der echten Trajektorien**, falls keine OpenDRIVE-
   Karte extrahiert wurde (siehe plan.md, Risiken: "Kartengeometrie") --
   zeigt Fahrspuren/Wege dort, wo Verkehrsteilnehmer laut den echten Daten
   tatsaechlich fahren/gehen, ist aber unschaerfer als die echte Kartengeometrie.

Ausgabe:
    frontend/map_background.png   -- RGBA-Rasterbild, transparent wo keine Daten
    frontend/map_background.json  -- Welt-Koordinaten-Grenzen (UTM) des Bildes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from scripts.build_scenes import RAW_DIR, load_raw_csv

# Farben passend zu den CSS-Variablen im Frontend (style.css: --vehicle/--bicycle/--pedestrian)
VEHICLE_COLOR = (79, 140, 255)
BICYCLE_COLOR = (255, 180, 84)
PEDESTRIAN_COLOR = (79, 209, 139)

# Fuellfarben fuer die echte OpenDRIVE-Geometrie (gedeckter/"asphalt-artiger" als
# die leuchtenden Dichte-Farben oben, damit Fahrbahnflaechen als Untergrund lesbar
# bleiben und nicht mit den Agenten-Markern/Trails konkurrieren).
OPENDRIVE_FILL = {
    "vehicle": (55, 65, 82, 235),
    "bicycle": (255, 180, 84, 150),
    "pedestrian": (90, 130, 110, 120),
}
OPENDRIVE_OUTLINE = {
    "vehicle": (90, 105, 130, 255),
    "bicycle": (255, 200, 130, 220),
    "pedestrian": (120, 160, 140, 200),
}
# Reihenfolge unten -> oben: Fahrbahn zuerst, Fuss-/Radwege darueber (sonst
# wuerden z.B. Gehwege an Kreuzungsecken von Fahrbahnflaechen verdeckt).
OPENDRIVE_LAYER_ORDER = ["vehicle", "bicycle", "pedestrian"]


def render_opendrive_background(
    polygons: list[dict], xmin: float, ymin: float, width_px: int, height_px: int, resolution: float
) -> Image.Image:
    img = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def to_px(pt: list[float]) -> tuple[float, float]:
        x, y = pt
        return ((x - xmin) / resolution, height_px - (y - ymin) / resolution)

    counts = {g: 0 for g in OPENDRIVE_LAYER_ORDER}
    for group in OPENDRIVE_LAYER_ORDER:
        for poly in polygons:
            if poly["type"] != group:
                continue
            pts = [to_px(p) for p in poly["points"]]
            if len(pts) < 3:
                continue
            draw.polygon(pts, fill=OPENDRIVE_FILL[group], outline=OPENDRIVE_OUTLINE[group])
            counts[group] += 1
    print(f"  OpenDRIVE-Polygone gezeichnet: {counts}")
    return img


def classify_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vektorisierte Variante von build_scenes.classify -- fuer mehrere
    100k Zeilen deutlich schneller als ein zeilenweises .apply().
    """
    probs = pd.DataFrame(
        {
            "pedestrian": df["classifications_pedestrian"],
            "bicycle": df["classifications_bicycle"],
            "vehicle": df[
                ["classifications_motorbike", "classifications_car", "classifications_van", "classifications_truck"]
            ].sum(axis=1),
        }
    )
    return probs.idxmax(axis=1)


def density_grid(xs: np.ndarray, ys: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, blur_sigma_px: float) -> np.ndarray:
    """Punktdichte-Histogramm, geglaettet zu durchgaengigen, spurbreiten Baendern
    statt einzelner Pixel-Punkte, und log-normalisiert auf 0..1.
    """
    hist, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges])
    hist = gaussian_filter(hist, sigma=blur_sigma_px)
    if hist.max() > 0:
        hist = np.log1p(hist) / np.log1p(hist.max())
    return hist.T  # (y_bins, x_bins) -- row-major wie ein Bild


def paint(img: np.ndarray, density: np.ndarray, color: tuple[int, int, int], max_alpha: int, min_alpha: int = 6) -> None:
    alpha = (density * max_alpha).astype(np.uint8)
    flipped = np.flipud(alpha)  # Zeile 0 = Norden (ymax) oben im Bild
    visible = flipped > min_alpha
    for c in range(3):
        img[..., c] = np.where(visible, color[c], img[..., c])
    img[..., 3] = np.where(visible, np.maximum(img[..., 3], flipped), img[..., 3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-files",
        nargs="+",
        default=[
            "trajectories_230924-080000_230924-081500.csv",
            "trajectories_230924-170000_230924-171500.csv",
        ],
        help="Mehrere Zeitfenster ergeben ein vollstaendigeres Bild des Fahrspurnetzes.",
    )
    parser.add_argument("--resolution", type=float, default=0.3, help="Meter pro Pixel")
    parser.add_argument("--blur-sigma-px", type=float, default=2.0)
    parser.add_argument("--opendrive-lanes", type=str, default="data/opendrive_lanes.json", help="Von extract_opendrive_lanes.py erzeugte Spur-Polygone (bevorzugt, falls vorhanden)")
    parser.add_argument("--out-image", type=str, default="frontend/map_background.png")
    parser.add_argument("--out-meta", type=str, default="frontend/map_background.json")
    args = parser.parse_args()

    frames = []
    for fname in args.raw_files:
        path = RAW_DIR / fname
        print(f"Lade {path}")
        df = load_raw_csv(path)
        df["agent_class"] = classify_vectorized(df)
        frames.append(df[["center_easting", "center_northing", "agent_class"]])
        print(f"  {len(df)} Zeilen")
    df = pd.concat(frames, ignore_index=True)

    pad = 5.0
    xmin, xmax = float(df["center_easting"].min() - pad), float(df["center_easting"].max() + pad)
    ymin, ymax = float(df["center_northing"].min() - pad), float(df["center_northing"].max() + pad)

    width_px = max(1, int((xmax - xmin) / args.resolution))
    height_px = max(1, int((ymax - ymin) / args.resolution))
    print(f"Rasterkarte: {width_px}x{height_px} px, Bereich {xmax - xmin:.0f}m x {ymax - ymin:.0f}m")

    opendrive_path = Path(args.opendrive_lanes)
    if opendrive_path.exists():
        print(f"Nutze echte OpenDRIVE-Fahrspurgeometrie: {opendrive_path}")
        with open(opendrive_path, encoding="utf-8") as f:
            polygons = json.load(f)
        img_out = render_opendrive_background(polygons, xmin, ymin, width_px, height_px, args.resolution)
    else:
        print(f"Keine OpenDRIVE-Extraktion gefunden ({opendrive_path}), falle auf Trajektorien-Dichte zurueck.")
        print("  (siehe scripts/extract_opendrive_lanes.py, um die echte Kartengeometrie zu nutzen)")
        xedges = np.linspace(xmin, xmax, width_px + 1)
        yedges = np.linspace(ymin, ymax, height_px + 1)
        img = np.zeros((height_px, width_px, 4), dtype=np.uint8)
        for label, color, max_alpha in [
            ("vehicle", VEHICLE_COLOR, 190),
            ("bicycle", BICYCLE_COLOR, 210),
            ("pedestrian", PEDESTRIAN_COLOR, 220),
        ]:
            mask = df["agent_class"] == label
            if not mask.any():
                continue
            density = density_grid(
                df.loc[mask, "center_easting"].to_numpy(),
                df.loc[mask, "center_northing"].to_numpy(),
                xedges,
                yedges,
                args.blur_sigma_px,
            )
            paint(img, density, color, max_alpha)
            print(f"  {label}: {mask.sum()} Punkte eingebrannt")
        img_out = Image.fromarray(img, mode="RGBA")

    out_image = Path(args.out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    img_out.save(out_image)

    meta = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"gespeichert: {out_image}")
    print(f"gespeichert: {args.out_meta}")


if __name__ == "__main__":
    main()
