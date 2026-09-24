"""Extrahiert reale Fahrspur-Polygone aus der offiziellen OpenDRIVE-Karte der
Braunschweiger Innenstadtring-Strassen (enthaelt die AIM-Forschungskreuzung),
beschraenkt auf den Kartenausschnitt der Demo-Szenen.

Quelle: M. Scholz, "OpenDRIVE dataset of the inner ring road in Brunswick", DLR,
2020, Zenodo DOI 10.5281/zenodo.4043193 (CC BY 4.0) -- referenziert in der
DLR-UT-Datensatzdokumentation, Abschnitt 9.2 "Additional Resources", ist aber
NICHT Teil des DLR-UT-Downloads selbst und muss separat geladen werden.

Verwendet die Bibliothek `pyxodr` (github.com/driskai/pyxodr) zum Parsen der
OpenDRIVE-Geometrie (Kurven, Breitenverlaeufe etc.), damit wir die komplexe
OpenDRIVE-Spec nicht von Hand nachbauen muessen.

Koordinatentransformation (siehe README der xodr-Karte): Die Karte nutzt eine
Variante von UTM 32N mit einem festen Versatz, um Fliesskomma-Ungenauigkeiten zu
vermeiden. Um auf echte UTM-32N-Koordinaten (wie im DLR-UT-Datensatz) zu kommen:
    x_utm = x_xodr + 604763.0
    y_utm = y_xodr + 5792795.0

Das volle Netz (46 MB, gesamter Innenstadtring von Braunschweig) enthaelt fuer
manche Strassen/Spuren degenerierte Geometrien, an denen pyxodr abbricht (siehe
GitHub-Issues von driskai/pyxodr). Wir laden deshalb Strasse fuer Strasse und
ueberspringen einzelne Strassen/Spuren, die einen Fehler werfen, statt das ganze
Netz auf einmal zu parsen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# pyxodr zeichnet bei bestimmten Geometrie-Fehlern (abgefangen, aber vor dem Raise)
# ungefragt eine Debug-Abbildung; unter Windows erschoepft das ohne Headless-Backend
# schnell die GDI-Ressourcen ("Fail to allocate bitmap"). Deshalb vor dem pyxodr-Import
# fest auf den nicht-interaktiven Agg-Backend umstellen.
matplotlib.use("Agg")

import numpy as np
from lxml import etree

from pyxodr.road_objects.road import Road

OFFSET_X = 604763.0
OFFSET_Y = 5792795.0

# Grobe Einteilung der OpenDRIVE-Spurtypen in die drei Klassen, die das Frontend
# schon kennt (vehicle/bicycle/pedestrian, siehe src/datapipeline/schema.py).
LANE_TYPE_GROUP = {
    "driving": "vehicle",
    "parking": "vehicle",
    "shoulder": "vehicle",
    "bidirectional": "vehicle",
    "biking": "bicycle",
    "sidewalk": "pedestrian",
    "border": "pedestrian",
    "curb": "pedestrian",
}


def reference_line_bbox_overlaps(ref_line: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = bbox
    rx_min, ry_min = ref_line[:, 0].min(), ref_line[:, 1].min()
    rx_max, ry_max = ref_line[:, 0].max(), ref_line[:, 1].max()
    return not (rx_max < xmin or rx_min > xmax or ry_max < ymin or ry_min > ymax)


def extract_lane_polygons(xodr_path: Path, bbox_utm: tuple[float, float, float, float], margin: float, resolution: float):
    bbox_raw = (
        bbox_utm[0] - OFFSET_X - margin,
        bbox_utm[1] - OFFSET_X + margin,
        bbox_utm[2] - OFFSET_Y - margin,
        bbox_utm[3] - OFFSET_Y + margin,
    )

    tree = etree.parse(str(xodr_path))
    road_elements = tree.getroot().findall("road")
    print(f"{len(road_elements)} Strassen im OpenDRIVE-Netz insgesamt")

    polygons = []
    n_candidate_roads = 0
    n_ok_roads = 0
    for road_xml in road_elements:
        # Vorfilter ueber die Referenzlinie, bevor wir die (teurere) Spurgeometrie
        # berechnen. Wichtig: mit der VOLLEN Aufloesung sampeln, nicht kuenstlich
        # vergroebert -- bei sehr wenigen Samples (z.B. nur 2 Punkte pro Bogen)
        # schlaegt pyxodrs interne Richtungs-Plausibilitaetspruefung fuer Arc-
        # Geometrien faelschlich an (der Sehnenvektor zwischen zwei weit auseinander
        # liegenden Samples weicht dann zu stark von der lokalen Tangente ab) und
        # die Strasse wuerde faelschlich komplett uebersprungen -- das war die
        # Ursache dafuer, dass an drei von vier Kreuzungsecken Fahrspuren fehlten.
        try:
            road = Road(road_xml, resolution=resolution)
            ref_line = road.reference_line
        except Exception:
            continue
        if len(ref_line) == 0 or not reference_line_bbox_overlaps(ref_line, bbox_raw):
            continue
        n_candidate_roads += 1

        try:
            road_ok = False
            for lane_section in road.lane_sections:
                try:
                    lanes = lane_section.lanes
                except Exception:
                    continue
                for lane in lanes:
                    if lane.id == 0 or lane.type is None:
                        continue
                    group = LANE_TYPE_GROUP.get(lane.type)
                    if group is None:
                        continue
                    try:
                        near = lane.lane_reference_line
                        far = lane.boundary_line
                    except Exception:
                        continue
                    if len(near) < 2 or len(far) < 2:
                        continue
                    ring = np.concatenate([near[:, :2], far[::-1, :2]], axis=0)
                    ring_utm = ring + np.array([OFFSET_X, OFFSET_Y])
                    polygons.append({"type": group, "lane_type": lane.type, "points": ring_utm.round(2).tolist()})
                    road_ok = True
            if road_ok:
                n_ok_roads += 1
        except Exception as exc:  # noqa: BLE001 - einzelne kaputte Strassen sollen die Extraktion nicht stoppen
            print(f"  Strasse {road_xml.get('id')} uebersprungen: {exc.__class__.__name__}: {exc}")
            continue

    print(f"{n_candidate_roads} Strassen im Kartenausschnitt, {n_ok_roads} davon erfolgreich extrahiert")
    print(f"{len(polygons)} Spur-Polygone insgesamt")
    return polygons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xodr", type=str, default="data/raw/opendrive/bs-inner-ring-road-v1.0.0/bs-inner-ring-road.xodr")
    parser.add_argument("--bounds-meta", type=str, default="frontend/map_background.json", help="JSON mit xmin/xmax/ymin/ymax (UTM) der Demo-Szenen")
    parser.add_argument("--margin", type=float, default=80.0, help="Zusaetzlicher Rand in Metern um den Szenen-Kartenausschnitt")
    parser.add_argument("--resolution", type=float, default=0.5, help="Meter pro Sample entlang der Spurgeometrie")
    parser.add_argument("--out", type=str, default="data/opendrive_lanes.json")
    args = parser.parse_args()

    with open(args.bounds_meta, encoding="utf-8") as f:
        meta = json.load(f)
    bbox_utm = (meta["xmin"], meta["xmax"], meta["ymin"], meta["ymax"])

    polygons = extract_lane_polygons(Path(args.xodr), bbox_utm, args.margin, args.resolution)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(polygons, f)
    print(f"gespeichert: {out_path}")


if __name__ == "__main__":
    main()
