# DLR-UT World-Model-Demo — Umsetzungsplan (Kurzsprint)

Stand: September 2026. Ziel: In wenigen Tagen eine funktionierende, interaktive Demo bauen, die anhand des öffentlichen **DLR-UT-Datensatzes** (AIM-Forschungskreuzung Braunschweig) den Mehrwert eines generativen "Traffic World Models" zeigt — nutzbar sowohl für Förderanträge/Gutachter als auch für Industriepartner-Gespräche.

## ⚠️ Anweisung für die Umsetzung mit Claude Code (zuerst lesen)

Dieses Projekt wird in **zwei getrennten Phasen auf zwei verschiedenen Maschinen** umgesetzt:

- **Phase A — Laptop, kein GPU:** Komplette Codebasis bauen (Datenpipeline, Modell-Code, FastAPI-Backend, Browser-UI) und über eine **Mock-Inferenz** end-to-end lauffähig und im Browser testbar machen. **Kein echtes Modelltraining ausführen.**
- **Phase B — VM mit GPU (8× V100):** Erst hier findet das eigentliche Training auf den vollständigen echten Daten statt.

**Wenn du (Claude Code) diese `plan.md` auf dem Laptop ausgeführt bekommst, gilt automatisch: Du befindest dich in Phase A.** Bearbeite ausschließlich die Aufgaben unter "Phase A" in Abschnitt 6. Erlaubt ist ein kurzer **Sanity-Check** des Trainingsskripts mit winzigen Dummy-Daten (wenige Zeilen, 1–2 Iterationen, CPU, nur um Code-Fehler zu finden) — das ist **kein** echtes Training und läuft in Sekunden. Nicht erlaubt: ein mehrepochiges Training auf den echten DLR-UT-Daten, das Herunterladen/Installieren von GPU-/CUDA-Paketen, oder der Versuch, die Trainingsschleife "einfach mal laufen zu lassen, um zu sehen was passiert". Wenn am Ende von Phase A alle Definition-of-Done-Punkte aus Abschnitt 6 erfüllt sind: **anhalten und dem Nutzer Bescheid geben, dass das Repo bereit zum Klonen auf die GPU-VM ist.** Führe Phase B nicht selbstständig aus, auch nicht testweise.

Erkennungsmerkmal, auf welcher Maschine du läufst, falls unklar: `nvidia-smi` bzw. `torch.cuda.is_available()` prüfen. Kein GPU erkannt → Phase A. GPU erkannt (die 8×V100-VM) → Phase B, siehe Abschnitt 6.

## 0. Zielsetzung & bewusster Scope

**Was die Demo zeigt:** Aus einer kurz beobachteten realen Verkehrssituation an der AIM-Kreuzung (ein paar Sekunden echte Trajektorien aus DLR-UT) erzeugt ein generatives, auf euren Daten trainiertes World Model mehrere plausible, physikalisch konsistente Fortsetzungen der Szene — sichtbar gemacht als interaktive Bird's-Eye-View-Kartenanimation, die "Real" gegen "vom Modell generierte Varianten" stellt. Ein Stretch-Ziel zeigt zusätzlich steuerbare Generierung ("was wäre, wenn Fußgänger X trotzdem quert / Fahrzeug Y stärker bremst").

**Bewusst NICHT Teil dieses Sprints:** photorealistische Videogenerierung (Cosmos-Klasse Pixel-World-Models) als Kernstück. Grund: Diese Modelle brauchen zwingend Ampere-oder-neuer-GPUs (eure 8×V100 reichen dafür laut Hersteller-Doku nicht aus) und sind für einen Wenige-Tage-Sprint zu aufwändig. Stattdessen: ein **Trajektorien-World-Model** — eine ebenso aktuelle, aber deutlich leichtgewichtigere Modellklasse, die auf V100 ohne Einschränkung läuft. Photorealistische Clips sind unten als optionales Stretch-Goal über eine gehostete API skizziert, nicht als Kernabhängigkeit.

## 1. Warum dieser Scope die richtige Wahl für "wenige Tage" ist

- **Nur DLR-UT, nicht DLR-HT:** Die Kreuzung bietet dichte Multi-Agent-Interaktion (Kfz, Rad, Fuß) auf engem Raum — visuell und inhaltlich pointierter für eine Kurz-Demo als reine Autobahn-Längstrajektorien, und mit direktem Bezug zur VRU-Sicherheit (Fußgänger/Radfahrer), einem Thema mit hoher Förder- und Industrierelevanz.
- **Trajektorien- statt Pixel-World-Model:** Generative Diffusionsmodelle über Trajektorien-Sequenzen (Methodik-Linie: NVIDIAs eigenes [NVlabs/CTG](https://github.com/NVlabs/CTG) "Guided Conditional Diffusion for Controllable Traffic Simulation", sowie Trajectron++/Trajeglish als verwandte Ansätze) sind selbst anerkannte, zitierte State-of-the-Art-Forschung zu "Traffic World Models" — nur eben auf Trajektorien- statt auf Pixelebene. Sie sind klein (typischerweise 1–10 Mio. Parameter), brauchen keine Ampere-Architektur und lassen sich in Stunden statt Tagen trainieren.
- **BEV-Kartenanimation als UI:** robust, schnell baubar, in Sekunden verständlich für Laien (Förder-Gutachter, Industriemanager) UND aussagekräftig genug für Fachpublikum.
- **Laptop/VM-Trennung:** Der komplette Code (Pipeline, Modell, Backend, UI) lässt sich ohne jede GPU schreiben und über eine Mock-Inferenz vollständig im Browser testen — nur der eigentliche Trainingslauf braucht die V100-VM. Das entkoppelt "Code fertig & UI sichtbar" von "Modell trainiert" und macht Phase A komplett unabhängig von VM-Verfügbarkeit.

## 2. Technischer Ansatz: Modellwahl

**Referenzmethode:** [NVlabs/CTG](https://github.com/NVlabs/CTG) (NVIDIA Research) dient als methodisches Vorbild — Diffusion über kurze Multi-Agent-Trajektorien, konditioniert auf Beobachtungshistorie und optional steuerbar über Guidance-Terme (z.B. Kollisionsvermeidung, Zieleinhaltung).

**Wichtige Entscheidung:** Das volle CTG-Repo 1:1 zu portieren ist für einen Wenige-Tage-Sprint ein vermeidbares Risiko — es ist fest auf nuScenes zugeschnitten (Kartenformat, `trajdata`-Abhängigkeit), nutzt alte, gepinnte Versionen (PyTorch 1.11, CUDA 11.3) und bringt Zusatzkomplexität (STL-Guidance, optionale ChatGPT-Anbindung) mit, die ihr nicht braucht.

**Empfehlung:** Eine **schlanke Eigenimplementierung nach demselben Prinzip**, von Anfang an auf das DLR-UT-Format zugeschnitten — mit klarem Verweis auf CTG/Trajectron++/Trajeglish als methodische Vorbilder in Doku und Pitch (wissenschaftliche Einordnung bleibt erhalten, ohne das Integrationsrisiko der fremden Codebasis). Kernkomponenten:

- Feature-Pipeline: kurze Zeitfenster (Beobachtungshistorie + Vorhersagehorizont) je Interaktionsszene aus DLR-UT extrahieren
- Modell: Transformer- oder 1D-Conv-Backbone + Diffusions-Sampling-Kopf über die zukünftigen Positionsdeltas aller Agenten in der Szene gemeinsam (nicht pro Agent isoliert — das ist der Kern dessen, was "World Model" statt Einzel-Trajektorien-Prädiktor bedeutet: gemeinsame, konsistente Fortsetzung der ganzen Szene)
- Konditionierung optional erweiterbar um einfache Guidance-Terme (Kollisionsvermeidung) für das Stretch-Goal "steuerbare Generierung"

**Wichtig für Phase A:** Dieses Modell wird in Phase A vollständig als Code geschrieben (Architektur, Trainingsskript, Sampling-Funktion) — es wird nur nicht auf den echten Daten trainiert. Siehe Abschnitt 5 für die austauschbare Inferenz-Schnittstelle, mit der die App trotzdem im Browser läuft.

## 3. Compute-Bedarf — konkret, je Phase

| Schritt | Phase | Ressource | Aufwand |
|---|---|---|---|
| Datenexploration & Feature-Pipeline | A (Laptop) | CPU, keine GPU | Minuten bis Stunden |
| Modell-, Backend- und UI-Code schreiben | A (Laptop) | CPU, keine GPU | Hauptteil des Sprints |
| Sanity-Check Trainingsskript (Dummy-Daten, 1–2 Iterationen) | A (Laptop) | CPU, keine GPU | Sekunden |
| App im Browser testen (Mock-Inferenz) | A (Laptop) | CPU, keine GPU | laufend während der Entwicklung |
| **Echtes Modelltraining** (Trajektorien-Diffusionsmodell) | **B (VM)** | **1× V100** (16 oder 32GB reichen) | geschätzt 2–6 Stunden, abhängig von Datenmenge/Epochenzahl — von den 8 verfügbaren GPUs wird nur eine gebraucht |
| Inferenz/Sampling für die Live-Demo mit echtem Modell | B (VM, danach auch wieder lokal möglich) | 1× V100 (oder CPU für einzelne Samples) | Sekunden bis wenige Minuten pro generierter Variante |
| **Optionales Stretch-Goal:** 1–2 photorealistische Vorschau-Clips | A oder B (egal) | **Keine eigene GPU** — nur gehostete Cosmos-API (z.B. NVIDIA build.nvidia.com), Verfügbarkeit/Kosten bei Umsetzung prüfen | Internetzugang + API-Zugang |

**Fazit:** Phase A braucht keinerlei GPU. Phase B braucht nur eine einzige der 8 V100-GPUs für wenige Stunden. Kein Cloud-GPU-Rental nötig, außer für das optionale Stretch-Goal (dort ohnehin keine eigene GPU, nur API-Zugriff).

## 4. Datenbasis

- Nur **DLR-UT** ([Zenodo](https://zenodo.org/records/13907201)), AIM-Forschungskreuzung Braunschweig.
- Laden über [TASI](https://github.com/DLR-TS/TASI):
  ```python
  from tasi.dlr import DLRUTDatasetManager, DLRUTVersion, DLRTrajectoryDataset
  dataset = DLRUTDatasetManager(DLRUTVersion.latest)
  path = dataset.load()
  ds = DLRTrajectoryDataset.from_csv(path)
  ```
  Ergebnis: Dataframe mit Multi-Index (Zeitstempel, Objekt-ID); Felder umfassen laut TASI-Doku Bounding-Box-Position, Geschwindigkeit, Dimension und Klassifikationstyp (Kfz/Rad/Fuß). Exakte Spaltennamen/Einheiten/Koordinatensystem in **Phase A, Tag 1** anhand der TASI-Beispiel-Notebooks verifizieren, nicht blind annehmen. Das Herunterladen des DLR-UT-Datensatzes selbst braucht keine GPU und ist Teil von Phase A.
- **Zu klären in Phase A:** Liegt für die AIM-Kreuzung bereits eine Kartengeometrie vor (Fahrspuren, Kreuzungsumriss, z.B. als OpenDRIVE)? Falls nicht: als Fallback aus den Trajektoriendaten selbst grob rekonstruieren (siehe Risiken unten).
- **Zu klären in Phase A:** Wie viele Trajektorien/Interaktionsszenen enthält der eine veröffentlichte Zenodo-Tag tatsächlich? Das bestimmt, ob die Datenmenge für ein robustes Modelltraining reicht oder ob Augmentierung nötig ist (siehe Risiken).

## 5. Architektur der Demo — inkl. austauschbarer Inferenz-Schnittstelle

- **Backend:** Python-Trainings-/Inferenzskripte; für die Live-Demo ein kleiner lokaler Service (z.B. FastAPI), der auf Anfrage neue Varianten sampelt und als JSON (Trajektorien-Koordinaten über Zeit) an die UI liefert.
- **Kernidee für den Phasenübergang — eine einzige austauschbare Schnittstelle:**
  ```python
  class VariantGenerator(ABC):
      def generate(self, scene_context: SceneContext, n_variants: int) -> list[Trajectory]:
          ...

  class MockVariantGenerator(VariantGenerator):
      # Phase A: erzeugt plausible Dummy-Varianten (z.B. leicht verrauschte
      # Kopien der realen Trajektorie), gleiche Rückgabestruktur wie das echte Modell
      ...

  class TrainedModelVariantGenerator(VariantGenerator):
      # Phase B: lädt den trainierten Checkpoint und sampelt echte Varianten
      ...
  ```
  Der FastAPI-Service bekommt die Implementierung über eine Config/Umgebungsvariable injiziert (z.B. `GENERATOR_MODE=mock` vs. `GENERATOR_MODE=trained`). So bleibt Backend und UI beim Wechsel von Phase A zu Phase B **unverändert** — es wird nur diese eine Zeile umgestellt, sobald auf der VM ein Checkpoint existiert.
- **Frontend:** kleine, selbstständige Web-UI mit:
  - Zeitstrahl/Play-Button: reale Szene abspielen (Kreuzungsumriss + bewegte Punkte/Icons je Objektklasse)
  - "Generiere Variante"-Button: ruft `VariantGenerator.generate()` auf (in Phase A den Mock, in Phase B das echte Modell) — für die UI ist das nicht unterscheidbar
  - **Stretch:** einfacher Interaktionsregler ("Fußgänger quert trotzdem" / "Fahrzeug bremst stärker") als steuerbare Generierung
  - **Stretch:** eingebetteter kurzer generierter Videoclip (Cosmos-API, siehe Abschnitt 3) als visueller Ausblick auf die größere Vision aus dem Hauptplan
- **Abhängigkeiten sauber trennen:** `requirements-cpu.txt` (Laptop, CPU-PyTorch, für Phase A ausreichend) und `requirements-gpu.txt` (VM, CUDA-PyTorch, für Phase B) als getrennte Dateien pflegen, damit das Repo auf beiden Maschinen ohne Anpassung installierbar ist.

## 6. Sprintplan mit Phasen-Trennung (Beispiel: 5 Arbeitstage, auf 3 komprimierbar)

### Phase A — Laptop, kein GPU (Tag 1–4)

1. **Tag 1 — Daten:** TASI-Setup, Datenexploration, Schema-/Koordinatensystem-Klärung, Extraktion kurzer Beobachtungs-/Vorhersagefenster je Interaktionsszene, Kartenumriss klären/rekonstruieren.
2. **Tag 2 — Modell-Code:** Implementierung des Diffusions-Trajektorienmodells (Architektur + Trainingsskript + Sampling-Funktion) und der `MockVariantGenerator`-Implementierung. Sanity-Check des Trainingsskripts mit winzigen Dummy-Daten (1–2 Iterationen, CPU) — **kein echtes Training**.
3. **Tag 3 — Backend & UI:** FastAPI-Service mit austauschbarer `VariantGenerator`-Schnittstelle (Abschnitt 5), BEV-Animation (Play/Compare/Generate), Anbindung Backend → UI, End-to-End-Test mit `GENERATOR_MODE=mock`.
4. **Tag 4 — Politur & Übergabe:** `requirements-cpu.txt` / `requirements-gpu.txt` sauberziehen, README mit VM-Anleitung schreiben (siehe Definition of Done unten), 2–3 Szenen für die spätere Demo kuratieren, Stretch-Goal-UI-Elemente (Regler, Video-Slot) vorbereiten (auch wenn sie mangels echtem Modell/API noch nicht "scharf" sind).

**Definition of Done für Phase A** (danach anhalten und Bescheid geben, nicht selbst weitermachen):
- App läuft lokal im Browser, Play/Compare/Generate funktionieren vollständig mit `MockVariantGenerator`
- Trainingsskript ist geschrieben und per Dummy-Sanity-Check als fehlerfrei bestätigt, aber **nicht real trainiert**
- `requirements-gpu.txt` ist vorbereitet, aber nicht auf dem Laptop installiert/getestet (keine GPU vorhanden)
- README enthält eine kurze "VM-Anleitung": Repo klonen → `pip install -r requirements-gpu.txt` → echtes Training starten → `GENERATOR_MODE=trained` setzen

### Phase B — VM mit 8× V100 (Tag 5, oder später)

5. **Tag 5 — Training & Feinschliff:** Repo auf die VM klonen, `requirements-gpu.txt` installieren, echtes Training auf einer V100 (siehe Abschnitt 3), `GENERATOR_MODE=trained` setzen, quantitative Plausibilitätschecks (Kollisionsrate, Geschwindigkeits-/Beschleunigungsverteilung generiert vs. real), finale Pitch-Vorbereitung mit echten (statt Mock-)Ergebnissen.

**Falls nur 3 Tage für Phase A bleiben:** Tag 1+2 zusammenlegen (kleineres, einfacheres Modell), Stretch-Goals (Interaktionsregler, Video-Clip) komplett streichen, UI auf "Real abspielen + eine (Mock-)Variante daneben zeigen" reduzieren. Das ist immer noch eine überzeugende, im Browser lauffähige Kernaussage, bevor überhaupt auf die VM gewechselt wird.

## 7. Was die Demo beweisen soll

- **Für Förderantrag/Gutachter:** technischer Nachweis, dass aus echten, an eurer eigenen Infrastruktur erhobenen Daten ein generatives Traffic-World-Model trainiert werden kann, das plausible, physikalisch konsistente Szenarienvarianten erzeugt — als belastbares Fundament, auf dem sich der größere Fahrzeugperspektiven-/Cosmos-Ausbau (siehe Hauptplan) aufbauen lässt.
- **Für Industriepartner-Gespräche:** eine anfassbare, in Minuten verständliche Demonstration des Grundprinzips "aus echten Infrastrukturdaten lernen, kontrollierbare synthetische Szenarienvarianten für Test-/Trainingszwecke erzeugen" — ohne dass der komplette Pixel-Video-Stack nötig ist, um den Wert zu vermitteln.

## 8. Risiken & Fallbacks

- **Datenmenge:** Nur ein Zenodo-Tag könnte für ein robustes Diffusionsmodell knapp sein. Fallback: kleineres Modell, stärkere Datenaugmentierung (Spiegelung an der Kreuzungsachse, Zeitverschiebung, Ausschnitts-Sampling), oder — falls schnell zugänglich — Ergänzung durch einen Teil eurer internen Zusatzdaten.
- **Kartengeometrie:** Falls keine saubere HD-Map/OpenDRIVE-Datei für die Kreuzung vorliegt: Kreuzungsumriss/Fahrspuren grob aus der Dichte der realen Trajektorien selbst schätzen (z.B. konvexe Hülle je Bewegungsrichtung) statt auf eine exakte Karte zu warten.
- **Zeitdruck:** MVP zuerst sicherstellen (reale vs. eine generierte Variante, auch als statischer Plot reicht im Notfall), Interaktivität und Stretch-Goals nur ergänzen, wenn Zeit bleibt.
- **Stretch-Goal-API:** Falls kein nutzbares Cosmos-API-Kontingent gefunden wird oder das Setup zu viel Zeit frisst: Stretch-Goal ersatzlos streichen, ohne den Kern der Demo zu gefährden — es ist bewusst optional gehalten.
- **Phasenvermischung:** Falls eine Claude-Code-Instanz versehentlich versucht, in Phase A doch zu trainieren (z.B. weil "es ja auch auf CPU ginge, nur langsam") — das explizit abbrechen. Selbst ein technisch mögliches CPU-Training auf den echten Daten würde Tage statt Stunden dauern und den Sinn der Phasentrennung unterlaufen.

## 9. Nächste Schritte nach der Demo

Diese Demo ist der erste, risikoarme Baustein des größeren Plans (siehe vorheriges Dokument `DLR_UT_HT_WorldModel_UseCases.md`, Abschnitte 5–9): DLR-HT einbeziehen, die Kameraperspektiven und den Infrastruktur-zu-Fahrzeug-Ansatz (I2V-GS-Linie) angehen, und die Ergebnisse als Grundlage für einen Förderantrag (z.B. Horizon-CCAM-Call, mFUND) bzw. konkrete Industriegespräche nutzen.

---
Quellen: [Zenodo DLR-UT](https://zenodo.org/records/13907201) · [NVlabs/CTG](https://github.com/NVlabs/CTG) · [DLR-TS/TASI GitHub](https://github.com/DLR-TS/TASI) · [TASI auf PyPI](https://pypi.org/project/tasi/0.15.4/) · [Cosmos Transfer2.5 System Requirements](https://deepwiki.com/nvidia-cosmos/cosmos-transfer2.5/3.1-system-requirements)
