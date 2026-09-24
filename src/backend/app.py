"""FastAPI-Service fuer die Demo (siehe plan.md Abschnitt 5).

Startet mit:
    uvicorn src.backend.app:app --reload --port 8000

GENERATOR_MODE=mock (Standard, Phase A) oder GENERATOR_MODE=trained (Phase B,
braucht einen Checkpoint) steuert, welcher VariantGenerator injiziert wird --
diese Endpunkte selbst aendern sich dabei nicht.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.backend.config import GENERATOR_MODE, SCENES_DIR, build_generator
from src.backend.scene_repo import SceneRepository

app = FastAPI(title="Traffic World Model Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

scene_repo = SceneRepository(SCENES_DIR)
generator = build_generator()


class GenerateRequest(BaseModel):
    n_variants: int = 3


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "generator_mode": GENERATOR_MODE, "n_scenes": len(scene_repo.list_summaries())}


@app.get("/api/scenes")
def list_scenes() -> list[dict]:
    return scene_repo.list_summaries()


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict:
    scene = scene_repo.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Szene '{scene_id}' nicht gefunden")
    return scene.to_dict()


@app.post("/api/scenes/{scene_id}/generate")
def generate_variants(scene_id: str, req: GenerateRequest) -> dict:
    scene = scene_repo.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Szene '{scene_id}' nicht gefunden")
    if not 1 <= req.n_variants <= 8:
        raise HTTPException(status_code=400, detail="n_variants muss zwischen 1 und 8 liegen")

    variants = generator.generate(scene, req.n_variants)
    return {
        "scene_id": scene_id,
        "generator_mode": GENERATOR_MODE,
        "variants": [v.to_dict() for v in variants],
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
