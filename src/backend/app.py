"""FastAPI service for the demo (see plan.md section 5).

Start with:
    uvicorn src.backend.app:app --reload --port 8000

GENERATOR_MODE=mock (default, Phase A) or GENERATOR_MODE=trained (Phase B,
needs a checkpoint) controls which VariantGenerator gets injected -- these
endpoints themselves stay unchanged either way.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.backend.config import GENERATOR_MODE, SCENES_DIR, build_generator
from src.backend.scene_repo import SceneRepository

app = FastAPI(title="Traffic World Model Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def no_cache(request: Request, call_next):
    # StaticFiles has no cache-busting by default, and browsers happily serve a
    # stale frontend/app.js or map_background.png from disk cache across page
    # loads (bit us twice during development). This is a local demo, not a
    # production CDN target, so simply disable caching everywhere.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response

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
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
    return scene.to_dict()


@app.post("/api/scenes/{scene_id}/generate")
def generate_variants(scene_id: str, req: GenerateRequest) -> dict:
    scene = scene_repo.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
    if not 1 <= req.n_variants <= 8:
        raise HTTPException(status_code=400, detail="n_variants must be between 1 and 8")

    variants = generator.generate(scene, req.n_variants)
    return {
        "scene_id": scene_id,
        "generator_mode": GENERATOR_MODE,
        "variants": [v.to_dict() for v in variants],
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
