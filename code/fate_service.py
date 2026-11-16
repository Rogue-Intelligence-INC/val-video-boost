#!/usr/bin/env python3
"""Fate consistency service -- the product API surface.

Wraps FateInterface + the consolidated pipeline in HTTP so any client
(web UI, editing tool, another agent) can use the stack without
importing our code:

  GET  /health            service + model/config status
  GET  /v1/route          framework route for a task (measured records)
  POST /v1/supply         {scene|scene_desc} -> pinned prompt + fingerprint
  POST /v1/enroll         {image_path} -> designate the anchor face
  POST /v1/verify         {image_path} -> {cos, passed, abstained}
  POST /v1/select         {pool: [paths]} -> chosen + all scores
  POST /v1/run            {scenes, seeds, k, hero?} -> full pipeline
                          manifest summary (synchronous; minutes-scale)

Paths in requests are server-side filesystem paths (the service is a
local/edge component next to the GPU, not a public upload API).
If FATE_API_KEY is set, every /v1 call needs header X-API-Key.
Run:  python fate_service.py        (binds FATE_HOST:FATE_PORT)
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings
from character_loader import list_characters, load_character

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

import adaptive_framework as af
from fate_interface import FateInterface

app = FastAPI(title="Valhalla Fate Consistency Service", version="1.0.0")
_fate = FateInterface(tau=settings.TAU)
_pipe = None
_lock = threading.Lock()


def _pipe_get():
    global _pipe
    if _pipe is None:
        import download
        from test_big_models import load_pipe
        _pipe = load_pipe(download.IMAGE_JUG_DIR)
    return _pipe


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if settings.API_KEY and request.url.path.startswith("/v1"):
        if request.headers.get("x-api-key") != settings.API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "invalid or missing X-API-Key"}, status_code=401)
    return await call_next(request)


class SupplyReq(BaseModel):
    scene: str | None = None
    scene_desc: str | None = None
    character_id: str | None = None


class PathReq(BaseModel):
    image_path: str


class SelectReq(BaseModel):
    pool: list[str]


class RunReq(BaseModel):
    scenes: list[str] = ["park", "harbour"]
    seeds: list[int] = [7]
    k: int = 4
    hero: str | None = None
    out: str | None = None
    character_id: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fate-consistency",
        "config": settings.describe(),
        "anchor_enrolled": _fate.anchor_path is not None,
        "anchor": str(_fate.anchor_path) if _fate.anchor_path else None,
        "available_characters": list_characters(),
        "current_character": _fate.character.id if _fate.character else None,
    }


@app.get("/v1/route")
def route(task: str = "identity_no_plugin"):
    return af.route(task, registry=af.load_measured_registry())


@app.post("/v1/supply")
def supply(req: SupplyReq):
    # Allow switching character for this request
    fate = _fate
    if req.character_id:
        fate = FateInterface(character_id=req.character_id, tau=settings.TAU)
    
    if req.scene_desc:
        return fate.supply_desc(req.scene_desc)
    return fate.supply(req.scene or "park")


@app.post("/v1/enroll")
def enroll(req: PathReq):
    ok = _fate.enroll_face(req.image_path)
    if not ok:
        raise HTTPException(422, "no face detectable in image; anchor not set")
    return {"enrolled": True, "anchor": str(_fate.anchor_path), "tau": _fate.tau}


@app.post("/v1/verify")
def verify(req: PathReq):
    if _fate.anchor_path is None:
        raise HTTPException(409, "no anchor enrolled; call /v1/enroll first")
    return _fate.verify(req.image_path)


@app.post("/v1/select")
def select(req: SelectReq):
    if _fate.anchor_path is None:
        raise HTTPException(409, "no anchor enrolled; call /v1/enroll first")
    with _lock:
        return _fate.select([Path(p) for p in req.pool])


@app.post("/v1/run")
def run(req: RunReq):
    import run_pipeline as rp

    with _lock:
        pipe = _pipe_get()
        out = Path(req.out) if req.out else settings.PIPELINE_OUT
        out.mkdir(parents=True, exist_ok=True)
        fate = FateInterface(tau=settings.TAU, character_id=req.character_id)
        
        # Use character's scenes or fallback to default scenes
        scene_text = fate.character.scenes if fate.character else rp.SCENE_TEXT
        
        hero_path = Path(req.hero) if req.hero else out / "hero_studio.png"
        if not req.hero:
            # Use first available scene or 'studio' for hero
            hero_scene = "studio" if "studio" in scene_text else list(scene_text.keys())[0]
            rp.gen(pipe, fate.supply(hero_scene), hero_path, req.seeds[0])
        if not fate.enroll_face(hero_path):
            raise HTTPException(422, "no face detectable in hero; run abstained")
        segments, chosen = [], [hero_path]
        seg = 0
        for seed in req.seeds:
            for scene in req.scenes:
                # Get scene description from character or use as-is
                if fate.character and fate.character.has_scene(scene):
                    scene_desc = fate.character.get_scene(scene)
                else:
                    scene_desc = scene_text.get(scene, scene)
                
                spec = fate.supply_desc(scene_desc)
                pool = []
                for k in range(req.k):
                    p = out / "pool" / f"seg{seg:02d}_s{seed}_{scene}_c{k}.png"
                    rp.gen(pipe, spec, p, 70000 + seg * 100 + k)
                    pool.append(p)
                picked = fate.select(pool)
                segments.append({
                    "segment": seg, "seed": seed, "scene": scene,
                    "fingerprint": spec["identity_fingerprint"],
                    "character_id": spec.get("character_id"),
                    "pool_scores": [s["cos"] for s in picked["scores"]],
                    "chosen": picked["chosen"], "chosen_cos": picked["chosen_cos"],
                    "status": "ABSTAINED" if picked["abstained"] else "ACCEPTED"
                })
                if picked["chosen"]:
                    chosen.append(Path(picked["chosen"]))
                seg += 1
        ok = [s["chosen_cos"] for s in segments if s["chosen_cos"] is not None]
        summary = {
            "segments": len(segments),
            "accepted": sum(1 for s in segments if s["status"] == "ACCEPTED"),
            "abstained": sum(1 for s in segments if s["status"] == "ABSTAINED"),
            "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
            "min_cos": min(ok) if ok else None,
            "character_id": fate.character.id if fate.character else None,
        }
        return {"anchor": str(hero_path), "summary": summary, "segments": segments}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
