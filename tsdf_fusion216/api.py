from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .delivery import build_result_zip
from .fusion import fuse_depth_frames
from .input_validation import InvalidDepthData, load_depth_npz
from .models import FusionParameters
from .surface import extract_zero_surface

app = FastAPI(title="TSDF Fusion 216", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/fuse")
async def fuse(
    file: UploadFile = File(...),
    params: str = Form(...),
) -> Response:
    raw_npz = await file.read()
    try:
        payload: dict[str, Any] = json.loads(params)
        if not isinstance(payload, dict):
            raise ValueError("params must be a JSON object")
        fusion_params = FusionParameters.from_payload(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        depth_input = load_depth_npz(raw_npz)
        volume = fuse_depth_frames(depth_input, fusion_params)
        mesh = extract_zero_surface(volume)
        result_zip = build_result_zip(volume, mesh)
    except InvalidDepthData as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=result_zip,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="tsdf_fusion_216.zip"'},
    )
