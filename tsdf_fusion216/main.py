"""FastAPI application exposing the TSDF fusion endpoint."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .errors import InputValidationError
from .io_npz import load_depth_sequence
from .packaging import build_zip
from .pipeline import run_fusion
from .schemas import parse_params

app = FastAPI(title="tsdf_fusion216", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(..., description="NPZ with depths, K, Tcw"),
    params: str = Form(..., description="JSON volume parameters"),
) -> Response:
    try:
        fusion_params = parse_params(params)
        payload = await file.read()
        seq = load_depth_sequence(payload)
        result = run_fusion(seq, fusion_params)
        archive = build_zip(result, fusion_params)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="reconstruction.zip"'},
    )
