"""ZIP delivery: PLY mesh, TSDF/weight NPZ, and run metadata JSON."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from .pipeline import FusionResult
from .schemas import FusionParams
from .surface import mesh_to_ply


def build_zip(result: FusionResult, params: FusionParams) -> bytes:
    npz_buf = io.BytesIO()
    np.savez(
        npz_buf,
        tsdf=result.tsdf.astype(np.float32),
        weight=result.weight.astype(np.float32),
    )

    meta = {
        "origin": list(params.origin),
        "dims": list(params.dims),
        "voxel_size": params.voxel_size,
        "trunc": params.trunc,
        "array_axis_order": ["z", "y", "x"],
        "units": "metres",
        "observed_voxels": result.observed_voxels,
        "num_vertices": int(len(result.vertices)),
        "num_faces": int(len(result.faces)),
    }

    ply = mesh_to_ply(result.vertices, result.faces)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mesh.ply", ply)
        zf.writestr("tsdf_weight.npz", npz_buf.getvalue())
        zf.writestr("run.json", json.dumps(meta, indent=2))
    return buf.getvalue()
