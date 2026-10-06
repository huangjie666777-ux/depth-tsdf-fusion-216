from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FusionParameters:
    origin: tuple[float, float, float]
    dims: tuple[int, int, int]
    voxel_size: float
    trunc: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FusionParameters":
        required = {"origin", "dims", "voxel_size", "trunc"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"missing volume parameters: {sorted(missing)}")

        origin = np.asarray(payload["origin"], dtype=np.float64)
        dims = np.asarray(payload["dims"])
        if origin.shape != (3,):
            raise ValueError("origin must contain exactly three metric coordinates")
        if dims.shape != (3,):
            raise ValueError("dims must contain exactly three integer sizes")
        if not np.all(np.isfinite(origin)):
            raise ValueError("origin values must be finite")
        if dims.dtype.kind not in "iu":
            raise ValueError("dims must be integers")

        dims_i64 = dims.astype(np.int64)
        if np.any(dims_i64 < 2) or np.any(dims_i64 > 64):
            raise ValueError("each axis dimension must be between 2 and 64")

        try:
            voxel_size = float(payload["voxel_size"])
            trunc = float(payload["trunc"])
        except (TypeError, ValueError) as exc:
            raise ValueError("voxel_size and trunc must be numbers") from exc
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size must be a positive finite metric value")
        if not np.isfinite(trunc) or trunc <= 0.0:
            raise ValueError("trunc must be a positive finite metric value")

        return cls(
            origin=tuple(float(x) for x in origin),
            dims=tuple(int(x) for x in dims_i64),
            voxel_size=voxel_size,
            trunc=trunc,
        )


@dataclass(frozen=True)
class DepthInput:
    depths: NDArray[np.float64]
    camera_matrix: NDArray[np.float64]
    world_from_camera: NDArray[np.float64]


@dataclass(frozen=True)
class FusedVolume:
    tsdf: NDArray[np.float64]
    weights: NDArray[np.int32]
    parameters: FusionParameters


@dataclass(frozen=True)
class FusionResult:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
