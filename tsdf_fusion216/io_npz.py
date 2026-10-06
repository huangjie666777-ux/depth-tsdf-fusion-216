"""Loading and validation of uploaded NPZ depth sequences."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .errors import InputValidationError

MAX_FRAMES = 16
MAX_HEIGHT = 128
MAX_WIDTH = 128
MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024  # 8 MiB


@dataclass
class DepthSequence:
    """Validated multi-view depth input.

    depths: (N, H, W) float64, metres, camera z, 0 means missing.
    K: (3, 3) shared pinhole intrinsics.
    Tcw: (N, 4, 4) camera-to-world rigid transforms.
    """

    depths: np.ndarray
    K: np.ndarray
    Tcw: np.ndarray


def _as_numeric(name: str, arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        raise InputValidationError(f"{name}: object arrays are not allowed")
    if not np.issubdtype(arr.dtype, np.number):
        raise InputValidationError(f"{name}: dtype {arr.dtype} is not numeric")
    return arr


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise InputValidationError(f"{name}: contains non-finite values")


def _validate_intrinsics(K: np.ndarray) -> None:
    if K.shape != (3, 3):
        raise InputValidationError(f"K must have shape (3, 3), got {K.shape}")
    _check_finite("K", K)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if fx <= 0 or fy <= 0:
        raise InputValidationError("K: focal lengths fx and fy must be positive")
    if not (0.0 <= cx < MAX_WIDTH) or not (0.0 <= cy < MAX_HEIGHT):
        raise InputValidationError("K: principal point outside the image bounds")
    if abs(K[0, 1]) > 1e-9:
        raise InputValidationError("K: skew must be zero")
    if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise InputValidationError("K: last row must be [0, 0, 1]")
    if abs(K[1, 0]) > 1e-9 or abs(K[2, 0]) > 1e-9 or abs(K[2, 1]) > 1e-9:
        raise InputValidationError("K: entries below the diagonal must be zero")


def _validate_poses(Tcw: np.ndarray, n: int) -> None:
    if Tcw.shape != (n, 4, 4):
        raise InputValidationError(
            f"Tcw must have shape ({n}, 4, 4), got {Tcw.shape}"
        )
    _check_finite("Tcw", Tcw)
    bottom = Tcw[:, 3, :]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise InputValidationError("Tcw: last row must be [0, 0, 0, 1]")
    R = Tcw[:, :3, :3]
    eye = np.eye(3)
    for i in range(n):
        if not np.allclose(R[i].T @ R[i], eye, atol=1e-6):
            raise InputValidationError(f"Tcw[{i}]: rotation block is not orthonormal")
        det = float(np.linalg.det(R[i]))
        if abs(det - 1.0) > 1e-6:
            raise InputValidationError(f"Tcw[{i}]: rotation determinant must be +1")


def load_depth_sequence(data: bytes) -> DepthSequence:
    """Parse and validate an uploaded NPZ payload."""
    try:
        archive = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise InputValidationError(f"cannot read NPZ archive: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise InputValidationError("uploaded file is not an NPZ archive")

    names = set(archive.files)
    required = {"depths", "K", "Tcw"}
    missing = required - names
    if missing:
        raise InputValidationError(f"NPZ missing arrays: {sorted(missing)}")

    total = 0
    arrays = {}
    for name in required:
        try:
            arr = archive[name]
        except Exception as exc:
            raise InputValidationError(f"cannot read array {name}: {exc}") from exc
        arr = _as_numeric(name, arr)
        total += arr.nbytes
        if total > MAX_DECOMPRESSED_BYTES:
            raise InputValidationError(
                "decompressed arrays exceed the 8 MiB limit"
            )
        arrays[name] = arr

    depths = np.asarray(arrays["depths"], dtype=np.float64)
    if depths.ndim != 3:
        raise InputValidationError(
            f"depths must have shape (N, H, W), got {depths.shape}"
        )
    n, h, w = depths.shape
    if n < 1 or n > MAX_FRAMES:
        raise InputValidationError(f"frame count N must be in [1, {MAX_FRAMES}]")
    if h < 1 or h > MAX_HEIGHT or w < 1 or w > MAX_WIDTH:
        raise InputValidationError(
            f"image size HxW must be within [1, {MAX_HEIGHT}]x[1, {MAX_WIDTH}]"
        )
    _check_finite("depths", depths)
    if np.any(depths < 0.0):
        raise InputValidationError("depths: negative values are not allowed")

    K = np.asarray(arrays["K"], dtype=np.float64)
    _validate_intrinsics(K)

    Tcw = np.asarray(arrays["Tcw"], dtype=np.float64)
    _validate_poses(Tcw, n)

    return DepthSequence(depths=depths, K=K, Tcw=Tcw)
