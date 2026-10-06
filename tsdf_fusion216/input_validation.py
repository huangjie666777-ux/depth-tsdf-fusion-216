from __future__ import annotations

import io
import zipfile

import numpy as np

from .models import DepthInput

MAX_FRAMES = 16
MAX_HEIGHT = 128
MAX_WIDTH = 128
MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024


class InvalidDepthData(ValueError):
    pass


def _finite_float_array(value: np.ndarray, name: str) -> np.ndarray:
    if value.dtype.kind == "O":
        raise InvalidDepthData(f"{name} must not be an object array")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidDepthData(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise InvalidDepthData(f"{name} contains a non-finite value")
    return result


def _validate_camera_matrix(k: np.ndarray) -> np.ndarray:
    k = _finite_float_array(k, "K")
    if k.shape != (3, 3):
        raise InvalidDepthData("K must have shape (3, 3)")
    if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise InvalidDepthData("K focal lengths must be positive")
    if not np.allclose(k[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-8):
        raise InvalidDepthData("K bottom row must be [0, 0, 1]")
    return k


def _validate_pose(pose: np.ndarray, index: int) -> np.ndarray:
    pose = _finite_float_array(pose, f"Tcw[{index}]")
    if pose.shape != (4, 4):
        raise InvalidDepthData(f"Tcw[{index}] must have shape (4, 4)")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8):
        raise InvalidDepthData(f"Tcw[{index}] bottom row must be [0, 0, 0, 1]")

    rotation = pose[:3, :3]
    identity = rotation @ rotation.T
    if not np.allclose(identity, np.eye(3), rtol=1e-6, atol=1e-6):
        raise InvalidDepthData(f"Tcw[{index}] rotation must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=1e-6, atol=1e-6):
        raise InvalidDepthData(f"Tcw[{index}] must be a right-handed rigid transform")
    return pose


def load_depth_npz(data: bytes) -> DepthInput:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise InvalidDepthData("uploaded file is not a valid NPZ/ZIP archive") from exc

    with archive:
        uncompressed_size = sum(info.file_size for info in archive.infolist())
        if uncompressed_size > MAX_DECOMPRESSED_BYTES:
            raise InvalidDepthData(
                f"decompressed NPZ size {uncompressed_size} bytes exceeds 8 MiB"
            )
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as npz:
                names = set(npz.files)
                required = {"depths", "K", "Tcw"}
                if names != required:
                    missing = sorted(required.difference(names))
                    extra = sorted(names.difference(required))
                    raise InvalidDepthData(
                        "NPZ must contain exactly depths, K and Tcw; "
                        f"missing={missing}, extra={extra}"
                    )
                depths_raw = npz["depths"]
                k_raw = npz["K"]
                poses_raw = npz["Tcw"]
        except InvalidDepthData:
            raise
        except Exception as exc:
            raise InvalidDepthData(f"could not read NPZ arrays: {exc}") from exc

    if depths_raw.dtype.kind == "O":
        raise InvalidDepthData("depths must not be an object array")
    if depths_raw.ndim != 3:
        raise InvalidDepthData("depths must have shape (N, H, W)")
    frame_count, height, width = depths_raw.shape
    if not 1 <= frame_count <= MAX_FRAMES:
        raise InvalidDepthData("frame count must be between 1 and 16")
    if not 1 <= height <= MAX_HEIGHT or not 1 <= width <= MAX_WIDTH:
        raise InvalidDepthData("depth frame dimensions must not exceed 128 x 128")

    depths = _finite_float_array(depths_raw, "depths")
    if np.any(depths < 0.0):
        raise InvalidDepthData("depth values must be non-negative; zero means missing")

    camera_matrix = _validate_camera_matrix(k_raw)
    if poses_raw.dtype.kind == "O":
        raise InvalidDepthData("Tcw must not be an object array")
    if poses_raw.ndim != 3 or poses_raw.shape[1:] != (4, 4):
        raise InvalidDepthData("Tcw must have shape (N, 4, 4)")
    if poses_raw.shape[0] != frame_count:
        raise InvalidDepthData("Tcw frame count must equal depth frame count")

    poses = np.stack([_validate_pose(poses_raw[i], i) for i in range(frame_count)])
    return DepthInput(depths=depths, camera_matrix=camera_matrix, world_from_camera=poses)
