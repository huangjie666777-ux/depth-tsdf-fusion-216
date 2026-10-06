"""Voxel-centre projection into depth cameras."""

from __future__ import annotations

import numpy as np


def voxel_centers(
    origin: np.ndarray, dims: tuple[int, int, int], voxel_size: float
) -> np.ndarray:
    """World-space voxel centres.

    Returns an array of shape (nz, ny, nx, 3) where the centre at integer
    index (iz, iy, ix) is origin + (index + 0.5) * voxel_size. Array axes
    are ordered (z, y, x).
    """
    nx, ny, nz = dims
    axes = [
        origin[0] + (np.arange(nx) + 0.5) * voxel_size,
        origin[1] + (np.arange(ny) + 0.5) * voxel_size,
        origin[2] + (np.arange(nz) + 0.5) * voxel_size,
    ]
    zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return np.stack([xx, yy, zz], axis=-1)


def project_to_camera(
    points_world: np.ndarray, K: np.ndarray, Tcw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into one camera.

    points_world: (..., 3) world coordinates.
    Returns (z_cam, u, v) with the leading shape of points_world; u, v are
    nearest-neighbour pixel indices computed as floor(coord + 0.5). Points
    with z_cam <= 0 are still returned; callers must mask them out.
    """
    flat = points_world.reshape(-1, 3)
    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    cam = (R.T @ (flat - t).T).T  # world -> camera
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cam[:, 0] / z
        y = cam[:, 1] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.floor(fx * x + cx + 0.5).astype(np.int64)
    v = np.floor(fy * y + cy + 0.5).astype(np.int64)
    shape = points_world.shape[:-1]
    return z.reshape(shape), u.reshape(shape), v.reshape(shape)
