"""TSDF fusion core (implemented from scratch, no off-the-shelf fuser)."""

from __future__ import annotations

import numpy as np

from .projection import project_to_camera, voxel_centers


class TSDFVolume:
    """Truncated signed distance volume with cumulative averaging."""

    def __init__(
        self,
        origin: tuple[float, float, float],
        dims: tuple[int, int, int],
        voxel_size: float,
        trunc: float,
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64)
        self.dims = tuple(int(d) for d in dims)  # (nx, ny, nz)
        self.voxel_size = float(voxel_size)
        self.trunc = float(trunc)
        nx, ny, nz = self.dims
        # Axes ordered (z, y, x); TSDF initialised to 1, weights to 0.
        self.tsdf = np.ones((nz, ny, nx), dtype=np.float64)
        self.weight = np.zeros((nz, ny, nx), dtype=np.float64)
        self._centers = voxel_centers(self.origin, self.dims, self.voxel_size)

    @property
    def observed_voxels(self) -> int:
        return int(np.count_nonzero(self.weight > 0.0))

    def integrate(self, depth: np.ndarray, K: np.ndarray, Tcw: np.ndarray) -> None:
        """Fuse one depth frame (metres, camera z, 0 = missing)."""
        h, w = depth.shape
        z_cam, u, v = project_to_camera(self._centers, K, Tcw)

        valid = z_cam > 0.0
        in_bounds = (
            valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        )
        if not np.any(in_bounds):
            return

        ui = u[in_bounds]
        vi = v[in_bounds]
        measured = depth[vi, ui]
        has_depth = measured > 0.0
        if not np.any(has_depth):
            return

        idx = np.flatnonzero(in_bounds.ravel())[has_depth]
        zi = z_cam.ravel()[idx]
        di = measured[has_depth]

        s = di - zi
        keep = s >= -self.trunc
        if not np.any(keep):
            return
        idx = idx[keep]
        tsdf_new = np.minimum(1.0, s[keep] / self.trunc)

        tsdf_flat = self.tsdf.ravel()
        weight_flat = self.weight.ravel()
        w_old = weight_flat[idx]
        w_new = w_old + 1.0
        tsdf_flat[idx] = (tsdf_flat[idx] * w_old + tsdf_new) / w_new
        weight_flat[idx] = w_new
