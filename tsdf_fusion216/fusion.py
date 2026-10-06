from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import DepthInput, FusionParameters, FusedVolume
from .projection import project_world_centers


def _voxel_centers(params: FusionParameters) -> NDArray[np.float64]:
    dim_z, dim_y, dim_x = params.dims
    z_index, y_index, x_index = np.indices(
        (dim_z, dim_y, dim_x), dtype=np.float64
    )
    origin = np.asarray(params.origin, dtype=np.float64)
    centers = np.empty((dim_z, dim_y, dim_x, 3), dtype=np.float64)
    centers[..., 0] = origin[0] + (x_index + 0.5) * params.voxel_size
    centers[..., 1] = origin[1] + (y_index + 0.5) * params.voxel_size
    centers[..., 2] = origin[2] + (z_index + 0.5) * params.voxel_size
    return centers


def fuse_depth_frames(data: DepthInput, params: FusionParameters) -> FusedVolume:
    dims = params.dims
    tsdf = np.ones(dims, dtype=np.float64)
    weights = np.zeros(dims, dtype=np.int32)
    world_centers = _voxel_centers(params).reshape(-1, 3)
    k = data.camera_matrix

    for frame_index in range(data.depths.shape[0]):
        frame = data.depths[frame_index]
        camera_points, pixel_y, pixel_x = project_world_centers(
            world_centers, k, data.world_from_camera[frame_index]
        )
        camera_z = camera_points[:, 2]
        in_front = camera_z > 0.0
        height, width = frame.shape
        valid = (
            in_front
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
        )
        observed_depth = np.zeros(valid.shape, dtype=np.float64)
        observed_depth[valid] = frame[pixel_y[valid], pixel_x[valid]]
        valid &= observed_depth > 0.0

        surface_distance = np.full(valid.shape, np.nan, dtype=np.float64)
        surface_distance[valid] = observed_depth[valid] - camera_z[valid]
        valid &= surface_distance >= -params.trunc
        if not np.any(valid):
            continue

        new_tsdf = np.minimum(1.0, surface_distance[valid] / params.trunc)
        old_weights = weights.reshape(-1)[valid].astype(np.float64)
        old_tsdf = tsdf.reshape(-1)[valid]
        updated_weights = old_weights + 1.0
        tsdf.reshape(-1)[valid] = (
            old_tsdf * old_weights + new_tsdf
        ) / updated_weights
        weights.reshape(-1)[valid] = updated_weights.astype(np.int32)

    return FusedVolume(tsdf=tsdf, weights=weights, parameters=params)
