from __future__ import annotations

import numpy as np
from skimage import measure

from .models import FusedVolume, FusionResult


def extract_zero_surface(volume: FusedVolume) -> FusionResult:
    known = volume.weights > 0
    valid_cells = (
        known[:-1, :-1, :-1]
        & known[1:, :-1, :-1]
        & known[:-1, 1:, :-1]
        & known[:-1, :-1, 1:]
        & known[1:, 1:, :-1]
        & known[1:, :-1, 1:]
        & known[:-1, 1:, 1:]
        & known[1:, 1:, 1:]
    )
    if not np.any(valid_cells):
        return FusionResult(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        )

    corner_values = np.stack(
        [
            volume.tsdf[:-1, :-1, :-1][valid_cells],
            volume.tsdf[1:, :-1, :-1][valid_cells],
            volume.tsdf[:-1, 1:, :-1][valid_cells],
            volume.tsdf[:-1, :-1, 1:][valid_cells],
            volume.tsdf[1:, 1:, :-1][valid_cells],
            volume.tsdf[1:, :-1, 1:][valid_cells],
            volume.tsdf[:-1, 1:, 1:][valid_cells],
            volume.tsdf[1:, 1:, 1:][valid_cells],
        ],
        axis=1,
    )
    crosses_zero = (corner_values.min(axis=1) <= 0.0) & (
        corner_values.max(axis=1) >= 0.0
    )
    if not np.any(crosses_zero):
        return FusionResult(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        )

    cell_mask = np.zeros(volume.tsdf.shape, dtype=bool)
    cell_mask[1:, 1:, 1:] = valid_cells
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            volume.tsdf,
            level=0.0,
            spacing=(volume.parameters.voxel_size,) * 3,
            mask=cell_mask,
            allow_degenerate=False,
        )
    except RuntimeError:
        return FusionResult(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        )

    origin = np.asarray(volume.parameters.origin, dtype=np.float64)
    world_vertices = np.empty_like(vertices)
    world_vertices[:, 0] = origin[0] + vertices[:, 2]
    world_vertices[:, 1] = origin[1] + vertices[:, 1]
    world_vertices[:, 2] = origin[2] + vertices[:, 0]
    return FusionResult(vertices=world_vertices, faces=faces.astype(np.int64))
