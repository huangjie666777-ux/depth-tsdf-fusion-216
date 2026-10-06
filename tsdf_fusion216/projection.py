from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def project_world_centers(
    world_centers: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    world_from_camera: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    rotation = world_from_camera[:3, :3]
    translation = world_from_camera[:3, 3]
    camera_points = (world_centers - translation) @ rotation
    camera_z = camera_points[:, 2]
    in_front = camera_z > 0.0
    safe_z = np.where(in_front, camera_z, 1.0)

    projected_x = (
        camera_matrix[0, 0] * camera_points[:, 0]
        + camera_matrix[0, 1] * camera_points[:, 1]
        + camera_matrix[0, 2] * safe_z
    ) / safe_z
    projected_y = (
        camera_matrix[1, 0] * camera_points[:, 0]
        + camera_matrix[1, 1] * camera_points[:, 1]
        + camera_matrix[1, 2] * safe_z
    ) / safe_z
    pixel_x = np.floor(projected_x + 0.5).astype(np.int64)
    pixel_y = np.floor(projected_y + 0.5).astype(np.int64)
    return camera_points, pixel_y, pixel_x
