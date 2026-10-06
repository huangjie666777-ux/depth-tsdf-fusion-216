from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def render_depth(world_from_camera: np.ndarray, size: int = 16) -> np.ndarray:
    depth = np.zeros((size, size), dtype=np.float32)
    k = np.array([[8.0, 0.0, 7.5], [0.0, 8.0, 7.5], [0.0, 0.0, 1.0]])
    rotation = world_from_camera[:3, :3]
    translation = world_from_camera[:3, 3]

    for pixel_y in range(size):
        for pixel_x in range(size):
            x_camera = (pixel_x - k[0, 2]) / k[0, 0]
            y_camera = (pixel_y - k[1, 2]) / k[1, 1]
            z_camera = (0.5 - (rotation[2] @ translation)) / rotation[2, 2]
            camera_point = np.array(
                [x_camera * z_camera, y_camera * z_camera, z_camera]
            )
            world_point = rotation @ camera_point + translation
            if abs(world_point[0]) <= 0.5 and abs(world_point[1]) <= 0.5:
                depth[pixel_y, pixel_x] = z_camera
    return depth


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "samples"
    sample_dir.mkdir(exist_ok=True)

    poses = np.stack(
        [
            np.eye(4, dtype=np.float64),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.25],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ]
    )
    depths = np.stack([render_depth(pose) for pose in poses])
    camera_matrix = np.array(
        [[8.0, 0.0, 7.5], [0.0, 8.0, 7.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    np.savez_compressed(
        sample_dir / "known_poses.npz", depths=depths, K=camera_matrix, Tcw=poses
    )

    params = {
        "origin": [-0.6, -0.6, 0.35],
        "dims": [2, 12, 12],
        "voxel_size": 0.1,
        "trunc": 0.2,
    }
    (sample_dir / "volume_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
