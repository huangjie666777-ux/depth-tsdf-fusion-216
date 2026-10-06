from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tsdf_fusion216.api import app
from tsdf_fusion216.fusion import fuse_depth_frames
from tsdf_fusion216.input_validation import load_depth_npz
from tsdf_fusion216.models import FusionParameters, FusedVolume
from tsdf_fusion216.surface import extract_zero_surface


def make_input() -> tuple[bytes, dict]:
    depth = np.zeros((1, 5, 5), dtype=np.float32)
    depth[0, 1:4, 1:4] = 1.0
    k = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, depths=depth, K=k, Tcw=pose[None])
    params = {
        "origin": [-0.5, -0.5, 0.4],
        "dims": [2, 5, 5],
        "voxel_size": 0.2,
        "trunc": 0.3,
    }
    return buffer.getvalue(), params


def test_fusion_averages_observed_voxels() -> None:
    raw, raw_params = make_input()
    data = load_depth_npz(raw)
    params = FusionParameters.from_payload(raw_params)
    volume = fuse_depth_frames(data, params)

    assert int(volume.weights[0, 2, 2]) == 1
    assert volume.tsdf[0, 2, 2] == pytest.approx(1.0)
    assert volume.tsdf[0, 0, 0] == pytest.approx(1.0)
    assert int(volume.weights[0, 0, 0]) == 0


def test_rejects_negative_depth_and_bad_pose() -> None:
    depth = np.full((1, 2, 2), -0.1, dtype=np.float32)
    k = np.eye(3)
    pose = np.eye(4)
    buffer = io.BytesIO()
    np.savez(buffer, depths=depth, K=k, Tcw=pose[None])
    with pytest.raises(ValueError):
        load_depth_npz(buffer.getvalue())

    depth[:] = 1.0
    pose[0, 0] = 2.0
    buffer = io.BytesIO()
    np.savez(buffer, depths=depth, K=k, Tcw=pose[None])
    with pytest.raises(ValueError):
        load_depth_npz(buffer.getvalue())


def test_unknown_region_does_not_create_face() -> None:
    raw, raw_params = make_input()
    data = load_depth_npz(raw)
    params = FusionParameters.from_payload(raw_params)
    mesh = extract_zero_surface(fuse_depth_frames(data, params))
    assert len(mesh.faces) == 0


def test_zero_surface_uses_complete_cells_and_world_coordinates() -> None:
    params = FusionParameters.from_payload(
        {
            "origin": [1.0, 2.0, 3.0],
            "dims": [2, 2, 2],
            "voxel_size": 1.0,
            "trunc": 1.0,
        }
    )
    tsdf = np.ones((2, 2, 2))
    tsdf[1, :, :] = -1.0
    weights = np.ones((2, 2, 2), dtype=np.int32)
    mesh = extract_zero_surface(FusedVolume(tsdf=tsdf, weights=weights, parameters=params))
    assert len(mesh.faces) > 0
    assert np.allclose(mesh.vertices[:, 2], 3.5)
    assert np.all(mesh.vertices[:, 0] >= 1.0)
    assert np.all(mesh.vertices[:, 1] >= 2.0)


def test_http_zip_delivery() -> None:
    raw, raw_params = make_input()
    client = TestClient(app)
    response = client.post(
        "/fuse",
        files={"file": ("depths.npz", raw, "application/octet-stream")},
        data={"params": json.dumps(raw_params)},
    )
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(archive.namelist()) == {
        "mesh.ply",
        "tsdf_weights.npz",
        "parameters.json",
        "stats.json",
    }
    stats = json.loads(archive.read("stats.json"))
    assert stats["observed_voxels"] > 0
    assert stats["faces"] == 0
    assert archive.read("mesh.ply").startswith(b"ply\nformat ascii 1.0\n")
