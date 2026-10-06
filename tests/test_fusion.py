import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tsdf_fusion216.errors import InputValidationError
from tsdf_fusion216.fusion import TSDFVolume
from tsdf_fusion216.io_npz import load_depth_sequence
from tsdf_fusion216.main import app
from tsdf_fusion216.projection import project_to_camera, voxel_centers
from tsdf_fusion216.surface import extract_mesh, mesh_to_ply, vertices_to_world

K = np.array([[50.0, 0.0, 7.5], [0.0, 50.0, 7.5], [0.0, 0.0, 1.0]])


def make_npz(**overrides):
    depths = np.full((2, 16, 16), 2.0)
    Tcw = np.repeat(np.eye(4)[None], 2, axis=0)
    arrays = dict(depths=depths, K=K, Tcw=Tcw)
    arrays.update(overrides)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_load_valid():
    seq = load_depth_sequence(make_npz())
    assert seq.depths.shape == (2, 16, 16)


def test_reject_negative_depth():
    depths = np.full((1, 4, 4), 1.0)
    depths[0, 0, 0] = -0.5
    with pytest.raises(InputValidationError):
        load_depth_sequence(make_npz(depths=depths, Tcw=np.eye(4)[None]))


def test_reject_nonfinite():
    depths = np.full((1, 4, 4), np.nan)
    with pytest.raises(InputValidationError):
        load_depth_sequence(make_npz(depths=depths, Tcw=np.eye(4)[None]))


def test_reject_object_array():
    # craft an NPZ whose depths entry has an object dtype descriptor
    header = b"{'descr': '<O8', 'fortran_order': False, 'shape': (2,), }"
    pad = 64 - (10 + len(header) + 1) % 64
    npy = (bytes([0x93]) + b"NUMPY" + bytes([1, 0]) + (len(header) + pad + 1).to_bytes(2, "little")
           + header + b" " * pad + bytes([10]) + bytes(16))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("depths.npy", npy)
        for name, arr in (("K", K), ("Tcw", np.eye(4)[None])):
            b = io.BytesIO()
            np.save(b, arr)
            zf.writestr(f"{name}.npy", b.getvalue())
    with pytest.raises(InputValidationError):
        load_depth_sequence(buf.getvalue())


def test_reject_bad_intrinsics():
    bad = K.copy()
    bad[0, 0] = -1.0
    with pytest.raises(InputValidationError):
        load_depth_sequence(make_npz(K=bad))


def test_reject_bad_pose():
    Tcw = np.eye(4)[None].repeat(2, axis=0)
    Tcw[0, 0, 0] = 2.0
    with pytest.raises(InputValidationError):
        load_depth_sequence(make_npz(Tcw=Tcw))


def test_reject_too_many_frames():
    depths = np.ones((17, 4, 4))
    Tcw = np.repeat(np.eye(4)[None], 17, axis=0)
    with pytest.raises(InputValidationError):
        load_depth_sequence(make_npz(depths=depths, Tcw=Tcw))


def test_voxel_centers_layout():
    c = voxel_centers(np.array([1.0, 2.0, 3.0]), (2, 3, 4), 0.5)
    assert c.shape == (4, 3, 2, 3)  # (z, y, x)
    assert np.allclose(c[0, 0, 0], [1.25, 2.25, 3.25])
    assert np.allclose(c[1, 2, 1], [1.75, 3.25, 3.75])


def test_projection_roundtrip():
    pts = np.array([[[0.0, 0.0, 2.0], [0.5, -0.5, 2.0]]])
    T = np.eye(4)
    z, u, v = project_to_camera(pts, K, T)
    assert np.allclose(z, 2.0)
    assert (u[0, 0], v[0, 0]) == (8, 8)  # floor(7.5 + 0.5)
    assert (u[0, 1], v[0, 1]) == (20, -5)  # out of bounds, caller masks


def test_fusion_flat_wall():
    vol = TSDFVolume((0.0, 0.0, 0.0), (8, 8, 8), 0.1, 0.15)
    depth = np.full((16, 16), 0.45)
    vol.integrate(depth, K, np.eye(4))
    assert vol.observed_voxels > 0
    # voxel centres at z = 0.05..0.75; wall at 0.45 -> zero crossing
    # between indices 3 (0.35) and 4 (0.45): s at idx4 = 0 -> tsdf 0
    assert vol.tsdf[4, 0, 0] == pytest.approx(0.0)
    assert vol.tsdf[3, 0, 0] == pytest.approx(min(1.0, 0.1 / 0.15))
    assert vol.weight[3, 0, 0] == 1.0
    # just behind the wall, within trunc: negative value
    assert vol.tsdf[5, 0, 0] == pytest.approx(-0.1 / 0.15)
    # beyond trunc behind the wall: no update
    assert vol.weight[7, 0, 0] == 0.0


def test_fusion_cumulative_average():
    vol = TSDFVolume((0.0, 0.0, 0.0), (4, 4, 4), 0.1, 0.2)
    vol.integrate(np.full((16, 16), 0.4), K, np.eye(4))
    vol.integrate(np.full((16, 16), 0.5), K, np.eye(4))
    # voxel idx3 (centre 0.35): s1=0.05->0.25, s2=0.15->0.75 -> mean 0.5
    assert vol.tsdf[3, 0, 0] == pytest.approx(0.5)
    assert vol.weight[3, 0, 0] == 2.0


def test_extract_mesh_empty_when_unknown():
    tsdf = np.ones((4, 4, 4))
    w = np.zeros((4, 4, 4))
    v, f = extract_mesh(tsdf, w)
    assert len(v) == 0 and len(f) == 0


def test_extract_mesh_open_at_unknown():
    tsdf = np.ones((6, 6, 6))
    tsdf[2:4, 2:4, 2:4] = -1.0
    w = np.zeros((6, 6, 6))
    w[1:5, 1:5, 1:5] = 1.0  # unknown border stays open
    v, f = extract_mesh(tsdf, w)
    assert len(v) > 0 and len(f) > 0
    # no vertex should sit on the outer boundary of the known region
    assert v[:, 0].min() > 1.0 and v[:, 0].max() < 4.0


def test_vertices_to_world():
    verts = np.array([[0.5, 0.5, 0.5]])
    world = vertices_to_world(verts, np.array([1.0, 2.0, 3.0]), 0.1)
    assert np.allclose(world, [[1.05, 2.05, 3.05]])


def test_ply_ascii():
    ply = mesh_to_ply(np.array([[0.0, 0.0, 0.0]]), np.zeros((0, 3), int))
    text = ply.decode()
    assert text.startswith("ply\nformat ascii 1.0")
    assert "element vertex 1" in text


PARAMS = json.dumps({
    "origin": [-0.4, -0.4, -0.4],
    "dims": [40, 40, 40],
    "voxel_size": 0.02,
    "trunc": 0.06,
})


def test_endpoint_reconstructs_sphere():
    client = TestClient(app)
    with open("samples/sample_sphere.npz", "rb") as fh:
        resp = client.post(
            "/reconstruct",
            files={"file": ("sample.npz", fh.read(), "application/octet-stream")},
            data={"params": PARAMS},
        )
    assert resp.status_code == 200, resp.text
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert set(zf.namelist()) == {"mesh.ply", "tsdf_weight.npz", "run.json"}
    meta = json.loads(zf.read("run.json"))
    assert meta["num_faces"] > 0
    assert meta["observed_voxels"] > 0
    ply = zf.read("mesh.ply").decode()
    assert ply.startswith("ply")
    # vertices should cluster near the sphere surface (radius 0.3)
    lines = ply.splitlines()
    nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    head = lines.index("end_header") + 1
    pts = np.array([[float(x) for x in l.split()] for l in lines[head:head + nv]])
    r = np.linalg.norm(pts, axis=1)
    assert abs(r.mean() - 0.3) < 0.03
    npz = np.load(io.BytesIO(zf.read("tsdf_weight.npz")))
    assert npz["tsdf"].shape == (40, 40, 40)


def test_endpoint_rejects_bad_params():
    client = TestClient(app)
    with open("samples/sample_sphere.npz", "rb") as fh:
        resp = client.post(
            "/reconstruct",
            files={"file": ("s.npz", fh.read(), "application/octet-stream")},
            data={"params": json.dumps({
                "origin": [0, 0, 0], "dims": [1, 40, 40],
                "voxel_size": 0.02, "trunc": 0.06})},
        )
    assert resp.status_code == 422


def test_endpoint_rejects_bad_npz():
    client = TestClient(app)
    resp = client.post(
        "/reconstruct",
        files={"file": ("s.npz", b"not an npz", "application/octet-stream")},
        data={"params": PARAMS},
    )
    assert resp.status_code == 422
