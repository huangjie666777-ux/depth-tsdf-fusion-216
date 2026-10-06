"""Generate a known-pose sample NPZ: a sphere rendered from 6 views.

Run with: .venv/bin/python samples/make_sample.py
Writes samples/sample_sphere.npz and samples/sample_params.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

H = W = 64
FX = FY = 70.0
CX = (W - 1) / 2.0
CY = (H - 1) / 2.0

CENTER = np.array([0.0, 0.0, 0.0])
RADIUS = 0.3


def look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world matrix with camera z pointing at the target."""
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    R = np.stack([right, true_up, forward], axis=1)  # camera axes in world
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return T


def render_depth(K: np.ndarray, Tcw: np.ndarray) -> np.ndarray:
    """Ray-sphere intersection per pixel; 0 where the ray misses."""
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    R = Tcw[:3, :3]
    eye = Tcw[:3, 3]
    dirs_world = dirs_cam @ R.T  # rows are world-space ray directions
    oc = eye - CENTER
    a = (dirs_world * dirs_world).sum(-1)
    b = 2.0 * (dirs_world * oc).sum(-1)
    c = float(oc @ oc) - RADIUS * RADIUS
    disc = b * b - 4.0 * a * c
    depth = np.zeros((H, W), dtype=np.float64)
    hit = disc >= 0.0
    aa = a[hit]
    t = (-b[hit] - np.sqrt(disc[hit])) / (2.0 * aa)
    t = np.where(t > 0.0, t, (-b[hit] + np.sqrt(disc[hit])) / (2.0 * aa))
    valid = t > 0.0
    # depth is the camera-z coordinate of the hit point
    pts = eye[None, :] + t[valid, None] * dirs_world[hit][valid]
    cam_pts = (pts - eye) @ R
    d = np.zeros((H, W))
    idx = np.flatnonzero(hit.ravel())[valid]
    d.ravel()[idx] = cam_pts[:, 2]
    depth = d
    return depth


def main() -> None:
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    positions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    Tcw = np.stack([look_at(p, CENTER) for p in positions])
    depths = np.stack([render_depth(K, T) for T in Tcw])

    np.savez(os.path.join(HERE, "sample_sphere.npz"), depths=depths, K=K, Tcw=Tcw)
    params = {
        "origin": [-0.4, -0.4, -0.4],
        "dims": [40, 40, 40],
        "voxel_size": 0.02,
        "trunc": 0.06,
    }
    with open(os.path.join(HERE, "sample_params.json"), "w") as fh:
        json.dump(params, fh, indent=2)
    print("wrote sample_sphere.npz and sample_params.json")


if __name__ == "__main__":
    main()
