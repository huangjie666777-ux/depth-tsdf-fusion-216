"""Zero-level-set extraction and PLY export."""

from __future__ import annotations

import numpy as np
from skimage import measure


def extract_mesh(
    tsdf: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the zero isosurface of the TSDF in voxel-index space.

    Only cells whose eight corners all have positive weight are meshed, so
    unknown regions are left open. Returns (vertices, faces) with vertices
    in (z, y, x) index coordinates, or two empty arrays when no zero
    crossing exists.
    """
    known = weight > 0.0
    if not known.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    tsdf_min = float(tsdf[known].min())
    tsdf_max = float(tsdf[known].max())
    if tsdf_min >= 0.0 or tsdf_max <= 0.0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    # Mask unknown cells by assigning them a constant positive value so no
    # zero crossing is ever introduced next to unobserved voxels.
    masked = np.where(known, tsdf, 1.0)

    # A cell is processed only when all eight corners are observed: pad the
    # known mask and require every corner of each unit cube to be known.
    known_f = known.astype(np.float64)
    fully_known = (
        known_f[:-1, :-1, :-1]
        * known_f[:-1, :-1, 1:]
        * known_f[:-1, 1:, :-1]
        * known_f[:-1, 1:, 1:]
        * known_f[1:, :-1, :-1]
        * known_f[1:, :-1, 1:]
        * known_f[1:, 1:, :-1]
        * known_f[1:, 1:, 1:]
    ) > 0.0
    if not fully_known.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    cell_min = np.minimum.reduce(
        [
            masked[:-1, :-1, :-1],
            masked[:-1, :-1, 1:],
            masked[:-1, 1:, :-1],
            masked[:-1, 1:, 1:],
            masked[1:, :-1, :-1],
            masked[1:, :-1, 1:],
            masked[1:, 1:, :-1],
            masked[1:, 1:, 1:],
        ]
    )
    cell_max = np.maximum.reduce(
        [
            masked[:-1, :-1, :-1],
            masked[:-1, :-1, 1:],
            masked[:-1, 1:, :-1],
            masked[:-1, 1:, 1:],
            masked[1:, :-1, :-1],
            masked[1:, :-1, 1:],
            masked[1:, 1:, :-1],
            masked[1:, 1:, 1:],
        ]
    )
    active = fully_known & (cell_min < 0.0) & (cell_max > 0.0)
    if not active.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    # Crop to the active region (plus one voxel of margin) to keep the
    # marching-cubes call small, then drop any triangles touching inactive
    # cells afterwards.
    zi, yi, xi = np.nonzero(active)
    z0, z1 = zi.min(), zi.max() + 2
    y0, y1 = yi.min(), yi.max() + 2
    x0, x1 = xi.min(), xi.max() + 2
    sub = masked[z0:z1, y0:y1, x0:x1]

    verts, faces, _, _ = measure.marching_cubes(sub, level=0.0)
    verts = verts + np.array([z0, y0, x0], dtype=np.float64)

    # Keep only faces fully surrounded by active (fully observed) cells:
    # every unit cell overlapped by the triangle's voxel bounding box must
    # be active, which guarantees no face borders unknown space.
    vox = np.floor(verts).astype(np.int64)
    tri = vox[faces]  # (F, 3, 3): vertex voxels in (z, y, x)
    lo = tri.min(axis=1)
    hi = tri.max(axis=1)
    keep = np.zeros(len(faces), dtype=bool)
    ashape = np.array(active.shape)
    for i in range(len(faces)):
        a = np.maximum(lo[i], 0)
        b = np.minimum(hi[i], ashape - 1)
        if np.any(b < a):
            continue
        block = active[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]
        keep[i] = bool(block.all())
    faces = faces[keep]
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    used = np.unique(faces)
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = verts[used]
    faces = remap[faces]
    return verts, faces


def vertices_to_world(
    verts: np.ndarray, origin: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Convert (z, y, x) index-space vertices to world metres (x, y, z).

    Index i along an axis corresponds to position origin + i * voxel_size
    (centre of voxel 0 sits at origin + 0.5 * voxel_size).
    """
    if len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    world = np.empty_like(verts, dtype=np.float64)
    world[:, 0] = origin[0] + verts[:, 2] * voxel_size
    world[:, 1] = origin[1] + verts[:, 1] * voxel_size
    world[:, 2] = origin[2] + verts[:, 0] * voxel_size
    return world


def mesh_to_ply(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Serialise a triangle mesh as ASCII PLY."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in vertices:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        lines.append(f"3 {int(f[0])} {int(f[1])} {int(f[2])}")
    return ("\n".join(lines) + "\n").encode("ascii")
