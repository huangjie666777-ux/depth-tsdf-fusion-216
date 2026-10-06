"""End-to-end fusion pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fusion import TSDFVolume
from .io_npz import DepthSequence
from .schemas import FusionParams
from .surface import extract_mesh, vertices_to_world


@dataclass
class FusionResult:
    tsdf: np.ndarray
    weight: np.ndarray
    vertices: np.ndarray  # world metres, (M, 3)
    faces: np.ndarray  # (F, 3) int
    observed_voxels: int


def run_fusion(seq: DepthSequence, params: FusionParams) -> FusionResult:
    volume = TSDFVolume(
        origin=params.origin,
        dims=params.dims,
        voxel_size=params.voxel_size,
        trunc=params.trunc,
    )
    for i in range(seq.depths.shape[0]):
        volume.integrate(seq.depths[i], seq.K, seq.Tcw[i])

    verts_idx, faces = extract_mesh(volume.tsdf, volume.weight)
    verts_world = vertices_to_world(
        verts_idx, volume.origin, volume.voxel_size
    )
    return FusionResult(
        tsdf=volume.tsdf,
        weight=volume.weight,
        vertices=verts_world,
        faces=faces,
        observed_voxels=volume.observed_voxels,
    )
