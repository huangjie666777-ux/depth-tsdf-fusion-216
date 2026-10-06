from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from .models import FusedVolume, FusionParameters, FusionResult


def write_ascii_ply(mesh: FusionResult) -> bytes:
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    buffer = io.StringIO(header)
    for vertex in mesh.vertices:
        buffer.write(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
    for face in mesh.faces:
        buffer.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    return buffer.getvalue().encode("ascii")


def parameters_payload(params: FusionParameters) -> dict:
    return {
        "origin": list(params.origin),
        "dims": list(params.dims),
        "voxel_size": params.voxel_size,
        "trunc": params.trunc,
        "units": {
            "origin": "metre",
            "voxel_size": "metre",
            "trunc": "metre",
            "dims": "voxel counts in z, y, x order",
        },
    }


def build_result_zip(volume: FusedVolume, mesh: FusionResult) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mesh.ply", write_ascii_ply(mesh))

        volume_buffer = io.BytesIO()
        np.savez_compressed(
            volume_buffer,
            tsdf=volume.tsdf,
            weights=volume.weights,
        )
        archive.writestr("tsdf_weights.npz", volume_buffer.getvalue())

        params = parameters_payload(volume.parameters)
        archive.writestr("parameters.json", json.dumps(params, indent=2, sort_keys=True))

        stats = {
            "observed_voxels": int(np.count_nonzero(volume.weights)),
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
        }
        archive.writestr("stats.json", json.dumps(stats, indent=2, sort_keys=True))
    return output.getvalue()
