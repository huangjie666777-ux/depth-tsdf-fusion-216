# tsdf_fusion216

Pure-backend multi-view depth fusion service: fuses metric depth maps into a
TSDF volume and extracts the object surface as a triangle mesh.

## Stack

- Python 3.10.12, FastAPI 0.115.12, NumPy 2.2.6, scikit-image 0.25.2
- Fusion core implemented from scratch (no point-cloud stitching, no
  off-the-shelf fuser); scikit-image is used only for marching cubes.

## Layout

- `tsdf_fusion216/io_npz.py` — NPZ upload parsing and validation
- `tsdf_fusion216/projection.py` — voxel-centre generation and pinhole projection
- `tsdf_fusion216/fusion.py` — TSDF fusion core (cumulative weighted average)
- `tsdf_fusion216/surface.py` — zero-level-set extraction, world transform, ASCII PLY
- `tsdf_fusion216/pipeline.py` — end-to-end orchestration
- `tsdf_fusion216/packaging.py` — ZIP delivery assembly
- `tsdf_fusion216/main.py` — FastAPI endpoint
- `samples/make_sample.py` — known-pose sample generator (sphere, 6 views)
- `tests/` — pytest suite

## Input

`POST /reconstruct` with multipart form:

- `file`: NPZ archive containing
  - `depths`: float array, shape `(N, H, W)`, metres, camera-z; `0` = missing.
  - `K`: shared `(3, 3)` pinhole intrinsics.
  - `Tcw`: `(N, 4, 4)` camera-to-world rigid transforms.
- `params`: JSON string, e.g.
  `{"origin": [-0.4, -0.4, -0.4], "dims": [40, 40, 40], "voxel_size": 0.02, "trunc": 0.06}`

Limits: N ≤ 16 frames, H, W ≤ 128 px, ≤ 8 MiB decompressed; dims per axis in
[2, 64]; `voxel_size` and `trunc` positive; all lengths in metres. Negative or
non-finite depths, object arrays, invalid intrinsics, and non-rigid poses are
rejected with HTTP 422.

## Fusion model

- Voxel centres: `origin + (index + 0.5) * voxel_size`; array axes `(z, y, x)`.
- Centres are transformed to each camera; only `z > 0` is projected.
- Pixels use nearest neighbour: `floor(coord + 0.5)`; out-of-bounds or
  missing depths are skipped.
- Distance `s = depth - z_cam`; voxels with `s < -trunc` are not updated,
  otherwise the update is `min(1, s / trunc)` with weight 1, accumulated as a
  running average. TSDF starts at 1, weights at 0.
- Marching cubes runs only on cells whose eight corners all have positive
  weight, so unknown regions stay open. Vertices are returned in world
  metres. If no zero crossing exists, an empty mesh is returned.

## Output

`application/zip` containing:

- `mesh.ply` — ASCII PLY triangle mesh (world metres)
- `tsdf_weight.npz` — `tsdf` and `weight` volumes (axes `z, y, x`)
- `run.json` — parameters, observed voxel count, vertex/face counts

## Run

```bash
.venv/bin/pip install -r requirements.txt   # already installed in .venv
.venv/bin/uvicorn tsdf_fusion216.main:app --port 8216
```

## Try it

```bash
.venv/bin/python samples/make_sample.py     # writes samples/sample_sphere.npz
curl -s -o reconstruction.zip \
  -F "file=@samples/sample_sphere.npz" \
  -F 'params={"origin": [-0.4, -0.4, -0.4], "dims": [40, 40, 40], "voxel_size": 0.02, "trunc": 0.06}' \
  http://127.0.0.1:8216/reconstruct
unzip -l reconstruction.zip
```

## Test

```bash
.venv/bin/python -m pytest tests -q
```

