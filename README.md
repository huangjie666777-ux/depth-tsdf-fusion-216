# TSDF Fusion 216

纯后端多视角米制定向深度图融合服务，包名为 `tsdf_fusion216`。实现自行完成体素投影、截断符号距离累计平均和零等值面提取；点云拼接或第三方现成融合器均未用于融合核心。

## 环境

- Python 3.10.12
- FastAPI 0.115.12
- NumPy 2.2.6
- scikit-image 0.25.2（仅用于 marching cubes 零等值面）

运行所有命令前使用仓库内解释器 `.venv/bin/python`。

## 输入 NPZ

`POST /fuse` 使用 `multipart/form-data`：

- `file`：NPZ，内部数组名必须恰好为 `depths`、`K`、`Tcw`。
- `params`：JSON 字符串，描述融合体积。

`depths` 形状为 `N x H x W`，单位米，深度是相机坐标系正 `z`，`0` 表示缺测。`K` 为共享的 3x3 针孔内参。`Tcw` 形状为 `N x 4 x 4`，表示相机到世界刚体变换。服务拒绝负值、非有限值、object 数组、非法内参和非刚体位姿。

限制：

- 最多 16 帧。
- 每帧最多 128 x 128 像素。
- NPZ 解压后总内容最多 8 MiB。

体积参数示例：

```json
{
  "origin": [-0.6, -0.6, 0.35],
  "dims": [2, 12, 12],
  "voxel_size": 0.1,
  "trunc": 0.2
}
```

`dims` 轴序是 `z, y, x`，每轴为 2 至 64 的整数。体素 `(iz, iy, ix)` 的中心为：

```text
world_x = origin_x + (ix + 0.5) * voxel_size
world_y = origin_y + (iy + 0.5) * voxel_size
world_z = origin_z + (iz + 0.5) * voxel_size
```

世界点通过 `Tcw` 的逆变换进入相机。仅相机 `z > 0` 的体素中心投影；像素使用 `floor(projected + 0.5)` 取最近邻，越界或缺测深度跳过。距离 `s = measured_depth - camera_z`，当 `s < -trunc` 时跳过，否则融合值为 `min(1, s / trunc)`。每帧每次观测权重为 1，并按旧权重累计平均。

## 输出 ZIP

响应为 `tsdf_fusion_216.zip`，包含：

- `mesh.ply`：ASCII PLY；没有零面时为空网格。
- `tsdf_weights.npz`：`tsdf` 与 `weights` 数组，轴序 `z, y, x`。
- `parameters.json`：实际使用的米制体积参数。
- `stats.json`：`observed_voxels`、`faces`、`vertices`。

零等值面只在八个角点权重都大于 0 的单元中提取，未知区域不会被封口。scikit-image 返回的是体素索引坐标，服务将其按 `z, y, x` 转回世界米制坐标。

## 样例与运行

生成已知位姿样例：

```bash
.venv/bin/python -m scripts.create_sample
```

启动服务：

```bash
.venv/bin/python -m tsdf_fusion216
```

用 curl 融合并下载结果：

```bash
curl -f -o result.zip \
  -F "file=@samples/known_poses.npz;type=application/octet-stream" \
  -F "params=<samples/volume_params.json" \
  http://127.0.0.1:8000/fuse
.venv/bin/python -c "import zipfile; print(zipfile.ZipFile('result.zip').namelist())"
```

## 自测

```bash
.venv/bin/python -m compileall -q tsdf_fusion216 scripts tests
.venv/bin/python -m pytest -q
```
