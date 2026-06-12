# Assignment 4 - 简化版 3D Gaussian Splatting 实验报告

本仓库实现了一个纯 PyTorch 版本的简化 3D Gaussian Splatting，并在 `chair` 多视角数据上完成 COLMAP 相机恢复、3D Gaussian 可微渲染训练，以及与官方 3DGS CUDA 实现的实测对比。

## Requirements

实验环境如下：

| 项目 | 版本 / 配置 |
| --- | --- |
| 系统 | Windows |
| Python | 3.13.9 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| COLMAP | 4.0.3 |

主要依赖包括：

```setup
pip install torch torchvision opencv-python numpy tqdm natsort plyfile joblib iopath
```

本实验还使用了本地编译安装的 `pytorch3d`。官方 3DGS 对比实验需要额外安装 `simple-knn` 与 `diff-gaussian-rasterization` 两个 CUDA 扩展。由于当前 Python/PyTorch/CUDA 版本较新，官方扩展源码做了兼容处理：CUDA 编译单元避免直接包含 `torch/extension.h`，改用 ATen Tensor 接口。`fused_ssim` 为可选模块，未安装时官方训练代码会自动回退到 Python SSIM。

## Training

### Task 1: COLMAP 相机恢复

本实验选择 `data/chair` 场景，共 100 张多视角图像。运行：

```train
python mvs_with_colmap.py --data_dir data/chair
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

由于 COLMAP 4.0.3 的参数名发生变化，脚本中的 GPU 参数已改为：

```text
FeatureExtraction.use_gpu
FeatureMatching.use_gpu
```

COLMAP 成功生成稀疏重建：

| 项目 | 结果 |
| --- | --- |
| 图像数量 | 100 |
| 稀疏 3D 点数量 | 13,631 |
| COLMAP 文本输出 | `data/chair/sparse/0_text` |
| 重投影可视化 | `data/chair/projections/` |

### Task 2: 简化版 3DGS 训练

核心 TODO 已完成：

| 模块 | 实现内容 |
| --- | --- |
| `gaussian_model.py` | 由四元数和缩放参数构造 3D 协方差 `Sigma = R S S^T R^T` |
| `gaussian_renderer.py` | 实现 3D 到 2D 投影与投影协方差 `Sigma_2D = J W Sigma_3D W^T J^T` |
| `gaussian_renderer.py` | 实现 2D Gaussian 取值 |
| `gaussian_renderer.py` | 实现基于深度排序的 alpha blending |

训练命令：

```train
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints --device cuda
```

训练中曾在 epoch 171 附近出现 CUDA out of memory。根据报错信息，主要问题是长时间训练后显存碎片化和中间张量峰值较高。已对训练脚本做了以下处理：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
zero_grad(set_to_none=True)
rendered_images.detach()
torch.cuda.empty_cache()
resume 时从 checkpoint epoch + 1 继续
训练结束补存最后一轮 checkpoint
```

### Task 3: 官方 3DGS 训练

官方 3DGS 源码位于 `gaussian-splatting-main/`。为了与简化版 100x100 分辨率评估保持一致，官方训练使用 `-r 8` 将 800x800 图像下采样到 100x100：

```train
python gaussian-splatting-main/train.py ^
  -s data/chair ^
  -m data/chair/official_3dgs_output_7000_black_r8 ^
  --iterations 7000 ^
  --test_iterations 7000 ^
  --save_iterations 7000 ^
  --checkpoint_iterations 7000 ^
  --eval --data_device cpu --disable_viewer -r 8
```

该数据的 PNG 为 RGBA，透明区域 RGB 为黑色。官方训练时使用 alpha mask 处理透明区域，因此后续定量评估也采用同样 mask 口径。

## Evaluation

简化版评估使用 `checkpoint_000180.pt` 在 100 个训练视角上渲染并计算 MAE、MSE、PSNR。

官方 3DGS 评估使用 `eval_official_3dgs.py`，加载 `iteration_7000` 模型，对 train/test 视角分别计算指标：

```eval
python eval_official_3dgs.py ^
  --model_path data/chair/official_3dgs_output_7000_black_r8 ^
  --iteration 7000 ^
  --resolution 8 ^
  --output data/chair/official_3dgs_metrics_7000_black_r8.json
```

官方渲染图生成命令：

```eval
python gaussian-splatting-main/render.py -m data/chair/official_3dgs_output_7000_black_r8 --iteration 7000 --quiet
```

## Pre-trained Models

本实验不提供可下载的预训练模型。训练得到的本地产物如下：

| 产物 | 路径 |
| --- | --- |
| 简化版 checkpoint | `data/chair/checkpoints/checkpoint_000180.pt` |
| 简化版 debug 视频 | `data/chair/checkpoints/debug_rendering.mp4` |
| 简化版 orbit 视频 | `data/chair/render_mv_checkpoint_180.mp4` |
| 官方 3DGS 点云 | `data/chair/official_3dgs_output_7000_black_r8/point_cloud/iteration_7000/point_cloud.ply` |
| 官方 3DGS 渲染图 | `data/chair/official_3dgs_output_7000_black_r8/train/ours_7000/` 与 `test/ours_7000/` |

其中 checkpoint、点云和官方完整输出体积较大，建议只在本地保留，不上传到普通 Git 仓库。为了便于报告展示，已将两个较小的视频复制到 `report_assets/`，可以随报告一起提交。

## Results

### Task 1: COLMAP 结果

COLMAP 从 100 张 `chair` 图像中恢复出 13,631 个稀疏 3D 点。重投影可视化结果表明，相机内外参与稀疏点云能够正确投影回多视角图像，可以作为后续 3DGS 初始化。

### Task 2: 简化版 3DGS 结果

| 指标 | 数值 |
| --- | --- |
| 训练图像数 | 100 |
| 渲染分辨率 | 100 x 100 |
| 最后评估 checkpoint | `checkpoint_000180.pt` |
| MAE | 0.021401 |
| MSE | 0.006115 |
| PSNR | 22.14 dB |

简化版能够恢复椅子的主体结构和主要颜色，但因为没有 adaptive densification、tile-based rasterizer 和 spherical harmonics 颜色模型，边缘、细节和遮挡处仍存在模糊与半透明伪影。

#### 可视化结果

训练视角 debug 视频展示了训练图像与简化版渲染结果的并排对比：

<video src="report_assets/chair_training_debug.mp4" controls width="720"></video>

绕物体一圈的 orbit 渲染视频展示了简化版 3DGS 在连续新视角下的重建效果：

<video src="report_assets/chair_orbit_render.mp4" controls width="720"></video>

### Task 3: 与官方 3DGS 的对比

| 对比项 | 本作业纯 PyTorch 简化版 | 官方 3DGS 实现 |
| --- | --- | --- |
| 数据 | `chair`，100 个训练视角 | `chair`，87 个 train + 13 个 test 视角 |
| 分辨率 | 100 x 100 | 100 x 100 |
| 训练长度 | 200 epochs，使用 `checkpoint_000180.pt` 评估 | 7000 iterations，`iteration_7000` |
| MAE | 0.021401 | train 0.005243，test 0.009576 |
| MSE | 0.006115 | train 0.000543，test 0.002331 |
| PSNR | 22.14 dB | train 33.44 dB，test 30.33 dB |
| 训练速度 | 基准约 0.997 s/step | 47.819 s / 7000 iterations，约 146 iter/s |
| 显存占用 | 5-step 基准峰值约 8995 MiB，完整训练曾触发 OOM | 100x100 训练峰值约 1792 MiB |

额外参考：官方实现在原始 800x800 分辨率下训练 7000 iterations 用时 140.804 s，显存峰值约 2982 MiB；在 alpha mask 口径下 train/test PSNR 分别约 34.66 dB / 33.84 dB。

差异来源主要包括：

1. 官方实现使用 CUDA tile-based rasterizer，只对相关 tile 内的 Gaussian 做排序和混合；简化版依赖 PyTorch 张量广播，容易产生巨大中间张量。
2. 官方实现包含 adaptive densification 和 pruning，可在训练中 clone/split/prune Gaussian，使表面覆盖更充分；简化版只使用 COLMAP 稀疏点初始化。
3. 官方实现使用 spherical harmonics 表达视角相关颜色；简化版每个 Gaussian 只有固定 RGB，表达能力较弱。
4. 官方实现具有可见性裁剪、屏幕空间半径统计和高效排序策略；简化版 alpha blending 更直接，遮挡边界更容易出现模糊。
5. 简化版更适合理解 3DGS 数学核心；官方版本工程复杂度更高，但渲染质量、训练速度和显存效率都明显更好。

## Contributing

本项目为课程作业代码，主要用于复现实验流程和提交报告。若继续维护，建议只提交核心源码、报告、必要脚本和小型配置文件；数据、训练产物、第三方源码与编译缓存通过 `.gitignore` 排除。

## 总结

本实验完成了从 COLMAP 相机恢复、3D Gaussian 初始化、投影、2D Gaussian 计算、alpha blending 到多视角渲染的完整流程。简化版实现清晰展示了 3DGS 的核心数学结构；官方实现则展示了工程优化、CUDA rasterizer、自适应增密和更强颜色模型对质量与效率的显著提升。
