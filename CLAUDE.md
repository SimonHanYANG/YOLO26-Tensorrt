# CLAUDE.md — YOLO26-TensorRT 项目

## 对话规则

1. 每次对话都要叫用户**涵哥**
2. 不要主动查看图片、视频等多模态内容（Read 工具不要用于 .jpg/.png/.mp4 等文件）
3. 运行代码时使用 **cuda:1**（`device=1`）
4. 使用名为 **yolo** 的 conda 环境运行所有命令：`conda run -n yolo python ...`

## 项目概述

这是一个基于 **Ultralytics YOLO** 的目标检测项目，流程为：训练 → 导出 ONNX → 导出 TensorRT Engine → TensorRT 推理（图片/视频/视频跟踪）。

预训练权重：`yolo26n.pt`（YOLO26 nano）

## 文件结构

| 文件 | 用途 |
|------|------|
| `train.py` | 训练脚本，加载 yolo26n.pt 进行 fine-tune |
| `val.py` | 验证脚本，评估 best.pt 的 mAP |
| `export_onnx.py` | 将 best.pt 导出为 ONNX 格式 |
| `export_engine.py` | 将 best.pt 导出为 TensorRT engine（FP32 + FP16） |
| `tensorrt_inference.py` | 单张图片的 TensorRT 推理 |
| `tensorrt_visualize_benchmark.py` | TensorRT 图片推理可视化 + 速度测试（支持命令行参数） |
| `tensorrt_video_inference.py` | 视频的 TensorRT 检测推理，输出带标注的视频和日志 |
| `tensorrt_video_tracker_inference.py` | 视频的 TensorRT 跟踪推理（ByteTrack/BotSort），输出带跟踪ID的视频和日志 |
| `tensorrt_video_test.py` | 视频推理主脚本：检测+跟踪，绿色bbox，轨迹绘制，统一1920x1200输出，支持命令行参数 |
| `det_seg_video.py` | **新** 检测+分割联合推理：TensorRT检测+ByteTrack+ENet分割，输出4种视频（full/det_track/seg/det_seg） |
| `visualize_predictions.py` | 用训练好的模型对图片做推理，保存原图大小（1920x1200）带 bbox 叠加的可视化结果 |

## 数据集

位于 `dataset/` 目录下，当前使用的数据集为 `dataset/Xiangya-yolo-head-dataset-260817/`：

- 类别：`sperm`（精子头部），nc=1
- 训练集：414 张图片
- 验证集：103 张图片
- 格式：标准 YOLO 格式（images/ + labels/ + data.yaml）

## 典型工作流（支持 n/s/m 多模型）

所有脚本支持 `--model` 参数选择模型大小，输出目录自动区分。

```
# YOLO26-N (默认)
1. 训练:       conda run -n yolo python train.py
2. 验证:       conda run -n yolo python val.py --model n
3. 可视化:     conda run -n yolo python visualize_predictions.py --model n
4. 导出TRT:    conda run -n yolo python export_engine.py --model n
5. 图片测速:   conda run -n yolo python tensorrt_visualize_benchmark.py --engine runs/detect/train_n/weights/best.engine
6. 视频推理:   conda run -n yolo python tensorrt_video_test.py --engine runs/detect/train_n/weights/best.engine
7. 视频跟踪:   conda run -n yolo python tensorrt_video_test.py --engine runs/detect/train_n/weights/best.engine --mode track
8. 检测+分割:  conda run -n yolo python det_seg_video.py --engine runs/detect/train_n/weights/best.engine --video dataset/XiangYa-test-videoes/gray_video_1920x1200.mp4

# YOLO26-S / M (把 n 换成 s 或 m 即可)
1. 训练:       conda run -n yolo python train.py --model yolo26s.pt
2. 验证:       conda run -n yolo python val.py --model s
3. 可视化:     conda run -n yolo python visualize_predictions.py --model s
4. 导出TRT:    conda run -n yolo python export_engine.py --model s
5. 图片测速:   conda run -n yolo python tensorrt_visualize_benchmark.py --engine runs/detect/train_s/weights/best.engine
6. 视频推理:   conda run -n yolo python tensorrt_video_test.py --engine runs/detect/train_s/weights/best.engine
7. 视频跟踪:   conda run -n yolo python tensorrt_video_test.py --engine runs/detect/train_s/weights/best.engine --mode track
8. 检测+分割:  conda run -n yolo python det_seg_video.py --engine runs/detect/train_s/weights/best.engine --video dataset/XiangYa-test-videoes/gray_video_1920x1200.mp4
```

## 当前进度

- [x] 环境配置：yolo conda 环境，ultralytics 8.4.120 已安装
- [x] 数据集配置：`data.yaml` 的 `path` 已改为服务器路径
- [x] 所有脚本支持 `--model n/s/m` 多模型参数
- [x] YOLO26-N: 200 epochs 训练完成
- [ ] YOLO26-S: 待训练
- [ ] YOLO26-M: 待训练
- [x] TensorRT engine 导出（FP32 + FP16）
- [x] `tensorrt_visualize_benchmark.py` 图片推理可视化 + 测速
- [x] `tensorrt_video_test.py` 视频推理（检测+跟踪），统一 1920x1200 输出
- [x] `det_seg_video.py` 检测+分割联合推理，输出 4 种视频
- [ ] 验证 mAP（val.py）
- [ ] 导出 ONNX（export_onnx.py）

## 输出目录约定

| 模型 | 训练目录 | Engine |
|------|----------|--------|
| N | `runs/detect/train_n/` | `best.engine`, `best_fp16.engine` |
| S | `runs/detect/train_s/` | `best.engine`, `best_fp16.engine` |
| M | `runs/detect/train_m/` | `best.engine`, `best_fp16.engine` |

## 环境

- Python 3.10, PyTorch 2.10.0+cu128
- Ultralytics 8.4.120
- GPU: NVIDIA RTX 5090 (32GB) × 2（默认使用 cuda:1）

## 重要：PyTorch 版本

**PyTorch 必须保持 cu128 版本**（2.10.0+cu128），不要升级到 cu130。
当前 NVIDIA 驱动版本 12080，只支持到 CUDA 12.8。
如果误升级到 cu130 会导致 `torch.cuda.is_available()=False`，所有 GPU 代码都会报错。

修复命令：
```
conda run -n yolo pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 外部依赖：分割项目

`det_seg_video.py` 依赖分割项目 `/root/Segmentation-Model-Zoo-for-Super-Small-Object/`：
- ENet 模型：`models/ENet_sperm_ROINAHead_Xiangya_260817/model.pth` + `config.yml`
- 分割类别（4类）：class 0=non-measurable head(红), class 1=nuclear(绿), class 2=acrosome(蓝), class 3=mid-piece(黄)
- 输入：64x64 ROI，**BGR** 格式（不做 RGB 转换）
- 输出：4 通道 sigmoid 二值 mask
- 直接 import `model_zoo.enet.ENet`，不走 archs.py（archs.py 依赖 timm 等）

### ⚠️ 分割预处理注意事项（重要）

训练代码 `dataset.py` 有**双重 /255** 的行为：
1. `A.Normalize()` 对 uint8 输入先 `/255` 再 ImageNet 归一化 → 输出 float32 ≈ [-2.1, 2.6]
2. `dataset.py` 之后又 `img / 255` → 输出 ≈ [-0.008, 0.010]

推理时必须复现同样的预处理：
```python
roi_float = roi_bgr.astype('float32') / 255.0          # 第一次 /255
roi_norm = (roi_float - mean) / std                     # ImageNet 归一化
roi_final = roi_norm / 255.0                            # 第二次 /255 (训练代码的 bug)
```
同时保持 **BGR** 格式（训练用 cv2.imread 默认 BGR，不做 RGB 转换）。

分割项目使用 1920x1200 的预处理视频 `gray_video_1920x1200.mp4`（已保存到 dataset 目录）。

## 注意事项

- 训练输出保存在 `runs/detect/train/`
- TensorRT engine 文件较大，已在 .gitignore 中排除
- 视频推理和跟踪推理会分别输出到 `video_results/` 和 `tracker_video_results/`
- Ultralytics `predict` 的 `project` 参数默认嵌套在 `runs/detect/` 下；要用绝对路径才能输出到指定目录（如 `visualize_predictions.py` 的做法）
- 原始图片分辨率为 **1920x1200**，推理内部 resize 到 640，但 `save=True` 输出的可视化图片保持原图大小
- `.gitignore` 已排除 dataset/、runs/、yolo26n.pt 等大文件
- `tensorrt_video_test.py` 会将所有帧统一 resize 到 1920x1200 后再推理和保存
- 视频测试素材：`dataset/XiangYa-test-videoes/gray_video.mp4`（原始 3456x2160）
- 预处理视频：`dataset/XiangYa-test-videoes/gray_video_1920x1200.mp4`（1920x1200，det_seg_video.py 使用）
- `tensorrt_visualize_benchmark.py` 和 `tensorrt_video_test.py` 支持命令行参数 `--engine`、`--mode` 等
