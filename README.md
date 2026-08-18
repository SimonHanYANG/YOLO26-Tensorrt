# YOLO26-TensorRT

基于 Ultralytics YOLO26 的目标检测项目，完成从训练到 TensorRT 部署的全流程。

## 项目流程

```
训练 (PyTorch) → 验证 (mAP) → 导出 TensorRT Engine (FP32/FP16) → 推理 (图片/视频/跟踪/检测+分割)
```

## 快速开始

### 环境要求

- Python 3.10
- PyTorch 2.10+cu128
- Ultralytics 8.4.120
- NVIDIA GPU（CUDA 12.8+）

### 安装

```bash
conda create -n yolo python=3.10
conda activate yolo
pip install ultralytics
```

### 典型工作流

所有脚本支持 `--model n/s/m` 参数，输出目录自动区分（`train_n/`、`train_s/`、`train_m/`）。

```bash
# 1. 训练
python train.py                          # YOLO26-N (默认)
python train.py --model yolo26s.pt       # YOLO26-S
python train.py --model yolo26m.pt       # YOLO26-M

# 2. 验证 mAP
python val.py --model n
python val.py --model s
python val.py --model m

# 3. 图片可视化
python visualize_predictions.py --model n

# 4. 导出 TensorRT（FP32 + FP16）
python export_engine.py --model n
python export_engine.py --model s
python export_engine.py --model m

# 5. TensorRT 图片推理 + 测速
python tensorrt_visualize_benchmark.py --engine runs/detect/train_n/weights/best.engine

# 6. TensorRT 视频推理（检测）
python tensorrt_video_test.py --engine runs/detect/train_n/weights/best.engine

# 7. TensorRT 视频推理（跟踪 + 轨迹）
python tensorrt_video_test.py --engine runs/detect/train_n/weights/best.engine --mode track

# 8. 检测 + 分割联合推理（需要先准备好 1920x1200 视频）
python det_seg_video.py --engine runs/detect/train_n/weights/best.engine --video gray_video_1920x1200.mp4
```

### 输出目录

| 模型 | 训练目录 | Engine 文件 |
|------|----------|-------------|
| YOLO26-N | `runs/detect/train_n/` | `best.engine`, `best_fp16.engine` |
| YOLO26-S | `runs/detect/train_s/` | `best.engine`, `best_fp16.engine` |
| YOLO26-M | `runs/detect/train_m/` | `best.engine`, `best_fp16.engine` |

## 文件说明

| 文件 | 用途 |
|------|------|
| `train.py` | 训练脚本，加载 yolo26n.pt 进行 fine-tune |
| `val.py` | 验证脚本，评估 best.pt 的 mAP |
| `export_engine.py` | 导出 TensorRT engine（FP32 + FP16） |
| `visualize_predictions.py` | 图片推理可视化（原图大小 1920x1200） |
| `tensorrt_visualize_benchmark.py` | TensorRT 图片推理 + 速度测试 |
| `tensorrt_video_test.py` | 视频推理主脚本：检测/跟踪，统一 1920x1200 输出 |
| `det_seg_video.py` | 检测+分割联合推理：TensorRT 检测 + ByteTrack 跟踪 + ENet 分割，输出 4 种视频 |
| `tensorrt_video_inference.py` | 视频检测推理（旧版） |
| `tensorrt_video_tracker_inference.py` | 视频跟踪推理（旧版） |

## 数据集

- 类别：`sperm`（精子头部），nc=1
- 训练集：414 张 | 验证集：103 张
- 格式：标准 YOLO 格式（images/ + labels/ + data.yaml）

## 命令速查

详见 [commands.txt](commands.txt)
