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
| `tensorrt_video_test.py` | **新** 视频推理主脚本：检测+跟踪，绿色bbox，轨迹绘制，统一1920x1200输出，支持命令行参数 |
| `visualize_predictions.py` | 用训练好的模型对图片做推理，保存原图大小（1920x1200）带 bbox 叠加的可视化结果 |

## 数据集

位于 `dataset/` 目录下，当前使用的数据集为 `dataset/Xiangya-yolo-head-dataset-260817/`：

- 类别：`sperm`（精子头部），nc=1
- 训练集：414 张图片
- 验证集：103 张图片
- 格式：标准 YOLO 格式（images/ + labels/ + data.yaml）

## 典型工作流

```
1. 训练:       conda run -n yolo python train.py
2. 验证:       conda run -n yolo python val.py
3. 可视化:     conda run -n yolo python visualize_predictions.py
4. 导出TRT:    conda run -n yolo python export_engine.py          # 导出 FP32 + FP16
5. 图片测速:   conda run -n yolo python tensorrt_visualize_benchmark.py --engine best.engine
6. 视频推理:   conda run -n yolo python tensorrt_video_test.py --engine best.engine
7. 视频跟踪:   conda run -n yolo python tensorrt_video_test.py --engine best.engine --mode track
```

## 当前进度

- [x] 环境配置：yolo conda 环境，ultralytics 8.4.120 已安装
- [x] 数据集配置：`data.yaml` 的 `path` 已改为服务器路径
- [x] `train.py` 已修改：数据集指向 `Xiangya-yolo-head-dataset-260817`，device=1
- [x] 1 epoch 训练测试通过（414 train / 103 val，1753 instances）
- [x] 200 epochs 正式训练完成
- [x] `visualize_predictions.py` 编写完成并测试通过
- [x] TensorRT engine 导出完成：`best.engine`（FP32）、`best_fp16.engine`（FP16）
- [x] `tensorrt_visualize_benchmark.py` 图片推理可视化 + 测速
- [x] `tensorrt_video_test.py` 视频推理（检测+跟踪），统一 1920x1200 输出
- [ ] 验证 mAP（val.py）
- [ ] 导出 ONNX（export_onnx.py）

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

## 注意事项

- 训练输出保存在 `runs/detect/train/`
- TensorRT engine 文件较大，已在 .gitignore 中排除
- 视频推理和跟踪推理会分别输出到 `video_results/` 和 `tracker_video_results/`
- Ultralytics `predict` 的 `project` 参数默认嵌套在 `runs/detect/` 下；要用绝对路径才能输出到指定目录（如 `visualize_predictions.py` 的做法）
- 原始图片分辨率为 **1920x1200**，推理内部 resize 到 640，但 `save=True` 输出的可视化图片保持原图大小
- `.gitignore` 已排除 dataset/、runs/、yolo26n.pt 等大文件
- `tensorrt_video_test.py` 会将所有帧统一 resize 到 1920x1200 后再推理和保存
- 视频测试素材：`dataset/XiangYa-test-videoes/gray_video.mp4`
- `tensorrt_visualize_benchmark.py` 和 `tensorrt_video_test.py` 支持命令行参数 `--engine`、`--mode` 等
