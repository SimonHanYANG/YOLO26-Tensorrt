"""
TensorRT 模型可视化 + 推理速度测试
用法:
  python tensorrt_visualize_benchmark.py --engine <engine路径>
  python tensorrt_visualize_benchmark.py --engine best.engine         # FP32
  python tensorrt_visualize_benchmark.py --engine best_fp16.engine    # FP16
"""

import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

# ── 默认配置 ──────────────────────────────────────────────────────
IMAGE_DIR = "dataset/Xiangya-yolo-head-dataset-260817/images/val"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMG_SIZE = 640
WARMUP_RUNS = 5
DEVICE = 1


def main():
    parser = argparse.ArgumentParser(description="TensorRT 可视化 + 推理测速")
    parser.add_argument("--engine", type=str, required=True,
                        help="TensorRT engine 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="可视化输出目录（默认: trt_vis_<engine名>）")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS,
                        help=f"warmup 次数（默认 {WARMUP_RUNS}）")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"置信度阈值（默认 {CONF_THRESHOLD}）")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                        help=f"IoU 阈值（默认 {IOU_THRESHOLD}）")
    args = parser.parse_args()

    engine_path = Path(args.engine)
    if not engine_path.exists():
        # 尝试在 weights 目录下找
        alt = Path("runs/detect/train/weights") / engine_path.name
        if alt.exists():
            engine_path = alt
        else:
            print(f"错误: 找不到 engine 文件 {args.engine}")
            return

    engine_name = engine_path.stem  # best / best_fp16
    output_dir = args.output or f"trt_vis_{engine_name}"

    # ── 加载模型 ──
    print(f"加载 TensorRT engine: {engine_path}")
    model = YOLO(str(engine_path), task="detect")

    # ── 收集图片路径 ──
    image_dir = Path(IMAGE_DIR)
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    print(f"找到 {len(image_paths)} 张图片\n")

    # ── Warmup ──
    print(f"Warmup {args.warmup} 次...")
    for i in range(args.warmup):
        _ = model.predict(
            source=str(image_paths[0]),
            conf=args.conf,
            iou=args.iou,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )
    print("Warmup 完成\n")

    # ── 正式推理 + 计时 ──
    print("开始推理...")
    times = []
    all_results = []

    for idx, img_path in enumerate(image_paths):
        t0 = time.perf_counter()
        result = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )[0]
        t1 = time.perf_counter()

        times.append(t1 - t0)
        all_results.append(result)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"  [{idx+1}/{len(image_paths)}] 已完成")

    # ── 保存可视化结果 ──
    print("\n保存可视化图片...")
    output_path = Path(output_dir).resolve() / "pred"
    output_path.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        plotted = result.plot()
        save_name = Path(result.path).name
        cv2.imwrite(str(output_path / save_name), plotted)

    print(f"可视化结果保存在: {output_path}\n")

    # ── 统计 ──
    times_arr = np.array(times)
    total_detections = sum(len(r.boxes) for r in all_results)

    print("=" * 60)
    print(f"TensorRT 推理统计 — {engine_name}")
    print("=" * 60)
    print(f"Engine 文件:       {engine_path}")
    print(f"图片数量:          {len(image_paths)}")
    print(f"Warmup 次数:       {args.warmup}")
    print(f"总检测数:          {total_detections}")
    print("-" * 60)
    print(f"平均每张推理时间:  {times_arr.mean() * 1000:.2f} ms")
    print(f"最快:              {times_arr.min() * 1000:.2f} ms")
    print(f"最慢:              {times_arr.max() * 1000:.2f} ms")
    print(f"中位数:            {np.median(times_arr) * 1000:.2f} ms")
    print(f"标准差:            {times_arr.std() * 1000:.2f} ms")
    print(f"FPS:               {1.0 / times_arr.mean():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
