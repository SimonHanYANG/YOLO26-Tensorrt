"""
tensorrt_visualize_benchmark.py - TensorRT 图片推理可视化 + 速度测试
用法:
    conda run -n yolo python tensorrt_visualize_benchmark.py --engine runs/detect/train_n/weights/best.engine
    conda run -n yolo python tensorrt_visualize_benchmark.py --engine runs/detect/train_s/weights/best.engine
    conda run -n yolo python tensorrt_visualize_benchmark.py --engine runs/detect/train_m/weights/best_fp16.engine
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="TensorRT 图片推理 + 测速")
    parser.add_argument("--engine", required=True, help="TensorRT engine 路径")
    parser.add_argument("--data", default="dataset/Xiangya-yolo-head-dataset-260817/data.yaml", help="数据集配置")
    parser.add_argument("--output", default=None, help="输出目录 (默认: trt_vis_<engine名>)")
    parser.add_argument("--warmup", type=int, default=5, help="warmup 次数")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    parser.add_argument("--device", type=int, default=1, help="GPU 设备号")
    args = parser.parse_args()

    engine_name = Path(args.engine).stem
    if args.output is None:
        args.output = f"trt_vis_{engine_name}"

    # 读取图片路径
    import yaml
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    image_dir = Path(data_cfg['path']) / data_cfg['val']
    images = sorted([str(p) for p in image_dir.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])

    print(f"Engine: {args.engine}")
    print(f"图片: {image_dir} ({len(images)} 张)")
    print(f"输出: {args.output}/pred/")

    # 加载模型
    model = YOLO(args.engine)

    # Warmup
    print(f"\nWarmup {args.warmup} 次 ...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(args.warmup):
        model(dummy, conf=args.conf, iou=args.iou, verbose=False, device=args.device)

    # 推理 + 计时
    print(f"\n开始推理 {len(images)} 张图片 ...")
    times = []
    out_dir = Path(args.output) / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        img = cv2.imread(img_path)
        t0 = time.time()
        results = model(img, conf=args.conf, iou=args.iou, verbose=False, device=args.device)
        t1 = time.time()
        times.append(t1 - t0)

        # 保存带标注的图片
        result_img = results[0].plot()
        cv2.imwrite(str(out_dir / Path(img_path).name), result_img)

    # 统计
    times_ms = np.array(times) * 1000
    print(f"\n{'='*60}")
    print(f"推理完成: {len(images)} 张")
    print(f"  平均: {times_ms.mean():.1f} ms")
    print(f"  中位: {np.median(times_ms):.1f} ms")
    print(f"  最快: {times_ms.min():.1f} ms")
    print(f"  最慢: {times_ms.max():.1f} ms")
    print(f"  FPS:  {1000 / times_ms.mean():.1f}")
    print(f"结果: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
