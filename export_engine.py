"""
export_engine.py - 导出 TensorRT Engine (FP32 + FP16)
用法:
    conda run -n yolo python export_engine.py                # 默认 n 模型
    conda run -n yolo python export_engine.py --model s      # s 模型
    conda run -n yolo python export_engine.py --model m      # m 模型
输出:
    runs/detect/train_{model}/weights/best.engine        (FP32)
    runs/detect/train_{model}/weights/best_fp16.engine   (FP16)
"""

import argparse
import shutil
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="导出 TensorRT Engine")
    parser.add_argument("--model", default="26n", help="模型标识 (26n/26s/26m)")
    parser.add_argument("--weights", default=None, help="自定义权重路径")
    parser.add_argument("--device", type=int, default=1, help="GPU 设备号")
    args = parser.parse_args()

    weights = args.weights or f"runs/detect/train_{args.model}/weights/best.pt"
    out_dir = weights.rsplit("/", 1)[0]

    print(f"权重: {weights}")

    # 导出 FP16
    print("\n导出 FP16 engine ...")
    model = YOLO(weights)
    model.export(format="engine", half=True, device=args.device)
    fp16_path = f"{out_dir}/best_fp16.engine"
    shutil.move(f"{out_dir}/best.engine", fp16_path)
    print(f"FP16 保存为: {fp16_path}")

    # 导出 FP32
    print("\n导出 FP32 engine ...")
    model = YOLO(weights)
    model.export(format="engine", half=False, device=args.device)
    fp32_path = f"{out_dir}/best.engine"
    print(f"FP32 保存为: {fp32_path}")


if __name__ == "__main__":
    main()
