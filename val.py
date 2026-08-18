"""
val.py - YOLO26 验证脚本 (mAP 评估)
用法:
    conda run -n yolo python val.py                       # 默认 n 模型
    conda run -n yolo python val.py --model s              # s 模型
    conda run -n yolo python val.py --model m              # m 模型
    conda run -n yolo python val.py --weights path/to/best.pt  # 自定义权重
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 验证 mAP")
    parser.add_argument("--model", default="n", help="模型大小 (n/s/m/l/x)")
    parser.add_argument("--weights", default=None, help="自定义权重路径 (优先于 --model)")
    parser.add_argument("--data", default="dataset/Xiangya-yolo-head-dataset-260817/data.yaml", help="数据集配置")
    parser.add_argument("--device", type=int, default=1, help="GPU 设备号")
    args = parser.parse_args()

    if args.weights:
        weights = args.weights
    else:
        weights = f"runs/detect/train_{args.model}/weights/best.pt"

    print(f"权重: {weights}")
    model = YOLO(weights)
    metrics = model.val(data=args.data, device=args.device)
    print(f"\nmAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
