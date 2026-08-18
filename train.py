"""
train.py - YOLO26 训练脚本
用法:
    conda run -n yolo python train.py                       # 默认 yolo26n
    conda run -n yolo python train.py --model yolo26s.pt    # YOLO26-S
    conda run -n yolo python train.py --model yolo26m.pt    # YOLO26-M
    conda run -n yolo python train.py --model yolo26s.pt --epochs 300 --imgsz 640
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 训练")
    parser.add_argument("--model", default="yolo26n.pt", help="预训练模型 (yolo26n.pt / yolo26s.pt / yolo26m.pt / ...)")
    parser.add_argument("--data", default="dataset/Xiangya-yolo-head-dataset-260817/data.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--device", type=int, default=1, help="GPU 设备号")
    parser.add_argument("--batch", type=int, default=-1, help="batch size (-1=auto)")
    parser.add_argument("--name", default=None, help="实验名称 (默认根据模型自动生成)")
    args = parser.parse_args()

    # 自动生成实验名称: yolo26n → train_26n, yolo26s → train_26s
    model_short = args.model.replace(".pt", "").replace(".yaml", "")  # yolo26s
    if args.name is None:
        args.name = f"train_{model_short.replace('yolo', '')}"  # train_26s

    print(f"模型: {args.model}")
    print(f"数据集: {args.data}")
    print(f"实验名称: {args.name}")
    print(f"epochs: {args.epochs}, imgsz: {args.imgsz}, device: cuda:{args.device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
