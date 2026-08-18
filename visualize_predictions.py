"""
visualize_predictions.py - 图片推理可视化
用法:
    conda run -n yolo python visualize_predictions.py                # 默认 n 模型
    conda run -n yolo python visualize_predictions.py --model s      # s 模型
    conda run -n yolo python visualize_predictions.py --model m      # m 模型
输出:
    visualize_results_{model}/pred/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="图片推理可视化")
    parser.add_argument("--model", default="n", help="模型大小 (n/s/m/l/x)")
    parser.add_argument("--weights", default=None, help="自定义权重路径")
    parser.add_argument("--data", default="dataset/Xiangya-yolo-head-dataset-260817/data.yaml", help="数据集配置")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    parser.add_argument("--device", type=int, default=1, help="GPU 设备号")
    args = parser.parse_args()

    weights = args.weights or f"runs/detect/train_{args.model}/weights/best.pt"
    output_dir = f"visualize_results_{args.model}"

    # 从 data.yaml 读取 val 图片路径
    import yaml
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    image_dir = str(Path(data_cfg['path']) / data_cfg['val'])

    print(f"权重: {weights}")
    print(f"图片: {image_dir}")
    print(f"输出: {output_dir}/pred/")

    model = YOLO(weights)
    results = model.predict(
        source=image_dir,
        save=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=640,
        project=str(Path(output_dir).resolve()),
        name="pred",
        exist_ok=True,
        device=args.device,
    )

    total_images = len(results)
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n{'='*60}")
    print(f"处理图片: {total_images}, 检测数: {total_detections}")
    print(f"结果: {output_dir}/pred/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
