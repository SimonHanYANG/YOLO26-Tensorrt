from ultralytics import YOLO
from pathlib import Path

# ============================================================
# 配置参数
# ============================================================
MODEL_PATH = "runs/detect/train/weights/best.pt"  # 训练好的模型
IMAGE_DIR = "dataset/Xiangya-yolo-head-dataset-260817/images/val"  # 待推理的图片目录
OUTPUT_DIR = "visualization_results"  # 可视化输出目录
CONF_THRESHOLD = 0.25  # 置信度阈值
IOU_THRESHOLD = 0.7    # NMS IOU 阈值
IMG_SIZE = 640         # 推理图片尺寸


def main():
    # 加载模型
    model = YOLO(MODEL_PATH)

    # 对整个目录做推理，save=True 会自动保存带 bbox 标注的图片
    results = model.predict(
        source=IMAGE_DIR,
        save=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        project=str(Path(OUTPUT_DIR).resolve()),
        name="pred",
        exist_ok=True,
        device=1,
    )

    # 打印统计
    total_images = len(results)
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n{'='*60}")
    print(f"可视化完成！")
    print(f"{'='*60}")
    print(f"处理图片数: {total_images}")
    print(f"总检测数:   {total_detections}")
    print(f"结果保存在: {Path(OUTPUT_DIR) / 'pred'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
