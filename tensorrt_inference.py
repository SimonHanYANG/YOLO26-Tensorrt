from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Load the exported TensorRT INT8 model
model = YOLO(r"/root/yolo26-tensorrt/runs/detect/train/weights/best.engine", task="detect")

# Run inference
result = model.predict(r"/root/yolo26-tensorrt/dataset/yolo_dataset2_better/val/images/263.png")