from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
# tail
# results = model.train(data=r"dataset/yolo_dataset2_better/data.yaml", epochs=200, imgsz=640)
# head
results = model.train(data=r"dataset/Xiangya-yolo-head-dataset-260817/data.yaml", epochs=200, imgsz=640, device=1)

