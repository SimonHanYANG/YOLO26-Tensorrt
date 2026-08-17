from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# 导出 FP16 engine
print("导出 FP16 engine ...")
model.export(format="engine", half=True, device=1)  # 输出 best.engine (被覆盖)

# 重命名为 best_fp16.engine
import shutil
shutil.move("runs/detect/train/weights/best.engine",
            "runs/detect/train/weights/best_fp16.engine")
print("FP16 保存为: runs/detect/train/weights/best_fp16.engine\n")

# 导出 FP32 engine
print("导出 FP32 engine ...")
model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="engine", half=False, device=1)  # 输出 best.engine
print("FP32 保存为: runs/detect/train/weights/best.engine")
