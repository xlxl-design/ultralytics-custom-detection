from ultralytics import YOLO

# 加载模型
model = YOLO("/mnt/workspace/ul/runs/detect/train35/weights/best.pt")

# 评估模型
metrics = model.val()  # 在验证集上评估
