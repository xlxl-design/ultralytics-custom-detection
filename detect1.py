from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11s.pt")

results = model("D:/new/yolo11/ultralytics-8.3.235/ultralytics/assets", conf=0.25, save=True)
