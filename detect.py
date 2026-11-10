import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/mnt/workspace/ul/runs/detect/train33/weights/best.pt")
    model.predict(
        source="/mnt/workspace/dataset/images/test",
        device=0,
        save=True,
    )

# metrics = model.val(split = 'test')
