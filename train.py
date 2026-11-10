# 模型配置文件
# model_yaml_path = '/mnt/workspace/ul/ultralytics/cfg/models/11/yolo11s.yaml'
# model_yaml_path = '/mnt/workspace/ul/ultralytics/cfg/models/11/att_11/yolo11s-CBAM+CA.yaml'
model_yaml_path = "/mnt/workspace/ul/ultralytics/cfg/models/11/yolo11s-CA+cut.yaml"
# model_yaml_path = '/mnt/workspace/ul/ultralytics/cfg/models/v8/yolov8s.yaml'
# model_yaml_path = '/mnt/workspace/ul/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml'


# 数据集配置文件
data_yaml_path = "/mnt/workspace/ul/data.yaml"
# 预训练配置文件
pre_model_name = "/mnt/workspace/ul/yolo11s.pt"

import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model_yaml_path).load(pre_model_name)
    # model = YOLO('fasterrcnn_resnet50_fpn.pt')
    # model = RTDETR(model_yaml_path)
    model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=50,
        # patience=20,
        batch=16,
        workers=8,
        device=0,
        optimizer="AdamW",
        amp=True,
        conf=0.3,
        iou=0.4,
        # lr0=0.0005,            # 核心：降低初始学习率
        # lrf=0.001,             # 核心：降低最终学习率衰减
    )
