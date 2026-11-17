from ultralytics import YOLO

model = YOLO("修复后的配置.yaml")
model.info()  # 打印模型结构，无报错则通道数匹配
