# from ultralytics import YOLO
# from ultralytics import RTDETR

# # 加载模型
# model = YOLO('/mnt/workspace/ul/runs/detect/train51/weights/best.pt')
# # model = RTDETR('/mnt/workspace/ul/runs/detect/train43/weights/best.pt')

# # 评估模型
# metrics = model.val(split='test')

from ultralytics import YOLO
import os
import json
import pandas as pd
from datetime import datetime

# 加载模型
model_path = '/mnt/workspace/ul/runs/detect/train54/weights/best.pt'
model = YOLO(model_path)

# 创建保存结果的文件夹
train_name = model_path.split('/')[-3]
save_dir = f'/mnt/workspace/ul/runs/evaluation/{train_name}_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(save_dir, exist_ok=True)

print(f"结果将保存到: {save_dir}")

# 评估模型（开启保存图片和JSON）
metrics = model.val(
    split='test',
    save_json=True,      # 保存JSON格式的指标
    plots=True,          # 生成并保存曲线图
    save_dir=save_dir
)

# ===== 提取所有需要的数据 =====
# 1. 核心指标
map50 = metrics.box.map50      # mAP@0.5
map50_95 = metrics.box.map     # mAP@0.5:0.95
precision = metrics.box.mp     # 精确度
recall = metrics.box.mr        # 召回率

# 2. 速度指标（用于计算FPS）
speed = metrics.speed
preprocess_ms = speed.get('preprocess', 0)
inference_ms = speed.get('inference', 0)
postprocess_ms = speed.get('postprocess', 0)
total_time_ms = preprocess_ms + inference_ms + postprocess_ms
fps = 1000 / total_time_ms if total_time_ms > 0 else 0

# 3. 保存为文本报告
report_path = os.path.join(save_dir, 'metrics_summary.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write(f"模型评估报告\n")
    f.write(f"训练名称: {train_name}\n")
    f.write(f"模型路径: {model_path}\n")
    f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    
    f.write("【检测精度指标】\n")
    f.write(f"mAP50 (IoU=0.5):      {map50:.4f}\n")
    f.write(f"mAP50-95:             {map50_95:.4f}\n")
    f.write(f"精确度 (Precision):   {precision:.4f}\n")
    f.write(f"召回率 (Recall):      {recall:.4f}\n\n")
    
    f.write("【推理速度指标】\n")
    f.write(f"前处理时间:   {preprocess_ms:.2f} ms\n")
    f.write(f"推理时间:     {inference_ms:.2f} ms\n")
    f.write(f"后处理时间:   {postprocess_ms:.2f} ms\n")
    f.write(f"单张图片总耗时: {total_time_ms:.2f} ms\n")
    f.write(f"FPS (帧/秒):  {fps:.2f}\n\n")
    
    f.write("【生成的图片文件】\n")
    f.write(f"- PR曲线:     {save_dir}/PR_curve.png\n")
    f.write(f"- P曲线:      {save_dir}/P_curve.png\n")
    f.write(f"- R曲线:      {save_dir}/R_curve.png\n")
    f.write(f"- 混淆矩阵:   {save_dir}/confusion_matrix.png\n")
    f.write("="*60 + "\n")

# 4. 保存为CSV格式（方便Excel打开）
csv_path = os.path.join(save_dir, f'{train_name}_metrics.csv')
metrics_df = pd.DataFrame([
    {'指标': 'mAP50', '数值': map50},
    {'指标': 'mAP50-95', '数值': map50_95},
    {'指标': 'Precision', '数值': precision},
    {'指标': 'Recall', '数值': recall},
    {'指标': 'FPS', '数值': fps},
    {'指标': '总耗时(ms)', '数值': total_time_ms}
])
metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 5. 保存完整JSON
json_path = os.path.join(save_dir, f'{train_name}_full_metrics.json')
full_data = {
    '训练名称': train_name,
    '模型路径': model_path,
    '评估时间': datetime.now().isoformat(),
    '精度指标': {
        'mAP50': float(map50),
        'mAP50-95': float(map50_95),
        'Precision': float(precision),
        'Recall': float(recall)
    },
    '速度指标': {
        'preprocess_ms': preprocess_ms,
        'inference_ms': inference_ms,
        'postprocess_ms': postprocess_ms,
        'total_time_ms': total_time_ms,
        'FPS': fps
    }
}
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(full_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("✅ 评估完成！所有结果已保存：")
print(f"📁 保存目录: {save_dir}")
print(f"📊 文本报告: {report_path}")
print(f"📈 CSV文件: {csv_path}")
print(f"🔢 JSON文件: {json_path}")
print(f"🖼️ 曲线图片: 已保存在目录中")
print("="*60)
print(f"\n🎯 最终结果：")
print(f"   mAP50: {map50:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
print(f"   FPS: {fps:.2f} 帧/秒 | 单张耗时: {total_time_ms:.2f} ms")