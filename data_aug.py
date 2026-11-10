import cv2
import numpy as np
import os
import shutil
from pathlib import Path


def adjust_annotation_rotation(label_lines, img_w, img_h, angle):
    """
    旋转图像后同步调整标注框坐标
    :param label_lines: 原始标注行列表（YOLO格式）
    :param img_w: 原始图像宽度
    :param img_h: 原始图像高度
    :param angle: 旋转角度（仅支持90/180/270°，避免非整数旋转导致的复杂计算）
    :return: 调整后的标注行列表
    """
    new_labels = []
    for line in label_lines:
        if not line.strip():
            continue
        # 解析YOLO标注：cls x y w h（归一化）
        parts = line.strip().split()
        cls = parts[0]
        x, y, w, h = map(float, parts[1:])
        # 提前初始化
        new_x, new_y, new_w, new_h = x, y, w, h
        # 反归一化：转像素坐标（中心x, 中心y, 宽, 高）
        x_pix = x * img_w
        y_pix = y * img_h
        w_pix = w * img_w
        h_pix = h * img_h

        # 旋转中心（图像中心）
        cx = img_w / 2
        cy = img_h / 2

        # 根据旋转角度调整坐标
        if angle == 90:
            # 90°顺时针旋转：新宽=原高，新高=原宽
            new_x_pix = cy + (y_pix - cy)
            new_y_pix = cx - (x_pix - cx)
            new_w_pix = h_pix
            new_h_pix = w_pix
            new_w = new_w_pix / img_h  # 新宽度是原高度
            new_h = new_h_pix / img_w
            new_x = new_x_pix / img_h
            new_y = new_y_pix / img_w
        elif angle == 180:
            # 180°旋转
            new_x_pix = cx - (x_pix - cx)
            new_y_pix = cy - (y_pix - cy)
            new_x = new_x_pix / img_w
            new_y = new_y_pix / img_h
            new_w = w
            new_h = h
        elif angle == 270:
            # 270°顺时针旋转
            new_x_pix = cy - (y_pix - cy)
            new_y_pix = cx + (x_pix - cx)
            new_w_pix = h_pix
            new_h_pix = w_pix
            new_w = new_w_pix / img_h
            new_h = new_h_pix / img_w
        else:
            # 0°不调整
            new_x, new_y, new_w, new_h = x, y, w, h

        # 确保坐标在0-1范围内（防止越界）
        new_x = np.clip(new_x, 0, 1)
        new_y = np.clip(new_y, 0, 1)
        new_w = np.clip(new_w, 0, 1)
        new_h = np.clip(new_h, 0, 1)

        new_labels.append(f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")
    return new_labels


def augment_single_image(img_path, label_path, save_img_dir, save_label_dir, augment_type):
    """
    对单张图像做指定类型的增强，并同步保存标注
    :param img_path: 原始图像路径
    :param label_path: 原始标注路径
    :param save_img_dir: 增强后图像保存目录
    :param save_label_dir: 增强后标注保存目录
    :param augment_type: 增强类型：flip（水平翻转）/ rotate90/ rotate180/ rotate270/ brightness
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图像 {img_path}，跳过")
        return
    h, w = img.shape[:2]
    img_name = Path(img_path).name
    label_name = Path(label_path).name

    # 读取标注
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
    else:
        print(f"警告：标注文件 {label_path} 不存在，跳过")
        label_lines = []

    # 根据增强类型处理图像和标注
    if augment_type == "flip":
        # 水平翻转
        aug_img = cv2.flip(img, 1)
        # 调整标注：x_center = 1 - x_center
        new_label_lines = []
        for line in label_lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = parts[0]
            x, y, w_box, h_box = map(float, parts[1:])
            new_x = 1 - x
            new_label_lines.append(f"{cls} {new_x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

    elif augment_type.startswith("rotate"):
        # 旋转（仅支持90/180/270°）
        angle = int(augment_type.replace("rotate", ""))
        if angle == 90:
            aug_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            aug_img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            aug_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"警告：不支持的旋转角度 {angle}，跳过")
            return
        # 调整标注
        new_label_lines = adjust_annotation_rotation(label_lines, w, h, angle)


    elif augment_type == "brightness":
        # 亮度调整：随机增减亮度（替代固定+30）
        beta_range = (-50, 80)  # 可自定义明暗范围
        beta = np.random.uniform(beta_range[0], beta_range[1])
        aug_img = cv2.convertScaleAbs(img, alpha=1, beta=beta)
        # 亮度调整不改变标注
        new_label_lines = label_lines

    else:
        print(f"警告：不支持的增强类型 {augment_type}，跳过")
        return

    # 保存增强后的图像和标注
    save_img_path = os.path.join(save_img_dir, f"{augment_type}_{img_name}")
    save_label_path = os.path.join(save_label_dir, f"{augment_type}_{label_name}")

    cv2.imwrite(save_img_path, aug_img)
    with open(save_label_path, 'w', encoding='utf-8') as f:
        f.writelines(new_label_lines)

    print(f"完成增强：{save_img_path}")


def batch_augment_dataset(raw_img_dir, raw_label_dir, save_root_dir, augment_types=["flip", "rotate90","rotate18","rotate270", "brightness"]):
    """
    批量增强数据集
    :param raw_img_dir: 原始图像目录（如 dataset/images/train）
    :param raw_label_dir: 原始标注目录（如 dataset/labels/train）
    :param save_root_dir: 增强后数据保存根目录（会自动创建images/labels子目录）
    :param augment_types: 要执行的增强类型列表
    """
    # 创建保存目录
    save_img_dir = os.path.join(save_root_dir, "images")
    save_label_dir = os.path.join(save_root_dir, "labels")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    # 复制原始文件到目标目录（保留原始数据）
    print("正在复制原始数据...")
    for img_file in os.listdir(raw_img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 复制图像
            shutil.copy(os.path.join(raw_img_dir, img_file), os.path.join(save_img_dir, img_file))
            # 复制对应标注
            label_file = Path(img_file).stem + ".txt"
            label_src = os.path.join(raw_label_dir, label_file)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(save_label_dir, label_file))

    # 批量执行增强
    print("开始批量增强...")
    for img_file in os.listdir(raw_img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        # 拼接原始图像和标注路径
        img_path = os.path.join(raw_img_dir, img_file)
        label_file = Path(img_file).stem + ".txt"
        label_path = os.path.join(raw_label_dir, label_file)

        # 对每种增强类型执行
        for aug_type in augment_types:
            augment_single_image(img_path, label_path, save_img_dir, save_label_dir, aug_type)

    print(f"批量增强完成！增强后数据保存在：{save_root_dir}")


if __name__ == "__main__":
    # ====================== 你需要修改的参数 ======================
    RAW_IMG_DIR = r"datasettest/images/train"  # 你的原始训练图像目录
    RAW_LABEL_DIR = r"datasettest/labels/train"  # 你的原始训练标注目录
    SAVE_ROOT_DIR = r"datasettest_aug/train"  # 增强后数据保存目录
    AUGMENT_TYPES = ["flip", "rotate90", "rotate180", "rotate270","brightness"]  # 要执行的增强类型
    # =============================================================

    # 执行批量增强
    batch_augment_dataset(RAW_IMG_DIR, RAW_LABEL_DIR, SAVE_ROOT_DIR, AUGMENT_TYPES)

# 本段代码主要是对数据集中的训练集进行训练前的数据增强，首先输入需要增强的文件夹
# 通过设置增强类型，对文件夹中的每一张图片进行翻转、旋转或亮度改变
# 最后将原始图像以及增强生成的图片全部保存到新的文件夹中
# 数据增强只需要对训练集进行操作即可，验证集和测试集不需要增强