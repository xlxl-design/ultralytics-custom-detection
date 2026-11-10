import os
from pathlib import Path
import time
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLODetectionDataset(Dataset):
    """
    支持目录结构（推荐）:
    dataset/
      images/
        train/xxx.jpg
        val/xxx.jpg
      labels/
        train/xxx.txt
        val/xxx.txt

    YOLO txt 每行:
    class_id cx cy w h   (归一化到 0~1)
    """
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        self.image_paths: List[Path] = []
        for p in self.images_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                self.image_paths.append(p)
        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _label_path_from_image(self, img_path: Path) -> Path:
        # 保留相对目录结构：images/train/a/b.jpg -> labels/train/a/b.txt
        rel = img_path.relative_to(self.images_dir).with_suffix(".txt")
        return self.labels_dir / rel

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self._label_path_from_image(img_path)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor = F.to_tensor(img)  # float32, [C,H,W], 0~1

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    cls_id, cx, cy, bw, bh = map(float, parts)
                    cls_id = int(cls_id)

                    # YOLO(归一化 xywh) -> FasterRCNN(绝对坐标 xyxy)
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h

                    # 裁剪到图像边界
                    x1 = max(0.0, min(x1, w - 1.0))
                    y1 = max(0.0, min(y1, h - 1.0))
                    x2 = max(0.0, min(x2, w - 1.0))
                    y2 = max(0.0, min(y2, h - 1.0))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])

                    # 关键：YOLO 类别从 0 开始，FasterRCNN 的 0 是背景
                    # 所以需要 +1
                    labels.append(cls_id + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=20):
    model.train()
    epoch_loss = 0.0
    start = time.time()

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        if i % print_freq == 0:
            detail = {k: round(v.item(), 4) for k, v in loss_dict.items()}
            print(f"Epoch {epoch} | Iter {i}/{len(loader)} | Loss {losses.item():.4f} | {detail}")

    avg = epoch_loss / max(1, len(loader))
    print(f"Epoch {epoch} done, avg loss={avg:.4f}, time={time.time()-start:.1f}s")
    return avg


@torch.no_grad()
def validate_forward_only(model, loader, device, max_batches=10):
    model.eval()
    cnt = 0
    for i, (images, _) in enumerate(loader):
        images = [img.to(device) for img in images]
        _ = model(images)
        cnt += len(images)
        if i + 1 >= max_batches:
            break
    print(f"Validation forward checked {cnt} images.")


def main():
    # ===== 按你的数据修改 =====
    train_images = "/mnt/workspace/dataset/images/train"
    train_labels = "/mnt/workspace/dataset/labels/train"
    val_images = "/mnt/workspace/dataset/images/val"
    val_labels = "/mnt/workspace/dataset/labels/val"
    test_images = "/mnt/workspace/dataset/images/test"
    test_labels = "/mnt/workspace/dataset/labels/test"

    yolo_nc = 1               # 你的 YOLO 类别数（不含背景）
    num_classes = yolo_nc + 1 # FasterRCNN 需要 + 背景类

    batch_size = 2
    num_workers = 2
    epochs = 20
    lr = 0.005
    save_dir = "checkpoints"
    # =======================

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = YOLODetectionDataset(train_images, train_labels)
    val_ds = YOLODetectionDataset(val_images, val_labels)
    

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )

    model = build_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        scheduler.step()

        validate_forward_only(model, val_loader, device, max_batches=5)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(save_dir, f"fasterrcnn_epoch_{epoch}.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            print("Best model updated.")

    print("Training complete.")


if __name__ == "__main__":
    main()