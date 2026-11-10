import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLOTestDataset(Dataset):
    """
    dataset/
      images/test/xxx.jpg
      labels/test/xxx.txt

    YOLO label: class cx cy w h (normalized)
    """
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_paths: List[Path] = sorted(
            [p for p in self.images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _label_path(self, img_path: Path) -> Path:
        rel = img_path.relative_to(self.images_dir).with_suffix(".txt")
        return self.labels_dir / rel

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self._label_path(img_path)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor = F.to_tensor(img)

        gt_boxes = []
        gt_labels = []

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

                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h

                    x1 = max(0.0, min(x1, w - 1.0))
                    y1 = max(0.0, min(y1, h - 1.0))
                    x2 = max(0.0, min(x2, w - 1.0))
                    y2 = max(0.0, min(y2, h - 1.0))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(cls_id)  # 0-based class id

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
            gt_labels = torch.tensor(gt_labels, dtype=torch.int64)

        target = {"boxes": gt_boxes, "labels": gt_labels}
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute AP by precision envelope integration.
    """
    if recall.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # area under curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_and_collect(model, loader, device, num_classes, score_thresh=0.001):
    """
    1) torchmetrics: mAP@[.50:.95] 等
    2) 手工收集每类检测结果，用于PR/F1曲线
    """
    model.eval()
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    # 每类预测记录: list[{"img_id": int, "score": float, "box": tensor(4)}]
    preds_by_class: Dict[int, List[Dict]] = {c: [] for c in range(num_classes)}
    # 每类GT记录: dict[(img_id, cls)] = tensor(N,4)
    gt_by_img_cls: Dict[Tuple[int, int], torch.Tensor] = {}

    img_global_id = 0

    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            preds_for_metric = []
            gts_for_metric = []

            for out, gt in zip(outputs, targets):
                # ---- predictions from FasterRCNN ----
                boxes = out["boxes"].detach().cpu()
                scores = out["scores"].detach().cpu()
                labels = out["labels"].detach().cpu() - 1  # 1..N -> 0..N-1

                keep = scores >= score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                valid = (labels >= 0) & (labels < num_classes)
                boxes = boxes[valid]
                scores = scores[valid]
                labels = labels[valid].to(torch.int64)

                # save for manual PR/F1
                for b, s, l in zip(boxes, scores, labels):
                    preds_by_class[int(l.item())].append(
                        {"img_id": img_global_id, "score": float(s.item()), "box": b.clone()}
                    )

                # ---- GT ----
                gt_boxes = gt["boxes"].to(torch.float32)
                gt_labels = gt["labels"].to(torch.int64)
                valid_gt = (gt_labels >= 0) & (gt_labels < num_classes)
                gt_boxes = gt_boxes[valid_gt]
                gt_labels = gt_labels[valid_gt]

                # save for manual PR/F1
                for c in range(num_classes):
                    cls_mask = gt_labels == c
                    cls_boxes = gt_boxes[cls_mask]
                    if cls_boxes.numel() > 0:
                        gt_by_img_cls[(img_global_id, c)] = cls_boxes.clone()

                # for torchmetrics
                preds_for_metric.append({
                    "boxes": boxes.to(torch.float32),
                    "scores": scores.to(torch.float32),
                    "labels": labels.to(torch.int64),
                })
                gts_for_metric.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                })

                img_global_id += 1

            map_metric.update(preds_for_metric, gts_for_metric)

    map_results = map_metric.compute()
    return map_results, preds_by_class, gt_by_img_cls


def compute_pr_f1_per_class(preds_by_class, gt_by_img_cls, num_classes, iou_thresh=0.5):
    """
    对每个类别计算:
    - PR曲线
    - AP50
    - F1-Confidence曲线
    """
    per_class = {}

    for c in range(num_classes):
        preds = preds_by_class[c]
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        # GT总数
        gt_keys = [k for k in gt_by_img_cls.keys() if k[1] == c]
        npos = int(sum(gt_by_img_cls[k].shape[0] for k in gt_keys))

        if len(preds) == 0:
            per_class[c] = {
                "scores": np.array([], dtype=np.float32),
                "precision": np.array([], dtype=np.float32),
                "recall": np.array([], dtype=np.float32),
                "f1": np.array([], dtype=np.float32),
                "ap50": 0.0,
                "best_f1": 0.0,
                "best_thresh": 0.0,
                "best_precision": 0.0,
                "best_recall": 0.0,
                "gt_count": npos,
                "pred_count": 0,
            }
            continue

        # 为每个(img,cls)维护是否已匹配
        matched_flags = {}
        for key in gt_keys:
            n = gt_by_img_cls[key].shape[0]
            matched_flags[key] = np.zeros(n, dtype=bool)

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        scores = np.array([p["score"] for p in preds], dtype=np.float32)

        for i, p in enumerate(preds):
            key = (p["img_id"], c)
            pred_box = p["box"].unsqueeze(0)  # [1,4]

            if key not in gt_by_img_cls:
                fp[i] = 1.0
                continue

            gt_boxes = gt_by_img_cls[key]  # [N,4]
            ious = box_iou(pred_box, gt_boxes).squeeze(0).numpy()  # [N]

            # 只考虑未匹配GT
            unmatched = ~matched_flags[key]
            if unmatched.sum() == 0:
                fp[i] = 1.0
                continue

            ious_unmatched = ious.copy()
            ious_unmatched[~unmatched] = -1.0
            max_idx = int(np.argmax(ious_unmatched))
            max_iou = float(ious_unmatched[max_idx])

            if max_iou >= iou_thresh:
                tp[i] = 1.0
                matched_flags[key][max_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        if npos > 0:
            recall = tp_cum / float(npos)
        else:
            recall = np.zeros_like(tp_cum)

        f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
        ap50 = compute_ap_from_pr(recall, precision)

        best_idx = int(np.argmax(f1)) if f1.size > 0 else 0
        best_f1 = float(f1[best_idx]) if f1.size > 0 else 0.0
        best_thresh = float(scores[best_idx]) if scores.size > 0 else 0.0
        best_precision = float(precision[best_idx]) if precision.size > 0 else 0.0
        best_recall = float(recall[best_idx]) if recall.size > 0 else 0.0

        per_class[c] = {
            "scores": scores,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": float(ap50),
            "best_f1": best_f1,
            "best_thresh": best_thresh,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "gt_count": npos,
            "pred_count": len(preds),
        }

    return per_class


def save_curves_and_tables(per_class_metrics, class_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) 每类 PR/F1 曲线
    for c, m in per_class_metrics.items():
        cname = class_names[c] if class_names and c < len(class_names) else f"class_{c}"

        # PR
        plt.figure(figsize=(7, 5))
        if m["recall"].size > 0:
            plt.plot(m["recall"], m["precision"], label=f"{cname} AP50={m['ap50']:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {cname}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(plots_dir / f"pr_{cname}.png", dpi=180)
        plt.close()

        # F1-Confidence
        plt.figure(figsize=(7, 5))
        if m["scores"].size > 0:
            plt.plot(m["scores"], m["f1"], label=f"best_f1={m['best_f1']:.4f}")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1")
        plt.title(f"F1 Curve - {cname}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(plots_dir / f"f1_{cname}.png", dpi=180)
        plt.close()

    # 2) Macro PR 曲线
    recall_grid = np.linspace(0.0, 1.0, 101)
    pr_interp = []
    for c, m in per_class_metrics.items():
        if m["recall"].size == 0:
            continue
        # 保证recall单调（通常已经单调）
        recall = np.clip(m["recall"], 0, 1)
        precision = np.clip(m["precision"], 0, 1)
        # np.interp 要求x递增
        order = np.argsort(recall)
        recall = recall[order]
        precision = precision[order]
        interp_p = np.interp(recall_grid, recall, precision, left=precision[0], right=precision[-1])
        pr_interp.append(interp_p)

    if len(pr_interp) > 0:
        macro_p = np.mean(np.stack(pr_interp, axis=0), axis=0)
        plt.figure(figsize=(7, 5))
        plt.plot(recall_grid, macro_p, label="macro PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Macro PR Curve")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(plots_dir / "pr_macro.png", dpi=180)
        plt.close()

    # 3) Macro F1-Confidence 曲线
    conf_grid = np.linspace(0.0, 1.0, 101)
    f1_interp = []
    for c, m in per_class_metrics.items():
        if m["scores"].size == 0:
            continue
        scores = np.clip(m["scores"], 0, 1)
        f1 = np.clip(m["f1"], 0, 1)
        # 分数通常降序，这里改成升序再插值
        order = np.argsort(scores)
        scores = scores[order]
        f1 = f1[order]
        interp_f1 = np.interp(conf_grid, scores, f1, left=f1[0], right=f1[-1])
        f1_interp.append(interp_f1)

    macro_best_f1 = 0.0
    macro_best_conf = 0.0
    if len(f1_interp) > 0:
        macro_f1 = np.mean(np.stack(f1_interp, axis=0), axis=0)
        best_idx = int(np.argmax(macro_f1))
        macro_best_f1 = float(macro_f1[best_idx])
        macro_best_conf = float(conf_grid[best_idx])

        plt.figure(figsize=(7, 5))
        plt.plot(conf_grid, macro_f1, label=f"best_f1={macro_best_f1:.4f}@{macro_best_conf:.3f}")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1")
        plt.title("Macro F1 Curve")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(plots_dir / "f1_macro.png", dpi=180)
        plt.close()

    # 4) 每类指标 CSV
    csv_path = Path(out_dir) / "per_class_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name", "gt_count", "pred_count",
            "ap50", "best_f1", "best_thresh", "precision_at_best", "recall_at_best"
        ])
        for c, m in per_class_metrics.items():
            cname = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
            writer.writerow([
                c, cname, m["gt_count"], m["pred_count"],
                f"{m['ap50']:.6f}", f"{m['best_f1']:.6f}", f"{m['best_thresh']:.6f}",
                f"{m['best_precision']:.6f}", f"{m['best_recall']:.6f}"
            ])

    return macro_best_f1, macro_best_conf


def tensor_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def main():
    # ====== 按需修改 ======
    test_images = "/mnt/workspace/dataset/images/test"
    test_labels = "/mnt/workspace/dataset/labels/test"
    best_weight = "checkpoints/best.pth"
    output_dir = "rcnnruns/eval_test"

    yolo_nc = 1
    num_classes = yolo_nc + 1  # FasterRCNN includes background
    batch_size = 2
    num_workers = 2

    # 可选: 类别名长度应等于 yolo_nc
    class_names = ["class0"]
    # =====================

    if len(class_names) != yolo_nc:
        raise ValueError("class_names length must equal yolo_nc")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = YOLOTestDataset(test_images, test_labels)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    model = build_model(num_classes).to(device)

    ckpt = torch.load(best_weight, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    # 评估 + 收集
    map_results, preds_by_class, gt_by_img_cls = evaluate_and_collect(
        model=model,
        loader=loader,
        device=device,
        num_classes=yolo_nc,   # 这里传前景类别数
        score_thresh=0.001
    )

    per_class_metrics = compute_pr_f1_per_class(
        preds_by_class=preds_by_class,
        gt_by_img_cls=gt_by_img_cls,
        num_classes=yolo_nc,
        iou_thresh=0.5
    )

    os.makedirs(output_dir, exist_ok=True)
    macro_best_f1, macro_best_conf = save_curves_and_tables(
        per_class_metrics=per_class_metrics,
        class_names=class_names,
        out_dir=output_dir
    )

    # 保存 summary.json
    summary = {
        "map_50_95": tensor_scalar(map_results["map"]),
        "map_50": tensor_scalar(map_results["map_50"]),
        "map_75": tensor_scalar(map_results["map_75"]),
        "mar_1": tensor_scalar(map_results["mar_1"]),
        "mar_10": tensor_scalar(map_results["mar_10"]),
        "mar_100": tensor_scalar(map_results["mar_100"]),
        "macro_best_f1": macro_best_f1,
        "macro_best_conf": macro_best_conf,
        "per_class_ap50_mean": float(np.mean([m["ap50"] for m in per_class_metrics.values()])),
    }

    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"Saved summary: {Path(output_dir) / 'summary.json'}")
    print(f"Saved table:   {Path(output_dir) / 'per_class_metrics.csv'}")
    print(f"Saved plots:   {Path(output_dir) / 'plots'}")


if __name__ == "__main__":
    main()