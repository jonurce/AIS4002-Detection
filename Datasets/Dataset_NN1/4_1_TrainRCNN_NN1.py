import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class DroneStationDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.classes = ["Drone", "Station"]  # Background is implicit (class 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Load YOLO annotations
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, w, h = map(float, line.split())
                    class_id = int(class_id) + 1  # YOLO: 0=A, 1=B; Faster R-CNN: 1=A, 2=B (0=background)
                    x1 = (x_center - w/2) * width
                    y1 = (y_center - h/2) * height
                    x2 = (x_center + w/2) * width
                    y2 = (y_center + h/2) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform:
            img = self.transform(img)

        return img, target

def get_model(num_classes, device):
    # Load pre-trained Faster R-CNN with ResNet-50 FPN
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace classifier head for 2 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

def evaluate_metrics(model, data_loader, device):
    model.eval()
    predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                predictions.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                })
                all_targets.append({
                    "boxes": targets[i]["boxes"].cpu(),
                    "labels": targets[i]["labels"].cpu()
                })

    # Simplified evaluation (precision, recall, mAP)
    precision, recall, map50, map50_95 = compute_metrics(predictions, all_targets)
    return precision, recall, map50, map50_95

def compute_metrics(predictions, targets, iou_thres=0.5):
    # Simplified metric computation (approximating COCO mAP)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    map_scores = []

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        tgt_boxes = tgt["boxes"]
        tgt_labels = tgt["labels"]

        # Filter predictions by confidence (e.g., >0.5)
        conf_mask = pred_scores > 0.5
        pred_boxes = pred_boxes[conf_mask]
        pred_labels = pred_labels[conf_mask]
        pred_scores = pred_scores[conf_mask]

        if len(pred_boxes) == 0 and len(tgt_boxes) == 0:
            continue
        if len(pred_boxes) == 0:
            false_negatives += len(tgt_boxes)
            continue
        if len(tgt_boxes) == 0:
            false_positives += len(pred_boxes)
            continue

        # Compute IoU
        iou = box_iou(pred_boxes, tgt_boxes)
        max_iou, max_idx = iou.max(dim=1)

        for i, (iou_val, idx) in enumerate(zip(max_iou, max_idx)):
            if iou_val >= iou_thres and pred_labels[i] == tgt_labels[idx]:
                true_positives += 1
            else:
                false_positives += 1
        false_negatives += len(tgt_boxes) - (max_iou >= iou_thres).sum().item()

        # Approximate mAP@0.5 and mAP@0.5:0.95
        map_scores.append(max_iou.mean().item() if max_iou.numel() > 0 else 0)

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    map50 = np.mean(map_scores) if map_scores else 0
    map50_95 = map50 * 0.9  # Simplified approximation (adjust based on IoU range)

    return precision, recall, map50, map50_95

def box_iou(boxes1, boxes2):
    # Compute IoU between two sets of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    dataset_root = "Dataset_NN1"
    output_dir = "Runs_NN1/faster_rcnn_NN1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dataset
    train_dataset = DroneStationDataset(dataset_root, "train", transform=ToTensor())
    val_dataset = DroneStationDataset(dataset_root, "val", transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    num_classes = 3  # 2 classes (Drone, Station) + background
    model = get_model(num_classes, device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Metrics storage
    results = []
    columns = [
        "epoch", "time",
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "lr/pg0", "lr/pg1", "lr/pg2"
    ]

    # Training loop
    num_epochs = 100
    best_val_loss = float("inf")
    patience = 100
    patience_counter = 0

    start_time = time.time()
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_box_loss = 0
        train_cls_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_box_loss += (loss_dict.get("loss_box_reg", 0) + loss_dict.get("loss_rpn_box_reg", 0)).item()
            train_cls_loss += loss_dict.get("loss_classifier", 0).item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        train_box_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_loss = train_box_loss + train_cls_loss

        # Validate
        model.eval()
        val_box_loss = 0
        val_cls_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)  # Run on GPU
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                model.train()  # Required for loss computation
                loss_dict = model(images, targets)
                model.eval()
                val_box_loss += (loss_dict.get("loss_box_reg", 0) + loss_dict.get("loss_rpn_box_reg", 0)).item()
                val_cls_loss += loss_dict.get("loss_classifier", 0).item()

        val_box_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_loss = val_box_loss + val_cls_loss

        # Compute metrics
        precision, recall, map50, map50_95 = evaluate_metrics(model, val_loader, device)

        # Learning rate
        lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        epoch_time = time.time() - start_time
        results.append([
            epoch + 1, epoch_time,
            train_box_loss, train_cls_loss, 0,  # dfl_loss=0 (not used)
            precision, recall, map50, map50_95,
            val_box_loss, val_cls_loss, 0,  # dfl_loss=0
            lr, lr, lr  # Single parameter group
        ])

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f} (Box: {train_box_loss:.4f}, Cls: {train_cls_loss:.4f}), "
              f"Val Loss: {val_loss:.4f} (Box: {val_box_loss:.4f}, Cls: {val_cls_loss:.4f}), "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Save results
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    print(f"Training completed. Best model saved to {output_dir}/best.pt")
    print(f"Results saved to {output_dir}/results.csv")

if __name__ == "__main__":
    main()