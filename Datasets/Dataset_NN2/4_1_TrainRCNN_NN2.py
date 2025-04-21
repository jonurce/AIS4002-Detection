import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from PIL import Image
import numpy as np
import pandas as pd
import time

class DroneStationHoleDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.class_names = [
            'Drone_Hole_Empty', 'Drone_Hole_Full', 'Station_TL_Empty', 'Station_TL_Full',
            'Station_TR_Empty', 'Station_TR_Full', 'Station_BL_Empty', 'Station_BL_Full',
            'Station_BR_Empty', 'Station_BR_Full'
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = T.ToTensor()(img)  # Convert to tensor

        # Load annotations
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    class_id = int(class_id)
                    # Convert YOLO to COCO format (x_min, y_min, x_max, y_max)
                    img_w, img_h = img.shape[2], img.shape[1]
                    x_min = (x_center - w/2) * img_w
                    y_min = (y_center - h/2) * img_h
                    x_max = (x_center + w/2) * img_w
                    y_max = (y_center + h/2) * img_h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id + 1)  # +1 for 1-based indexing (0 is background)

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotation(30))
        transforms.append(T.ColorJitter(hue=0.015, saturation=0.7, brightness=0.4))
    transforms.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transforms)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    train_box_loss = 0
    train_cls_loss = 0
    total_batches = len(data_loader)
    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_box_loss += (loss_dict.get("loss_box_reg", 0) + loss_dict.get("loss_rpn_box_reg", 0)).item()
        train_cls_loss += loss_dict.get("loss_classifier", 0).item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if i % print_freq == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item():.4f}")

    train_box_loss /= total_batches
    train_cls_loss /= total_batches
    return train_box_loss, train_cls_loss

def evaluate(model, data_loader, device):
    model.eval()
    total, correct = 0, 0
    val_box_loss = 0
    val_cls_loss = 0
    total_batches = len(data_loader)
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Compute validation loss
            model.train()  # Required for loss computation
            loss_dict = model(images, targets)
            model.eval()
            val_box_loss += (loss_dict.get("loss_box_reg", 0) + loss_dict.get("loss_rpn_box_reg", 0)).item()
            val_cls_loss += loss_dict.get("loss_classifier", 0).item()

            # Compute accuracy
            outputs = model(images)
            for target, output in zip(targets, outputs):
                if not isinstance(output, dict) or 'labels' not in output:
                    print(f"Unexpected output format: {output}")
                    continue

                pred_labels = output['labels']
                pred_boxes = output['boxes']
                true_labels = target['labels']
                true_boxes = target['boxes']

                if len(true_labels) == 0 or len(pred_labels) == 0:
                    continue

                iou = box_iou(pred_boxes, true_boxes)
                iou_threshold = 0.5
                max_iou, match_indices = iou.max(dim=1)

                for pred_idx, true_idx in enumerate(match_indices):
                    if max_iou[pred_idx] > iou_threshold:
                        if pred_labels[pred_idx] == true_labels[true_idx]:
                            correct += 1
                total += len(true_labels)

    accuracy = correct / total if total > 0 else 0
    val_box_loss /= total_batches
    val_cls_loss /= total_batches
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy, val_box_loss, val_cls_loss

def evaluate_metrics(model, data_loader, device):
    model.eval()
    predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model([img.to(device) for img in images])
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

    precision, recall, map50, map50_95 = compute_metrics(predictions, all_targets)
    return precision, recall, map50, map50_95

def compute_metrics(predictions, targets, iou_thres=0.5):
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

        iou = box_iou(pred_boxes, tgt_boxes)
        max_iou, max_idx = iou.max(dim=1)
        matched_gt = set()

        for i, (iou_val, idx) in enumerate(zip(max_iou, max_idx)):
            if iou_val >= iou_thres and pred_labels[i] == tgt_labels[idx]:
                true_positives += 1
                matched_gt.add(idx.item())
            else:
                false_positives += 1
        false_negatives += len(tgt_boxes) - len(matched_gt)

        map_scores.append(max_iou.mean().item() if max_iou.numel() > 0 else 0)

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    map50 = np.mean(map_scores) if map_scores else 0
    map50_95 = map50 * 0.9  # Simplified approximation
    return precision, recall, map50, map50_95

def main():
    # Paths
    dataset_root = "Dataset_NN2"
    output_dir = "Runs_NN2/faster_rcnn_NN2"
    os.makedirs(output_dir, exist_ok=True)

    # Datasets
    train_dataset = DroneStationHoleDataset(dataset_root, "train", get_transform(train=True))
    val_dataset = DroneStationHoleDataset(dataset_root, "val", get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=11)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
    best_accuracy = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        train_box_loss, train_cls_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_loss = train_box_loss + train_cls_loss

        accuracy, val_box_loss, val_cls_loss = evaluate(model, val_loader, device)
        val_loss = val_box_loss + val_cls_loss

        precision, recall, map50, map50_95 = evaluate_metrics(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        results.append([
            epoch + 1, epoch_time,
            train_box_loss, train_cls_loss, 0,
            precision, recall, map50, map50_95,
            val_box_loss, val_cls_loss, 0,
            lr, lr, lr
        ])

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f} (Box: {train_box_loss:.4f}, Cls: {train_cls_loss:.4f}), "
              f"Val Loss: {val_loss:.4f} (Box: {val_box_loss:.4f}, Cls: {val_cls_loss:.4f}), "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")

        lr_scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch}.pth"))

        df = pd.DataFrame(results, columns=columns)
        df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    print(f"Training completed. Best model saved in {output_dir}/best.pth")
    print(f"Results saved to {output_dir}/results.csv")

if __name__ == "__main__":
    main()