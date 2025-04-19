import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class DroneStationHoleDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.image_dir = os.path.join(root, split, "Images_NN2")
        self.label_dir = os.path.join(root, split, "Annotations_NN2_10_classes")
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
    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if i % print_freq == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item():.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                pred_labels = output['labels']
                true_labels = target['labels']
                total += len(true_labels)
                correct += (pred_labels == true_labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # Paths
    dataset_root = "/home/tmkristi/PycharmProjects/AIS4002-Detection/Datasets/Dataset_NN2/Dataset_NN2"
    output_dir = "/home/tmkristi/PycharmProjects/AIS4002-Detection/Datasets/Dataset_NN2/Runs_NN2/faster_rcnn_holes"
    os.makedirs(output_dir, exist_ok=True)

    # Datasets
    train_dataset = DroneStationHoleDataset(dataset_root, "train", get_transform(train=True))
    val_dataset = DroneStationHoleDataset(dataset_root, "val", get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2(num_classes=11, weights="COCO_V1")  # 10 classes + background
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10
    best_accuracy = 0
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        accuracy = evaluate(model, val_loader, device)
        lr_scheduler.step()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch}.pth"))

    print(f"Training completed. Best model saved in {output_dir}/best.pth")

if __name__ == "__main__":
    main()