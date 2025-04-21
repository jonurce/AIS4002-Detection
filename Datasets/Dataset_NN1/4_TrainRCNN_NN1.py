import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml

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

def main():
    #Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    num_classes = 3  # 2 classes (A, B) + background
    model = get_model(num_classes, device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 100
    best_val_loss = float("inf")
    patience = 100
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

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

    print(f"Training completed. Best model saved to {output_dir}/best.pt")

if __name__ == "__main__":
    main()