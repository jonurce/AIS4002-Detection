import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
import numpy as np

def get_model(num_classes, model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

def main():
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    model_path = "Runs_Drone_Station/faster_rcnn_drone_station/best.pt"
    test_image_dir = "Dataset_Drone_Station/test/images"
    output_dir = "Tests_Drone_Station/faster_rcnn_test_predictions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Model
    num_classes = 3  # A, B + background
    model = get_model(num_classes, model_path, device)
    model.eval()

    # Classes
    classes = ["background", "Drone", "Station"]

    # Test images
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(".jpg")]
    transform = ToTensor()

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(test_image_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).to(device)

            # Predict
            predictions = model([img_tensor])[0]
            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()

            # Filter by confidence
            conf_threshold = 0.7
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # Draw boxes
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                class_name = classes[label]
                label_text = f"{class_name} {score:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")

    print(f"Test predictions saved to {output_dir}")

if __name__ == "__main__":
    main()