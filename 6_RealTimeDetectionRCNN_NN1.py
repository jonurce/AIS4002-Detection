import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torchvision.ops import nms
import numpy as np
import datetime

# Paths
model_path = "Datasets/Dataset_NN1/Runs_NN1/faster_rcnn_NN1/best.pt"
output_dir = "Detections/RCNN_NN1"

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load Faster R-CNN model
def get_model(num_classes, model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


num_classes = 3  # Background, A, B
model = get_model(num_classes, model_path, device)

# Initialize L515 camera
camera_index = 1  # Adjust if needed
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if camera opened
if not cap.isOpened():
    print(f"Error: Could not open camera at index {camera_index}")
    exit()

# Get camera settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened at {width}x{height} @ {fps} FPS")

# Initial thresholds
conf = 0.7
iou = 0.5

# Class names
class_names = ["background", "Drone", "Station"]

# Transform for input
transform = ToTensor()

try:
    print("Real-time detection started. Press SPACE to save frame, 'q' to quit, 'u'/'d' for conf, 'i'/'o' for IoU.")
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Preprocess frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).to(device)

        # Run Faster R-CNN inference
        with torch.no_grad():
            predictions = model([img_tensor])[0]

        # Extract predictions
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        # Apply NMS
        keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores), iou_threshold=iou)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Filter by confidence
        mask = scores >= conf
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Draw boxes and labels
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[label]
            label_text = f"{class_name} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display thresholds
        cv2.putText(frame, f"Conf: {conf:.2f} IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)

        # Show frame
        cv2.imshow("L515 Faster R-CNN Detection", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Save frame
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename = os.path.join(output_dir, f'detection_{timestamp}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Detection saved: {filename}")
        elif key == ord('q'):  # Quit
            print("Exiting...")
            break
        elif key == ord('u'):  # Increase conf
            conf = min(conf + 0.1, 1.0)
        elif key == ord('d'):  # Decrease conf
            conf = max(conf - 0.1, 0.1)
        elif key == ord('i'):  # Increase IoU
            iou = min(iou + 0.1, 1.0)
        elif key == ord('o'):  # Decrease IoU
            iou = max(iou - 0.1, 0.1)

finally:
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()