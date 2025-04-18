import cv2
import os
import datetime
from ultralytics import YOLO
import numpy as np

# Paths
model_path = "Datasets/Dataset_NN1/Runs_NN1/yolov8l_NN1/weights/best.pt"
output_dir = "Detections/YOLOv8l_NN1"

# Create output directory for saved frames
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the YOLOv8 model
model = YOLO(model_path)

# Initialize the L515 camera
camera_index = 1  # Adjust if needed (0, 1, 2, etc.)
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

# Set camera resolution (match training or test supported resolutions)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if camera opened
if not cap.isOpened():
    print(f"Error: Could not open camera at index {camera_index}")
    exit()

# Get actual camera settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened at {width}x{height} @ {fps} FPS")

try:
    print("Real-time detection started. Press SPACE to save frame, 'q' to quit.")
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Run YOLOv8 inference
        results = model.predict(frame, conf=0.7, iou=0.5, device="cpu")  # CPU, adjust conf if needed

        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs
            class_names = result.names  # Class names (A, B)

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names[int(cls)]} {score:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('L515 YOLOv8 Detection', frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Save frame on spacebar
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename = os.path.join(output_dir, f'detection_{timestamp}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Detection saved: {filename}")

        elif key == ord('q'):  # Quit on 'q'
            print("Exiting...")
            break

finally:
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()