from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("Runs_Drone_Station/yolo_drone_station/weights/best.pt")

# Test on test set
results = model.predict(
    source="Dataset_Drone_Station/test/images",
    conf=0.5,  # Confidence threshold
    iou=0.5,   # IoU threshold for NMS
    save=True, # Save predictions
    save_txt=True,  # Save YOLO-format predictions
    name="YOLOv8_test_predictions",  # Output folder
    project="Tests_Drone_Station"
)

# Print results
print("Predictions saved in runs/detect/test_predictions")