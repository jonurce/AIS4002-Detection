from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("runs/detect/yolo_a_b/weights/best.pt")

# Test on test set
results = model.predict(
    source="Dataset_Drone_Station/test/images",
    conf=0.5,  # Confidence threshold
    iou=0.5,   # IoU threshold for NMS
    save=True, # Save predictions
    save_txt=True,  # Save YOLO-format predictions
    name="YOLOv8_test_predictions"  # Output folder
)

# Print results
print("Predictions saved in runs/detect/test_predictions")