from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("Runs_NN1/yolov8l_NN1/weights/last.pt")

# Test on test set
results = model.predict(
    source="Dataset_NN1/test/images",
    conf=0.7,  # Confidence threshold
    iou=0.5,   # IoU threshold for NMS
    save=True, # Save predictions
    name="YOLOv8l_test_predictions_NN1",  # Output folder
    project="Tests_NN1"
)

# Print results
print("Predictions saved in Runs_NN1/YOLOv8l_test_predictions_NN1")