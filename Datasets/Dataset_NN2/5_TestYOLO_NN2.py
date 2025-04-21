from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("Runs_NN2/yolov8l_NN2/weights/last.pt")

# Test on test set
results = model.predict(
    source="Dataset_NN2/test/images",
    conf=0.7,  # Confidence threshold
    iou=0.5,   # IoU threshold for NMS
    save=True, # Save predictions
    name="YOLOv8l_test_predictions_NN2",  # Output folder
    project="Tests_NN2"
)

# Print results
print("Predictions saved in Tests_NN2/YOLOv8l_test_predictions_NN2")