from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("Runs_NN1/yolov8l_NN1/weights/best.pt")

# Test on test set
results = model.predict(
    source="Dataset_NN1/test/images",
    conf=0.5,  # Confidence threshold
    iou=0.5,   # IoU threshold for NMS
    save=True, # Save predictions
    save_txt=True,  # Save YOLO-format predictions
    name="YOLOv8_test_predictions_NN1",  # Output folder
    project="Tests_NN1"
)

# Print results
print("Predictions saved in Runs_NN1/YOLOv8_test_predictions_NN1")