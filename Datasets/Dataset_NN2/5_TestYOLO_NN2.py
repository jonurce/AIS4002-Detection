from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("Runs_NN2/yolov8l_NN2/weights/best.pt")

# Evaluate on test set with metrics
results = model.val(
    data="data_NN2.yaml",   # <-- Make sure this points to a YOLO-format data.yaml file
    split="test",                   # Tells it to use the test set
    conf=0.7,
    iou=0.5,
    save=True,                      # Save predictions
    save_txt=True,                  # Save predictions in YOLO format
    save_hybrid=True,               # Save both labels and predictions
    project="Tests_NN2",
    name="YOLOv8l_test_metrics_NN2",
    imgsz=640                      # Optional: set test resolution
)

print("Evaluation complete. Results and predictions saved to:")
print("→ Tests_NN1/YOLOv8l_test_metrics_NN2/")
