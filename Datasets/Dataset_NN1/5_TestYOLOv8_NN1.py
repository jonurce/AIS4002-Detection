from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("Runs_NN1/yolov8n_NN1/weights/best.pt")

# Evaluate on test set with metrics
results = model.val(
    data="data_NN1.yaml",   # <-- Make sure this points to a YOLO-format data.yaml file
    split="test",                   # Tells it to use the test set
    conf=0.7,
    iou=0.5,
    save=True,                      # Save predictions
    save_txt=True,                  # Save predictions in YOLO format
    save_hybrid=True,               # Save both labels and predictions
    project="Tests_NN1",
    name="YOLOv8n_test_metrics_NN1",
    imgsz=640                      # Optional: set test resolution
)

print("Evaluation complete. Results and predictions saved to:")
print("→ Tests_NN1/YOLOv8s_test_metrics_NN1/")
