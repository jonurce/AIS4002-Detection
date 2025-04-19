from sympy import false
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano for speed, or medium/large for accuracy)
model = YOLO("yolov8n.pt")  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data="data_NN1.yaml",  # Path to dataset config
    epochs=100,         # Number of training epochs
    imgsz=640,         # Image size (resize to 640x640)
    batch=8,          # Batch size (adjust based on GPU memory)
    workers = 4,        # CPU cores
    device="cuda",          # Use GPU (set to -1 for CPU)
    name="yolov8n_NN1",   # Experiment name
    project="Runs_NN1",
    patience=0,       # Early stopping after 10 epochs with no improvement
    save=False,         # Save checkpoints
    save_period=10,     # Save checkpoint every 10 epochs
# Data augmentation
    hsv_h=0.02,                     # Hue augmentation
    hsv_s=0.7,                       # Saturation augmentation
    hsv_v=0.4,                       # Value (brightness) augmentation
    degrees=50,                       # Rotation up to ±10 degrees
    translate=0.2,                    # Translation up to 10% of image
    scale=0.5                        # Scaling up to 50%
)

# Print training results
print("Training completed. Results saved in Runs_NN1/yolov8n_NN1")