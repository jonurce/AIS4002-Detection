from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano for speed, or medium/large for accuracy)
model = YOLO("yolov8n.pt")  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data="data_drone_station.yaml",  # Path to dataset config
    epochs=50,         # Number of training epochs
    imgsz=640,         # Image size (resize to 640x640)
    batch=8,          # Batch size (adjust based on GPU memory)
    workers = 4,        # CPU cores
    device=-1,          # Use GPU (set to -1 for CPU)
    name="yolo_drone_station",   # Experiment name
    project="Runs_Drone_Station",
    patience=10,       # Early stopping after 10 epochs with no improvement
    save=True,         # Save checkpoints
    save_period=10     # Save checkpoint every 10 epochs
)

# Print training results
print("Training completed. Results saved in runs/detect/yolo_drone_station")