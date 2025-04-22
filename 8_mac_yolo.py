from ptflops import get_model_complexity_info
from ultralytics import YOLO

# Load the YOLO model (e.g., YOLOv8n)
nn = 2
model_name = f"yolov8l_NN{nn}"
model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/weights/best.pt"
model = YOLO(model_path)  # Pretrained model or your custom model

# Get the PyTorch model from the YOLO object
pytorch_model = model.model

# Compute MACs and parameters (input shape: 3 channels, 1280x720)
macs, params = get_model_complexity_info(
    pytorch_model,
    (3, 1280, 736),  # Input shape: (channels, height, width)
    as_strings=True,
    print_per_layer_stat=True,  # Detailed layer-wise stats
    verbose=True
)

print(f"Model: YOLOv8n")
print(f"MACs: {macs}")
print(f"Parameters: {params}")