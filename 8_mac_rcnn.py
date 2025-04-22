from ptflops import get_model_complexity_info
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the YOLO model (e.g., YOLOv8n)
nn = 2
model_name = f"faster_rcnn_NN{nn}"
model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/best.pth"

num_classes = 11  # Replace with the number of classes in your trained model (including background)

# Initialize the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# Load the saved weights
checkpoint = torch.load(model_path, map_location=torch.device('cuda'))  # Use 'cuda' if GPU is available
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

# Set model to evaluation mode
model.eval()

# Compute MACs and parameters (input shape: 3 channels, 1280x720)
macs, params = get_model_complexity_info(
    model,
    (3, 1280, 736),  # Input shape: (channels, height, width)
    as_strings=True,
    print_per_layer_stat=True,  # Detailed layer-wise stats
    verbose=True
)

print(f"Model: {model_name}")
print(f"MACs: {macs}")
print(f"Parameters: {params}")