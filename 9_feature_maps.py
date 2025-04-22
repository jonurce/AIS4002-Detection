import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === Setup paths ===
nn = 2
model_name = f"yolov8s_NN{nn}"
model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/weights/best.pt"
output_dir = f"FeatureMaps/{model_name}"
os.makedirs(output_dir, exist_ok=True)

# === Check for CUDA ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load the YOLO model ===
model = YOLO(model_path)

# === Load a sample image ===
image_path = "Datasets/Dataset_NN2/Dataset_NN2/test/images/image_2025_04_16_16_14_37.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Preprocess image ===
image_resized = cv2.resize(image_rgb, (1280, 736))
image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

# === Hook to capture feature maps ===
feature_maps = {}
def hook_fn(module, input, output, name):
    feature_maps[name] = output.detach().cpu()

# === Register hooks on all convolutional layers ===
model.model.eval()
layer_names = []
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Conv2d):  # Target all conv layers
        layer_names.append(name)
        module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

print(f"Capturing feature maps from {len(layer_names)} convolutional layers: {layer_names}")

# === Run direct inference to capture feature maps ===
with torch.no_grad():
    _ = model.model(image_tensor)  # Direct forward pass to trigger hooks
    print(f"Captured feature maps: {list(feature_maps.keys())}")  # Debug

# === Run prediction separately for final output ===
with torch.no_grad():
    results = model.predict(image_tensor, conf=0.7, iou=0.5, device=device)

# === Save input image ===
input_image_path = os.path.join(output_dir, "0000_input_image.png")
plt.figure(figsize=(12.8, 7.36))  # For 1280x736 at 100 DPI
plt.imshow(image_resized)
plt.title("Input Image")
plt.axis('off')
plt.savefig(input_image_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved input image to {input_image_path}")

# === Plot and save feature maps for all layers ===
for idx, layer_name in enumerate(layer_names):
    if layer_name not in feature_maps:
        print(f"Warning: No feature maps captured for layer {layer_name}")
        continue
    fmap = feature_maps[layer_name][0]  # Shape: [channels, height, width]
    num_channels = fmap.shape[0]
    print(f"Layer {layer_name}: {num_channels} channels, shape {fmap.shape[1:]}")

    # Plot up to 16 channels per grid
    num_plots = min(16, num_channels)
    cols = 4
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(12.8, 7.36))  # For 1280x736 at 100 DPI
    for i in range(num_plots):
        plt.subplot(rows, cols, i + 1)
        fmap_i = fmap[i].numpy()
        # Normalize for visualization
        fmap_i = (fmap_i - fmap_i.min()) / (fmap_i.max() - fmap_i.min() + 1e-8)
        plt.imshow(fmap_i, cmap='viridis')
        plt.title(f"Channel {i}", fontsize=8)
        plt.axis('off')
    plt.suptitle(f"Layer {layer_name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save with sequential naming for video
    output_path = os.path.join(output_dir, f"{idx + 1:04d}_layer_{layer_name.replace('.', '_')}.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved feature maps for {layer_name} to {output_path}")

# === Save prediction image ===
result = results[0]
image_pred = image_resized.copy()
for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    label = f"{result.names[int(cls)]} {score:.2f}"
    cv2.rectangle(image_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_pred, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

pred_image_path = os.path.join(output_dir, f"{len(layer_names) + 1:04d}_prediction.png")
plt.figure(figsize=(12.8, 7.36))  # For 1280x736 at 100 DPI
plt.imshow(image_pred)
plt.title("Prediction")
plt.axis('off')
plt.savefig(pred_image_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved prediction image to {pred_image_path}")

# === Instructions for video creation ===
print("\nTo create a video from the images, use ffmpeg:")
print(f"cd {output_dir}")
print(f"ffmpeg -framerate 1 -i %04d_*.png -c:v libx264 -pix_fmt yuv420p feature_map_video.mp4")
print("Or run create_video.py from the FeatureMaps/yolov8s_NN2 directory.")
