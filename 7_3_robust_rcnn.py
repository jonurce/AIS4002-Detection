import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np
import os
import glob


def faster_rcnn_inference(image_input, output_dir, model_path, nn, model_name, classes, conf_threshold=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare list of image paths
    image_paths = []
    if isinstance(image_input, str):
        if os.path.isdir(image_input):
            image_extensions = ['*.jpg', '*.jpeg', '*.png']
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(image_input, ext)))
        elif os.path.isfile(image_input):
            image_paths = [image_input]
    elif isinstance(image_input, list):
        image_paths = [path for path in image_input if os.path.isfile(path)]

    if not image_paths:
        raise ValueError("No valid image files found in the input")

    # Process each image
    for image_path in image_paths:
        try:
            # Read and preprocess the image
            img = Image.open(image_path).convert('RGB')
            img_tensor = F.to_tensor(img).to(device)  # Convert to tensor and move to device
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                predictions = model(img_tensor)[0]  # Get predictions for the first (and only) image

            # Load image with OpenCV for visualization
            img_cv2 = cv2.imread(image_path)
            if img_cv2 is None:
                print(f"Could not load image at {image_path}")
                continue

            # Process predictions
            boxes = predictions['boxes'].cpu().numpy()  # Bounding boxes
            scores = predictions['scores'].cpu().numpy()  # Confidence scores
            labels = predictions['labels'].cpu().numpy()  # Class labels

            # Filter predictions based on confidence threshold
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Draw bounding boxes and labels on the image
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.astype(int)
                # Draw rectangle
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label and score
                label_text = f"{classes[label]}: {score:.2f}"
                cv2.putText(img_cv2, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Generate output path
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"predicted_{nn}_{model_name}_{image_name}")

            # Save the annotated image
            cv2.imwrite(output_path, img_cv2)
            print(f"Predicted image saved to {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    # Example usage
    image_input = "Datasets/Robustness"  # Can be directory or single image path
    output_dir = "Plots/Robustness"  # Output directory for annotated images
    nn = 2
    model_name = f"faster_rcnn_NN{nn}"
    model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/best.pth"  # Path to saved Faster R-CNN model

    classes = []
    if nn == 1:
        classes = ["Background", "Drone", "Station"]
    else:
        classes = [ "Background",
            "Drone_Hole_Empty", "Drone_Hole_Full",
            "Station_TL_Empty", "Station_TL_Full",
            "Station_TR_Empty", "Station_TR_Full",
            "Station_BL_Empty", "Station_BL_Full",
            "Station_BR_Empty", "Station_BR_Full"
        ]

    faster_rcnn_inference(image_input, output_dir, model_path, nn, model_name, classes, conf_threshold=0.3)