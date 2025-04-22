from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import glob


def yolo_inference(image_input, output_dir, model_path, nn, model_name):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path)

    # Prepare list of image paths
    image_paths = []
    if isinstance(image_input, str):
        if os.path.isdir(image_input):
            # Support common image extensions
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
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image at {image_path}")
                continue

            # Generate output path
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"predicted_{nn}_{model_name}_{image_name}")

            # Perform inference
            results = model(img)

            # Process results
            annotated_img = results[0].plot()  # Get the annotated image with bounding boxes

            # Convert BGR to RGB for display and saving
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            # Save the annotated image
            cv2.imwrite(output_path, cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR))
            print(f"Predicted image saved to {output_path}")


        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    # Example usage
    image_input = "Datasets/Robustness"  # Can be directory or single image path
    output_dir = "Plots/Robustness"  # Output directory for annotated images
    nn = 2
    model_name = f"yolov8l_NN{nn}"
    model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/weights/best.pt"
    yolo_inference(image_input, output_dir, model_path, nn, model_name)