import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO  # For YOLOv8 models
from torchvision import models, transforms  # For Faster R-CNN


# Function to resize the image to the required input size for YOLOv8 and Faster R-CNN
def resize_image(image, target_size=(1280, 720)):
    """ Resize image while keeping the aspect ratio (if necessary) or resizing to a fixed size. """
    h, w, _ = image.shape
    if target_size:
        resized_image = cv2.resize(image, target_size)  # Resize to fixed size for YOLO
    return resized_image


# Load the trained Faster R-CNN model
def load_faster_rcnn_model(model_path, num_classes):
    # Load the Faster R-CNN model with the number of classes you have in your dataset
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Set pretrained=False for custom weights
    # Get the number of input features for the classification layer
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Change the box predictor to match the number of classes
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load the saved state dict
    state_dict = torch.load(model_path)

    # Load the state_dict while ignoring missing or mismatched layers
    model.load_state_dict(state_dict, strict=False)

    model.eval()  # Set to evaluation mode
    return model


# Preprocessing function for Faster R-CNN
def preprocess_faster_rcnn(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# Prediction for YOLOv8
def get_yolo_predictions(model, image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image to 640x640 for YOLOv8
    image_resized = resize_image(image_rgb, target_size=(1280, 720))

    # Perform inference with YOLOv8 model
    results = model.predict(image_rgb, conf=0.5)

    # Initialize predictions list
    predictions = []

    # Process results
    for result in results:  # Results is typically a list from Ultralytics
        # Access bounding boxes, confidences, and class IDs
        if result.boxes is not None:  # Ensure boxes exist
            boxes = result.boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, conf, cls in zip(boxes, confs, classes):
                x_min, y_min, x_max, y_max = map(int, box)  # Convert to integers
                predictions.append([x_min, y_min, x_max, y_max, float(conf), int(cls)])

    return predictions


# Prediction for Faster R-CNN
def get_faster_rcnn_predictions(model, image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image for Faster R-CNN
    image_resized = resize_image(image_rgb, target_size=(1280, 720))  # Adjust as needed for Faster R-CNN

    # Preprocess the image
    image_tensor = preprocess_faster_rcnn(image_resized)

    # Run inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]  # Get predictions for the first (and only) image in the batch

    # Parse predictions (boxes, labels, and scores)
    predictions = []
    for i in range(len(prediction['boxes'])):
        box = prediction['boxes'][i].cpu().numpy()
        score = prediction['scores'][i].cpu().item()
        label = prediction['labels'][i].cpu().item()
        if score > 0.5:  # Confidence threshold, adjust as needed
            x_min, y_min, x_max, y_max = map(int, box)
            predictions.append([x_min, y_min, x_max, y_max, score, label])

    return predictions


# Get predictions for a model
def get_predictions(model, image_path, task):
    if isinstance(model, YOLO):  # YOLO model
        return get_yolo_predictions(model, image_path)
    elif isinstance(model, torch.nn.Module):  # Faster R-CNN model (PyTorch model)
        return get_faster_rcnn_predictions(model, image_path)
    else:
        raise ValueError("Unknown model type!")


# Function to load YOLOv8 models
def load_yolo_models(model_paths):
    models = [YOLO(model_path) for model_path in model_paths]
    return models


def plot_bounding_boxes(image, predictions, class_names, ax, model_name, row_name):
    # Draw bounding boxes on the image
    for box in predictions:
        x_min, y_min, x_max, y_max, confidence, class_id = box
        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   edgecolor='r', facecolor='none', linewidth=2))
        ax.text(x_min, y_min, f'{class_name} {confidence:.2f}', color='red', fontsize=20)

    ax.imshow(image)
    ax.set_title(model_name, fontsize=40)
    ax.set_ylabel(row_name, fontsize=40, rotation=90, labelpad=50, va='center')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_results(models, image_paths, class_names, task, save_path, model_names, row_titles):
    num_images = len(image_paths)
    num_models = len(models)

    # Create a grid of subplots (rows = images, columns = models)
    fig, axes = plt.subplots(num_images, num_models, figsize=(num_models * 5, num_images * 3))

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)  # Read the image (in BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

        for j, model in enumerate(models):
            predictions = get_predictions(model, image_path, task)  # Get predictions for the image
            ax = axes[i, j]  # Select the subplot for this image and model

            # Set custom model name for the top row only
            if i == 0 and model_names:
                model_name = model_names[j]  # Use the custom model name
            else:
                model_name = ''  # Remove the title for the rest of the rows

            if row_titles and j == 0:
                row_name = row_titles[i]
            else:
                row_name = ''

            plot_bounding_boxes(image_rgb, predictions, class_names, ax, model_name, row_name)

    plt.tight_layout()
    plt.suptitle(f'{task} - Model Predictions on Test Images', fontsize=44)
    plt.subplots_adjust(top=0.88, left=0.05)  # Adjust title
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    row_titles = ["Normal", "Dark", "Far away", "Rotated", "Crowded"]

    # Custom model names
    model_names = [
        "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l",
        "Faster R-CNN"
    ]

    # Example class names for two tasks
    task1_classes = ["Drone", "Station"]
    task2_classes = [
        "Drone_Hole_Empty", "Drone_Hole_Full",
        "Station_TL_Empty", "Station_TL_Full",
        "Station_TR_Empty", "Station_TR_Full",
        "Station_BL_Empty", "Station_BL_Full",
        "Station_BR_Empty", "Station_BR_Full"
    ]

    # Paths to YOLOv8 model files for Task 1
    task1_yolo_model_paths = [
        "Datasets/Dataset_NN1/Runs_NN1/yolov8n_NN1/weights/best.pt",
        "Datasets/Dataset_NN1/Runs_NN1/yolov8s_NN1/weights/best.pt",
        "Datasets/Dataset_NN1/Runs_NN1/yolov8m_NN1/weights/best.pt",
        "Datasets/Dataset_NN1/Runs_NN1/yolov8l_NN1/weights/best.pt"
    ]

    # Paths to YOLOv8 model files for Task 2
    task2_yolo_model_paths = [
        "Datasets/Dataset_NN2/Runs_NN2/yolov8n_NN2/weights/best.pt",
        "Datasets/Dataset_NN2/Runs_NN2/yolov8s_NN2/weights/best.pt",
        "Datasets/Dataset_NN2/Runs_NN2/yolov8m_NN2/weights/best.pt",
        "Datasets/Dataset_NN2/Runs_NN2/yolov8l_NN2/weights/best.pt"
    ]

    # Paths to your saved Faster R-CNN model files for Task 1 and Task 2
    task1_faster_rcnn_model_path = "Datasets/Dataset_NN1/Runs_NN1/faster_rcnn_NN1/best.pt"
    task2_faster_rcnn_model_path = "Datasets/Dataset_NN2/Runs_NN2/faster_rcnn_NN2/best.pth"

    # Load YOLO models for Task 1
    task1_yolo_models = load_yolo_models(task1_yolo_model_paths)
    task2_yolo_models = load_yolo_models(task2_yolo_model_paths)

    # Load your trained Faster R-CNN models
    task1_faster_rcnn_model = load_faster_rcnn_model(task1_faster_rcnn_model_path,
                                                     num_classes=3)  # Adjust num_classes as per your task
    task2_faster_rcnn_model = load_faster_rcnn_model(task2_faster_rcnn_model_path,
                                                     num_classes=11)  # Adjust num_classes as per your task

    # Combine Task 1 models and Task 2 models into one list
    task1_models = task1_yolo_models + [task1_faster_rcnn_model]
    task2_models = task2_yolo_models + [task2_faster_rcnn_model]

    # List of image paths for testing (replace with your actual test images)
    image_folder = "Datasets/Robustness"
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))  # Assuming images are in .jpg format

    # Output file paths for the results
    save_task1_path = "Plots/robust_task1.png"
    save_task2_path = "Plots/robust_task2.png"

    # Plot the results for Task 1
    plot_results(task1_models, image_paths, task1_classes, task="Task 1", save_path=save_task1_path, model_names=model_names, row_titles=row_titles)

    # Plot the results for Task 2
    plot_results(task2_models, image_paths, task2_classes, task="Task 2", save_path=save_task2_path, model_names=model_names, row_titles=row_titles)


if __name__ == "__main__":
    main()
