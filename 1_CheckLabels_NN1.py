import cv2
import os
import numpy as np

# Paths
image_dir = "Datasets/Dataset_NN1/Images_NN1"
annotation_dir = "Datasets/Dataset_NN1/Annotations_NN1" # class_id, x_center, y_center, box_width, box_height
classes = ["Drone", "Station"]  # First coordinate of annotations: 0 for drone; 1 for station

# Get list of images
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Check a few images
for image_file in image_files[:5]:  # Check first 5 images
    # Load image
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Load annotation
    annotation_path = os.path.join(annotation_dir, image_file.replace(".jpg", ".txt"))
    if not os.path.exists(annotation_path):
        print(f"No annotation for {image_file}")
        continue

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    # Draw bounding boxes
    for line in lines:
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())
        class_id = int(class_id)
        class_name = classes[class_id]

        # Convert normalized coordinates to pixel values
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        # Calculate top-left and bottom-right corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Annotation Check", image)
    print(f"Showing {image_file}. Press any key to continue, 'q' to quit.")
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()