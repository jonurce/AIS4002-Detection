import os
import cv2
import numpy as np


def check_annotations(label_dir, image_dir, output_dir):
    # Create output directory for annotated images
    os.makedirs(output_dir, exist_ok=True)

    # Class names
    hole_names = ['Drone_Hole', 'Station_TL', 'Station_TR', 'Station_BL', 'Station_BR']
    empty_full_names = ['Empty', 'Full']

    # Colors for each hole_id (BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # Green, Blue, Red, Yellow, Cyan

    # Lists for errors
    missing_images = []
    missing_labels = []
    invalid_lines = []
    invalid_values = []
    out_of_bounds = []

    # Check for matching files
    label_files = set(f for f in os.listdir(label_dir) if f.endswith(".txt"))
    image_files = set(f.replace(".jpg", ".txt") for f in os.listdir(image_dir) if f.endswith(".jpg"))

    missing_images.extend([f.replace(".txt", ".jpg") for f in label_files - image_files])
    missing_labels.extend([f.replace(".txt", ".jpg") for f in image_files - label_files])

    # Process each label file
    for label_file in sorted(label_files):
        img_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
        label_path = os.path.join(label_dir, label_file)
        output_img_path = os.path.join(output_dir, label_file.replace(".txt", ".jpg"))

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            missing_images.append(img_path)
            continue
        height, width = img.shape[:2]

        # Read annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()

        valid = True
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 6:
                invalid_lines.append(f"{label_file}: Line {i + 1}: Expected 6 values, got {len(parts)}")
                valid = False
                continue

            try:
                hole_id, x, y, w, h, empty_full = map(float, parts)
                hole_id = int(hole_id)
                empty_full = int(empty_full)
            except ValueError:
                invalid_lines.append(f"{label_file}: Line {i + 1}: Non-numeric values")
                valid = False
                continue

            # Check valid values
            if hole_id not in range(5):
                invalid_values.append(f"{label_file}: Line {i + 1}: Invalid hole_id {hole_id}, expected 0–4")
                valid = False
            if empty_full not in [0, 1]:
                invalid_values.append(f"{label_file}: Line {i + 1}: Invalid empty_full {empty_full}, expected 0 or 1")
                valid = False
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                invalid_values.append(f"{label_file}: Line {i + 1}: Invalid coordinates x={x}, y={y}, w={w}, h={h}")
                valid = False

            # Check bounding box bounds
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            x2 = (x + w / 2) * width
            y2 = (y + h / 2) * height
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                out_of_bounds.append(
                    f"{label_file}: Line {i + 1}: Box out of bounds (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
                valid = False

            # Draw box and label
            if valid:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[hole_id], 2)
                label = f"{hole_names[hole_id]} {empty_full_names[empty_full]}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[hole_id], 2)

        # Save annotated image
        cv2.imwrite(output_img_path, img)

        # Display image for manual inspection
        cv2.imshow("Annotation Check", img)
        print(f"Displaying {label_file}. Press 'n' for next, 'q' to quit")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return  # Exit early

    cv2.destroyAllWindows()

    # Print error report
    print("\n=== Annotation Check Report ===")
    if missing_images:
        print("Missing Images:")
        for img in missing_images:
            print(f"  {img}")
    if missing_labels:
        print("Missing Labels:")
        for img in missing_labels:
            print(f"  {img}")
    if invalid_lines:
        print("Invalid Lines:")
        for err in invalid_lines:
            print(f"  {err}")
    if invalid_values:
        print("Invalid Values:")
        for err in invalid_values:
            print(f"  {err}")
    if out_of_bounds:
        print("Out of Bounds Boxes:")
        for err in out_of_bounds:
            print(f"  {err}")
    if not (missing_images or missing_labels or invalid_lines or invalid_values or out_of_bounds):
        print("All annotations passed checks!")
    print(f"Annotated images saved to {output_dir}")


# Paths
label_dir = "Annotations_NN2_complete"
image_dir = "Images_NN2"
output_dir = "Annotations_NN2_check"

# Run
check_annotations(label_dir, image_dir, output_dir)