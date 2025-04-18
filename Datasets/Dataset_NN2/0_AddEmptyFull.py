import os
import cv2


def add_empty_full(label_dir, image_dir, output_label_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_label_dir, exist_ok=True)

    # Get list of images
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    for label_file in label_files:
        # Paths
        img_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
        output_label_path = os.path.join(output_label_dir, label_file)

        # Load image
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        new_lines = []
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid line in {label_file}: {line}")
                continue
            hole_id, x, y, w, h = map(float, parts)

            # Convert normalized coordinates to pixel values and display image with bounding box
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display image
            cv2.imshow("Check Empty/Full", img)
            print(f"Hole ID {int(hole_id)} for {label_file}. Enter 0 (Empty) or 1 (Full):")

            # Get user input with validation
            while True:
                try:
                    key = cv2.waitKey(0)
                    if key == ord('0'):
                        empty_full = 0  # Convert '0' to integer 0
                        break
                    elif key == ord('1'):
                        empty_full = 1  # Convert '1' to integer 1
                        break
                    print("Please enter 0 (Empty) or 1 (Full)")
                except ValueError:
                    print("Invalid input. Enter 0 (Empty) or 1 (Full)")

            cv2.destroyAllWindows()
            new_lines.append(f"{int(hole_id)} {x} {y} {w} {h} {empty_full}")

            # Save new annotations to output directory
            with open(output_label_path, 'w') as f:
                f.write("\n".join(new_lines))
            print(f"Saved new annotations to {output_label_path}")


# Paths
label_dir = "Annotations_NN2_uncomplete"
image_dir = "Images_NN2"
output_label_dir = "Annotations_NN2_complete"

# Run
add_empty_full(label_dir, image_dir, output_label_dir)