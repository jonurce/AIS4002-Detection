import os
import cv2


def add_empty_full(label_dir, image_dir, output_label_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_label_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            # Paths
            img_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
            output_label_path = os.path.join(output_label_dir, label_file)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            new_lines = []
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid line in {label_file}: {line.strip()}")
                        continue
                    hole_id, x, y, w, h = map(float, parts)

                    # Display image with bounding box
                    x1 = int((x - w / 2) * img.shape[1])
                    y1 = int((y - h / 2) * img.shape[0])
                    x2 = int((x + w / 2) * img.shape[1])
                    y2 = int((y + h / 2) * img.shape[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Check Empty/Full", img)
                    print(f"Hole ID {int(hole_id)} for {label_file}. Enter 0 (Empty) or 1 (Full):")

                    # Get user input with validation
                    while True:
                        try:
                            empty_full = int(input())
                            if empty_full in [0, 1]:
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
dataset_root = ""
label_dir = os.path.join(dataset_root, "Annotations_NN2_uncomplete")
image_dir = os.path.join(dataset_root, "Images_NN2")
output_label_dir = os.path.join(dataset_root, "Annotations_NN2_complete")

# Run
add_empty_full(label_dir, image_dir, output_label_dir)