import os


def validate_split(split, dataset_dir):
    image_dir = os.path.join(dataset_dir, split, "images")
    label_dir = os.path.join(dataset_dir, split, "labels")

    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    labels = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    print(f"{split.capitalize()} split: {len(images)} images, {len(labels)} labels")

    # Check for missing labels
    for image in images:
        label = image.replace(".jpg", ".txt")
        if label not in labels:
            print(f"Missing label for {image}")

    # Check for orphaned labels
    for label in labels:
        image = label.replace(".txt", ".jpg")
        if image not in images:
            print(f"Missing image for {label}")


# Validate all splits
dataset_dir = "Datasets/Dataset_NN2/Dataset_NN2"
for split in ["train", "val", "test"]:
    validate_split(split, dataset_dir)