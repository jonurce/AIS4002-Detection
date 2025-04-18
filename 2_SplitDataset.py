import os
import shutil
import random

# Paths
image_dir = "Datasets/Dataset_NN2/Images_NN2"
annotation_dir = "Datasets/Dataset_NN2/Annotations_NN2_hole_id" # class_id, x_center, y_center, box_width, box_height
dataset_dir = "Datasets/Dataset_NN2/Dataset_NN2"
splits = {"train": 0.8, "val": 0.1, "test": 0.1}  # 80/10/10 split

# Create dataset directories
for split in splits:
    os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(image_files)  # Randomize for unbiased split

# Calculate split sizes
total_images = len(image_files)
train_size = int(splits["train"] * total_images)
val_size = int(splits["val"] * total_images)
test_size = total_images - train_size - val_size

# Assign images to splits
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Copy files to respective directories
for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    for image_file in files:
        # Copy image
        src_image_path = os.path.join(image_dir, image_file)
        dst_image_path = os.path.join(dataset_dir, split, "images", image_file)
        shutil.copy(src_image_path, dst_image_path)

        # Copy annotation
        annotation_file = image_file.replace(".jpg", ".txt")
        src_annotation_path = os.path.join(annotation_dir, annotation_file)
        dst_annotation_path = os.path.join(dataset_dir, split, "labels", annotation_file)
        if os.path.exists(src_annotation_path):
            shutil.copy(src_annotation_path, dst_annotation_path)
        else:
            print(f"Warning: No annotation for {image_file}")

print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")