import cv2
import os
import glob

# Define output video path
output_video = "feature_map_video.mp4"
target_size = (1280, 736)  # Match input/prediction image size

# Collect all PNG files in the current directory
image_files = sorted(glob.glob("*.png"))
if not image_files:
    print("Error: No PNG files found in the current directory")
    exit()

# Print included images for debugging
print(f"Found {len(image_files)} images: {[os.path.basename(f) for f in image_files]}")

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, 1.5, target_size)

# Write images to video
for img_file in image_files:
    img = cv2.imread(img_file)
    if img is None:
        print(f"Warning: Could not load image {img_file}")
        continue
    img = cv2.resize(img, target_size)
    video_writer.write(img)
    # Hold input image (first image) and prediction image (last image) for 2 seconds (4 frames at 2 FPS)
    if "input_image" in img_file or "prediction" in img_file:
        for _ in range(5):  # 3 additional frames (1 + 3 = 4 frames total)
            video_writer.write(img)

video_writer.release()
print(f"Video saved to {output_video}")