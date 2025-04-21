import os
import json
from collections import defaultdict
import cv2

for nn in range(1,3):
    # Paths
    labels_json = f"Datasets/Dataset_NN{nn}/Tests_NN{nn}/faster_rcnn_test_predictions_NN{nn}/labels.json"
    predictions_json = f"Datasets/Dataset_NN{nn}/Tests_NN{nn}/faster_rcnn_test_predictions_NN{nn}/predictions.json"  # ← Your predictions file
    image_dir = f"Datasets/Dataset_NN{nn}/Dataset_NN{nn}/test/images"              # ← Where the images are
    output_dir = f"Datasets/Dataset_NN{nn}/Tests_NN{nn}/faster_rcnn_test_predictions_NN{nn}/labels"             # ← Where to save YOLO .txt files

    os.makedirs(output_dir, exist_ok=True)

    # Load image list and sort it to match COCO order
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    # Map image_id (index) to filename
    id_to_filename = {i: fname for i, fname in enumerate(image_files)}

    # Load predictions
    with open(predictions_json, "r") as f:
        predictions = json.load(f)

    # Group predictions by image_id
    grouped_preds = defaultdict(list)
    for pred in predictions:
        grouped_preds[pred["image_id"]].append(pred)

    # Convert predictions to YOLO format
    for image_id, preds in grouped_preds.items():
        filename = id_to_filename.get(image_id)
        if not filename:
            continue

        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        yolo_lines = []
        for pred in preds:
            class_id = pred["category_id"] - 1  # convert COCO (1-based) to YOLO (0-based)
            x, y, bw, bh = pred["bbox"]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        # Save YOLO-style label
        txt_name = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(output_dir, txt_name), "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"✅ Converted predictions to YOLO .txt format at: {output_dir}")
