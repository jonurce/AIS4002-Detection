import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_model(num_classes, model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    model_path = "Runs_NN1/faster_rcnn_NN1/best.pt"
    test_image_dir = "Dataset_NN1/test/images"
    label_dir = "Dataset_NN1/test/labels"
    output_dir = "Tests_NN1/faster_rcnn_test_predictions_NN1"
    os.makedirs(output_dir, exist_ok=True)

    # Model
    num_classes = 3  # Background, Drone, Station
    classes = ["background", "Drone", "Station"]
    model = get_model(num_classes, model_path, device)
    model.eval()

    # Transforms
    transform = ToTensor()

    # Initialize containers
    all_predictions = []
    all_ground_truths = []
    image_id = 0
    y_true = []
    y_pred = []

    # Test images
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(".jpg")]

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(test_image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).to(device)
            img_h, img_w = img.shape[:2]

            # Predict
            outputs = model([img_tensor])[0]
            boxes = outputs["boxes"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()

            # Filter predictions by confidence
            conf_threshold = 0.7
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # Save prediction JSON
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                all_predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, width, height],
                    "score": float(score)
                })
                y_pred.append(int(label))

            # Save image with boxes
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                label_text = f"{classes[label]} {score:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(os.path.join(output_dir, img_file), img)

            # Load YOLO-format ground truths
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        cls, x_center, y_center, w, h = map(float, line.strip().split())
                        x = (x_center - w / 2) * img_w
                        y = (y_center - h / 2) * img_h
                        w *= img_w
                        h *= img_h
                        all_ground_truths.append({
                            "image_id": image_id,
                            "category_id": int(cls) + 1,  # offset for background class
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "id": len(all_ground_truths)
                        })
                        y_true.append(int(cls) + 1)  # offset for background
            image_id += 1

    # Save COCO-format files
    coco_gt_dict = {
        "images": [{"id": i} for i in range(image_id)],
        "annotations": all_ground_truths,
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }

    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(coco_gt_dict, f)
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(all_predictions, f)

    # Evaluate using pycocotools
    coco_gt = COCO(os.path.join(output_dir, "labels.json"))
    coco_dt = coco_gt.loadRes(os.path.join(output_dir, "predictions.json"))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save metrics to file
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write("Precision, Recall, mAP@0.5, mAP@0.5:0.95\n")
        f.write(f"{coco_eval.stats[0]:.4f}, {coco_eval.stats[1]:.4f}, "
                f"{coco_eval.stats[1]:.4f}, {coco_eval.stats[2]:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes[1:])
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
