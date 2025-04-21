import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from glob import glob

def load_yolo_labels(label_dir, class_offset=0):
    data = {}
    for path in glob(os.path.join(label_dir, "*.txt")):
        with open(path, "r") as f:
            labels = [int(line.strip().split()[0]) + class_offset for line in f.readlines()]
            filename = os.path.basename(path).replace(".txt", ".jpg")
            data[filename] = labels
    return data

def get_flattened_labels(gt_dict, pred_dict, filenames, background_label=None):
    y_true, y_pred = [], []
    for fname in filenames:
        gt = gt_dict.get(fname, [])
        pred = pred_dict.get(fname, [])

        # Naively match by padding with background
        max_len = max(len(gt), len(pred))
        gt += [background_label] * (max_len - len(gt))
        pred += [background_label] * (max_len - len(pred))

        y_true.extend(gt)
        y_pred.extend(pred)
    return y_true, y_pred

def plot_conf_matrices(model_dirs, gt_dir, class_names, task_name, save_path, model_names=None):
    print(f"\nGenerating confusion matrices for {task_name}...")
    background_label = len(class_names)  # For unmatched predictions/GTs
    class_names = class_names + ["Background"]

    gt_labels = load_yolo_labels(gt_dir, class_offset=0)
    filenames = sorted(gt_labels.keys())

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_dirs))]

    # Create 3 rows, 2 columns layout
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for i, (model_dir, model_name) in enumerate(zip(model_dirs, model_names)):
        model_labels = load_yolo_labels(model_dir, class_offset=0)
        y_true, y_pred = get_flattened_labels(gt_labels, model_labels, filenames, background_label)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

        disp.plot(
            ax=axes[i],
            cmap="Blues",
            colorbar=False,
            values_format="d",
            xticks_rotation=45
        )
        axes[i].set_title(model_name, fontsize=20)
        axes[i].tick_params(labelsize=12)
        for text in disp.text_.ravel():
            if text:
                text.set_fontsize(12)

    # Hide any unused subplots
    for j in range(len(model_dirs), rows * cols):
        fig.delaxes(axes[j])

    plt.suptitle(f"Confusion Matrices - {task_name}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")



def main():
    model_names = [
        "YOLOv8n",
        "YOLOv8s",
        "YOLOv8m",
        "YOLOv8l",
        "Faster R-CNN"
    ]

    # TASK 1
    task1_models = [
        "Datasets/Dataset_NN1/Tests_NN1/YOLOv8n_test_metrics_NN1/labels",
        "Datasets/Dataset_NN1/Tests_NN1/YOLOv8s_test_metrics_NN1/labels",
        "Datasets/Dataset_NN1/Tests_NN1/YOLOv8m_test_metrics_NN1/labels",
        "Datasets/Dataset_NN1/Tests_NN1/YOLOv8l_test_metrics_NN1/labels",
        "Datasets/Dataset_NN1/Tests_NN1/faster_rcnn_test_predictions_NN1/labels"
    ]
    task1_gt = "Datasets/Dataset_NN1/Dataset_NN1/test/labels"
    task1_classes = ["Drone", "Station"]
    plot_conf_matrices(task1_models, task1_gt, task1_classes, "Task 1", "Plots/conf_matrix_task1.png", model_names=model_names)

    # TASK 2
    task2_models = [
        "Datasets/Dataset_NN2/Tests_NN2/YOLOv8n_test_metrics_NN2/labels",
        "Datasets/Dataset_NN2/Tests_NN2/YOLOv8s_test_metrics_NN2/labels",
        "Datasets/Dataset_NN2/Tests_NN2/YOLOv8m_test_metrics_NN2/labels",
        "Datasets/Dataset_NN2/Tests_NN2/YOLOv8l_test_metrics_NN2/labels",
        "Datasets/Dataset_NN2/Tests_NN2/faster_rcnn_test_predictions_NN2/labels"
    ]
    task2_gt = "Datasets/Dataset_NN2/Dataset_NN2/test/labels"
    task2_classes = [
        "Drone_Hole_Empty", "Drone_Hole_Full",
        "Station_TL_Empty", "Station_TL_Full",
        "Station_TR_Empty", "Station_TR_Full",
        "Station_BL_Empty", "Station_BL_Full",
        "Station_BR_Empty", "Station_BR_Full"
    ]
    plot_conf_matrices(task2_models, task2_gt, task2_classes, "Task 2", "Plots/conf_matrix_task2.png", model_names=model_names)

if __name__ == "__main__":
    main()
