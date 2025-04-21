import pandas as pd
import matplotlib.pyplot as plt
import os

output_dir = 'Plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for nn in range(1,3):

    font_size = 16

    # Load the CSV files
    data_n = pd.read_csv(f'Datasets/Dataset_NN{nn}/Runs_NN{nn}/yolov8n_NN{nn}/results.csv')
    data_s = pd.read_csv(f'Datasets/Dataset_NN{nn}/Runs_NN{nn}/yolov8s_NN{nn}/results.csv')
    data_m = pd.read_csv(f'Datasets/Dataset_NN{nn}/Runs_NN{nn}/yolov8m_NN{nn}/results.csv')
    data_l = pd.read_csv(f'Datasets/Dataset_NN{nn}/Runs_NN{nn}/yolov8l_NN{nn}/results.csv')
    data_rcnn = pd.read_csv(f'Datasets/Dataset_NN{nn}/Runs_NN{nn}/faster_rcnn_NN{nn}/results.csv')

    # Compute total train and validation losses
    data_n['total_train_loss'] = data_n['train/box_loss'] + data_n['train/cls_loss'] + data_n['train/dfl_loss']
    data_n['total_val_loss'] = data_n['val/box_loss'] + data_n['val/cls_loss'] + data_n['val/dfl_loss']

    data_s['total_train_loss'] = data_s['train/box_loss'] + data_s['train/cls_loss'] + data_s['train/dfl_loss']
    data_s['total_val_loss'] = data_s['val/box_loss'] + data_s['val/cls_loss'] + data_s['val/dfl_loss']

    data_m['total_train_loss'] = data_m['train/box_loss'] + data_m['train/cls_loss'] + data_m['train/dfl_loss']
    data_m['total_val_loss'] = data_m['val/box_loss'] + data_m['val/cls_loss'] + data_m['val/dfl_loss']

    data_l['total_train_loss'] = data_l['train/box_loss'] + data_l['train/cls_loss'] + data_l['train/dfl_loss']
    data_l['total_val_loss'] = data_l['val/box_loss'] + data_l['val/cls_loss'] + data_l['val/dfl_loss']

    data_rcnn['total_train_loss'] = data_rcnn['train/box_loss'] + data_rcnn['train/cls_loss'] + data_rcnn['train/dfl_loss']
    data_rcnn['total_val_loss'] = data_rcnn['val/box_loss'] + data_rcnn['val/cls_loss'] + data_rcnn['val/dfl_loss']

    # Plot 1: Total Train Loss vs. Epoch Yolo models (Log Scale)
    plt.figure(figsize=(8, 6))
    plt.plot(data_n['epoch'], data_n['total_train_loss'], 'b-', label='YOLOv8n')
    plt.plot(data_s['epoch'], data_s['total_train_loss'], 'r-', label='YOLOv8s')
    plt.plot(data_m['epoch'], data_m['total_train_loss'], 'g-', label='YOLOv8m')
    plt.plot(data_l['epoch'], data_l['total_train_loss'], 'orange', label='YOLOv8l')
    plt.yscale('log')  # Set logarithmic scale for y-axis
    plt.title(f'Total Training Loss for YOLO Models in Task {nn}', fontsize=font_size+2)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Total Train Loss (Log Scale)', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # Tick labels
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.legend(fontsize=font_size)
    plt.savefig(f'{output_dir}/train_loss_yolo_nn{nn}.png')
    plt.close()

    # Plot 2: Total Validation Loss vs. Epoch Yolo models (Log Scale)
    plt.figure(figsize=(8, 6))
    plt.plot(data_n['epoch'], data_n['total_val_loss'], 'b-', label='YOLOv8n')
    plt.plot(data_s['epoch'], data_s['total_val_loss'], 'r-', label='YOLOv8s')
    plt.plot(data_m['epoch'], data_m['total_val_loss'], 'g-', label='YOLOv8m')
    plt.plot(data_l['epoch'], data_l['total_val_loss'], 'orange', label='YOLOv8l')
    plt.yscale('log')  # Set logarithmic scale for y-axis
    plt.title(f'Total Validation Loss for YOLO Models in Task {nn}', fontsize=font_size+2)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Total Validation Loss (Log Scale)', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # Tick labels
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.legend(fontsize=font_size)
    plt.savefig(f'{output_dir}/val_loss_nn{nn}.png')
    plt.close()

    # Plot 3: Total Train and Validation Loss vs. Epoch for Faster R-CNN (Log Scale)
    plt.figure(figsize=(8, 6))
    plt.plot(data_rcnn['epoch'], data_rcnn['total_train_loss'], 'r-', label='Training Loss')
    plt.plot(data_rcnn['epoch'], data_rcnn['total_val_loss'], 'b-', label='Validation')
    plt.yscale('log')  # Set logarithmic scale for y-axis
    plt.title(f'Total Training and Validation Loss for Faster R-CNN in Task {nn}', fontsize=font_size + 2)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Total Validation Loss (Log Scale)', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # Tick labels
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.legend(fontsize=font_size)
    plt.savefig(f'{output_dir}/loss_rcnn_nn{nn}.png')
    plt.close()

    # Plot 4: Precision vs. Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(data_n['epoch'], data_n['metrics/precision(B)'], 'b-', label='YOLOv8n')
    plt.plot(data_s['epoch'], data_s['metrics/precision(B)'], 'r-', label='YOLOv8s')
    plt.plot(data_m['epoch'], data_m['metrics/precision(B)'], 'g-', label='YOLOv8m')
    plt.plot(data_l['epoch'], data_l['metrics/precision(B)'], 'orange', label='YOLOv8l')
    plt.plot(data_rcnn['epoch'], data_rcnn['metrics/precision(B)'], 'pink', label='Faster R-CNN')
    plt.title(f'Precision in Task {nn}', fontsize=font_size)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # Tick labels
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.legend(fontsize=font_size)
    plt.savefig(f'{output_dir}/precision_nn{nn}.png')
    plt.close()

    # Plot 5: Recall vs. Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(data_n['epoch'], data_n['metrics/recall(B)'], 'b-', label='YOLOv8n')
    plt.plot(data_s['epoch'], data_s['metrics/recall(B)'], 'r-', label='YOLOv8s')
    plt.plot(data_m['epoch'], data_m['metrics/recall(B)'], 'g-', label='YOLOv8m')
    plt.plot(data_l['epoch'], data_l['metrics/recall(B)'], 'orange', label='YOLOv8l')
    plt.plot(data_rcnn['epoch'], data_rcnn['metrics/recall(B)'], 'pink', label='Faster R-CNN')
    plt.title(f'Recall in Task {nn}', fontsize=font_size+2)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Recall', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # Tick labels
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.legend(fontsize=font_size)
    plt.savefig(f'{output_dir}/recall_nn{nn}.png')
    plt.close()

    # Plot 6: mAP@0.5:0.95 vs. Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(data_n['epoch'], data_n['metrics/mAP50-95(B)'], 'b-', label='YOLOv8n')
    plt.plot(data_s['epoch'], data_s['metrics/mAP50-95(B)'], 'r-', label='YOLOv8s')
    plt.plot(data_m['epoch'], data_m['metrics/mAP50-95(B)'], 'g-', label='YOLOv8m')
    plt.plot(data_l['epoch'], data_l['metrics/mAP50-95(B)'], 'orange', label='YOLOv8l')
    plt.plot(data_rcnn['epoch'], data_rcnn['metrics/mAP50-95(B)'], 'pink', label='Faster R-CNN')
    plt.title(f'mAP@0.5:0.95 in Task {nn}', fontsize=font_size)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('mAP@0.5:0.95', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # Tick labels
    plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
    plt.legend(fontsize=font_size)
    plt.savefig(f'{output_dir}/mAP50-95_nn{nn}.png')
    plt.close()