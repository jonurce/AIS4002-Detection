import cv2
import os
import datetime
import time
import numpy as np
import psutil
import csv
import torch
from ultralytics import YOLO

# === Try importing pynvml if CUDA is available ===
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetPowerUsage
    )
    nvml_available = True
except ImportError:
    nvml_available = False

# === Setup paths ===
nn = 1
model_name = f"yolov8n_NN{nn}"
model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/weights/best.pt"
output_dir = f"RealTime/{model_name}"
os.makedirs(output_dir, exist_ok=True)

# === CSV logging setup ===
csv_path = os.path.join(output_dir, "performance_metrics.csv")
write_header = not os.path.exists(csv_path)
csv_file = open(csv_path, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["timestamp", "inference_time_ms", "fps", "cpu_percent", "gpu_percent", "vram_gb", "power_w"])

# === Check for CUDA ===
use_cuda = torch.cuda.is_available()
if use_cuda and nvml_available:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

# === Load the YOLOv8 model ===
model = YOLO(model_path)

# === Initialize the camera ===
camera_index = 6  # Adjust as needed
cap = cv2.VideoCapture(camera_index) # cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {camera_index}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened at {width}x{height} @ {fps} FPS")

try:
    print("Real-time detection started. Press SPACE to save frame, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # === Start inference timing ===
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        results = model.predict(frame, conf=0.7, iou=0.5, device="cuda" if use_cuda else "cpu")
        end = time.perf_counter()

        inference_time_ms = (end - start) * 1000
        current_fps = 1000 / inference_time_ms

        # === Resource monitoring ===
        cpu_util = psutil.cpu_percent()
        gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu if use_cuda and nvml_available else "N/A"
        power_w = nvmlDeviceGetPowerUsage(handle) / 1000 if use_cuda and nvml_available else "N/A"
        vram_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else "N/A"

        # === Draw detections ===
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            class_names = result.names

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names[int(cls)]} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # === Overlay metrics ===
        metrics = [
            f"Inference: {inference_time_ms:.1f} ms",
            f"FPS: {current_fps:.1f}",
            f"CPU: {cpu_util:.1f}%",
            f"GPU: {gpu_util}%" if gpu_util != "N/A" else "GPU: N/A",
            f"VRAM: {vram_gb:.2f} GB" if isinstance(vram_gb, float) else "VRAM: N/A",
            f"Power: {power_w:.1f} W" if isinstance(power_w, float) else "Power: N/A",
        ]

        for i, text in enumerate(metrics):
            cv2.putText(frame, text, (10, 20 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # === Save metrics to CSV ===
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([
            timestamp,
            round(inference_time_ms, 2),
            round(current_fps, 2),
            round(cpu_util, 2),
            gpu_util,
            round(vram_gb, 3) if isinstance(vram_gb, float) else "N/A",
            round(power_w, 2) if isinstance(power_w, float) else "N/A"
        ])
        csv_file.flush()

        # === Show frame ===
        cv2.imshow('L515 YOLOv8 Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            timestamp_img = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename = os.path.join(output_dir, f'detection_{timestamp_img}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Detection saved: {filename}")
        elif key == ord('q'):
            print("Exiting...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
