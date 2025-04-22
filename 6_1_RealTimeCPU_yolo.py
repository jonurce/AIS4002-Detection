import cv2
import os
import datetime
import time
import numpy as np
import psutil
import csv
import torch
from ultralytics import YOLO

# === Setup paths ===
nn = 2
model_name = f"yolov8l_NN{nn}"
model_path = f"Datasets/Dataset_NN{nn}/Runs_NN{nn}/{model_name}/weights/best.pt"
output_dir = f"RealTime/{model_name}"
os.makedirs(output_dir, exist_ok=True)

# === Check for CUDA ===
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Using device: {device}")

# === Load the YOLOv8 model ===
model = YOLO(model_path)

# === Initialize the camera ===
camera_index = 0  # Adjust as needed
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {camera_index}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened at {width}x{height} @ {fps} FPS")

# === Metrics storage ===
metrics_list = []
start_time = time.time()
duration = 10  # Run for 10 seconds

# === CSV setup ===
csv_path = os.path.join(output_dir, "metrics.csv")
csv_columns = ["timestamp", "inference_time_ms", "fps", "cpu_percent", "ram_used_gb", "power_watts"]

# === Power estimation setup ===
def estimate_power(cpu_percent, use_cuda=False):
    # Refined model: Base power + CPU/GPU contribution
    base_power = 5.0  # Idle system power (screen, etc.), typical for Zenbook
    max_cpu_power = 15.0  # Typical TDP for U-series CPU (e.g., Intel i5/i7 U); adjust to 28.0 for H-series
    cpu_power = (cpu_percent / 100.0) * max_cpu_power
    gpu_power = 10.0 if use_cuda else 0.0  # Approximate GPU power for NVIDIA or Iris Xe
    return base_power + cpu_power + gpu_power

# === Check battery measurement availability ===
battery_available = psutil.sensors_battery() is not None
if not battery_available:
    print("Warning: Battery power measurement unavailable. Using CPU-based estimation.")
elif psutil.sensors_battery().power_plugged:
    print("Warning: Laptop is plugged in. Battery power data unavailable; using CPU-based estimation.")

try:
    print("Real-time detection started for 10 seconds. Press SPACE to save frame, 'q' to quit early.")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # === Start inference timing ===
            start = time.perf_counter()
            results = model.predict(frame, conf=0.7, iou=0.5, device=device)
            end = time.perf_counter()

            inference_time_ms = (end - start) * 1000
            current_fps = 1000 / inference_time_ms

            # === Resource monitoring ===
            cpu_util = psutil.cpu_percent()
            ram_used_gb = psutil.virtual_memory().used / 1e9  # System RAM in GB

            # === Power monitoring ===
            power_watts = 0.0
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged and battery_available:
                # Use battery discharge rate (system-wide power in watts)
                # Note: Negative values indicate discharge; convert to positive
                power_watts = abs(battery.power) if hasattr(battery, 'power') else 0.0
                if power_watts == 0.0:
                    # Fallback if battery.power is not populated
                    power_watts = estimate_power(cpu_util, use_cuda)
            else:
                # Fallback to estimation if plugged in or no battery data
                power_watts = estimate_power(cpu_util, use_cuda)

            # === Store metrics ===
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics = {
                "timestamp": timestamp,
                "inference_time_ms": inference_time_ms,
                "fps": current_fps,
                "cpu_percent": cpu_util,
                "ram_used_gb": ram_used_gb,
                "power_watts": power_watts
            }
            metrics_list.append(metrics)
            writer.writerow(metrics)

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
            display_metrics = [
                f"Inference: {inference_time_ms:.1f} ms",
                f"FPS: {current_fps:.1f}",
                f"CPU: {cpu_util:.1f}%",
                f"RAM: {ram_used_gb:.2f} GB",
                f"Power: {power_watts:.2f} W"
            ]
            for i, text in enumerate(display_metrics):
                cv2.putText(frame, text, (10, 20 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # === Show frame ===
            cv2.imshow('L515 YOLOv8 Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                timestamp_img = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                filename = os.path.join(output_dir, f'detection_{timestamp_img}.jpg')
                cv2.imwrite(filename, frame)
                print(f"Detection saved: {filename}")
            elif key == ord('q'):
                print("Exiting early...")
                break

    # === Compute and print averages ===
    if metrics_list:
        avg_inference_time = np.mean([m["inference_time_ms"] for m in metrics_list])
        avg_fps = np.mean([m["fps"] for m in metrics_list])
        avg_cpu_util = np.mean([m["cpu_percent"] for m in metrics_list])
        avg_ram_used = np.mean([m["ram_used_gb"] for m in metrics_list])
        avg_power_watts = np.mean([m["power_watts"] for m in metrics_list])

        print("\n=== Performance Summary ===")
        print(f"Average Inference Time: {avg_inference_time:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average CPU Utilization: {avg_cpu_util:.2f}%")
        print(f"Average RAM Usage: {avg_ram_used:.2f} GB")
        print(f"Average Power Consumption: {avg_power_watts:.2f} W")
        if battery_available and not psutil.sensors_battery().power_plugged:
            print("(Power measured via battery discharge rate)")
        else:
            print("(Power estimated based on CPU/GPU utilization)")

finally:
    cap.release()
    cv2.destroyAllWindows()