# Detection Model for Compartment Detection and Classification
This project trains and compares 5 object detection and classification models (YOLOv8n, YOLOv8s, YOLOv8, YOLOv8l, Faster-RCNN) for two different detection tasks:

- Task 1: Detect a compartment and classify it as:
  - 0: Drone
  - 1: Station
 
<img width="1920" height="546" alt="image" src="https://github.com/user-attachments/assets/3eca4fc6-d149-4f1a-a8a5-7efbd386ede9" />
  
- Task 2: Detect a compartment and classify it as:
  - 0: Drone Empty
  - 1: Drone Full
  - 2: Station Top Left Empty
  - 3: Station Top Left Full
  - 4: Station Top Right Empty
  - 5: Station Top Right Full
  - 6: Station Bottom Left Empty
  - 7: Station Bottom Left Full
  - 8: Station Bottom Right Empty
  - 9: Station Bottom Right Full

<img width="1611" height="917" alt="image" src="https://github.com/user-attachments/assets/363dfbe8-e91f-4e2b-a4fb-9c0f6a07ca3b" />


All the details are described in the project's report:
[View Report](Project_Report/Object_Detection_Report.pdf)

## Folder and Files Structure
The folders and files are structured in the next way. It's a little bit messy, as there are many model-task combinations, but the number in the .py  files indicate the order of execution:

```
├── Datasets
│    ├── Dataset_NN1
│    │    ├── Dataset_NN1
│    │    │      ├── test
│    │    │      ├── train
│    │    │      └── val
│    │    ├── 1_CheckLabels_NN1.py
│    │    ├── 4_TrainRCNN_NN1.py
│    │    ├── 4_TrainYOLOv8_NN1.py
│    │    ├── 5_TestRCNN_NN1.py
│    │    ├── 5_TestYOLOv8_NN1.py
│    │    ├── 6_RealTime_YOLO_NN1.py
│    │    ├── 6_RealTimeRCNN_NN1.py
│    │    ├── classes_NN1.txt
│    │    └── data_NN1.yaml
│    ├── Dataset_NN2
│    │    ├── Dataset_NN2
│    │    │      ├── test
│    │    │      ├── train
│    │    │      └── val
│    │    ├── 0_AddEmptyFull.py
│    │    ├── 1_CheckLabels_NN2.py
│    │    ├── 2_1_ConvertToHoleID.py
│    │    ├── 4_TrainRCNN_NN2.py
│    │    ├── 4_TrainYOLOv8_NN2.py
│    │    ├── 5_TestRCNN_NN2.py
│    │    ├── 5_TestYOLOv8_NN2.py
│    │    ├── 6_RealTimeDetectionRCNN_NN2.py
│    │    ├── classes_NN2.txt
│    │    └── data_NN2.yaml
│    ├── Original
│    └── Robustness
├── FeatureMaps
├── Plots
├── ProjectReports
├── 0_TakePictures
├── 2_SplitDataset
├── 3_ValidateDataset
├── 6_1_RealTime_yolo
├── 6_1_RealTimeCPU_yolo
├── 6_2_RealTime_RCNN
├── 6_2_RealTimeCPU_RCNN
├── 7_1_json_to_yolo
├── 7_2_confusion_plots_tests
├── 7_3_robust_rcnn
├── 7_3_robust_yolo
├── 7_plot_total_losses
├── 8_mac_rcnn
└── 9_feature_maps
```
