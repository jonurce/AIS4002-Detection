import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_v2
import torchvision.transforms as T

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_v2(num_classes=11)
model.load_state_dict(torch.load("/Runs_NN2/faster_rcnn_NN2/best.pth"))
model.to(device)
model.eval()

# Class names
class_names = ['Background', 'Drone_Hole_Empty', 'Drone_Hole_Full', 'Station_TL_Empty', 'Station_TL_Full',
               'Station_TR_Empty', 'Station_TR_Full', 'Station_BL_Empty', 'Station_BL_Full',
               'Station_BR_Empty', 'Station_BR_Full']

# Camera
cap = cv2.VideoCapture(0)  # Replace with L515 pipeline
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img = T.ToTensor()(frame).to(device)
    with torch.no_grad():
        outputs = model([img])
    for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_names[label]} {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Faster R-CNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()