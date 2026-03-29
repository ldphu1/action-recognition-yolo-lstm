from collections import deque
from ultralytics import YOLO
import cv2
import numpy as np
from model import *
import torch

CLASSES = ['BodyWeightSquats', 'BoxingPunchingBag', 'JumpingJack', 'Lunges', 'PushUps', 'TennisSwing', 'WalkingWithDog']

MAX_FRAME = 60
INPUT_SIZE = 34
HIDDEN_SIZE = 128
NUM_CLASSES = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolov8n-pose.pt")

lstm_model = LSTM( input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, num_classes = NUM_CLASSES).to(device)
lstm_model.load_state_dict(torch.load("best_model.pt", map_location=device))
lstm_model.eval()

cap = cv2.VideoCapture(r"C:\Users\Asus\Desktop\DL\project2\UFC101\train\Lunges\v_Lunges_g06_c01.avi")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(width), int(height)))

sequence = deque(maxlen=MAX_FRAME)
pred_history = deque(maxlen=30)

prev_frame_data = np.zeros(34)

def normalize_keypoints(keypoints, bbox):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    keypoints[:, 0] = (keypoints[:, 0] - cx) / w
    keypoints[:, 1] = (keypoints[:, 1] - cy) / h

    return keypoints

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break

    results = yolo_model(frame, verbose=False)
    frame_data = np.zeros(INPUT_SIZE)

    xy = []

    if results[0].boxes is not None and results[0].keypoints is not None and len(results[0].boxes) > 0:
        keypoints_all = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Chọn bbox lớn nhất
        max_area = 0
        main_actor_index = 0
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > max_area:
                xy = [box[0], box[1]]
                max_area = area
                main_actor_index = i

        main_actor_result = results[0][main_actor_index]
        frame = main_actor_result.plot(labels=False, conf=False)

        #normalize
        keypoints = normalize_keypoints(keypoints_all[main_actor_index], boxes[main_actor_index])
        hip_center = (keypoints[11] + keypoints[12]) / 2
        keypoints = keypoints - hip_center

        if frame_data is None:
            if prev_frame_data is not None:
                frame_data = prev_frame_data
            else:
                frame_data = np.zeros(34)

        frame_data = keypoints.flatten()
        sequence.append(frame_data)

    else:
        if len(sequence) > 0:
            sequence.append(sequence[-1])

    # predict
    if len(sequence) == MAX_FRAME:
        with torch.inference_mode():
            seq_tensor = torch.tensor(
                np.array(sequence),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            outputs = lstm_model(seq_tensor)

            _, predicted = torch.max(outputs, 1)

            pred_history.append(predicted.item())

            final_pred = max(
                set(pred_history),
                key=pred_history.count
            )

            current_action = CLASSES[final_pred]

            text_pos = (20, 40)

            if len(xy) == 2:
                text_pos = (int(xy[0]), int(xy[1]))

        cv2.putText(frame, current_action, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()