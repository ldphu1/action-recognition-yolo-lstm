from ultralytics import YOLO
import numpy as np
import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_VIDEO_DIR = "UFC101"
OUT_DIR = "extracted_data"

model =  YOLO('yolov8n-pose.pt')

def normalize_keypoints(keypoints, bbox):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    keypoints[:, 0] = (keypoints[:, 0] - cx) / w
    keypoints[:, 1] = (keypoints[:, 1] - cy) / h

    return keypoints

def extract_and_save(video_path, output_path):
    all_frames_data = []

    cap = cv2.VideoCapture(video_path)

    prev_frame_data = None

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        results = model.track(frame, verbose=False)

        frame_data = None

        if len(results[0].boxes) > 0 and results[0].keypoints is not None:
            keypoints_all = results[0].keypoints.xy.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            max_area = 0
            main_actor_index = 0
            for i, box in enumerate(boxes):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    main_actor_index = i

            #normalize
            keypoints = normalize_keypoints(keypoints_all[main_actor_index], boxes[main_actor_index])
            hip_center = (keypoints[11] + keypoints[12]) / 2
            keypoints = keypoints - hip_center

            if keypoints.shape == (17, 2):
                frame_data = keypoints.flatten()
                prev_frame_data = frame_data

        #fallback if miss detect
        if frame_data is None:
            if prev_frame_data is not None:
                frame_data = prev_frame_data
            else:
                frame_data = np.zeros(34)

        all_frames_data.append(frame_data)

    cap.release()

    npy_data = np.array(all_frames_data)
    np.save(output_path, npy_data)

def process_dataset(csv_file, split_name):
    df = pd.read_csv(csv_file)

    for row in tqdm(df.itertuples(), total=len(df)):
        video_path = os.path.join(RAW_VIDEO_DIR, row.clip_path.lstrip("/"))
        video_name = row.clip_name
        label = row.label

        class_out_dir = os.path.join(OUT_DIR, split_name, label)
        os.makedirs(class_out_dir, exist_ok=True)

        npy_filename = video_name + '.npy'
        npy_out_path = os.path.join(class_out_dir, npy_filename)

        if not os.path.exists(npy_out_path):
            if os.path.exists(video_path):
                extract_and_save(video_path, npy_out_path)
            else:
                print("video not exist!")

if __name__ == '__main__':
    process_dataset("UFC101/train.csv", "train")
    process_dataset("UFC101/val.csv", "val")
    print("DONE")
