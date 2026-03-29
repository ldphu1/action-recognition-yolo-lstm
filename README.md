# Introduction

Here is my python source code for Human Action Recognition - a skeleton-based action classification system. With my code, you could:
* Extract normalized human body keypoints from videos using YOLOv8-Pose (`pose_extraction.py`)
* Train an Attention-LSTM network to classify temporal action sequences (`train.py`)
* Run a LIVE real-time inference application using your webcam to track and predict actions (`predict.py`)
  <p align="center">
  <img src="https://github.com/user-attachments/assets/placeholder-image-1" width="30%"/>
  <img src="https://github.com/user-attachments/assets/placeholder-image-2" width="30%"/>
  <img src="https://github.com/user-attachments/assets/placeholder-image-3" width="30%"/></p>

# Real-Time Action Action Recognition

**How it works:**
1.  Captures video frames continuously from the webcam.
2.  Detects and tracks the main person (largest bounding box) using `yolov8n-pose.pt`.
3.  Dynamically extracts and normalizes 17 skeleton keypoints frame-by-frame.
4.  Maintains a rolling window of **60 frames** (`MAX_FRAME = 60`) of keypoints.
5.  Feeds the sequence into the trained Attention-LSTM network.
6.  Smoothes the final prediction over a 30-frame history to display a stable current action label on the screen.
7.  
### How to run the Demo
1.  Connect your webcam.
2.  Ensure your trained model weights (`best_model.pt`) are in the root directory.
3.  Run the live inference script:
    ```bash
    python3 predict.py
    ```
    *(Note: The application will open a window showing your webcam feed with the skeleton drawn and the predicted action displayed at the top. To exit the live feed, press **`q`** on your keyboard).*

*(Note for Performance: Running both YOLOv8-pose and an LSTM in real-time is computationally demanding. For a smooth experience (>= 15 FPS), a dedicated GPU (e.g., NVIDIA) is highly recommended. If running on a CPU, expect lower FPS).*
# Dataset

The dataset used for training my model is a subset of the **[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)** dataset.
The raw videos should be placed in the `UFC101/` folder, and the train/val splits are defined in `UFC101/train.csv` and `UFC101/val.csv`.

# Categories

The model is currently trained to classify **7 specific action categories**. The table below shows the categories used for classification:

| BodyWeightSquats | BoxingPunchingBag | JumpingJack | Lunges | PushUps | TennisSwing | WalkingWithDog |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

# Training

You need to download the **[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)** videos and split files, then store them in your local folder. 
1. First, you need to extract the skeleton keypoints using YOLO-Pose by configuring the input paths and running the script:
   `python3 extract_keypoints.py`
   This script will automatically run pose estimation on every frame, normalize the coordinates, and save the sequences into `.npy` files for the train/val sets.
2. If you want to train your model with a different set of hyper-parameters, you only need to change the arguments (like `hidden_dim`, `num_layers`, `sequence_length`) in `train.py`.
3. Then you could simply run PyTorch LSTM training using the generated keypoint data:
   `python3 train.py --epochs 30 --batch_size 16`

# Experiments

<p align = "center">
<img width="700" src="https://github.com/user-attachments/assets/93fdf792-9140-4cc2-84c2-9ec0922d946d" />
  
The model structure (`model.py`) utilizes LSTM combined with an Attention mechanism and a fully connected layer with Dropout/LayerNorm for robust feature learning.

I trained the model for 30 epochs using the Adam optimizer and CrossEntropyLoss. The model's performance was monitored using TensorBoard (runs/tensorboard). During training, the checkpoint with the highest macro F1-score on the validation set is saved as (`best_model.pt`).
</p>

# Requirements

* python 3.8+
* pytorch
* ultralytics (YOLOv8-Pose)
* opencv-python (cv2)
* numpy
