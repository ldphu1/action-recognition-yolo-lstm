# Introduction

Here is my python source code for Human Action Recognition - a skeleton-based action classification system. With my code, you could:
* Extract normalized human body keypoints from videos using YOLOv8-Pose (`pose_extraction.py`)
* Train an Attention-LSTM network to classify temporal action sequences (`train.py`)
* Run an inference app which tracks the main actor's skeleton and predicts actions from a single video file (`predict.py`)
  <p align="center">
  <img src="https://github.com/user-attachments/assets/f92cf236-7f3e-433c-8858-a8f5a1b40d52" width="30%">
  <img src="https://github.com/user-attachments/assets/b138c178-989a-4a1a-9948-227bb0e45618" width="30%"/>
  <img src="https://github.com/user-attachments/assets/b08f5626-6360-484a-bb42-cd69ea490931" width="30%"/>


# Action Recognition
In order to use this repo, you need an action video. When a person appears in the frame, their 17 skeleton keypoints will be detected and tracked using `yolov8n-pose.pt`. The keypoints are dynamically normalized based on the bounding box and shifted relative to the hip center.

These temporal sequences of keypoints are collected in a rolling window of **60 frames** (`MAX_FRAME = 60`) and fed into a PyTorch LSTM network. A prediction history buffer (30 frames) is used to smooth the output and predict the most stable current action.

* **For video inference:** simply run `python3 predict.py`. 
  *(Note: You can change the input video path and output path inside the `predict.py` file).*

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

<img width="680" height="622" alt="Screenshot 2026-03-29 212903" src="https://github.com/user-attachments/assets/4d2110d2-e5cc-4b97-82d3-f16017777583" />

The model structure (`model.py`) utilizes LSTM combined with an Attention mechanism and a fully connected layer with Dropout/LayerNorm for robust feature learning.

I trained the model for 30 epochs using the Adam optimizer and CrossEntropyLoss. The model's performance was monitored using TensorBoard (runs/tensorboard). During training, the checkpoint with the highest macro F1-score on the validation set is saved as (`best_model.pt`).
</p>

# Requirements

* python 3.8+
* pytorch
* ultralytics (YOLOv8-Pose)
* opencv-python (cv2)
* numpy
