# Introduction

Here is my python source code for Human Action Recognition - a skeleton-based action classification system. With my code, you could:
* Run an app which you could extract skeleton keypoints and classify actions from a video file (`.mp4` or `.avi`)
* Run an app which you could evaluate the model's accuracy on the test dataset
  <p align="center">
  <img src="https://github.com/user-attachments/assets/placeholder-image-1" width="30%"/>
  <img src="https://github.com/user-attachments/assets/placeholder-image-2" width="30%"/>
  <img src="https://github.com/user-attachments/assets/placeholder-image-3" width="30%"/></p>

# Action Recognition

In order to use this repo, you need an action video or a sequence of extracted frames. When a person appears in the frame, their skeleton keypoints will be detected and tracked using YOLO-Pose. These temporal sequences of keypoints are then fed into a PyTorch LSTM network; whenever the sequence reaches a fixed length, the model will predict the current action being performed. 
Below are the scripts to run the demo:
* **For video:** simply run `python3 test_video.py`
* **For dataset evaluation:** simply run `python3 test_dataset.py`

# Dataset

The dataset used for training my model is the **[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)** dataset, which is a challenging real-world action recognition dataset collected from YouTube. 
Due to its large size, I only uploaded a small sample of extracted keypoint sequences (`sample_sequence/`) and a demo video in the `data/` folder for testing purposes. You can download the full original dataset from their official website.

# Categories

The table below shows some examples of the 101 action categories my model used for classification:

| BodyWeightSquats | BoxingPunchingBag | JumpingJack | Lunges | PushUps | TennisSwing | WalkingWithDog |
|:---:|:---:|:---:|:---:|:---:|

# Training

You need to download the **[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)** videos and split files, then store them in your local folder. 
1. First, you need to extract the skeleton keypoints using YOLO-Pose by configuring the input paths and running the script:
   `python3 extract_keypoints.py`
   This script will automatically run pose estimation on every frame, normalize the coordinates, and save the sequences into `.npy` files for the train/val sets.
2. If you want to train your model with a different set of hyper-parameters, you only need to change the arguments (like `hidden_dim`, `num_layers`, `sequence_length`) in `train.py`.
3. Then you could simply run PyTorch LSTM training using the generated keypoint data:
   `python3 train.py --epochs 50 --batch_size 32`

# Experiments
<p align = "center">
<img width="700" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/placeholder-loss-curve-image" />
</p>

The LSTM model was trained using PyTorch on cloud platforms (Google Colab/Kaggle) to leverage GPU acceleration. I trained the model for 50 epochs. The model reached its convergence point and achieved impressive classification performance on the UCF101 scenarios. The key metrics for the best epoch are shown below:
* **Accuracy (Top-1):** ~XX.X%
* **Validation Loss:** X.XXX

# Requirements

* python 3.8+
* pytorch
* ultralytics (YOLOv8-Pose)
* opencv-python (cv2)
* numpy
