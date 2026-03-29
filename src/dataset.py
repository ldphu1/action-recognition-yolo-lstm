from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch

MAX_FRAME = 60
INPUT_SIZE = 34

class SkeDataset(Dataset):
    def __init__(self, root, train = True):
        self.root = root
        if train:
            self.root = os.path.join(root, "train" )
        else:
            self.root = os.path.join(root, "val")

        self.categories = sorted(os.listdir(self.root))
        self.file_paths = []
        self.labels = []
        for class_name in self.categories:
            class_folder = os.path.join(self.root, class_name)
            if not os.path.isdir(class_folder):
                continue

            label = self.categories.index(class_name)
            for file in os.listdir(class_folder):
                if file.endswith(".npy"):
                    self.file_paths.append(os.path.join(class_folder, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        frames = data.shape[0]

        if frames < MAX_FRAME:
            pad = np.zeros((MAX_FRAME - frames, INPUT_SIZE))
            data = np.vstack((data, pad))

        if frames > MAX_FRAME:
            data = data[:MAX_FRAME, :]

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
