from model import *
from dataset import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

DATA_DIR = "extracted_data"
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 30
LEARNING_RATE = 0.001

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = SkeDataset(DATA_DIR, train=True)
    val_data   = SkeDataset(DATA_DIR, train=False)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_data.categories)
    model = LSTM(input_size=34, hidden_size=128, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter("/tensorboard")

    best_f1 = 0
    global_step = 0
    for epoch in range(EPOCHS):
        # ===== train =====
        model.train()
        running_loss = 0
        progress_bar = tqdm(train_loader)

        for keypoint, label in progress_bar:

            keypoint = keypoint.to(device)
            label    = label.to(device)

            output = model(keypoint)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("Loss", loss.item(), global_step)
            global_step += 1

            progress_bar.set_description(
                f"Epoch {epoch+1}/{EPOCHS} | loss {loss.item():.4f}"
            )

        train_loss = running_loss / len(train_loader)

        # ===== validate =====
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.inference_mode():
            for keypoint, label in val_loader:
                keypoint = keypoint.to(device)
                label    = label.to(device)

                output = model(keypoint)

                pred = output.argmax(dim=1)

                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_f1 = f1_score(all_labels, all_predictions, average="macro")
        acc = np.mean(np.array(all_predictions) == np.array(all_labels))

        print("val acc:", acc)
        print("val f1:", val_f1)

        print(classification_report(
            all_labels,
            all_predictions,
            target_names=train_data.categories
        ))

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Metrics/accuracy", acc, epoch)
        writer.add_scalar("Metrics/F1", val_f1, epoch)

        torch.save(model.state_dict(), "last_model.pt")

        if val_f1 > best_f1:

            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
