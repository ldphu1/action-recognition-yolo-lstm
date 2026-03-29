import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=num_classes)
            )

    def forward(self, x):
        out,_ = self.lstm(x)
        w = torch.softmax(self.attention(out), dim=1)

        final_memory = (w * out).sum(dim=1)
        return self.fc(final_memory)
