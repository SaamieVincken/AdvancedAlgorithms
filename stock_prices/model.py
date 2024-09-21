import torch
from torch import nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
num_layers=2)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = torch.relu(out[:, -1, :])  # Take the last output of the LSTM
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
