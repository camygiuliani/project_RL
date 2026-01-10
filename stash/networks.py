import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNCNN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # Input: (B, C, 84, 84)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute conv output size: 84 -> 20 -> 9 -> 7  (classic)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: float in [0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values
