import torch
import torch.nn as nn
import torch.nn.functional as F

class StateNet(nn.Module):
    """
    give current state and action predicts future state
    """
    def __init__(self, obs_dim, hidden_dim = 128):
        super(StateNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim * 2 + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, obs_dim * 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x