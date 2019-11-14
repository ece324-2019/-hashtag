import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, in_size):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(in_size, 100);
        self.fc2 = nn.Linear(100, 1);

    def forward(self, x, lengths=None):
        x=self.fc1(x);
        x=self.fc2(x);

	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return x