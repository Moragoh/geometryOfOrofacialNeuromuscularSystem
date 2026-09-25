"""ResNet for time-series classification from Wang et al. 2017 (arXiv 1611.06455), applied to (22, 7500) EMG trials."""


import torch.nn as nn


class residualBlock(nn.Module):
    """Eq. 3 of the paper: h1 = Block(x), h2 = Block(h1), h3 = Block(h2), output = ReLU(h3 + x),
    where each Block (Eq. 2) is conv -> BN -> ReLU."""

    def __init__(self, inChannels, outChannels):
        super(residualBlock, self).__init__()
        # Stride 1 with "same" padding everywhere, so the time length (7500) never changes.
        self.convolutions = nn.Sequential(
            nn.Conv1d(inChannels, outChannels, kernel_size = 8, padding = "same"),
            nn.BatchNorm1d(outChannels), # Batchnorm: prevents internal covariate shift between layers. Basically prevents drift between layers
            nn.ReLU(),
            nn.Conv1d(outChannels, outChannels, kernel_size = 5, padding = "same"),
            nn.BatchNorm1d(outChannels),
            nn.ReLU(),
            nn.Conv1d(outChannels, outChannels, kernel_size = 3, padding = "same"),
            nn.BatchNorm1d(outChannels),
            nn.ReLU(),
        )
        # The paper adds x directly. When the channel count changes that is impossible, and the paper doesn't say
        # what to do, so a 1x1 convolution projects x to outChannels (the projection shortcut of He et al. 2016).
        if inChannels != outChannels:
            self.shortcut = nn.Conv1d(inChannels, outChannels, kernel_size = 1) 
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.convolutions(x) + self.shortcut(x)) # Where shortcut is added. 


class resnet1dNet(nn.Module):
    def __init__(self, numberClasses, numberChannels = 22):
        super(resnet1dNet, self).__init__()
        self.blocks = nn.Sequential(
            residualBlock(numberChannels, 64),
            residualBlock(64, 128),
            residualBlock(128, 128),
        )
        self.linear = nn.Linear(128, numberClasses)

    def forward(self, x):
        # x: (batch, 22 channels, 7500 samples) -> (batch, 128, 7500) -> mean over time -> (batch, 128) -> (batch, classes)
        features = self.blocks(x)
        return self.linear(features.mean(dim = -1))
