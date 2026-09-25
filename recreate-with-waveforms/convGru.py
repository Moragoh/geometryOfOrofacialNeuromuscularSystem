"""Conv+GRU baseline on raw (22, 7500) EMG, windowed on the same 46-step grid as the SPD-RNN."""


import torch
import torch.nn as nn


class convGruNet(nn.Module):
    def __init__(self, classifications, numberChannels = 22, numberFilters = 40, kernelSize = 25, hiddenSize = 64):
        super(convGruNet, self).__init__()
        self.spatialConv = nn.Conv1d(numberChannels, numberFilters, kernel_size = 1, bias = False)
        self.temporalConv = nn.Conv1d(numberFilters, numberFilters, kernel_size = kernelSize, groups = numberFilters, padding = kernelSize // 2, bias = False)
        # Windows [i*150, i*150 + 750) for i = 0..45, identical to the SPD-RNN notebooks' covariance windows.
        self.windowPool = nn.AvgPool1d(kernel_size = 750, stride = 150)
        self.gru = nn.GRU(numberFilters, hiddenSize, num_layers = 1, batch_first = True, bidirectional = True)
        self.hiddenSize = hiddenSize
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hiddenSize, classifications)

    def forward(self, x):
        x = self.temporalConv(self.spatialConv(x))
        x = self.windowPool(x ** 2)
        x = torch.log(torch.clamp(x, min = 1e-6))
        x = x.transpose(1, 2)
        gruOutput, _ = self.gru(x)
        # Mean over time per direction, then sum directions (as in manifoldRnn/spdRnn.py rnnNet.forward).
        forwardMean = gruOutput[:, :, :self.hiddenSize].mean(dim = 1)
        backwardMean = gruOutput[:, :, self.hiddenSize:].mean(dim = 1)
        return self.linear(self.dropout(forwardMean + backwardMean))
