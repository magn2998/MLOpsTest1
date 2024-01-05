import torch
from torch import nn


class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc15 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 10)

        self.act1 = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout(x)
        x = self.act1(self.fc15(x))
        x = self.dropout(x)
        x = self.act1(self.fc2(x))
        x = self.dropout(x)
        x = self.act1(self.fc3(x))
        x = self.dropout(x)
        return self.softmax(self.output(x))


class MyCnnNetwork(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MyCnnNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(7 * 7 * 16, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,10), 
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = x.view(x.size(0), 28, 28)
        out = out.unsqueeze(1)
        out = self.cnn1(out)
        cnn_out = self.cnn2(out)
        out = cnn_out.reshape(cnn_out.size(0), -1)
        out = self.fc(out)
        return out, cnn_out
