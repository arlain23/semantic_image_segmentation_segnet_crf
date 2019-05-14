import torch
import torch.nn.functional as F
import torch.nn as nn


class CNN(nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Input channels = 3, output channels = 18
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Sequential(
            torch.nn.Linear(32 * 8 * 8, 100),
            nn.ReLU())

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(100, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def output_size(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1

        return output