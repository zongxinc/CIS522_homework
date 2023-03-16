import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize parameters of ConvNet

        Args:
            num_channels (int): Number of channels in the input image
            num_classes (int): Number of classes in the dataset

        Returns:
            None
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ConvNet

        Args:
          x: torch.tensor
            Input data

        Returns:
          x: torch.tensor
            Output after passing x through Conv2d layer
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x
