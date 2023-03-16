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
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
