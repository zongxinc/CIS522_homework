from typing import Callable
import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation()
        self.initializer = initializer

        self.layers = torch.nn.ModuleList()
        for i in range(self.hidden_count):
            self.layers.append(torch.nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.layers.apply(self._init_weights)

        self.out = torch.nn.Linear(hidden_size, num_classes)

    def _init_weights(self, m):
        """weight initialization.

        Arguments:
            x: layer

        Returns:
        """
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            self.initializer(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = self.activation(layer(x))

        out = self.out(x)
        return out