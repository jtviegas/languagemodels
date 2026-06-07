"""Activation functions for transformer neural network architectures."""

import torch
from torch import nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function.

    A smooth, non-linear activation that weights inputs by their magnitude.
    Unlike ReLU, GELU allows small negative values through probabilistically,
    providing smoother gradients. Commonly used in transformer architectures.
    """

    def __init__(self) -> None:
        """Initialize the GELU activation module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with GELU activation applied element-wise.
        """
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
