import torch
import math

__all__ = ["Sigma", "sigma"]


class Sigma(torch.nn.Module):
    """
    Implementation of Sigma activation function.

    This custom activation function applies a non-linear transformation to the input tensor using
    a logarithmic function, providing a way to introduce non-linearity in neural networks.

    Methods:
        forward(output): Computes the forward pass of the activation function.
        f(output): Helper function to compute the logarithmic transformation.
    """

    def __init__(self):
        """
        Initializes the Sigma activation function module.
        """
        super(Sigma, self).__init__()

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the Sigma activation function.

        Args:
            output (torch.Tensor): The input tensor to the activation function.

        Returns:
            torch.Tensor: The transformed tensor after applying the Sigma activation function.
        """
        return sigma(output)


def sigma(output: torch.Tensor | float) -> torch.Tensor:
    """
    Computes the Sigma activation function.

    Args:
        output (torch.Tensor | float): The input tensor or scalar to transform.

    Returns:
        torch.Tensor: The transformed tensor after applying the Sigma activation function.
    """
    threshold = -0.5
    k = 1

    output_tensor = torch.as_tensor(output)

    return torch.where(
        output_tensor <= threshold,
        (1 / k) * (__f(output_tensor) ** k - __f(torch.tensor(0.0)) ** k),
        output_tensor,
    )


def __f(output: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute the logarithmic transformation.

    Args:
        output (torch.Tensor | float): The input tensor or scalar to transform.

    Returns:
        torch.Tensor: The transformed tensor after applying the logarithmic function.
    """
    M = 20
    u0 = 1

    # Uses the log-sum-exp trick for numerical stability
    # Stable computation of (1/M) * ln(1 + e^(M * (u0 + x)))
    z = M * (u0 + output)
    max_z = torch.clamp(z, min=0)  # Ensure z is non-negative for stability
    return (1 / M) * (max_z + torch.log(torch.exp(-max_z) + torch.exp(z - max_z)))
