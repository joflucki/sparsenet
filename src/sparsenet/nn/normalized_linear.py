import torch
import torch.nn as nn


class NormalizedLinear(nn.Module):
    """
    A custom linear layer with weight normalization.

    This layer performs a linear transformation with normalized weights,
    which can help stabilize training by ensuring the weights have a consistent scale.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (torch.nn.Parameter): The learnable weights of the module.
        bias (torch.nn.Parameter or None): The learnable bias of the module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initializes the NormalizedLinear module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
            device (torch.device, optional): The device on which to place the parameters.
            dtype (torch.dtype, optional): The data type of the parameters.
        """
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weights and biases of the module.

        Weights are initialized using a normal distribution with mean 0 and standard deviation 1.
        Biases are initialized similarly.
        """
        nn.init.normal_(self.weight, mean=0.0, std=0.1)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the normalized linear transformation.
        """
        normalized_weight = self.weight / torch.norm(self.weight)
        if self.bias is None:
            return torch.matmul(input, normalized_weight.t())
        else:
            return torch.matmul(input, normalized_weight.t()) + self.bias
