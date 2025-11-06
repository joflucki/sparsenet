from typing import Callable, List
import torch
from ..nn import sigma
from . import HappyLambdaLinear
import math


__all__ = ["HappyLambdaMLP"]


class HappyLambdaMLP(HappyLambdaLinear):
    """
    An object used to compute the optimal penalty weight, a.k.a the `Happy` lambda.

    Attributes:
        dataset (torch.Tensor): The dataset used for computation.
        distribution_count (int): The number of random distributions to draw.
        neuron_count (List[int]): The number of neurons in each layer of the neural network.
        activation_fn (Callable): The activation function used in the neural network.
        std (float): The standard deviation of the Gaussian distributions.
        mean (float): The mean of the Gaussian distributions.
        device (torch.device): The device where computation will be performed.
        generator (torch.Generator): The random number generator for reproducibility.
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        distribution_count: int,
        neuron_count: List[int],
        activation_fn: Callable = sigma,
        std: float = 1,
        mean: float = 0,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        generator: torch.Generator | None = None,
    ):
        """
        Initializes the HappyLambda instance with a given dataset, distribution count, standard deviation, mean, device, and seed.

        Args:
            dataset (torch.Tensor): The dataset used for computation.
            distribution_count (int): The number of random distributions to draw.
            neuron_count (List[int]): The number of neurons in each layer of the neural network.
            activation_fn (Callable): The activation function used in the neural network.
            std (float, optional): The standard deviation of the Gaussian distributions. Defaults to 1.
            mean (float, optional): The mean of the Gaussian distributions. Defaults to 0.
            device (torch.device, optional): The device where computation will be performed. Defaults to "cuda" if available, else "cpu".
            generator (torch.Generator | None, optional): A generator for random number generation. Defaults to None.
        """
        super().__init__(
            dataset,
            distribution_count,
            std,
            mean,
            device,
            generator,
        )
        self._distribution_count = distribution_count
        self._layer_count = len(neuron_count)
        self._neuron_count = neuron_count
        self._activation_fn = activation_fn

    def run(self) -> float:
        """
        Computes the optimal lambda (Happy lambda).

        Returns:
            float: The computed optimal lambda.
        """

        # Get initial happy lambda value

        happy_lambda = super().run()

        # Compute weight constant C
        pi = math.sqrt(math.prod(self._neuron_count[2:]))
        activation_input = torch.tensor(0.0, requires_grad=True)
        activation_value = self._activation_fn(activation_input)
        activation_value.backward()

        if activation_input.grad is not None:
            c = (
                pi * pow(activation_input.grad.item(), self._layer_count - 1)
                if self.layer_count > 2
                else 1
            )
            return happy_lambda * c
        else:
            raise TypeError("Gradient computation failed for activation input.")

    @property
    def layer_count(self) -> int:
        """The number of layers in the neural network."""
        return self._layer_count

    @property
    def neuron_count(self) -> List[int]:
        """The number of neurons in each layer of the neural network."""
        return self._neuron_count

    @property
    def activation_fn(self) -> Callable:
        """The activation function used in the neural network."""
        return self._activation_fn
