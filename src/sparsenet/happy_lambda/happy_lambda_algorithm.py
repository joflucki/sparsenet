from typing import Callable, List
from abc import ABC, abstractmethod
import torch

__all__ = ["HappyLambdaAlgorithm"]


class HappyLambdaAlgorithm(ABC):
    """
    Abstract base class for optimal penalty weight computation, a.k.a the `Happy` lambda computation.

    Attributes:
        dataset (torch.Tensor): The dataset used for computation.
        distribution_count (int): The number of random distributions to draw.
        layer_count (int): The number of layers in the neural network.
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
        distribution_count,
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
            std (float, optional): The standard deviation of the Gaussian distributions. Defaults to 1.
            mean (float, optional): The mean of the Gaussian distributions. Defaults to 0.
            device (torch.device, optional): The device where computation will be performed. Defaults to "cuda" if available, else "cpu".
            generator (torch.Generator | None, optional): A generator for random number generation. Defaults to None.
        """
        # Send dataset to specified device
        if dataset.device != device:
            dataset = dataset.to(device)

        # Init instance members
        self._distribution_count = distribution_count
        self._dataset = dataset
        self._std = std
        self._mean = mean
        self._device = device
        self._generator = torch.Generator(device) if generator is None else generator

        if device != self._generator.device:
            raise ValueError("dataset and generator must live on the same device")

    @abstractmethod
    def run(self) -> float:
        """
        Computes the optimal lambda (Happy lambda).

        Returns:
            float: The computed optimal lambda.
        """
        pass

    @property
    def dataset(self) -> torch.Tensor:
        """The dataset."""
        return self._dataset

    @property
    def distribution_count(self) -> int:
        """The number of distributions to draw during computation."""
        return self._distribution_count

    @property
    def std(self) -> float:
        """The standard deviation for Gaussian distributions."""
        return self._std

    @property
    def mean(self) -> float:
        """The mean value for Gaussian distributions."""
        return self._mean

    @property
    def device(self) -> torch.device:
        """The device on which to compute the optimal lambda value."""
        return self._device

    @property
    def generator(self) -> torch.Generator:
        """The random number generator for reproducibility."""
        return self._generator
