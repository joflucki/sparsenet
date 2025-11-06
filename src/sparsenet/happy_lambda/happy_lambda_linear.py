import torch

from sparsenet.happy_lambda import HappyLambdaAlgorithm

__all__ = ["HappyLambdaLinear"]


class HappyLambdaLinear(HappyLambdaAlgorithm):
    """
    An object used to compute the optimal lambda, a.k.a the `Happy` lambda for linear models.

    Attributes:
        dataset (torch.Tensor): The dataset used for computation.
        distribution_count (int): The number of random distributions to draw.
        std (float): The standard deviation of the Gaussian distributions.
        mean (float): The mean of the Gaussian distributions.
        device (torch.device): The device where computation will be performed.
        generator (torch.Generator): The random number generator for reproducibility.
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        distribution_count: int,
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
            distribution_count (int, optional): The number of random distributions to draw. Defaults to 1000.
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

    def run(self) -> float:
        """
        Computes the optimal lambda (Happy lambda).

        Returns:
            float: The computed optimal lambda.
        """
        n = self._dataset.size(0)
        lambdas = []
        for _ in range(self._distribution_count):
            # Draw random numbers in gaussian distribution
            gaussian = torch.normal(
                self._mean,
                self._std,
                size=([n, 1]),
                device=self._device,
                generator=self._generator,
            )

            if n > 1:
                # Center the distribution around mean
                gaussian -= torch.mean(gaussian)

            # Compute new lambda
            lambda_value = HappyLambdaLinear.gaussian_to_lambda(self._dataset, gaussian)

            # Append to the list
            lambdas.append(lambda_value)

        # Sort lambdas
        lambdas = sorted(lambdas, key=lambda t: t.item())

        # Return the value on the 95th percentile's lower bound
        happy_lambda = float(lambdas[int(len(lambdas) * 0.95)])
        return happy_lambda

    @staticmethod
    def gaussian_to_lambda(dataset: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        Computes lambda for a given gaussian distribution and dataset.

        Args:
            dataset (torch.Tensor): The dataset used for computation.
            sample (torch.Tensor): The gaussian distribution for which to compute lambda.

        Returns:
            torch.Tensor: The computed lambda.
        """
        return torch.norm(
            torch.matmul(torch.transpose(dataset, 0, 1), sample), p=torch.inf
        ) / torch.norm(sample, p=2)
