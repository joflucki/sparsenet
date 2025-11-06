from typing import List
import torch
from . import ISTA
import sparsenet


class LassoLinearTrainer:
    """
    Helper class to run a training session using the Lasso pipeline on a linear model, using ISTA and PenalizedL2Loss.
    It is recommended to write you own pipeline, as shown in the examples, for a better control over the training.

    When running a Lasso training session, the training will happen in multiple "stages".
    Each stage is assigned a penalty weight (lambda) and a treshold. When the variation of the training loss
    is smaller than the assigned threshold, the next stage is launched.


    Attributes:
        dataset (torch.Tensor): The input dataset. X.
        target (torch.Tensor): The target values. Y.
        weights (torch.nn.Module): The weights of the linear model to be trained.
        lambdas (List[float]): The list of lambda values for regularization.
        thresholds (List[float]): The list of thresholds for stage transitionning.
        optimizer (torch.optim.Optimizer): The optimizer for the training session.
        criterion (torch.nn.Module): The loss function with L2 penalty.
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        lambdas: List[float] | float,
        thresholds: List[float] | float,
    ):
        """
        Initializes the LassoMLPTrainer instance.

        Args:
            dataset (torch.Tensor): The input dataset.
            target (torch.Tensor): The target values.
            weights (torch.nn.Module): The weights of the linear model to be trained.
            lambdas (List[float] | float): The lambda value for regularization. Can be a single value or a list of values.
            thresholds (List[float] | float): The threshold for stage transitionning. Can be a single value or a list of values.
        """
        if not isinstance(lambdas, list):
            lambda_list = [lambdas]
        else:
            lambda_list = lambdas

        if not isinstance(thresholds, list):
            threshold_list = [thresholds]
        else:
            threshold_list = thresholds

        self.dataset = dataset
        self.target = target
        self.weights = weights
        self.thresholds = threshold_list
        self.lambdas = lambda_list

        if len(threshold_list) != len(lambda_list):
            raise ValueError(
                "There must be the same number of thresholds as the number of lambdas"
            )

    def run(self) -> List[float]:
        """
        Runs the training session.

        Returns:
            List[float]: A list of loss values recorded during training.
        """

        losses = []
        for lambda_val, threshold in zip(self.lambdas, self.thresholds):
            self.optimizer = ISTA([self.weights], lambda_val)
            self.criterion = sparsenet.nn.PenalizedL2Loss(
                [self.weights[1:]],
                lambda_val,
            )

            def closure() -> float:
                """
                Closure function for evaluating the model and returning the loss.

                Returns:
                    float: The computed loss value.
                """
                with torch.no_grad():
                    output = torch.matmul(self.dataset, self.weights)
                    loss, penalty = self.criterion(output, self.target)
                    return (loss + penalty).item()

            # Training loop
            while len(losses) < 2 or abs(losses[-2] - losses[-1]) > threshold:
                # Clear pre-existing gradients
                self.optimizer.zero_grad()

                # Compute output
                output = torch.matmul(self.dataset, self.weights)

                # Compute loss
                loss, penalty = self.criterion(output, self.target)

                # Compute gradients
                loss.backward()

                # Perform optimization step
                self.optimizer.step(closure, (loss + penalty).item())

                losses.append((loss + penalty).item())

        return losses
