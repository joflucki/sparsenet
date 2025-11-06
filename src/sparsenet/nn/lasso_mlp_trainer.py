from typing import List
import torch
from . import ISTA, PenalizedL2Loss


class LassoMLPTrainer:
    """
    Helper class to run a training session using the Lasso pipeline on an MLP, using ISTA and PenalizedL2Loss.
    It is recommended to write you own pipeline, as shown in the examples, for a better control over the training.

    When running a Lasso training session, the training will happen in multiple "stages".
    Each stage is assigned a penalty weight (lambda) and a treshold. When the variation of the training loss
    is smaller than the assigned threshold, the next stage is launched.


    Attributes:
        dataset (torch.Tensor): The input dataset. X.
        target (torch.Tensor): The target values. Y.
        model (torch.nn.Module): The model to be trained.
        lambdas (List[float]): The list of lambda values for regularization.
        thresholds (List[float]): The list of thresholds for stage transitionning.
        optimizer (torch.optim.Optimizer): The optimizer for the training session.
        criterion (torch.nn.Module): The loss function with L2 penalty.
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        lambdas: List[float] | float,
        thresholds: List[float] | float,
    ):
        """
        Initializes the LassoMLPTrainer instance.

        Args:
            dataset (torch.Tensor): The input dataset.
            target (torch.Tensor): The target values.
            model (torch.nn.Module): The model to be trained.
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
        self.model = model
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
            self.optimizer = ISTA(self.model.param_groups(), lambda_val)
            self.criterion = PenalizedL2Loss(
                [
                    p
                    for g in list(
                        filter(lambda g: g["penalized"], self.model.param_groups())
                    )
                    for p in g["params"]
                ],
                lambda_val,
            )

            def closure() -> float:
                """
                Closure function for evaluating the model and returning the loss.

                Returns:
                    float: The computed loss value.
                """
                with torch.no_grad():
                    output = self.model.forward(self.dataset)
                    loss, penalty = self.criterion.forward(output, self.target)
                    return loss.item() + penalty.item()

            # Training loop
            while len(losses) < 2 or abs(losses[-2] - losses[-1]) > threshold:
                # Forward pass, compute model output
                output = self.model.forward(self.dataset)

                # Compute loss and penalty
                l2_loss, l1_penalty = self.criterion.forward(output, self.target)
                loss = l2_loss + l1_penalty

                # Record loss for plots
                losses.append(loss.item())

                # Backward pass, compute gradients
                l2_loss.backward()

                # Optimize
                self.optimizer.step(closure, loss.item())  # type: ignore

                # Reset gradients
                self.optimizer.zero_grad()

        return losses
