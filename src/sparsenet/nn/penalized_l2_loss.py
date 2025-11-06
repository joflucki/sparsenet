from typing import List, Tuple
import torch
import torch.nn as nn


class PenalizedL2Loss(nn.Module):
    """
    MSE loss function with an added L1 penalty for sparse models.

    This loss function combines the L2 loss (mean squared error) with an L1 penalty
    on the model parameters to promote sparsity.

    Attributes:
        params (List[torch.Tensor]): Parameters of the model to apply the L1 penalty on.
        happy_lambda (float): Regularization parameter for the L1 penalty.
    """

    def __init__(self, params: List[torch.Tensor], happy_lambda: float):
        """
        Initializes the L1PenaltyLoss module.

        Args:
            params (List[torch.Tensor]): Parameters of the model to apply the L1 penalty on.
            happy_lambda (float): Regularization parameter for the L1 penalty.
        """
        assert params
        super(PenalizedL2Loss, self).__init__()
        self.__happy_lambda = happy_lambda
        self.__params = params

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the loss function.

        Args:
            output (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The true target values.

        Returns:
            torch.Tensor: The combined L2 loss and L1 penalty.
        """
        # Loss
        loss = torch.norm(output - target)

        # Penalty
        penalty = self.happy_lambda * torch.linalg.vector_norm(
            torch.cat([tensor.reshape([torch.numel(tensor)]) for tensor in self.__params]),
            ord=1,
        )

        # Combine both losses
        return loss, penalty

    @property
    def happy_lambda(self) -> float:
        """
        Returns the regularization parameter for the L1 penalty.

        Returns:
            float: The regularization parameter.
        """
        return self.__happy_lambda

    @property
    def params(self) -> List[torch.Tensor]:
        """
        Returns the parameters of the model to apply the L1 penalty on.

        Returns:
            List[torch.Tensor]: The model parameters.
        """
        return self.__params
