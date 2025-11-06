from typing import Callable
import torch


class ISTA(torch.optim.Optimizer):
    """
    Implements the Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse optimization.

    If you use this optimizer in combination with the PenalizedL2Loss, it is important to call the backward method
    on the L2 Loss only, and not on the complete loss or L1 penalty. You can find an example usage in the user guide.

    Attributes:
        param_groups (list): A list of parameter groups, each containing parameters to optimize and associated options.
    """

    def __init__(self, params, happy_lambda: float):
        """
        Initializes the ISTA optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
            happy_lambda (float): Regularization parameter.
        """
        defaults = dict(happy_lambda=happy_lambda)
        super(ISTA, self).__init__(params, defaults)

    def step(self, closure: Callable[[], float], initial_loss: float | None):  # type: ignore
        """
        Performs a single optimization step.

        Args:
            closure (Callable[[], float]): A closure that reevaluates the model and returns the loss.
            initial_loss (float): The initial loss value before the step.
        """
        if initial_loss is None:
            initial_loss = closure()

        for group in self.param_groups:
            hl = group["happy_lambda"]
            for p in group["params"]:
                if not isinstance(p, torch.Tensor):
                    continue

                if p.grad is None:
                    continue

                # Clone initial tensor
                initial = p.clone().detach()

                # Parameter groups MUST contain a "penalized" param, that is either True of False.
                if group["penalized"]:
                    t = 1
                    a = initial.data - t * p.grad

                    p.data = torch.sign(a) * torch.max(
                        torch.abs(a) - hl * t, torch.tensor(0.0)
                    )

                    while initial_loss <= closure() and not torch.allclose(
                        initial, p, rtol=0.0001
                    ):
                        t /= 2
                        a = initial.data - t * p.grad
                        p.data = torch.sign(a) * torch.max(
                            torch.abs(a) - hl * t, torch.tensor(0.0)
                        )

                else:
                    t = 1
                    p.data -= t * p.grad
                    while initial_loss <= closure() and not torch.allclose(
                        initial, p, rtol=0.0001
                    ):

                        t /= 2
                        p.data += t * p.grad
