from typing import Callable
import torch


class GD(torch.optim.Optimizer):
    """
    Implements Gradient Descent (GD) optimization.

    This optimizer performs gradient descent steps using a given loss function.
    The developer does not need to perform forward/backward passes manually, as the optimizer
    will call the provided loss function and compute the gradients internally.

    Attributes:
        param_groups (list): A list of parameter groups, each containing parameters to optimize and associated options.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
    ):
        """
        Initializes the GD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
            lr (float): Learning rate.
        """
        defaults = dict(lr=lr)
        super(GD, self).__init__(params, defaults)

    def step(self, closure: Callable[[], float], initial_loss: float | None) -> float:  # type: ignore
        """
        Performs a single optimization step.

        The optimizer will call the loss function and compute the gradients internally,
        so the developer does not need to perform forward/backward passes manually.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
            initial_loss (float, optional): The initial loss value before the step.
        """
        if initial_loss is None:
            initial_loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            for p in group["params"]:
                if not isinstance(p, torch.Tensor):
                    continue

                if p.grad is None:
                    continue

                initial = p.clone().detach()

                t = 1
                p.data = initial - t * lr * initial

                while initial_loss < closure():
                    t /= 2
                    p.data = initial - t * lr * initial
