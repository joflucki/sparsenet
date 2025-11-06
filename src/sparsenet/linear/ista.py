import torch


class ISTA(torch.optim.Optimizer):
    """
    Implements the Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse optimization.

    This optimizer performs iterative shrinkage-thresholding.
    The optimizer requires a closure function to reevaluate the model's loss during the optimization step.
    The optimizer also assumes your parameters include an intercept at the 0th index of the tensor.

    Attributes:
        param_groups (list): A list of parameter groups, each containing parameters to optimize and associated options.
    """

    def __init__(
        self,
        params,
        happy_lambda: float,
    ):
        """
        Initializes the ISTA optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
            happy_lambda (float): Regularization parameter.
        """
        defaults = dict(happy_lambda=happy_lambda)
        super(ISTA, self).__init__(params, defaults)

    def step(self, closure, initial_loss):  # type: ignore
        """
        Performs a single optimization step.

        The optimizer will call the loss and gradient functions and compute the updates internally,
        so the developer does not need to perform forward/backward passes manually.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
            initial_loss (float, optional): The initial loss value before the step.
        """
        if initial_loss is None:
            initial_loss = closure()

        for group in self.param_groups:
            hl: float = group["happy_lambda"]

            for p in group["params"]:
                if not isinstance(p, torch.Tensor):
                    continue

                if p.grad is None:
                    continue

                initial = p.clone().detach()

                t = 1
                b = initial.data - t * p.grad
                p.data[0] = b[0]
                p.data[1:] = torch.sign(b[1:]) * torch.max(
                    torch.abs(b[1:]) - hl * t, torch.tensor(0.0)
                )

                while initial_loss <= closure() and not torch.allclose(initial, p):
                    t /= 2
                    b = initial.data - t * p.grad
                    p.data[0] = b[0]
                    p.data[1:] = torch.sign(b[1:]) * torch.max(
                        torch.abs(b[1:]) - hl * t, torch.tensor(0.0)
                    )
