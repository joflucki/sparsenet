import torch
from tqdm import tqdm
import sparsenet
import matplotlib.pyplot as plt
import os


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(501, 20)
        self.fc2 = sparsenet.nn.NormalizedLinear(20, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = sparsenet.nn.sigma(self.fc1(input))
        b = self.fc2(a)
        return b

    def param_groups(self):
        return [
            {  # This is used to indicate which parameters are used in the penalty.
                "params": [self.fc1.weight, self.fc1.bias],
                "penalized": True,
            },
            {  # This is used to indicate which parameters are used in the penalty.
                "params": [self.fc2.weight, self.fc2.bias],
                "penalized": False,
            },
        ]


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset, weights, biases, target = sparsenet.nn.Simulation.load(
        "data/simulations/Mark", format="csv"
    )

    # Neural net
    model = Net()
    model.to(device)

    # HappyLambda
    happy_lambda = sparsenet.happy_lambda.HappyLambdaMLP(
        dataset=dataset,
        distribution_count=1_000,
        neuron_count=[20, 1],
    ).run()

    # Optimizer
    optimizer = sparsenet.nn.ISTA(model.param_groups(), happy_lambda)  # For ISTA
    # optimizer = torch.optim.Adam(model.parameters())  # For PyTorch optimizers

    # Loss
    criterion = sparsenet.nn.PenalizedL2Loss(
        model.param_groups()[0]["params"], happy_lambda
    )

    # Helper function to reevaluate model loss
    def closure() -> float:
        with torch.no_grad():
            output = model.forward(dataset)
            loss, penalty = criterion.forward(output, target)
            return loss.item() + penalty.item()

    # Training loop
    losses = []
    weights = []
    for _ in tqdm(
        range(500), f"Optimizing loss function using {type(optimizer).__name__}"
    ):
        # Save weights for later plots
        weights.append(model.fc1.weight.data.clone().detach())

        # Forward pass, compute model output
        output = model.forward(dataset)

        # Compute loss and penalty
        loss, penalty = criterion.forward(output, target)
        full_loss = loss + penalty

        # Record loss for plots
        losses.append(full_loss.item())

        # Backward pass, compute gradients
        loss.backward() # For ISTA
        # full_loss.backward()  # For PyTorch optimizers

        # Optimize
        optimizer.step(closure, full_loss.item())  # For ISTA
        # optimizer.step()  # For PyTorch optimizers

        # Reset gradients
        optimizer.zero_grad()

    # ---
    # Plots & results
    # ---

    print("Final loss:", losses[-1])
    print("MSE on prediction:", ((output - target) ** 2).mean().item())  # type: ignore
    plt.plot(losses)
    plt.title(f"Minimizing loss of MLP using {type(optimizer).__name__}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"plot-loss-{type(optimizer).__name__}-MLP.png")
    plt.clf()
    print(f"Saved loss plot in plot-loss-{type(optimizer).__name__}-MLP.png")

    # Convert tensors to numpy arrays
    output_np = output.detach().cpu().numpy()  # type: ignore
    target_np = target.detach().cpu().numpy()

    # Create a scatter plot
    plt.scatter(output_np, target_np, color="blue")

    # Add titles and labels
    plt.title(f"Prediction vs ground truth, {type(optimizer).__name__}")
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")

    # Optionally, add a line y=x for reference
    plt.plot(
        [target_np.min(), target_np.max()],
        [target_np.min(), target_np.max()],
        color="red",
        linestyle="--",
    )
    plt.savefig(f"plot-results-{type(optimizer).__name__}-MLP.png")
    plt.clf()
    print(f"Saved predictions plot in plot-results-{type(optimizer).__name__}-MLP.png")

    exit()
    if not os.path.isdir("weights_plots"):
        os.makedirs("weights_plots")

    weights = torch.stack(weights)
    if losses:
        for i in tqdm(range(weights.size(2)), "Generating weight plots"):
            [
                plt.plot(
                    list(range(len(losses))),
                    weights[:, j, i].cpu().numpy(),
                    label="Neuron #{}".format(j),
                )
                for j in range(weights.size(1))
            ]
            plt.title(
                "Weights of {}th feature in each neuron during training with {} ".format(
                    i, type(optimizer).__name__
                )
            )
            plt.xlabel("Epochs")
            plt.ylabel("Weight")
            plt.legend(prop={"size": 6})
            plt.savefig(
                "weights_plots/{}th_feature_weights_{}.png".format(
                    i, type(optimizer).__name__
                )
            )
            plt.clf()
    print(f"Saved weights plots in weights_plots/")


if __name__ == "__main__":
    main()
