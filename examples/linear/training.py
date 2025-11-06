import csv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sparsenet


def main():
    # Set device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset, weights, _, target = sparsenet.linear.Simulation.load(
        "data/simulations/Steve",
        format="csv",
        device=device,
    )

    # Define training weights
    alpha = torch.full(
        [dataset.size(1), 1],
        fill_value=-25.0,
        device=device,
        requires_grad=True,
    )

    # Create a simple linear model. Usually, this model is a neural network.
    def model(inputs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(inputs, alpha)

    # Compute lambda
    happy_lambda = sparsenet.happy_lambda.HappyLambdaLinear(
        dataset,
        distribution_count=1000,
        device=device,
    ).run()

    # Define loss
    criterion = sparsenet.nn.PenalizedL2Loss([alpha[1:]], happy_lambda)

    # Define optimizer
    optimizer = sparsenet.linear.ISTA([alpha], happy_lambda)  # For ISTA
    # optimizer = torch.optim.SGD([alpha]) # For PyTorch optimizers

    def closure():
        with torch.no_grad():
            output = torch.matmul(dataset, alpha)
            loss, penalty = criterion(output, target)
            return (loss + penalty).item()

    losses = []
    for _ in tqdm(
        range(500), f"Optimizing loss function using {type(optimizer).__name__}"
    ):
        # Clear pre-existing gradients
        optimizer.zero_grad()

        # Compute output
        output = model(dataset)

        # Compute loss
        loss, penalty = criterion(output, target)
        complete_loss = loss + penalty

        # Compute gradients
        loss.backward()  # For ISTA
        # complete_loss.backward()  # For PyTorch optimizers

        # Perform optimization step
        optimizer.step(closure, (loss + penalty).item())  # For ISTA
        # optimizer.step() # For PyTorch optimizers

        losses.append((loss + penalty).item())

    # ---
    # Plots & results
    # ---
    print("Final loss:", losses[-1])
    print("MSE on weights:", ((weights - alpha) ** 2).mean().item())
    print("Weights close to target:", torch.allclose(weights, alpha))
    print("MSE on prediction:", ((output - target) ** 2).mean().item())  # type: ignore


    # Save weights in CSV file
    with open(f"alpha-{type(optimizer).__name__}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(alpha.detach().cpu().numpy())
    print(f"Saved weight data in alpha-{type(optimizer).__name__}.csv")

    # Plot loss evolution
    plt.plot(losses)
    plt.title(f"Minimizing loss using {type(optimizer).__name__}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"graph-{type(optimizer).__name__}.png")
    print(f"Saved loss plot in graph-{type(optimizer).__name__}.png")
    plt.clf()

    # Convert tensors to numpy arrays
    output_np = output.detach().cpu().numpy()  # type: ignore
    target_np = target.detach().cpu().numpy()

    # Plot prediction accuracy
    plt.scatter(output_np, target_np, color="blue")
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
    plt.savefig(f"graph-results-{type(optimizer).__name__}.png")
    print(f"Saved prediction plot in graph-results-{type(optimizer).__name__}.png")


if __name__ == "__main__":
    main()
