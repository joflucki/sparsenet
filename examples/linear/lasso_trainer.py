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

    # Compute lambda
    happy_lambda = sparsenet.happy_lambda.HappyLambdaLinear(
        dataset,
        distribution_count=1000,
        device=device,
    ).run()

    # Create a trainer with specified params
    trainer = sparsenet.linear.LassoLinearTrainer(
        dataset,
        target,
        alpha,
        [happy_lambda],
        [0.001],
    )

    # Run the training session
    losses = trainer.run()

    # Compute an output with the trained weights
    output = torch.matmul(dataset, trainer.weights)

    # ---
    # Plots & results
    # ---
    print("Final loss:", losses[-1])
    print("MSE on weights:", ((weights - alpha) ** 2).mean().item())
    print("Weights close to target:", torch.allclose(weights, alpha))
    print("MSE on prediction:", ((output - target) ** 2).mean().item())  # type: ignore

    # Save weights in CSV file
    with open(f"alpha-lasso-trainer.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(alpha.detach().cpu().numpy())
    print(f"Saved weight data in alpha-lasso-trainer.csv")

    # Plot loss evolution
    plt.plot(losses)
    plt.title(f"Minimizing loss using LassoLinearTrainer")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"graph-lasso-trainer.png")
    print(f"Saved loss plot in graph-lasso-trainer.png")
    plt.clf()

    # Convert tensors to numpy arrays
    output_np = output.detach().cpu().numpy()  # type: ignore
    target_np = target.detach().cpu().numpy()

    # Plot prediction accuracy
    plt.scatter(output_np, target_np, color="blue")
    plt.title(f"Prediction vs ground truth, LassoLinearTrainer")
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")

    # Optionally, add a line y=x for reference
    plt.plot(
        [target_np.min(), target_np.max()],
        [target_np.min(), target_np.max()],
        color="red",
        linestyle="--",
    )
    plt.savefig(f"graph-results-lasso-trainer.png")
    print(f"Saved prediction plot in graph-results-lasso-trainer.png")


if __name__ == "__main__":
    main()
