import torch
from sparsenet.linear import Simulation


def main():
    run_and_save()
    load_and_print()


def run_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(100)
    simulation = Simulation(
        sample_count=1_000,
        feature_count=500,
        generator=generator,
        device=device,
    )
    simulation.run()
    simulation.save("data/simulations/Steve", format="csv")


def load_and_print():
    device = torch.device("cuda")
    dataset, weights, error, target = Simulation.load(
        "data/simulations/Steve", format="csv"
    )
    print("Dataset shape:", dataset.shape)
    print(
        "Sparsity percentage: {:.2%}".format(
            (weights == 0).sum().item() / weights.size(0)
        )
    )
    print("Mean error:", error.mean().item())
    print("Target shape:", target.shape)


if __name__ == "__main__":
    main()
