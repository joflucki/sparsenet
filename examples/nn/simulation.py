import torch
from sparsenet.nn import Simulation, sigma


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(192)
    simulation = Simulation(
        sample_count=10_000,
        feature_count=501,
        neuron_counts=[20, 1],
        activation_fn=sigma,
        generator=generator,
    )
    simulation.run()
    simulation.save("./data/simulations/Mark")

    dataset, weights, biases, target = Simulation.load("./data/simulations/Mark")

    assert torch.allclose(simulation.dataset, dataset)
    assert all([torch.allclose(sw, w) for sw, w in zip(simulation.weights, weights)])
    assert all([torch.allclose(sb, b) for sb, b in zip(simulation.biases, biases)])
    assert torch.allclose(simulation.target, target)


if __name__ == "__main__":
    main()
