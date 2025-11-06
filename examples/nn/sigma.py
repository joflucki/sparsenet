"""
This script is used to compare two gradients of the same activation function.

One is computed using PyTorch's autrograd, the other is computer analytically, and results are then plotted.
"""

from sparsenet.nn import sigma
import torch
import matplotlib.pyplot as plt


def analytical_sigma_gradient(input: torch.Tensor):
    M = 20
    k = 1
    u = 1
    return (torch.e ** (M * (input + u))) / (k * (torch.e ** (M * (input + 1)) + 1))


def main():

    generator = torch.Generator().manual_seed(42)
    input = torch.normal(-1, 1, size=[500], generator=generator, requires_grad=True)

    values = sigma(input)
    values.backward(torch.full([500], 1))

    gradients = analytical_sigma_gradient(input)

    x = input.detach().numpy()
    y = values.detach().numpy()

    print("Plotting Sigma activation function...")
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="blue", s=0.1)
    plt.title("2D Scatter Plot of sigma activation function")
    plt.xlabel("Autograd gradient values")
    plt.ylabel("Analytical gradient values")
    plt.grid(True)
    plt.savefig("activation_function.png")
    plt.clf()
    print("Saved plot in activation_function.png")

    if input.grad is not None:
        x = input.grad.detach().numpy()
        y = gradients.detach().numpy()

        print("Plotting analytical gradient vs automatic gradient...")
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color="blue", s=0.1)
        plt.title("2D Scatter Plot of analytical vs autograd gradients")
        plt.xlabel("Autograd gradient values")
        plt.ylabel("Analytical gradient values")
        plt.grid(True)
        plt.savefig("comparison_autograd_vs_analytical.png")
        plt.clf()
        print("Saved plot in comparison_autograd_vs_analytical.png")


if __name__ == "__main__":
    main()
