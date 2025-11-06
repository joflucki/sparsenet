import torch
from csv import reader
import csv
from sparsenet.happy_lambda import HappyLambdaLinear


def main():
    seed = 1_123_581_321  # Use a seed for determinism in random numbers
    distribution_count = 10_000  # The number of random distribution to draw

    # Set device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generator for reproduceability
    generator = torch.Generator(device).manual_seed(seed)

    # Load dataset
    with open("data/unige/X.csv", "r") as file:
        csvreader = reader(file, quoting=csv.QUOTE_NONNUMERIC)
        dataset = list(csvreader)[1:]
        dataset = torch.tensor(dataset)

    # Load dataset
    with open("data/unige/happylambda.csv", "r") as file:
        csvreader = reader(file, quoting=csv.QUOTE_NONNUMERIC)
        true_happylambda = list(csvreader)[1:]
        true_happylambda = torch.tensor(true_happylambda)

    # Compute lambda
    lambda_calculator = HappyLambdaLinear(
        dataset,
        distribution_count,
        generator=generator,
    )
    print("Computing Happy Lambda...")
    lambda_value = lambda_calculator.run()
    print("Computed HappyLambda:", lambda_value)
    print("True HappyLambda:", true_happylambda.item())


if __name__ == "__main__":
    main()
