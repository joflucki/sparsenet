import unittest
import torch
from sparsenet.happy_lambda import HappyLambdaMLP


class TestHappyLambdaPyTorch(unittest.TestCase):
    def setUp(self):
        self.mean = 0
        self.std = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = torch.normal(
            mean=0,
            std=1,
            size=[30, 10],
        )
        self.layer_count = 3
        self.neuron_counts = [20, 10, 1]
        self.distributions_counts = [1, 50, 100, 250, 500, 1000]
        self.seeds = [i**5 for i in range(100)]

        self.lambdas = torch.tensor(
            [
                [self.__generate_lambda(seed, dc) for seed in self.seeds]
                for dc in self.distributions_counts
            ]
        )

    def test_precision_increases(self):
        for i in range(len(self.distributions_counts) - 1):
            a = torch.std(self.lambdas[i], dim=0).item()
            b = torch.std(self.lambdas[i + 1], dim=0).item()
            self.assertLess(b, a)

    def __generate_lambda(self, seed: int, distribution_count: int) -> float:
        return HappyLambdaMLP(
            self.dataset,
            distribution_count,
            self.neuron_counts,
            device=self.device,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).run()
