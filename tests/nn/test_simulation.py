import unittest
import torch
from sparsenet.nn import Simulation, sigma


class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.generator = torch.Generator(self.device).manual_seed(192)
        self.sample_count = 100
        self.feature_count = 20
        self.neuron_counts = [20, 10, 1]
        self.sim = Simulation(
            self.sample_count,
            self.feature_count,
            self.neuron_counts,
            sigma,
            self.generator,
            self.device,
        )
        self.sim.run()

    def test_dataset(self):
        self.assertEqual(self.sim.dataset.size(0), self.sample_count)
        self.assertEqual(self.sim.dataset.size(1), self.feature_count)

    def test_weights(self):
        self.assertEqual(len(self.sim.weights), len(self.neuron_counts))

        dimensions = [self.feature_count] + self.neuron_counts
        for i, weight in enumerate(self.sim.weights):
            self.assertEqual(weight.size(0), dimensions[i + 1])
            self.assertEqual(weight.size(1), dimensions[i])

    def test_biases(self):
        self.assertEqual(len(self.sim.biases), len(self.neuron_counts))

        for i, bias in enumerate(self.sim.biases):
            self.assertEqual(bias.size(0), self.neuron_counts[i])

    def test_target(self):
        self.assertEqual(self.sim.target.size(0), self.sample_count)
        self.assertEqual(self.sim.target.size(1), self.neuron_counts[-1])
