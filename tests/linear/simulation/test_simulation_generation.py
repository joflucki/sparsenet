import unittest

from sparsenet.linear import Simulation
import torch


class TestSimulationGeneration(unittest.TestCase):
    def setUp(self):
        self.sample_count = 30
        self.feature_count = 10
        self.seed = 123
        self.generator = torch.Generator().manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.error_rate = 0.01
        self.non_zero_first = False

        self.simulation = Simulation(
            self.sample_count,
            self.feature_count,
            self.generator,
            self.device,
            self.error_rate,
            self.non_zero_first,
        )

        self.simulation.run()

    def test_dataset_size(self):
        self.assertEqual(
            self.sample_count * (self.feature_count + 1),
            self.simulation.dataset.shape.numel(),
        )

    def test_dataset_shape(self):
        self.assertEqual(
            (self.sample_count, self.feature_count + 1),
            tuple(self.simulation.dataset.shape),
        )

    def test_weights_size(self):
        self.assertEqual(
            self.feature_count + 1,
            self.simulation.weights.shape.numel(),
        )

    def test_weights_shape(self):
        self.assertEqual(
            (self.feature_count + 1, 1),
            tuple(self.simulation.weights.shape),
        )

    def test_error_size(self):
        self.assertEqual(
            self.sample_count,
            self.simulation.error.shape.numel(),
        )

    def test_error_shape(self):
        self.assertEqual(
            (self.sample_count, 1),
            tuple(self.simulation.error.shape),
        )

    def test_target_size(self):
        self.assertEqual(
            self.sample_count,
            self.simulation.target.shape.numel(),
        )

    def test_target_shape(self):
        self.assertEqual(
            (self.sample_count, 1),
            tuple(self.simulation.target.shape),
        )

    def test_results(self):
        target = torch.add(
            torch.matmul(
                self.simulation.dataset,
                self.simulation.weights,
            ),
            self.simulation.error,
        )
        self.assertTrue(torch.allclose(target, self.simulation.target))
