import unittest

from sparsenet.linear import Simulation
import torch


class TestSimulationAttributes(unittest.TestCase):
    def setUp(self):
        self.sample_count = 300
        self.feature_count = 100
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

    def test_sample_count_getter(self):
        self.assertEqual(self.sample_count, self.simulation.sample_count)

    def test_feature_count_getter(self):
        self.assertEqual(self.feature_count, self.simulation.feature_count)

    def test_generator_getter(self):
        self.assertEqual(self.generator, self.simulation.generator)

    def test_device_getter(self):
        self.assertEqual(self.device, self.simulation.device)

    def test_error_rate_getter(self):
        self.assertEqual(self.error_rate, self.simulation.error_rate)

    def test_non_zero_first_getter(self):
        self.assertEqual(self.non_zero_first, self.simulation.non_zero_first)
