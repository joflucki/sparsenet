import unittest
from sparsenet.happy_lambda import HappyLambdaLinear
import torch


class TestHappyLambdaAttributes(unittest.TestCase):
    def setUp(self):
        self.mean = 0
        self.std = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(self.device)
        self.dataset = torch.normal(mean=0, std=1, size=[30, 10], device=self.device)
        self.distribution_count = 50

        self.hl = HappyLambdaLinear(
            self.dataset,
            self.distribution_count,
            self.std,
            self.mean,
            self.device,
            self.generator,
        )

    def test_dataset_getter(self):
        self.assertTrue(torch.equal(self.dataset, self.hl.dataset))

    def test_distribution_count_getter(self):
        self.assertEqual(self.distribution_count, self.hl.distribution_count)

    def test_std_getter(self):
        self.assertEqual(self.std, self.hl.std)

    def test_mean_getter(self):
        self.assertEqual(self.mean, self.hl.mean)

    def test_device_getter(self):
        self.assertEqual(self.device, self.hl.device)

    def test_generator_getter(self):
        self.assertEqual(self.generator, self.hl.generator)
