import unittest
import torch
from sparsenet.nn import PenalizedL2Loss


class TestPenalizedL2Loss(unittest.TestCase):
    def setUp(self):
        self.input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.target = torch.tensor(
            [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]
        )
        self.criterion = PenalizedL2Loss([self.input], 1)

    def test_loss(self):
        loss, _ = self.criterion(self.input, self.target)
        self.assertAlmostEqual(loss.item(), 140.2426469, places=5)

    def test_penalty(self):
        _, penalty = self.criterion(self.input, self.target)
        self.assertEqual(penalty.item(), 55)
