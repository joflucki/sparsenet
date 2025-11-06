import unittest
import torch
from sparsenet.nn import NormalizedLinear


class TestNormalizedLinear(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(self.device).manual_seed(78915789069)

    def test_with_bias(self):

        inputs = torch.randn([100, 20], device=self.device, generator=self.generator)
        layer = NormalizedLinear(20, 1, device=self.device)

        truths = (
            torch.matmul(inputs, (layer.weight.t() / torch.norm(layer.weight)))
            + layer.bias
        )
        outputs = layer(inputs)
        self.assertTrue(torch.allclose(outputs, truths))

    def test_without_bias(self):

        inputs = torch.randn([100, 20], device=self.device, generator=self.generator)
        layer = NormalizedLinear(20, 1, bias=False, device=self.device)

        truths = torch.matmul(inputs, (layer.weight.t() / torch.norm(layer.weight)))
        outputs = layer(inputs)
        self.assertTrue(torch.allclose(outputs, truths))
