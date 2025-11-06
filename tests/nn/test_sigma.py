import unittest
import torch
from sparsenet.nn import sigma


class TestSigma(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(self.device).manual_seed(53456787)

    def test_results_tensor(self):
        inputs = torch.tensor(
            [-10, -1.1, -1.05, -1, -0.95, -0.9, -0.85, -0.75, -0.5, 0, 1, 10, 100],
            device=self.device,
        )
        truths = torch.tensor(
            [
                -1.0000000001031,
                -0.9936535995509,
                -0.9843369157271,
                -0.9653426410751,
                -0.9343369157271,
                -0.8936535995509,
                -0.8475706325244,
                -0.7496642326786,
                -0.4999977301581,
                0,
                0.9999999998969,
                9.9999999998969,
                100,
            ],
            device=self.device,
        )  # Computed using external calculator (geogebra.org)

        outputs = sigma(inputs)
        self.assertTrue(torch.allclose(outputs, truths))

    def test_results_floats(self):
        inputs = [-10, -1.1, -1.05, -1, -0.95, -0.9, -0.85, -0.75, -0.5, 0, 1, 10, 100]

        truths = [
            -1.0000000001031,
            -0.9936535995509,
            -0.9843369157271,
            -0.9653426410751,
            -0.9343369157271,
            -0.8936535995509,
            -0.8475706325244,
            -0.7496642326786,
            -0.4999977301581,
            0,
            0.9999999998969,
            9.9999999998969,
            100,
        ]

        # Computed using external calculator (geogebra.org)

        outputs = [float(sigma(input).item()) for input in inputs]
        [
            self.assertAlmostEqual(output, truth, places=6)
            for output, truth in zip(outputs, truths)
        ]

    def test_gradients(self):
        inputs = torch.tensor(
            [-10, -1.1, -1.05, -1, -0.95, -0.9, -0.85, -0.75, -0.5, 0, 1, 10, 100],
            device=self.device,
            requires_grad=True,
        )
        truths = torch.tensor(
            [
                0,
                0.1192029220221,
                0.26894142137,
                0.5,
                0.73105857863,
                0.8807970779779,
                0.9525741268224,
                0.9933071490757,
                0.9999546021313,
                0.9999999979388,
                1,
                1,
                1,
            ],
            device=self.device,
        )  # Computed using external calculator (geogebra.org)

        outputs = sigma(inputs) # Compute values
        loss = outputs.sum() # Create a scalar value for backward pass
        loss.backward() # Run backward pass

        self.assertIsNotNone(inputs.grad)
        if inputs.grad is not None:
            self.assertTrue(torch.allclose(inputs.grad, truths))
