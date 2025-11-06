from typing import Callable, List, Literal, Tuple
import torch
import os
import pandas as pd


class Simulation:
    def __init__(
        self,
        sample_count: int,
        feature_count: int,
        neuron_counts: List[int],
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        generator: torch.Generator | None = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        non_zero_first: bool = True,
    ):
        self.__sample_count = sample_count
        self.__feature_count = feature_count
        self.__neuron_counts = neuron_counts
        self.__activation_fn = activation_fn
        self.__generator = torch.Generator(device) if generator is None else generator
        self.__device = device
        self.__non_zero_first = non_zero_first

        if device != self.__generator.device:
            raise ValueError("simulation and generator must live on the same device")

        if not self.__neuron_counts:
            raise ValueError("neuron counts can not be empty")

    def run(self):
        # Create inputs
        self.__dataset = torch.normal(
            0,
            1,
            [self.__sample_count, self.__feature_count],
            generator=self.__generator,
            device=self.__device,
        )

        # Create first layer weights to be sparse
        # Calculate the number of non-zero elements
        num_non_zero_elements = max(int(self.__feature_count * 0.01), 1)

        if self.__non_zero_first:
            # Select the first indices for the non-zero elements
            indices = list(range(num_non_zero_elements))
        else:
            # Randomly select indices for the non-zero elements
            indices = torch.randperm(
                self.__feature_count,
                generator=self.__generator,
            )[:num_non_zero_elements]
        fc1_weight = torch.zeros(
            [self.__neuron_counts[0], self.__feature_count],
            device=self.__device,
        )
        fc1_weight[:, indices] = torch.randn(
            [self.__neuron_counts[0], len(indices)],
            generator=self.__generator,
            device=self.__device,
        )
        fc1_bias = torch.zeros(
            [self.__neuron_counts[0]],
            device=self.__device,
        )
        fc1_bias[indices] = torch.randn(
            [len(indices)],
            generator=self.__generator,
            device=self.__device,
        )
        self.__weights = [fc1_weight]
        self.__biases = [fc1_bias]

        # Create other layer weights
        self.__weights.extend(
            [
                torch.normal(
                    0,
                    1,
                    [
                        self.__neuron_counts[i],
                        self.__neuron_counts[i - 1] if i > 0 else self.__feature_count,
                    ],
                    generator=self.__generator,
                    device=self.__device,
                )
                for i in range(1, len(self.__neuron_counts))
            ]
        )

        # Create biases
        self.__biases.extend(
            [
                self.__initialize_sparse_tensor(self.__neuron_counts[i])
                for i in range(1, len(self.__neuron_counts) - 1)
            ]
        )
        self.__biases.append(
            torch.randn(
                self.__neuron_counts[-1],
                generator=self.generator,
                device=self.device
            ),
        )

        # Compute target
        target = self.__dataset.clone().detach()
        for i, (weight, bias) in enumerate(zip(self.__weights, self.__biases)):
            # Linear product
            target = torch.matmul(target, weight.t()) + bias

            # Activation function not applied to last layer
            if i < len(self.weights) - 1:
                target = self.__activation_fn(target)
        self.__target = target

    def __initialize_sparse_tensor(self, N: int, sparsity=0.01):
        # Create a tensor of size N x M initialized with zeros
        tensor = torch.zeros(
            [N],
            device=self.__device,
        )

        # Calculate the number of non-zero elements
        num_non_zero_elements = max(int(N * sparsity), 1)

        if self.__non_zero_first:
            # Select the first indices for the non-zero elements
            indices = list(range(num_non_zero_elements))
        else:
            # Randomly select indices for the non-zero elements
            indices = torch.randperm(
                N,
                generator=self.__generator,
            )[:num_non_zero_elements]

        tensor_flat = tensor.view(-1)  # Flatten the tensor to 1D for easy indexing
        tensor_flat[indices] = torch.randn(
            num_non_zero_elements,
            generator=self.__generator,
            device=self.__device,
        )
        return tensor

    def save(self, path: str, format: Literal["pytorch", "csv"] = "csv"):
        if os.path.isdir(path):
            if os.listdir(path):
                raise OSError("output directory is not empty")
        else:
            os.makedirs(path)

        if format == "pytorch":
            torch.save(self.__dataset, os.path.join(path, "dataset.pt"))
            torch.save(self.__weights, os.path.join(path, "weights.pt"))
            torch.save(self.__biases, os.path.join(path, "biases.pt"))
            torch.save(self.__target, os.path.join(path, "target.pt"))
        elif format == "csv":
            # Save dataset
            pd.DataFrame(self.__dataset.cpu().numpy()).to_csv(
                os.path.join(path, "dataset.csv"), index=False
            )
            # Save weights
            for i, weight in enumerate(self.__weights):
                pd.DataFrame(weight.cpu().numpy()).to_csv(
                    os.path.join(path, f"weight_{i}.csv"), index=False
                )
            # Save biases
            for i, bias in enumerate(self.__biases):
                pd.DataFrame(bias.cpu().numpy()).to_csv(
                    os.path.join(path, f"bias_{i}.csv"), index=False
                )
            # Save target
            pd.DataFrame(self.__target.cpu().numpy()).to_csv(
                os.path.join(path, "target.csv"), index=False
            )

    @staticmethod
    def load(
        path: str,
        format: Literal["pytorch", "csv"] = "csv",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        if format == "pytorch":
            dataset = torch.load(
                os.path.join(path, "dataset.pt"),
                map_location=device,
            )
            weights = torch.load(
                os.path.join(path, "weights.pt"),
                map_location=device,
            )
            biases = torch.load(
                os.path.join(path, "biases.pt"),
                map_location=device,
            )
            target = torch.load(
                os.path.join(path, "target.pt"),
                map_location=device,
            )
        elif format == "csv":
            # Load dataset
            dataset = torch.tensor(
                pd.read_csv(os.path.join(path, "dataset.csv")).values,
                device=device,
                dtype=torch.float,
            )
            # Load weights
            weights = []
            i = 0
            while os.path.exists(os.path.join(path, f"weight_{i}.csv")):
                weights.append(
                    torch.tensor(
                        pd.read_csv(os.path.join(path, f"weight_{i}.csv")).values,
                        device=device,
                        dtype=torch.float,
                    )
                )
                i += 1
            # Load biases
            biases = []
            i = 0
            while os.path.exists(os.path.join(path, f"bias_{i}.csv")):
                biases.append(
                    torch.tensor(
                        pd.read_csv(os.path.join(path, f"bias_{i}.csv")).values,
                        device=device,
                        dtype=torch.float,
                    ).view(-1)
                )
                i += 1
            # Load target
            target = torch.tensor(
                pd.read_csv(os.path.join(path, "target.csv")).values,
                device=device,
                dtype=torch.float,
            )
        else:
            raise ValueError("Unknown format")

        return (dataset, weights, biases, target)

    @property
    def sample_count(self) -> int:
        return self.__sample_count

    @property
    def feature_count(self) -> int:
        return self.__feature_count

    @property
    def neuron_counts(self) -> List[int]:
        return self.__neuron_counts

    @property
    def dataset(self) -> torch.Tensor:
        return self.__dataset

    @property
    def weights(self) -> List[torch.Tensor]:
        return self.__weights

    @property
    def biases(self) -> List[torch.Tensor]:
        return self.__biases

    @property
    def target(self) -> torch.Tensor:
        return self.__target

    @property
    def generator(self) -> torch.Generator:
        return self.__generator

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def non_zero_first(self) -> bool:
        return self.__non_zero_first
