"""
This module contains the `Simulation` class, which is used to generate
coherent datasets for sparse models. The datasets include a standard
Gaussian distribution, weights with a small number of non-zero entries,
and an error term.
"""

import torch
import math
from typing import Any, List, Literal, Tuple
import csv
import os

__all__ = ["Simulation"]


class Simulation:
    """A numerical simulation, used to generate coherent datasets for sparse models.

    Attributes:
        sample_count (int): The number of samples in the dataset.
        feature_count (int): The number of features in the dataset.
        generator (torch.Generator) : The generator used in random number generation.
        device (torch.device) : The device on which the simulation lives.
        dataset (torch.Tensor): The generated dataset with an additional column of ones for the intercept.
        weights (torch.Tensor): The weights for the linear model, including the intercept.
        error (torch.Tensor): The error term added to the target.
        target (torch.Tensor): The computed target values.
        non_zero_first (bool): Whether the non zero weights should be placed in the first positions of the weights vector.
    """

    def __init__(
        self,
        sample_count: int,
        feature_count: int,
        generator: torch.Generator | None = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        error_rate: float | None = None,
        non_zero_first: bool = True,
    ):
        """
        Initializes the Simulation instance with the specified sample count,
        feature count, random generator, and device.

        Args:
            sample_count (int, optional): The number of samples in the dataset.
            feature_count (int, optional): The number of features in the dataset.
            generator (torch.Generator, optional): The random number generator for reproducibility. Defaults to None.
            device (torch.device, optional): The device on which to perform computations. Defaults to GPU if available, otherwise CPU.
            error_rate (int | None): The size of errors to add to the simulation. If None, no errors are added. Defaults to None.
            non_zero_first (bool): Whether the non zero weights should be placed in the first positions of the weights vector. Defaults to True.
        """
        self.__sample_count = sample_count
        self.__feature_count = feature_count
        self.__device = device
        self.__generator = torch.Generator(device) if generator is None else generator
        self.__error_rate = error_rate
        self.__non_zero_first = non_zero_first

        if device != self.__generator.device:
            raise ValueError("simulation and generator must live on the same device")

    def save(
        self,
        path: str,
        format: Literal["pytorch", "csv"] = "csv",
    ):
        """Saves the data to multiple files.

        Args:
            path (str): The path of the folder where to save the files.
            format (Literal["pytorch", "csv"], optional): The format to save the files in. Defaults to "pytorch".
        """
        if os.path.isdir(path):
            if os.listdir(path):
                raise OSError("output directory is not empty")
        else:
            os.makedirs(path)

        if format == "pytorch":
            torch.save(self.__dataset, os.path.join(path, "dataset.pt"))
            torch.save(self.__weights, os.path.join(path, "weights.pt"))
            torch.save(self.__error, os.path.join(path, "error.pt"))
            torch.save(self.__target, os.path.join(path, "target.pt"))
        elif format == "csv":
            Simulation.__save_csv(self.__dataset, os.path.join(path, "dataset.csv"))
            Simulation.__save_csv(self.__weights, os.path.join(path, "weights.csv"))
            Simulation.__save_csv(self.__error, os.path.join(path, "error.csv"))
            Simulation.__save_csv(self.__target, os.path.join(path, "target.csv"))

        else:
            raise ValueError("Unknown format")

    @staticmethod
    def load(
        path: str,
        format: Literal["pytorch", "csv"] = "pytorch",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loads simulated data from multiple files.

        Args:
            path (str): The folder path for loading the files.
            format (Literal["pytorch", "csv"], optional): The format to load the files from. Defaults to "pytorch".
            device (torch.device, optional): The device on which to load the data. Defaults to GPU if available, otherwise CPU.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The loaded dataset, weights, error, and target tensors.
        """
        if format == "pytorch":
            dataset = torch.load(
                os.path.join(path, "dataset.pt"),
                map_location=device,
            )
            weights = torch.load(
                os.path.join(path, "weights.pt"),
                map_location=device,
            )
            error = torch.load(
                os.path.join(path, "error.pt"),
                map_location=device,
            )
            target = torch.load(
                os.path.join(path, "target.pt"),
                map_location=device,
            )
        elif format == "csv":
            dataset = torch.tensor(
                Simulation.__load_csv(os.path.join(path, "dataset.csv")),
                device=device,
            )
            weights = torch.tensor(
                Simulation.__load_csv(os.path.join(path, "weights.csv")),
                device=device,
            )
            error = torch.tensor(
                Simulation.__load_csv(os.path.join(path, "error.csv")),
                device=device,
            )
            target = torch.tensor(
                Simulation.__load_csv(os.path.join(path, "target.csv")),
                device=device,
            )
        else:
            raise ValueError("Unknown format")

        return (dataset, weights, error, target)

    @staticmethod
    def __save_csv(data: torch.Tensor, path: str):
        """Saves a tensor to a CSV file.

        Args:
            data (torch.Tensor): The tensor to save.
            path (str): The file path to save the CSV.
        """
        with open(path, "w") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(data.cpu().numpy())

    @staticmethod
    def __load_csv(path: str) -> List[Any]:
        """Loads data from a CSV file into a list.

        Args:
            path (str): The file path to load the CSV from.

        Returns:
            List[Any]: The loaded data.
        """
        with open(path, "r") as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            return list(reader)

    def run(self):
        """Runs the simulation to generate the dataset, weights, error, and target."""
        self.__generate_dataset()
        self.__generate_weights()
        self.__generate_error()
        self.__compute_target()

    def __generate_dataset(self):
        """Generates the dataset using a standard Gaussian distribution."""

        ones = torch.ones([self.__sample_count, 1], device=self.__device)
        data = torch.normal(
            mean=0.0,
            std=1.0,
            size=[self.__sample_count, self.__feature_count],
            device=self.__device,
            generator=self.__generator,
        )
        self.__dataset = torch.cat([ones, data], 1)

    def __generate_weights(self):
        """Generates the weights with a small number of non-zero entries."""

        intercept = (
            torch.rand(
                [1, 1],
                device=self.__device,
                generator=self.__generator,
            )
            * 10
            - 5
        )
        body = torch.zeros([self.__feature_count, 1], device=self.__device)

        count = round(self.__feature_count * 0.01)
        if self.__non_zero_first:
            non_zero = torch.normal(
                0,
                3,
                size=[count, 1],
                device=self.__device,
                generator=self.__generator,
            )
            body = torch.cat([non_zero, body[count:]])
        else:
            while count > 0:
                index = torch.randint(
                    0,
                    self.__feature_count,
                    size=[1],
                    device=self.__device,
                    generator=self.__generator,
                ).item()
                index = int(index)
                if math.isclose(body[index].item(), 0):
                    body[index] = torch.normal(
                        0,
                        3,
                        size=[1],
                        device=self.__device,
                        generator=self.__generator,
                    ).item()
                    count -= 1

        self.__weights = torch.cat([intercept, body], 0)

    def __generate_error(self):
        """Generates the error term."""
        if self.__error_rate is None:
            self.__error = torch.zeros([self.__sample_count, 1], device=self.__device)

        else:
            self.__error = torch.normal(
                mean=0,
                std=self.__error_rate,
                size=[self.__sample_count, 1],
                device=self.__device,
                generator=self.__generator,
            )

    def __compute_target(self):
        """Computes the target variable."""
        self.__target = torch.add(
            torch.matmul(self.__dataset, self.__weights),
            0 if self.__error_rate is None else self.__error,
        )

    @property
    def sample_count(self) -> int:
        """The sample count of the dataset."""
        return self.__sample_count

    @property
    def feature_count(self) -> int:
        """The feature count of the dataset."""
        return self.__feature_count

    @property
    def generator(self) -> torch.Generator:
        """The generator used in random number generation."""
        return self.__generator

    @property
    def device(self) -> torch.device:
        """The device on which the simulation lives."""
        return self.__device

    @property
    def target(self) -> torch.Tensor:
        """The target values."""
        return self.__target

    @property
    def dataset(self) -> torch.Tensor:
        """The dataset."""
        return self.__dataset

    @property
    def weights(self) -> torch.Tensor:
        """The linear weights."""
        return self.__weights

    @property
    def error(self) -> torch.Tensor:
        """The error term added to the target."""
        return self.__error

    @property
    def error_rate(self) -> float | None:
        """The size of errors in the dataset."""
        return self.__error_rate

    @property
    def non_zero_first(self) -> bool:
        """Whether the non zero weights should be placed in the first positions of the weights vector."""
        return self.__non_zero_first
