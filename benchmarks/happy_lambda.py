from typing import Dict, List
from tqdm import tqdm
from sparsenet.happy_lambda import HappyLambdaLinear
import torch
import time
from csv import DictWriter
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Main function to run the benchmarking for both GPU and CPU."""

    # GPU
    fc_gpu = benchmark(
        feature_count=variate(0, 50_000, 10),
        sample_count=25_000,
        distribution_count=1_000,
        name="Benchmarking feature count on GPU",
        device=torch.device("cuda:1"),
    )
    dc_gpu = benchmark(
        feature_count=20_000,
        sample_count=25_000,
        distribution_count=variate(0, 10_000, 10),
        name="Benchmarking distribution count on GPU",
        device=torch.device("cuda:1"),
    )
    both_gpu = benchmark(
        feature_count=variate(0, 50_000, 10),
        sample_count=25_000,
        distribution_count=variate(0, 10_000, 10),
        name="Benchmarking distribution and feature count on GPU",
        device=torch.device("cuda:1"),
    )

    # CPU
    fc_cpu = benchmark(
        feature_count=variate(0, 50_000, 10),
        sample_count=25_000,
        distribution_count=1_000,
        name="Benchmarking feature count on CPU",
        device=torch.device("cpu"),
    )
    dc_cpu = benchmark(
        feature_count=20_000,
        sample_count=25_000,
        distribution_count=variate(0, 10_000, 10),
        name="Benchmarking distribution count on CPU",
        device=torch.device("cpu"),
    )
    both_cpu = benchmark(
        feature_count=variate(0, 50_000, 10),
        sample_count=25_000,
        distribution_count=variate(0, 10_000, 10),
        name="Benchmarking distribution and feature count on CPU",
        device=torch.device("cpu"),
    )

    plot_distribution_count(dc_gpu, dc_cpu)
    plot_feature_count(fc_gpu, fc_cpu)


def variate(start: int, stop: int, n: int) -> List[int]:
    """Generates a list of integers from start to stop with n steps.

    Args:
        start (int): The starting value of the range.
        stop (int): The ending value of the range.
        n (int): The number of steps in the range.

    Returns:
        List[int]: A list of integers in the specified range.
    """
    return [i if i != 0 else 1 for i in range(start, stop + 1, int(stop / n))]


def benchmark(
    feature_count: int | List[int],
    sample_count: int | List[int],
    distribution_count: int | List[int],
    name: str,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Benchmark the computation of the HappyLambda for different configurations.

    Args:
        feature_count (int | List[int]): The number of features, or a list of feature counts to benchmark.
        sample_count (int | List[int]): The number of samples, or a list of sample counts to benchmark.
        distribution_count (int | List[int]): The number of distributions, or a list of distribution counts to benchmark.
        name (str): The name of the benchmark.
        device (torch.device): The device to run the benchmark on.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing benchmark results.
    """
    lists: List[List[int]] = list(
        filter(
            lambda int_or_list: isinstance(int_or_list, list),
            [feature_count, sample_count, distribution_count],
        )
    )  # type: ignore

    if (
        isinstance(feature_count, list)
        or isinstance(sample_count, list)
        or isinstance(distribution_count, list)
    ):
        size = len(lists[0])
        if not all([len(item) == size for item in lists]):
            raise ValueError("All lists must have the same size")

        if isinstance(feature_count, list):
            fc_list = feature_count
        else:
            fc_list = [feature_count]

        if isinstance(sample_count, list):
            sc_list = sample_count
        else:
            sc_list = [sample_count]

        if isinstance(distribution_count, list):
            dc_list = distribution_count
        else:
            dc_list = [distribution_count]
    else:
        fc_list = [feature_count]
        sc_list = [sample_count]
        dc_list = [distribution_count]

    benchmark_results = []
    size = len(fc_list) * len(sc_list) * len(dc_list)
    generator = torch.Generator(device).manual_seed(10)
    for i in tqdm(
        range(size),
        desc=name,
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
    ):
        fc_index = i // (len(dc_list) * len(sc_list))
        dc_index = (i % (len(dc_list) * len(sc_list))) // len(sc_list)
        sc_index = i % len(sc_list)

        dataset = torch.rand(
            [sc_list[sc_index], fc_list[fc_index]],
            generator=generator,
            device=device,
        )
        computer = HappyLambdaLinear(
            dataset=dataset,
            distribution_count=dc_list[dc_index],
            device=device,
            generator=generator,
        )

        start = time.time()
        computer.run()
        time_taken = time.time() - start
        benchmark_results.append(
            {
                "sample_count": sc_list[sc_index],
                "feature_count": fc_list[fc_index],
                "distribution_count": dc_list[dc_index],
                "time_taken": time_taken,
            },
        )

    name_snakecase = "_".join(name.lower().split(" "))
    with open(f"results_{name_snakecase}.csv", "w") as file:
        writer = DictWriter(file, benchmark_results[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_results)

    return benchmark_results


def plot_feature_count(
    fc_gpu: List[Dict[str, float]],
    fc_cpu: List[Dict[str, float]],
):
    """Plots the benchmark results for feature count on GPU and CPU.

    Args:
        fc_gpu (List[Dict[str, float]]): Benchmark results for feature count on GPU.
        fc_cpu (List[Dict[str, float]]): Benchmark results for feature count on CPU.
    """
    # Data from the first dataset
    feature_count_gpu = [dict["feature_count"] for dict in fc_gpu]
    time_taken_gpu = [float(dict["time_taken"]) for dict in fc_gpu]
    # Data from the second dataset
    feature_count_cpu = [dict["feature_count"] for dict in fc_cpu]
    time_taken_cpu = [float(dict["time_taken"]) for dict in fc_cpu]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(feature_count_gpu))

    bar1 = ax.bar(index, time_taken_gpu, bar_width, label="GPU")
    bar2 = ax.bar(index + bar_width, time_taken_cpu, bar_width, label="CPU")

    ax.set_xlabel("Feature Count")
    ax.set_ylabel("Time Taken (seconds)")
    ax.set_title(
        "Computation time comparison when variating feature count (P)\nN=25000 and M=1000"
    )
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(feature_count_gpu)
    ax.legend()

    plt.tight_layout()
    plt.savefig("feature_count_benchmark.png")
    plt.clf()


def plot_distribution_count(
    dc_gpu: List[Dict[str, float]],
    dc_cpu: List[Dict[str, float]],
):
    """Plots the benchmark results for distribution count on GPU and CPU.

    Args:
        dc_gpu (List[Dict[str, float]]): Benchmark results for distribution count on GPU.
        dc_cpu (List[Dict[str, float]]): Benchmark results for distribution count on CPU.
    """
    # Data from the first dataset
    x_1 = [dict["distribution_count"] for dict in dc_gpu]
    y_1 = [float(dict["time_taken"]) for dict in dc_gpu]
    # Data from the second dataset
    x_2 = [dict["distribution_count"] for dict in dc_cpu]
    y_2 = [float(dict["time_taken"]) for dict in dc_cpu]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(x_1))

    bar1 = ax.bar(index, y_1, bar_width, label="GPU")
    bar2 = ax.bar(index + bar_width, y_2, bar_width, label="CPU")

    ax.set_xlabel("Distribution Count")
    ax.set_ylabel("Time Taken (seconds)")
    ax.set_title(
        "Computation time comparison when variating distribution count (M)\nN=25000 and P=20000"
    )
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x_1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("distribution_count_benchmark.png")
    plt.clf()


if __name__ == "__main__":
    main()
