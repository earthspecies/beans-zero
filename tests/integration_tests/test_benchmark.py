"""Tests the benchmarking code on a dummy dataset"""

import json
import numpy as np
from pathlib import Path
from datasets import Dataset

from beans_zero.benchmark import run_benchmark
from beans_zero.config import beans_cfg


# generate a dummy dataset based on the component datasets
# in beans_cfg
def generate_dummy_dataset() -> Dataset:
    components = beans_cfg["metadata"]["components"]
    # generate a dummy dataset with 1000 samples per component
    # for each component dataset
    n_samples = 1000
    dummy_data = []
    for component in components:
        if component["name"] == "captioning":
            continue
        name = component["name"]
        unique_labels = component["labels"]
        # generate random labels from the unique labels
        labels = np.random.choice(unique_labels, size=n_samples)
        for i in range(n_samples):
            dummy_data.append({"dataset_name": name, "output": str(labels[i])})

    return Dataset.from_list(dummy_data)


def test_benchmark(tmpdir: Path) -> None:
    """Test the benchmark function with a dummy dataset."""
    # generate a dummy dataset
    dummy_dataset = generate_dummy_dataset()
    # save the dummy dataset to disk
    dummy_dataset.save_to_disk(tmpdir / "dummy_dataset")

    # run the benchmark function on the loaded dataset
    # in batched mode
    batch_size = 2
    run_benchmark(
        model=lambda x: ["my prediction"] * batch_size,
        path_to_dataset=tmpdir / "dummy_dataset",
        streaming=False,
        batched=True,
        batch_size=batch_size,
        output_path=tmpdir / "metrics.json",
    )

    # load the metrics file
    with open(tmpdir / "metrics.json", "r") as metrics_file:
        metrics = json.load(metrics_file)

    # check that the metrics file is not empty
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "esc50" in metrics
    assert "Accuracy" in metrics["esc50"]
    assert "F1 Score" in metrics["esc50"]
    assert metrics["esc50"]["Accuracy"] < 0.1

    assert "hiceas" in metrics
    assert "F1" in metrics["hiceas"]
    assert "mAP" in metrics["hiceas"]
    assert metrics["hiceas"]["F1"] < 0.1


def test_benchmark_streaming(tmpdir: Path) -> None:
    """Test the benchmark function with a dummy dataset."""
    # generate a dummy dataset
    dummy_dataset = generate_dummy_dataset()
    # save the dummy dataset to disk
    dummy_dataset.save_to_disk(tmpdir / "dummy_dataset")

    # test with streaming
    batch_size = 2
    run_benchmark(
        model=lambda x: ["my prediction"] * batch_size,
        path_to_dataset=tmpdir / "dummy_dataset",
        streaming=True,
        batched=True,
        batch_size=batch_size,
        output_path=tmpdir / "metrics.json",
    )

    # load the metrics file
    with open(tmpdir / "metrics.json", "r") as metrics_file:
        metrics = json.load(metrics_file)

    # check that the metrics file is not empty
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "esc50" in metrics
    assert "Accuracy" in metrics["esc50"]
    assert "F1 Score" in metrics["esc50"]
    assert metrics["esc50"]["Accuracy"] < 0.1

    assert "hiceas" in metrics
    assert "F1" in metrics["hiceas"]
    assert "mAP" in metrics["hiceas"]
    assert metrics["hiceas"]["F1"] < 0.1
