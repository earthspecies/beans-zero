import doctest
import numpy as np
import pandas as pd
import json
from beans_zero import evaluate
from beans_zero.evaluate import compute_metrics

np.random.seed(0)


def test_utils_doctests() -> None:
    """Run doctests in the utils module."""
    failures, _ = doctest.testmod(evaluate)
    assert failures == 0


def generate_classification_dummy_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
):
    """Generate dummy data for testing classification metrics.
    Returns
    -------
    true_labels (np.ndarray): Array of true labels.
    predicted_labels (np.ndarray): Array of predicted labels.
    scores (np.ndarray): Array of scores.
    label_set (list[str]): List of labels.
    """
    label_set = ["A", "B", "C", "D", "E"]
    n_samples = 100
    n_classes = len(label_set)

    # Generate random true labels
    true_labels = np.random.choice(label_set, n_samples)

    # Generate random predicted labels
    predicted_labels = np.random.choice(label_set, n_samples)

    # Generate random scores
    scores = np.random.rand(n_samples, n_classes)

    # Normalize scores to sum to 1
    scores = scores / np.sum(scores, axis=1, keepdims=True)
    return true_labels, predicted_labels, scores, label_set


def generate_esc50_dummy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate dummy data according to the esc50 dataset config
    for testing classification metrics.

    Returns
    -------
    true_labels (np.ndarray): Array of true labels.
    predicted_labels (np.ndarray): Array of predicted labels.
    label_set (list[str]): List of labels.
    """
    with open("beans_zero_dataset_config.json", "r") as cfg_file:
        beans_cfg = json.load(cfg_file)

    esc50_cfg = [
        ds for ds in beans_cfg["metadata"]["components"] if ds["name"] == "esc50"
    ][0]
    label_set = esc50_cfg["labels"]

    n_samples = 400
    # Generate random true labels
    true_labels = np.random.choice(label_set, n_samples)

    # Generate random predicted labels
    predicted_labels = np.random.choice(label_set, n_samples)
    return true_labels, predicted_labels, label_set


def test_compute_metrics() -> None:
    """Test the compute_metrics function."""
    true_labels, predicted_labels, label_set = generate_esc50_dummy_data()
    num_classes = len(label_set)
    # first make dataframe with incorrect columns, should raise ValueErrr
    outputs_df = pd.DataFrame(
        {
            "prediction": predicted_labels,
            "label": true_labels,
            "dataset_name": ["esc50"] * len(true_labels),
        }
    )
    # Compute metrics
    metrics = compute_metrics(
        outputs=outputs_df,
        verbose=True,
    )
    metrics = metrics["esc50"]

    # Check if the metrics are computed correctly
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1 Score" in metrics
    assert "Top-1 Accuracy" in metrics

    # all metrics should not be much higher than chance level
    chance_level = 1 / num_classes

    assert metrics["Accuracy"] < chance_level + 0.1
    assert metrics["Precision"] < chance_level + 0.1
    assert metrics["Recall"] < chance_level + 0.1
    assert metrics["F1 Score"] < chance_level + 0.1
    assert metrics["Top-1 Accuracy"] < chance_level + 0.1

    # Compute again, but now with all predictions = "None"
    outputs_df["prediction"] = "None"
    metrics = compute_metrics(
        outputs=outputs_df,
        verbose=True,
    )

    metrics = metrics["esc50"]
    assert metrics["Accuracy"] < chance_level + 0.1
    assert metrics["Precision"] < chance_level + 0.1
    assert metrics["Recall"] < chance_level + 0.1
    assert metrics["F1 Score"] < chance_level + 0.1
    assert metrics["Top-1 Accuracy"] < chance_level + 0.1

    # Compute again, but now with all predictions = true_labels
    outputs_df["prediction"] = outputs_df["label"]
    metrics = compute_metrics(
        outputs=outputs_df,
        verbose=True,
    )
