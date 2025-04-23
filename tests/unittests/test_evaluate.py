import doctest
import pytest
import numpy as np
from beans_zero import evaluate


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


def test_edge_cases_evaluate() -> None:
    """Test edge cases for the evaluate function."""
    # Test with empty predictions and true labels
    predictions = []
    true_labels = []
    with pytest.raises(ValueError):
        evaluate.evaluate(predictions, true_labels, task="classification")

    # test with wrong input data types
    predictions = ["A", "B", "C", 1]
    true_labels = ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        evaluate.evaluate(predictions, true_labels, task="classification")

    # Test unequal lengths of predictions and true labels
    predictions = ["A", "B", "C"]
    true_labels = ["A", "B"]
    with pytest.raises(ValueError):
        evaluate.evaluate(predictions, true_labels, task="classification")

    # Test with predictions == true_labels
    predictions = ["A", "B", "C", "D"]
    true_labels = ["A", "B", "C", "D"]
    metrics = evaluate.evaluate(predictions, true_labels, task="classification")
    assert metrics is not None
    assert metrics["Accuracy"] == 1.0


def test_classification_metrics() -> None:
    # generate random
    true_labels, predictions, _, label_set = generate_classification_dummy_data()
    metrics = evaluate.evaluate(
        predictions, true_labels, task="classification", labels=label_set
    )
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "Accuracy" in metrics
    assert "F1 Score" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics

    assert metrics["Accuracy"] < 0.5


def test_detection_metrics() -> None:
    # generate random
    true_labels, predictions, _, label_set = generate_classification_dummy_data()
    metrics = evaluate.evaluate(
        predictions, true_labels, task="detection", labels=label_set
    )
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "F1" in metrics

    assert metrics["F1"] < 0.5
