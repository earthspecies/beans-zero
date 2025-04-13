import doctest
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from beans_zero import metrics
from beans_zero.metrics import AveragePrecision


def test_utils_doctests() -> None:
    """Run doctests in the utils module."""
    failures, _ = doctest.testmod(metrics)
    assert failures == 0


def generate_dummy_data(
    n_samples: int = 100, n_classes: int = 3, random_seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy data for testing average precision
    Arguments
    ---------
        n_samples (int): Number of samples.
        n_classes (int): Number of classes.
        random_seed (int): Random seed for reproducibility.

    Returns
    -------
        scores (torch.Tensor): Random scores of shape (n_samples, n_classes).
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Generate random scores (model predictions)
    scores = torch.rand(n_samples, n_classes)

    # Generate random targets (ground truth)
    targets = torch.zeros(n_samples, n_classes, dtype=torch.int64)
    for i in range(n_samples):
        # Randomly assign each sample to some classes
        for j in range(n_classes):
            if np.random.random() > 0.7:  # 30% chance to belong to each class
                targets[i, j] = 1

    return scores, targets


def compare_implementations() -> None:
    """Compare AveragePrecision with scikit-learn implementation"""
    n_classes = 3
    scores, targets = generate_dummy_data(n_samples=100, n_classes=n_classes)
    # Compute using AveragePrecision class
    ap_meter = AveragePrecision()
    ap_meter.update(scores, targets)
    custom_ap = ap_meter.get_metric()

    # Compute using scikit-learn
    sklearn_ap = torch.zeros(n_classes)
    for k in range(n_classes):
        sklearn_ap[k] = average_precision_score(
            targets[:, k].numpy(), scores[:, k].numpy()
        )

    assert torch.allclose(
        custom_ap, sklearn_ap, atol=1e-6
    ), "Custom AP should match sklearn AP for the same data"


def test_edge_cases() -> None:
    """Test edge cases for AveragePrecision"""
    print("\nTesting Edge Cases:")

    # Case 1: Perfect predictions
    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([[1, 0], [0, 1]])

    ap_meter = AveragePrecision()
    ap_meter.update(scores, targets)
    custom_ap = ap_meter.get_metric()

    sklearn_ap = torch.zeros(2)
    for k in range(2):
        sklearn_ap[k] = average_precision_score(
            targets[:, k].numpy(), scores[:, k].numpy()
        )

    assert torch.allclose(
        custom_ap, sklearn_ap, atol=1e-6
    ), "Custom AP should match sklearn AP for perfect predictions"

    # Case 2: All predictions wrong
    scores = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    targets = torch.tensor([[1, 0], [0, 1]])

    ap_meter = AveragePrecision()
    ap_meter.update(scores, targets)
    custom_ap = ap_meter.get_metric()

    sklearn_ap = torch.zeros(2)
    for k in range(2):
        sklearn_ap[k] = average_precision_score(
            targets[:, k].numpy(), scores[:, k].numpy()
        )

    assert torch.allclose(
        custom_ap, sklearn_ap, atol=1e-6
    ), "Custom AP should match sklearn AP for all wrong predictions"

    # Case 3: No positive samples for a class
    scores = torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, 0.6]])
    targets = torch.tensor([[1, 0, 0], [0, 1, 0]])

    ap_meter = AveragePrecision()
    ap_meter.update(scores, targets)
    custom_ap = ap_meter.get_metric()

    sklearn_ap = torch.zeros(3)
    for k in range(3):
        if targets[:, k].sum() > 0:  # Handle case with no positive samples
            sklearn_ap[k] = average_precision_score(
                targets[:, k].numpy(), scores[:, k].numpy()
            )
        else:
            sklearn_ap[k] = 0.0

    assert torch.allclose(
        custom_ap, sklearn_ap, atol=1e-6
    ), "Custom AP should match sklearn AP for no positive samples"


def test_with_weights() -> None:
    """Test AveragePrecision with sample weights"""
    print("\nTesting with Weights:")

    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    targets = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    weights = torch.tensor(
        [1.0, 0.5, 1.0, 0.5]
    )  # First sample of each class has higher weight

    ap_meter = AveragePrecision()
    ap_meter.update(scores, targets, weights)
    custom_ap = ap_meter.get_metric()

    # scikit-learn also supports sample_weight
    sklearn_ap = torch.zeros(2)
    for k in range(2):
        sklearn_ap[k] = average_precision_score(
            targets[:, k].numpy(), scores[:, k].numpy(), sample_weight=weights.numpy()
        )

    print(f"Custom AP:  {custom_ap}")
    print(f"Sklearn AP: {sklearn_ap}")
    print(f"Difference: {torch.abs(custom_ap - sklearn_ap)}")


def test_incremental_updates() -> None:
    """Test incremental updates to AveragePrecision"""
    print("\nTesting Incremental Updates:")

    # Generate data
    scores, targets = generate_dummy_data(n_samples=100, n_classes=3, random_seed=42)

    # Update in one go
    ap_meter1 = AveragePrecision()
    ap_meter1.update(scores, targets)
    ap_one_go = ap_meter1.get_metric()

    # Update incrementally
    ap_meter2 = AveragePrecision()
    batch_size = 20
    for i in range(0, 100, batch_size):
        ap_meter2.update(scores[i : i + batch_size], targets[i : i + batch_size])
    ap_incremental = ap_meter2.get_metric()

    print(f"AP (One Update):     {ap_one_go}")
    print(f"AP (Incremental):    {ap_incremental}")
    print(f"Difference:          {torch.abs(ap_one_go - ap_incremental)}")

    assert torch.allclose(
        ap_one_go, ap_incremental, atol=1e-6
    ), "Incremental updates should give the same result"
