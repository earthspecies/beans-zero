from typing import Iterable

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import eval_cfg, TASK_TYPES
from .metrics import MeanAveragePrecision, MulticlassBinaryF1Score, compute_spider


def _evaluate_captioning(
    references: list[str], hypotheses: list[str], verbose: bool = eval_cfg.verbose
) -> dict[str, float]:
    """
    Evaluates captioning task using SPIDEr (SPICE + CIDEr)

    Arguments
    ---------
    references: list[str]
        List of reference captions
    hypotheses: list[str]
        List of predicted captions
    verbose: bool
        Whether to print the computed metrics

    Returns
    -------
        SPIDEr score for the captioning task

    Examples
    --------
    >>> references = ["a cat is sitting on a mat", "a dog is running in the park"]
    >>> hypotheses = ["a cat is sitting on a mat", "a dog is running in the park"]
    >>> evaluate_captioning(references, hypotheses)
    {'spider_score': 1.0}
    """
    spider_score = compute_spider(references, hypotheses)
    if verbose:
        print(f"SPIDEr Score: {spider_score:.4f}")

    return {"spider_score": spider_score}


def parse_detection_output(
    output: str | dict[str, float], num_labels: int, label_to_id: dict[str, int]
) -> torch.Tensor:
    """
    This function parses the output, which can be either a string (text from an LLM) or
    a dictionary mapping labels to scores.

    Arguments
    ---------
    output: str | dict[str, float]
        Output text from a model, which is expected to be a comma-separated list of labels
        which are detected by a model in the input, or a mapping from labels to scores for each label.
    num_labels: int
        Number of labels.
    label_to_id: dict[str, int]
        Mapping from a label to its index.

    Returns
    -------
        tensor: A tensor representing the scores for each label.

    Examples
    --------
    >>> num_labels = 3
    >>> label_to_id = {"cat": 0, "dog": 1, "bird": 2}
    >>> parse_detection_output("cat, dog", num_labels, label_to_id)
    tensor([1., 1., 0.])
    >>> parse_detection_output({"cat": 0.9, "dog": 0.8}, num_labels, label_to_id)
    tensor([0.9, 0.8, 0.0])
    >>> parse_detection_output("None", num_labels, label_to_id)
    tensor([0., 0., 0.])
    >>> parse_detection_output({"mouse": 0.9, "dog": 0.8}, num_labels, label_to_id)
    tensor([0., 0.8, 0.])
    """
    if num_labels == 0:
        raise ValueError("Number of labels must be greater than 0.")

    # This is the default tensor to return, which means no labels were detected
    tensor = torch.zeros(num_labels)

    # If output is a string, process it as before
    if isinstance(output, str):
        # TODO: This is some special case handling for "None" labels.
        # We have to make this clear in the documentation.
        if output.lower() == "none":
            return tensor

        # FIXME: what about doing lowercase here ?
        # original code: output = output.split(", ")
        output = [s.strip() for s in output.split(",")]
        for label in output:
            if label in label_to_id:
                tensor[label_to_id[label]] = 1

    # If output is a dictionary, process it as a map of label-to-score
    elif isinstance(output, dict):
        for label, score in output.items():
            if label in label_to_id:
                tensor[label_to_id[label]] = score

    return tensor


def _evaluate_detection(
    predictions: Iterable[str],
    true_labels: Iterable[str],
    labels: Iterable[str] = None,
    binarization_threshold: float = eval_cfg.binarization_threshold,
    verbose: bool = eval_cfg.verbose,
) -> dict[str, float]:
    """
    Evaluates outputs on a detection task.

    Arguments
    ---------
    output: Iterable[str]
        Iterable of output predictions, each element is either text (comma separated detected labels)
        or a dictionary of label-to-score mappings.
    target: list
        Iterable of true labels in text format.
    labels: Iterable[str], optional
        Iterable of possible labels. If None, inferred from the true_labels.
    binarization_threshold: float, optional
        Threshold for binarizing the output scores.
    verbose: bool, optional
        Whether to print the computed metrics.

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Examples
    --------
    >>> predictions = ["cat, dog", {"cat": 0.9, "dog": 0.8}, "None"]
    >>> true_labels = ["cat, dog", "cat", "None"]
    >>> evaluate_detection(predictions, true_labels)

    """
    if labels is None:
        # Infer labels from both target and output
        label_set = set()
        for t in true_labels:
            if isinstance(t, str):
                if t.lower() != "none":
                    label_set.update([s.strip() for s in t.split(",")])

        labels = sorted(label_set)

    num_labels = len(labels)
    if verbose:
        print("Labels are:\n", labels)

    label_to_id = {label: i for i, label in enumerate(labels)}

    # Convert target and output to tensors
    # This will produce tensors of shape (num_examples, num_labels)
    labels_tensor = torch.stack([parse_detection_output(t, num_labels, label_to_id) for t in true_labels])
    predictions_tensor = torch.stack([parse_detection_output(o, num_labels, label_to_id) for o in predictions])

    # Compute Mean Average Precision (mAP)
    map_metric = MeanAveragePrecision()
    map_metric.update(predictions_tensor, labels_tensor)
    map_value = map_metric.get_metric()["map"]

    # Compute Multi-Label Classification Metrics
    # Binarize the outputs using a threshold (e.g., 0.5) if outputs are scores
    # Adjust the threshold as needed based on your specific use case
    # if predictions_tensor.max() > 1.0 or predictions_tensor.min() < 0.0:
    #     # Assume outputs are not probabilities, apply sigmoid or another activation if necessary
    #     output_binary = (predictions_tensor >= binarization_threshold).int()
    # else:
    #     # If outputs are probabilities
    #     output_binary = (output_tensor >= binarization_threshold).int()

    # # Similarly, ensure target is binary
    # target_binary = (target_tensor > 0).int()

    # # Convert tensors to numpy arrays for sklearn
    # y_true = target_binary.cpu().numpy()
    # y_pred = output_binary.cpu().numpy()

    # Compute metrics
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    # recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    # f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    map = MulticlassBinaryF1Score(num_labels)
    map.update(predictions_tensor, labels_tensor)
    f1 = map.get_metric()["macro_f1"]
    recall = map.get_metric()["macro_rec"]
    precision = map.get_metric()["macro_prec"]

    if verbose:
        print("Multi-Label Classification Metrics:")
        print(f"  Mean Average Precision (mAP): {map_value:.4f}")
        print(f"  Macro avg Precision: {precision:.4f}")
        print(f"  Macro avg Recall   : {recall:.4f}")
        print(f"  Macro avg F1 Score : {f1:.4f}")

    return {"mAP": map_value, "F1": f1, "Recall": recall, "Precision": precision}


def _evaluate_classification(
    predictions: Iterable[str],
    true_labels: Iterable[str],
    score_average: str = eval_cfg.classification_score_average,
    verbose: bool = eval_cfg.verbose,
) -> dict:
    """
    Evaluates classification metrics including Accuracy, Precision, Recall, F1 Score,
    and Top-1 Accuracy where a prediction is considered correct if it matches any of
    the true labels (which may contain multiple labels separated by commas).

    Arguments
    ---------
        predictions (Iterable[str]): Predicted labels.
        true_labels (Iterable[str]): True labels, possibly containing multiple labels separated by commas.
        score_average (str): Type of averaging to use for Precision, Recall, and F1 Score.
            see for e.g. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        verbose (bool): Whether to print the computed metrics

    Returns
    -------
        dict: Dictionary containing all computed metrics.
    """
    # Parse true labels into lists of lists
    # FIXME: Why would a true label be anything other than a string ?
    true_labels_list = [label.split(", ") if isinstance(label, str) else [] for label in true_labels]

    # Compute Top-1 Accuracy
    correct_top1 = 0
    for pred, true in zip(predictions, true_labels_list):
        if pred in true:
            correct_top1 += 1
    top1_accuracy = correct_top1 / len(true_labels)  # FIXME: This will be nan if len(true_labels) == 0

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average=score_average, zero_division=0)
    recall = recall_score(true_labels, predictions, average=score_average, zero_division=0)
    f1 = f1_score(true_labels, predictions, average=score_average, zero_division=0)

    if verbose:
        print("Classification Metrics:")
        print(f"  Accuracy       : {accuracy:.4f}")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
        print(f"  F1 Score       : {f1:.4f}")
        print(f"  Top-1 Accuracy : {top1_accuracy:.4f}")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Top-1 Accuracy": top1_accuracy,
    }


def evaluate(predictions: Iterable[str], true_labels: Iterable[str], task: str, labels: Iterable[str] = None):
    """
    Evaluate the predictions against the true labels for a given task.
    Args:
        predictions (list): List of predictions.
        true_labels (list): List of ground truth labels.
        task (str): The task type ("detection", "classification", "captioning").
    """
    if len(predictions) != len(true_labels):
        raise ValueError("Number of predictions and true labels must match.")

    if len(predictions) == 0 or len(true_labels) == 0:
        raise ValueError("No predictions or true labels provided.")

    if task == "detection":
        return _evaluate_detection(predictions, true_labels, labels)
    elif task == "classification":
        return _evaluate_classification(predictions, true_labels)
    elif task == "captioning":
        return _evaluate_captioning(true_labels, predictions)  # reference captions are the true labels
    else:
        raise NotImplementedError(f"task {task} has no metrics implemented. Choose from {TASK_TYPES}")
