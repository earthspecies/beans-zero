"""Main evaluation module for evaluating model predictions against ground truth labels
and returning metrics.
"""

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from beans_zero.config import eval_cfg
from beans_zero.metrics import (
    MeanAveragePrecision,
    MulticlassBinaryF1Score,
    compute_spider,
)
from beans_zero.post_processor import EvalPostProcessor

logger = logging.getLogger("beans_zero")


def _evaluate_captioning(
    references: list[str], hypotheses: list[str], verbose: bool = False
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
        Whether to print the computed metrics. Defaults to False.

    Returns
    -------
        SPIDEr score for the captioning task

    Examples
    --------
    >>> references = ["a cat is sitting on a mat", "a dog is running in the park"]
    >>> hypotheses = ["a cat is sitting on a mat", "a dog is running in the park"]
    >>> _evaluate_captioning(references, hypotheses)
    {'spider_score': 1.0}
    """
    spider_score = compute_spider(references, hypotheses)
    if verbose:
        logger.info(f"SPIDEr Score: {spider_score:.4f}")

    return {"spider_score": spider_score}


def _parse_detection_output(
    output: str | dict[str, float], num_labels: int, label_to_id: dict[str, int]
) -> torch.Tensor:
    """
    This function parses the model soutput, which can be either a string or
    a dictionary mapping labels to scores.

    Arguments
    ---------
    output: str | dict[str, float]
        Output text from a model, which is expected to be a comma-separated list
        of labels which are detected by a model in the input,
        or a mapping from labels to scores for each label.
    num_labels: int
        Number of labels.
    label_to_id: dict[str, int]
        Mapping from a label to its index.

    Returns
    -------
        tensor: A tensor representing the scores for each label.

    Raises
    ------
    ValueError: If the number of labels is <= 0

    Examples
    --------
    >>> import torch; num_labels = 3
    >>> label_to_id = {"cat": 0, "dog": 1, "bird": 2}
    >>> parse_detection_output("cat, dog, mouse", num_labels, label_to_id)
    tensor([1., 1., 0.])
    >>> out = parse_detection_output({"cat": 0.9, "dog": 0.8}, num_labels, label_to_id)
    >>> torch.allclose(out, torch.tensor([0.9, 0.8, 0.]))
    True
    >>> out = parse_detection_output("None", num_labels, label_to_id)
    >>> torch.allclose(out, torch.tensor([0., 0., 0.]))
    True
    >>> out = parse_detection_output({"mouse": 0.9, "dog": 0.8},
    ... num_labels,
    ... label_to_id)
    >>> torch.allclose(out, torch.tensor([0., 0.8, 0.]))
    True
    """
    if num_labels <= 0 or not isinstance(num_labels, int):
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
        output = output.split(", ")
        # THIS IS SAFER: because what if the label was like 'cat,dog,mouse' ??
        # output = [s.strip() for s in output.split(",")]
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
    labels: Iterable[str] | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Evaluates outputs on a detection task.

    Arguments
    ---------
    output: Iterable[str]
        Iterable of output predictions, each element is either text
        (comma separated detected labels) or,
        a dictionary of label-to-score mappings.
    target: list
        Iterable of true labels in text format.
    labels: Iterable[str], optional
        Iterable of possible labels. If None, inferred from the true_labels.
    verbose: bool
        Whether to print the computed metrics. Defaults to False.

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Examples
    --------
    >>> predictions = ["cat, dog", {"cat": 0.9, "dog": -0.1, "mouse": -0.1}, "mouse"]
    >>> true_labels = ["cat, dog", "cat", "mouse"]
    >>> _evaluate_detection(predictions, true_labels)
    {'mAP': 1.0, 'F1': 1.0, 'Recall': 1.0, 'Precision': 1.0}
    """
    if labels is None:
        # Infer labels from true_labels
        label_set = set()
        for t in true_labels:
            if isinstance(t, str):
                if t.lower() != "none":
                    label_set.update([s.strip() for s in t.split(",")])

        labels = sorted(label_set)

    num_labels = len(labels)
    if verbose:
        logger.info("Labels are:\n", labels)

    label_to_id = {label: i for i, label in enumerate(labels)}

    # Convert target and output to tensors
    # This will produce tensors of shape (num_examples, num_labels)
    labels_tensor = torch.stack(
        [_parse_detection_output(t, num_labels, label_to_id) for t in true_labels]
    )
    predictions_tensor = torch.stack(
        [_parse_detection_output(o, num_labels, label_to_id) for o in predictions]
    )

    # Compute Mean Average Precision (mAP)
    map_metric = MeanAveragePrecision()
    map_metric.update(predictions_tensor, labels_tensor)
    map_value = map_metric.get_metric()["map"]

    map = MulticlassBinaryF1Score(num_labels)
    map.update(predictions_tensor, labels_tensor)
    f1 = map.get_metric()["macro_f1"]
    recall = map.get_metric()["macro_rec"]
    precision = map.get_metric()["macro_prec"]

    if verbose:
        logger.info(f"""Multi-Label Classification Metrics:\n
                Mean Average Precision (mAP): {map_value:.4f}
                Macro avg Precision: {precision:.4f}
                Macro avg Recall   : {recall:.4f}
                Macro avg F1 Score : {f1:.4f}""")

    return {"mAP": map_value, "F1": f1, "Recall": recall, "Precision": precision}


def _evaluate_classification(
    predictions: Iterable[str],
    true_labels: Iterable[str],
    score_average: str = eval_cfg.classification_score_average,
    verbose: bool = False,
) -> dict:
    """
    Evaluates classification metrics including Accuracy, Precision, Recall, F1 Score,
    and Top-1 Accuracy where a prediction is considered correct if it matches any of
    the true labels (which may contain multiple labels separated by commas).

    Arguments
    ---------
    predictions: Iterable[str]
        Predicted labels.
    true_labels: Iterable[str]
        True labels, possibly containing multiple labels separated by commas.
    score_average: str
        Type of averaging to use for Precision, Recall, and F1 Score.
        see for e.g. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    verbose: bool
        Whether to print the computed metrics

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Examples
    --------
    >>> predictions = ["cat", "dog", "bird"]
    >>> true_labels = ["cat", "dog", "bird"]
    >>> metric = _evaluate_classification(predictions, true_labels)
    >>> metric["Accuracy"] == 1
    True
    """
    # Parse true labels into lists of lists
    # FIXME: Why would a true label be anything other than a string ?
    # FIXME: again, why split like ", " ? what if the label was "cat,dog,mouse" ?
    true_labels_list = [
        [part.strip() for part in label.split(",")] for label in true_labels
    ]

    # Compute Top-1 Accuracy
    correct_top1 = 0
    for pred, true in zip(predictions, true_labels_list, strict=False):
        if pred in true:
            correct_top1 += 1
    top1_accuracy = correct_top1 / len(
        true_labels
    )  # FIXME: This will be nan if len(true_labels) == 0

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(
        true_labels, predictions, average=score_average, zero_division=0
    )
    recall = recall_score(
        true_labels, predictions, average=score_average, zero_division=0
    )
    f1 = f1_score(true_labels, predictions, average=score_average, zero_division=0)

    if verbose:
        logger.info(f"""Classification Metrics:\n
                Accuracy: {accuracy:.4f}
                Precision: {precision:.4f}
                Recall   : {recall:.4f}
                F1 Score : {f1:.4f}
                Top-1 Accuracy: {top1_accuracy:.4f}""")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Top-1 Accuracy": top1_accuracy,
    }


def evaluate(
    predictions: Iterable[str],
    true_labels: Iterable[str],
    task: str,
    labels: Iterable[str] = None,
    verbose: bool = eval_cfg.verbose,
) -> dict:
    """
    Evaluate the predictions against the true labels for a given task.

    Arguments
    ---------
    predictions: Iterable[str]
        List of predictions.
    true_labels: Iterable[str]
        List of ground truth labels.
    task: str
        The task type ("detection", "classification", "captioning").
    labels: Iterable[str], optional
        List of possible labels. Required for detection task.
    verbose: bool, optional

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Raises
    ------
        ValueError: If the number of predictions and true labels do not match.
        NotImplementedError: If the task is not supported.

    Examples
    --------
    >>> predictions = ["cat", "dog", "bird"]
    >>> true_labels = ["cat", "dog", "bird"]
    >>> task = "classification"
    >>> metrics = evaluate(predictions, true_labels, task)
    >>> metrics["Accuracy"] == 1
    True
    """
    if len(predictions) != len(true_labels):
        raise ValueError("Number of predictions and true labels must match.")

    if len(predictions) == 0 or len(true_labels) == 0:
        raise ValueError("No predictions or true labels provided.")

    if not all([isinstance(t, str) for t in true_labels]):
        raise ValueError("True labels must be strings.")

    if not all([isinstance(p, str) for p in predictions]):
        raise ValueError("Predictions must be strings.")

    if task == "detection":
        return _evaluate_detection(predictions, true_labels, labels, verbose=verbose)
    elif task == "classification":
        return _evaluate_classification(predictions, true_labels, verbose=verbose)
    elif task == "captioning":
        return _evaluate_captioning(
            true_labels, predictions
        )  # reference captions are the true labels
    else:
        raise NotImplementedError(
            f"task {task} has no metrics implemented. Choose from {eval_cfg.task_types}"
        )


def compute_metrics(outputs: pd.DataFrame, verbose: bool = False) -> dict:
    """Compute metrics from a model output dataframe.

    Arguments
    ---------
    outputs: pd.DataFrame
        DataFrame containing the model outputs.
        The dataframe must contain the following columns:
            * dataset_name: The name of the dataset
            * prediction: The model's prediction
            * label: The ground truth label

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Raises
    ------
        ValueError: If the required columns are not found in the dataframe.

    Examples
    --------
    >>> outputs = pd.DataFrame({
    ...     "dataset_name": ["esc50", "esc50"],
    ...     "prediction": ["cat", "dog"],
    ...     "label": ["cat", "dog"]
    ... })
    >>> metrics = compute_metrics(outputs)
    >>> metrics["esc50"]["Accuracy"] == 1
    True
    """
    if not all(
        col in outputs.columns for col in eval_cfg.required_keys_in_predictions_file
    ):
        raise ValueError(
            f"""Model outputs dataframe must contain the following columns:
            {eval_cfg.required_keys_in_predictions_file}"""
        )
    root_dir = Path(__file__).resolve().parent.parent
    with open(str(root_dir / "beans_zero_dataset_config.json"), "r") as cfg_file:
        beans_cfg = json.load(cfg_file)

    components = beans_cfg["metadata"]["components"]
    ds_names = [d["name"] for d in components]
    ds_tasks = [d["task"] for d in components]

    all_metrics = {}
    for i, name in enumerate(ds_names):
        # subset the predictions dataframe
        sub = outputs[outputs["dataset_name"] == name]
        if sub.empty:
            logger.warning(f"No predictions found for dataset {name}")
            continue

        task = ds_tasks[i]
        labels = sub["label"].to_list()
        # TODO switch to using labels from the dataset config
        label_set = set(labels)

        processor = EvalPostProcessor(target_label_set=label_set, task=task)
        predictions = processor(sub["prediction"].to_list())

        metrics = evaluate(predictions, labels, task, None, verbose=verbose)

        logger.info(f"\nMetrics for dataset {name}:\n{metrics}")

        all_metrics[name] = metrics

    return all_metrics
