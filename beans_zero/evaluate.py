"""Evaluation module for evaluating predictions against ground truth labels."""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from config import TASK_TYPES, eval_cfg
from metrics import MeanAveragePrecision, MulticlassBinaryF1Score, compute_spider
from post_processor import EvalPostProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("beans_zero")


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
        logger.info(f"SPIDEr Score: {spider_score:.4f}")

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

    Raises
    ------
    ValueError: If the number of labels is <= 0

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
    if num_labels <= 0 or not isinstance(num_labels, int):
        raise ValueError("Number of labels must be greater than 0.")

    # This is the default tensor to return, which means no labels were detected
    tensor = torch.zeros(num_labels)

    # If output is a string, process it as before
    if isinstance(output, str):
        # TODO: This is some special case handling for "None" labels.
        # We have to make this clear in the documentation.
        # if output.lower() == "none":
        #     return tensor
        if output == "None":
            return tensor

        # FIXME: what about doing lowercase here ?
        output = output.split(", ")
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
    labels: Iterable[str] = None,
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
    {'mAP': 1.0, 'F1': 1.0, 'Recall': 1.0, 'Precision': 1.0}
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
        logger.info("Labels are:\n", labels)

    label_to_id = {label: i for i, label in enumerate(labels)}

    # Convert target and output to tensors
    # This will produce tensors of shape (num_examples, num_labels)
    labels_tensor = torch.stack([parse_detection_output(t, num_labels, label_to_id) for t in true_labels])
    predictions_tensor = torch.stack([parse_detection_output(o, num_labels, label_to_id) for o in predictions])

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
    verbose: bool = eval_cfg.verbose,
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


def evaluate(predictions: Iterable[str], true_labels: Iterable[str], task: str, labels: Iterable[str] = None) -> dict:
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

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Raises
    ------
        ValueError: If the number of predictions and true labels do not match.
        NotImplementedError: If the task is not supported.
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


def compute_metrics(outputs: pd.DataFrame, verbose: bool = True) -> dict:
    """Compute metrics from a model output dataframe.

    Arguments
    ---------
    outputs: pd.DataFrame
        DataFrame containing the model outputs. The dataframe must contain the following columns:
        - dataset_name: The name of the dataset
        - prediction: The model's prediction
        - label: The ground truth label

    verbose: bool
        Whether to print the computed metrics for each dataset.

    Returns
    -------
        dict: Dictionary containing all computed metrics.

    Raises
    ------
        ValueError: If the required columns are not found in the dataframe.
    """
    if not all(col in outputs.columns for col in eval_cfg.required_keys_in_predictions_file):
        raise ValueError(
            f"Model outputs dataframe must contain the following columns: {eval_cfg.required_keys_in_predictions_file}"
        )

    with open("../beans_zero_dataset_config.json", "r") as cfg_file:
        beans_cfg = json.load(cfg_file)

    components = beans_cfg["metadata"]["components"]
    ds_names = [d["name"] for d in components]
    ds_tasks = [d["task"] for d in components]

    all_metrics = {}
    for i, name in enumerate(ds_names):
        # subset the predictions dataframe
        sub = outputs[outputs["dataset_name"] == name]

        task = ds_tasks[i]

        labels = sub["label"].to_list()
        label_set = set(labels)

        processor = EvalPostProcessor(target_label_set=label_set, task=task)

        predictions = processor(sub["prediction"].to_list())
        metrics = evaluate(predictions, labels, task, None)

        logger.info(f"\nMetrics for dataset {name}:\n{metrics}")

        all_metrics[name] = metrics

    return all_metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns
    -------
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate a predictions file.")
    parser.add_argument("predictions_file", type=str, required=True, help="Path to the predictions file.")
    parser.add_argument("output_path", type=str, required=False, help="Path to save the evaluation results.")
    return parser.parse_args()


def main() -> None:
    """Evaluate a predictions file. The predictions file must have the following columns:

    - dataset_name: The name of the dataset
    - prediction: The model's prediction
    - label: The ground truth label

    The predictions file can be in CSV or JSON or JSONL format.
    The output will be saved to the specified output path.

    Raises
    ------
    FileNotFoundError: If the predictions file does not exist.
    ValueError: If the predictions file is not in CSV or JSON format.

    """
    args = parse_args()

    # Load the predictions file
    predictions_path = Path(args.predictions_file)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")

    if predictions_path.suffix == ".csv":
        outputs = pd.read_csv(predictions_path)
    elif predictions_path.suffix == ".json" or predictions_path.suffix == ".jsonl":
        outputs = pd.read_json(predictions_path, orient="records", lines=True)
    else:
        raise ValueError("Predictions file must be a CSV or JSON file.")

    all_metrics = compute_metrics(outputs)

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
            logger.info(f"Metrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
