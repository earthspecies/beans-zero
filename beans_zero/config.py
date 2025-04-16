"""Config definitions for zero-shot model evaluation."""

import json
from pathlib import Path
from dataclasses import dataclass, field

TASK_TYPES = ["detection", "classification", "captioning"]
REQUIRED_KEYS_IN_PREDICTIONS = ["prediction", "label", "dataset_name"]


def load_beans_zero_dataset_config() -> dict:
    """Load the BEANS-Zero dataset configuration from a JSON file
    located in the root directory of the package.

    Returns
    -------
    beans_cfg : dict
        The configuration dictionary containing metadata and components.
    """
    root_dir = Path(__file__).resolve().parent.parent
    with open(str(root_dir / "beans_zero_dataset_config.json"), "r") as cfg_file:
        beans_cfg = json.load(cfg_file)
    return beans_cfg


@dataclass
class EvaluationConfig:
    """Default configuration for evaluation of zero-shot learning models."""

    max_distance_for_match: int = 5
    binarization_threshold: float = 0.5
    multi_label: bool = False
    max_levenstein_distance: int = 5
    end_of_text_token: str = "<|end_of_text|>"

    default_label_for_detection: str = "None"
    verbose: bool = False
    classification_score_average: str = "weighted"

    task_types: list[str] = field(default_factory=lambda: TASK_TYPES)
    required_keys_in_predictions_file: list[str] = field(
        default_factory=lambda: REQUIRED_KEYS_IN_PREDICTIONS
    )


eval_cfg = EvaluationConfig()
beans_cfg = load_beans_zero_dataset_config()
