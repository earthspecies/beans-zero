from dataclasses import dataclass

TASK_TYPES = ["detection", "classification", "captioning"]


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

    tasks: list[str] = TASK_TYPES
    required_keys_in_predictions_file: list[str] = ["prediction", "true_label", "dataset_name"]


eval_cfg = EvaluationConfig()
