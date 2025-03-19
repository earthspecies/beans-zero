from pydantic import BaseModel

TASK_TYPES = ["detection", "classification", "captioning"]


class EvaluationConfig(BaseModel):
    """Default configuration for evaluation of zero-shot learning models."""

    max_distance_for_match: int = 5
    binarization_threshold: float = 0.5
    embedding_match: bool = False
    multi_label: bool = False

    default_label_for_detection: str = "None"
    verbose: bool = False
    classification_score_average: str = "weighted"


eval_cfg = EvaluationConfig()
