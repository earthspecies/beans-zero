"""Model prediction post-processing for beans-zero evaluation"""

from dataclasses import dataclass
from typing import Literal

from Levenshtein import distance as levenshtein_distance
from beans_zero.config import eval_cfg


@dataclass
class EvalPostProcessor:
    """Post-process the predictions from the NatureLM-audio model.
    This can also be used for evaluating other models.

    Arguments
    ---------
    target_label_set : set[str]
        The set of labels to match against, generally determined from the dataset
        as the set of unique labels in the dataset
    task : str
        The task to perform, either 'classification', 'detection', or 'captioning'
    end_token : str
        The end token for the prediction
    max_levenstein_distance : int
        The maximum levenshtein distance to find a match between
        the prediction and the labels

    Examples
    --------
    >>> processor = EvalPostProcessor(set(["dog", "cat", "bird"]),"classification")
    >>> predictions = ["dog", "cat", "bird"]
    >>> labels = ["dog", "cat", "bird"]
    >>> processor(predictions)
    ['dog', 'cat', 'bird']

    """

    target_label_set: set[str]
    task: Literal["classification", "detection", "captioning"]
    end_token: str = eval_cfg.end_of_text_token
    max_levenstein_distance: int = eval_cfg.max_levenstein_distance

    def __post_init__(self) -> None:
        if self.task not in ["classification", "detection", "captioning"]:
            raise ValueError(
                "The task must be one of 'classification', 'detection', 'captioning'"
            )

        self.multi_label = False  # classification
        if self.task == "detection":
            self.multi_label = True
            self.target_label_set.add("None")

        self.match_to_labels = True
        if self.task == "captioning":
            # this is not a match issue, so just return the input
            self.match_to_labels = False
            self.target_label_set = set()

    def remove_eos_token(self, text: str) -> str:
        return text.split(self.end_token)[0]

    def get_nearest_label(self, text: str) -> str:
        """
        Find the nearest label to the text using levenshtein distance.
        If the distance is greater than max_distance_for_match, return 'None'.

        If multi_label is True, return a comma-separated string of labels.

        Arguments
        ---------
        text : str
            The text to match

        Returns
        -------
        str
            The matched label

        Examples
        --------
        >>> processor = EvalPostProcessor(set(["dog", "cat", "bird"]),"classification")
        >>> processor.get_nearest_label("dog")
        'dog'
        """
        # Check for exact match first
        text = self.remove_eos_token(text)
        # TODO do lower()?
        # equal_labels = [label for label in labels if label == text]
        if text in self.target_label_set:
            return text

        nearest_label = min(
            list(self.target_label_set),
            key=lambda label: levenshtein_distance(text, label),
        )
        if (
            levenshtein_distance(nearest_label, text) > self.max_levenstein_distance
            and self.multi_label
        ):
            return "None"  # DETECTION only: no strong match, choose None

        return nearest_label

    def get_nearest_labels(self, text: str) -> str:
        """
        Parse text into multiple predictions based on a separator ','.
        For each prediction, get the nearest label.

        Arguments
        ---------
        text : str
            The text to match
        labels : list[str]
            The list of labels to match against

        Returns
        -------
        str
            The matched labels
        """
        predictions = [
            self.get_nearest_label(prediction.strip()) for prediction in text.split(",")
        ]
        return ", ".join(predictions)

    def __call__(self, predictions: list[str]) -> list[str]:
        """
        Post-process the predictions from a model.
        If the models output several predictions, they are separated by a comma.

        Arguments
        ---------
        predictions : list[str]
            The predictions to post-process

        Returns
        -------
        list[str]
            The post-processed predictions
        """
        if not self.match_to_labels:  # Return raw outputs
            return predictions

        matched_predictions = []
        for prediction in predictions:
            if self.multi_label:
                matched_predictions.append(self.get_nearest_labels(prediction))
            else:
                matched_predictions.append(self.get_nearest_label(prediction))

        return matched_predictions
