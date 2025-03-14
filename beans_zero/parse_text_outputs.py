import numpy as np
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from .config import eval_cfg


# def parse_prediction(self, text: str):
#     return text.split(END_TOKEN)[0]


def get_nearest_label(
    text: str,
    labels: list[str],
    max_distance_for_match: int = eval_cfg.max_distance_for_match,
    embedding_match: bool = False,
    label_embeddings: np.array = None,
    multi_label: bool = eval_cfg.multi_label,
    embedding_model=None,
) -> str:
    """
    Get the nearest label in labels to the text.

    If the text is in the labels, return the text.
    If not, return the label with the smallest Levenshtein distance to the text.
    Or, if embedding_match is True, return the label with the highest cosine similarity to the text embedding.

    Args:
        text: str
        labels: list[str]
    """
    # Check for exact match first
    # FIXME: move this to model
    # text = self.parse_prediction(text)

    # test exact match first
    if text in labels:
        return text

    # No exact match. Use embedder or levenshtein distance for approximate match.
    if embedding_match:
        text_embedding = embedding_model.embed([text])
        max_similarity = np.argmax(cosine_similarity(text_embedding, label_embeddings)[0])
        return labels[max_similarity]

    nearest_label = min(labels, key=lambda label: levenshtein_distance(text, label))
    if levenshtein_distance(nearest_label, text) > max_distance_for_match and multi_label:
        return eval_cfg.default_label_for_detection  # DETECTION only: no strong match, choose None

    return nearest_label


def get_nearest_labels(text: str, labels: list[str], separator: str = ",") -> str:
    """
    Parse text into multiple predictions based on a separator.
    For each prediction, get the nearest label.

    Args:
        text: str
        labels: list[str]
        separator: str
    """
    predictions = [get_nearest_label(prediction, labels) for prediction in text.split(separator)]
    # print(f"prediction {text} getting labels {predictions}")
    # FIXME: This is very specific to a model ??
    # return ", ".join(predictions)
    return separator.join(predictions)


def get_approximate_labels(text: str, labels: list[str], max_distance: int = eval_cfg.max_distance_for_match) -> str:
    """
    Find labels that are nearly contained within the text, case-insensitive, and using a distance function.
    If none of the labels is found, return a default label.
    """
    text = text.lower()
    matched_labels = []

    for label in labels:
        label_lower = label.lower()  # TODO: in general, we need to normalize the text and labels

        # slide the label over the text and check for a match using levenstein distance
        for i in range(len(text) - len(label_lower) + 1):
            substring = text[i : i + len(label_lower)]
            if levenshtein_distance(substring, label_lower) <= max_distance:
                matched_labels.append(label)
                break

    print(f"TEXT {text} getting labels {matched_labels}")
    if matched_labels:
        return ", ".join(matched_labels)

    return eval_cfg.default_label_for_detection
