"""Model inference interface for benchmarking."""

from typing import Iterable, Union


class ModelWrapper:
    """Use this class to wrap your model for benchmarking."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        pass

    def __call__(self, *args, **kwargs) -> Union[str | dict[str, Iterable[float]]]:
        # inference code here
        pass
