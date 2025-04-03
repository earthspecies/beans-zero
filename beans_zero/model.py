"""Model inference interface for benchmarking."""

import torch


class ModelWrapper(torch.nn.Module):
    """Use this class to wrap your model for benchmarking."""

    def __init__(self) -> None:
        """Initialize the model."""
        # Implement the model loading code here
        # make sure to set it in eval() mode and move it to the right device
        pass

    @torch.no_grad()
    def forward(self, example: dict) -> str | list[str]:
        # inference code here
        # your model should either return a single string or a list of strings
        # depending on batched True/False.

        # the example dict will either be a dict of arrays or
        # a dict of single elements depending on batched True/False (in benchmark.py)
        # If batched, batch_size is the len(example["audio"]).

        # Each example will contain two *essential* keys "audio" (numpy array) and "instruction" (text)
        # any other keys can be ignored (please ignore)
        # You may find the "task" key useful.

        pass
