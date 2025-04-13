r"""Benchmark your audio-text model on the BEANS-Zero dataset.

This is how you use this module:

Create a file `model.py` (or any other name, this may also be an installed module).
with a `predict` function that takes a single example
as a dictionary and returns the model's prediction.

For example, your model.py could look like this:
```python
import torch

class MyModel(torch.nn.Module):
    "A simple example model."
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize your model here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "A simple forward pass."
        # Do something with the input tensor
        return x

def predict(example: dict) -> str | list[str]:
    # The example contains the 'audio' and "instruction_text" fields
    # You can use your model to make a prediction.

    # if batched is True, 'example' will be a dict of arrays. The 'audio'
    # field will be a list[list[float]]
    # and the 'instruction_text' field will be a list[str].

    # if batched is False, 'example' will be a dict of single elements
    # The 'audio' field will be a list[float] and the 'instruction_text'
    # field will be a str.

    audio = torch.Tensor(example["audio"])
    ... # Do something with the audio and instruction_text
    # Return the model's prediction
    # the prediction can be a single string or a list of strings
    # depending on the batched = False / True respectively.
    return prediction
```

Then, you can run the benchmark like this:
```bash
beanz-benchmark \
    --path-to-model-module model.py \
    --path-to-dataset EarthSpeciesProject/BEANS-Zero \
    --batched \
    --batch-size 32 \
    --output-path metrics.json
```
"""

import json
import logging
from pathlib import Path
from importlib import import_module

import pandas as pd
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from beans_zero.evaluate import compute_metrics
from beans_zero.utils import load_module_from_path

logger = logging.getLogger("beans_zero")


def run_benchmark(
    path_to_model_module: str,
    path_to_dataset: str,
    streaming: bool,
    batched: bool,
    batch_size: int,
    output_path: str,
) -> None:
    """Main function to load the BEANS-Zero dataset, iterate over it,
    perform inference with a model and compute metrics.

    This function handles command line arguments, loads the dataset,
    and processes it in batches.

    Arguments
    ---------
        path_to_model_module (str): Path to the model module.
        path_to_dataset (str): Path to the dataset.
        streaming (bool): Whether to stream the dataset.
        batched (bool): Whether to batch the dataset.
        batch_size (int): The batch size.
        output_path (str): Path to save the metrics output as a json file.

    Raises
    ------
        ValueError: If the model function is not found in the module.
        ImportError: If the module cannot be imported.

    Examples
    ---------
        >>> run_benchmark(
        ...     path_to_model_module="path/to/your/model.py",
        ...     path_to_dataset="EarthSpeciesProject/BEANS-Zero",
        ...     streaming=True,
        ...     batched=True,
        ...     batch_size=32,
        ...     output_path="metrics.json",
        ... )
    """
    # Check if the dataset path is local
    datapath = Path(path_to_dataset)
    if not datapath.exists():
        # Load the remote dataset
        dataset = load_dataset(
            "EarthSpeciesProject/BEANS-Zero", streaming=streaming, split="test"
        )
    else:
        # Load the local dataset
        dataset = load_from_disk(path_to_dataset)

    if batched:
        dataset = dataset.batch(batch_size=batch_size)

    # Print info about the dataset
    logger.info("Dataset loaded. Info:")
    logger.info(f"Dataset features: {dataset.features}")

    logger.info("Loading your model prediction function...")
    # use the path to the model module
    try:
        # Check if it's a file path or a module name
        if Path(path_to_model_module).exists() and path_to_model_module.endswith(".py"):
            model_module = load_module_from_path(path_to_model_module)
        else:
            # Fall back to regular import for module names
            model_module = import_module(path_to_model_module)
    except Exception as e:
        raise ImportError(
            f"Could not import the model module {path_to_model_module}"
        ) from e

    prediction_func = getattr(model_module, "predict", None)
    if not prediction_func:
        raise ValueError(
            f"""Prediction function 'predict' not found
            in the module {path_to_model_module}"""
        )

    outputs = {"prediction": [], "label": [], "dataset_name": []}
    for _, example in tqdm(enumerate(dataset)):
        # example can either be a dict of arrays or
        # a dict of single elements depending on batched True/False
        output = prediction_func(example)

        if batched and isinstance(output, list):
            # output is a list so extend
            outputs["prediction"].extend(output)
            outputs["label"].extend(example["output"])
            outputs["dataset_name"].extend(example["dataset_name"])

        else:
            # append single elements
            outputs["prediction"].append(output)
            outputs["label"].append(example["output"])
            outputs["dataset_name"].append(example["dataset_name"])

    # Compute metrics
    metrics = compute_metrics(pd.DataFrame(outputs), verbose=True)

    # Save metrics to a json file
    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")

    # Print metrics
    logger.info("Metrics:")
    logger.info(metrics)
