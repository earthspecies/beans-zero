r"""Benchmark your audio-text model on the BEANS-Zero dataset.

1. Import the run_benchmark function into your model file.
2. Your model class / prediction function must be a Callable.
3. Call the run_benchmark function with your model class / prediction function
as an argument.

Here is an example of how to use the run_benchmark function:
```python
from beans_zero.benchmark import run_benchmark

class MyModel():
    "A simple example model class"
    def __init__(self):
        # Initialize your model here

    def predict(self, audio: torch.Tensor, text: str) -> torch.Tensor:
        "A simple forward pass."
        # Do something with the input tensor
        return x

    def __call__(self, example: dict) -> str:
        "A simple call method."
        # Extract the audio and text from the example
        audio = torch.Tensor(example["audio"])
        instruction = self.tokenizer(example["instruction_text"])

        # Perform inference
        prediction = self.predict(audio, instruction)
        return prediction

# Create an instance of your model
my_model = MyModel()
path_to_dataset = "EarthSpeciesProject/BEANS-Zero"

run_benchmark(
    model=my_model,
    path_to_dataset=path_to_dataset,
    streaming=True,
    batched=False,
    batch_size=0,
    output_path="metrics.json",
)
```
"""

import json
import logging
from typing import Callable
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from beans_zero.evaluate import compute_metrics

logger = logging.getLogger("beans_zero")


def _collate_fn(batch: list[dict]) -> dict[list]:
    # convert to a single dict with arrays
    # for each key in the batch
    collated = {}
    for key in batch[0].keys():
        collated[key] = [d[key] for d in batch]
    return collated


def run_benchmark(
    model: Callable,
    path_to_dataset: str | Path,
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
        model (Callable): The model class or prediction function.
            The model must be a callable function or class that takes
            a dictionary as input and returns a string as output if
            batched is False, or a list of strings if batched is True.
        path_to_dataset (str ): Path to the dataset.
        streaming (bool): Whether to stream the dataset.
        batched (bool): Whether to batch the dataset.
        batch_size (int): The batch size.
        output_path (str): Path to save the metrics output as a json file.

    Raises
    ------
        ValueError: If the model is not a callable function or class.
    """
    # check that the 'model' is a callable
    if not callable(model):
        raise ValueError("The 'model' must be a callable function or class.")

    # Check if the dataset path is local
    datapath = Path(path_to_dataset)
    if not datapath.exists():
        # Load the remote dataset
        logger.warning("""=====Loading dataset from huggingface hub====
        WARNING: This will need about 180 GB of space.
        Please check your ~/.cache/huggingface/datasets/downloads folder
        and remove  the downloaded cache chunks if you want to save space.
        If you want to save space, please use the streaming option.
        """)
        dataset = load_dataset(
            "EarthSpeciesProject/BEANS-Zero", streaming=streaming, split="test"
        )
    else:
        # Load the local dataset
        dataset = load_from_disk(path_to_dataset)

    # Print info about the dataset
    logger.info("Dataset loaded. Info:")
    logger.info(f"Dataset features: {dataset.features}")

    if batched:
        # If batched, use DataLoader for batching
        dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=_collate_fn,
        )

    outputs = {"prediction": [], "label": [], "dataset_name": []}
    for _, example in tqdm(enumerate(dataset)):
        # example can either be a dict of arrays or
        # a dict of single elements depending on batched True/False
        prediction = model(example)

        if batched and isinstance(prediction, list):
            if not isinstance(prediction[0], str):
                raise ValueError("The model must return a list of strings.")

        if not batched and not isinstance(prediction, str):
            raise ValueError("The model must return a string.")

        if batched and isinstance(prediction, list):
            # output is a list so extend
            outputs["prediction"].extend(prediction)
            outputs["label"].extend(example["output"])
            outputs["dataset_name"].extend(example["dataset_name"])
        else:
            # append single elements
            outputs["prediction"].append(prediction)
            outputs["label"].append(example["output"])
            outputs["dataset_name"].append(example["dataset_name"])

    # Compute metrics
    metrics = compute_metrics(pd.DataFrame(outputs), verbose=True)

    # Save metrics to a json file
    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")
