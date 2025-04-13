import json
import logging
from pathlib import Path
import click
import pandas as pd
from datasets import load_dataset
from beans_zero.evaluate import compute_metrics
from beans_zero.benchmark import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("beans_zero")


@click.group()
def cli() -> None:
    """Command line interface for BEANS-Zero"""
    pass


@cli.command()
def fetch_dataset() -> None:
    """Download the BEANS-Zero dataset from Hugging Face Hub."""
    load_dataset(
        "EarthSpeciesProject/BEANS-Zero",
        split="test",
    )
    logger.info("BEANS-Zero dataset downloaded successfully.")


@cli.command()
@click.argument("predictions_file", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def evaluate(predictions_file: str, output_path: str) -> None:
    """Evaluate a predictions file.
    The predictions file must have the following columns:

    - dataset_name (str): The name of the component BEANS-Zero dataset
        for that row (sample).
    - prediction (str): The model's prediction for that sample,
         either a single string or comma-separated list
    - label (str): The ground truth label(s) for that sample

    'dataset_name'
    The predictions file can be in CSV or JSON or JSONL format.
    The output will be saved to the specified output path.

    Raises
    ------
    FileNotFoundError: If the predictions file does not exist.
    ValueError: If the predictions file is not in CSV or JSON format.

    """
    # Load the predictions file
    predictions_path = Path(predictions_file)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")

    if predictions_path.suffix == ".csv":
        outputs = pd.read_csv(predictions_path)
    elif predictions_path.suffix == ".json" or predictions_path.suffix == ".jsonl":
        logger.info("Reading jsonl as an array of records")
        outputs = pd.read_json(predictions_path, orient="records", lines=True)
    else:
        raise ValueError(
            """Predictions file must either be a CSV,
            or a 'jsonl' file in a records orientation.
            """
        )

    all_metrics = compute_metrics(outputs)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
            logger.info(f"Metrics saved to {output_path}")


@cli.command()
@click.option(
    "--path-to-model-module",
    required=True,
    help="Path to the model module. This should be a path to a Python file.",
)
@click.option(
    "--path-to-dataset",
    default="EarthSpeciesProject/BEANS-Zero",
    help="Path to the dataset, can be a path to a local dir.",
)
@click.option("--streaming", is_flag=True, help="Whether to stream the dataset.")
@click.option("--batched", is_flag=True, help="Whether to batch the dataset.")
@click.option(
    "--batch-size", type=int, default=32, help="The batch size if batched=True."
)
@click.option(
    "--as-torch",
    is_flag=True,
    help="Whether to load the dataset as a torch dataset.",
)
@click.option(
    "--output-path",
    default="metrics.json",
    help="Path to save the metrics output as a json file.",
)
def benchmark(
    path_to_model_module: str,
    path_to_dataset: str,
    streaming: bool,
    batched: bool,
    batch_size: int,
    as_torch: bool,
    output_path: str,
) -> None:
    """Run benchmark on your audio-text model against the BEANS-Zero dataset."""
    run_benchmark(
        path_to_model_module=path_to_model_module,
        path_to_dataset=path_to_dataset,
        streaming=streaming,
        batched=batched,
        batch_size=batch_size,
        as_torch=as_torch,
        output_path=output_path,
    )


if __name__ == "__main__":
    cli()
