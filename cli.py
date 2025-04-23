import json
import logging
from pathlib import Path
import click
import pandas as pd
from datasets import load_dataset
from beans_zero.evaluate import compute_metrics
from beans_zero.pretty_print_dataset import print_component, print_component_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("beans_zero")

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """BEANS-Zero: Zero-shot evaluation for audio-text bioacoustics models.

    \b
    Available commands:
      * evaluate - Evaluate predictions against ground truth
      * fetch - Download the BEANS-Zero dataset
      * info - List all component datasets or,
        get info on a specific component of the dataset

    \b
    For detailed help on each command, run:
      python cli.py <command> --help

    \b
    Examples:
      beans-evaluate predictions.csv metrics.json
      beans-fetch
      beans-info zf-indiv
    """
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
@click.argument("component_name", type=str, default=None, required=False)
def info(component_name: str) -> None:
    """Print a specific component of the dataset.

    Parameters
    ----------
    component_name : str
        The name of the component to print.
    """
    if component_name:
        print_component(component_name)
    else:
        print_component_list()


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


if __name__ == "__main__":
    cli()
