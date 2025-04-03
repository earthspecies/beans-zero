import argparse
import json
import logging

import pandas as pd
from datasets import load_dataset
from evaluate import compute_metrics
from model import ModelWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("beans_zero")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Load the BEANS-Zero dataset.")
    parser.add_argument("--streaming", action="store_true", help="Whether to stream the dataset.")
    parser.add_argument("--batched", action="store_true", help="Whether to batch the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--as_torch", action="store_true", help="Whether to load the dataset as a torch dataset.")
    parser.add_argument(
        "--output_path", type=str, default="metrics.json", help="Path to save the metrics output as a json file."
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function to load the BEANS-Zero dataset, iterate over it, perform inference with a model
    and compute metrics.

    This function handles command line arguments, loads the dataset, and processes it in batches.
    """

    args = parse_args()
    # Load the dataset
    dataset = load_dataset("EarthSpeciesProject/BEANS-Zero", streaming=args.streaming, split="train")
    if args.batched:
        dataset = dataset.batch(batch_size=args.batch_size)

    if args.as_torch:
        # Load the dataset as a torch dataset
        dataset = dataset.with_format("torch")

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Print info about the dataset
    logger.info("Dataset loaded. Info:")
    logger.info(dataset)

    logger.info("Loading your model...")
    model = ModelWrapper()

    outputs = {"prediction": [], "label": [], "dataset_name": []}
    for _, example in tqdm(enumerate(dl)):
        # example can either be a dict of arrays or
        # a dict of single elements depending on batched True/False
        output = model(example)

        if args.batched:
            # output is a list so extend
            outputs["prediction"].extend(output)
            outputs["label"].extend(example["label"])
            outputs["dataset_name"].extend(example["dataset_name"])

        else:
            # append single elements
            outputs["prediction"].append(output)
            outputs["label"].append(example["label"])
            outputs["dataset_name"].append(example["dataset_name"])

    # Compute metrics
    metrics = compute_metrics(pd.DataFrame(outputs), verbose=True)

    # Save metrics to a json file
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {args.output_path}")

    # Print metrics
    logger.info("Metrics:")
    logger.info(metrics)


if __name__ == "__main__":
    main()
