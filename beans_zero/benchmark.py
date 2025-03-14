import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Load the BEANS-Zero dataset.")
    parser.add_argument("--streaming", action="store_true", help="Whether to stream the dataset.")
    parser.add_argument("--batched", action="store_true", help="Whether to batch the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--as_torch", action="store_true", help="Whether to load the dataset as a torch dataset.")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("EarthSpeciesProject/BEANS-Zero", streaming=args.streaming, split="train")
    if args.batched:
        dataset = dataset.batch(batch_size=args.batch_size)

    # Print the dataset
    print(dataset)

    for example in dataset:
        # example can either be a dict of arrays or a dict of single elements depending on batched True/False
        pass


if __name__ == "__main__":
    main()
