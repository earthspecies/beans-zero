"""A utility that prints the info about a dataset"""

from beans_zero.config import beans_cfg
from rich import print


def print_component_list() -> None:
    """Print the components of the dataset."""
    print("Dataset components:")
    component_datasets = beans_cfg["metadata"]["components"]
    print(f"  Number of components: {len(component_datasets)}")
    for component in component_datasets:
        print(f"  - {component['name']}")


def print_component(dataset_name: str) -> None:
    """Pretty print the dataset info.

    Arguments
    ---------
    dataset : str
        The dataset to be printed.
    """
    print("Dataset info:")
    component_datasets = beans_cfg["metadata"]["components"]
    found = False
    for component in component_datasets:
        if component["name"] == dataset_name:
            print(f"  Name: {component['name']}")
            print(f"  Description: {component['description']}")
            print(f"  Labels: {component.get('labels', 'None')}")
            print(f"  License: {component['license']}")
            print(f"  Sample rate (Hz): {component['sample_rate']}")
            print(f"  Max duration (secs): {component['max_duration']}")
            found = True
            break
    if not found:
        print(f"Component dataset '{dataset_name}' not found.")
        print_component_list()
