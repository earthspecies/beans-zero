"""A utility that prints the info about a dataset"""

from beans_zero.config import beans_cfg
from rich import print


def print_component_list() -> None:
    """Print the components of the dataset."""
    print("Available dataset components:")
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

    Examples
    ---------
    >>> print_component("captioning")
    Dataset info:
      Name: captioning
      Description: Captioning the audio. Derived from iNaturalist.
      Labels: None
      License: per file licenses, please see individual files
      Sample rate (Hz): 16000
      Max duration (secs): 10
    >>> print_component("unknown")
    Component dataset 'unknown' not found.
    Available dataset components:
      Number of components: 22
      - esc50
      - watkins
      - cbi
      - humbugdb
      - enabirds
      - hiceas
      - rfcx
      - gibbons
      - dcase
      - lifestage
      - call-type
      - unseen-species-cmn
      - unseen-species-sci
      - unseen-species-tax
      - unseen-genus-cmn
      - unseen-genus-sci
      - unseen-genus-tax
      - unseen-family-cmn
      - unseen-family-sci
      - unseen-family-tax
      - captioning
      - zf-indiv
    """
    component_datasets = beans_cfg["metadata"]["components"]
    found = False
    for component in component_datasets:
        if component["name"] == dataset_name:
            print("Dataset info:")
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
