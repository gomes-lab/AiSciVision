import argparse
import json
import os

import numpy as np

import config
from dataloaders.datasets import get_dataloaders


def create_test_set_selection(dataset_name: str, output_dir: str, seed: int = 1994):
    """
    Create and save random test set selections for each dataset.

    Args:
        dataset_name (str): Name of the dataset, must be one of `config.dataset_names`.
        output_dir (str): Directory where experiment outputs and metadata will be stored.
        seed (int): Seed for random number generation. Default is 1994.
    """
    assert dataset_name in config.dataset_names

    # Set random seed
    np.random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create metadata directory
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Get the test dataloader
    _, _, test_loader = get_dataloaders(data_root_dir="Data/", dataset_name=dataset_name, batch_size=1)

    # Get the total number of samples in the test set
    total_samples = len(test_loader.dataset)

    print(f"* {dataset_name=}, {total_samples=}")
    # Generate random indices
    indices = np.arange(total_samples).tolist()
    np.random.shuffle(indices)

    # Save indices to JSON file
    json_filename = os.path.join(metadata_dir, f"{dataset_name}_test_indices.json")
    with open(json_filename, "w") as json_file:
        json.dump({"indices": indices}, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test set selections for datasets.")
    parser.add_argument("--dataset_name", type=str, choices=config.dataset_names, help="Name of the dataset")
    parser.add_argument("--output_dir", type=str, help="Directory for experiment outputs and metadata.")
    parser.add_argument("--seed", type=int, default=1994, help="Seed for random number generation.")

    args = parser.parse_args()

    create_test_set_selection(args.dataset_name, args.output_dir, args.seed)
