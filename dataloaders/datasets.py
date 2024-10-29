"""
Classes and functions to load image datasets.
"""

from __future__ import annotations

import abc
import json
import os
import random

import geobench
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
import utils


class ImageDataset(Dataset):
    def __init__(
        self, data_root_dir: str, split: str = "train", percent_labeled: int = 100, test_order_path: str = None
    ) -> None:
        f"""
        Args:
            split (str, optional): One of {config.data_split_choices}. Defaults to "train".
            percent_labeled (int, optional): Percentage of train data to be subsampled. Only used when split is train.
                Defaults to 100.
        """
        super().__init__()
        assert split in config.data_split_choices
        self.data_root_dir = data_root_dir
        self.split = split
        self.percent_labeled = percent_labeled
        self.test_order_path = test_order_path

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_label_names() -> list[str]:
        """
        Returns:
            list[str]: Label names for the image dataset (ordered).

        Example:
            >>> print(ImageDataset.label_names)
            ["label0", "label1"]
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_val_split(train_dataset: ImageDataset) -> tuple[ImageDataset, ImageDataset]:
        f"""
        Creates a validation dataset (val_dataset) given the train_dataset.

        We use the dataset's specified validation split when it exists.

        If the dataset doesn't specify a validation split, the val_dataset is {config.val_dataset_frac} fraction of the images
        in the folder `train_dataset.data_dir`. The function guarantees that val_dataset and train_dataset do not overlap.
        Images are only added to the validation split if they do not exist in the given train_dataset. As a result,
        the val_dataset is at most {config.val_dataset_frac} fraction of the images, 
        and **smaller** if `train_dataset.percent_labeled`% >= {1 - config.val_dataset_frac}.

        Args:
            train_dataset (ImageDataset): Training dataset.

        Returns:
            tuple[ImageDataset, ImageDataset]: train_dataset, val_dataset
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing four elements:
                - unormalized_image (torch.Tensor): The original image without normalization.
                    Shape: (C, H, W) where C is the number of channels, H is height, and W is width.
                - normalized_image (torch.Tensor): The normalized version of the image.
                    Shape: (C, H, W) where C is the number of channels, H is height, and W is width.
                - label (int): The class label for the image.
                - meta_data (dict): A dictionary containing additional information about the image.
                    This can include any relevant metadata for tooling methods.

        Example:
            >>> dataset = ImageDataset()
            >>> unorm_img, norm_img, label, meta = dataset[0]
            >>> print(unorm_img.shape, norm_img.shape, label, meta)
            torch.Size([3, 224, 224]) torch.Size([3, 224, 224]) 5 {'filename': 'image_001.jpg', 'capture_date': '2023-01-15'}
        """
        pass


class SolarDataset(ImageDataset):
    """
    Using Geobench https://github.com/ServiceNow/geo-bench accesses the py4vger dataset
    """

    def __init__(
        self, data_root_dir: str, split: str = "train", percent_labeled: int = 100, test_order_path: str = None
    ):
        super().__init__(data_root_dir, split, percent_labeled, test_order_path)
        self.transforms = self.get_transforms()
        self.to_tensor_transform = transforms.ToTensor()

        # Find correct task from geobench, and get split
        self.task_name = "pv4ger"
        task_hf_path = f"classification_v1.0/m-{self.task_name}"
        if not (geobench.GEO_BENCH_DIR / task_hf_path).exists():
            utils.download_geobench_tasks([f"{task_hf_path}.zip"])

        for task in geobench.task_iterator(benchmark_name="classification_v1.0"):
            if self.task_name in task.dataset_name:
                break
        self.dataset = task.get_dataset(split=self.split)

        # Subsample the dataset if it's the training split
        if self.split == "train":
            num_samples = len(self.dataset)
            num_labeled = int(num_samples * (self.percent_labeled / 100))
            indices = random.sample(range(num_samples), num_labeled)
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        # If test_order_path is provided, use it to subset/order the test set
        if self.test_order_path and self.split == "test":
            json_filename = os.path.join(self.test_order_path, "solar_test_indices.json")
            if os.path.exists(json_filename):
                with open(json_filename, "r") as json_file:
                    test_indices = json.load(json_file)["indices"]
                self.dataset = torch.utils.data.Subset(self.dataset, test_indices)
            else:
                print(f"Warning: {json_filename} not found. Using original test set order.")

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_val_split(train_dataset: SolarDataset) -> tuple[SolarDataset, SolarDataset]:
        # Geobench provides earmarked val splits, __init__ will figure it out
        val_dataset = SolarDataset(data_root_dir=train_dataset.data_root_dir, split="valid")
        return train_dataset, val_dataset

    def get_transforms(self) -> transforms.Compose:
        if self.split == "train":
            img_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    transforms.RandomVerticalFlip(),  # Random vertical flip
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
                ]
            )
        else:
            img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
                ]
            )
        return img_transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
        sample = self.dataset[index]
        img, _ = sample.pack_to_3d(band_names=("red", "green", "blue"))
        label = sample.label

        # Unnormalized image
        unorm_img = self.to_tensor_transform(img)

        # Normalize to be between 0-1
        img = img / img.max()
        im = Image.fromarray((img * 255).astype(np.uint8))

        # Normalized image
        norm_img = self.transforms(im)

        # Convert label to tensor of type long
        label = torch.tensor(label).long()

        return unorm_img, norm_img, label, {}

    @staticmethod
    def get_label_names() -> list[str]:
        return ["no solar panel", "solar panel"]


class AquacultureDataset(ImageDataset):
    """
    Dataset for aquaculture images.

    This dataset is designed to handle google earth images of ponds,
    and then classify them as being aquaculture ponds or not.
    """

    def __init__(self, data_root_dir: str, split="train", percent_labeled: int = 100, test_order_path: str = None):
        super().__init__(data_root_dir, split, percent_labeled, test_order_path)
        # Setup data location, and transforms
        dir_suff = "train" if self.split == "valid" else self.split
        self.data_dir = os.path.join(self.data_root_dir, f"aquaculture/{dir_suff}")
        self.transforms = self.get_transforms()
        self.to_tensor_transform = transforms.ToTensor()

        # Load images
        self.load_images()

    def get_transforms(self) -> transforms.Compose:
        if self.split == "train":
            img_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        return img_transform

    @staticmethod
    def get_label_names() -> list[str]:
        # Placeholder labels - adjust based on your specific aquaculture classes
        return ["no aquaculture pond", "aquaculture pond"]

    @staticmethod
    def get_val_split(train_dataset: AquacultureDataset) -> tuple[AquacultureDataset, AquacultureDataset]:
        val_dataset = AquacultureDataset(train_dataset.data_root_dir, split="valid")

        # AquacultureDataset.load_images() by default will load all training images as val_dataset.
        # Reload validation images to exclude images in train_dataset.

        # Get fixed size validation split, but only select from training images not in the training split
        all_image_names = [x for x in os.listdir(train_dataset.data_dir) if x.endswith(".png")]
        remaining_image_names = [x for x in all_image_names if x not in train_dataset.selected_image_names]

        num_samples = int(len(all_image_names) * config.val_dataset_frac)  # So val dataset is always of same size
        indices = torch.randperm(len(remaining_image_names))[
            :num_samples
        ]  # Only take images that are not in train images
        selected_val_image_names = [remaining_image_names[i] for i in indices]

        # Load validation and image labels
        images = []
        labels = []
        metadata = []
        for im in selected_val_image_names:
            img = Image.open(os.path.join(train_dataset.data_dir, im))
            images.append(img.copy())
            img.close()
            lab = 1 if "positive" in im else 0
            labels.append(lab)
            # Load json file
            with open(os.path.join(train_dataset.data_dir, im.replace(".png", ".json"))) as f:
                data = json.load(f)
                data["zoom"] = 19
                metadata.append(data)
        val_dataset.images = images
        val_dataset.labels = labels
        val_dataset.metadata = metadata

        return train_dataset, val_dataset

    def load_images(self) -> None:
        image_names = [x for x in os.listdir(self.data_dir) if x.endswith(".png")]

        if self.split == "train" and self.percent_labeled < 100:
            num_samples = int(len(image_names) * self.percent_labeled / 100.0)
            indices = torch.randperm(len(image_names))[:num_samples]
            self.selected_image_names = [image_names[i] for i in indices]
        elif self.split == "test" and self.test_order_path is not None:
            # Load the test indices
            with open(os.path.join(self.test_order_path, "aquaculture_test_indices.json"), "r") as f:
                test_indices = json.load(f)["indices"]
            # Reorder the images based on the test indices
            self.selected_image_names = [image_names[i] for i in test_indices]
        else:
            self.selected_image_names = image_names

        images = []
        labels = []
        metadata = []
        for im in self.selected_image_names:
            img = Image.open(os.path.join(self.data_dir, im))
            images.append(img.copy())
            img.close()
            lab = 1 if "positive" in im else 0
            labels.append(lab)
            # Load json file
            with open(os.path.join(self.data_dir, im.replace(".png", ".json"))) as f:
                data = json.load(f)
                data["zoom"] = 19
                metadata.append(data)
        self.images = images
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
        # Get original image
        orig_image = self.images[index].convert("RGB")

        # Make image just to tensor, no transforms
        no_transform_img = self.to_tensor_transform(orig_image)

        # Make image with transforms
        transformed_img = self.transforms(orig_image)

        # Get label
        label = self.labels[index]

        # Make label a tensor
        label = torch.tensor(label).long()

        # And return, in this case no metadata is needed
        return no_transform_img, transformed_img, label, self.metadata[index]


class EelgrassDataset(ImageDataset):
    """
    Takes the segmentation data set and turn it into classification

    By - gridding image up, originally images are 512 by 512, so will change to 64 by 64

    Only image that contain any eelgrass blade (normal or lesioned) will be included

    """

    def __init__(self, data_root_dir: str, split="train", percent_labeled: int = 100, test_order_path: str = None):
        super().__init__(data_root_dir, split, percent_labeled, test_order_path)
        # Setup data location, and transforms
        dir_suff = "train" if self.split == "valid" else self.split
        self.data_dir = os.path.join(self.data_root_dir, f"eelgrass/class_{dir_suff}")
        self.transforms = self.get_transforms()
        self.to_tensor_transform = transforms.ToTensor()

        # Load images
        self.load_images()

    def get_transforms(self) -> transforms.Compose:
        if self.split == "train":
            img_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    transforms.RandomVerticalFlip(),  # Random vertical flip
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
                ]
            )
        else:
            img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
                ]
            )
        return img_transform

    @staticmethod
    def get_label_names() -> list[str]:
        # Order of labels is consistent with data folder:
        # - Label 0 is for no disease
        # - Label 1 is for diseased plant
        return ["plant with no disease", "plant with disease"]

    @staticmethod
    def get_val_split(train_dataset: EelgrassDataset) -> tuple[EelgrassDataset, EelgrassDataset]:
        val_dataset = EelgrassDataset(train_dataset.data_root_dir, split="valid")

        # EelgrassDataset.load_images() by default will load all training images as val_dataset.
        # Reload validation images to exclude images in train_dataset.

        # Get fixed size validation split, but only select from training images not in the training split
        all_image_names = [x for x in os.listdir(train_dataset.data_dir)]
        remaining_image_names = [x for x in all_image_names if x not in train_dataset.selected_image_names]

        num_samples = int(len(all_image_names) * config.val_dataset_frac)  # So val dataset is always of same size
        indices = torch.randperm(len(remaining_image_names))[
            :num_samples
        ]  # Only take images that are not in train images
        selected_val_image_names = [remaining_image_names[i] for i in indices]

        # Load validation and image labels
        images = []
        labels = []
        for im in selected_val_image_names:
            img = Image.open(os.path.join(train_dataset.data_dir, im))
            images.append(img.copy())
            img.close()
            labels.append(int(im.split("_")[-1].split(".")[0]))

        val_dataset.images = images
        val_dataset.labels = labels

        return train_dataset, val_dataset

    def load_images(self) -> None:
        image_names = [x for x in os.listdir(self.data_dir)]

        if self.split == "train" and self.percent_labeled < 100:
            num_samples = int(len(image_names) * self.percent_labeled / 100.0)
            indices = torch.randperm(len(image_names))[:num_samples]
            self.selected_image_names = [image_names[i] for i in indices]
        elif self.split == "test" and self.test_order_path is not None:
            with open(os.path.join(self.test_order_path, "eelgrass_test_indices.json"), "r") as f:
                test_order = json.load(f)["indices"]
            self.selected_image_names = [image_names[i] for i in test_order]
        else:
            self.selected_image_names = image_names

        images = []
        labels = []
        for im in self.selected_image_names:
            img = Image.open(os.path.join(self.data_dir, im))
            images.append(img.copy())
            img.close()
            labels.append(int(im.split("_")[-1].split(".")[0]))

        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
        # Get original image
        orig_image = self.images[index]

        # Make image just to tensor, no transforms
        no_transform_img = self.to_tensor_transform(orig_image)

        # Make image with transforms
        transformed_img = self.transforms(orig_image)

        # Get label
        label = self.labels[index]

        # Make label a tensor
        label = torch.tensor(label).long()

        # And return, in this case no metadata is needed
        return no_transform_img, transformed_img, label, {}


dataset_name2ImageDatasetClass: dict[str, ImageDataset] = {
    "eelgrass": EelgrassDataset,
    "solar": SolarDataset,
    "aquaculture": AquacultureDataset,
}


def get_dataloaders(
    data_root_dir: str,
    dataset_name: str,
    batch_size: int,
    percent_labeled: int = 100,
    test_order_path: str = None,
    use_val_dataset: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """
    Returns a training, validation, and test dataloader for the specified dataset.
    If use_val_dataset is False, validation dataloader is None.

    Args:
        dataset_name (str): The name of the dataset to load.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to run for data loading.
    """
    assert dataset_name in dataset_name2ImageDatasetClass.keys()

    # Load the dataset
    image_dataset_cls = dataset_name2ImageDatasetClass[dataset_name]
    train_dataset: ImageDataset = image_dataset_cls(data_root_dir, split="train", percent_labeled=percent_labeled)
    test_dataset: ImageDataset = image_dataset_cls(data_root_dir, split="test", test_order_path=test_order_path)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if use_val_dataset:
        train_dataset, val_dataset = image_dataset_cls.get_val_split(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_loader = None

    return train_loader, val_loader, test_loader
