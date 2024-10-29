from __future__ import annotations

import abc

import torch

from metrics import BinaryMetrics


class ImageClassifier(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, metrics: BinaryMetrics, **kwargs) -> None:
        """
        Train the classifier for one epoch on the given dataloader, and update metrics.
        The given metrics is reset before training starts.

        Args:
            dataloader (torch.utils.data.DataLoader): An iterator for the dataset.
            metrics (BinaryMetrics): Object for logging metrics.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataloader: torch.utils.data.DataLoader, metrics: BinaryMetrics, **kwargs) -> None:
        """
        Evaluate the trained classifier on the given dataloader and update metrics.
        The given metrics is reset before evaluation starts.

        Args:
            dataloader (torch.utils.data.DataLoader): An iterator for the dataset.
            metrics (BinaryMetrics): Object for logging metrics.
        """
        pass

    @abc.abstractmethod
    def save_model(self, model_path: str) -> None:
        """
        Save the model at specified model_path.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load_model(cls, model_path: str) -> ImageClassifier:
        """
        Load the model at specified model_path, and return an object of this class.
        """
        pass
