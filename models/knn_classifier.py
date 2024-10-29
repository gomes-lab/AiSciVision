from __future__ import annotations

import pickle
from dataclasses import dataclass

import torch
from sklearn.neighbors import KNeighborsClassifier

import embeddingModel
from metrics import BinaryMetrics

from . import ImageClassifier


@dataclass
class SeparateImageLabelDataset:
    """
    An object separately stores images and labels of a dataset.
    """

    no_transformed_images: torch.Tensor
    embeddings: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        embedding_model: embeddingModel.EmbeddingModel,
    ) -> SeparateImageLabelDataset:
        all_no_transformed_images = []
        all_embeddings = []
        all_labels = []
        for no_transformed_images, _, labels, _ in dataloader:
            all_no_transformed_images.append(no_transformed_images)
            with torch.no_grad():
                all_embeddings.append(embedding_model.embed(no_transformed_images))
            all_labels.append(labels)

        return cls(
            no_transformed_images=torch.cat(all_no_transformed_images),
            embeddings=torch.cat(all_embeddings),
            labels=torch.cat(all_labels),
        )


class KNNImageClassifier(ImageClassifier):
    def __init__(self, n_neighbors: int, embedding_model: embeddingModel.EmbeddingModel):
        """
        Initialize a k-Nearest Neighbor classifier with specified k=n_neighbors.

        Args:
            n_neighbors (int): Number of neighbors used for inference.
            embedding_model (embeddingModel.EmbeddingModel): Embedding model to use.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.embedding_model = embedding_model
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, metrics: BinaryMetrics, **kwargs) -> None:
        # Separate images and labels
        dataset = SeparateImageLabelDataset.from_dataloader(dataloader, self.embedding_model)
        self.clf.fit(dataset.embeddings.numpy(), dataset.labels.numpy())

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: BinaryMetrics,
        number_to_test: int | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """

        Args:
            number_to_test (int | None, optional): Number of examples to evaluate, sliced from the beginning of the dataset.
                Defaults to None.

        Returns:
            dict[str, list]: Dictionary of test example ids, classifier predictions, and ground truth labels.
        """
        metrics.reset()

        # Separate images and labels
        dataset = SeparateImageLabelDataset.from_dataloader(dataloader, self.embedding_model)

        # Limit the number of samples to evaluate
        number_to_test = number_to_test or dataset.embeddings.size(0)  # Default test all of them
        limited_embeddings = dataset.embeddings[:number_to_test]
        limited_labels = dataset.labels[:number_to_test]

        # Get probabilities for the positive class
        proba = self.clf.predict_proba(limited_embeddings.numpy())[:, 1]

        # Create the result dictionary
        result = {
            "ids": list(range(1, number_to_test + 1)),
            "knn_preds": [{"class": 1 if p >= 0.5 else 0, "probability": float(p)} for p in proba],
            "ground_truths": limited_labels.numpy().tolist(),
        }

        # Update metrics
        preds = torch.Tensor([1 if p >= 0.5 else 0 for p in proba])
        metrics.update(preds, limited_labels)

        return result

    def save_model(self, model_path: str) -> None:
        with open(model_path, "wb") as f:
            pickle.dump(self.clf, f)

    @classmethod
    def load_model(cls, model_path: str) -> KNNImageClassifier:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        return obj
