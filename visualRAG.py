"""
The VisualRAG is an object that can provide additional image context
to the LMM in order to make a classification.

The class will be instantied with the full training set, and an image embedding model, and is then free
to construct the VisualRAG in any format.

When called, the method should take an input image (an image to be classified) and return
a dictionary where the key is the identifier for the type of context, and the value is the image.

How the images should be presented/sent to the LMM needs to be setup in the promptSchema, and thus
the prompt schema needs to know how to handle the specific identifiers for each VisualRAG method being
tested.

In our case, we are only concerned with no additional visual context, and the positive/negative
visual context.
"""

import abc

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embeddingModel import EmbeddingModel


class VisualRAG(abc.ABC):
    """
    Abstract class for a VisualRAG system.
    """

    def __init__(self, train_dataloader: DataLoader, embedding_model: EmbeddingModel):
        self.train_dataloader = train_dataloader
        self.embedding_model = embedding_model

    @abc.abstractmethod
    def get_context(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """
        Returns a list of tuples, where the first element in the tuple is an image (from the training set)
        and the second element is an identifier for what the image is (postiveExample, negativeExample,
        etc.)

        Args:
            image (torch.Tensor): The image to provide context to.

        Returns:
            dict[str, torch.Tensor | None]: A mapping from image "type names" to the images from the training set.
                "Type names" can be "positiveExample, negativeExample".
                Some mappings might not contain a value, in which case the key is mapped to None.
        """
        pass


class NoContextVisualRAG(VisualRAG):
    """
    VisualRAG that provides no additional context.
    """

    def __init__(self, train_dataloader: DataLoader, embedding_model: EmbeddingModel):
        super().__init__(train_dataloader, embedding_model)

    def get_context(self, image):
        return {"noContext": None}


class PositiveNegativeVisualRAG(VisualRAG):
    """
    VisualRAG that provides positive and negative context.
    """

    def __init__(self, train_dataloader: DataLoader, embedding_model: EmbeddingModel):
        super().__init__(train_dataloader, embedding_model)
        self.positive_examples = []
        self.negative_examples = []
        self.positive_embeddings = []
        self.negative_embeddings = []
        self._preprocess_training_data()

    def _preprocess_training_data(self):
        """
        Separates positive and negative label images and applies the embedding model
        to create `self.positive_embeddings` and `self.negative_embeddings` lists.
        """
        positive_embeddings = []
        negative_embeddings = []
        total_batches = len(self.train_dataloader)
        with tqdm(total=total_batches, desc="Preprocessing training data") as pbar:
            for unnormalized_images, _, labels, _ in self.train_dataloader:
                for image, label in zip(unnormalized_images, labels):
                    embedding = self.embedding_model.embed(image.unsqueeze(0))
                    if label == 1:
                        self.positive_examples.append(image)
                        positive_embeddings.append(embedding)
                    else:
                        self.negative_examples.append(image)
                        negative_embeddings.append(embedding)
                pbar.update(1)

        self.positive_embeddings = torch.cat(positive_embeddings, dim=0)
        self.negative_embeddings = torch.cat(negative_embeddings, dim=0)

    def get_context(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        query_embedding = self.embedding_model.embed(image)

        positive_similarities = [
            torch.cosine_similarity(query_embedding, pos_emb, dim=1) for pos_emb in self.positive_embeddings
        ]
        negative_similarities = [
            torch.cosine_similarity(query_embedding, neg_emb, dim=1) for neg_emb in self.negative_embeddings
        ]

        most_similar_positive_idx = max(range(len(positive_similarities)), key=lambda i: positive_similarities[i])
        most_similar_negative_idx = max(range(len(negative_similarities)), key=lambda i: negative_similarities[i])

        return {
            "positiveExample": self.positive_examples[most_similar_positive_idx],
            "negativeExample": self.negative_examples[most_similar_negative_idx],
        }


rag_type2VisualRAGClass: dict[str, VisualRAG] = {
    "NoContext": NoContextVisualRAG,
    "PositiveNegative": PositiveNegativeVisualRAG,
}


def get_visual_rag(rag_type: str, train_dataloader: DataLoader, embedding_model: EmbeddingModel):
    """
    Factory function to get the appropriate VisualRAG based on the name.

    Args:
        name (str): The name of the VisualRAG to use.
        training_set (DataLoader): The training dataloader to use.
        embedding_model (EmbeddingModel): The embedding model to use.
    """
    assert rag_type in rag_type2VisualRAGClass.keys()

    visualRAG_cls = rag_type2VisualRAGClass[rag_type]
    return visualRAG_cls(train_dataloader, embedding_model)
