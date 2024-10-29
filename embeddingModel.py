"""
The VisualRAG method requires a method to embed images into a vector space,
in order to comparisons/similarity scores between images.

When called it will be passed an un-normalized image tensor, i.e. its the raw pixel values.
Should the method require a normalized image, it should perform the normalization within the method.

"""

import abc
from typing import Any

import torch
from torchvision import transforms
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, CLIPVisionModel

import config


class EmbeddingModel(abc.ABC):
    """
    Abstract class for an embedding model.
    """

    @abc.abstractmethod
    def embed(self, image: torch.Tensor) -> torch.Tensor:
        """
        Embeds images into a vector space.

        Args:
            images (torch.Tensor): A batch of images to embed, tensor of shape (batch, channels, height, width).
                * Very important that these are un-normalized images *

        Returns:
            torch.Tensor: Embeddings of images.
        """
        pass


class RandomEmbeddingModel(EmbeddingModel):
    """
    Random embedding model, which returns a random embedding for images.
    """

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return torch.randn(images.size(0), 100)


class MeanEmbeddingModel(EmbeddingModel):
    """
    Mean embedding model, which returns a mean of the channels for images.
    """

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return torch.mean(images, axis=1).view(images.size(0), images.size(-1) ** 2)


class CLIPEmbeddingModel(EmbeddingModel):
    """
    CLIP embedding model, which returns the CLIP embedding for images.
    """

    def __init__(self, clip_model_config: str = config.clip_model_config) -> None:
        super().__init__()
        self.clip_model_config = clip_model_config
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_config)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_config)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        images = [transforms.ToPILImage()(image) for image in images]
        images: dict[str, Any] = self.clip_processor(
            text=None,
            images=images,
            return_tensors="pt",
        )
        embeddings = self.clip_model.get_image_features(**images)
        return embeddings


class CLIPVisionEmbeddingModel(EmbeddingModel):
    """
    CLIP vision embedding model, which returns the CLIP embedding for images.
    Uses CLIPVisionModel instead of CLIPModel from huggingface.
    """

    def __init__(self, clip_model_config: str = config.clip_model_config) -> None:
        super().__init__()
        self.clip_model_config = clip_model_config
        self.c_model = CLIPVisionModel.from_pretrained(self.clip_model_config)
        self.c_processor = AutoProcessor.from_pretrained(self.clip_model_config)

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        # Change the image (torch.tensor) to a PIL image
        image = transforms.ToPILImage()(image.squeeze())
        inputs = self.c_processor(images=image, return_tensors="pt")
        outputs = self.c_model(**inputs)
        pooled_output = outputs.pooler_output  # pooled CLS states
        pooled_output = pooled_output.detach().cpu()

        return pooled_output


embedding_name2model_cls: dict[str, EmbeddingModel] = {
    "random": RandomEmbeddingModel,
    "mean": MeanEmbeddingModel,
    "clip": CLIPEmbeddingModel,
    "clip_vision": CLIPVisionEmbeddingModel,
}


def get_embedding_model(name: str, embedding_model_init_kwargs: dict) -> EmbeddingModel:
    """
    Factory function to get the appropriate EmbeddingModel based on the name.

    Args:
        name (str): The name of the embedding model to use. Must be in `embedding_name2model_cls.keys()`

    Returns:
        EmbeddingModel: An instance of the appropriate EmbeddingModel subclass.

    Raises:
        AssertionError: If the specified embedding model is not implemented.
    """
    assert name in embedding_name2model_cls.keys(), f"Embedding model '{name}' is not implemented."
    return embedding_name2model_cls[name](**embedding_model_init_kwargs)
