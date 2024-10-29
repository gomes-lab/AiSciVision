from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from metrics import BinaryMetrics

from . import ImageClassifier


def clip_transform(clip_processor: CLIPProcessor, images: torch.Tensor) -> torch.Tensor:
    # Get embeddings for images
    images = [transforms.ToPILImage()(image) for image in images]
    images: dict[str, Any] = clip_processor(
        text=None,
        images=images,
        return_tensors="pt",
    )
    return images


class CLIPZeroShotImageClassifier(ImageClassifier):
    def __init__(
        self,
        clip_model_config: str,
        label_names: list[str],
    ):
        super().__init__()
        self.clip_model_config = clip_model_config
        self.label_names = label_names

        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_config)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_config)

    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader, metrics: BinaryMetrics, device: str = "cpu", **kwargs
    ) -> None:
        device = "cpu"
        # Get embeddings for labels
        label_tokens = self.clip_processor(
            text=self.label_names,
            padding=True,
            images=None,
            return_tensors="pt",
        ).to(device)
        self.label_embeddings = self.clip_model.get_text_features(**label_tokens)
        self.label_embeddings = self.label_embeddings.detach()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: BinaryMetrics,
        number_to_test: int | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> dict[str, list]:
        metrics.reset()

        all_proba = []
        all_labels = []
        all_ids = []

        number_to_test = number_to_test or len(dataloader)  # Default test all of them

        # Iterate through data and evaluate
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating...")
        with torch.inference_mode():
            for batch_idx, (no_transformed_images, _, labels, _) in pbar:
                # Set to correct device
                no_transformed_images, labels = no_transformed_images.to(device), labels.to(device)

                images = clip_transform(self.clip_processor, no_transformed_images)

                image_embeddings = self.clip_model.get_image_features(**images)
                image_embeddings = image_embeddings.detach()

                sims = torch.cosine_similarity(
                    image_embeddings[:, None, :], self.label_embeddings[None, :, :], dim=-1
                )  # shape (#images, #labels)
                probs = torch.softmax(sims, dim=1)[:, 1]  # Probability of positive class

                all_proba.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(range(batch_idx * labels.size(0) + 1, (batch_idx + 1) * labels.size(0) + 1))

                if len(all_proba) >= number_to_test:
                    break

        # Limit to number_to_test
        all_proba = all_proba[:number_to_test]
        all_labels = all_labels[:number_to_test]
        all_ids = all_ids[:number_to_test]

        # Create the result dictionary
        result = {
            "ids": [int(id) for id in all_ids],  # Ensure ids are integers
            "clip_zero_preds": [
                {
                    "class": int(1 if p >= 0.5 else 0),  # Ensure class is an integer
                    "probability": float(p),  # Ensure probability is a float
                }
                for p in all_proba
            ],
            "ground_truths": [int(label) for label in all_labels],  # Ensure labels are integers
        }

        # Update metrics
        preds = torch.tensor([1 if p >= 0.5 else 0 for p in all_proba], dtype=torch.long)
        labels = torch.tensor(all_labels, dtype=torch.long)
        metrics.update(preds, labels)

        return result

    def save_model(self, model_path: str) -> None:
        print("No need to save CLIP Zero Shot Classifier.")

    @classmethod
    def load_model(cls, model_path: str) -> CLIPZeroShotImageClassifier:
        return torch.load(model_path)


class CLIPMLPModel(nn.Module):
    def __init__(self, clip_model_config: str, num_labels: int):
        super().__init__()
        self.clip_model_config = clip_model_config
        self.num_labels = num_labels

        # Setup CLIP model
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_config)

        # Setup classifier
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_labels),
        )

    def forward(self, x: dict) -> torch.Tensor:
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**x)

        logits = self.mlp(image_embeddings)
        return logits


class CLIPMLPImageClassifier(ImageClassifier):
    def __init__(
        self,
        clip_model_config: str,
        num_labels: int,
    ):
        super().__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_config)
        self.clip_mlp_model = CLIPMLPModel(clip_model_config, num_labels)

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: BinaryMetrics,
        loss_function,
        optimizer,
        epoch: int,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        # Setup for training
        metrics.reset()
        self.clip_mlp_model.to(device)
        self.clip_mlp_model.train()

        running_loss = 0.0

        # Iterate through data and train
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train epoch {}".format(epoch))
        for batch_idx, (no_transformed_images, _, labels, _) in pbar:
            # Set to correct device
            no_transformed_images, labels = no_transformed_images.to(device), labels.to(device)

            images = clip_transform(self.clip_processor, no_transformed_images)
            images = images.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = self.clip_mlp_model(images)
            loss = loss_function(logits, labels)

            # Optimization step
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            metrics.update(preds, labels)

            # Update progress bar
            pbar.set_postfix({"loss": running_loss / (batch_idx + 1.0), "acc": metrics.accuracy.compute().item()})

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: BinaryMetrics,
        loss_function,
        number_to_test: int | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> dict[str, list]:
        metrics.reset()
        self.clip_mlp_model.to(device)
        self.clip_mlp_model.eval()

        all_proba = []
        all_labels = []
        all_ids = []

        running_loss = 0.0
        number_to_test = number_to_test or len(dataloader)  # Default test all of them

        # Iterate through data and evaluate
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating...")
        with torch.inference_mode():
            for batch_idx, (no_transformed_images, _, labels, _) in pbar:
                # Set to correct device
                no_transformed_images, labels = no_transformed_images.to(device), labels.to(device)

                images = clip_transform(self.clip_processor, no_transformed_images)
                images = images.to(device)

                # Forward pass
                logits = self.clip_mlp_model(images)
                loss = loss_function(logits, labels)

                # Update metrics
                running_loss += loss.item()
                # preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class

                all_proba.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(range(batch_idx * labels.size(0) + 1, (batch_idx + 1) * labels.size(0) + 1))

                if len(all_proba) >= number_to_test:
                    break

        # Limit to number_to_test
        all_proba = all_proba[:number_to_test]
        all_labels = all_labels[:number_to_test]
        all_ids = all_ids[:number_to_test]

        # Create the result dictionary
        result = {
            "ids": [int(id) for id in all_ids],  # Ensure ids are integers
            "clip_supervised": [
                {
                    "class": int(1 if p >= 0.5 else 0),  # Ensure class is an integer
                    "probability": float(p),  # Ensure probability is a float
                }
                for p in all_proba
            ],
            "ground_truths": [int(label) for label in all_labels],  # Ensure labels are integers
        }

        # Update metrics
        preds = torch.tensor([1 if p >= 0.5 else 0 for p in all_proba], dtype=torch.long)
        labels = torch.tensor(all_labels, dtype=torch.long)
        metrics.update(preds, labels)

        return result

    def predict_proba(self, image: Image.Image, device: str = "cpu") -> float:
        self.clip_mlp_model.to(device)
        self.clip_mlp_model.eval()

        # Transform the image
        image_tensor = clip_transform(self.clip_processor, transforms.ToTensor()(image).unsqueeze(0).to(device))

        # Forward pass
        with torch.no_grad():
            logits = self.clip_mlp_model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        # Return probability of positive class (assuming binary classification)
        return probabilities[0, 1].item() * 100.0

    def save_model(self, model_path: str) -> None:
        torch.save(self.clip_mlp_model.state_dict(), model_path)

    @classmethod
    def load_model(cls, model_path: str, model_init_kwargs: dict) -> CLIPMLPImageClassifier:
        # self.clip_mlp_model.load_state_dict(torch.load(model_path))
        obj = cls(**model_init_kwargs)
        obj.clip_mlp_model.load_state_dict(torch.load(model_path))
        return obj

    def load_model_inplace(self, model_path: str) -> CLIPMLPImageClassifier:
        if not torch.cuda.is_available():
            self.clip_mlp_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        else:
            self.clip_mlp_model.load_state_dict(torch.load(model_path))
