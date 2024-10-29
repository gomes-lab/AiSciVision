import json
import os

import torch
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, Sum


class BinaryMetrics:
    def __init__(self):
        self.num_total = Sum()
        self.num_correct = Sum()

        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

    def __repr__(self) -> str:
        return str(self._metrics2dict())

    def reset(self) -> None:
        self.num_total.reset()
        self.num_correct.reset()

        self.accuracy.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_cpu = preds.to("cpu")
        target_cpu = target.to("cpu")

        self.num_total.update(torch.ones_like(preds_cpu))
        self.num_correct.update(preds_cpu == target_cpu)

        self.accuracy.update(preds_cpu, target_cpu)
        self.f1.update(preds_cpu, target_cpu)
        self.precision.update(preds_cpu, target_cpu)
        self.recall.update(preds_cpu, target_cpu)

    def _metrics2dict(self) -> dict[str, float | int]:
        return {
            "num_total": self.num_total.compute().item(),
            "num_correct": self.num_correct.compute().item(),
            "accuracy": self.accuracy.compute().item(),
            "f1": self.f1.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
        }

    def save_metrics(self, save_path: str) -> None:
        assert os.path.splitext(save_path)[1] == ".json"

        with open(save_path, "w") as f:
            json.dump(self._metrics2dict(), f, indent=4, sort_keys=True)
