import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch) -> float:
        return (logits.argmax(dim=-1) == labels).float().mean().item()
