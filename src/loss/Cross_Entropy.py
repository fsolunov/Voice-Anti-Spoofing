import torch
from torch import nn
from torch.nn import functional


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer(
                "weight", torch.tensor(list(weight), dtype=torch.float)
            )
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        return {"loss": loss}
