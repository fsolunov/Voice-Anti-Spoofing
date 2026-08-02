import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(tensor, chunks=2, dim=self.dim)
        return torch.maximum(first, second)


class BLSTMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, output_dim // 2, bidirectional=True, batch_first=True
        )

    def forward(self, tensor: torch.Tensor):
        output, _ = self.lstm(tensor)
        return output


class LCNN(nn.Module):
    def __init__(
        self,
        feature_dim=60,
        num_classes=2,
        trunk_dropout=0.7,
        head_dropout=0.7,
        hidden_dim=160,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        reduced_dim = feature_dim // 16
        self.embedding_dim = 32 * reduced_dim

        mfm = MaxFeatureMap

        def pool():
            return nn.MaxPool2d(kernel_size=2, stride=2)

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            mfm(),
            pool(),
            nn.Conv2d(32, 64, kernel_size=1),
            mfm(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            mfm(),
            pool(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1),
            mfm(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            mfm(),
            pool(),
            nn.Conv2d(64, 128, kernel_size=1),
            mfm(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            mfm(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1),
            mfm(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            mfm(),
            pool(),
            nn.Dropout(trunk_dropout),
        )

        self.recurrent = nn.Sequential(
            BLSTMLayer(self.embedding_dim, self.embedding_dim),
            BLSTMLayer(self.embedding_dim, self.embedding_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            MaxFeatureMap(dim=1),
            nn.Dropout(head_dropout),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor, **batch):
        if features.dim() == 3:
            features = features.unsqueeze(1)

        hidden = self.trunk(features)
        batch_size, channels, frames, bins = hidden.shape
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch_size, frames, channels * bins)
        hidden = self.recurrent(hidden)
        hidden = hidden.mean(dim=1)

        logits = self.head(hidden)
        scores = logits[:, 1] - logits[:, 0]
        return {"logits": logits, "scores": scores}

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable}"
        return result_info
