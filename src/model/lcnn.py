import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(tensor, chunks=2, dim=self.dim)
        return torch.maximum(first, second)


def _linear_triangular_filterbank(
    n_fft_bins: int, n_filters: int, sample_rate: int
):
    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft_bins)
    edges = torch.linspace(0.0, sample_rate / 2, n_filters + 2)
    filters = []
    for index in range(n_filters):
        left, center, right = edges[index : index + 3]
        rising = (frequencies - left) / (center - left)
        falling = (right - frequencies) / (right - center)
        filters.append(torch.clamp(torch.minimum(rising, falling), min=0.0))
    return torch.stack(filters)


class LCNN(nn.Module):
    def __init__(
        self,
        n_fft_bins=257,
        n_filters=60,
        n_frames=750,
        sample_rate=16000,
        dropout=0.75,
        log_epsilon=1e-12,
    ):
        super().__init__()
        self.n_fft_bins = n_fft_bins
        self.n_filters = n_filters
        self.n_frames = n_frames
        self.log_epsilon = log_epsilon

        self.filterbank = nn.Linear(n_fft_bins, n_filters, bias=False)
        with torch.no_grad():
            self.filterbank.weight.copy_(
                _linear_triangular_filterbank(n_fft_bins, n_filters, sample_rate)
            )

        mfm = MaxFeatureMap

        def pool():
            return nn.MaxPool2d(kernel_size=2, stride=2)

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),
            mfm(),
            pool(),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            mfm(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1, bias=False),
            mfm(),
            pool(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            mfm(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1, bias=False),
            mfm(),
            pool(),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            mfm(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            mfm(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            mfm(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            mfm(),
            pool(),
        )

        final_time = n_frames
        final_frequency = n_filters
        for _ in range(4):
            final_time //= 2
            final_frequency //= 2
        flatten_size = 32 * final_time * final_frequency
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 160, bias=False),
            MaxFeatureMap(dim=1),
            nn.Dropout(p=dropout), 
            nn.BatchNorm1d(80),
            nn.Linear(80, 2),
        )
        self._initialize_backbone()

    def _initialize_backbone(self):
        for module in list(self.backbone.modules()) + list(self.classifier.modules()):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, **batch):
        expected_tail = (1, self.n_frames, self.n_fft_bins)
        features = self.filterbank(features.squeeze(1))
        features = torch.log10(features.clamp_min(self.log_epsilon)).unsqueeze(1)
        hidden = self.backbone(features).flatten(start_dim=1)
        logits = self.classifier(hidden)
        scores = logits[:, 1] - logits[:, 0]
        return {"logits": logits, "scores": scores}

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable}"

        return result_info
