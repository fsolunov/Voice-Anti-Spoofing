import random

import torch
from torch import nn
from torch.nn import functional as F


def linear_triangular_filterbank(
    n_freqs,
    n_filters,
    sample_rate,
    f_min=0.0,
    f_max=None,
):
    if f_max is None:
        f_max = sample_rate / 2
    frequencies = torch.linspace(0.0, sample_rate / 2, n_freqs)
    edges = torch.linspace(f_min, f_max, n_filters + 2)
    filters = []
    for index in range(n_filters):
        left, center, right = edges[index : index + 3]
        rising = (frequencies - left) / (center - left)
        falling = (right - frequencies) / (right - center)
        filters.append(torch.clamp(torch.minimum(rising, falling), min=0.0))
    return torch.stack(filters)


def delta_kernel(win_length=5):
    half = win_length // 2
    offsets = torch.arange(-half, half + 1, dtype=torch.float32)
    return offsets / offsets.square().sum()


def compute_deltas(features: torch.Tensor, kernel: torch.Tensor):
    half = kernel.numel() // 2
    padded = F.pad(features.unsqueeze(0), (half, half), mode="replicate")
    weight = kernel.flip(0).view(1, 1, -1).expand(features.shape[0], 1, -1)
    return F.conv1d(padded, weight, groups=features.shape[0]).squeeze(0)


def dct_matrix(n_out, n_in):
    steps = torch.arange(n_in, dtype=torch.float32)
    coefficients = torch.arange(n_out, dtype=torch.float32)
    matrix = torch.cos(
        torch.pi / n_in * (steps.unsqueeze(0) + 0.5) * coefficients.unsqueeze(1)
    )
    matrix *= (2.0 / n_in) ** 0.5
    matrix[0] *= 1.0 / (2.0**0.5)
    return matrix


class LFCCFrontend(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        num_samples=64600,
        random_crop=False,
        n_fft=512,
        win_length=320,
        hop_length=160,
        n_filters=20,
        n_lfcc=20,
        f_min=0.0,
        f_max=8000.0,
        with_energy=True,
        with_deltas=True,
        delta_window=5,
        eps=1e-10,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.random_crop = random_crop
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.with_energy = with_energy
        self.with_deltas = with_deltas
        self.eps = eps

        self.register_buffer("window", torch.hamming_window(win_length))
        self.register_buffer(
            "filterbank",
            linear_triangular_filterbank(
                n_fft // 2 + 1, n_filters, sample_rate, f_min, f_max
            ),
        )
        self.register_buffer("dct", dct_matrix(n_lfcc, n_filters))
        self.register_buffer("delta_kernel", delta_kernel(delta_window))

    @property
    def feature_dim(self):
        return self.dct.shape[0] * (3 if self.with_deltas else 1)

    def fix_length(self, waveform: torch.Tensor):
        length = waveform.shape[-1]
        if length == 0:
            return waveform.new_zeros((1, self.num_samples))
        if length < self.num_samples:
            repeats = self.num_samples // length + 1
            waveform = waveform.repeat(1, repeats)
        length = waveform.shape[-1]
        if length > self.num_samples:
            offset = length - self.num_samples
            start = random.randint(0, offset) if self.random_crop else 0
            waveform = waveform[..., start : start + self.num_samples]
        return waveform

    def power_spectrum(self, waveform: torch.Tensor):
        spectrum = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return spectrum.real.square() + spectrum.imag.square()

    def forward(self, waveform: torch.Tensor):
        waveform = self.fix_length(waveform)
        power = self.power_spectrum(waveform)

        filtered = torch.log(self.filterbank @ power + self.eps)
        features = self.dct @ filtered
        if self.with_energy:
            energy = torch.log(power.sum(dim=0) + self.eps)
            features = torch.cat([energy.unsqueeze(0), features[1:]], dim=0)

        if self.with_deltas:
            delta = compute_deltas(features, self.delta_kernel)
            delta2 = compute_deltas(delta, self.delta_kernel)
            features = torch.cat([features, delta, delta2], dim=0)

        return features.transpose(0, 1).unsqueeze(0)
