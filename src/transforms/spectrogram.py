import random

import torch
from torch import nn
from torch.nn import functional as F


class FixedLengthPowerSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=320,
        hop_length=160,
        num_frames=750,
        random_crop=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.random_crop = random_crop
        self.register_buffer("window", torch.hamming_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < self.win_length:
            waveform = F.pad(waveform, (0, self.win_length - waveform.shape[-1]))

        complex_spectrum = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        # [frequency, time] -> [1, time, frequency] for the model's FC frontend.
        power = complex_spectrum.abs().square().transpose(0, 1).unsqueeze(0)
        current_frames = power.shape[1]
        if current_frames > self.num_frames:
            max_start = current_frames - self.num_frames
            start = random.randint(0, max_start) if self.random_crop else max_start // 2
            power = power[:, start : start + self.num_frames]
        elif current_frames < self.num_frames:
            power = F.pad(power, (0, 0, 0, self.num_frames - current_frames))

        return power
