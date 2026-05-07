"""Signal transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


STFTBackend = Literal["torch", "scipy"]
STFTOutput = Literal["magnitude", "power", "log_power"]


@dataclass(frozen=True)
class STFTTransform:
    n_fft: int = 128
    hop_length: int = 32
    window: str = "hann"
    output: STFTOutput = "log_power"
    backend: STFTBackend = "torch"
    center: bool = True

    def __post_init__(self) -> None:
        if self.window != "hann":
            raise ValueError("Only hann window is currently supported")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must be <= n_fft")

    def __call__(self, iq: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = self._to_tensor(iq)
        if tensor.shape != (2, tensor.shape[-1]):
            if tensor.ndim != 2 or tensor.shape[0] != 2:
                raise ValueError(f"Expected IQ tensor with shape (2, T), got {tuple(tensor.shape)}")

        complex_signal = torch.complex(tensor[0], tensor[1])
        if self.backend == "torch":
            spec = self._torch_stft(complex_signal)
        else:
            spec = self._scipy_stft(complex_signal)
        return self._postprocess(spec)

    def _to_tensor(self, iq: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(iq, torch.Tensor):
            return iq.detach().cpu().to(torch.float32)
        return torch.as_tensor(iq, dtype=torch.float32)

    def _torch_stft(self, complex_signal: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft)
        spec = torch.stft(
            complex_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=self.center,
            return_complex=True,
            onesided=False,
        )
        spec = torch.fft.fftshift(spec, dim=0)
        return spec

    def _scipy_stft(self, complex_signal: torch.Tensor) -> torch.Tensor:
        from scipy import signal

        _, _, spec = signal.stft(
            complex_signal.numpy(),
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            return_onesided=False,
            boundary="zeros" if self.center else None,
        )
        spec = np.fft.fftshift(spec, axes=0)
        return torch.from_numpy(spec)

    def _postprocess(self, spec: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(spec)
        if self.output == "magnitude":
            return magnitude.unsqueeze(0)
        power = magnitude.square()
        if self.output == "power":
            return power.unsqueeze(0)
        if self.output == "log_power":
            return torch.log10(power + 1e-8).unsqueeze(0)
        raise ValueError(f"Unsupported output type: {self.output}")
