"""Noise-power helpers shared by spectrum sensing components."""

from __future__ import annotations


def estimate_noise_power_from_observation(total_power: float, snr_db: float) -> float:
    """Estimate AWGN power from an observed signal-plus-noise sample.

    RadioML samples already include channel and noise impairments. If the stored
    sample power is interpreted as the total observed power

        P_total = P_signal + P_noise

    and the nominal SNR is

        SNR = P_signal / P_noise

    then the matching noise-only hypothesis should use

        P_noise = P_total / (1 + SNR_linear)

    rather than treating ``P_total`` as pure signal power.
    """

    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    return float(total_power / (1.0 + max(snr_linear, 1e-8)))
