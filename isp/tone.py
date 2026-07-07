"""Stage 6 - Tone / gamma.

Everything upstream works in *linear* light. Displays and human perception are
non-linear, so the final encode applies the sRGB opto-electronic transfer
function (a ~1/2.4 gamma with a small linear toe). This must come after WB/CCM
(which are linear operations) and, in our pipeline, before the perceptual
denoise so the denoiser sees display-referred noise like a real ISP tail.
"""
import numpy as np


def linear_to_srgb(rgb):
    """Encode linear [0,1] -> sRGB gamma [0,1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    a = 0.055
    lo = rgb <= 0.0031308
    out = np.where(lo, rgb * 12.92, (1 + a) * np.power(rgb, 1 / 2.4) - a)
    return np.clip(out, 0.0, 1.0)


def srgb_to_linear(rgb):
    """Decode sRGB gamma [0,1] -> linear [0,1] (inverse of the above)."""
    rgb = np.clip(rgb, 0.0, 1.0)
    a = 0.055
    lo = rgb <= 0.04045
    out = np.where(lo, rgb / 12.92, np.power((rgb + a) / (1 + a), 2.4))
    return np.clip(out, 0.0, 1.0)


def apply_gamma(rgb, mode="srgb", gamma=2.2):
    if mode == "srgb":
        return linear_to_srgb(rgb)
    if mode == "power":
        return np.clip(np.power(np.clip(rgb, 0, 1), 1.0 / gamma), 0, 1)
    if mode in ("none", None):
        return np.clip(rgb, 0, 1)
    raise ValueError(f"unknown gamma mode {mode!r}")
