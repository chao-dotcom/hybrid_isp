"""Stage 7 - Classical spatial denoise (bilateral filter).

Demosaicing and gamma amplify the sensor noise, so the ISP tail denoises. We use
an edge-preserving *bilateral* filter: a Gaussian in space weighted by a Gaussian
in intensity difference, so it averages flat regions but not across edges (unlike
a plain Gaussian blur, which would smear detail).

This is the classical counterpart to the learned CNN denoiser in ``learned/``;
run_smoke compares the two on identical inputs.
"""
import numpy as np

try:
    from skimage.restoration import denoise_bilateral
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


def bilateral(rgb, sigma_color=0.08, sigma_spatial=2.0):
    """Edge-preserving denoise. Uses skimage if available, else a manual fallback."""
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)
    if _HAVE_SK:
        return np.clip(
            denoise_bilateral(rgb, sigma_color=sigma_color,
                              sigma_spatial=sigma_spatial, channel_axis=-1),
            0.0, 1.0)
    return _bilateral_manual(rgb, sigma_color, sigma_spatial)


def _bilateral_manual(rgb, sigma_color, sigma_spatial, radius=2):
    """Small dependency-free bilateral (used only if skimage is missing)."""
    h, w, c = rgb.shape
    pad = radius
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(rgb)
    wsum = np.zeros((h, w, 1))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w, :]
            spatial = np.exp(-(dy * dy + dx * dx) / (2 * sigma_spatial ** 2))
            diff = shifted - rgb
            color = np.exp(-(diff ** 2) / (2 * sigma_color ** 2))
            wgt = spatial * color
            out += wgt * shifted
            wsum += wgt.mean(axis=2, keepdims=True) * 0 + wgt.sum(axis=2, keepdims=True) / c
    return np.clip(out / np.maximum(wsum, 1e-8), 0.0, 1.0)
