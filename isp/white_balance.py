"""Stage 4 - Auto white balance (gray-world).

The scene illuminant tints the raw signal (tungsten -> orange, shade -> blue).
White balance rescales the R and B channels so a neutral surface reads neutral.
The classic assumption is *gray world*: the average of a natural scene is achromatic,
so we scale each channel to match the green channel's mean.

Order note: WB is applied in linear light (after demosaic, before the CCM and
gamma). Doing it after gamma would be wrong because the per-channel gains are
multiplicative on linear radiance.
"""
import numpy as np


def gray_world(rgb, clip=True):
    """Return WB'd image and the (gain_r, gain_g, gain_b) applied."""
    means = rgb.reshape(-1, 3).mean(axis=0) + 1e-8
    g = means[1]
    gains = np.array([g / means[0], 1.0, g / means[2]], np.float64)
    out = rgb * gains
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out, gains
