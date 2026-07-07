"""Stage 5 - Color correction matrix (CCM).

Sensor RGB primaries are not the display's primaries, so even after white
balance the colours are off (typically desaturated / hue-shifted). A 3x3 CCM
maps sensor RGB -> (linear) display RGB. Rows are usually normalised to sum to 1
so the matrix preserves neutral gray (doesn't change exposure/white point).

Applied in linear light, after WB and before gamma. Here we default to identity
because our synthetic RAW is generated in the same (sRGB-linear) space as the
ground truth, so a colour-space rotation would only add error. The stage is
kept modular and documented so a real per-sensor CCM can be dropped in.
"""
import numpy as np

IDENTITY = np.eye(3, dtype=np.float64)


def apply_ccm(rgb, matrix=None, clip=True):
    """Apply a 3x3 CCM (row-vector convention: out = rgb @ M.T)."""
    if matrix is None:
        matrix = IDENTITY
    matrix = np.asarray(matrix, np.float64)
    flat = rgb.reshape(-1, 3) @ matrix.T
    out = flat.reshape(rgb.shape)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out
