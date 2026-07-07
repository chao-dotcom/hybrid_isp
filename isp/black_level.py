"""Stage 1 - Black-level subtraction (and optional white-level normalisation).

Real sensors report a non-zero pedestal even in total darkness (the "optical
black" level) coming from the amplifier bias. If you demosaic/white-balance
before removing it, the pedestal leaks into colour ratios and tone mapping and
tints the shadows. So this is the *first* thing an ISP does on the RAW Bayer
frame, before any colour reasoning.
"""
import numpy as np


def subtract_black_level(raw, black_level=0.0, white_level=1.0):
    """Remove the sensor pedestal and rescale to [0, 1].

    Parameters
    ----------
    raw : np.ndarray
        Single-channel Bayer mosaic, float, in [0, white_level].
    black_level : float
        Optical-black pedestal in the same units as ``raw``.
    white_level : float
        Saturation level of the sensor (max valid code).

    Returns
    -------
    np.ndarray
        Float image clipped to [0, 1].
    """
    raw = raw.astype(np.float64)
    scale = max(white_level - black_level, 1e-8)
    out = (raw - black_level) / scale
    return np.clip(out, 0.0, 1.0)
