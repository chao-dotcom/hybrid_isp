"""Stage 2 - Lens-shading correction (vignetting), optional.

Lenses transmit less light toward the frame corners (cos^4 falloff plus mount
mechanics), so a flat scene reads darker at the edges. LSC multiplies by the
inverse of a measured/monotone gain map. It runs after black-level (the falloff
is multiplicative on real signal, not on the pedestal) and before demosaic so
the correction is applied per-photosite.

Here we synthesise a smooth radial gain so the stage is runnable without a
calibration file; on real hardware you would load a per-unit LSC mesh.
"""
import numpy as np


def radial_gain_map(shape, strength=0.0):
    """Return a smooth radial gain map that brightens the corners.

    strength=0 -> all ones (no-op). strength~0.4 -> corners ~40% boosted.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r /= r.max() + 1e-8
    return 1.0 + strength * (r ** 2)


def correct(raw, gain_map=None, strength=0.0):
    """Apply lens-shading correction. If no gain_map given, synthesise one."""
    if strength <= 0 and gain_map is None:
        return raw
    if gain_map is None:
        gain_map = radial_gain_map(raw.shape, strength)
    return np.clip(raw * gain_map, 0.0, 1.0)
