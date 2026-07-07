"""Stage 3 - Demosaic (CFA interpolation).

The sensor sees one colour per pixel through a Bayer colour-filter array (we use
the RGGB layout). Demosaicing reconstructs the two missing channels at every
site. This is the ISP stage with the biggest quality impact and the classic
interview topic, so we implement two variants:

    * bilinear         - separable linear interpolation. Cheap, but produces
                         zippering and colour fringing at edges because it
                         interpolates each channel independently.
    * Malvar-He-Cutler - the "high-quality linear" method (Malvar et al. 2004,
                         used in MATLAB's demosaic). It adds gradient-corrected
                         cross-channel terms: e.g. when interpolating G at an R
                         site it mixes in a Laplacian of R. Same cost class as
                         bilinear (5x5 FIR) but far fewer edge artefacts.

Bayer RGGB index map (row, col), 0-based:
    (0,0)=R  (0,1)=G
    (1,0)=G  (1,1)=B
"""
import numpy as np
from scipy.signal import convolve2d


# --- CFA masks --------------------------------------------------------------
def bayer_masks(shape, pattern="RGGB"):
    """Boolean masks (R, G, B) marking which sites carry which colour."""
    h, w = shape
    r = np.zeros((h, w), bool)
    g = np.zeros((h, w), bool)
    b = np.zeros((h, w), bool)
    # positions within the 2x2 tile for each colour
    pos = {
        "RGGB": {"R": (0, 0), "G": [(0, 1), (1, 0)], "B": (1, 1)},
    }[pattern]
    ry, rx = pos["R"]
    by, bx = pos["B"]
    r[ry::2, rx::2] = True
    b[by::2, bx::2] = True
    for gy, gx in pos["G"]:
        g[gy::2, gx::2] = True
    return r, g, b


def mosaic_to_channels(raw, pattern="RGGB"):
    """Split the 1-channel mosaic into 3 sparse channels (missing = 0)."""
    r_m, g_m, b_m = bayer_masks(raw.shape, pattern)
    R = raw * r_m
    G = raw * g_m
    B = raw * b_m
    return R, G, B, (r_m, g_m, b_m)


def _conv(x, k):
    return convolve2d(x, k, mode="same", boundary="symm")


# --- Bilinear ---------------------------------------------------------------
def demosaic_bilinear(raw, pattern="RGGB"):
    """Separable bilinear interpolation of each colour plane."""
    R, G, B, (r_m, g_m, b_m) = mosaic_to_channels(raw, pattern)

    # Green sits on a quincunx -> 4-neighbour average kernel.
    kg = np.array([[0, 1, 0],
                   [1, 4, 1],
                   [0, 1, 0]], np.float64) / 4.0
    # Red/Blue sit on a rectangular grid -> full bilinear kernel.
    krb = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]], np.float64) / 4.0

    Gf = _conv(G, kg)
    Rf = _conv(R, krb)
    Bf = _conv(B, krb)

    # Keep the truly-measured samples exact.
    Gf[g_m] = raw[g_m]
    Rf[r_m] = raw[r_m]
    Bf[b_m] = raw[b_m]

    out = np.stack([Rf, Gf, Bf], axis=-1)
    return np.clip(out, 0.0, 1.0)


# --- Malvar-He-Cutler (2004) -----------------------------------------------
# Filters from the paper, scaled by 1/8. Applied to the full mosaic (raw), then
# selected per site by CFA position.
_F = 1.0 / 8.0
# G at R/B locations
_G_AT_RB = _F * np.array([
    [0,  0, -1,  0,  0],
    [0,  0,  2,  0,  0],
    [-1, 2,  4,  2, -1],
    [0,  0,  2,  0,  0],
    [0,  0, -1,  0,  0]], np.float64)
# R at green in R row, B column  (and B at green in B row, R col by symmetry)
_RB_AT_G_RBROW = _F * np.array([
    [0,  0,  0.5, 0,  0],
    [0, -1,  0,  -1,  0],
    [-1, 4,  5,   4, -1],
    [0, -1,  0,  -1,  0],
    [0,  0,  0.5, 0,  0]], np.float64)
# R at green in B row, R column (and symmetric)
_RB_AT_G_BRROW = _RB_AT_G_RBROW.T.copy()
# R at blue location (and B at red location)
_RB_AT_BR = _F * np.array([
    [0,  0, -1.5, 0,  0],
    [0,  2,  0,   2,  0],
    [-1.5, 0, 6,  0, -1.5],
    [0,  2,  0,   2,  0],
    [0,  0, -1.5, 0,  0]], np.float64)


def demosaic_malvar(raw, pattern="RGGB"):
    """Malvar-He-Cutler high-quality linear demosaic (gradient-corrected)."""
    if pattern != "RGGB":
        raise NotImplementedError("MHC implemented for RGGB only")
    raw = raw.astype(np.float64)
    r_m, g_m, b_m = bayer_masks(raw.shape, pattern)

    # Start from the raw samples placed in their own channel.
    R = raw * r_m
    G = raw * g_m
    B = raw * b_m

    g_at_rb = _conv(raw, _G_AT_RB)
    rb_at_g_h = _conv(raw, _RB_AT_G_RBROW)   # interpolate along rows
    rb_at_g_v = _conv(raw, _RB_AT_G_BRROW)   # interpolate along columns
    rb_at_br = _conv(raw, _RB_AT_BR)

    # --- Green: fill at R and B sites --------------------------------------
    G = np.where(r_m | b_m, g_at_rb, G)

    # --- Red channel -------------------------------------------------------
    # Green sites split into two kinds for RGGB:
    #   row 0 (R row): green pixels flanked by R horizontally  -> use H filter
    #   row 1 (B row): green pixels flanked by R vertically    -> use V filter
    g_rrow = np.zeros_like(r_m)
    g_brow = np.zeros_like(r_m)
    g_rrow[0::2, 1::2] = True   # green in the R row
    g_brow[1::2, 0::2] = True   # green in the B row
    R = np.where(g_rrow, rb_at_g_h, R)
    R = np.where(g_brow, rb_at_g_v, R)
    R = np.where(b_m,    rb_at_br,  R)

    # --- Blue channel (mirror of red) --------------------------------------
    # Blue in B row flanked horizontally, blue in R row flanked vertically.
    B = np.where(g_brow, rb_at_g_h, B)
    B = np.where(g_rrow, rb_at_g_v, B)
    B = np.where(r_m,    rb_at_br,  B)

    out = np.stack([R, G, B], axis=-1)
    return np.clip(out, 0.0, 1.0)


def demosaic(raw, method="malvar", pattern="RGGB"):
    if method == "bilinear":
        return demosaic_bilinear(raw, pattern)
    if method in ("malvar", "mhc"):
        return demosaic_malvar(raw, pattern)
    raise ValueError(f"unknown demosaic method {method!r}")
