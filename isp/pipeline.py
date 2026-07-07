"""Classical RAW -> RGB pipeline: wires the stages in the canonical order.

    RAW (Bayer, linear)
      1. black-level subtraction        black_level.subtract_black_level
      2. lens-shading correction (opt)  lens_shading.correct
      3. demosaic                       demosaic.demosaic  (bilinear | malvar)
      4. auto white balance             white_balance.gray_world
      5. color correction matrix        ccm.apply_ccm
      6. tone / gamma                   tone.apply_gamma
      7. denoise (bilateral)            denoise.bilateral
    -> display-referred sRGB

Why this order (a standard ISP interview question):
  * black level first: colour math needs a true zero.
  * WB / CCM in LINEAR light, before gamma: they are multiplicative on radiance.
  * gamma near the end: everything upstream is linear; encode for display last.
  * denoise at the tail here so it acts on display-referred noise; alternatively
    a RAW-domain denoiser can run right after black level (documented in README).
"""
from dataclasses import dataclass, field

from . import black_level, lens_shading, demosaic, white_balance, ccm, tone, denoise


@dataclass
class ISPConfig:
    black_level: float = 0.0
    white_level: float = 1.0
    lsc_strength: float = 0.0            # 0 disables lens-shading correction
    demosaic_method: str = "malvar"       # 'bilinear' | 'malvar'
    do_awb: bool = True
    ccm_matrix: object = None             # None -> identity
    gamma_mode: str = "srgb"
    do_denoise: bool = True
    denoise_sigma_color: float = 0.08
    denoise_sigma_spatial: float = 2.0
    pattern: str = "RGGB"
    debug: dict = field(default_factory=dict)


def run_classical_pipeline(raw, cfg: ISPConfig = None, keep_intermediates=False):
    """Run the full classical ISP on a single-channel Bayer frame -> sRGB RGB."""
    if cfg is None:
        cfg = ISPConfig()
    inter = {}

    x = black_level.subtract_black_level(raw, cfg.black_level, cfg.white_level)
    if keep_intermediates:
        inter["black_level"] = x.copy()

    x = lens_shading.correct(x, strength=cfg.lsc_strength)
    if keep_intermediates:
        inter["lens_shading"] = x.copy()

    rgb = demosaic.demosaic(x, method=cfg.demosaic_method, pattern=cfg.pattern)
    if keep_intermediates:
        inter["demosaic"] = rgb.copy()

    if cfg.do_awb:
        rgb, gains = white_balance.gray_world(rgb)
        cfg.debug["awb_gains"] = gains
        if keep_intermediates:
            inter["awb"] = rgb.copy()

    rgb = ccm.apply_ccm(rgb, cfg.ccm_matrix)
    if keep_intermediates:
        inter["ccm"] = rgb.copy()

    rgb = tone.apply_gamma(rgb, cfg.gamma_mode)
    if keep_intermediates:
        inter["gamma"] = rgb.copy()

    if cfg.do_denoise:
        rgb = denoise.bilateral(rgb, cfg.denoise_sigma_color, cfg.denoise_sigma_spatial)
        if keep_intermediates:
            inter["denoise"] = rgb.copy()

    if keep_intermediates:
        return rgb, inter
    return rgb
