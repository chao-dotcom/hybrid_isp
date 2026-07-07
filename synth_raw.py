"""Synthesise a realistic RAW frame from a clean RGB image.

We need ground-truth RGB to score the ISP against, but real RAW+GT datasets are
large downloads. Instead we forward-model the sensor:

    clean sRGB image
      -> sRGB decode to LINEAR light         (the sensor sees linear radiance)
      -> Bayer mosaic (RGGB): keep one colour per photosite
      -> Poisson-Gaussian noise:
             shot noise   ~ Poisson(signal * gain) / gain   (photon counting)
             read noise   ~ Gaussian(0, sigma_read)         (amplifier)
      -> quantise to the sensor bit depth, add a black-level pedestal
    -> noisy RAW Bayer (what the ISP ingests)

The GROUND TRUTH we compare against is the clean image in the SAME display space
the ISP outputs (sRGB), so PSNR/SSIM measure end-to-end reconstruction quality.

Noise model reference: Poisson-Gaussian (Foi et al. 2008). ``gain`` is the
electrons-per-unit scaling: lower gain -> stronger shot noise (fewer photons).
"""
import numpy as np

from isp import tone


# --------------------------------------------------------------------------- #
# Test images
# --------------------------------------------------------------------------- #
def load_test_image(name="astronaut", size=256):
    """Return a clean sRGB image in [0,1], shape (size, size, 3).

    ``name`` may be a skimage sample ('astronaut', 'coffee', 'chelsea', ...),
    'macbeth' for a synthetic colour chart, or a path to an image file.

    If ``size`` is None the image is returned at its NATIVE resolution (no
    resize), rounded down to even height/width so the RGGB Bayer tiling is
    well-defined. This is what the real-benchmark driver (run_real.py) uses so
    the standard Kodak/McMaster PSNR numbers are computed at full resolution,
    not on a resampled crop.
    """
    from skimage.transform import resize

    if name == "macbeth":
        img = synthetic_scene(size if size is not None else 256)
    else:
        import skimage.data as skd
        import os
        if os.path.isfile(name):
            import imageio.v3 as iio
            img = iio.imread(name)
        elif hasattr(skd, name):
            img = getattr(skd, name)()
        else:
            raise ValueError(f"unknown image {name!r}")
        img = _to_float_rgb(img)
        if size is None:
            h, w = img.shape[:2]
            img = img[:h - (h % 2), :w - (w % 2)]      # even dims for RGGB
        else:
            img = resize(img, (size, size), anti_aliasing=True)
    return np.clip(img, 0.0, 1.0).astype(np.float64)


def _to_float_rgb(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img / 255.0
    elif img.dtype == np.uint16:
        img = img / 65535.0
    else:
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / img.max()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.float64)


def synthetic_scene(size=256):
    """A Macbeth-style test scene: colour patches + gradients + hard edges.

    Exercises the parts of the ISP that matter: flat colour regions (WB/CCM),
    smooth gradients (banding/denoise), and sharp edges (demosaic zippering).
    """
    img = np.zeros((size, size, 3), np.float64)

    # Top half: 4x6 grid of colour patches (Macbeth-ish).
    patches = [
        (0.45, 0.32, 0.27), (0.76, 0.58, 0.50), (0.36, 0.48, 0.61),
        (0.35, 0.42, 0.26), (0.51, 0.50, 0.69), (0.39, 0.74, 0.67),
        (0.85, 0.48, 0.16), (0.29, 0.36, 0.64), (0.76, 0.33, 0.38),
        (0.36, 0.24, 0.42), (0.62, 0.73, 0.25), (0.89, 0.63, 0.18),
        (0.20, 0.24, 0.59), (0.28, 0.58, 0.29), (0.69, 0.20, 0.24),
        (0.93, 0.78, 0.13), (0.73, 0.33, 0.58), (0.00, 0.53, 0.64),
        (0.95, 0.95, 0.95), (0.78, 0.78, 0.78), (0.63, 0.63, 0.63),
        (0.48, 0.48, 0.48), (0.33, 0.33, 0.33), (0.20, 0.20, 0.20),
    ]
    top = size // 2
    ph, pw = top // 4, size // 6
    for i, col in enumerate(patches):
        r, c = divmod(i, 6)
        img[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw, :] = col

    # Bottom-left quadrant: horizontal luminance gradient (banding test).
    grad = np.linspace(0, 1, size // 2)[None, :, None]
    img[top:, :size // 2, :] = np.repeat(grad, size - top, axis=0)

    # Bottom-right quadrant: high-contrast edges / bars (demosaic stress).
    br = img[top:, size // 2:, :]
    for k in range(0, br.shape[1], 8):
        br[:, k:k + 4, :] = 1.0
    # radial edge
    yy, xx = np.mgrid[0:br.shape[0], 0:br.shape[1]]
    cy, cx = br.shape[0] / 2, br.shape[1] / 2
    ring = (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) % 20) < 10
    br[ring] = np.array([0.9, 0.1, 0.1])
    img[top:, size // 2:, :] = br

    img = np.clip(img, 0.0, 1.0)
    # Make the scene gray-world VALID: equalise the per-channel means in linear
    # light so a neutral illuminant maps to equal channel averages. This makes
    # gray-world AWB a controlled, well-posed test (it can recover a known cast).
    lin = tone.srgb_to_linear(img)
    m = lin.reshape(-1, 3).mean(axis=0) + 1e-8
    lin = np.clip(lin * (m.mean() / m), 0.0, 1.0)
    return tone.linear_to_srgb(lin)


# --------------------------------------------------------------------------- #
# Sensor forward model
# --------------------------------------------------------------------------- #
def mosaic_rggb(linear_rgb):
    """Sample one colour per pixel following the RGGB Bayer layout."""
    h, w, _ = linear_rgb.shape
    raw = np.zeros((h, w), np.float64)
    R, G, B = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    raw[0::2, 0::2] = R[0::2, 0::2]   # R
    raw[0::2, 1::2] = G[0::2, 1::2]   # G
    raw[1::2, 0::2] = G[1::2, 0::2]   # G
    raw[1::2, 1::2] = B[1::2, 1::2]   # B
    return raw


def add_poisson_gaussian(raw, gain=200.0, sigma_read=0.005, rng=None):
    """Apply shot (Poisson) + read (Gaussian) noise to a clean linear Bayer frame.

    Parameters
    ----------
    gain : float
        Photon-conversion gain (a.k.a. full-well scale). We draw
        Poisson(raw * gain) / gain, so larger gain => less relative shot noise.
    sigma_read : float
        Std-dev of additive read noise in signal units.
    """
    rng = np.random.default_rng() if rng is None else rng
    raw = np.clip(raw, 0.0, 1.0)
    shot = rng.poisson(raw * gain) / gain
    read = rng.normal(0.0, sigma_read, size=raw.shape)
    noisy = shot + read
    return np.clip(noisy, 0.0, 1.0)


def synth_raw_from_rgb(srgb_img, gain=200.0, sigma_read=0.005,
                       black_level=0.0, bit_depth=12, seed=0, illuminant=None):
    """Full forward model. Returns (noisy_raw, meta).

    ``srgb_img`` is clean display-referred RGB in [0,1]. The returned RAW is a
    single-channel Bayer frame in [0,1] (already includes pedestal if requested).
    The ground truth for scoring is ``srgb_img`` itself.

    ``illuminant`` : optional (3,) RGB multiplier applied in linear light before
    mosaicing to model a coloured light source (e.g. [1.25,1,0.8] ~ tungsten).
    The ISP's auto white balance is expected to undo it. None -> neutral.
    """
    rng = np.random.default_rng(seed)
    linear = tone.srgb_to_linear(srgb_img)          # sensor sees linear light
    if illuminant is not None:
        linear = np.clip(linear * np.asarray(illuminant, np.float64), 0.0, 1.0)
    clean_raw = mosaic_rggb(linear)
    noisy_raw = add_poisson_gaussian(clean_raw, gain, sigma_read, rng)

    # Quantise to sensor bit depth and add optical-black pedestal.
    maxc = (2 ** bit_depth) - 1
    coded = np.round(noisy_raw * maxc)
    if black_level > 0:
        coded = np.clip(coded + black_level * maxc, 0, maxc)
    raw01 = coded / maxc

    # Estimate the effective noise sigma on a mid-gray patch for honest reporting.
    mid = 0.5
    sigma_shot_mid = np.sqrt(mid / gain)
    sigma_total_mid = float(np.sqrt(sigma_shot_mid ** 2 + sigma_read ** 2))

    meta = {
        "gain": gain,
        "sigma_read": sigma_read,
        "bit_depth": bit_depth,
        "black_level_frac": black_level,
        "sigma_total_at_mid_gray": sigma_total_mid,
        "pattern": "RGGB",
        "illuminant": None if illuminant is None else list(np.asarray(illuminant, float)),
        "clean_linear": linear,
    }
    return raw01, meta
