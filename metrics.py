"""Image-quality metrics: PSNR and SSIM.

We prefer skimage's reference implementations (well-tested), but also provide
NumPy fallbacks so the project runs with only numpy. Both operate on float
images in [0, 1].
"""
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    from skimage.metrics import structural_similarity as _sk_ssim
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


def psnr(gt, test, data_range=1.0):
    """Peak signal-to-noise ratio in dB. Higher is better."""
    gt = np.clip(gt, 0, data_range).astype(np.float64)
    test = np.clip(test, 0, data_range).astype(np.float64)
    if _HAVE_SK:
        return float(_sk_psnr(gt, test, data_range=data_range))
    mse = np.mean((gt - test) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(gt, test, data_range=1.0):
    """Structural similarity index (mean over the image). Higher is better."""
    gt = np.clip(gt, 0, data_range).astype(np.float64)
    test = np.clip(test, 0, data_range).astype(np.float64)
    if _HAVE_SK:
        return float(_sk_ssim(gt, test, data_range=data_range, channel_axis=-1))
    return _ssim_numpy(gt, test, data_range)


def _ssim_numpy(gt, test, data_range=1.0):
    """Global (single-window) SSIM fallback; per-channel then averaged."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    vals = []
    chans = gt.shape[-1] if gt.ndim == 3 else 1
    for c in range(chans):
        a = gt[..., c] if gt.ndim == 3 else gt
        b = test[..., c] if test.ndim == 3 else test
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        s = (((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
             ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))
        vals.append(s)
    return float(np.mean(vals))


def score(gt, test):
    """Convenience: return dict with psnr and ssim."""
    return {"psnr": psnr(gt, test), "ssim": ssim(gt, test)}
