"""Train the DnCNN-lite denoiser on patches from TRAINING images.

Honest protocol
---------------
The denoiser is trained on images that are DIFFERENT from the evaluation image
(default train = ['chelsea', 'coffee'], eval = 'astronaut' in run_smoke). So the
reported learned-stage numbers are a genuine generalisation result, not
train-on-test. Everything runs on CPU in a few seconds/epochs (proof-of-concept).

Training pair for each patch:
    input  = classical pipeline output WITHOUT the bilateral denoise (noisy sRGB)
    target = clean ground-truth sRGB
The CNN thus learns to denoise the ISP tail, directly comparable to the classical
bilateral stage on identical inputs.
"""
import numpy as np
import torch
import torch.nn as nn

from isp.pipeline import ISPConfig, run_classical_pipeline
from synth_raw import load_test_image, synth_raw_from_rgb, mosaic_rggb
from isp import tone
from .model import DnCNNLite


def _pipeline_no_denoise(cfg: ISPConfig):
    c = ISPConfig(**{**cfg.__dict__})
    c.do_denoise = False
    c.debug = {}
    return c


def build_training_pairs(train_images, size, cfg, gain, sigma_read, seed=0,
                         illuminant=None):
    """Return (noisy_frontend, clean_frontend) pairs for each training image.

    KEY DESIGN CHOICE: the CNN is trained as a PURE DENOISER. Both input and
    target pass through the SAME classical front-end (Malvar demosaic + gray-world
    AWB + CCM + gamma, no denoise); only the input's RAW carries sensor noise:

        input  = frontend( noisy  RAW )      # demosaic/AWB/gamma of a noisy frame
        target = frontend( clean  RAW )      # same pipeline on a noiseless frame

    This decouples the network from colour/demosaic modelling (which the classical
    stages already handle and which are scene-dependent through gray-world). If we
    instead used the original ground-truth sRGB as the target, the net would try
    to also learn the per-scene AWB correction and would MIS-generalise a colour
    shift to unseen scenes. Training on the front-end delta = noise makes the CNN
    a drop-in replacement for the bilateral denoiser, on identical inputs.
    """
    pairs = []
    nd_cfg = _pipeline_no_denoise(cfg)
    for i, name in enumerate(train_images):
        clean = load_test_image(name, size=size)
        # noisy front-end (network input)
        raw_noisy, _ = synth_raw_from_rgb(clean, gain=gain, sigma_read=sigma_read,
                                          seed=seed + 100 + i, illuminant=illuminant)
        inp = run_classical_pipeline(raw_noisy, nd_cfg)
        # clean front-end (network target): same forward geometry, no sensor noise
        lin = tone.srgb_to_linear(clean)
        if illuminant is not None:
            lin = np.clip(lin * np.asarray(illuminant, np.float64), 0.0, 1.0)
        raw_clean = mosaic_rggb(lin)
        tgt = run_classical_pipeline(raw_clean, nd_cfg)
        pairs.append((inp.astype(np.float32), tgt.astype(np.float32)))
    return pairs


def _sample_patches(pairs, patch=48, per_image=64, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    xs, ys = [], []
    for noisy, clean in pairs:
        h, w, _ = noisy.shape
        for _ in range(per_image):
            y = rng.integers(0, h - patch)
            x = rng.integers(0, w - patch)
            xs.append(noisy[y:y + patch, x:x + patch])
            ys.append(clean[y:y + patch, x:x + patch])
    X = np.stack(xs).transpose(0, 3, 1, 2)   # N,C,H,W
    Y = np.stack(ys).transpose(0, 3, 1, 2)
    return torch.from_numpy(X), torch.from_numpy(Y)


def train_denoiser(train_images, size, cfg, gain, sigma_read,
                   epochs=30, patch=48, per_image=96, lr=1e-3,
                   features=32, depth=6, seed=0, verbose=True, illuminant=None):
    """Train and return a DnCNNLite model (on CPU)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    pairs = build_training_pairs(train_images, size, cfg, gain, sigma_read, seed,
                                 illuminant=illuminant)
    X, Y = _sample_patches(pairs, patch, per_image, rng)

    model = DnCNNLite(channels=3, features=features, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.3)
    lossf = nn.MSELoss()

    n = X.shape[0]
    batch = 16
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [train] epoch {ep+1:>2}/{epochs}  mse={tot/n:.6f}")
    model.eval()
    return model


@torch.no_grad()
def infer(model, noisy_srgb):
    """Run the denoiser on a full-frame sRGB image (H,W,3) in [0,1]."""
    x = torch.from_numpy(noisy_srgb.astype(np.float32).transpose(2, 0, 1))[None]
    out = model(x)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
    return out.astype(np.float64)
