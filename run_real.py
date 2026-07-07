"""Turnkey REAL benchmark on the standard Kodak 24-image demosaicing set.

This is the honest, standard-practice evaluation (not the toy synthetic scene):

  PART A - Demosaic benchmark (NOISELESS), the recognised yardstick
  ----------------------------------------------------------------
  Standard demosaicing protocol: take a high-quality RGB image, MOSAIC it to a
  Bayer (RGGB) frame (sample one colour per photosite), then DEMOSAIC and score
  the reconstruction against the original. We compare:
      (a) bilinear            - separable linear interpolation
      (b) Malvar-He-Cutler    - gradient-corrected high-quality linear
  Metric: CPSNR / SSIM over the whole image, mean +/- std across the set.
  This is exactly how Kodak/McMaster demosaicing numbers are reported in the
  literature (Malvar 2004; Zhang & Wu). HEADLINE: MHC vs bilinear on Kodak.

  PART B - Hybrid pipeline WITH noise, on Kodak
  ---------------------------------------------
  Forward-model a noisy RAW (linear light -> RGGB mosaic -> Poisson-Gaussian
  shot+read noise -> 12-bit) and reconstruct three ways, scored vs the clean
  sRGB ground truth:
      (a) naive bilinear   (black-level + bilinear demosaic + gamma)
      (b) full classical   (Malvar + gray-world AWB + CCM + gamma + bilateral)
      (c) classical+LEARNED (same front-end, DnCNN-lite denoiser)  <- AI-ISP half
  The DnCNN-lite is TRAINED on a Kodak split and EVALUATED on the HELD-OUT split
  (no train-on-test). Everything runs on CPU.

Usage:
    python run_real.py                 # download Kodak if missing, run A + B
    python run_real.py --no-torch      # classical only (skip learned stage)
    python run_real.py --limit 8       # quick smoke on first 8 images
    python run_real.py --epochs 15     # cap CNN training epochs

If ALL download mirrors are blocked, the script prints instructions and refuses
to invent numbers (it will say "not run - download blocked").

Later scaling (see README): McMaster (18 img, harder, edge-rich) uses the SAME
Part-A protocol; real RAW noise would use SIDD/DND for Part B.
"""
import argparse
import sys
import time

import numpy as np

from data_kodak import download_kodak
from isp.pipeline import ISPConfig, run_classical_pipeline
from isp import demosaic as demo, tone, black_level, white_balance, ccm as ccm_mod, denoise as den
from synth_raw import load_test_image, synth_raw_from_rgb
import metrics


# --------------------------------------------------------------------------- #
# helpers (mirror run_smoke's definitions so numbers are directly comparable)
# --------------------------------------------------------------------------- #
def naive_bilinear(raw, cfg):
    x = black_level.subtract_black_level(raw, cfg.black_level, cfg.white_level)
    rgb = demo.demosaic_bilinear(x, pattern=cfg.pattern)
    return tone.apply_gamma(rgb, cfg.gamma_mode)


def frontend(raw, cfg, method):
    """Demosaic(method) + AWB + CCM + gamma, no denoise. Returns sRGB."""
    x = black_level.subtract_black_level(raw, cfg.black_level, cfg.white_level)
    rgb = demo.demosaic(x, method=method, pattern=cfg.pattern)
    if cfg.do_awb:
        rgb, _ = white_balance.gray_world(rgb)
    rgb = ccm_mod.apply_ccm(rgb, cfg.ccm_matrix)
    return tone.apply_gamma(rgb, cfg.gamma_mode)


def mean_std(vals):
    a = np.asarray(vals, np.float64)
    return float(a.mean()), float(a.std())


def prow(name, ps, ss, base=None):
    pm, psd = mean_std(ps)
    sm, ssd = mean_std(ss)
    delta = f"{pm - base:+.2f}" if base is not None else ""
    return f"{name:<34} {pm:>6.2f}+-{psd:<4.2f} {sm:>6.4f}+-{ssd:<6.4f} {delta:>7}"


def hdr(title):
    print("-" * 72)
    print(title)
    print(f"{'method':<34} {'CPSNR(dB)':>11} {'SSIM':>14} {'dPSNR':>7}")
    print("-" * 72)


# --------------------------------------------------------------------------- #
# PART A - noiseless demosaic benchmark
# --------------------------------------------------------------------------- #
def demosaic_benchmark(paths):
    """Mosaic each RGB -> RGGB -> demosaic (bilinear, MHC); score vs original."""
    from synth_raw import mosaic_rggb
    bil_p, bil_s, mhc_p, mhc_s = [], [], [], []
    print(f"\nPART A  |  NOISELESS demosaic benchmark on {len(paths)} images")
    print("  (mosaic sRGB -> RGGB Bayer -> demosaic -> CPSNR/SSIM vs original)")
    for p in paths:
        gt = load_test_image(p, size=None)             # native res, even dims
        raw = mosaic_rggb(gt)                            # standard: mosaic as-is
        rb = demo.demosaic_bilinear(raw, "RGGB")
        rm = demo.demosaic_malvar(raw, "RGGB")
        sb, sm = metrics.score(gt, rb), metrics.score(gt, rm)
        bil_p.append(sb["psnr"]); bil_s.append(sb["ssim"])
        mhc_p.append(sm["psnr"]); mhc_s.append(sm["ssim"])
    print()
    hdr("PART A: Kodak demosaic (noiseless) - mean +/- std over the set")
    print(prow("(a) bilinear", bil_p, bil_s))
    print(prow("(b) Malvar-He-Cutler (MHC)", mhc_p, mhc_s, mean_std(bil_p)[0]))
    print("-" * 72)
    print(f"HEADLINE: MHC beats bilinear by "
          f"{mean_std(mhc_p)[0] - mean_std(bil_p)[0]:+.2f} dB CPSNR "
          f"(+{mean_std(mhc_s)[0] - mean_std(bil_s)[0]:.4f} SSIM) on {len(paths)} Kodak images.")
    return {
        "n": len(paths),
        "bilinear": (mean_std(bil_p), mean_std(bil_s)),
        "mhc": (mean_std(mhc_p), mean_std(mhc_s)),
    }


# --------------------------------------------------------------------------- #
# PART B - hybrid pipeline WITH noise (train/eval split)
# --------------------------------------------------------------------------- #
def hybrid_benchmark(paths, args):
    n = len(paths)
    n_train = max(1, n // 2)
    train_paths = paths[:n_train]
    eval_paths = paths[n_train:]
    if not eval_paths:                     # tiny --limit: reuse split
        eval_paths = paths[-1:]

    cfg = ISPConfig(demosaic_method="malvar", do_awb=True, gamma_mode="srgb",
                    do_denoise=True, denoise_sigma_color=0.06, denoise_sigma_spatial=1.8)
    illum = None if args.illuminant.lower() == "none" else \
        [float(v) for v in args.illuminant.split(",")]

    print(f"\nPART B  |  HYBRID pipeline WITH noise on Kodak")
    print(f"  noise: Poisson-Gaussian gain={args.gain} sigma_read={args.sigma_read} (12-bit RGGB)")
    print(f"  illuminant cast: {illum}")
    print(f"  train split: {n_train} imgs   eval (held-out): {len(eval_paths)} imgs")

    # ---- train the learned denoiser on the TRAIN split -------------------
    learned_ok = False
    model = None
    if not args.no_torch:
        try:
            import torch  # noqa
            from learned.train import train_denoiser, infer
            print(f"\n  training DnCNN-lite on {n_train} Kodak images (CPU, "
                  f"{args.epochs} epochs, native-res patches)...")
            t = time.time()
            model = train_denoiser(train_paths, None, cfg, args.gain, args.sigma_read,
                                   epochs=args.epochs, per_image=args.per_image,
                                   seed=args.seed, illuminant=illum, verbose=True)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  trained: {n_params:,} params in {time.time()-t:.1f}s")
            learned_ok = True
        except Exception as e:
            print(f"[!] learned stage skipped: {e}", file=sys.stderr)
    else:
        print("\n  learned stage disabled (--no-torch)")

    # ---- evaluate on the HELD-OUT split ---------------------------------
    _infer = None
    if learned_ok:
        from learned.train import infer as _infer
    a_p, a_s = [], []      # naive bilinear
    b_p, b_s = [], []      # full classical
    c_p, c_s = [], []      # classical + learned
    nd_p, nd_s = [], []    # front-end, no denoise (ablation base)
    bd_p, bd_s = [], []    # front-end + bilateral (ablation)
    for i, p in enumerate(eval_paths):
        clean = load_test_image(p, size=None)
        raw, _ = synth_raw_from_rgb(clean, gain=args.gain, sigma_read=args.sigma_read,
                                    bit_depth=12, seed=args.seed + i, illuminant=illum)
        out_a = naive_bilinear(raw, cfg)
        out_b = run_classical_pipeline(raw, cfg)
        out_nd = frontend(raw, cfg, "malvar")
        out_bd = den.bilateral(out_nd, cfg.denoise_sigma_color, cfg.denoise_sigma_spatial)
        sa, sb = metrics.score(clean, out_a), metrics.score(clean, out_b)
        snd, sbd = metrics.score(clean, out_nd), metrics.score(clean, out_bd)
        a_p.append(sa["psnr"]); a_s.append(sa["ssim"])
        b_p.append(sb["psnr"]); b_s.append(sb["ssim"])
        nd_p.append(snd["psnr"]); nd_s.append(snd["ssim"])
        bd_p.append(sbd["psnr"]); bd_s.append(sbd["ssim"])
        if learned_ok:
            out_c = _infer(model, out_nd)
            sc = metrics.score(clean, out_c)
            c_p.append(sc["psnr"]); c_s.append(sc["ssim"])

    print()
    hdr("PART B-MAIN: reconstruction vs GT (mean +/- std, held-out Kodak)")
    print(prow("(a) naive bilinear", a_p, a_s))
    print(prow("(b) full classical", b_p, b_s, mean_std(a_p)[0]))
    if learned_ok:
        print(prow("(c) classical + LEARNED (hybrid)", c_p, c_s, mean_std(a_p)[0]))
    print("-" * 72)
    print("dPSNR = mean gain over the naive baseline (dB).")

    print()
    hdr("PART B-ABLATION: denoiser only (front-end fixed = Malvar+AWB+gamma)")
    print(prow("no denoise", nd_p, nd_s))
    print(prow("classical bilateral", bd_p, bd_s, mean_std(nd_p)[0]))
    if learned_ok:
        print(prow("learned DnCNN-lite", c_p, c_s, mean_std(nd_p)[0]))
    print("-" * 72)
    return learned_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="use only first N images (0=all)")
    ap.add_argument("--gain", type=float, default=150.0)
    ap.add_argument("--sigma-read", type=float, default=0.008)
    ap.add_argument("--illuminant", default="none",
                    help="scene illuminant RGB cast, comma-sep, or 'none' (default)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-image", type=int, default=64, help="patches sampled per train image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--skip-hybrid", action="store_true", help="only run Part A")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("REAL BENCHMARK  |  Kodak 24-image demosaicing + hybrid ISP")
    print("=" * 72)

    print("ensuring Kodak set is present...")
    paths, errors = download_kodak(verbose=False)
    if not paths:
        print("\n*** NOT RUN - DOWNLOAD BLOCKED ***")
        print("All Kodak mirrors were unreachable; refusing to invent numbers.")
        print("Fix: manually place kodim01.png..kodim24.png in data/kodak/ and re-run.")
        print("Mirrors tried:")
        for n, e in errors[:3]:
            print(f"  {n}: {e}")
        return 2
    if errors:
        print(f"  note: {len(errors)} image(s) could not be fetched; "
              f"running on the {len(paths)} present.")
    if args.limit:
        paths = paths[:args.limit]
    print(f"  {len(paths)} images ready.")

    demosaic_benchmark(paths)
    if not args.skip_hybrid:
        hybrid_benchmark(paths, args)

    print(f"\ntotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
