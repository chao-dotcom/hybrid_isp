"""One-command smoke test: synth RAW -> run pipelines -> print PSNR/SSIM tables.

MAIN comparison (three reconstructions vs the clean ground-truth sRGB):
    (a) naive bilinear baseline   (black-level + bilinear demosaic + gamma;
                                   NO white balance, NO denoise)
    (b) full CLASSICAL pipeline   (Malvar demosaic + gray-world AWB + CCM
                                   + gamma + bilateral denoise)
    (c) CLASSICAL + LEARNED        (same front-end, DnCNN-lite denoiser instead
                                   of bilateral)  <- the hybrid / "AI-ISP" half

ABLATIONS (isolate one stage at a time):
    * demosaic:  bilinear vs Malvar-He-Cutler  (AWB on, denoise off)
    * denoise:   none vs bilateral vs learned   (front-end fixed)

Usage:
    python run_smoke.py                              # synthetic scene, mild cast
    python run_smoke.py --eval astronaut --size 256
    python run_smoke.py --gain 120 --sigma-read 0.01 --illuminant 1.3,1,0.75
    python run_smoke.py --no-torch                   # skip the learned stage
    python run_smoke.py --save-images out/           # dump PNGs for the README

Every number is MEASURED at runtime; nothing is hard-coded.
"""
import argparse
import sys
import time

import numpy as np

from isp.pipeline import ISPConfig, run_classical_pipeline
from isp import demosaic as demo, tone, black_level, white_balance, ccm as ccm_mod, denoise as den
from synth_raw import load_test_image, synth_raw_from_rgb
import metrics


def naive_bilinear(raw, cfg):
    """Minimal baseline: black-level -> bilinear demosaic -> gamma. No WB/CCM/denoise."""
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


def row(name, s, base_psnr=None):
    delta = f"{s['psnr'] - base_psnr:+.2f}" if base_psnr is not None else ""
    return f"{name:<38} {s['psnr']:>7.2f} {s['ssim']:>8.4f} {delta:>8}"


def header(title):
    print("-" * 66)
    print(title)
    print(f"{'method':<38} {'PSNR':>7} {'SSIM':>8} {'dPSNR':>8}")
    print("-" * 66)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="macbeth",
                    help="eval image: 'macbeth' (synthetic, controlled) / skimage name / path")
    ap.add_argument("--train", nargs="+", default=["chelsea", "coffee"],
                    help="training images for the learned stage (held out from eval)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gain", type=float, default=150.0, help="photon gain (lower = more shot noise)")
    ap.add_argument("--sigma-read", type=float, default=0.008)
    ap.add_argument("--illuminant", default="1.25,1.0,0.8",
                    help="scene illuminant RGB cast, comma-sep, or 'none'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-torch", action="store_true", help="skip the learned CNN stage")
    ap.add_argument("--save-images", default=None, help="dir to dump PNGs")
    args = ap.parse_args()

    illum = None if args.illuminant.lower() == "none" else \
        [float(v) for v in args.illuminant.split(",")]

    t0 = time.time()
    print("=" * 66)
    print("HYBRID RAW->RGB MINI-ISP  |  classical vs learned")
    print("=" * 66)

    # 1. Clean GT + synthetic RAW ------------------------------------------
    clean = load_test_image(args.eval, size=args.size)
    raw, meta = synth_raw_from_rgb(clean, gain=args.gain, sigma_read=args.sigma_read,
                                   bit_depth=12, seed=args.seed, illuminant=illum)
    print(f"eval image      : {args.eval}  ({clean.shape[0]}x{clean.shape[1]})")
    print(f"noise model     : Poisson-Gaussian  gain={meta['gain']}  sigma_read={meta['sigma_read']}")
    print(f"effective sigma : ~{meta['sigma_total_at_mid_gray']:.4f} at mid-gray  (12-bit RAW, RGGB)")
    print(f"illuminant cast : {meta['illuminant']}")

    cfg = ISPConfig(demosaic_method="malvar", do_awb=True, gamma_mode="srgb",
                    do_denoise=True, denoise_sigma_color=0.06, denoise_sigma_spatial=1.8)

    # ---- MAIN table ------------------------------------------------------
    out_a = naive_bilinear(raw, cfg);            s_a = metrics.score(clean, out_a)
    out_b = run_classical_pipeline(raw, cfg);    s_b = metrics.score(clean, out_b)
    out_nd = frontend(raw, cfg, "malvar");       s_nd = metrics.score(clean, out_nd)

    learned_ok = False
    out_c = None
    if not args.no_torch:
        try:
            import torch  # noqa
            from learned.train import train_denoiser, infer
            print("\ntraining learned denoiser (CPU)...")
            model = train_denoiser(args.train, args.size, cfg, args.gain,
                                   args.sigma_read, epochs=args.epochs,
                                   seed=args.seed, illuminant=illum)
            out_c = infer(model, out_nd)          # denoise the SAME noisy front-end
            s_c = metrics.score(clean, out_c)
            n_params = sum(p.numel() for p in model.parameters())
            learned_ok = True
            print(f"learned model   : DnCNN-lite, {n_params:,} params, "
                  f"train={args.train}, epochs={args.epochs}")
        except Exception as e:
            print(f"[!] learned stage skipped: {e}", file=sys.stderr)
    else:
        print("\nlearned stage disabled (--no-torch)")

    print()
    header("MAIN: full-pipeline reconstruction vs ground truth")
    print(row("(a) naive bilinear baseline", s_a))
    print(row("(b) full classical (Malvar+AWB+bilateral)", s_b, s_a["psnr"]))
    if learned_ok:
        print(row("(c) classical + LEARNED (hybrid)", s_c, s_a["psnr"]))
    print("-" * 66)
    print("dPSNR = gain over the naive baseline (dB).")

    # ---- Ablation 1: demosaic -------------------------------------------
    fe_bil = frontend(raw, cfg, "bilinear"); s_feb = metrics.score(clean, fe_bil)
    header("ABLATION 1: demosaic quality (AWB on, denoise off)")
    print(row("bilinear demosaic", s_feb))
    print(row("Malvar-He-Cutler demosaic", s_nd, s_feb["psnr"]))

    # ---- Ablation 2: denoiser (front-end fixed = Malvar+AWB) -------------
    out_bil_dn = den.bilateral(out_nd, cfg.denoise_sigma_color, cfg.denoise_sigma_spatial)
    s_bdn = metrics.score(clean, out_bil_dn)
    header("ABLATION 2: denoiser (front-end fixed: Malvar+AWB+gamma)")
    print(row("no denoise", s_nd))
    print(row("classical bilateral", s_bdn, s_nd["psnr"]))
    if learned_ok:
        print(row("learned DnCNN-lite", s_c, s_nd["psnr"]))
    print("-" * 66)
    print(f"total runtime   : {time.time() - t0:.1f}s")

    # ---- optional image dump --------------------------------------------
    if args.save_images:
        import os, imageio.v3 as iio
        os.makedirs(args.save_images, exist_ok=True)
        u8 = lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
        iio.imwrite(os.path.join(args.save_images, "00_clean_gt.png"), u8(clean))
        iio.imwrite(os.path.join(args.save_images, "01_noisy_raw.png"), u8(raw))
        iio.imwrite(os.path.join(args.save_images, "02_naive_bilinear.png"), u8(out_a))
        iio.imwrite(os.path.join(args.save_images, "03_classical.png"), u8(out_b))
        if learned_ok:
            iio.imwrite(os.path.join(args.save_images, "04_hybrid_learned.png"), u8(out_c))
        print(f"saved PNGs to   : {args.save_images}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
