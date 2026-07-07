# Hybrid RAW→RGB Mini-ISP (classical + learned)

A small, runnable **image signal processor** that turns a sensor's RAW Bayer
frame into a display RGB image, built two ways and compared with real numbers:

1. a **classical**, stage-by-stage ISP (NumPy/SciPy), and
2. a **learned** denoising stage (a tiny DnCNN-style CNN in PyTorch)

…with a quantitative **PSNR / SSIM** comparison of naive vs classical vs hybrid.
Everything runs on CPU in ~2 minutes; no GPU and no dataset download required
(test images ship with scikit-image, plus a synthetic Macbeth-style scene).

> Built as an edge-AI portfolio artifact: it exercises exactly the on-device
> pipeline skills — real RAW→RGB image processing, a small quantized-friendly
> CNN, and PSNR/SSIM measurement — with an honest, measured evaluation.

---

## Pipeline

```
        RAW Bayer (RGGB, linear, noisy)
                 │
   ┌─────────────▼──────────────┐
   │ 1. black-level subtraction │  remove sensor pedestal, normalise to [0,1]
   ├────────────────────────────┤
   │ 2. lens-shading (optional) │  correct radial vignetting
   ├────────────────────────────┤
   │ 3. demosaic                │  bilinear  OR  Malvar-He-Cutler (gradient-corr.)
   ├────────────────────────────┤
   │ 4. auto white balance      │  gray-world: neutralise the illuminant cast
   ├────────────────────────────┤
   │ 5. color correction matrix │  sensor RGB → display RGB (3×3)
   ├────────────────────────────┤
   │ 6. tone / gamma            │  linear → sRGB encode
   ├────────────────────────────┤
   │ 7. denoise                 │  classical bilateral  ──┐
   └────────────────────────────┘                         │  swap-in point
                 │                                         ▼
                 │                            LEARNED  DnCNN-lite denoiser
                 ▼
        display sRGB  ──►  PSNR / SSIM vs ground truth
```

### Why this stage order (a classic ISP interview question)

- **Black level first.** All colour math needs a true zero; the optical-black
  pedestal must go before any ratios (WB) or non-linear mapping (gamma).
- **Demosaic before colour.** WB/CCM operate on full RGB; they need all three
  channels present at every pixel.
- **WB and CCM in *linear* light, before gamma.** Both are multiplicative on
  radiance (per-channel gains, a 3×3 mix). Doing them after gamma would apply
  linear operations to non-linear values and shift hues/exposure.
- **Gamma near the end.** Everything upstream is linear; we encode for the
  display last.
- **Denoise placement.** Here the denoiser runs at the tail (display-referred),
  so the classical bilateral and the learned CNN see *identical* inputs and the
  comparison is apples-to-apples. A production ISP may instead denoise in the
  RAW domain right after black-level; the modular design supports either.

---

## The two demosaic variants (the highest-impact stage)

- **Bilinear** — separable linear interpolation of each colour plane. Cheap but
  causes *zippering* and colour fringing at edges (each channel interpolated
  independently, ignoring cross-channel correlation).
- **Malvar-He-Cutler (2004)** — "high-quality linear" demosaic (the method
  behind MATLAB's `demosaic`). Same 5×5 FIR cost class, but adds
  gradient-corrected cross-channel terms (e.g. a Laplacian of the known channel
  when interpolating another), which sharply reduces edge artefacts.

Measured: on a noiseless mosaic of the synthetic scene, Malvar beats bilinear by
**~3 dB** (see `RESULTS.md`), confirming the implementation.

---

## The learned stage ("AI-ISP" half)

`learned/model.py` is a **DnCNN-lite** residual CNN (~39k params, 6 conv layers):
it predicts the *noise* residual and subtracts it (`clean = noisy − net(noisy)`).
Residual learning converges fast on little data — key for a CPU proof-of-concept.

**Honest training protocol** (`learned/train.py`):
- Trained on images **held out** from the evaluation image (default train =
  `chelsea, coffee`, eval = `astronaut`/`macbeth`) — a genuine generalisation
  test, not train-on-test.
- Trained as a **pure denoiser**: input = classical front-end on a *noisy* RAW,
  target = the *same* front-end on a *noiseless* RAW. This decouples the CNN from
  colour/demosaic modelling (already handled by the classical stages and
  scene-dependent through gray-world), so it drops in exactly where the bilateral
  filter sits and the two are compared on identical inputs.

---

## Input handling (no big downloads)

`synth_raw.py` forward-models a sensor from a clean RGB image, giving us
ground-truth RGB to score against:

```
clean sRGB → sRGB-decode to LINEAR → (optional illuminant cast)
           → Bayer mosaic (RGGB) → Poisson (shot) + Gaussian (read) noise
           → 12-bit quantise (+ optional black-level pedestal) → noisy RAW
```

The **Poisson-Gaussian** model (Foi et al. 2008) is the standard sensor-noise
model: shot noise scales with √signal (photon counting), read noise is additive.
`gain` sets photon conversion (lower gain ⇒ stronger shot noise).

Test images: any scikit-image sample (`astronaut`, `coffee`, `chelsea`, …), a
path to your own image, or `macbeth` — a synthetic scene with colour patches,
gradients, and hard edges whose linear channel-means are equalised so gray-world
AWB is a well-posed test.

---

## Files

```
isp/                 classical pipeline, one module per stage
  black_level.py     1. pedestal subtraction
  lens_shading.py    2. vignetting correction (optional)
  demosaic.py        3. bilinear + Malvar-He-Cutler
  white_balance.py   4. gray-world AWB
  ccm.py             5. colour correction matrix
  tone.py            6. sRGB gamma (encode/decode)
  denoise.py         7. bilateral denoise
  pipeline.py        wires the stages (ISPConfig)
learned/
  model.py           DnCNN-lite residual CNN
  train.py           patch training + full-frame inference (CPU)
synth_raw.py         RGB → Bayer + Poisson-Gaussian noise + synthetic scene
metrics.py           PSNR / SSIM (skimage, with NumPy fallback)
run_smoke.py         one command → synthetic-scene comparison + ablation tables
data_kodak.py        download the standard Kodak 24-image benchmark (mirrors)
run_real.py          REAL benchmark on Kodak (auto-download → prints tables)
requirements.txt
RESULTS.md           measured numbers + résumé bullet + interview framing
```

---

## Run it

```bash
pip install -r requirements.txt

# default: synthetic controlled scene, mild illuminant cast, learned stage on
python run_smoke.py

# showcase the learned denoiser on a natural image
python run_smoke.py --eval astronaut --train chelsea coffee --illuminant 1.15,1,0.9

# knobs
python run_smoke.py --gain 120 --sigma-read 0.01 --epochs 40 --save-images out/
python run_smoke.py --no-torch          # classical only (skip the CNN)
```

`run_smoke.py` prints a MAIN table (naive / classical / hybrid) plus two
ablations (demosaic: bilinear vs Malvar; denoise: none vs bilateral vs learned).
All numbers are measured at runtime.

### Real standard benchmark (Kodak 24)

`run_real.py` runs the **recognised demosaicing yardstick** — the Kodak 24-image
set — end to end: it downloads Kodak if missing, then reports (A) the noiseless
demosaic benchmark (bilinear vs Malvar-He-Cutler, CPSNR/SSIM, mean±std over the
set) and (B) the hybrid pipeline with Poisson-Gaussian noise, with the DnCNN-lite
**trained on a Kodak split and evaluated on the held-out split**.

```bash
python run_real.py                 # download Kodak → Part A + Part B tables
python run_real.py --no-torch      # classical only (skip learned stage)
python run_real.py --skip-hybrid   # just the noiseless demosaic benchmark
python run_real.py --limit 8       # quick smoke on the first 8 images
```

If every download mirror is blocked it prints `NOT RUN - DOWNLOAD BLOCKED` and
refuses to invent numbers (drop the PNGs into `data/kodak/` and re-run).
Measured Kodak numbers are in `RESULTS.md` (the "REAL — Kodak" section).

---

## Using real RAW datasets later

The pipeline consumes a single-channel Bayer array in `[0,1]`; swap the
synthetic front-end for real captures:

- **Kodak** (demosaic benchmark): **done** — `run_real.py` is turnkey (see above).
- **McMaster (18 img)** — the harder, edge-rich sibling of Kodak, same Part-A
  protocol. To add it: fetch the 18 `.tif` crops, point the loader at them, and
  reuse `demosaic_benchmark`; expect lower CPSNR than Kodak (more saturated,
  high-frequency content). A good next real number.
- **SIDD / DND** (real smartphone denoising, the honest Part-B upgrade): load the
  RAW (`rawpy` for DNG),
  set the real `black_level`/`white_level`/CFA pattern in `ISPConfig`, and use
  the provided clean references as ground truth for the denoise comparison.
- Retrain the CNN on real noisy/clean pairs (same `train.py` interface) — the
  residual-denoiser design and PSNR/SSIM harness are dataset-agnostic.

---

## References

- H.S. Malvar, L. He, R. Cutler, *High-quality linear interpolation for demosaicing of Bayer-patterned color images*, ICASSP 2004.
- A. Foi et al., *Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data*, IEEE TIP 2008.
- K. Zhang et al., *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN)*, IEEE TIP 2017.
- Y. Blau, T. Michaeli, *The Perception-Distortion Tradeoff*, CVPR 2018.
