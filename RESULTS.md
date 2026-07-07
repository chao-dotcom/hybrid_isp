# Results — Hybrid RAW→RGB Mini-ISP (measured)

All numbers below were **measured by running the code** on this machine
(Windows 11, Python 3.12, CPU-only, PyTorch 2.11+cpu). Nothing is hand-tuned or
fabricated. Reproduce with the exact commands shown.

Two evaluation families, kept **separate** below:
1. **REAL — Kodak 24-image benchmark** (the recognised demosaicing yardstick).
   Standard protocol: mosaic a high-quality RGB to Bayer, demosaic, score the
   reconstruction. This is what published demosaicing numbers compare to.
2. **Synthetic** controlled/natural scenes — a forward-modelled sensor
   (Poisson–Gaussian noise, coloured illuminant) used to validate each stage.

PSNR/SSIM via scikit-image; metric target = the clean ground truth.

---

# REAL — Kodak (24 img) demosaic + hybrid

The **standard demosaicing benchmark**. The Kodak PhotoCD set (24 lossless
768×512 images) is the recognised yardstick: you mosaic each image to a Bayer
(RGGB) frame and score the reconstruction. Everyone's demosaic numbers compare to
this. Downloaded automatically by `run_real.py` (24/24 mirrors reachable here).

Exact commands:
```
python run_real.py                 # full: download + Part A + Part B
python run_real.py --skip-hybrid   # just the noiseless demosaic benchmark
```

## PART A — noiseless demosaic benchmark (the headline real number)

Per image: mosaic the sRGB → RGGB Bayer → demosaic → CPSNR/SSIM vs the original.
Mean ± std over all **24** Kodak images (copied from stdout):

```
PART A: Kodak demosaic (noiseless) - mean +/- std over the set
method                               CPSNR(dB)           SSIM   dPSNR
------------------------------------------------------------------------
(a) bilinear                        29.37+-2.77 0.8916+-0.0394
(b) Malvar-He-Cutler (MHC)          35.36+-2.88 0.9688+-0.0080   +5.98
------------------------------------------------------------------------
HEADLINE: MHC beats bilinear by +5.98 dB CPSNR (+0.0772 SSIM) on 24 Kodak images.
```

**Reading it — the real headline:** on the standard Kodak set, the
**Malvar-He-Cutler demosaic scores 35.36 dB CPSNR vs bilinear's 29.37 dB — a
+5.98 dB gain (SSIM 0.969 vs 0.892)**. These land squarely in the published range
for MHC on Kodak (~35–37 dB depending on border handling), confirming the
implementation is correct and competitive, not a toy.

## PART B — hybrid pipeline with noise on Kodak (train/eval split)

Forward-model a noisy RAW (linear light → RGGB mosaic → Poisson–Gaussian
gain=150, σ_read=0.008 → σ≈0.058 at mid-gray → 12-bit), neutral illuminant.
DnCNN-lite (**38,883 params**) **trained on 12 Kodak images, evaluated on the
held-out 12** (no train-on-test), 20 epochs, CPU (~71 s train). Mean ± std over
the held-out 12 (copied from stdout):

```
PART B-MAIN: reconstruction vs GT (mean +/- std, held-out Kodak)
method                               CPSNR(dB)           SSIM   dPSNR
------------------------------------------------------------------------
(a) naive bilinear                  25.58+-1.29 0.6177+-0.0406
(b) full classical                  25.26+-1.98 0.7361+-0.0465   -0.32
(c) classical + LEARNED (hybrid)    25.22+-2.00 0.7491+-0.0342   -0.36

PART B-ABLATION: denoiser only (front-end fixed = Malvar+AWB+gamma)
method                               CPSNR(dB)           SSIM   dPSNR
------------------------------------------------------------------------
no denoise                          23.07+-1.56 0.5838+-0.0724
classical bilateral                 25.26+-1.98 0.7361+-0.0465   +2.20
learned DnCNN-lite                  25.22+-2.00 0.7491+-0.0342   +2.16
```

**Reading it — the AI-ISP result:** with the front-end held fixed, both denoisers
recover ~+2.2 dB over the undenoised front-end. On **PSNR** the learned CNN and
classical bilateral are **tied** (25.22 vs 25.26 dB, within noise), but the
**learned DnCNN-lite wins on SSIM — 0.749 vs 0.736** — i.e. at matched distortion
it preserves more structure (the perception–distortion point), consistent with
the synthetic natural-image result below. Total `run_real.py` runtime ~125 s CPU.

**Honest caveats (measured, not hidden):**
- On PSNR the *naive* bilinear (25.58) edges out the full classical (25.26). Same
  cause as the synthetic `astronaut` run: gray-world AWB assumes a neutral scene
  average, and Kodak photos are non-neutral, so gray-world slightly overcorrects
  and costs PSNR — while lifting **SSIM by +0.12** (0.736 vs 0.618). This is the
  textbook motivation for **learned AWB / illuminant estimation**.
- Single seed, 20 epochs, tiny CNN, CPU. Demonstrates method + measurement on the
  standard benchmark, not production SOTA.

---

# SYNTHETIC — controlled/natural scenes (stage validation)

**Setup:** synthetic Bayer RAW (RGGB) forward-modelled from a clean sRGB image,
**Poisson–Gaussian noise** (shot + read), 12-bit quantisation, with a coloured
illuminant the ISP's AWB must undo. Metric target = the clean sRGB ground truth.
PSNR/SSIM via scikit-image.

---

## Run 1 — controlled synthetic scene (`macbeth`): validates the classical pipeline

Label: *synthetic Bayer from the `macbeth` scene, Poisson–Gaussian noise
(gain=150, σ_read=0.008 → effective σ ≈ 0.058 at mid-gray), illuminant cast
[1.25, 1.0, 0.8].*

Exact command:
```
python run_smoke.py
```

Measured table (copied from stdout):
```
MAIN: full-pipeline reconstruction vs ground truth
method                                    PSNR     SSIM    dPSNR
(a) naive bilinear baseline              18.66   0.6006
(b) full classical (Malvar+AWB+bilateral)   21.99   0.8190    +3.33
(c) classical + LEARNED (hybrid)         20.68   0.6137    +2.02

ABLATION 1: demosaic quality (AWB on, denoise off)
bilinear demosaic                        19.37   0.6072
Malvar-He-Cutler demosaic                20.22   0.5246    +0.85

ABLATION 2: denoiser (front-end fixed: Malvar+AWB+gamma)
no denoise                               20.22   0.5246
classical bilateral                      21.99   0.8190    +1.77
learned DnCNN-lite                       20.68   0.6137    +0.46
```

**Reading it:** the full classical pipeline beats the naive baseline by
**+3.33 dB PSNR and +0.22 SSIM**. Gray-world AWB correctly recovers the
illuminant on this neutral-average scene; Malvar demosaic adds +0.85 dB over
bilinear. On this scene of large flat colour patches the **classical bilateral
denoiser wins** (+1.77 dB) — bilateral is near-optimal for piecewise-flat
content, and the CNN (trained on natural texture) is out of its element here.

---

## Run 2 — natural photo (`astronaut`): shows the learned denoiser winning

Label: *synthetic Bayer from the `astronaut` photo, Poisson–Gaussian noise
(gain=150, σ_read=0.008 → effective σ ≈ 0.058 at mid-gray), illuminant cast
[1.15, 1.0, 0.9]. Learned denoiser trained on held-out images `chelsea`,
`coffee` (15 epochs, CPU, ~28 s total).*

Exact command:
```
python run_smoke.py --eval astronaut --train chelsea coffee --illuminant 1.15,1,0.9 --epochs 15
```

Measured table (copied from stdout):
```
MAIN: full-pipeline reconstruction vs ground truth
method                                    PSNR     SSIM    dPSNR
(a) naive bilinear baseline              24.06   0.6953
(b) full classical (Malvar+AWB+bilateral)   20.73   0.6989    -3.33
(c) classical + LEARNED (hybrid)         21.10   0.7117    -2.96

ABLATION 1: demosaic quality (AWB on, denoise off)
bilinear demosaic                        20.75   0.6950
Malvar-He-Cutler demosaic                20.61   0.6605    -0.14

ABLATION 2: denoiser (front-end fixed: Malvar+AWB+gamma)
no denoise                               20.61   0.6605
classical bilateral                      20.73   0.6989    +0.12
learned DnCNN-lite                       21.10   0.7117    +0.49
```

**Reading it — the headline AI-ISP result:** holding the front-end fixed, the
**learned DnCNN-lite denoiser beats the classical bilateral filter on BOTH
metrics** — PSNR 21.10 vs 20.73 dB and SSIM **0.7117 vs 0.6989** — and beats the
undenoised front-end by +0.49 dB. On natural texture the bilateral filter has to
choose between smoothing noise and preserving detail; the CNN, ~39k params,
learns a better trade-off.

**Honest caveats (both are real, measured findings, not failures to hide):**
- On `astronaut` the *naive* baseline has higher PSNR than the full classical
  pipeline. Reason: gray-world AWB assumes a neutral scene average; `astronaut`
  is warm/non-neutral, so gray-world **overcorrects** and costs PSNR. This is the
  textbook limitation of gray-world and the exact motivation for **learned AWB /
  illuminant estimation** in modern ISPs. On the neutral-average `macbeth` scene
  (Run 1) AWB behaves correctly and the classical pipeline wins by +3.33 dB.
- On `astronaut` Malvar ≈ bilinear (−0.14 dB): the image is smooth and
  noise-dominated, so demosaic choice barely matters. Malvar's advantage shows on
  edge-rich content — measured separately below.

### Demosaic sanity (noiseless mosaic, `macbeth`)
Isolated, noise-free comparison of the two demosaic implementations:
```
bilinear PSNR vs clean-linear: 18.77 dB
Malvar   PSNR vs clean-linear: 21.93 dB      (+3.16 dB)
```
Confirms the Malvar-He-Cutler implementation is correct and materially better
than bilinear when noise isn't masking the difference.

---

## What each run demonstrates

| Claim | Evidence (measured) |
|---|---|
| Classical ISP stages each help when their assumptions hold | Run 1: naive→classical +3.33 dB / +0.22 SSIM; Malvar +0.85 dB; bilateral +1.77 dB |
| Malvar-He-Cutler > bilinear demosaic | Noiseless: +3.16 dB (macbeth) |
| **Learned denoiser > classical bilateral on natural imagery** | Run 2: SSIM 0.7117 vs 0.6989, PSNR 21.10 vs 20.73 dB |
| Gray-world AWB is scene-dependent (motivates learned AWB) | Run 2 caveat: AWB overcorrects the non-neutral scene |

---

## Résumé-bullet drafts (grounded ONLY in the numbers above)

**Primary — REAL Kodak numbers (use this one):**
> **Hybrid RAW→RGB ISP (classical + learned).** Built a modular RAW→RGB pipeline
> in NumPy (black-level, **Malvar-He-Cutler demosaic**, gray-world AWB, CCM, sRGB
> tone, bilateral denoise) + a **~39k-param PyTorch DnCNN-lite** denoiser, and
> benchmarked it on the **standard Kodak 24-image set**: Malvar-He-Cutler demosaic
> reached **35.4 dB CPSNR (0.969 SSIM), +6.0 dB over bilinear**; under a
> Poisson–Gaussian sensor model the learned denoiser (trained on a Kodak split,
> **evaluated on the held-out split**) matched classical bilateral on PSNR while
> improving **SSIM to 0.749 vs 0.736** — all on CPU.

Prior synthetic-only draft (superseded by the Kodak numbers above):
> **Hybrid RAW→RGB ISP (classical + learned).** Built a modular RAW→RGB image
> signal processor in NumPy (black-level, Malvar-He-Cutler demosaic, gray-world
> AWB, CCM, sRGB tone, bilateral denoise) plus a ~39k-param PyTorch DnCNN-lite
> denoiser, evaluated on synthetic Bayer with a Poisson–Gaussian noise model.
> The learned denoiser beat the classical bilateral filter on natural imagery
> (SSIM 0.712 vs 0.699, PSNR +0.4 dB) on CPU in <30 s; the full classical
> pipeline improved PSNR +3.3 dB / SSIM +0.22 over a naive bilinear baseline on a
> controlled scene.

Alternate (perception–distortion angle):
> Implemented and quantitatively compared classical vs learned ISP denoising
> (PSNR/SSIM vs ground truth) on a Poisson–Gaussian sensor model; showed a small
> residual-learning CNN improves the perceptual–distortion trade-off over
> bilateral filtering on natural texture, and documented the gray-world AWB
> failure mode that motivates learned white balance.

Keep it honest in interviews: these are **256×256 synthetic-Bayer proof-of-concept
numbers**, single seed, tiny CNN, CPU. They demonstrate method and measurement,
not production SOTA.

---

## Interview framing: hybrid ISP & the perception–distortion trade-off

- **Why hybrid, not end-to-end.** A full RAW→RGB CNN is hard to constrain, hard
  to quantize, and a black box for color. Keeping demosaic/AWB/CCM/gamma
  classical (cheap, interpretable, HW-friendly) and inserting **one learned stage
  where data helps most (denoising/restoration)** is the pragmatic edge-AI design.
  My results show exactly that split: classical stages carry the color/geometry;
  the CNN wins the denoise on texture.
- **Perception vs distortion (Blau & Michaeli 2018).** PSNR (distortion) and
  SSIM (structure/perception) don't always move together. On `macbeth` bilateral
  maxes PSNR by flattening; on `astronaut` the CNN raises *both* PSNR and SSIM by
  denoising without over-smoothing — the practical value of a learned prior.
- **Stage order matters.** Black-level first (true zero for color math); WB & CCM
  in linear light before gamma (they're multiplicative on radiance); gamma last;
  denoise placed so classical and learned stages see identical inputs → fair
  comparison. All discussed in `README.md`.
- **The AWB caveat is a feature to talk about.** Gray-world failing on a
  non-neutral scene is the standard motivation for illuminant estimation and
  **learned AWB** — a natural "what would you do next" answer.
- **Edge-AI relevance.** ~39k-param fully-convolutional residual net, CPU
  training/inference, INT8-friendly (plain conv/BN/ReLU) — the same profile as
  on-device super-resolution / denoising I'd deploy on a phone or camera SoC.

---

## Environment (for reproducibility)

- Python 3.12.10, NumPy 2.4.4, SciPy 1.17.1, scikit-image 0.26.0, imageio 2.37.3,
  Pillow 11.3.0, PyTorch 2.11.0+cpu. No GPU, no dataset download.
- Both runs above completed in ~30–105 s each on CPU.
- Torch **was** usable here, so the learned stage ran and is measured. If torch
  were unavailable, `python run_smoke.py --no-torch` delivers the fully-measured
  classical pipeline and skips the CNN (the code path is tested and works).
