# Mini-ISP: From Raw Sensor Data to a Photo

> A small, runnable camera pipeline that turns raw sensor data into a clean color image, built two ways (classical image processing and a tiny neural network) and compared with real numbers. Runs on CPU in about 2 minutes.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20only-orange.svg)
![Runtime](https://img.shields.io/badge/runtime-~2%20min-green.svg)
![Benchmark](https://img.shields.io/badge/benchmark-Kodak%2024-lightgrey.svg)

## Results

All numbers are measured by running this code (CPU only, single seed). PSNR and SSIM both measure closeness to the ground-truth image; higher is better.

### A. Demosaicing on the Kodak 24 benchmark (no noise)

The standard yardstick for demosaicing: turn each image into a Bayer mosaic, reconstruct it, and score against the original. Mean ± std over all 24 images.

| Method | CPSNR (dB) ↑ | SSIM ↑ | Notes |
| --- | --- | --- | --- |
| Bilinear | 29.37 ± 2.77 | 0.892 ± 0.039 | Fast, but zipper edges and color fringing |
| **Malvar-He-Cutler** | **35.36 ± 2.88** | **0.969 ± 0.008** | **+5.98 dB**, in line with published numbers |

### B. Full pipeline with sensor noise (Kodak, held-out images)

Noisy raw frames are simulated with a Poisson-Gaussian sensor model. The neural network (38,883 parameters) is trained on 12 Kodak images and tested on the other 12, so it is never evaluated on images it has seen.

| Method | CPSNR (dB) ↑ | SSIM ↑ | Notes |
| --- | --- | --- | --- |
| Naive bilinear | 25.58 ± 1.29 | 0.618 ± 0.041 | Basic conversion, no corrections |
| Full classical | 25.26 ± 1.98 | 0.736 ± 0.047 | All 7 stages, bilateral denoise |
| **Hybrid (classical + CNN)** | 25.22 ± 2.00 | **0.749 ± 0.034** | Learned denoiser swapped in |

### Denoiser comparison (everything else held fixed)

| Denoiser | CPSNR (dB) ↑ | SSIM ↑ | Gain over none |
| --- | --- | --- | --- |
| None | 23.07 ± 1.56 | 0.584 ± 0.072 | n/a |
| Classical bilateral filter | 25.26 ± 1.98 | 0.736 ± 0.047 | +2.20 dB |
| **Learned (DnCNN-lite)** | 25.22 ± 2.00 | **0.749 ± 0.034** | +2.16 dB |

**Takeaways**

- Both denoisers recover about **2.2 dB** over no denoising. On PSNR they are tied; the learned one keeps **more structure** (higher SSIM).
- On a natural photo (`astronaut`) the learned denoiser beats the bilateral filter on **both** metrics (21.10 vs 20.73 dB, SSIM 0.712 vs 0.699). On a scene of flat color patches (`macbeth`), the classical filter wins, because bilateral filtering is near-optimal for flat content.
- **Honest caveat:** the naive baseline scores slightly higher PSNR than the full classical pipeline. Gray-world white balance assumes the scene averages to neutral gray, and Kodak photos usually don't, so it over-corrects. SSIM still improves a lot (0.736 vs 0.618). This is the textbook reason modern ISPs use learned white balance.
- Single seed, 20 epochs, tiny CNN, CPU. This demonstrates the method and measurement, not production state of the art.

Full details, every run, and exact commands are in [`RESULTS.md`](RESULTS.md).

---

## How the pipeline works

Each step fixes one specific problem with the raw sensor data:

```
 Raw sensor data (one color per pixel, noisy)
        │
        ▼
 1. Black level      "Zero" on the sensor isn't really zero. Subtract the baseline.
        ▼
 2. Lens shading     Lenses make corners darker. Brighten them back up. (optional)
        ▼
 3. Demosaic         Each pixel only saw one color. Estimate the other two.
        ▼
 4. White balance    Remove the color cast from the lighting (gray-world method).
        ▼
 5. Color matrix     Convert the sensor's colors to standard display colors.
        ▼
 6. Gamma            Encode brightness the way screens and eyes expect (sRGB).
        ▼
 7. Denoise          Clean up the noise  ◄── classical filter OR learned CNN
        ▼
 Final color image  →  scored against ground truth (PSNR / SSIM)
```

<details>
<summary><b>Why this order? (click to expand)</b></summary>

- **Black level first.** Every color calculation needs a true zero.
- **Demosaic before color.** White balance and color correction need all three colors at every pixel.
- **White balance and color matrix before gamma.** These assume brightness is *linear* (proportional to light). Applying them after gamma would shift hues and exposure.
- **Gamma near the end.** Everything upstream is linear; we convert for the display last.
- **Denoise at the end (in this project).** The classical filter and the neural network then see identical inputs, so the comparison is fair. Real ISPs often denoise earlier, directly on raw data, and this design supports that too.

</details>

### The learned part

The neural network (`learned/model.py`) is a small **DnCNN-style** denoiser with 6 convolution layers. Instead of predicting the clean image directly, it predicts *the noise* and subtracts it. This "residual" approach trains quickly on very little data, which is what makes a CPU-only demo possible.

It is trained only as a denoiser, so it drops in exactly where the classical filter sits. Color and demosaicing stay with the classical stages.

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
python run_smoke.py
```

Builds a synthetic test scene, simulates a noisy sensor, runs every pipeline variant, and prints comparison tables.

### 3. Try it on a natural photo

```bash
python run_smoke.py --eval astronaut --train chelsea coffee --illuminant 1.15,1,0.9
```

### 4. Reproduce the Kodak benchmark

```bash
python run_real.py                 # downloads Kodak, runs Parts A and B
python run_real.py --limit 8       # quick version, first 8 images only
python run_real.py --no-torch      # classical only, skip the neural net
python run_real.py --skip-hybrid   # just the noiseless demosaic comparison
```

If the download is blocked, the script prints `NOT RUN - DOWNLOAD BLOCKED` rather than inventing numbers. Drop the PNGs into `data/kodak/` yourself and re-run.

### Handy options

| Option | What it does |
| --- | --- |
| `--gain 120` | Sensor gain. Lower gain means more noise. |
| `--sigma-read 0.01` | Amount of electronic read noise |
| `--epochs 40` | Training length for the neural network |
| `--save-images out/` | Save output images to a folder |
| `--no-torch` | Skip the neural network (classical only) |

---

## Using your own data

The pipeline takes a single-channel Bayer array scaled to `[0, 1]`, so real camera data plugs in directly:

| Dataset | Use | Status |
| --- | --- | --- |
| **Kodak 24** | Standard demosaicing benchmark | Done (`run_real.py`) |
| **McMaster (18 images)** | Harder, sharper-edged demosaicing test | Next step. Same protocol as Kodak; expect lower scores. |
| **SIDD / DND** | Real smartphone noise, the honest denoising upgrade | Load RAW with `rawpy`, set the real black level, white level, and Bayer pattern in `ISPConfig`, then retrain the CNN on real noisy/clean pairs. |
