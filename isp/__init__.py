"""Classical RAW -> RGB mini-ISP.

Each stage of the image signal processor lives in its own module so the pipeline
is easy to read, reorder and interview about. The canonical stage order is:

    black-level subtraction
    -> lens-shading correction (optional)
    -> demosaic (bilinear or Malvar-He-Cutler)
    -> auto white balance (gray-world)
    -> color correction matrix (CCM)
    -> tone / gamma
    -> denoise (bilateral)

`pipeline.run_classical_pipeline` wires them together.
"""
from . import black_level, lens_shading, demosaic, white_balance, ccm, tone, denoise, pipeline  # noqa: F401
