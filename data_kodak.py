"""Download and locate the Kodak 24-image demosaicing benchmark.

The Kodak PhotoCD set (24 images, 768x512, lossless PNG) is the recognised
yardstick for demosaicing: you mosaic each high-quality RGB image to Bayer and
score the reconstruction. It is small (~15 MB total) and freely downloadable.

We try a few well-known mirrors in order. If ALL are blocked, the caller is told
so it can print instructions and refuse to invent numbers (see run_real.py).

Sibling set for later scaling: McMaster (18 images) — same protocol, harder,
edge-rich content. Real *raw* noise benchmarking would use SIDD/DND. Both noted
in the README; this module only fetches Kodak.
"""
import os
import ssl
import time
import urllib.request

# 24 images, canonical names kodim01.png .. kodim24.png
KODAK_NAMES = [f"kodim{ i:02d}.png" for i in range(1, 25)]

# Mirrors: each is a format string with {name}. Tried in order per image.
MIRRORS = [
    "https://r0k.us/graphics/kodak/kodak/{name}",
    "http://r0k.us/graphics/kodak/kodak/{name}",
    "https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/{name}",
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (hybrid-isp-benchmark)"}


def _ssl_ctx():
    ctx = ssl.create_default_context()
    # Some mirrors have flaky certs; we only fetch public PNGs, so tolerate it.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download_one(name, dest_dir, timeout=60, retries=2):
    dest = os.path.join(dest_dir, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 10000:
        return dest, None
    last_err = None
    for tmpl in MIRRORS:
        url = tmpl.format(name=name)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=_HEADERS)
                data = urllib.request.urlopen(
                    req, timeout=timeout, context=_ssl_ctx()).read()
                if len(data) < 10000:
                    raise IOError(f"suspiciously small ({len(data)} bytes)")
                with open(dest, "wb") as f:
                    f.write(data)
                return dest, None
            except Exception as e:  # noqa: BLE001
                last_err = f"{url} -> {e!r}"
                time.sleep(0.5 * (attempt + 1))
    return None, last_err


def download_kodak(dest_dir=None, verbose=True):
    """Ensure the 24 Kodak PNGs exist locally; return (paths, errors).

    ``paths``  : list of absolute paths that are present (may be < 24 if some
                 downloads failed).
    ``errors`` : list of (name, last_error) for images that could not be fetched.
    """
    if dest_dir is None:
        dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "kodak")
    os.makedirs(dest_dir, exist_ok=True)

    paths, errors = [], []
    for name in KODAK_NAMES:
        path, err = _download_one(name, dest_dir)
        if path is not None:
            paths.append(os.path.abspath(path))
        else:
            errors.append((name, err))
        if verbose:
            done = len(paths)
            tag = "ok " if path else "FAIL"
            print(f"  [{tag}] {name}   ({done}/24 present)")
    return paths, errors


if __name__ == "__main__":
    ps, errs = download_kodak()
    print(f"\n{len(ps)}/24 Kodak images available")
    if errs:
        print("failed:")
        for n, e in errs:
            print(f"  {n}: {e}")
