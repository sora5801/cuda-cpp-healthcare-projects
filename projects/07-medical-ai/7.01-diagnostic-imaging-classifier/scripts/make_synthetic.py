#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 7.1 : Diagnostic Imaging Classifier   (REDUCED-SCOPE teaching version)
#
# WHY THIS EXISTS
#   The real datasets named in the catalog (MIMIC-CXR, CheXpert, LIDC-IDRI, TCIA)
#   require registration/credentials and forbid casual redistribution, so we CANNOT
#   commit them. To keep the demo fully offline we deterministically generate a
#   clearly-SYNTHETIC stand-in: a tiny model (fixed, hand-designed weights) plus a
#   small batch of grayscale "image patches" with known labels. Everything here is
#   synthetic and labeled as such (CLAUDE.md section 8); it is NOT medical data.
#
#   This MIRRORS make_builtin() in src/reference_cpu.cpp so the committed sample and
#   the built-in fallback produce the same result -- the file is the inspectable,
#   editable copy; the C++ fallback keeps the program runnable with no arguments.
#
# FILE LAYOUT (whitespace-separated floats; parsed by load_sample in reference_cpu.cpp)
#     n
#     conv_w  [NUM_F*KERNEL*KERNEL]     (filter taps, row-major per filter)
#     conv_b  [NUM_F]
#     dense_w [NUM_CLS*FLAT]            (dense weights, row-major per class)
#     dense_b [NUM_CLS]
#     then n times:
#         label                         (0 normal, 1 lesion, -1 unknown)
#         pixels [IMG_H*IMG_W]          (row-major, in [0,1])
#
# USAGE
#   python scripts/make_synthetic.py                 # -> data/sample/imaging_sample.txt
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

# ---- Model geometry: MUST match the constants in src/reference_cpu.h ----------
IMG_H, IMG_W = 16, 16
NUM_F        = 4
KERNEL       = 3
CONV_H       = IMG_H - KERNEL + 1        # 14
CONV_W       = IMG_W - KERNEL + 1        # 14
POOL         = 2
POOL_H       = CONV_H // POOL            # 7
POOL_W       = CONV_W // POOL            # 7
FLAT         = NUM_F * POOL_H * POOL_W   # 196
NUM_CLS      = 2

ROOT = Path(__file__).resolve().parent.parent            # the project folder
OUT  = ROOT / "data" / "sample" / "imaging_sample.txt"


def build_weights():
    """Fixed, hand-designed 'lesion detector' weights (see reference_cpu.cpp)."""
    # Conv filters (row-major, 4 filters x 9 taps).
    lap = [-1, -1, -1, -1, 8, -1, -1, -1, -1]            # filter 0: center-surround
    sx  = [-1, 0, 1, -2, 0, 2, -1, 0, 1]                 # filter 1: Sobel-x
    sy  = [-1, -2, -1, 0, 0, 0, 1, 2, 1]                 # filter 2: Sobel-y
    bl  = [t / 9.0 for t in [1, 1, 1, 1, 1, 1, 1, 1, 1]] # filter 3: box blur
    conv_w = lap + sx + sy + bl
    conv_b = [0.0] * NUM_F

    # Dense layer: reward filter-0 (Laplacian) pooled energy for class 1 (lesion),
    # penalize it for class 0 (normal). Only the first POOL_H*POOL_W features
    # (filter 0's block) are weighted; the rest stay zero.
    dense_w = [0.0] * (NUM_CLS * FLAT)
    for j in range(POOL_H * POOL_W):
        dense_w[1 * FLAT + j] = 0.25                     # class 1: lesion
        dense_w[0 * FLAT + j] = -0.25                    # class 0: normal
    dense_b = [0.5, 0.0]                                 # small prior toward normal
    return conv_w, conv_b, dense_w, dense_b


def blob_image(bg, fg):
    """A patch with a centered 4x4 bright blob (a synthetic 'lesion')."""
    im = [bg] * (IMG_H * IMG_W)
    for y in range(6, 10):
        for x in range(6, 10):
            im[y * IMG_W + x] = fg
    return im


def flat_image(v):
    """A uniform, low-contrast patch (synthetic 'normal' tissue)."""
    return [v] * (IMG_H * IMG_W)


def gradient_image(lo, hi):
    """A gentle horizontal gradient (normal tissue with mild shading)."""
    return [lo + (hi - lo) * ((k % IMG_W) / IMG_W) for k in range(IMG_H * IMG_W)]


def build_dataset():
    """4 labeled images: 2 lesions, 2 normals (matches make_builtin)."""
    imgs = [
        (1, blob_image(0.10, 0.95)),     # strong lesion
        (1, blob_image(0.15, 0.80)),     # weaker lesion
        (0, flat_image(0.20)),           # flat normal
        (0, gradient_image(0.10, 0.30)), # gradient normal
    ]
    return imgs


def fmt(vals):
    return " ".join(f"{v:.6g}" for v in vals)


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic imaging sample.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    conv_w, conv_b, dense_w, dense_b = build_weights()
    imgs = build_dataset()

    lines = [str(len(imgs)), fmt(conv_w), fmt(conv_b), fmt(dense_w), fmt(dense_b)]
    for label, pixels in imgs:
        lines.append(str(label))
        lines.append(fmt(pixels))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({len(imgs)} images, {IMG_H}x{IMG_W}; SYNTHETIC, not medical data)")


if __name__ == "__main__":
    main()
