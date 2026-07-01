#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic fMRI sample dataset
# ---------------------------------------------------------------------------
# Project 4.16 : Functional MRI Analysis
#
# WHY THIS EXISTS
#   Real fMRI (HCP, OpenNeuro, ABIDE, UK Biobank -- see data/README.md) is large,
#   in NIfTI format, and often needs registration/credentials. To keep the demo
#   RUNNABLE OFFLINE with zero dependencies, this script deterministically
#   generates a CLEARLY-SYNTHETIC stand-in in the loader's plain-text layout.
#   Synthetic data is always LABELED synthetic (in the file header comment,
#   data/README.md, and the program output).
#
# WHAT IT GENERATES
#   A tiny "brain" of V voxels, each a T-scan BOLD time-series built as:
#       y[t] = baseline + drift*ramp(t) + (amp * task_regressor[t] if active) + noise
#   The task_regressor is the SAME boxcar-convolved-with-HRF used by the C++ GLM
#   (glm.h), so ACTIVE voxels carry a genuine task response and the GLM recovers
#   them as the top-t voxels -- an interpretable, verifiable planted answer
#   (docs/PATTERNS.md §6). Noise is a small deterministic LCG (no numpy needed),
#   so re-running reproduces the sample byte-for-byte and expected_output.txt
#   stays stable.
#
#   File layout (whitespace-separated; matches load_fmri in reference_cpu.cpp):
#       V  T  TR_seconds  block_scans
#       then V rows:  active_flag  y_0 y_1 ... y_{T-1}
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --V 64 --T 120  # a bigger synthetic brain
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "fmri_sample.txt"


# --- The HRF + design, mirrored EXACTLY from src/glm.h so active voxels carry --
# --- the same regressor the C++ GLM fits. Keep these in lockstep with glm.h.  --
def gamma_pdf(t, shape, rate):
    if t <= 0.0:
        return 0.0
    log_pdf = (shape * math.log(rate)
               + (shape - 1.0) * math.log(t)
               - rate * t
               - math.lgamma(shape))
    return math.exp(log_pdf)


def canonical_hrf(t):
    if t <= 0.0:
        return 0.0
    return gamma_pdf(t, 6.0, 1.0) - (1.0 / 6.0) * gamma_pdf(t, 16.0, 1.0)


def boxcar_on(scan, block_scans):
    return 1 if ((scan // block_scans) & 1) == 0 else 0


def task_regressor(ti, tr, block_scans):
    acc = 0.0
    for k in range(ti + 1):
        if boxcar_on(k, block_scans):
            acc += canonical_hrf((ti - k) * tr)
    return acc


class LCG:
    """A tiny 64-bit linear congruential generator (Numerical Recipes constants)
    so noise is deterministic and dependency-free -> reproducible sample."""
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next_unit(self):
        # Advance and return a float in [-1, 1).
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.state >> 11) / float(1 << 53) * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic fMRI GLM sample.")
    ap.add_argument("--V", type=int, default=48, help="number of voxels")
    ap.add_argument("--T", type=int, default=80, help="number of scans (time points)")
    ap.add_argument("--tr", type=float, default=2.0, help="repetition time (seconds/scan)")
    ap.add_argument("--block", type=int, default=10, help="task boxcar half-period (scans)")
    ap.add_argument("--amp", type=float, default=6.0, help="task response amplitude (active voxels)")
    ap.add_argument("--noise", type=float, default=1.2, help="noise standard-deviation scale")
    ap.add_argument("--seed", type=int, default=20260630, help="LCG seed (reproducibility)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    V, T, tr, block = args.V, args.T, args.tr, args.block
    rng = LCG(args.seed)

    # Precompute the shared task regressor once (voxel-independent).
    reg = [task_regressor(t, tr, block) for t in range(T)]

    # Every 6th voxel is "active" -> a sparse, interpretable activation map.
    active = [1 if (v % 6 == 0) else 0 for v in range(V)]

    lines = [f"{V} {T} {tr:g} {block}"]
    for v in range(V):
        baseline = 100.0                      # typical BOLD baseline units
        drift = 0.8 * (v % 3 - 1)             # small per-voxel linear drift
        amp = args.amp if active[v] else 0.0
        row = []
        for t in range(T):
            ramp = (2.0 * t / (T - 1) - 1.0) if T > 1 else 0.0   # matches design_column1
            noise = args.noise * rng.next_unit()
            y = baseline + drift * ramp + amp * reg[t] + noise
            row.append(f"{y:.4f}")
        lines.append(f"{active[v]} " + " ".join(row))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_active = sum(active)
    print(f"[make_synthetic] wrote {args.out}  "
          f"(V={V}, T={T}, TR={tr}s, block={block}, {n_active} active; SYNTHETIC)")


if __name__ == "__main__":
    main()
