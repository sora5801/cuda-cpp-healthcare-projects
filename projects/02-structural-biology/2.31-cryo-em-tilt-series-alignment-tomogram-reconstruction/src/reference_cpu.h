// ===========================================================================
// src/reference_cpu.h  --  Cryo-ET geometry, alignment, ramp filter, WBP (CPU)
// ---------------------------------------------------------------------------
// Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
//
// WHAT THIS PROJECT COMPUTES  (reduced-scope, 2-D teaching version)
//   In cryo electron tomography (cryo-ET) we tilt a frozen-hydrated specimen
//   to a series of angles in the microscope and record a 2-D projection image
//   at each tilt. From that "tilt series" we reconstruct the 3-D density of the
//   specimen. The classic, GPU-friendly reconstruction is Weighted Back-
//   Projection (WBP) -- mathematically the same inverse Radon transform that
//   drives CT filtered back-projection (project 4.01), with three twists that
//   make cryo-ET its own problem:
//
//     1. TILT-SERIES ALIGNMENT. Stage drift and beam-induced motion shift each
//        recorded projection by an unknown amount. Before reconstruction we must
//        ALIGN the projections to a common origin. We do a simplified, fiducial-
//        free 1-D translational alignment: cross-correlate each projection
//        against the untilted (0 deg) reference and undo the integer shift that
//        maximizes the correlation. (Production tools -- IMOD, AreTomo2 -- also
//        solve for rotation/magnification and use gold fiducial beads.)
//
//     2. ARBITRARY, LIMITED tilt angles. A CT scan spans a full 180 deg in even
//        steps; a cryo-ET tilt series spans only about +-60 deg (the holder
//        blocks higher tilts) at irregular angles. The cone of directions never
//        sampled is the famous MISSING WEDGE, which smears the reconstruction
//        along the beam axis. Our back-projection therefore loops over a list of
//        ARBITRARY tilt angles (not theta_k = k*pi/n).
//
//     3. RAMP (high-pass) FILTERING via FFT. Back-projection alone blurs the
//        density by 1/r in Fourier space; WBP first multiplies each projection's
//        spectrum by |frequency| (the "weighting" in Weighted Back-Projection).
//        We do this with cuFFT on the GPU (R2C -> ramp multiply -> C2R), the
//        exact CUDA library the catalog names for this project.
//
//   We reconstruct a 2-D slice (one detector row -> one image plane) so the
//   geometry stays legible; the 3-D tomogram is a stack of independent slices
//   built by exactly this kernel (see THEORY.md "Where this sits...").
//
// WHY A GPU
//   Back-projection is a per-PIXEL GATHER: each output pixel reads one
//   interpolated sample from every projection. A real tomogram is e.g.
//   1024 x 1024 x 256 voxels x ~40 tilts ~ 10^10 voxel-projection pairs -- the
//   archetypal embarrassingly-parallel, bandwidth-bound GPU workload. The ramp
//   filter is a batched FFT, also a GPU staple (cuFFT).
//
//   This header is PURE C++ (no CUDA) so the host compiler can build the
//   reference. The per-sample interpolation math shared by CPU and GPU lives in
//   "wbp_core.h" as __host__ __device__ inline functions (PATTERNS.md sec.2) so
//   both paths run identical arithmetic and verification is tight.
//
// READ THIS AFTER: wbp_core.h (the shared math), util/io.hpp.
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// One reconstruction problem: a tilt series plus the geometry needed to invert
// it back into a 2-D slice.
//   tilt[k]                 = tilt angle of projection k, in DEGREES (signed).
//   proj[k*n_det + j]       = recorded intensity of projection k at detector
//                             bin j (row-major; one 1-D projection per tilt).
//   The detector has n_det bins of spacing ds (world units). Bin j sits at
//   world offset s_j = (j - (n_det-1)/2) * ds along the (rotated) detector axis.
//   The reconstructed slice is `img` x `img` pixels covering the square world
//   region [-world_half, world_half]^2.
struct TiltSeries {
    int   n_tilts    = 0;     // number of projection images (tilt angles)
    int   n_det      = 0;     // detector bins per projection
    int   img        = 0;     // output slice side length (pixels)
    float ds         = 0.0f;  // detector bin spacing (world units)
    float world_half = 0.0f;  // slice spans [-world_half, world_half] in x and y
    std::vector<float> tilt;  // [n_tilts] tilt angles, DEGREES
    std::vector<float> proj;  // [n_tilts * n_det] recorded (shifted) projections
};

// ---- File I/O -------------------------------------------------------------
// Load a TiltSeries from the text format documented in data/README.md:
//   header line : "<n_tilts> <n_det> <ds> <img> <world_half>"
//   then n_tilts records, each: "<tilt_deg>  p_0 p_1 ... p_{n_det-1}".
// Throws std::runtime_error on a malformed/short file so demos fail loudly.
TiltSeries load_tilt_series(const std::string& path);

// ---- Step 1: tilt-series alignment ----------------------------------------
// Estimate, for every projection, the INTEGER detector shift (in bins) that
// best aligns it to the 0-degree reference projection by cross-correlation, and
// write those shifts into `shift` (length n_tilts; shift[ref]=0 by construction).
//   search : maximum |shift| in bins to test (a small window; +-search).
// This is the teaching core of "tilt-series alignment": find the lag that
// maximizes overlap. Returns the index of the reference (smallest |tilt|).
int estimate_shifts(const TiltSeries& ts, int search, std::vector<int>& shift);

// Apply integer shifts to produce an ALIGNED copy of the projections: row k is
// projection k translated by -shift[k] bins (so its features line up with the
// reference). Out-of-range bins are filled with 0. `aligned` is sized to proj.
void apply_shifts(const TiltSeries& ts, const std::vector<int>& shift,
                  std::vector<float>& aligned);

// ---- Step 2: ramp filter (CPU reference, spatial-domain) ------------------
// Ram-Lak ramp filter of each aligned projection row -- the "Weighted" in WBP.
// The CPU does it by direct convolution with the discrete ramp kernel; the GPU
// does the mathematically equivalent operation in the FREQUENCY domain with
// cuFFT (multiply the spectrum by |f|). We verify the two filtered sinograms
// agree (see main.cu). `filtered` is sized to the projection array.
void ramp_filter_cpu(const TiltSeries& ts, const std::vector<float>& aligned,
                     std::vector<float>& filtered);

// ---- Step 3: precompute trig of the tilt angles ---------------------------
// cos/sin of each (signed) tilt angle, computed ONCE on the host so the CPU and
// GPU back-projections use bit-identical trig (avoids cos vs cosf drift).
void compute_trig(const TiltSeries& ts, std::vector<float>& cosv,
                  std::vector<float>& sinv);

// ---- Step 4: CPU reference weighted back-projection -----------------------
// slice[py*img+px] = (pi/n_tilts) * sum_k interp(filtered_row_k, s),
//   where s = wx*cos(tilt_k) + wy*sin(tilt_k) is where pixel (px,py)'s ray
//   crosses projection k's detector. The trusted baseline the GPU is checked
//   against. `slice` is sized to img*img.
void backproject_cpu(const TiltSeries& ts, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& slice);
