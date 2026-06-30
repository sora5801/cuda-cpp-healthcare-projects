// ===========================================================================
// src/reference_cpu.h  --  Micrograph model, loader, and CPU CTF-fit reference
// ---------------------------------------------------------------------------
// Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
//
// WHAT THIS PROJECT COMPUTES
//   Given a cryo-EM micrograph (a noisy 2-D electron image), estimate the
//   microscope's DEFOCUS by:
//     1. Computing the micrograph's 2-D power spectrum  |FFT(image)|^2.
//     2. Rotationally averaging it into a 1-D RADIAL PROFILE (the "Thon-ring"
//        profile), with a smooth background subtracted so only the rings remain.
//     3. GRID-SEARCHING over candidate defocus values, scoring each by the
//        normalized cross-correlation between the model |CTF(k)|^2 and the
//        observed profile, and reporting the argmax defocus.
//   This is a reduced-scope teaching version of CTF estimation (CTFFIND4 /
//   RELION CtfFind). The full pipeline also estimates astigmatism (defocus along
//   two axes) and does particle picking; we describe those in THEORY "real world".
//
// WHY A GPU
//   Stage 1 is a 2-D FFT -- a solved problem we hand to cuFFT (NOT a black box:
//   kernels.cu documents exactly what cufftExecR2C computes). Stages 2 and 3 are
//   embarrassingly parallel: the radial average is one thread per pixel scattering
//   into ring bins, and the defocus search is one thread per CANDIDATE defocus
//   (the independent-jobs pattern, PATTERNS.md §1). A real facility fits thousands
//   of micrographs per session, each over hundreds of candidate defoci -- the GPU
//   turns minutes into seconds.
//
//   This header is PURE C++ (no CUDA), so both cl.exe and nvcc can include it.
//   The actual per-frequency physics lives in ctf_model.h (shared host+device).
//
// READ THIS AFTER: ctf_model.h.  READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "ctf_model.h"   // CtfParams + the shared __host__ __device__ physics

// ---------------------------------------------------------------------------
// Micrograph: a square N x N single-precision image plus the optics constants
// needed to interpret its frequencies. Stored row-major: pix[y*N + x].
//   We keep the pixels as float because cuFFT's real transform is single
//   precision (cufftReal == float); the CPU reference casts to double internally.
// ---------------------------------------------------------------------------
struct Micrograph {
    int n = 0;                  // side length (pixels); square, power of two
    CtfParams optics;           // lambda, cs, amp_contrast, pixel_size, n
    std::vector<float> pix;     // [n*n] image, row-major
    // Ground-truth defocus stored in the synthetic sample's header so the demo
    // can report "recovered vs true". -1 means "unknown / real data".
    double true_dz = -1.0;
};

// ---------------------------------------------------------------------------
// CtfFitConfig: the defocus search grid and the radial fitting band. Kept in one
// struct so the CPU and GPU paths are guaranteed to search the IDENTICAL grid.
// ---------------------------------------------------------------------------
struct CtfFitConfig {
    double dz_min;   // smallest candidate defocus (A)
    double dz_max;   // largest candidate defocus (A)
    int    n_dz;     // number of candidates (grid points, inclusive of both ends)
    int    nbins;    // number of radial bins (== n/2)
    int    r_lo;     // first radial bin used for fitting (skip DC spike)
    int    r_hi;     // one-past-last radial bin used for fitting (skip noisy tail)
};

// ---------------------------------------------------------------------------
// CtfFitResult: what either path returns. `scores[i]` is the NCC of candidate i;
// best_idx is its argmax; best_dz is the defocus of that candidate.
// ---------------------------------------------------------------------------
struct CtfFitResult {
    std::vector<double> scores;  // [n_dz] NCC per candidate defocus
    int    best_idx = -1;        // argmax over scores
    double best_dz  = 0.0;       // dz_min + best_idx * step
};

// dz_of_index: map a grid index i in [0, n_dz) to its defocus (A). Inline so the
// loader, the CPU fitter, and main all agree on the grid spacing.
inline double dz_of_index(const CtfFitConfig& c, int i) {
    if (c.n_dz <= 1) return c.dz_min;
    const double step = (c.dz_max - c.dz_min) / (c.n_dz - 1);
    return c.dz_min + step * i;
}

// ---------------------------------------------------------------------------
// load_micrograph: read the text sample format (see data/README.md):
//   line 1:  n  pixel_size  lambda  cs  amp_contrast  true_dz
//   then n*n whitespace-separated floats (row-major pixels).
// Throws std::runtime_error on any malformed input so the demo fails loudly.
// ---------------------------------------------------------------------------
Micrograph load_micrograph(const std::string& path);

// ---------------------------------------------------------------------------
// radial_power_profile_cpu: stages 1+2 on the CPU (the trusted reference).
//   * Removes the image mean (kills the DC term so it does not swamp the rings).
//   * Computes the full 2-D DFT power spectrum |X|^2 via a STRAIGHTFORWARD (slow)
//     transform -- transparently correct, the whole point of a reference.
//   * Rotationally averages |X|^2 into `nbins` radial bins.
//   * Subtracts a smooth running-mean background so only the oscillating ring
//     signal remains (CTF fitting matches ring POSITIONS, not the envelope).
//   Output: prof[0..nbins-1], the background-flattened radial profile.
//
// NOTE: a naive 2-D DFT is O(N^4) and would be far too slow for a real 4k x 4k
// micrograph; that is precisely why the GPU uses cuFFT (O(N^2 log N)). For our
// tiny teaching image (N is small, see data/sample) it runs in well under a
// second and gives the exact spectrum to verify cuFFT against.
// ---------------------------------------------------------------------------
void radial_power_profile_cpu(const Micrograph& m, int nbins, std::vector<double>& prof);

// ---------------------------------------------------------------------------
// flatten_background: shared post-processing applied to BOTH the CPU and GPU
// radial profiles so the comparison is apples-to-apples. Subtracts a windowed
// running mean (a smooth low-pass background estimate) from the raw radial power,
// leaving the oscillating Thon-ring signal centred on zero. Declared here so
// main.cu can apply the identical step to the cuFFT profile.
//   raw  : input radial power (length nbins)
//   win  : half-window (bins) of the running-mean background
//   out  : flattened profile (length nbins)
// ---------------------------------------------------------------------------
void flatten_background(const std::vector<double>& raw, int win, std::vector<double>& out);

// ---------------------------------------------------------------------------
// fit_ctf_cpu: stage 3 on the CPU. Given a radial profile and the search config,
// score every candidate defocus with ncc_model_vs_profile() (from ctf_model.h)
// and return the full score curve + argmax. This is the exact computation the GPU
// kernel parallelizes, one thread per candidate.
// ---------------------------------------------------------------------------
CtfFitResult fit_ctf_cpu(const std::vector<double>& prof, const CtfParams& optics,
                         const CtfFitConfig& cfg);
