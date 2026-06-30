// ===========================================================================
// src/reference_cpu.h  --  Subtomogram model, geometry helpers, CPU reference
// ---------------------------------------------------------------------------
// Project 2.4 : Cryo-ET Subtomogram Averaging  (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   The inner loop of SUBTOMOGRAM AVERAGING (STA). In cryo-electron tomography
//   the same molecular machine (a ribosome, a nuclear-pore complex, ...) appears
//   THOUSANDS of times inside one extremely noisy 3-D tomogram, each copy at a
//   different orientation. STA recovers a clean structure by (1) ALIGNING every
//   noisy copy ("subtomogram") to a common reference and (2) AVERAGING the
//   aligned copies -- noise averages out, signal reinforces.
//
//   The compute-heavy step we accelerate is the ALIGNMENT SEARCH: for each
//   candidate subtomogram we try a small set of trial ROTATIONS and, for each,
//   measure how well the rotated candidate matches the reference using
//   CROSS-CORRELATION over all translational shifts. The rotation+shift with the
//   highest correlation is the candidate's pose; we then average the best copies.
//
//   Cross-correlation over all 3-D shifts is what the GPU does in FOURIER SPACE
//   via the cross-correlation theorem (cuFFT) -- see kernels.cu. This header is
//   pure C++ (no CUDA), so it is shared by main.cu (nvcc) and reference_cpu.cpp
//   (host compiler), guaranteeing the CPU baseline and the GPU path model the
//   SAME geometry and the SAME correlation metric (PATTERNS.md §2).
//
// WHY THIS IS A REDUCED-SCOPE TEACHING VERSION (CLAUDE.md §13)
//   Production STA (RELION-4, Dynamo, emClarity) does a full 3-D orientation
//   search (3 Euler angles), Bayesian/CTF weighting, missing-wedge masking and
//   iterative refinement over millions of particles. We teach the load-bearing
//   idea -- FFT-based cross-correlation alignment + averaging -- on tiny 3-D
//   cubes with a discrete in-plane rotation search. THEORY.md "real world"
//   maps every simplification back to the full algorithm.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// A set of cubic subtomograms loaded from data/sample (see data/README.md).
//   File format (whitespace-separated floats):
//     header:  <n_sub> <d> <n_angles>
//       n_sub    : number of candidate subtomograms
//       d        : cube edge length in voxels (volume is d*d*d)
//       n_angles : number of trial in-plane rotation angles in the search
//     then ONE reference cube  (d*d*d floats, the current running average)
//     then n_sub candidate cubes (each d*d*d floats)
//
//   Voxels are stored z-major then y then x:  vox[(z*d + y)*d + x].
//   Intensities are arbitrary units (synthetic densities); the loader subtracts
//   each cube's mean so cross-correlation peaks are meaningful (see below).
// ---------------------------------------------------------------------------
struct SubtomogramSet {
    int n_sub = 0;                  // number of candidate subtomograms
    int d = 0;                      // cube edge (voxels); volume = d*d*d
    int n_angles = 0;               // trial rotation angles per candidate
    std::vector<float> ref;         // [d*d*d]        the reference (running average)
    std::vector<float> cand;        // [n_sub*d*d*d]  candidate cubes, row-major
    int vol() const { return d * d * d; }   // voxels per cube (convenience)
};

// ---------------------------------------------------------------------------
// The discrete in-plane rotation search.
//   We rotate each candidate about the cube's z-axis (the simplest non-trivial
//   3-D rotation to teach) by angle_k = k * (2*pi / n_angles), k = 0..n_angles-1.
//   trial_angle(k, n) returns that angle in radians. Declared here (defined in
//   reference_cpu.cpp) and ALSO recomputed identically inside the kernel, so the
//   host reference and the device kernel search the very same angles.
// ---------------------------------------------------------------------------
double trial_angle(int k, int n_angles);

// ---------------------------------------------------------------------------
// Loading & preprocessing
// ---------------------------------------------------------------------------

// Parse the data/sample text format above. Throws std::runtime_error on any
// malformed input so demos fail loudly rather than computing on garbage.
SubtomogramSet load_subtomograms(const std::string& path);

// Subtract the mean of a cube in place. Cross-correlation of zero-mean signals
// is what makes the *peak* meaningful (a constant DC offset would otherwise
// dominate every shift). Both reference and every candidate are zero-meaned on
// load. vol = number of voxels in the cube (d*d*d).
void normalize_zero_mean(std::vector<float>& cube, int vol);

// ---------------------------------------------------------------------------
// Geometry: rotate a cube in-plane by `theta` radians about its z-axis.
//   Output voxel (x,y,z) samples the INPUT at the point obtained by rotating the
//   (x,y) offset-from-center by -theta (the inverse map, so the output grid is
//   fully covered), using bilinear interpolation in x,y (z is unchanged).
//   Out-of-bounds samples read 0. This is the "rotate the candidate" step of the
//   alignment search; the GPU kernel rotate_kernel() mirrors it voxel-for-voxel.
//
//   in/out  : [d*d*d] cubes (out must be distinct storage from in)
//   d       : cube edge length
//   theta   : rotation angle (radians), counter-clockwise in the x-y plane
// ---------------------------------------------------------------------------
void rotate_cube_cpu(const float* in, float* out, int d, double theta);

// ---------------------------------------------------------------------------
// CPU reference correlation: the TRUSTED baseline the GPU is checked against.
//
//   For one candidate, for each trial angle:
//     1. rotate the candidate by that angle (rotate_cube_cpu),
//     2. compute the NORMALIZED CROSS-CORRELATION (NCC) at ZERO shift between
//        the rotated candidate and the reference:
//            NCC = sum(ref .* rot) / sqrt(sum(ref^2) * sum(rot^2))
//        (both are zero-mean, so this is the Pearson correlation coefficient).
//   The best (highest NCC) angle is the candidate's recovered orientation.
//
//   We deliberately score at ZERO translational shift on the CPU so the
//   reference is short and obviously correct; the GPU additionally searches ALL
//   translational shifts via FFT (cross-correlation theorem) and we verify the
//   GPU's zero-shift value matches this CPU number. See THEORY "verify".
//
//   Outputs (sized by this function):
//     ncc_zero_shift[s*n_angles + k] = NCC of candidate s at angle k, zero shift
//     best_angle[s]                  = argmax_k of that candidate's NCC (ties ->
//                                      lowest k, so the result is deterministic)
// ---------------------------------------------------------------------------
void correlate_cpu(const SubtomogramSet& set,
                   std::vector<double>& ncc_zero_shift,
                   std::vector<int>& best_angle);

// ---------------------------------------------------------------------------
// Build the refined average from the best-aligned candidates.
//   Each candidate is rotated by ITS chosen angle, then all are averaged voxel-
//   wise. This is the "averaging" half of STA: the output is the new, less-noisy
//   reference. We also return a single deterministic scalar -- the mean absolute
//   intensity of the averaged cube -- so stdout has a stable number to print.
//
//   best_angle : per-candidate angle index chosen by correlate_cpu / the GPU.
//   out_avg    : [d*d*d] the averaged cube (resized + filled by this function).
//   returns    : mean(|voxel|) of the averaged cube (a deterministic scalar).
// ---------------------------------------------------------------------------
double build_average_cpu(const SubtomogramSet& set,
                         const std::vector<int>& best_angle,
                         std::vector<float>& out_avg);
