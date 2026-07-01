// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for DECT decomposition
// ---------------------------------------------------------------------------
// Project 4.20 : Dual-Energy / Spectral CT Reconstruction
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the dual-energy sinogram container,
//   the spectral-model builder, the file loader) and the CPU reference prototype
//   live here. kernels.cuh also includes this header so the GPU side reuses the
//   exact same types -- nothing CUDA-specific leaks in either direction. The
//   per-bin PHYSICS is one level down in dect.h (the __host__ __device__ core).
//
// THE PROBLEM (one line; full derivation in dect.h and ../THEORY.md)
//   A dual-energy scan gives, for each of n sinogram bins, TWO measured
//   log-attenuations (m_lo at ~80 kVp, m_hi at ~140 kVp). We recover, per bin,
//   the TWO basis-material path lengths (t1 = water-equivalent cm, t2 = iodine-
//   equivalent cm) by solving a 2x2 nonlinear system with Newton's method. Every
//   bin is INDEPENDENT -> one GPU thread per bin (kernels.cu).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Physics is in dect.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "dect.h"   // SpectralModel, DecompResult, decompose_bin (pure C++/HD)

// ---------------------------------------------------------------------------
// Newton solver settings shared by the CPU reference and the GPU kernel so both
// take the EXACT same iteration path (hence bit-identical results).
//   MAX_NEWTON_ITER : hard iteration cap (safety). Quadratic convergence reaches
//                     tol in ~5-8 steps here, so this is rarely hit.
//   NEWTON_TOL      : residual threshold in log-attenuation units. 1e-12 is near
//                     double-precision floor for these O(1) magnitudes.
// ---------------------------------------------------------------------------
constexpr int    MAX_NEWTON_ITER = 50;
constexpr double NEWTON_TOL      = 1.0e-12;

// ---------------------------------------------------------------------------
// DectSinogram: a loaded dual-energy sinogram.
//   n        : number of bins (rays). In this teaching sample n is tiny; a real
//              scan has ~10^8 bins, all decomposed by the same code.
//   m_lo[i]  : measured log-attenuation of bin i at the LOW-kVp spectrum.
//   m_hi[i]  : measured log-attenuation of bin i at the HIGH-kVp spectrum.
//   true_t1/true_t2 (optional, size n or 0): the ground-truth path lengths the
//              synthetic data was generated from. Present for synthetic samples
//              so the demo can report recovery error against a KNOWN answer
//              (PATTERNS.md §6); empty for real data where truth is unknown.
// ---------------------------------------------------------------------------
struct DectSinogram {
    int n = 0;
    std::vector<double> m_lo;      // [n]
    std::vector<double> m_hi;      // [n]
    std::vector<double> true_t1;   // [n] or empty
    std::vector<double> true_t2;   // [n] or empty
};

// ---------------------------------------------------------------------------
// build_spectral_model: construct the fixed scanner physics (the two spectra and
// the two basis-material attenuation curves) used to generate AND decompose the
// data. Defined in reference_cpu.cpp. It is deterministic and identical on every
// run, so the demo output is reproducible. See THEORY "The science" for how the
// analytic curves approximate real 80/140 kVp spectra and water/iodine mu(E).
// ---------------------------------------------------------------------------
SpectralModel build_spectral_model();

// ---------------------------------------------------------------------------
// load_sinogram: parse the tiny text dataset (format documented in
// data/README.md):
//   line 1 : "<n> <has_truth>"   (has_truth = 1 if truth columns follow)
//   next n : "<m_lo> <m_hi>"  or  "<m_lo> <m_hi> <true_t1> <true_t2>"
// Throws std::runtime_error on a missing file or a malformed line.
// ---------------------------------------------------------------------------
DectSinogram load_sinogram(const std::string& path);

// ---------------------------------------------------------------------------
// linear_init: a cheap starting guess for Newton, from the LINEARISED problem.
//   At small path lengths the forward model f_e ~ t1*mubar1_e + t2*mubar2_e,
//   where mubar are the spectrum-averaged attenuation coefficients. Solving that
//   2x2 LINEAR system gives a guess close enough for Newton to converge fast.
//   Shared so CPU and GPU seed identically. Defined in reference_cpu.cpp.
// ---------------------------------------------------------------------------
void linear_init(const SpectralModel& sm, double m_lo, double m_hi,
                 double& t1_init, double& t2_init);

// ---------------------------------------------------------------------------
// decompose_cpu: the trusted serial baseline. For each bin, seed with
// linear_init() then call decompose_bin() (the shared HD core). Fills t1[i],
// t2[i] and the per-bin iteration count. This is what the GPU result is verified
// against and the timing baseline that makes the speed-up legible. All output
// vectors are resized to sino.n.
// ---------------------------------------------------------------------------
void decompose_cpu(const DectSinogram& sino, const SpectralModel& sm,
                   std::vector<double>& t1, std::vector<double>& t2,
                   std::vector<int>& iters);
