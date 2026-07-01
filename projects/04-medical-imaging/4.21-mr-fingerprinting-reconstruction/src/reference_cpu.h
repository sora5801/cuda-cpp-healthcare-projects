// ===========================================================================
// src/reference_cpu.h  --  Data model + the CPU reference MRF reconstruction
// ---------------------------------------------------------------------------
// Project 4.21 : MR Fingerprinting Reconstruction
//
// WHY A SEPARATE, PURE-C++ HEADER
//   The CPU reference (reference_cpu.cpp) is compiled by the plain host C++
//   compiler and must NOT see any CUDA/__global__ syntax, so its prototypes
//   cannot live in kernels.cuh. Both main.cu and reference_cpu.cpp include THIS
//   header so they agree on the data structures and the reference API. The
//   actual per-element math is shared with the GPU through mrf_core.h.
//
// WHAT LIVES HERE
//   * MrfProblem     -- everything loaded from data/sample: the acquisition
//                       schedule, the (T1,T2) dictionary grid, the measured
//                       voxel signals, and the synthetic ground truth.
//   * build_dictionary_cpu / normalize_* -- the reference dictionary build.
//   * match_cpu      -- the reference matcher (argmax cosine per voxel).
//   * load_problem   -- parse the whitespace sample file into an MrfProblem.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu. See mrf_core.h for the shared
// scalar formulas and kernels.cuh for the GPU twin of the reconstruction.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// MrfProblem: the fully-loaded reconstruction problem (all host memory).
//
//   Sizes:
//     T  = number of time frames (fingerprint length)
//     D  = number of dictionary atoms (T1xT2 grid points)
//     V  = number of voxels in the (synthetic) image
//
//   Layout choices (documented because they drive the GPU code):
//     * dict[d*T + t]   -- row-major DxT: atom d's signal at frame t. Storing
//                          atoms as rows keeps each fingerprint contiguous, so
//                          simulating/normalizing one atom touches a contiguous
//                          run of memory (coalesced on the GPU).
//     * signal[v*T + t] -- row-major VxT: voxel v's measured signal at frame t.
//   These two row-major matrices are EXACTLY the operands the cuBLAS SGEMM in
//   kernels.cu multiplies to get the VxD cosine-score matrix.
// ---------------------------------------------------------------------------
struct MrfProblem {
    int T = 0;                        // time-course length (frames)
    int D = 0;                        // dictionary size (atoms)
    int V = 0;                        // number of voxels

    // Acquisition schedule (length T), shared by every atom and voxel.
    std::vector<double> alpha;        // [T] flip angles, radians
    std::vector<double> tr;           // [T] repetition times, ms
    std::vector<double> te;           // [T] echo times, ms

    // Dictionary parameter grid (length D), the tissue candidates.
    std::vector<double> dict_t1;      // [D] T1 of atom d, ms
    std::vector<double> dict_t2;      // [D] T2 of atom d, ms

    // Measured voxel signals (row-major VxT). In this teaching demo they are
    // synthesized from known ground-truth atoms plus noise (see data/README.md).
    std::vector<float> signal;        // [V*T]

    // Synthetic ground truth (present only because the data is synthetic; a real
    // scan has none). truth_atom[v] is the dictionary index the voxel was drawn
    // from, so we can score reconstruction accuracy honestly.
    std::vector<int> truth_atom;      // [V] index in [0, D) or -1 if unknown
};

// ---------------------------------------------------------------------------
// MatchResult: the reconstruction output, one entry per voxel.
//   best_atom : argmax-cosine dictionary index (the matched tissue)
//   score     : the winning cosine in [-1, 1] (1 = perfect shape match)
//   t1_ms/t2_ms : the matched atom's relaxation times (the parameter maps)
//   pd        : proton-density estimate = ||signal|| * score (the scale we
//               removed by normalizing; recovered here as a bonus map)
// ---------------------------------------------------------------------------
struct MatchResult {
    int    best_atom;
    float  score;
    double t1_ms;
    double t2_ms;
    float  pd;
};

// load_problem: parse the committed sample file into an MrfProblem.
//   The file format is documented in data/README.md and scripts/make_synthetic.py.
//   Throws std::runtime_error on a malformed/absent file so demos fail loudly.
MrfProblem load_problem(const std::string& path);

// build_dictionary_cpu: simulate every atom's fingerprint and L2-normalize it.
//   Fills dict_norm (row-major D*T) using mrf::simulate_atom + mrf::normalize.
//   This is the serial baseline for the GPU build_dictionary kernel.
void build_dictionary_cpu(const MrfProblem& p, std::vector<float>& dict_norm);

// normalize_signals_cpu: L2-normalize each voxel signal into sig_norm (V*T).
//   Also returns each voxel's original norm in sig_scale[V] (for the PD map).
void normalize_signals_cpu(const MrfProblem& p, std::vector<float>& sig_norm,
                           std::vector<float>& sig_scale);

// match_cpu: for each voxel, find the dictionary atom of maximum cosine.
//   Reference implementation of the whole matcher: it recomputes each cosine as
//   a direct inner product (mrf::dot) -- the same scalar the GPU's SGEMM forms.
//   Ties are broken by SMALLEST atom index, deterministically, so the reference
//   answer is unique and reproducible.
void match_cpu(const MrfProblem& p, const std::vector<float>& dict_norm,
               const std::vector<float>& sig_norm,
               const std::vector<float>& sig_scale,
               std::vector<MatchResult>& out);
