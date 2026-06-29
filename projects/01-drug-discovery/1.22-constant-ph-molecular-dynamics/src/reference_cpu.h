// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU titration reference
// ---------------------------------------------------------------------------
// Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
//
// The CPU reference runs the SAME Monte Carlo chains as the GPU (same RNG, same
// energy from cph_core.h), so the two protonation tallies must be IDENTICAL.
// This header is pure C++ (no CUDA constructs), so kernels.cu can also include
// it to reuse the CphProblem struct and the result layout.
//
// READ THIS AFTER: cph_core.h (the shared physics). READ BEFORE: reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cph_core.h"   // CphSystem, Residue, Rng, run_chain (host+device safe)

// ---------------------------------------------------------------------------
// CphProblem: a complete, loadable titration experiment.
//   * sys      : the residues + MC controls (shared with the GPU, cph_core.h).
//   * pH_min / pH_max / n_pH : the pH grid the titration curve is sampled on.
//                The k-th pH is  pH_min + k*(pH_max-pH_min)/(n_pH-1).
//   * replicas : independent MC chains per pH (different RNG seeds). More
//                replicas = lower Monte Carlo noise on the fraction-protonated;
//                we average over them. They are extra independent work, so they
//                widen the ensemble the GPU parallelises over.
//   * seed     : the base RNG seed; chain (pH index k, replica r) is seeded from
//                (seed, encode(k,r)) so every chain is reproducible.
// The TOTAL number of independent chains is  n_pH * replicas  -- that is the
// ensemble size the GPU launches one thread per.
// ---------------------------------------------------------------------------
struct CphProblem {
    CphSystem sys{};
    double pH_min = 0.0;
    double pH_max = 0.0;
    int    n_pH   = 0;       // number of pH grid points (titration-curve x-axis)
    int    replicas = 1;     // independent chains averaged per pH
    uint64_t seed = 0;       // base RNG seed
};

// ---------------------------------------------------------------------------
// CphResult: the tallied output, laid out so CPU and GPU fill the SAME array and
// a byte-compare verifies them.
//   * prot_count : a flat [n_pH * n_res] array of unsigned 64-bit COUNTS.
//                  prot_count[k*n_res + i] = total protonated-sweep tallies for
//                  residue i summed over all replicas at pH index k. Integer, so
//                  the GPU's parallel accumulation is exact and order-independent.
//   * tallied_per_pH : sweeps tallied per pH = replicas * (sweeps - burn_in).
//                  Divide prot_count by this to get the fraction protonated.
// We keep the raw integer counts (not pre-divided floats) precisely so the
// verification is an EXACT == check, not a tolerance check (PATTERNS.md §4).
// ---------------------------------------------------------------------------
struct CphResult {
    std::vector<uint64_t> prot_count;   // [n_pH * n_res] protonated-sweep tallies
    uint64_t tallied_per_pH = 0;        // denominator to form the fraction
};

// Load a CphProblem from the simple text format documented in data/README.md.
// Throws std::runtime_error with a clear message on any malformed input so the
// demo fails loudly instead of silently running on garbage.
CphProblem load_cph_problem(const std::string& path);

// CPU reference: run every (pH, replica) chain serially, tallying integer
// protonation counts into `out`. Because the counts are integers, this result
// must equal the GPU's exactly. `out.prot_count` is sized to n_pH*n_res here.
void titrate_cpu(const CphProblem& prob, CphResult& out);

// Helper (host-only) shared by main.cu's reporting: estimate the pKa of residue
// i from its titration curve by locating where the fraction-protonated crosses
// 0.5, using linear interpolation between the two bracketing pH points. Returns
// NaN if the curve never crosses 0.5 on the sampled grid. Defined in
// reference_cpu.cpp so both the CPU and GPU report paths can call it.
double estimate_pKa(const CphProblem& prob, const CphResult& res, int residue);
