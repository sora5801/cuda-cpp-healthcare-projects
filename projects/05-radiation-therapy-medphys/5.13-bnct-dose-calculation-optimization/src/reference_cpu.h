// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU Monte Carlo reference
// ---------------------------------------------------------------------------
// Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
//
// The CPU reference runs the SAME neutron histories as the GPU (same RNG, same
// transport in bnct_physics.h), so the two per-component dose tallies must be
// identical. This header is pure C++ (no CUDA); kernels.cu reuses BnctProblem
// so the GPU and CPU share one problem definition.
//
// READ THIS AFTER: bnct_physics.h. READ BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bnct_physics.h"   // SimParams, Rng, simulate_neutron, DoseComponent

// ---------------------------------------------------------------------------
// BnctProblem: a complete BNCT dose job -- the slab/cross-section physics plus
// how many neutron histories to run and the RNG seed. history i uses the
// independent stream (seed, i) so the CPU and GPU cover the identical set of
// histories regardless of thread order.
// ---------------------------------------------------------------------------
struct BnctProblem {
    SimParams sp{};                     // slab + cross-section parameters
    unsigned long long n_histories = 0; // number of neutron histories to simulate
    uint64_t seed = 0;                  // base RNG seed
    double gray_per_keV = 0.0;          // scale: keV-quanta tally -> physical Gy
                                        //   (a single documented constant so the
                                        //    Gy report is reproducible; see below)
};

// ---------------------------------------------------------------------------
// DoseTally: the raw result of a run. dose[c] is a length-n_bins vector of the
// INTEGER keV quanta deposited by component c in each depth bin. Integer so the
// GPU's atomicAdd order does not change the result -> exact CPU==GPU match.
//   dose[DC_BORON][b], dose[DC_NITROGEN][b], dose[DC_GAMMA][b], dose[DC_FAST][b]
// ---------------------------------------------------------------------------
struct DoseTally {
    std::vector<std::vector<unsigned long long>> dose;  // [DC_COUNT][n_bins]
    void reset(int n_bins) {
        dose.assign(DC_COUNT, std::vector<unsigned long long>(n_bins, 0ULL));
    }
};

// Load a BnctProblem from the one-line text format (see data/README.md):
//   "L n_bins Sig_s_fast p_thermalize Sig_a_B Sig_a_N Sig_a_H Sig_s_th
//    Q_boron_keV Q_nitro_keV Q_gamma_keV Q_fast_keV n_histories seed gray_per_keV"
BnctProblem load_bnct_problem(const std::string& path);

// CPU reference: simulate all n_histories neutron histories serially and tally
// integer per-component dose per depth bin. `t` is sized to n_bins. Because
// energy is integer keV quanta, the tally is exact and order-independent -- it
// must equal the GPU's bin-for-bin.
void dose_cpu(const BnctProblem& prob, DoseTally& t);
