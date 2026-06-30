// ===========================================================================
// src/reference_cpu.h  --  Problem loader + CPU Monte Carlo reference (interface)
// ---------------------------------------------------------------------------
// Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
//
// The CPU reference runs the SAME walks as the GPU (same RNG, same moves, same
// Boltzmann tables from mc_moves.h), so the two per-replica energies must be
// identical. This header is pure C++ (no CUDA constructs); kernels.cu reuses
// McProblem / McResult from mc_moves.h, included below.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "mc_moves.h"   // McProblem, McResult, run_replica, build_boltzmann_table

// Load an McProblem from the tiny text format documented in data/README.md:
//   line 1: n sweeps n_replicas t_min t_max seed
//   line 2: the HP sequence as n characters from {H,P} (e.g. HPHPPHHPHH)
// Throws std::runtime_error on a missing file or malformed contents so the demo
// fails loudly rather than silently running on garbage.
McProblem load_mc_problem(const std::string& path);

// CPU reference: run all n_replicas walks SERIALLY and fill `out` (sized to
// n_replicas) with each replica's {best_energy, final_energy}. Because the HP
// energy is an integer and the Boltzmann tables are shared, this result must
// equal the GPU's element-for-element -- that exact match is the verification.
//   tables : flat array of n_replicas * (2*MC_DE_RANGE+1) doubles, prebuilt by
//            the caller (main.cu); replica r's table starts at r*table_stride.
void sample_cpu(const McProblem& prob, const std::vector<double>& tables,
                std::vector<McResult>& out);

// Number of doubles in one replica's Boltzmann table. Exposed so main.cu can
// size the flat `tables` buffer consistently with mc_moves.h (MC_DE_RANGE).
inline int boltzmann_table_size() { return 2 * MC_DE_RANGE + 1; }
