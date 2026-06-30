// ===========================================================================
// src/reference_cpu.h  --  RefineConfig loader + CPU reference annealer
// ---------------------------------------------------------------------------
// Project 2.18 : NMR Structure Refinement
//
// This header declares the host-only pieces: how we LOAD a refinement job from
// the text sample, and the CPU REFERENCE that anneals every replica serially.
// The actual per-replica physics (RNG, energy, the SA loop) lives in nmr_refine.h
// and is shared with the GPU so the two agree to round-off. kernels.cu reuses
// RefineConfig and ReplicaResult straight from nmr_refine.h.
//
// WHY A SEPARATE HOST HEADER
//   reference_cpu.cpp is compiled by the plain C++ compiler and must NOT see any
//   CUDA/__global__ syntax, so its prototypes cannot live in kernels.cuh. Both
//   main.cu and reference_cpu.cpp include THIS pure-C++ header so they agree on
//   the loader and reference signatures.
//
// READ THIS AFTER: nmr_refine.h.  READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "nmr_refine.h"   // RefineConfig, ReplicaResult, anneal_one, NMR_HD

// Load a RefineConfig from the text format documented in data/README.md:
//   line 1:  n_beads n_restraints bond_len k_bond k_noe
//   line 2:  n_replicas n_steps T_hot T_cold step_sigma base_seed
//   next n_restraints lines:  i j upper
// Throws std::runtime_error on a missing file or malformed/oversized input so the
// demo fails loudly rather than silently annealing garbage.
RefineConfig load_config(const std::string& path);

// CPU reference: run the full SA ensemble SERIALLY (one replica after another),
// filling results[r] for every replica. This is the trusted baseline the GPU
// ensemble is checked against -- it calls the same anneal_one() the kernel does,
// so a given replica yields identical numbers on both. Scratch buffers are
// allocated once here and reused across replicas.
void anneal_ensemble_cpu(const RefineConfig& c, std::vector<ReplicaResult>& results);
