// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for conformer generation
// ---------------------------------------------------------------------------
// Project 1.14 : Conformer Ensemble Generation
//
// WHY A PURE-C++ HEADER (no CUDA here)
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared host-side declarations live here. The actual
//   per-conformer PHYSICS is in conformer.h (the __host__ __device__ core that
//   BOTH this reference and the GPU kernel call), so nothing is duplicated.
//
// WHAT THE REFERENCE PROVIDES
//   1. enumerate_energies_cpu() : the trusted serial energy of every conformer,
//      the baseline the GPU energies are verified against.
//   2. rmsd_cluster()           : greedy RMSD pruning of the ensemble into a set
//      of distinct representative conformers -- the "breadth-first conformer
//      pruning (RMSD clustering)" algorithm from the catalog. This runs on the
//      CPU on purpose: it is an inherently sequential greedy scan (each decision
//      depends on the ones before it), so it is the wrong shape for the GPU --
//      the GPU's win is the embarrassingly-parallel ENERGY step.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.  Physics lives in conformer.h.
// ===========================================================================
#pragma once

#include <array>
#include <vector>

#include "conformer.h"   // N_CONFORMER, Vec3, conformer_energy, coord_rmsd, ...

// enumerate_energies_cpu: fill energy[c] with conformer c's potential energy
// (kcal/mol) for c = 0 .. N_CONFORMER-1, computed serially on the CPU by calling
// the SAME conformer_energy() the GPU kernel uses. This is the verification
// baseline (and the timing baseline that makes the GPU speed-up legible).
//   energy : resized to N_CONFORMER and filled (output parameter).
void enumerate_energies_cpu(std::vector<double>& energy);

// rmsd_cluster: prune the full ensemble to a non-redundant set of conformers.
//   Algorithm (greedy "leader" clustering, the standard conformer-dedup recipe):
//     1. sort all conformers by ascending energy (lowest first);
//     2. walk that sorted list; ACCEPT a conformer as a new cluster
//        representative only if its coordinate RMSD to EVERY already-accepted
//        representative exceeds rmsd_threshold; otherwise discard it as a
//        near-duplicate of a lower-energy shape.
//   The result is the list of representative conformer indices, in ascending
//   energy order -- the "conformer ensemble" a docking run would actually use.
//
//   energy         : per-conformer energies (from enumerate_energies_cpu).
//   rmsd_threshold : Angstrom; two conformers closer than this are "the same".
//   returns        : representative conformer indices (low energy -> high).
std::vector<long> rmsd_cluster(const std::vector<double>& energy,
                               double rmsd_threshold);
