// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for covalent docking
// ---------------------------------------------------------------------------
// Project 1.28 : Covalent Docking
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax. So the shared DATA MODEL (how a docking problem is loaded from
//   a file) and the CPU reference prototype live here. kernels.cuh also includes
//   this header so the GPU side reuses the SAME DockProblem type and loader --
//   nothing CUDA-specific leaks across the boundary. The per-element PHYSICS is
//   one level deeper, in docking.h (the __host__ __device__ shared core).
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   We search a grid of ligand torsion angles for the lowest-energy covalent
//   pose. Each conformation is scored by score_conformation() (docking.h). The
//   CPU reference loops over all conformations serially; the GPU kernel gives
//   each conformation its own thread. Both then argmin to find the docked pose.
//
// READ THIS AFTER: docking.h. READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "docking.h"   // DockProblem, score_conformation, n_conformations (pure C++)

// ---------------------------------------------------------------------------
// DockResult: the outcome of a full search.
//   best_id     : flat conformation index of the lowest-energy pose found.
//   best_energy : that pose's total energy (kcal/mol). Lower is better.
// We return the id (not the coordinates) so the result is a small, exactly-
// comparable scalar pair -- ideal for CPU-vs-GPU verification. The caller can
// rebuild the coordinates from the id via build_conformation() if needed.
// ---------------------------------------------------------------------------
struct DockResult {
    long long best_id;       // argmin conformation index
    double    best_energy;   // its energy (kcal/mol)
};

// ---------------------------------------------------------------------------
// load_problem: read a DockProblem from the text format documented in
// data/README.md. The format is line-oriented and human-readable so the sample
// is easy to inspect and tweak. Throws std::runtime_error on a missing/short
// file so demos fail loudly instead of silently scoring garbage.
// ---------------------------------------------------------------------------
DockProblem load_problem(const std::string& path);

// ---------------------------------------------------------------------------
// score_all_cpu: fill `energies` with the energy of EVERY conformation on the
// grid (energies[id] = score_conformation(p, id)). This is the trusted, obvious
// baseline the GPU result is checked against AND the timing baseline that makes
// the speed-up legible. `energies` is resized to n_conformations().
// We materialize the whole energy array (rather than just the min) so main.cu
// can verify the GPU array element-by-element, not just the final argmin.
// ---------------------------------------------------------------------------
void score_all_cpu(const DockProblem& p, std::vector<double>& energies);

// ---------------------------------------------------------------------------
// argmin_energy: reduce an energy array to the best (lowest-energy) pose.
//   Ties are broken by the LOWER id so the winner is deterministic regardless of
//   iteration direction. Shared by the CPU and GPU paths (both copy energies to
//   the host and call this) so the reported pose is identical.
// ---------------------------------------------------------------------------
DockResult argmin_energy(const std::vector<double>& energies);
