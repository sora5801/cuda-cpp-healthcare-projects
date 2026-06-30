// ===========================================================================
// src/reference_cpu.h  --  Problem loader + serial MDFF reference (no CUDA)
// ---------------------------------------------------------------------------
// Project 2.12 : Flexible Fitting / MDFF
//
// ROLE IN THE PROJECT
//   Pure C++ (compiled by cl.exe/g++, never nvcc). It declares:
//     * the in-memory PROBLEM (the density map + the atom set + parameters),
//     * how to LOAD that problem from data/sample (or build it synthetically),
//     * the CPU reference fitting loop (the trusted baseline), and
//     * the correctness METRICS (RMSD-to-target and density cross-correlation).
//
//   The per-atom physics and trilinear sampling are NOT here -- they live in
//   mdff.h as __host__ __device__ functions so the GPU kernel reuses the exact
//   same math. This header only owns host-side orchestration and I/O.
//
// READ THIS AFTER: mdff.h. Then reference_cpu.cpp, then main.cu / kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "mdff.h"   // Vec3, MdffParams, sample_density, mdff_step_atom

// ---------------------------------------------------------------------------
// MdffProblem : a self-contained fitting problem instance.
//   * params  : grid size, weights, step, iteration count (see mdff.h).
//   * rho      : the density map, length nx*ny*nz, x-fastest (grid_index order).
//   * x0       : the STARTING atom positions (the misfitted model).
//   * x_ref    : the restraint anchors (where the harmonic restraint pulls back
//                to). Here equal to x0: we anchor the restraint at the start so
//                the density force does the deforming and the restraint only
//                damps runaway. (Production MDFF restrains to bonded geometry.)
//   * x_target : the GROUND-TRUTH positions used ONLY to score the fit (the atoms
//                that generated the synthetic density). Never read by the solver
//                -- it is the answer key, so reporting RMSD-to-target is honest.
// ---------------------------------------------------------------------------
struct MdffProblem {
    MdffParams params;
    std::vector<double> rho;       // density map  [nx*ny*nz]
    std::vector<Vec3>   x0;        // starting (misfitted) atom positions
    std::vector<Vec3>   x_ref;     // harmonic-restraint anchors
    std::vector<Vec3>   x_target;  // ground-truth positions (answer key, scoring only)
};

// Load a problem from a whitespace-separated text file (format in data/README.md).
// Throws std::runtime_error if the file is missing or malformed so demos fail
// loudly rather than silently fitting garbage.
MdffProblem load_problem(const std::string& path);

// Build the built-in synthetic problem (used when no data file is supplied).
// Deterministic: a small lattice of atoms, each displaced from a target, with a
// density map made of Gaussian blobs centred on the targets. The EXACT values
// this produces are what demo/expected_output.txt encodes.
MdffProblem make_synthetic();

// CPU reference solver: run `params.iters` steepest-descent fitting iterations,
// advancing every atom each iteration via mdff_step_atom (from mdff.h). Returns
// the final fitted positions. This is the trusted baseline the GPU must match.
std::vector<Vec3> fit_cpu(const MdffProblem& prob);

// ---- Correctness / quality metrics (host-only, used by main.cu) -----------

// rmsd : root-mean-square distance between two equal-length atom sets. The
//   headline "how well did we fit" number; we report it for the start and the
//   end (against x_target) so the learner sees the model snap onto the density.
double rmsd(const std::vector<Vec3>& a, const std::vector<Vec3>& b);

// cross_correlation : the density cross-correlation score CC that MDFF actually
//   optimises -- here computed as the mean interpolated density at the atom
//   positions (higher = atoms sit on denser regions = better fit). It is the
//   quantity whose gradient is the fitting force, so it must rise during fitting.
double cross_correlation(const std::vector<Vec3>& x, const MdffProblem& prob);
