// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for rigid-body docking
// ---------------------------------------------------------------------------
// Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see CUDA
//   syntax, so the shared DATA MODEL (the receptor energy grid, the ligand, the
//   pose search space, the file loader) and the CPU reference prototypes live
//   here. kernels.cuh also includes this header to reuse the same types, and the
//   per-pose PHYSICS lives in docking_core.h (shared __host__ __device__). Thus
//   nothing CUDA-specific leaks into the host build, and CPU and GPU score poses
//   with identical math.
//
// THE PROBLEM IN ONE PARAGRAPH (full derivation in ../THEORY.md)
//   Molecular docking predicts how a small-molecule ligand binds in a protein
//   pocket. We precompute the pocket as a 3D ENERGY GRID (energy a probe atom
//   feels at each point), then SAMPLE many rigid poses of the ligand (a grid of
//   translations x rotations) and SCORE each by summing grid energy over the
//   ligand's atoms (docking_core.h::score_pose). The pose with the lowest energy
//   is the predicted binding mode. Every pose is scored INDEPENDENTLY -> perfect
//   data parallelism: one GPU thread per pose (kernels.cu).
//
//   This is a deliberately REDUCED-SCOPE teaching version: rigid ligand, single
//   generic energy grid, exhaustive pose grid (no genetic algorithm / BFGS).
//   ../THEORY.md S"Where this sits in the real world" explains what AutoDock-GPU
//   / Vina-GPU add (per-atom-type grids, LGA search, torsional flexibility).
//
// READ THIS BEFORE: docking_core.h (the physics), reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "docking_core.h"   // GridDims, Pose, score_pose (pure-C++-safe header)

// ---------------------------------------------------------------------------
// Ligand: a rigid set of atoms in LIGAND-LOCAL coordinates (offsets from the
//   ligand centroid, Angstrom). Stored as Structure-of-Arrays (separate x/y/z
//   vectors) rather than Array-of-Structs because the GPU reads all atoms' x,
//   then all y, etc. -- SoA gives coalesced, cache-friendly device access
//   (THEORY S"GPU mapping").
//   weight[k] is a per-atom scale (think |partial charge| or atom-type factor);
//   all-ones is a valid generic probe.
// ---------------------------------------------------------------------------
struct Ligand {
    std::vector<double> x, y, z;     // [n_atoms] local offsets from centroid (A)
    std::vector<double> weight;      // [n_atoms] per-atom probe weight
    int n_atoms = 0;
};

// ---------------------------------------------------------------------------
// SearchSpace: the discrete grid of poses to evaluate (the "conformational
//   sampling" of the catalog). We sweep:
//     translation : n_trans points per axis, evenly spanning [-trans_range,
//                   +trans_range] Angstrom around the pocket centre (tcx,tcy,tcz);
//     rotation    : n_rot points per axis, evenly spanning [0, 2pi) radians.
//   Total poses = (n_trans^3) * (n_rot^3). A single linear pose index p is
//   decoded into these six sub-indices by unrank_pose() so the CPU loop and the
//   GPU's flat thread index enumerate EXACTLY the same poses in the same order.
// ---------------------------------------------------------------------------
struct SearchSpace {
    int    n_trans = 0;              // samples per translation axis
    int    n_rot   = 0;              // samples per rotation axis
    double trans_range = 0.0;        // +/- Angstrom around pocket centre
    double tcx = 0.0, tcy = 0.0, tcz = 0.0;   // pocket centre (world, A)

    // Total number of poses = (n_trans^3) * (n_rot^3).
    long long n_poses() const {
        const long long t = static_cast<long long>(n_trans) * n_trans * n_trans;
        const long long r = static_cast<long long>(n_rot)   * n_rot   * n_rot;
        return t * r;
    }
};

// ---------------------------------------------------------------------------
// DockingProblem: everything loaded from the sample file -- the receptor energy
//   grid, the ligand, and the pose search space. Passed whole to CPU and GPU.
// ---------------------------------------------------------------------------
struct DockingProblem {
    GridDims            dims;        // receptor grid geometry
    std::vector<double> grid;        // [dims.count()] energies (kcal/mol)
    Ligand              ligand;      // rigid ligand (local coords)
    SearchSpace         space;       // pose grid to evaluate
};

// ---------------------------------------------------------------------------
// unrank_pose: map a flat pose index p in [0, n_poses) to the concrete Pose.
//   This is the linchpin of CPU/GPU parity: the GPU thread with global index p
//   and the CPU loop iteration p must produce the IDENTICAL Pose. We decode p in
//   mixed-radix order (translation x,y,z then rotation x,y,z), each axis a
//   uniform sample of its range. Declared here, defined in reference_cpu.cpp,
//   and ALSO usable on the device because it only does integer/double math --
//   so kernels.cu provides its own __device__ copy with the same body (kept in
//   sync deliberately; see the note there).
//
//   p layout (fastest-varying first):
//     it_x, it_y, it_z   each in [0, n_trans)
//     ir_x, ir_y, ir_z   each in [0, n_rot)
// ---------------------------------------------------------------------------
Pose unrank_pose(const SearchSpace& s, long long p);

// ---------------------------------------------------------------------------
// load_problem: parse the tiny text dataset (format in data/README.md) into a
//   DockingProblem. Throws std::runtime_error on a missing file / malformed input
//   so demos fail loudly rather than scoring garbage.
// ---------------------------------------------------------------------------
DockingProblem load_problem(const std::string& path);

// ---------------------------------------------------------------------------
// dock_cpu: the trusted serial baseline. Scores EVERY pose with score_pose()
//   and returns the best (lowest-energy) pose's flat index and its energy.
//   out_energy / out_index are written with the global minimum. Deterministic:
//   on an exact tie it keeps the LOWER pose index (matches the GPU's tie rule).
//   This is both the correctness oracle and the timing baseline that makes the
//   GPU speed-up legible.
// ---------------------------------------------------------------------------
void dock_cpu(const DockingProblem& prob, double* out_energy, long long* out_index);
