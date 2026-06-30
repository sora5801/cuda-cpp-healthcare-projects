// ===========================================================================
// src/reference_cpu.h  --  Data model, pose grid, loader + CPU reference
// ---------------------------------------------------------------------------
// Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the DockingProblem container, the
//   pose enumeration, the text loader) and the CPU reference prototype live
//   here. kernels.cuh also includes this header to reuse the same types -- the
//   only CUDA-aware file (docking_core.h) is HD-guarded so it is safe in both.
//
// THE PROBLEM (one paragraph; full derivation in ../THEORY.md)
//   We are given a rigid PROTEIN and a rigid NUCLEIC-ACID fragment (the
//   "ligand"). We slide and rotate the ligand over a discrete 6-D grid of rigid
//   poses (3 translations x a set of orientations) and score each pose by how
//   well its atoms pack against the protein surface (favourable contacts +
//   matched electrostatics, minus steric clashes). The output is the BEST pose
//   and a ranked shortlist -- exactly the inner loop of rigid-body
//   protein-nucleic-acid docking (ZDOCK/PIPER-style), minus the FFT speed-up
//   and the all-atom force field. Every pose is INDEPENDENT, so the search is
//   embarrassingly parallel: one GPU thread per pose (PATTERNS.md sec 1,
//   "score one query vs N items").
//
// READ THIS AFTER: docking_core.h.   READ THIS BEFORE: reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "docking_core.h"   // Atom, ScoreParams, Rot3, score_pose (HD-safe)

// ---------------------------------------------------------------------------
// PoseGrid: the discrete translational search box, in fixed-point units.
//   The ligand's reference centre is swept over a regular 3-D lattice of
//   translations:  tx = tx0 + ix*step  for ix in [0, nx), and likewise y,z.
//   Orientations come from a separate list of cube-group rotations (below).
//   Total poses = nx * ny * nz * n_rot.
// ---------------------------------------------------------------------------
struct PoseGrid {
    int32_t tx0, ty0, tz0;   // translation origin (milli-Angstrom)
    int32_t step;            // lattice spacing (milli-Angstrom)
    int     nx, ny, nz;      // lattice counts per axis
};

// ---------------------------------------------------------------------------
// DockingProblem: everything one search needs.
//   protein : Np fixed atoms.
//   ligand  : Nl atoms in their reference frame (pose (R=identity, t=0)).
//   rots    : the orientation set (cube-group rotations; rots[0] is identity).
//   grid    : the translational lattice.
//   params  : the scoring thresholds/weights.
//   n_poses(): convenience = nx*ny*nz*rots.size().
// All vectors are host-side; kernels.cu uploads flat copies to the GPU.
// ---------------------------------------------------------------------------
struct DockingProblem {
    std::vector<Atom>   protein;
    std::vector<Atom>   ligand;
    std::vector<Rot3>   rots;
    PoseGrid            grid{};
    ScoreParams         params{};

    int Np() const { return (int)protein.size(); }
    int Nl() const { return (int)ligand.size(); }
    int n_rot() const { return (int)rots.size(); }
    long long n_poses() const {
        return (long long)grid.nx * grid.ny * grid.nz * (long long)rots.size();
    }
};

// ---------------------------------------------------------------------------
// decode_pose: map a flat pose index `p` in [0, n_poses) to its (rotation,
//   translation) -- the SAME decoding the CPU loop and the GPU thread use, so
//   pose p means the identical transform on both sides. Layout (fastest-varying
//   first): rotation, then ix, then iy, then iz.
//     r  = p % n_rot
//     ix = (p / n_rot) % nx
//     iy = (p / (n_rot*nx)) % ny
//     iz =  p / (n_rot*nx*ny)
//   Fills tx,ty,tz from the grid and returns the rotation index r.
//   HD so the GPU kernel can call the exact same function.
// ---------------------------------------------------------------------------
HD inline int decode_pose(long long p, const PoseGrid& g, int n_rot,
                          int32_t& tx, int32_t& ty, int32_t& tz) {
    const int       r  = (int)(p % n_rot);
    const long long q  = p / n_rot;
    const int       ix = (int)(q % g.nx);
    const int       iy = (int)((q / g.nx) % g.ny);
    const int       iz = (int)(q / ((long long)g.nx * g.ny));
    tx = g.tx0 + ix * g.step;
    ty = g.ty0 + iy * g.step;
    tz = g.tz0 + iz * g.step;
    return r;
}

// ---------------------------------------------------------------------------
// cube_rotations: build the 24 proper rotations of a cube (the orientation
//   set). Defined in reference_cpu.cpp; exposed here so main/kernels can ask
//   for the canonical list. Each matrix has det = +1 and entries in {-1,0,+1}.
// ---------------------------------------------------------------------------
std::vector<Rot3> cube_rotations();

// ---------------------------------------------------------------------------
// load_problem: parse the text format documented in data/README.md.
//   Throws std::runtime_error on a missing file or a malformed header.
//   The format is deliberately simple, integer, and human-readable so the
//   committed sample is auditable (CLAUDE.md sec 6 "no black boxes").
// ---------------------------------------------------------------------------
DockingProblem load_problem(const std::string& path);

// ---------------------------------------------------------------------------
// dock_cpu: the trusted reference. Score EVERY pose with score_pose() and
//   return the per-pose int64 scores in `scores` (indexed by flat pose id).
//   This is the obviously-correct baseline the GPU result is checked against
//   (exact integer equality) and the timing baseline that makes the GPU
//   speed-up legible. `scores` is resized to n_poses().
// ---------------------------------------------------------------------------
void dock_cpu(const DockingProblem& prob, std::vector<int64_t>& scores);
