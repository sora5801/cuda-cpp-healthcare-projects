// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for trajectory analysis
// ---------------------------------------------------------------------------
// Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the Trajectory container, the file
//   loader) and the CPU reference prototypes live here. kernels.cuh also
//   includes this header to reuse the Trajectory type and N_ATOMS -- nothing
//   CUDA-specific leaks across the boundary. The per-frame *math* lives in
//   rmsd_core.h (the shared __host__ __device__ core), which both sides call.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   A molecular-dynamics TRAJECTORY is a movie of a molecule: F "frames", each a
//   snapshot of N atoms in 3D. Post-simulation we want, per frame:
//     * RMSD to a reference frame after OPTIMAL SUPERPOSITION (Kabsch/QCP) --
//       "how far has the structure moved, ignoring rigid translation/rotation?"
//     * the FRACTION OF NATIVE CONTACTS Q -- "how many of the reference's close
//       atom pairs are still close?" -- a standard conformational coordinate.
//   Every frame is INDEPENDENT of every other, so the natural GPU mapping is one
//   thread per frame (kernels.cu) -- the same "independent jobs" pattern as the
//   1.12 fingerprint search, but the per-job work is a small linear-algebra
//   computation instead of a popcount.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Math is in rmsd_core.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "rmsd_core.h"   // N_ATOMS, the __host__ __device__ per-frame math

// A loaded trajectory: n_frames snapshots, each N_ATOMS atoms x 3 coordinates.
//   coords : [n_frames * N_ATOMS * 3] doubles, frame-major then atom-major
//            (frame f, atom a, axis k) lives at coords[(f*N_ATOMS + a)*3 + k].
//   ref    : which frame index is the reference for RMSD / native contacts.
//            We use frame 0 by convention (the "starting structure").
// We store coordinates as DOUBLE so the QCP eigenvalue and the contact distances
// are computed in double precision on both CPU and GPU (FP64), which is what
// makes the two agree to ~machine epsilon (see ../THEORY.md "Numerical").
struct Trajectory {
    int n_frames = 0;                 // number of frames F
    int ref = 0;                      // reference frame index (0-based)
    std::vector<double> coords;       // [n_frames * N_ATOMS * 3], frame-major
};

// Per-frame results we compute and verify. Two parallel arrays of length
// n_frames so the GPU and CPU can be compared element-by-element.
//   rmsd[f] : optimal-superposition RMSD of frame f vs. the reference frame.
//   qnc[f]  : fraction of native contacts of frame f (in [0,1]).
struct FrameMetrics {
    std::vector<double> rmsd;   // [n_frames]
    std::vector<double> qnc;    // [n_frames]  (Q, native-contact fraction)
};

// Load a trajectory from the text format documented in data/README.md:
//   line 1:  "<n_frames> <N_ATOMS> <ref_index>"
//   then n_frames blocks, each of N_ATOMS lines "x y z" (whitespace-separated).
// Throws std::runtime_error on a missing file or an atom-count mismatch (the
// file's N_ATOMS must equal the compile-time N_ATOMS so the layout is fixed).
Trajectory load_trajectory(const std::string& path);

// CPU reference: fill out.rmsd[f] and out.qnc[f] for every frame f by calling
// the SAME rmsd_core.h functions the GPU kernel calls -- so when the two agree
// we have a genuine correctness proof, not a coincidence. out is sized to
// traj.n_frames. This is the trusted baseline and the timing baseline.
void analyze_trajectory_cpu(const Trajectory& traj, FrameMetrics& out);

// Tiny clustering step (the "clustering" in the project title), kept on the CPU
// because it is a reduction over the already-computed per-frame RMSDs, not a
// per-frame parallel job. It is the GROMOS-style fixed-radius idea in its
// simplest, deterministic form: bin each frame into an RMSD shell
// b = floor(rmsd / width), i.e. group conformations by how far they have drifted
// from the reference. Returns counts[b] = number of frames in shell b. See
// ../THEORY.md "The algorithm" for how this relates to true GROMOS clustering.
//   width  : RMSD bin width (same length unit as coordinates); must be > 0
//   counts : resized to (max_bin+1); counts[b] = #frames with bin == b
void cluster_by_rmsd(const FrameMetrics& m, double width, std::vector<int>& counts);
