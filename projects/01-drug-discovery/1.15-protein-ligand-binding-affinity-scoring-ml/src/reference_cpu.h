// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for 3D-CNN affinity scoring
// ---------------------------------------------------------------------------
// Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must never see
//   CUDA syntax, so the shared DATA MODEL (the parsed dataset, the file loader)
//   and the CPU reference prototypes live here. kernels.cuh also includes this
//   header to reuse the ComplexSet type -- nothing CUDA-specific leaks either way.
//   The per-element math (voxelization, conv, weights) is in scoring_core.h, which
//   BOTH this CPU file and the GPU kernels include (the HD-macro idiom).
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   Given N docked protein-ligand poses, predict each one's binding affinity pKd
//   with a small 3D convolutional neural network:
//       voxelize atoms -> conv3d + ReLU -> global average pool -> dense -> pKd.
//   Every pose is independent => embarrassingly parallel batch inference, the
//   real-world "rescore millions of docking poses" workload.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.  READ scoring_core.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "scoring_core.h"   // Atom, GRID, CIN, COUT, geometry + per-element math

// ---------------------------------------------------------------------------
// A batch of complexes to score. The atoms of all N complexes are stored in ONE
// flat vector `atoms` (cache-friendly and trivial to upload to the GPU in a
// single cudaMemcpy); `offset[i]..offset[i+1]` delimits complex i's atoms (the
// classic CSR-style "ragged array" layout, since complexes have different atom
// counts). `label[i]` is the synthetic ground-truth pKd we generated the pose to
// have, used only as an interpretability check in the report (NOT in scoring).
// ---------------------------------------------------------------------------
struct ComplexSet {
    int n = 0;                        // number of complexes (poses) in the batch
    std::vector<Atom>   atoms;        // all atoms of all complexes, concatenated
    std::vector<int>    offset;       // size n+1; complex i = atoms[offset[i]..offset[i+1])
    std::vector<double> label;        // size n; synthetic "true" pKd per complex

    int atom_count(int i) const { return offset[i + 1] - offset[i]; }
};

// Load a batch from the text format documented in data/README.md. Throws
// std::runtime_error on a missing file or malformed record.
ComplexSet load_complexes(const std::string& path);

// ---------------------------------------------------------------------------
// score_cpu: the trusted serial forward pass. For each complex it voxelizes the
//   atoms, runs the conv+ReLU+pool+dense network from scoring_core.h, and writes
//   the predicted pKd into out[i]. This is the obviously-correct baseline the GPU
//   kernel is verified against, and the timing baseline that makes the speed-up
//   legible. out is resized to cs.n.
//
//   We compute everything in DOUBLE precision with a fixed summation order so the
//   result is reproducible and matches the GPU term-for-term (THEORY "verify").
// ---------------------------------------------------------------------------
void score_cpu(const ComplexSet& cs, std::vector<double>& out);
