// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for the NNP
// ---------------------------------------------------------------------------
// Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see CUDA
//   syntax, so the shared DATA MODEL (the loaded structure, the model builder,
//   the file loader) and the CPU reference prototypes live here. kernels.cuh
//   also includes this header to reuse the same types -- nothing CUDA-specific
//   leaks across the boundary. The per-atom PHYSICS lives one level deeper, in
//   nnp.h (the __host__ __device__ core shared by CPU and GPU).
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   Given the 3-D coordinates of n atoms, compute:
//     * a per-atom energy E_i (descriptor -> small MLP), and
//     * the total potential energy E = sum_i E_i.
//   Every E_i depends only on neighbors within a cutoff, so the n atoms are
//   independent jobs -> one GPU thread per atom (kernels.cu).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.  READ nnp.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "nnp.h"   // AtomicNet, AcsfParams, N_DESC, N_HID, atomic_energy(...)

// ---------------------------------------------------------------------------
// Structure: one loaded molecular system.
//   pos is FLAT and row-major: atom k is at (pos[3k], pos[3k+1], pos[3k+2]),
//   in Angstrom. We store coordinates flat (not as a vector<vec3>) because that
//   is exactly the layout we copy to the GPU -- contiguous doubles that coalesce
//   nicely and need no per-atom struct on the device.
// ---------------------------------------------------------------------------
struct Structure {
    int n = 0;                  // number of atoms
    std::vector<double> pos;    // [3*n] flat coordinates (Angstrom), row-major
};

// ---------------------------------------------------------------------------
// build_model: construct the (deterministic) ACSF hyperparameters + MLP weights.
//   In a real NNP the MLP weights are LEARNED from quantum-chemistry data. For a
//   self-contained, reproducible demo we synthesize them with a fixed, seeded
//   rule (the SAME rule scripts/make_synthetic.py documents). The descriptor
//   hyperparameters (cutoff Rc, width eta, shell centers Rs) are chosen to span
//   typical bonded/non-bonded distances (~0.8 .. Rc Angstrom).
//
//   These functions live in reference_cpu.cpp so both main.cu (via this header)
//   and the build see a single definition. They take no arguments and are
//   deterministic, so every run -- and the CPU and GPU paths -- use identical
//   parameters.
// ---------------------------------------------------------------------------
AcsfParams build_acsf_params();   // cutoff + Gaussian shells
AtomicNet  build_atomic_net();    // the fixed/"trained" per-atom network

// ---------------------------------------------------------------------------
// load_structure: read a structure from the text format in data/README.md:
//   line 1:  "<n>"                      (number of atoms)
//   next n:  "<x> <y> <z>"              (coordinates in Angstrom)
//   '#' begins a comment line (ignored). Throws std::runtime_error on a missing
//   file or a malformed/short body so demos fail loudly instead of on garbage.
// ---------------------------------------------------------------------------
Structure load_structure(const std::string& path);

// ---------------------------------------------------------------------------
// nnp_energy_cpu: the trusted CPU reference.
//   Fills e_atom[i] with atom i's energy E_i and returns the total E = sum E_i.
//   It loops atoms serially, calling the SAME atomic_energy() from nnp.h that
//   the GPU thread calls -- so the GPU result must match this to round-off. This
//   is both the correctness oracle and the timing baseline that makes the GPU
//   speed-up legible. e_atom is resized to s.n.
// ---------------------------------------------------------------------------
double nnp_energy_cpu(const Structure& s, const AcsfParams& p, const AtomicNet& net,
                      std::vector<double>& e_atom);
