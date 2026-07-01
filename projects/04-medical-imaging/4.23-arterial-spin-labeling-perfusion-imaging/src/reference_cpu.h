// ===========================================================================
// src/reference_cpu.h  --  ASL dataset loader + CPU reference fit
// ---------------------------------------------------------------------------
// Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
//
// Pure C++ (no CUDA). The Buxton model, its Jacobian, and the Gauss-Newton solver
// live in asl.h as __host__ __device__ functions; kernels.cu reuses AslDataset +
// AslFit. Because the CPU reference here and the GPU kernel both call the SAME
// asl_fit_voxel(), their per-voxel fits agree to round-off (verified in main.cu).
//
// This header also owns the on-disk representation: a HostDataset struct that
// actually OWNS the pld[] and signal[] vectors (AslDataset only holds pointers),
// plus ground-truth arrays for the science check (recovered CBF/ATT vs. truth).
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.  AFTER: asl.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "asl.h"   // AslDataset, AslFit, AslConstants, asl_fit_voxel

// ---------------------------------------------------------------------------
// HostDataset: the loaded ASL study, owning its buffers.
//   AslDataset (in asl.h) is a lightweight VIEW with raw pointers so the same
//   struct can point at host OR device memory. HostDataset owns the host vectors
//   and can hand out an AslDataset view via `view()`.
//
//   The file also carries GROUND TRUTH (the true CBF/ATT used to synthesize each
//   voxel's noise-free curve) so we can report how well the fit recovers the
//   known physiology -- the "embed a known answer" idiom (docs/PATTERNS.md §6).
// ---------------------------------------------------------------------------
struct HostDataset {
    int n_voxels = 0;
    int n_plds   = 0;
    std::vector<double> pld;        // [n_plds]            delay schedule (s)
    std::vector<double> signal;     // [n_voxels*n_plds]   measured delta-M
    std::vector<double> true_cbf;   // [n_voxels]          ground-truth CBF (mL/100g/min)
    std::vector<double> true_att;   // [n_voxels]          ground-truth ATT (s)
    AslConstants consts{};          // acquisition constants used for the fit
    int    max_iters = 30;          // Gauss-Newton cap
    double f_init   = 30.0;         // initial CBF guess (mL/100g/min)
    double att_init = 0.7;          // initial ATT guess (s)

    // Build a pointer-view over the host buffers for asl_fit_voxel().
    AslDataset view() const {
        AslDataset d;
        d.n_voxels  = n_voxels;
        d.n_plds    = n_plds;
        d.pld       = pld.data();
        d.signal    = signal.data();
        d.consts    = consts;
        d.max_iters = max_iters;
        d.f_init    = f_init;
        d.att_init  = att_init;
        return d;
    }
};

// Load the ASL sample text file (format documented in data/README.md):
//   line 1:  n_voxels  n_plds  max_iters  f_init  att_init
//   line 2:  pld_0 pld_1 ... pld_{n_plds-1}
//   then n_voxels lines, each:  true_cbf  true_att  s_0 s_1 ... s_{n_plds-1}
// Throws std::runtime_error on any malformed input so demos fail loudly.
HostDataset load_asl(const std::string& path);

// CPU reference: fit every voxel serially with asl_fit_voxel(). `fits` is sized
// to n_voxels. This is the trusted baseline the GPU result is checked against.
void fit_cpu(const HostDataset& ds, std::vector<AslFit>& fits);
