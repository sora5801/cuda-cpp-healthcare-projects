// ===========================================================================
// src/reference_cpu.h  --  Data model + host helpers + CPU reference for DTI
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the diffusion volume container, the
//   file loader, the fixed gradient scheme, the OLS pseudo-inverse builder) and
//   the CPU reference prototypes live here. kernels.cuh also includes this header
//   to reuse the data model -- nothing CUDA-specific leaks in either direction.
//   The actual per-voxel PHYSICS is one level deeper, in dti_core.h (shared by
//   host and device), which this header pulls in.
//
// THE PROBLEM (short version; full derivation in ../THEORY.md)
//   A diffusion MRI acquisition gives, for every voxel, NMEAS signal intensities:
//   one b=0 image plus NDIR diffusion-weighted images. Fitting the 3x3 diffusion
//   tensor is an independent least-squares problem per voxel -> perfect data
//   parallelism (one GPU thread per voxel, in kernels.cu). From each tensor we
//   derive FA/MD scalar maps and the principal fiber direction, then trace a few
//   deterministic streamlines through the direction field (tractography).
//
// READ THIS AFTER: dti_core.h. READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "dti_core.h"   // NDIR, NMEAS, NPARAM, VoxelResult, fit_voxel (host+device)

// ---------------------------------------------------------------------------
// GradientScheme: the fixed acquisition geometry shared by every voxel.
//   bval    : NMEAS b-values (s/mm^2). Entry 0 is 0 (the b=0 image); the rest
//             share one nonzero shell b-value in this teaching set.
//   gx/gy/gz: NMEAS unit gradient directions (entry 0 is the zero vector).
//   These populate the design matrix B (NMEAS x NPARAM) whose pseudo-inverse
//   Minv (NPARAM x NMEAS) is the fixed OLS operator used by fit_voxel().
// ---------------------------------------------------------------------------
struct GradientScheme {
    std::vector<double> bval;         // [NMEAS]
    std::vector<double> gx, gy, gz;   // [NMEAS] each
};

// ---------------------------------------------------------------------------
// DwiVolume: a loaded diffusion-weighted dataset (a small 3-D grid of voxels).
//   nx,ny,nz : grid dimensions; nvox = nx*ny*nz.
//   signal   : nvox * NMEAS row-major -- voxel v occupies
//              signal[v*NMEAS .. v*NMEAS + NMEAS-1]. signal[v*NMEAS+0] is the
//              b=0 intensity of voxel v.
//   mask     : nvox flags (1 = tissue, 0 = background). Background voxels are
//              still fitted (their FA is ~0); the mask exists so tractography
//              seeds only in tissue. Stored as int for simple text I/O.
// ---------------------------------------------------------------------------
struct DwiVolume {
    int nx = 0, ny = 0, nz = 0;
    int nvox = 0;
    std::vector<double> signal;   // [nvox * NMEAS]
    std::vector<int>    mask;     // [nvox]
};

// Load a DWI volume from the text format documented in data/README.md.
//   Throws std::runtime_error on a missing file or an NMEAS mismatch.
DwiVolume load_dwi(const std::string& path);

// Build the fixed 12-direction gradient scheme used by this teaching project.
// (Defined in reference_cpu.cpp so the CPU and GPU sides agree on the geometry.)
GradientScheme make_gradient_scheme();

// Build the OLS pseudo-inverse Minv (NPARAM x NMEAS, row-major) from a gradient
// scheme: Minv = (B^T B)^{-1} B^T, where row k of B is
//   [ 1, -b_k gx^2, -b_k gy^2, -b_k gz^2, -2 b_k gx gy, -2 b_k gx gz, -2 b_k gy gz ].
// This is the SAME operator for every voxel; we compute it once (small 7x7
// inverse) and upload it to GPU constant memory. Returns a length NPARAM*NMEAS
// vector. Defined in reference_cpu.cpp.
std::vector<double> build_pseudo_inverse(const GradientScheme& scheme);

// CPU reference: fit every voxel, filling out[v] with its VoxelResult. This is
// the trusted baseline the GPU kernel is checked against (and the timing
// baseline that makes the speed-up legible). out is resized to vol.nvox.
void fit_all_voxels_cpu(const DwiVolume& vol, const std::vector<double>& Minv,
                        std::vector<VoxelResult>& out);

// ---------------------------------------------------------------------------
// Streamline (deterministic tractography) -- one traced polyline.
//   pts    : flattened x0,y0,z0, x1,y1,z1, ... in continuous voxel coordinates.
//   nsteps : number of points recorded (path length in vertices).
// ---------------------------------------------------------------------------
struct Streamline {
    std::vector<float> pts;   // 3 * nsteps floats
    int nsteps = 0;
};

// Trace all seeds on the CPU (reference). `seeds` is a flat list of (x,y,z)
// voxel coordinates. Fills `lines` (one Streamline per seed) using the fitted
// per-voxel principal directions. Stops a streamline when FA < fa_min, the path
// leaves the volume, or the local direction turns by more than acos(cos_min).
//   step     : integration step length in voxels.
//   max_steps: cap on vertices per streamline.
void trace_streamlines_cpu(const DwiVolume& vol,
                           const std::vector<VoxelResult>& fit,
                           const std::vector<float>& seeds,
                           int max_steps, float step, float fa_min, float cos_min,
                           std::vector<Streamline>& lines);
