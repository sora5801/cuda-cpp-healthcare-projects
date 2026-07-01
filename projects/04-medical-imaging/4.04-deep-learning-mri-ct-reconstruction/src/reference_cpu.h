// ===========================================================================
// src/reference_cpu.h  --  Prototypes of the CPU reference + data plumbing
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHY A SEPARATE HEADER
//   reference_cpu.cpp is compiled by the plain C++ compiler and must NOT see any
//   CUDA/__global__ syntax. Its prototypes therefore live here (pure C++), not in
//   kernels.cuh. Both main.cu and reference_cpu.cpp include this header so they
//   agree on the signatures. It reuses the Acquisition/ReconParams types declared
//   in kernels.cuh (which is itself plain C++), so we include that.
//
// WHAT THE CPU SIDE PROVIDES
//   * load_acquisition       -- parse a data/sample file into an Acquisition.
//   * make_synthetic_acquisition -- build the built-in synthetic phantom + scan
//                                   when no file is supplied.
//   * recon_cpu              -- the trusted, serial unrolled reconstruction; the
//                              GPU result (recon_gpu) is verified against it.
//   * rms_error              -- root-mean-square error between two images, our
//                              science-level score (recon vs ground truth).
//
//   The CPU reference exists for two reasons (CLAUDE.md section 5): (a) it is the
//   readable baseline that makes the GPU speed-up legible, and (b) the demo runs
//   BOTH and asserts they agree within tolerance.
//
// READ THIS AFTER: kernels.cuh. READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "kernels.cuh"   // Acquisition, ReconParams (plain C++ types)

// Build the built-in synthetic acquisition: a small piecewise-constant phantom
//   (a bright disk + a square, on a dim background), forward-transformed to
//   k-space and then UNDER-SAMPLED by a mask that keeps the low frequencies plus
//   a fraction of the high ones. Everything is clearly synthetic (see data/README).
//   ny,nx : image size to generate (kept small so the O(N^2) DFT is instant).
Acquisition make_synthetic_acquisition(int ny, int nx);

// Parse an Acquisition from a whitespace-separated sample file (see data/README
//   for the exact layout). Returns true on success; false if the file is missing
//   or malformed, so the caller can fall back to the synthetic phantom.
bool load_acquisition(const std::string& path, Acquisition& acq);

// The CPU reference reconstruction: the SAME unrolled cascade as recon_gpu(),
//   run serially. Fills `recon` ([ny*nx], row-major). This is the ground truth
//   the GPU result is compared against within tolerance.
void recon_cpu(const Acquisition& acq, const ReconParams& p,
               std::vector<float>& recon);

// Root-mean-square error between two equal-length images. Our science score:
//   how close a reconstruction is to the ground-truth phantom (lower = better).
double rms_error(const std::vector<float>& a, const std::vector<float>& b);
