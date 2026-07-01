// ===========================================================================
// src/reference_cpu.h  --  Dataset + shared helpers + serial ICP reference
// ---------------------------------------------------------------------------
// Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
//
// Pure C++ (NO CUDA): compiled by the host compiler for the CPU reference and
// also included by kernels.cu (nvcc happily compiles plain C++ too). The
// per-point math lives in icp.h (the __host__ __device__ core). This file adds:
//   * Clouds        -- the two point clouds (moving P, fixed Q) + ground truth.
//   * load_clouds   -- parse the tiny text sample in data/sample/.
//   * rms_error     -- report the alignment quality (mm), a shared metric.
//   * icp_cpu       -- the trusted serial ICP that the GPU must reproduce.
//
// READ THIS AFTER: icp.h.  READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "icp.h"   // Vec3, Rigid, AccumFixed, nearest_index, solve_rigid, ...

// ---------------------------------------------------------------------------
// Clouds: everything the demo needs, loaded from one text file.
//   P  -- the MOVING cloud (pre-operative surface points), np of them.
//   Q  -- the FIXED  cloud (intra-operative surface points), nq of them.
//   gt -- the GROUND-TRUTH rigid transform that was applied to build P from a
//         subset of Q (synthetic data only). ICP should recover its INVERSE:
//         the transform that maps P back onto Q. We keep gt only to describe
//         how the sample was made; the demo's correctness metric is the RMS
//         alignment error, which needs no ground truth.
// ---------------------------------------------------------------------------
struct Clouds {
    std::vector<Vec3> P;     // moving points (pre-op surface)
    std::vector<Vec3> Q;     // fixed  points (intra-op surface)
    Rigid gt;                // ground-truth transform used to synthesize P
    bool  has_gt = false;    // whether the file carried a ground-truth block
};

// Load the sample format (see data/README.md):
//   line: "np nq"
//   optional line: "GT" followed by 12 numbers (R row-major 9, then t 3)
//   np lines of "x y z"  (moving cloud P)
//   nq lines of "x y z"  (fixed  cloud Q)
Clouds load_clouds(const std::string& path);

// Root-mean-square nearest-neighbour distance (mm) between the (transformed)
// moving cloud and the fixed cloud: our headline alignment-quality number. It
// is the same value ICP is driving down. Shared by CPU and GPU for one metric.
double rms_error(const std::vector<Vec3>& P, const std::vector<Vec3>& Q, const Rigid& g);

// Serial ICP reference: run `iters` fixed iterations starting from identity.
// Fills `history` with the RMS error (mm) AFTER each iteration so main.cu can
// print a deterministic convergence curve. Returns the final recovered
// transform (maps P onto Q). This is the trusted baseline the GPU reproduces.
Rigid icp_cpu(const Clouds& c, int iters, std::vector<double>& history);
