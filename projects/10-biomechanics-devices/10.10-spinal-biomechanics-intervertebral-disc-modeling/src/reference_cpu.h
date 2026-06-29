// ===========================================================================
// src/reference_cpu.h  --  Prototype of the CPU reference computation
// ---------------------------------------------------------------------------
// Project 10.10 -- Spinal Biomechanics & Intervertebral Disc Modeling   (template skeleton)
//
// WHY A SEPARATE HEADER
//   The CPU reference (reference_cpu.cpp) is compiled by the plain C++ compiler
//   and must NOT see any CUDA/__global__ syntax, so its prototype cannot live in
//   kernels.cuh. Both main.cu and reference_cpu.cpp include THIS pure-C++ header
//   so they agree on the function signature.
//
// THE CONTRACT (this template's placeholder computation):
//   SAXPY -- "Single-precision A*X Plus Y":  out[i] = a * x[i] + y[i].
//   This is the canonical first GPU kernel; here it stands in as a buildable
//   placeholder. TODO(impl): replace saxpy_cpu with this project's real
//   reference computation, and update the prototype + callers accordingly.
//
//   The CPU reference exists for two reasons (CLAUDE.md section 5):
//     (a) it is the readable baseline that makes the GPU speed-up legible, and
//     (b) the demo runs BOTH and asserts they agree within tolerance.
// ===========================================================================
#pragma once

#include <vector>

// Compute out = a*x + y on the CPU, element by element.
//   x, y : input vectors of equal length n
//   a    : the scalar multiplier
//   out  : resized to n and filled with the result (output parameter)
void saxpy_cpu(int n, float a, const std::vector<float>& x,
               const std::vector<float>& y, std::vector<float>& out);
