// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 13.18 -- Pharmacogenomics-Guided Precision Dosing   (template skeleton)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop, no parallelism, no
//   cleverness -- so that when the GPU and CPU agree, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// TODO(impl): replace the SAXPY placeholder below with this project's real
//             reference algorithm (keep it simple and readable on purpose).
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

// out[i] = a * x[i] + y[i], computed serially on the CPU.
//   Complexity: O(n) time, O(1) extra space. This is the baseline whose wall
//   time (timed in main.cu via util::CpuTimer) we compare with the GPU kernel.
void saxpy_cpu(int n, float a, const std::vector<float>& x,
               const std::vector<float>& y, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(n), 0.0f);  // allocate + zero n outputs
    for (int i = 0; i < n; ++i) {
        // The whole computation in one line. Each output element depends only
        // on its own inputs -> no data dependencies between iterations, which
        // is exactly WHY this maps perfectly onto independent GPU threads
        // (one thread per i) in kernels.cu.
        out[static_cast<std::size_t>(i)] = a * x[static_cast<std::size_t>(i)]
                                             + y[static_cast<std::size_t>(i)];
    }
}
