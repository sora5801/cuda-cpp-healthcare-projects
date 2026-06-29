// ===========================================================================
// src/kernels.cu  --  GPU ensemble induced-dipole solver (one thread per system)
// ---------------------------------------------------------------------------
// Project 1.21 : Polarizable / AMOEBA Force Field MD
//
// WHAT THIS FILE DOES
//   The GPU twin of integrate_cpu(): each thread runs the SAME matrix-free
//   conjugate-gradient solve (solve_induced_dipoles, amoeba.h) for one ensemble
//   member and writes one PerSystemResult. The host wrapper handles the standard
//   CUDA lifecycle (allocate, copy H2D, launch + time, copy D2H, free). main.cu
//   compares the per-member results against the CPU reference.
//
//   This is the "custom CUDA conjugate-gradient solver for induced dipoles" the
//   catalog calls for -- written by hand (no library) so nothing is a black box.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), amoeba.h (the physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good default here: each thread does a fair amount
// of register-heavy work (the CG arrays are on-stack), so we keep the block
// modest to leave registers for occupancy on sm_75..sm_89. A multiple of the
// 32-lane warp keeps the scheduler happy. (Tune per GPU; see THEORY.md.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// dipole_ensemble_kernel: thread idx solves member idx.
//   Launch config (set in solve_ensemble_gpu):
//     grid  = ceil(M / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x  ->  member idx.
//
//   Memory: reads systems[idx] from global memory (a struct load), keeps the
//   entire CG working set (mu, r, p, Ap -- each AMOEBA_MAX_ATOMS x 3 doubles) in
//   LOCAL memory / registers, writes out[idx]. No shared memory, no atomics, no
//   cross-thread dependence: the cleanest possible "independent jobs" mapping.
//
//   Divergence is mild: members may take slightly different CG iteration counts,
//   so threads in a warp finish their loops at different times. That is fine for
//   a teaching ensemble; for tightly-matched workloads one would pad to a fixed
//   iteration count.
// ---------------------------------------------------------------------------
__global__ void dipole_ensemble_kernel(const AtomSystem* __restrict__ systems,
                                       int M, double tol, int max_iter,
                                       PerSystemResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;                  // guard the ragged last block

    // Copy this thread's system into a local (register/stack) AtomSystem so the
    // inner O(n^2) matvec reads from fast local memory rather than re-fetching
    // global memory on every CG iteration. The struct is small (<= 32 atoms).
    AtomSystem s = systems[idx];

    double mu[AMOEBA_MAX_ATOMS][3];        // the converged dipoles (scratch)
    out[idx] = solve_induced_dipoles(s, tol, max_iter, mu);
}

// ---------------------------------------------------------------------------
// solve_ensemble_gpu: host wrapper. The canonical CUDA lifecycle:
//   (1) allocate device buffers  (2) copy the ensemble H2D
//   (3) launch + TIME the kernel (CUDA events)  (4) copy results D2H  (5) free.
//   We time ONLY the kernel (step 3) so the figure reflects compute, not PCIe.
// ---------------------------------------------------------------------------
void solve_ensemble_gpu(const EnsembleConfig& c,
                        std::vector<PerSystemResult>& results,
                        float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), PerSystemResult{});

    const std::size_t sys_bytes = static_cast<std::size_t>(M) * sizeof(AtomSystem);
    const std::size_t res_bytes = static_cast<std::size_t>(M) * sizeof(PerSystemResult);

    // (1) Device buffers. d_ marks DEVICE pointers (dereferencing on host crashes).
    AtomSystem*      d_sys = nullptr;      // the M input systems
    PerSystemResult* d_out = nullptr;      // the M output summaries
    CUDA_CHECK(cudaMalloc(&d_sys, sys_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_out, res_bytes));

    // (2) Copy the whole ensemble H2D in one contiguous transfer. AtomSystem is a
    //     plain-old-data struct (fixed arrays, no pointers), so it is trivially
    //     copyable -- a flat memcpy is correct, no per-atom marshalling needed.
    CUDA_CHECK(cudaMemcpy(d_sys, c.systems.data(), sys_bytes, cudaMemcpyHostToDevice));

    // (3) Launch one thread per member; ceil-divide so blocks cover all M.
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    dipole_ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_sys, M, c.tol, c.max_iter, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("dipole_ensemble_kernel");   // catch launch + execution errors

    // (4) Bring the per-member results back.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out, res_bytes, cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_sys));
    CUDA_CHECK(cudaFree(d_out));
}
