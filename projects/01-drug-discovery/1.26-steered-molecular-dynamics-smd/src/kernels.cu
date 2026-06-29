// ===========================================================================
// src/kernels.cu  --  SMD-ensemble kernel (one thread per pulling trajectory)
// ---------------------------------------------------------------------------
// Project 1.26 : Steered Molecular Dynamics (SMD)
//
// WHAT THIS FILE DOES
//   Implements the device kernel (smd_kernel) and the host glue (run_gpu) that
//   allocates the work buffer, launches one thread per trajectory, times the
//   kernel, and copies the per-trajectory work back. This is the GPU twin of the
//   serial loop in reference_cpu.cpp -- and because BOTH call the shared
//   run_trajectory() in smd_core.h with the same per-trajectory seed, the two
//   work vectors are bit-for-bit identical. main.cu runs both and asserts it.
//
//   There is NO inter-thread communication and NO atomics here: each trajectory
//   writes its own slot work[i]. The only reduction (Jarzynski's average) is
//   done afterwards on the host, in a fixed order, so the demo's stdout is
//   deterministic (PATTERNS.md §3).
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), smd_core.h (physics).
// ===========================================================================
#include "kernels.cuh"
#include "smd_core.h"            // SmdParams, run_trajectory
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default for a register-heavy, compute-bound
// per-thread integrator on sm_75..sm_89: a multiple of the 32-lane warp, enough
// warps to hide latency, and it keeps per-thread register pressure (each thread
// holds the full Langevin state) from capping occupancy too hard. (Tune per GPU;
// THEORY.md "GPU mapping" discusses the register/occupancy trade-off.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// smd_kernel: thread `i` owns SMD trajectory i.
//   Launch config (set in run_gpu):
//     grid  = ceil(n_traj / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x * blockDim.x + threadIdx.x -> work[i].
//   Memory: the entire Langevin time loop runs in REGISTERS/local memory (xi,
//   center, W, the RNG state); the only global write is the single result
//   work[i]. No shared memory, no atomics -- pure embarrassing parallelism over
//   trajectories, exactly like the SEIR ensemble (9.02).
//   Divergence: every thread runs the same number of steps; the only data-
//   dependent branches are inside the RNG/log/cos, so warps stay coherent.
// ---------------------------------------------------------------------------
__global__ void smd_kernel(SmdParams p, double* __restrict__ work) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's traj
    if (i >= p.n_traj) return;                             // guard ragged last block

    // run_trajectory() is the SAME function the CPU reference calls; seeding from
    // (p.seed, i) makes trajectory i reproducible and equal on host and device.
    work[i] = run_trajectory(p, static_cast<uint64_t>(i));
}

// ---------------------------------------------------------------------------
// run_gpu: host wrapper. Output-only, so the only transfer is device->host.
//   (1) allocate the work buffer on the device,
//   (2) launch one thread per trajectory (timed with CUDA events),
//   (3) copy the per-trajectory work back, (4) free.
//   We time ONLY the launch so the reported figure is kernel cost, not the tiny
//   D2H copy (THEORY.md "GPU mapping" notes copies are negligible here because
//   only n_traj doubles cross the bus -- the compute dominates).
// ---------------------------------------------------------------------------
void run_gpu(const SmdParams& p, std::vector<double>& work, float* kernel_ms) {
    const int n = p.n_traj;
    work.assign(static_cast<std::size_t>(n), 0.0);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);

    // (1) Device output buffer (d_ prefix = DEVICE pointer; CLAUDE.md §12).
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, bytes));   // can fail: out of device memory

    // (2) Launch. Blocks must cover all n trajectories -> ceiling division.
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    smd_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_work);
    *kernel_ms = timer.stop_ms();             // GPU-measured kernel time
    CUDA_CHECK_LAST("smd_kernel");            // catch launch + execution errors

    // (3) Bring the per-trajectory work back to the host vector.
    CUDA_CHECK(cudaMemcpy(work.data(), d_work, bytes, cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_work));
}
