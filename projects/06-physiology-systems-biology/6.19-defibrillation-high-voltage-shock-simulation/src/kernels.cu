// ===========================================================================
// src/kernels.cu  --  GPU defibrillation-threshold sweep kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 6.19 : Defibrillation & High-Voltage Shock Simulation
//
// WHAT THIS FILE DOES
//   Implements sweep_gpu() (host glue) and sweep_kernel() (device entry point).
//   The kernel is the GPU twin of simulate_one_cpu(): thread k runs one full
//   1-D cable simulation for shock amplitude amps[k], calling the SAME shared
//   physics (cable_step / activity_metric from defib.h) the CPU reference uses.
//   main.cu runs both paths and asserts they agree to within a tight tolerance.
//
//   Pattern: ensemble of independent trajectories (PATTERNS.md section 1, like
//   9.02 / 13.02). One thread == one shock amplitude == one whole cable.
//
// READ THIS AFTER: kernels.cuh (the mapping idea), defib.h (the physics).
// ===========================================================================
#include "kernels.cuh"
#include "defib.h"               // FhnParams + shared cable_step / activity_metric
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here: each thread does a LOT of work
// (a whole time-stepped cable) and touches its own scratch, so we are latency-
// bound on global-memory ping-pong, not occupancy-bound. 128 gives 4 warps to
// hide that latency while keeping registers/thread comfortable. (Tune per GPU.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// sweep_kernel: thread k simulates the cable for shock amplitude d_amps[k].
//   Launch config (set in sweep_gpu):
//     grid  = ceil(namp / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: k = blockIdx.x * blockDim.x + threadIdx.x  (one shock).
//
//   MEMORY MODEL -- per-thread ping-pong scratch in GLOBAL memory:
//     Each thread needs two voltage buffers and two recovery buffers of length
//     ncell (double-buffered so a step reads the OLD field and writes the NEW).
//     We cannot size that on the stack (ncell is a runtime value and can exceed
//     the register/local budget), so the host pre-allocates one big slab and
//     hands each thread FOUR non-overlapping slices via simple index arithmetic:
//         Va = d_scratch + (0*namp + k) * ncell    (voltage buffer A)
//         Vb = d_scratch + (1*namp + k) * ncell    (voltage buffer B)
//         wa = d_scratch + (2*namp + k) * ncell    (recovery buffer A)
//         wb = d_scratch + (3*namp + k) * ncell    (recovery buffer B)
//     Because slices are disjoint per thread, there are NO data races and NO
//     atomics: threads never touch each other's memory.
//
//   The whole time loop lives inside this one thread, swapping its own pointers
//   each step -- byte-for-byte the same sequence of operations as the CPU. That
//   is what makes the GPU-vs-CPU comparison in main.cu essentially exact.
// ---------------------------------------------------------------------------
__global__ void sweep_kernel(FhnParams p, const double* __restrict__ d_amps,
                             int namp, double* __restrict__ d_scratch,
                             double* __restrict__ d_residual) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's shock
    if (k >= namp) return;                                 // guard ragged block

    const int n = p.ncell;
    const double amp = d_amps[k];

    // Carve this thread's four private buffers out of the shared slab (see above).
    double* Va = d_scratch + (static_cast<size_t>(0) * namp + k) * n;
    double* Vb = d_scratch + (static_cast<size_t>(1) * namp + k) * n;
    double* wa = d_scratch + (static_cast<size_t>(2) * namp + k) * n;
    double* wb = d_scratch + (static_cast<size_t>(3) * namp + k) * n;

    // Initial condition: left `initial_excited` cells excited (V=1), rest at
    // rest (V=0, w=0). Identical to simulate_one_cpu's initialisation.
    for (int i = 0; i < n; ++i) {
        Va[i] = (i < p.initial_excited) ? 1.0 : 0.0;
        wa[i] = 0.0;
    }

    // Ping-pong pointers, mirroring the CPU reference exactly.
    double* Vsrc = Va; double* Vdst = Vb;
    double* wsrc = wa; double* wdst = wb;

    // Time loop: advance the whole cable one step at a time using the shared
    // physics, then swap so the new field feeds the next step.
    for (int s = 0; s < p.nsteps; ++s) {
        cable_step(s, amp, p, Vsrc, wsrc, Vdst, wdst);
        double* tV = Vsrc; Vsrc = Vdst; Vdst = tV;
        double* tw = wsrc; wsrc = wdst; wdst = tw;
    }

    // Reduce this thread's final cable to a single residual-activity number.
    d_residual[k] = activity_metric(n, Vsrc);
}

// ---------------------------------------------------------------------------
// sweep_gpu: host wrapper. The canonical CUDA-computation steps:
//   (1) allocate device memory (amplitudes, per-thread scratch, residuals)
//   (2) copy amplitudes host->device
//   (3) launch the sweep kernel (timed with CUDA events -- kernel cost only)
//   (4) copy residuals device->host
//   (5) free device memory
// ---------------------------------------------------------------------------
void sweep_gpu(const FhnParams& p, const std::vector<double>& amps,
               std::vector<double>& residual, float* kernel_ms) {
    const int namp = static_cast<int>(amps.size());
    residual.assign(static_cast<std::size_t>(namp), 0.0);

    // (1) Device buffers. d_ prefix marks DEVICE pointers (CLAUDE.md section 12).
    //     Scratch holds 4 arrays (Va,Vb,wa,wb) of ncell doubles for each of the
    //     namp threads -> 4 * namp * ncell doubles total.
    const std::size_t scratch_elems =
        static_cast<std::size_t>(4) * namp * p.ncell;
    double *d_amps = nullptr, *d_scratch = nullptr, *d_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amps,     static_cast<std::size_t>(namp) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scratch,  scratch_elems * sizeof(double)));   // may OOM on big sweeps
    CUDA_CHECK(cudaMalloc(&d_residual, static_cast<std::size_t>(namp) * sizeof(double)));

    // (2) Upload the amplitudes (the only real input; scratch is thread-private).
    CUDA_CHECK(cudaMemcpy(d_amps, amps.data(),
                          static_cast<std::size_t>(namp) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // (3) Launch: one thread per amplitude, blocks cover all namp (ceiling div).
    const int blocks = (namp + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    sweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_amps, namp, d_scratch, d_residual);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("sweep_kernel");       // catch launch + execution errors

    // (4) Bring the per-amplitude residuals back to the host vector.
    CUDA_CHECK(cudaMemcpy(residual.data(), d_residual,
                          static_cast<std::size_t>(namp) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_amps));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_residual));
}
