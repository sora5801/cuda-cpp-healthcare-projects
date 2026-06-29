// ===========================================================================
// src/kernels.cu  --  GPU ensemble GaMD kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   Implements ensemble_kernel (one thread per GaMD-boosted Langevin walker) and
//   run_ensemble_gpu (the host glue: allocate + zero the device tally, launch,
//   time, copy back). This is the GPU twin of run_ensemble_cpu(); main.cu runs
//   both and asserts the fixed-point tallies match BIT-FOR-BIT.
//
//   The per-walker physics is NOT here -- it is the shared run_walker() in gamd.h,
//   which both this kernel and the CPU reference call (PATTERNS.md §2). This file
//   only supplies the GPU's *adder*: a deterministic fixed-point atomicAdd.
//
// READ THIS AFTER: kernels.cuh (the thread-per-walker idea), gamd.h (the physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default for a register-heavy integrator like
// this on sm_75..sm_89: a multiple of the 32-lane warp, enough warps to hide
// latency, and modest register pressure so occupancy stays reasonable. The walker
// loop lives entirely in registers/local memory, so there is no shared-memory
// constraint to balance against.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// device_add: the GPU's deterministic fixed-point adder, passed into run_walker.
//   run_walker() deposits int64 increments (a count of 1, and fixed-point dV /
//   dV^2). Many walkers hit the same bin, so the add MUST be atomic. We use the
//   64-bit atomicAdd, which CUDA exposes only for `unsigned long long`. Adding
//   signed two's-complement values THROUGH an unsigned atomic is well-defined:
//   the bit pattern of the sum is identical, so reinterpreting back to int64
//   gives the correct signed total. Crucially, INTEGER addition is associative
//   and commutative, so the final tally is independent of the (nondeterministic)
//   order in which threads arrive -> the GPU result equals the serial CPU result
//   EXACTLY (PATTERNS.md §3 rule 2). A float atomicAdd would NOT have this
//   property and verification could only be approximate.
// ---------------------------------------------------------------------------
struct DeviceAdder {
    long long* acc;   // device tally base pointer ([count | S1 | S2], 3*n_bins)
    __device__ void operator()(int idx, int64_t v) const {
        // Reinterpret the int64 slot as unsigned long long for the atomic, add the
        // (possibly negative) increment reinterpreted the same way; two's-complement
        // wraparound yields the correct signed sum bit-for-bit.
        atomicAdd(reinterpret_cast<unsigned long long*>(&acc[idx]),
                  static_cast<unsigned long long>(v));
    }
};

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx == walker idx.
//   Launch config (set in run_ensemble_gpu):
//     grid  = ceil(n_walkers / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x*blockDim.x + threadIdx.x  (walker index).
//   Memory: each thread keeps its walker state (x, RNG counters) in registers and
//   only touches GLOBAL memory through the atomic histogram deposits and the final
//   d_final_x[idx] write. The whole time loop runs in run_walker() (gamd.h).
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(GamdConfig c,
                                long long* __restrict__ d_acc,
                                double* __restrict__ d_final_x) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's walker
    if (idx >= c.n_walkers) return;                         // guard ragged last block

    // Hand run_walker() the device atomic adder; it returns the final position.
    DeviceAdder add{d_acc};
    d_final_x[idx] = run_walker(c, static_cast<uint32_t>(idx), add);
}

// ---------------------------------------------------------------------------
// run_ensemble_gpu: host wrapper. The canonical CUDA steps, with the twist that
//   the OUTPUT is an accumulator that must start at zero on the device:
//     (1) allocate device tally (3*n_bins int64) + final-x (n_walkers double)
//     (2) ZERO the tally with cudaMemset (atomicAdd accumulates onto it)
//     (3) launch one thread per walker (time only this with CUDA events)
//     (4) copy tally + final positions device->host
//     (5) free device memory
// ---------------------------------------------------------------------------
void run_ensemble_gpu(const GamdConfig& c,
                      std::vector<int64_t>& acc,
                      std::vector<double>& final_x,
                      float* kernel_ms) {
    const int    n_acc   = acc_total(c);                    // 3*n_bins int64 slots
    const int    n_walk  = c.n_walkers;
    const std::size_t acc_bytes = static_cast<std::size_t>(n_acc)  * sizeof(long long);
    const std::size_t fx_bytes  = static_cast<std::size_t>(n_walk) * sizeof(double);

    acc.assign(static_cast<std::size_t>(n_acc), 0);
    final_x.assign(static_cast<std::size_t>(n_walk), 0.0);

    // (1) Device buffers. d_ prefix = DEVICE pointer (CLAUDE.md §12).
    long long* d_acc     = nullptr;   // the [count|S1|S2] fixed-point tally
    double*    d_final_x = nullptr;   // each walker's final position
    CUDA_CHECK(cudaMalloc(&d_acc,     acc_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_final_x, fx_bytes));

    // (2) Zero the tally: every bit must be 0 because the kernel ONLY adds to it.
    //     cudaMemset sets bytes, and 0 bytes == integer 0, so this is correct for
    //     int64. (final_x is fully overwritten, so it need not be pre-zeroed.)
    CUDA_CHECK(cudaMemset(d_acc, 0, acc_bytes));

    // (3) Launch one thread per walker; round the grid up to cover all walkers.
    const int blocks = (n_walk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_acc, d_final_x);
    *kernel_ms = timer.stop_ms();                    // GPU-measured kernel time
    CUDA_CHECK_LAST("ensemble_kernel");              // catch launch + execution errors

    // (4) Bring the tally and final positions back to host vectors.
    CUDA_CHECK(cudaMemcpy(acc.data(),     d_acc,     acc_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(final_x.data(), d_final_x, fx_bytes,  cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_final_x));
}
