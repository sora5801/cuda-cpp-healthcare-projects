// ===========================================================================
// src/kernels.cu  --  Monte Carlo kernel: one thread = one replica walk
// ---------------------------------------------------------------------------
// Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
//
// WHAT THIS FILE DOES
//   GPU twin of sample_cpu(): it runs the IDENTICAL per-replica walks (the
//   shared run_replica() in mc_moves.h), but all replicas at once -- one CUDA
//   thread per replica. main.cu runs both paths and asserts the per-replica
//   {best, final} energies match exactly. See ../THEORY.md "GPU mapping".
//
//   The kernel itself is almost trivial precisely because the hard part (the
//   physics + RNG) lives in the shared header. That separation is the lesson:
//   write the per-history math ONCE, run it serially for the reference and
//   massively-parallel for the GPU, and the results are bit-identical.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), mc_moves.h (the walk).
// ===========================================================================
#include "kernels.cuh"
#include "mc_moves.h"            // run_replica, McProblem, McResult (host+device)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide latency, and it leaves many blocks resident
// for occupancy. Each replica walk is heavy and register-hungry (it keeps the
// chain coordinates x[],y[] in registers/local memory), so a moderate block
// size avoids spilling pressure -- tune per GPU.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// sample_kernel: one thread runs one replica's entire Metropolis walk.
//   Launch config (set in sample_gpu):
//     grid  = ceil(n_replicas / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: r = blockIdx.x * blockDim.x + threadIdx.x  (the replica).
//   Memory:
//     * prob is passed BY VALUE -> it lands in the kernel's parameter space and
//       is broadcast to every thread (read-only); no copy-in code needed.
//     * tables is a read-only device array; thread r reads its own slice
//       tables[r*table_stride .. ]. Many threads read different slices, no
//       contention, no atomics.
//     * out[r] is written by exactly one thread -> independent outputs, so
//       again no atomics are required (contrast project 5.01's dose tally).
//     * The chain coordinates inside run_replica live in per-thread local
//       memory (register-backed for small n) -- private to each walk.
//   Divergence note: different replicas accept/reject different moves, so warps
//   diverge on the Metropolis branch. That is intrinsic to MC; the walks are
//   short and balanced here, so it costs little (THEORY.md discusses sorting /
//   warp-coherent moves used by production codes).
// ---------------------------------------------------------------------------
__global__ void sample_kernel(McProblem prob, const double* __restrict__ tables,
                              int table_stride, McResult* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's replica
    if (r >= prob.n_replicas) return;                      // guard the ragged block

    // This replica's prebuilt Boltzmann table slice (read-only).
    const double* tbl = tables + (size_t)r * table_stride;

    // Run the SAME walk the CPU runs for replica r. Identical RNG stream
    // (seed,r) + identical table => identical sequence of accept/reject =>
    // identical {best,final} energy. We just write it to our own output slot.
    out[r] = run_replica(prob, r, tbl);
}

// ---------------------------------------------------------------------------
// sample_gpu: host wrapper. The canonical CUDA steps:
//   (1) allocate device buffers for the tables and the results
//   (2) copy the prebuilt Boltzmann tables host->device
//   (3) launch one thread per replica (timed with CUDA events)
//   (4) copy the per-replica results device->host
//   (5) free device memory
// We time ONLY the kernel (step 3), not the tiny copies, so the reported figure
// is the compute cost (PCIe transfer is negligible here and discussed in THEORY).
// ---------------------------------------------------------------------------
void sample_gpu(const McProblem& prob, const std::vector<double>& tables,
                std::vector<McResult>& out, float* kernel_ms) {
    const int R = prob.n_replicas;
    const int stride = boltzmann_table_size();          // doubles per replica table
    out.assign((std::size_t)R, McResult{});

    // (1) Device buffers.
    double*   d_tables = nullptr;                        // R * stride doubles
    McResult* d_out    = nullptr;                        // R results
    const std::size_t tbl_bytes = (std::size_t)R * stride * sizeof(double);
    const std::size_t out_bytes = (std::size_t)R * sizeof(McResult);
    CUDA_CHECK(cudaMalloc(&d_tables, tbl_bytes));        // can fail: out of memory
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    // (2) Upload the Boltzmann tables (computed once on the host in main.cu).
    CUDA_CHECK(cudaMemcpy(d_tables, tables.data(), tbl_bytes, cudaMemcpyHostToDevice));

    // (3) Launch: enough blocks to cover all replicas (ceiling division).
    const int blocks = (R + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    sample_kernel<<<blocks, THREADS_PER_BLOCK>>>(prob, d_tables, stride, d_out);
    *kernel_ms = timer.stop_ms();                        // GPU-measured kernel time
    CUDA_CHECK_LAST("sample_kernel");                    // launch + execution errors

    // (4) Bring the results back.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_tables));
    CUDA_CHECK(cudaFree(d_out));
}
