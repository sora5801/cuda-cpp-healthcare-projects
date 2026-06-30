// ===========================================================================
// src/kernels.cu  --  GPU REST2 sampler (one thread per replica)
// ---------------------------------------------------------------------------
// Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
//
// GPU twin of cpu_sample_round(): each thread runs the same Metropolis MC loop
// (rest2.h) for one replica, then writes back its updated coordinates and accept
// count. The host (main.cu) does the periodic REST2 exchange between rounds.
// Because the per-replica math is shared and the RNG is a deterministic counter
// hash, the GPU result matches the CPU reference exactly. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event kernel timing)

// Threads per block. Replica counts in REST2 are modest (tens, occasionally low
// hundreds), so one block usually covers the whole ladder. 64 is a comfortable
// default: a multiple of the 32-thread warp, low enough that a small ladder does
// not waste a giant block. The grid is sized to cover any n_replicas.
static constexpr int THREADS_PER_BLOCK = 64;

// ---------------------------------------------------------------------------
// sample_round_kernel: thread r owns replica r.
//   Thread-to-data map: r = blockIdx.x * blockDim.x + threadIdx.x indexes the
//   replica. The guard `if (r >= n)` retires threads in the ragged last block.
//   The replica's N_SOLUTE coordinates are loaded into a small register array,
//   advanced by sweeps_per_round MC sweeps entirely in registers (no global
//   traffic in the inner loop -> fast and divergence-free except the Metropolis
//   accept branch), then written back. There is NO inter-thread communication,
//   so no __syncthreads / shared memory / atomics are needed here.
// ---------------------------------------------------------------------------
__global__ void sample_round_kernel(SimConfig cfg,
                                    const ReplicaParams* __restrict__ reps,
                                    double*   __restrict__ coords,
                                    long*     __restrict__ accepted,
                                    uint64_t* __restrict__ rng_ctr) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's replica
    if (r >= cfg.n_replicas) return;                       // guard ragged last block

    // Pull this replica's beads into registers/local memory. N_SOLUTE is a
    // compile-time constant, so the compiler can keep this array in registers.
    double x[N_SOLUTE];
    #pragma unroll
    for (int i = 0; i < N_SOLUTE; ++i)
        x[i] = coords[static_cast<long long>(r) * N_SOLUTE + i];

    const ReplicaParams p = reps[r];   // (lambda, seed) for this replica
    uint64_t ctr = rng_ctr[r];         // this replica's RNG stream cursor
    long acc = 0;                      // accepted moves this round

    // The full sampling loop runs in this single thread -- exactly the inner work
    // cpu_sample_round() does serially for one replica. mc_sweep() is the shared
    // __host__ __device__ routine, so the numbers match the CPU bit-for-bit.
    for (int s = 0; s < cfg.sweeps_per_round; ++s)
        acc += mc_sweep(x, p, cfg, ctr);

    // Write the updated state back to global memory for the host to read and
    // (possibly) exchange between rounds.
    #pragma unroll
    for (int i = 0; i < N_SOLUTE; ++i)
        coords[static_cast<long long>(r) * N_SOLUTE + i] = x[i];
    accepted[r] += acc;     // accumulate (each thread owns its own slot -> no race)
    rng_ctr[r]   = ctr;     // persist the advanced cursor for the next round
}

// ---------------------------------------------------------------------------
// gpu_sample_round: host wrapper -- upload state, launch, download state.
//   We allocate fresh device buffers each call for clarity (the per-round data
//   is tiny). A production code would allocate once and keep state resident; the
//   re-upload here makes the host<->device data flow visible to the learner and
//   lets the host own the exchange step. Kernel time is measured with CUDA events
//   (util/timer.cuh) and reported as a teaching artifact, never a benchmark.
// ---------------------------------------------------------------------------
void gpu_sample_round(const SimConfig& cfg,
                      const std::vector<ReplicaParams>& reps,
                      std::vector<double>& coords,
                      std::vector<long>& accepted,
                      std::vector<uint64_t>& rng_ctr,
                      float* kernel_ms) {
    const int M = cfg.n_replicas;
    const std::size_t n_coords = static_cast<std::size_t>(M) * N_SOLUTE;

    // --- device buffers ----------------------------------------------------
    ReplicaParams* d_reps = nullptr;
    double*        d_coords = nullptr;
    long*          d_acc = nullptr;
    uint64_t*      d_ctr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reps,   static_cast<std::size_t>(M) * sizeof(ReplicaParams)));
    CUDA_CHECK(cudaMalloc(&d_coords, n_coords * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_acc,    static_cast<std::size_t>(M) * sizeof(long)));
    CUDA_CHECK(cudaMalloc(&d_ctr,    static_cast<std::size_t>(M) * sizeof(uint64_t)));

    // --- upload current state (H2D) ---------------------------------------
    CUDA_CHECK(cudaMemcpy(d_reps,   reps.data(),     static_cast<std::size_t>(M) * sizeof(ReplicaParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coords, coords.data(),   n_coords * sizeof(double),                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_acc,    accepted.data(), static_cast<std::size_t>(M) * sizeof(long),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctr,    rng_ctr.data(),  static_cast<std::size_t>(M) * sizeof(uint64_t),      cudaMemcpyHostToDevice));

    // --- launch: one thread per replica -----------------------------------
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    sample_round_kernel<<<blocks, THREADS_PER_BLOCK>>>(cfg, d_reps, d_coords, d_acc, d_ctr);
    *kernel_ms = timer.stop_ms();        // blocks until the kernel finishes
    CUDA_CHECK_LAST("sample_round_kernel");

    // --- download updated state (D2H) -------------------------------------
    CUDA_CHECK(cudaMemcpy(coords.data(),   d_coords, n_coords * sizeof(double),                  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accepted.data(), d_acc,    static_cast<std::size_t>(M) * sizeof(long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rng_ctr.data(),  d_ctr,    static_cast<std::size_t>(M) * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // --- free -------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_reps));
    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_ctr));
}
