// ===========================================================================
// src/kernels.cu  --  Ensemble SSA kernel (one thread per trajectory)
// ---------------------------------------------------------------------------
// Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
//
// WHAT THIS FILE DOES
//   The GPU twin of simulate_cpu(). Each thread runs the SAME event-by-event
//   Gillespie loop (simulate_trajectory in ssa.h) for ONE trajectory, then
//   writes one TrajectoryResult. main.cu runs both sides and compares them.
//   Because the SSA core + RNG are shared and each trajectory is seeded from its
//   index, the GPU results are BIT-IDENTICAL to the CPU reference.
//
//   NO ATOMICS, NO SHARED MEMORY, NO SYNC: trajectories never touch each other's
//   data, so this is pure independent-work parallelism. The only per-thread cost
//   is the RNG state and the tiny fixed-size network state, all in registers /
//   local memory.
//
// READ THIS AFTER: kernels.cuh (declarations + mapping), ssa.h (the SSA core).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 balances occupancy against the fact that each thread
// here is HEAVY (a whole SSA run with a per-thread RNG and local arrays) rather
// than a one-line SAXPY: a smaller block keeps register pressure per SM in
// check while still giving the scheduler several warps to hide latency. 128 is a
// multiple of the 32-lane warp, so no lanes are wasted.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ssa_kernel: thread `idx` owns trajectory `idx`.
//   Launch config (set in simulate_gpu):
//     grid  = ceil(n_traj / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x.
//   Each thread reads the (by-value) network, runs the FULL SSA time loop, and
//   writes exactly out[idx]. Divergence is inherent to the algorithm: different
//   trajectories fire different numbers of events, so warps wait for their
//   slowest lane. That is expected for Monte-Carlo SSA and is the price of an
//   *exact* method; THEORY.md discusses tau-leaping as the load-balancing fix.
// ---------------------------------------------------------------------------
__global__ void ssa_kernel(ReactionNetwork net, int n_traj,
                           TrajectoryResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_traj) return;                 // guard the ragged last block

    // Run this trajectory. simulate_trajectory seeds its own RNG stream from
    // (net.base_seed, idx), so trajectory idx here matches trajectory idx on the
    // CPU exactly. The result is written once -- no read-modify-write, no race.
    out[idx] = simulate_trajectory(net, static_cast<uint64_t>(idx));
}

// ---------------------------------------------------------------------------
// simulate_gpu: host wrapper. The canonical CUDA steps, minus input copies
//   (the only "input" is the small POD network, passed by value in the launch):
//     (1) build the network + allocate the device result buffer
//     (2) launch one thread per trajectory
//     (3) copy results device->host
//     (4) free device memory
//   We time ONLY the kernel (step 2) with CUDA events; the D2H copy of the
//   results is reported separately in spirit but excluded from the kernel figure
//   so the number reflects compute, not PCIe (THEORY.md, "honest timing").
// ---------------------------------------------------------------------------
void simulate_gpu(const EnsembleConfig& c, std::vector<TrajectoryResult>& results,
                  float* kernel_ms) {
    const int n = c.n_traj;
    results.assign(static_cast<std::size_t>(n), TrajectoryResult{});

    // Build the SAME network the CPU builds (single source of truth).
    const ReactionNetwork net = build_gene_network(c);

    // (1) Device output buffer: one TrajectoryResult per trajectory.
    TrajectoryResult* d_out = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(TrajectoryResult);
    CUDA_CHECK(cudaMalloc(&d_out, bytes));     // can fail: out of device memory

    // (2) Launch. Blocks must cover all n trajectories -> ceiling division.
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ssa_kernel<<<blocks, THREADS_PER_BLOCK>>>(net, n, d_out);
    *kernel_ms = timer.stop_ms();              // GPU-measured kernel time
    CUDA_CHECK_LAST("ssa_kernel");             // catch launch + execution errors

    // (3) Bring the per-trajectory results back to the host.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
