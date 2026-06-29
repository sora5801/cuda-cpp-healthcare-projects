// ===========================================================================
// src/kernels.cu  --  Ensemble Monte Carlo titration kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
//
// WHAT THIS FILE DOES
//   Implements the device kernel (titrate_kernel) -- one thread per Monte Carlo
//   chain -- and the host-side glue (titrate_gpu) that allocates the device
//   tally, launches the chains, times the kernel, and brings the integer counts
//   back. This is the GPU twin of titrate_cpu() in reference_cpu.cpp; both call
//   the SAME shared physics (cph_core.h run_chain), so their integer protonation
//   tallies must match exactly. main.cu runs both and verifies that.
//
//   The interesting CUDA content is NOT arithmetic-heavy kernels but the MAPPING:
//   an embarrassingly-parallel ensemble of independent stochastic chains, scored
//   with integer atomics so the parallel reduction is deterministic.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and cph_core.h.
// ===========================================================================
#include "kernels.cuh"
#include "cph_core.h"            // CphSystem, run_chain, rng_seed, chain_id
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, eight warps to hide latency, and many blocks resident for
// occupancy. Each thread's work (a whole MC chain) is heavy and register-bound,
// so the exact value matters little here -- the ensemble is far larger than the
// machine and we are compute/divergence-bound, not occupancy-bound.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// titrate_kernel: ONE THREAD RUNS ONE (pH, replica) CHAIN.
//
//   Launch config (set in titrate_gpu):
//     total chains  N = n_pH * replicas
//     grid          = a fixed number of blocks; a grid-stride loop lets that
//                     fixed grid cover any N (robust + simple).
//     block         = THREADS_PER_BLOCK threads.
//
//   Thread-to-data map: global id g = blockIdx.x*blockDim.x + threadIdx.x is
//   decoded into a (pH index k, replica r) pair by
//        k = g / replicas ;  r = g % replicas
//   so consecutive threads in a warp mostly share a pH (k) and differ by replica
//   -- they run structurally similar chains, which keeps warp divergence modest.
//
//   Memory spaces:
//     * `sys` is passed BY VALUE -> it lands in constant/parameter memory and is
//       broadcast to every thread (read-only, never changes during the launch).
//     * the per-chain state[] and chain_counts[] live in registers/local memory
//       (n_res <= 16), so the chain runs without touching global memory at all...
//     * ...until the very end, when the thread ATOMICALLY adds its integer counts
//       into the shared global tally d_prot. Integer atomics commute, so the sum
//       is order-independent and reproduces the CPU's exactly (PATTERNS.md §3).
//
//   Divergence note (the classic Monte-Carlo GPU lesson): different chains accept
//   different moves, so threads in a warp take different branches each step. We
//   accept that here for clarity; production CpHMD reduces it with replica
//   batching and sorting. See ../THEORY.md "GPU mapping".
// ---------------------------------------------------------------------------
__global__ void titrate_kernel(CphSystem sys, double pH_min, double pH_max,
                               int n_pH, int replicas, unsigned long long seed,
                               unsigned long long* __restrict__ d_prot) {
    const int n_res = sys.n_res;
    const long long N = static_cast<long long>(n_pH) * replicas;  // total chains
    const long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;           // grid-stride
    const long long start =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Per-chain protonation tally, register/local resident (tiny n_res).
    int chain_counts[CPH_MAX_RESIDUES];

    for (long long g = start; g < N; g += stride) {
        // Decode this chain's (pH index, replica) from its flat id.
        const int k = static_cast<int>(g / replicas);   // pH grid index
        const int r = static_cast<int>(g % replicas);   // replica index
        // The k-th pH on the linear grid (n_pH >= 2 guaranteed by the loader).
        const double pH = pH_min + (pH_max - pH_min) * k / (n_pH - 1);

        // Seed this chain identically to the CPU (same chain_id packing), so the
        // random decisions match move-for-move and the tally is bit-identical.
        Rng rng = rng_seed(seed, chain_id(k, r));
        for (int i = 0; i < n_res; ++i) chain_counts[i] = 0;

        // Run the SHARED Monte Carlo chain (same code path as the CPU).
        run_chain(sys, pH, rng, chain_counts);

        // Score: atomically fold this chain's integer counts into the per-pH
        // tally. Many replica-threads target the same (k,i) slot; integer adds
        // make the order irrelevant -> deterministic, CPU-matching result.
        for (int i = 0; i < n_res; ++i) {
            const unsigned long long add =
                static_cast<unsigned long long>(chain_counts[i]);
            atomicAdd(&d_prot[static_cast<size_t>(k) * n_res + i], add);
        }
    }
}

// ---------------------------------------------------------------------------
// titrate_gpu: host wrapper. The canonical CUDA steps, specialised for an
// ensemble that produces a small integer tally (no big input arrays to copy --
// the "input" is just the parameter struct passed by value):
//   (1) allocate + zero the device tally d_prot ([n_pH * n_res] uint64)
//   (2) launch one thread per chain (grid-stride), timed with CUDA events
//   (3) copy the integer tally device->host
//   (4) free device memory
// We time ONLY the kernel (step 2) so the reported figure is compute, not the
// trivial copy of a few hundred integers.
// ---------------------------------------------------------------------------
void titrate_gpu(const CphProblem& prob, CphResult& out, float* kernel_ms) {
    const int n_res = prob.sys.n_res;
    const size_t n_slots = static_cast<size_t>(prob.n_pH) * n_res;
    const size_t bytes = n_slots * sizeof(unsigned long long);

    // Size the host result and set the denominator (same for every chain).
    out.prot_count.assign(n_slots, 0ULL);
    out.tallied_per_pH = static_cast<uint64_t>(prob.replicas) *
                         (prob.sys.sweeps - prob.sys.burn_in);

    // (1) Device tally. d_ prefix = DEVICE pointer (deref on host would crash).
    unsigned long long* d_prot = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prot, bytes));        // can fail: out of device mem
    CUDA_CHECK(cudaMemset(d_prot, 0, bytes));      // start every counter at zero

    // (2) Launch. A fixed grid of `blocks` blocks; the kernel's grid-stride loop
    //     covers all n_pH*replicas chains regardless of how many that is. 1024
    //     blocks gives the GPU plenty of resident warps to hide the divergence.
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    titrate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        prob.sys, prob.pH_min, prob.pH_max, prob.n_pH, prob.replicas,
        static_cast<unsigned long long>(prob.seed), d_prot);
    *kernel_ms = timer.stop_ms();                  // GPU-measured kernel time
    CUDA_CHECK_LAST("titrate_kernel");             // catch launch + run errors

    // (3) Bring the integer tally back to the host result vector.
    CUDA_CHECK(cudaMemcpy(out.prot_count.data(), d_prot, bytes,
                          cudaMemcpyDeviceToHost));

    // (4) Always free (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_prot));
}
