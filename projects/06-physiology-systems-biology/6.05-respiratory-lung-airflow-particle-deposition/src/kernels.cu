// ===========================================================================
// src/kernels.cu  --  Lagrangian deposition kernel (one thread per particle)
// ---------------------------------------------------------------------------
// Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
//
// WHAT THIS FILE DOES
//   Implements the device kernel (deposition_kernel) and the host-side glue
//   (deposition_gpu) that allocates the GPU tally, launches the kernel, times
//   it, and brings the counts back. This is the GPU twin of the serial CPU
//   reference in reference_cpu.cpp: it runs the SAME particle histories (shared
//   lung_physics.h -> identical RNG + track_particle), just in parallel and with
//   atomicAdd instead of a plain ++. main.cu runs both and asserts the tallies
//   are IDENTICAL. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea),
//                  lung_physics.h (the shared per-particle physics).
// ===========================================================================
#include "kernels.cuh"
#include "lung_physics.h"        // Particle, Airway, Rng, track_particle
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: it is a multiple of
// the 32-lane warp, gives the scheduler 8 warps to hide latency, and leaves
// plenty of blocks resident for occupancy. Particle tracking is compute-bound
// (a short double-precision loop per generation), so this is a fine choice.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// deposition_kernel: grid-stride over particle histories. Each iteration tracks
// ONE aerosol particle through the airway tree and atomically bumps the counter
// for the generation it deposits in (or the "exhaled" bucket at index n_gen).
//
//   LAUNCH CONFIG (set in deposition_gpu):
//     grid  = 1024 blocks (a fixed, GPU-filling grid)
//     block = THREADS_PER_BLOCK = 256 threads
//   THREAD-TO-DATA MAP: the grid-stride loop lets a fixed grid of ~262k threads
//     cover ANY number of particles n. Thread with global id `start` handles
//     particles start, start+stride, start+2*stride, ... where stride = total
//     thread count. (This is preferred over "one thread == one particle" so the
//     same launch works for 1e3 or 1e9 particles.)
//
//   MEMORY / ATOMICS:
//     * No shared memory: the tally (n_gen+1 counters) is tiny but written by
//       many threads, so it lives in global memory and is updated with atomicAdd.
//     * INTEGER quanta => the atomic adds COMMUTE => the result is deterministic
//       and matches the CPU tally exactly (docs/PATTERNS.md section 3). A float
//       tally would depend on the (nondeterministic) atomic ordering.
//
//   DIVERGENCE NOTE: different particles deposit after different numbers of
//     generations, so threads in a warp finish their loops at different times --
//     the classic Lagrangian-tracking challenge. Production codes sort/compact
//     particles by state to reduce it; here we keep the straight loop for
//     clarity (THEORY.md "real world").
// ---------------------------------------------------------------------------
__global__ void deposition_kernel(lung::Particle p, lung::Airway aw,
                                  uint64_t n, uint64_t seed, int n_gen,
                                  unsigned long long* __restrict__ tally) {
    // Total number of threads in the grid == the grid-stride step.
    const uint64_t stride =
        static_cast<uint64_t>(blockDim.x) * gridDim.x;
    // This thread's first particle index.
    const uint64_t start =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (uint64_t i = start; i < n; i += stride) {
        lung::Rng rng = lung::rng_seed(seed, i);      // this particle's stream
        const int g = lung::track_particle(p, aw, rng);  // deposited-in generation
        // g is in [0, n_gen]; index n_gen is the exhaled bucket. Many threads may
        // target the same bucket, so the increment MUST be atomic.
        atomicAdd(&tally[g], 1ULL);
    }
}

// ---------------------------------------------------------------------------
// deposition_gpu: host wrapper. The canonical CUDA steps, minus input H2D copies
// (there is no per-particle input array -- particles are generated on the fly
// from their index via the shared RNG, which is itself a nice lesson: we move
// almost no data across PCIe, only the tiny tally comes back).
//   (1) allocate + zero the device tally
//   (2) launch the kernel over all particles      <-- timed with CUDA events
//   (3) copy the tally device->host
//   (4) free device memory
// We time ONLY step (2) so the reported figure is the kernel cost.
// ---------------------------------------------------------------------------
void deposition_gpu(const DepositionProblem& prob, const lung::Airway& aw,
                    std::vector<uint64_t>& tally, float* kernel_ms) {
    const int n_slots = prob.n_gen + 1;              // n_gen generations + exhaled
    tally.assign(static_cast<std::size_t>(n_slots), 0ULL);

    // Particle properties are identical for every history (monodisperse aerosol),
    // so compute them once on the host and pass by value into the kernel.
    const lung::Particle p = lung::make_particle(prob.d_p, prob.rho_p);

    // (1) Device tally. atomicAdd on 64-bit counters needs unsigned long long.
    unsigned long long* d_tally = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n_slots) * sizeof(unsigned long long);
    CUDA_CHECK(cudaMalloc(&d_tally, bytes));         // can fail: out of device memory
    CUDA_CHECK(cudaMemset(d_tally, 0, bytes));       // start every counter at zero

    // (2) Launch. A fixed 1024-block grid gives the GPU plenty of resident warps;
    //     the grid-stride loop inside the kernel covers all prob.n_particles.
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    deposition_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        p, aw, prob.n_particles, prob.seed, prob.n_gen, d_tally);
    *kernel_ms = timer.stop_ms();                    // GPU-measured kernel time
    CUDA_CHECK_LAST("deposition_kernel");            // catch launch + exec errors

    // (3) Copy the small tally back into the host vector (element by element the
    //     two 64-bit integer types are bit-compatible, so a raw memcpy is fine).
    CUDA_CHECK(cudaMemcpy(tally.data(), d_tally, bytes, cudaMemcpyDeviceToHost));

    // (4) Free (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_tally));
}
