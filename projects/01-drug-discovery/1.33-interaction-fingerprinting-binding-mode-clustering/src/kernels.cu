// ===========================================================================
// src/kernels.cu  --  IFP generation + consensus-bit Tanimoto k-means (GPU)
// ---------------------------------------------------------------------------
// Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
//
// This is the GPU twin of build_ifps() + ifp_cluster_cpu() in reference_cpu.cpp.
// It calls the SAME __host__ __device__ math in ifp.h, so its results are
// bit-identical to the CPU reference (main.cu asserts this). Two stages:
//
//   STAGE A  build_ifp_kernel  : one thread per pose -> that pose's IFP bits.
//   STAGE B  assign_kernel     : one thread per pose -> nearest centroid index.
//            tally_kernel       : one thread per pose -> atomicAdd integer per-bit
//                                 counters (integer -> order-independent -> exact).
//            (the majority-vote UPDATE runs on the host: update_centroids().)
//
// See ../THEORY.md "GPU mapping" for the launch-config and memory reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "ifp.h"                  // Residue, Pose, ifp_* device-callable math
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89. Our grids are 1-D over "poses", so one constant suffices.
static constexpr int THREADS_PER_BLOCK = 256;

// ===========================================================================
// STAGE A -- IFP GENERATION
// ===========================================================================
// build_ifp_kernel: thread p owns pose p. It scans all NUM_RESIDUES residues,
//   ORing each residue's interaction nibble into pose p's fingerprint row.
//   * Mapping: p = blockIdx.x*blockDim.x + threadIdx.x  ->  pose index.
//   * Memory : residues[] and poses[] in global memory (read-only here); each
//     thread writes ONLY its own FP_WORDS-word output row -> no races, no atomics.
//   * The residue loop is tiny and identical for every thread, so warps stay
//     converged (no divergence beyond the per-bit `if`s inside the shared
//     ifp_residue_nibble()).
//   Because it calls the same ifp_residue_nibble() as the CPU, the bits match.
// ---------------------------------------------------------------------------
__global__ void build_ifp_kernel(const Residue* __restrict__ residues,
                                 const Pose* __restrict__ poses, int P,
                                 uint64_t* __restrict__ fps) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;                       // guard the ragged last block

    // Accumulate into registers first, then write the whole row once.
    uint64_t row[FP_WORDS];
    #pragma unroll
    for (int w = 0; w < FP_WORDS; ++w) row[w] = 0ull;

    const Pose pose = poses[p];               // this thread's pose (to registers)
    for (int r = 0; r < NUM_RESIDUES; ++r) {
        const int nibble = ifp_residue_nibble(pose, residues[r]);   // shared math
        #pragma unroll
        for (int t = 0; t < NUM_ITYPES; ++t) {
            if (nibble & (1 << t)) {
                const int idx = r * NUM_ITYPES + t;        // global bit index
                row[idx >> 6] |= (1ull << (idx & 63));     // set bit idx
            }
        }
    }

    uint64_t* out = fps + static_cast<std::size_t>(p) * FP_WORDS;
    #pragma unroll
    for (int w = 0; w < FP_WORDS; ++w) out[w] = row[w];   // single write of the row
}

// build_ifps_gpu: host wrapper for STAGE A (the canonical 5 CUDA steps).
void build_ifps_gpu(const Dataset& d, std::vector<uint64_t>& fps, float* kernel_ms) {
    const int P = d.P;
    fps.assign(static_cast<std::size_t>(P) * FP_WORDS, 0ull);

    // (a) Allocate device buffers for residues, poses, and the output IFPs.
    Residue* d_res = nullptr; Pose* d_pose = nullptr; uint64_t* d_fps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res,  NUM_RESIDUES * sizeof(Residue)));
    CUDA_CHECK(cudaMalloc(&d_pose, static_cast<std::size_t>(P) * sizeof(Pose)));
    CUDA_CHECK(cudaMalloc(&d_fps,  static_cast<std::size_t>(P) * FP_WORDS * sizeof(uint64_t)));

    // (b) Upload inputs H2D.
    CUDA_CHECK(cudaMemcpy(d_res, d.residues.data(), NUM_RESIDUES * sizeof(Residue),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pose, d.poses.data(),
                          static_cast<std::size_t>(P) * sizeof(Pose), cudaMemcpyHostToDevice));

    // (c) Launch one thread per pose; time only the kernel (CUDA events).
    const int grid = (P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer; timer.start();
    build_ifp_kernel<<<grid, THREADS_PER_BLOCK>>>(d_res, d_pose, P, d_fps);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("build_ifp_kernel");

    // (d) Copy fingerprints D2H, then (e) free.
    CUDA_CHECK(cudaMemcpy(fps.data(), d_fps,
                          static_cast<std::size_t>(P) * FP_WORDS * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_pose));
    CUDA_CHECK(cudaFree(d_fps));
}

// ===========================================================================
// STAGE B -- CONSENSUS-BIT TANIMOTO k-MEANS
// ===========================================================================
// assign_kernel: thread i owns pose i; writes the index of its nearest centroid.
//   Reads the K centroids from global memory (small: K*FP_WORDS words). Pure
//   read + popcount, no writes besides its own label -> no atomics, no races.
// ---------------------------------------------------------------------------
__global__ void assign_kernel(const uint64_t* __restrict__ fps, int P,
                              const uint64_t* __restrict__ centroids, int K,
                              int* __restrict__ labels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    labels[i] = ifp_nearest_centroid(
        fps + static_cast<std::size_t>(i) * FP_WORDS, centroids, K);
}

// tally_kernel: thread i scatters pose i's set bits into its cluster's integer
//   per-bit counters via atomicAdd, and bumps the cluster size.
//   * WHY ATOMICS: many poses share a cluster, so their adds collide. We add 1u
//     (an INTEGER), and integer addition is associative AND commutative, so the
//     final counts are independent of thread order -> deterministic AND equal to
//     the CPU's serial tally. (A float atomicAdd would NOT be reproducible --
//     this is the determinism lesson from flagship 11.09 / PATTERNS.md sec 3.)
//   * bit_counts is [K*IFP_BITS] unsigned ints; counts is [K] unsigned ints.
// ---------------------------------------------------------------------------
__global__ void tally_kernel(const uint64_t* __restrict__ fps, int P,
                             const int* __restrict__ labels,
                             unsigned int* __restrict__ bit_counts,
                             unsigned int* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    const int k = labels[i];
    atomicAdd(&counts[k], 1u);                 // this cluster gained a member
    const uint64_t* fp = fps + static_cast<std::size_t>(i) * FP_WORDS;
    for (int b = 0; b < IFP_BITS; ++b)         // scatter set bits into counters
        if (fp[b >> 6] & (1ull << (b & 63)))
            atomicAdd(&bit_counts[static_cast<std::size_t>(k) * IFP_BITS + b], 1u);
}

// ifp_cluster_gpu: STAGE B host wrapper. ASSIGN + TALLY on the GPU each iter;
//   majority-vote UPDATE on the host (update_centroids) -- identical to the CPU
//   reference, so labels + centroids match exactly. Time only the two kernels.
double ifp_cluster_gpu(const std::vector<uint64_t>& fps, int P, int K, int iters,
                       std::vector<uint64_t>& centroids, std::vector<int>& labels,
                       std::vector<unsigned int>& sizes, float* kernel_ms) {
    init_centroids(fps, P, K, centroids);       // host: same deterministic seeding
    labels.assign(P, 0);
    sizes.assign(K, 0);

    // Device buffers.
    uint64_t* d_fps = nullptr; uint64_t* d_cent = nullptr; int* d_labels = nullptr;
    unsigned int* d_bitcounts = nullptr; unsigned int* d_counts = nullptr;
    const std::size_t fps_words  = static_cast<std::size_t>(P) * FP_WORDS;
    const std::size_t cent_words = static_cast<std::size_t>(K) * FP_WORDS;
    const std::size_t bc_words   = static_cast<std::size_t>(K) * IFP_BITS;
    CUDA_CHECK(cudaMalloc(&d_fps,       fps_words  * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cent,      cent_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_labels,    static_cast<std::size_t>(P) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bitcounts, bc_words   * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_counts,    static_cast<std::size_t>(K) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_fps, fps.data(), fps_words * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned int> bit_counts(bc_words);   // host staging for UPDATE
    const int grid = (P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer; timer.start();
    for (int it = 0; it < iters; ++it) {
        // Upload the current centroids, then ASSIGN.
        CUDA_CHECK(cudaMemcpy(d_cent, centroids.data(), cent_words * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
        assign_kernel<<<grid, THREADS_PER_BLOCK>>>(d_fps, P, d_cent, K, d_labels);

        // TALLY: zero the integer counters, then atomic-accumulate.
        CUDA_CHECK(cudaMemset(d_bitcounts, 0, bc_words * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_counts, 0, static_cast<std::size_t>(K) * sizeof(unsigned int)));
        tally_kernel<<<grid, THREADS_PER_BLOCK>>>(d_fps, P, d_labels, d_bitcounts, d_counts);

        // Bring the integer tallies back; UPDATE centroids on the host (shared code).
        CUDA_CHECK(cudaMemcpy(bit_counts.data(), d_bitcounts,
                              bc_words * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_counts,
                              static_cast<std::size_t>(K) * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        update_centroids(K, bit_counts, sizes, centroids);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("assign/tally kernels");

    // Final labels back to the host for verification + reporting.
    CUDA_CHECK(cudaMemcpy(labels.data(), d_labels,
                          static_cast<std::size_t>(P) * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_fps));
    CUDA_CHECK(cudaFree(d_cent));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_bitcounts));
    CUDA_CHECK(cudaFree(d_counts));
    return cluster_cost(fps, P, centroids, labels);
}
