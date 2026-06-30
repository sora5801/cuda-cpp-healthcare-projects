// ===========================================================================
// src/kernels.cu  --  Pangenome 1-D SMACOF layout kernels + host loop
// ---------------------------------------------------------------------------
// Project 3.30 : Pangenome Graph Construction
//
// GPU twin of layout_cpu(). Per sweep we run two tiny kernels:
//   scatter_kernel : one thread per TERM -> compute the shared Guttman numerator
//                    (layout.h) and atomic-scatter the FIXED-POINT numerator (+
//                    weight) onto its two endpoint nodes (the scatter-reduction).
//   apply_kernel   : one thread per NODE -> x[k] = num[k]/den[k] (Guttman update).
// Because the accumulation is in integers (commutative), the GPU result is
// deterministic and equals the CPU bit-for-bit. main.cu compares them. See
// ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp, 8 warps to hide latency, a
// solid default across sm_75..sm_89. Tune per GPU; tiny inputs are launch-bound.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// SIGNED <-> UNSIGNED FIXED-POINT BRIDGE
//   CUDA provides atomicAdd for `unsigned long long int` but NOT for signed
//   `long long`. Our fixed-point quanta are signed (numerator contributions and
//   positions can be negative). The fix is two's-complement reinterpretation:
//   adding the unsigned bit patterns of two signed integers yields the unsigned
//   bit pattern of their signed sum (modular 2^64 arithmetic is identical for
//   signed and unsigned). So we cast the signed quantum to unsigned, atomicAdd it,
//   and cast the final accumulator back. static_cast to/from unsigned is
//   well-defined and lossless for this round-trip.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long ll_to_ull(long long v) {
    return static_cast<unsigned long long>(v);
}
__host__ __device__ __forceinline__ long long ull_to_ll(unsigned long long v) {
    return static_cast<long long>(v);
}

// ---------------------------------------------------------------------------
// scatter_kernel: one thread per term.
//   grid  = ceil(num_terms / THREADS_PER_BLOCK)
//   block = THREADS_PER_BLOCK
//   thread (blockIdx.x, threadIdx.x) -> term index `t`.
//   Memory: reads the two endpoint positions from global `x` (the sweep source);
//   performs FOUR atomicAdds into global accumulators (numerator + denominator,
//   for each endpoint). No shared memory: the collisions are spread across all
//   nodes, and integer atomics are cheap. The numerator math (LO_term_numerator)
//   is the SAME inline function the CPU calls -> identical results.
// ---------------------------------------------------------------------------
__global__ void scatter_kernel(const double* __restrict__ x,
                               const LayoutTerm* __restrict__ terms, int num_terms,
                               unsigned long long* __restrict__ num,
                               unsigned long long* __restrict__ den) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_terms) return;                 // guard the ragged last block

    const LayoutTerm term = terms[t];           // this thread's soft constraint
    const double xi = x[term.i];
    const double xj = x[term.j];

    // Guttman numerator contributions for each endpoint (shared with the CPU).
    const double ni = LO_term_numerator(xi, xj, term.target_d, term.weight);  // for node i
    const double nj = LO_term_numerator(xj, xi, term.target_d, term.weight);  // for node j

    // Quantise each contribution (mirrors the CPU's LO_to_fixed calls), then
    // atomic-add the unsigned bit pattern onto the two endpoints' accumulators.
    atomicAdd(&num[term.i], ll_to_ull(LO_to_fixed(ni)));
    atomicAdd(&num[term.j], ll_to_ull(LO_to_fixed(nj)));
    atomicAdd(&den[term.i], ll_to_ull(LO_to_fixed(term.weight)));
    atomicAdd(&den[term.j], ll_to_ull(LO_to_fixed(term.weight)));
}

// ---------------------------------------------------------------------------
// apply_kernel: one thread per node.
//   x[k] = numerator[k] / denominator[k] (the weighted average = Guttman update).
//   Fully independent across nodes -> no atomics. A node with no terms keeps its
//   position (denominator 0). This runs AFTER scatter_kernel (separate launch =
//   barrier), so reading the just-summed accumulators and overwriting x is a
//   correct Jacobi step.
// ---------------------------------------------------------------------------
__global__ void apply_kernel(double* __restrict__ x, int num_nodes,
                             const unsigned long long* __restrict__ num,
                             const unsigned long long* __restrict__ den) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_nodes) return;                 // guard the ragged last block
    const long long dfix = ull_to_ll(den[k]);
    if (dfix != 0)
        x[k] = LO_from_fixed(ull_to_ll(num[k])) / LO_from_fixed(dfix);
}

// ---------------------------------------------------------------------------
// layout_gpu: host wrapper. Allocate device buffers once, then loop the two
//   kernels for `iters` sweeps (zeroing the accumulators each sweep), copy the
//   final positions back, and compute the stress with the SHARED host helper so
//   CPU and GPU report the identical metric.
//
//   We time the whole sweep LOOP with CUDA events (a teaching artifact, not a
//   benchmark): tiny graphs are dominated by per-sweep launch overhead -- the
//   lesson of PATTERNS.md section 7 (many small launches are launch-bound; the
//   GPU's edge appears only at ODGI's real scale).
// ---------------------------------------------------------------------------
double layout_gpu(const LayoutProblem& p, std::vector<double>& x, float* kernel_ms) {
    const int N = static_cast<int>(p.init_x.size());
    const int T = static_cast<int>(p.terms.size());

    // ---- (1) Device buffers ------------------------------------------------
    double*             d_x     = nullptr;   // [N] node positions (source + result)
    LayoutTerm*         d_terms = nullptr;   // [T] constraints (read-only on device)
    unsigned long long* d_num   = nullptr;   // [N] fixed-point numerator accumulators
    unsigned long long* d_den   = nullptr;   // [N] fixed-point denominator accumulators
    CUDA_CHECK(cudaMalloc(&d_x,     static_cast<std::size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_terms, static_cast<std::size_t>(T) * sizeof(LayoutTerm)));
    CUDA_CHECK(cudaMalloc(&d_num,   static_cast<std::size_t>(N) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_den,   static_cast<std::size_t>(N) * sizeof(unsigned long long)));

    // ---- (2) Upload the initial positions and the (constant) term list -----
    CUDA_CHECK(cudaMemcpy(d_x, p.init_x.data(),
                          static_cast<std::size_t>(N) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_terms, p.terms.data(),
                          static_cast<std::size_t>(T) * sizeof(LayoutTerm), cudaMemcpyHostToDevice));

    const int term_blocks = (T + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;   // cover terms
    const int node_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;   // cover nodes

    // ---- (3) Sweep loop, timed with CUDA events ----------------------------
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < p.iters; ++it) {
        // Zero the accumulators for this sweep (all-bits-zero == integer 0).
        CUDA_CHECK(cudaMemset(d_num, 0, static_cast<std::size_t>(N) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_den, 0, static_cast<std::size_t>(N) * sizeof(unsigned long long)));
        // SCATTER then APPLY. They are separate launches because every node's
        // numerator/denominator must be fully accumulated (a global reduction)
        // before any node moves -- the launch boundary is the synchronization
        // point that also makes the in-place x update a correct Jacobi step.
        scatter_kernel<<<term_blocks, THREADS_PER_BLOCK>>>(d_x, d_terms, T, d_num, d_den);
        apply_kernel<<<node_blocks,  THREADS_PER_BLOCK>>>(d_x, N, d_num, d_den);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("layout kernels");          // catch any launch/execution error

    // ---- (4) Copy final positions back -------------------------------------
    x.assign(N, 0.0);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x,
                          static_cast<std::size_t>(N) * sizeof(double), cudaMemcpyDeviceToHost));

    // ---- (5) Free -----------------------------------------------------------
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_terms));
    CUDA_CHECK(cudaFree(d_num));
    CUDA_CHECK(cudaFree(d_den));

    return compute_stress(p, x);                // shared metric -> identical to CPU
}
