// ===========================================================================
// src/kernels.cu  --  Per-site Felsenstein likelihood kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.9 : Phylogenetic Likelihood / Tree Inference
//
// WHAT THIS FILE DOES
//   Implements the device kernel (site_lnL_kernel) that computes ONE alignment
//   site's log-likelihood per thread via Felsenstein's pruning recursion, and
//   the host glue (score_trees_gpu) that uploads data, loops over candidate
//   trees, launches the kernel, and reduces per-site log-likelihoods into each
//   tree's total. This is the GPU twin of score_trees_cpu(); main.cu runs both
//   and asserts the per-tree totals are identical.
//
//   The per-site MATH is the shared site_log_likelihood() in felsenstein.h --
//   the SAME function the CPU reference calls -- so the device and host do
//   bit-for-bit identical arithmetic. The only GPU-specific machinery here is:
//     * the tree in CONSTANT memory (broadcast to every thread), and
//     * a deterministic FIXED-POINT integer atomicAdd reduction over sites.
//
// READ THIS AFTER: kernels.cuh, felsenstein.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#include "kernels.cuh"
#include "felsenstein.h"         // site_log_likelihood, PhyloNode, to_fixed, ...
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide global-memory latency, and small enough
// that the per-thread CLV scratch (see below) does not blow the occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// THE TREE IN CONSTANT MEMORY.
//   Every thread (every site) walks the SAME tree, reading each node's children
//   and branch lengths but never writing them. That is the textbook case for
//   __constant__ memory: its hardware cache broadcasts one address to a whole
//   warp in a single transaction, instead of each thread issuing its own global
//   load. We re-upload this array (cudaMemcpyToSymbol) once per candidate tree.
//   Size: MAX_INTERNAL_NODES * sizeof(PhyloNode) ~= 2 KB, well within 64 KB.
// ---------------------------------------------------------------------------
__constant__ PhyloNode c_nodes[MAX_INTERNAL_NODES];

// ---------------------------------------------------------------------------
// site_lnL_kernel: one thread computes ONE alignment site's log-likelihood and
// atomically adds its FIXED-POINT value into a shared 64-bit accumulator.
//
//   Launch config (set in score_trees_gpu):
//     grid  = ceil(n_sites / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: site index j = blockIdx.x * blockDim.x + threadIdx.x.
//
//   Parameters:
//     align_cm   : [n_sites * n_taxa] column-major alignment (site j's states at
//                  align_cm[j*n_taxa .. j*n_taxa + n_taxa-1]).
//     n_sites    : number of alignment columns (== number of logical threads).
//     n_taxa     : leaves per tree (also the CLV index offset for internals).
//     n_internal : internal nodes (== n_taxa-1); the tree is in c_nodes[].
//     kappa      : K2P transition/transversion ratio.
//     clv_scratch: [n_sites * (n_taxa+n_internal) * NSTATES] doubles. Thread j
//                  uses the slice starting at j*clv_stride so threads never
//                  collide. We keep CLVs in GLOBAL scratch (not registers/local)
//                  because the buffer is dynamically sized by n_taxa -- a clean,
//                  always-correct teaching choice; THEORY discusses the faster
//                  register/shared-memory variants for small fixed n_taxa.
//     clv_stride : (n_taxa + n_internal) * NSTATES, the per-thread slice length.
//     fixed_acc  : pointer to ONE long long; the per-site fixed-point lnL values
//                  are atomicAdd'd here. Integer adds commute -> deterministic.
//
//   Memory spaces touched: constant (c_nodes), global (align_cm, clv_scratch,
//   fixed_acc). Atomics: one atomicAdd(long long) per thread into fixed_acc.
// ---------------------------------------------------------------------------
__global__ void site_lnL_kernel(const unsigned char* __restrict__ align_cm,
                                int n_sites, int n_taxa, int n_internal,
                                double kappa,
                                double* __restrict__ clv_scratch, int clv_stride,
                                long long* __restrict__ fixed_acc) {
    // This thread's alignment site. The grid is sized to cover all sites, so we
    // only need to guard the ragged last block (sites with j >= n_sites idle).
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_sites) return;

    // This site's column (one observed state per taxon) and this thread's private
    // CLV scratch slice. Both are simple base-pointer offsets -- no allocation.
    const unsigned char* column = align_cm + static_cast<std::size_t>(j) * n_taxa;
    double* clv = clv_scratch + static_cast<std::size_t>(j) * clv_stride;

    // The whole pruning recursion for this site, in the SHARED host/device
    // function -> identical math to the CPU reference. c_nodes is the tree in
    // constant memory; passing it as a const PhyloNode* lets the same function
    // signature serve both sides.
    const double site_lnL = site_log_likelihood(column, c_nodes, n_internal,
                                                n_taxa, kappa, clv);

    // Deterministic reduction: round to fixed-point, then integer atomicAdd.
    // A float atomicAdd here would sum in nondeterministic order and NOT match
    // the CPU; the integer path is reproducible and exact (PATTERNS.md sec 3).
    atomicAdd(reinterpret_cast<unsigned long long*>(fixed_acc),
              static_cast<unsigned long long>(to_fixed(site_lnL)));
}

// ---------------------------------------------------------------------------
// score_trees_gpu: host wrapper. Uploads the alignment + per-thread CLV scratch
// ONCE, then for each candidate tree uploads its nodes to constant memory, zeroes
// the fixed-point accumulator, launches one thread per site, and reads the total
// back. We time the kernels only (CUDA events), not the H2D/D2H copies.
// ---------------------------------------------------------------------------
void score_trees_gpu(const PhyloProblem& prob, std::vector<double>& tree_lnL,
                     float* kernel_ms) {
    const int n_taxa     = prob.align.n_taxa;
    const int n_sites    = prob.align.n_sites;
    const int n_internal = n_taxa - 1;
    const int clv_stride = (n_taxa + n_internal) * PHYLO_NSTATES;   // doubles/thread
    tree_lnL.assign(prob.trees.size(), 0.0);

    if (n_internal > MAX_INTERNAL_NODES) {
        // Defensive: the constant-memory tree array is fixed-size. Fail loudly
        // rather than silently truncate (CLAUDE.md "no black boxes").
        std::fprintf(stderr, "[score_trees_gpu] tree too large: %d internal nodes > "
                             "MAX_INTERNAL_NODES=%d (raise it in kernels.cuh)\n",
                     n_internal, MAX_INTERNAL_NODES);
        std::exit(EXIT_FAILURE);
    }

    // (a) Upload the column-major alignment once (shared by every tree).
    const std::size_t align_bytes = static_cast<std::size_t>(n_sites) * n_taxa;
    unsigned char* d_align = nullptr;
    CUDA_CHECK(cudaMalloc(&d_align, align_bytes));
    CUDA_CHECK(cudaMemcpy(d_align, prob.align.data.data(), align_bytes,
                          cudaMemcpyHostToDevice));

    // (b) Allocate the per-thread CLV scratch: one private slice per site so
    //     threads never overwrite each other's partial likelihoods.
    const std::size_t clv_count = static_cast<std::size_t>(n_sites) * clv_stride;
    double* d_clv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_clv, clv_count * sizeof(double)));

    // (c) One 64-bit fixed-point accumulator on the device.
    long long* d_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_acc, sizeof(long long)));

    const int blocks = (n_sites + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Time the SUM of all per-tree kernel launches (a single GpuTimer spanning
    // the loop would also time the small cudaMemset/MemcpyToSymbol in between, so
    // we accumulate each launch's own event-measured time instead).
    float total_ms = 0.0f;

    for (std::size_t t = 0; t < prob.trees.size(); ++t) {
        const CandidateTree& tree = prob.trees[t];

        // Upload THIS tree's nodes to constant memory (broadcast to all threads).
        CUDA_CHECK(cudaMemcpyToSymbol(c_nodes, tree.nodes.data(),
                                      tree.nodes.size() * sizeof(PhyloNode)));
        // Zero the accumulator before summing this tree's sites.
        CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(long long)));

        GpuTimer timer;
        timer.start();
        site_lnL_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_align, n_sites, n_taxa, n_internal, prob.kappa,
            d_clv, clv_stride, d_acc);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("site_lnL_kernel");

        // Read the fixed-point total back and convert to a floating lnL. This is
        // the SAME integer the CPU produced -> tree_lnL[t] matches exactly.
        long long fixed_total = 0;
        CUDA_CHECK(cudaMemcpy(&fixed_total, d_acc, sizeof(long long),
                              cudaMemcpyDeviceToHost));
        tree_lnL[t] = from_fixed(fixed_total);
    }

    *kernel_ms = total_ms;

    // Free everything (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_align));
    CUDA_CHECK(cudaFree(d_clv));
    CUDA_CHECK(cudaFree(d_acc));
}
