// ===========================================================================
// src/kernels.cu  --  The GPU pose-scoring kernel and its host wrapper
// ---------------------------------------------------------------------------
// Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
//
// WHAT THIS FILE DOES
//   Implements the device kernel (dock_kernel) and the host-side glue
//   (dock_gpu) that uploads the inputs, launches the kernel, times it, and
//   brings the per-pose scores back. This is the GPU twin of dock_cpu() in
//   reference_cpu.cpp; main.cu runs both and asserts they are EXACTLY equal
//   (integer scores from the shared docking_core.h -> no float tolerance).
//
//   The whole computation reuses the SHARED HD functions:
//     decode_pose()  (reference_cpu.h)  -- flat pose id -> (rotation, t)
//     score_pose()   (docking_core.h)   -- the O(Np*Nl) integer pair potential
//   so the kernel body is short: the physics is defined once and called here.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (same default as the other flagships; see THEORY "GPU mapping").
static constexpr int THREADS_PER_BLOCK = 256;

// Upper bound on the number of orientations we keep in constant memory. The
// cube group is 24; we size the array generously (still tiny: 64*9*4 = 2304 B,
// far inside the 64 KB constant bank) so a finer orientation set could slot in.
static constexpr int MAX_ROTS = 64;

// ---------------------------------------------------------------------------
// The rotation set in CONSTANT memory.
//   Every thread reads the SAME 9-int matrix for its pose's orientation and
//   NONE writes it -> constant memory is ideal: its hardware cache broadcasts
//   one address to a whole warp in a single transaction (exactly the trick used
//   for the query in 1.12). We copy the host rotation list here in dock_gpu().
// ---------------------------------------------------------------------------
__constant__ Rot3 c_rots[MAX_ROTS];

// ---------------------------------------------------------------------------
// dock_kernel: one thread scores one pose, via a grid-stride loop so a modest,
//   fixed-size grid covers an arbitrarily large pose space.
//
//   Launch config (set in dock_gpu):
//     grid  = min(ceil(n_poses / B), cap) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: the loop variable p IS the flat pose index; thread
//     (blockIdx.x, threadIdx.x) starts at p = block*blockDim + thread and
//     strides by the total thread count until p >= n_poses.
//
//   Memory: protein/ligand atoms from GLOBAL memory (read-only, reused across
//     poses -> cached in L2/registers); the rotation matrix from CONSTANT
//     memory (c_rots, broadcast); the score written once to GLOBAL out[p].
//   No shared memory or atomics: every pose's score is independent, written to
//     its own slot -> no contention, fully deterministic.
// ---------------------------------------------------------------------------
__global__ void dock_kernel(const Atom* __restrict__ protein, int Np,
                            const Atom* __restrict__ ligand,  int Nl,
                            PoseGrid grid, int n_rot, ScoreParams sp,
                            long long n_poses,
                            long long* __restrict__ out) {
    const long long stride = (long long)blockDim.x * gridDim.x;   // total threads
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < n_poses; p += stride) {
        // Decode this pose EXACTLY as the CPU does (same shared function).
        int32_t tx, ty, tz;
        const int r = decode_pose(p, grid, n_rot, tx, ty, tz);
        // Score it with the shared integer core. c_rots[r] is the broadcast
        // rotation matrix; the result is an exact int64, identical to the CPU.
        out[p] = score_pose(protein, Np, ligand, Nl,
                            c_rots[r], tx, ty, tz, sp);
    }
}

// ---------------------------------------------------------------------------
// dock_gpu: host wrapper. The canonical CUDA steps, with the small rotation set
//   going to constant memory instead of a global buffer. We time ONLY the
//   kernel (CUDA events), not the H2D/D2H copies (discussed separately in
//   THEORY "GPU mapping").
// ---------------------------------------------------------------------------
void dock_gpu(const DockingProblem& prob, std::vector<int64_t>& out,
              float* kernel_ms) {
    const int       Np = prob.Np();
    const int       Nl = prob.Nl();
    const int       nr = prob.n_rot();
    const long long N  = prob.n_poses();
    out.assign((std::size_t)N, 0);

    // Guard: the constant-memory rotation array is fixed-size.
    if (nr > MAX_ROTS) {
        std::fprintf(stderr, "[dock_gpu] too many rotations (%d > %d)\n", nr, MAX_ROTS);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t pro_bytes = (std::size_t)Np * sizeof(Atom);
    const std::size_t lig_bytes = (std::size_t)Nl * sizeof(Atom);
    const std::size_t out_bytes = (std::size_t)N  * sizeof(long long);

    // (a) Upload the rotation set to the __constant__ symbol (a special copy
    //     that targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_rots, prob.rots.data(), nr * sizeof(Rot3)));

    // (b) Allocate + upload the atoms, and allocate the per-pose score array.
    Atom*      d_pro = nullptr;   // [Np] protein atoms (device)
    Atom*      d_lig = nullptr;   // [Nl] ligand atoms (device)
    long long* d_out = nullptr;   // [N] int64 scores (device)
    CUDA_CHECK(cudaMalloc(&d_pro, pro_bytes));
    CUDA_CHECK(cudaMalloc(&d_lig, lig_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_pro, prob.protein.data(), pro_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lig, prob.ligand.data(),  lig_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover N one-thread-per-pose, but capped so
    //     the grid stays modest; the grid-stride loop handles any remainder.
    long long blocks_ll = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks = (blocks_ll > 1024) ? 1024 : (int)blocks_ll;
    if (blocks < 1) blocks = 1;
    GpuTimer timer;
    timer.start();
    dock_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_pro, Np, d_lig, Nl,
                                               prob.grid, nr, prob.params, N, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("dock_kernel");        // catch launch + execution errors

    // (d) Copy scores back, then (e) free device memory (no GPU GC exists).
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pro));
    CUDA_CHECK(cudaFree(d_lig));
    CUDA_CHECK(cudaFree(d_out));
}
