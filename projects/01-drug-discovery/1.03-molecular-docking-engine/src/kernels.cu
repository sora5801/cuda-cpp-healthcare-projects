// ===========================================================================
// src/kernels.cu  --  Pose-scoring kernel + deterministic min-reduction
// ---------------------------------------------------------------------------
// Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   Implements dock_kernel (one thread per pose, scores it with the SHARED
//   docking_core.h::score_pose, then reduces to the best pose) and dock_gpu (the
//   host glue: upload, launch, time, download). This is the GPU twin of dock_cpu
//   in reference_cpu.cpp -- both call the same score_pose(), so they agree to
//   ~1e-12 (../THEORY.md S"verify").
//
//   The interesting CUDA content here is the INDEX-CARRYING MIN-REDUCTION: we
//   need not just the minimum energy but WHICH pose achieved it, and we need the
//   answer to be DETERMINISTIC (byte-identical stdout every run, PATTERNS.md S3).
//   Floating-point atomicMin is both unavailable for double and order-sensitive,
//   so we pack (energy, pose-index) into ONE unsigned 64-bit key whose integer
//   ordering matches the docking preference, and reduce that with atomicMin.
//
// READ THIS AFTER: kernels.cuh (interface + the thread-mapping idea), then
//   docking_core.h (the score_pose math this kernel calls).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cfloat>                // FLT_MAX
#include <limits>

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89, and it gives the warp-shuffle reduction below a tidy 8 warps.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// 2*pi literal -- MUST match TWO_PI in reference_cpu.cpp so the device pose
// enumeration is byte-identical to the host's (the whole point of the parity
// design). A literal avoids M_PI portability issues on both compilers.
// ---------------------------------------------------------------------------
__device__ __constant__ double D_TWO_PI = 6.283185307179586476925286766559;

// ---------------------------------------------------------------------------
// d_unrank_pose: the DEVICE copy of reference_cpu.cpp::unrank_pose. It is a
//   deliberate, documented duplication (kept in sync by hand) rather than a
//   shared function, because unrank_pose lives in the pure-C++ reference_cpu.cpp
//   (host-only translation unit) and we must not pull CUDA into that file. The
//   bodies are identical line-for-line; if you change one, change the other.
//   Decodes a flat pose index p into a concrete Pose (see reference_cpu.h).
// ---------------------------------------------------------------------------
__device__ inline Pose d_unrank_pose(const SearchSpace& s, long long p) {
    const long long nt = s.n_trans;
    const long long nr = s.n_rot;
    const int it_x = static_cast<int>(p % nt); p /= nt;
    const int it_y = static_cast<int>(p % nt); p /= nt;
    const int it_z = static_cast<int>(p % nt); p /= nt;
    const int ir_x = static_cast<int>(p % nr); p /= nr;
    const int ir_y = static_cast<int>(p % nr); p /= nr;
    const int ir_z = static_cast<int>(p % nr); p /= nr;

    // translation sub-index -> Angstrom offset over [-R, +R] (midpoint 0 if n==1)
    double tox = 0.0, toy = 0.0, toz = 0.0;
    if (s.n_trans > 1) {
        const double step = (2.0 * s.trans_range) / (s.n_trans - 1);
        tox = -s.trans_range + it_x * step;
        toy = -s.trans_range + it_y * step;
        toz = -s.trans_range + it_z * step;
    }
    // rotation sub-index -> angle over [0, 2pi) (angle 0 if n==1)
    double ra = 0.0, rb = 0.0, rc = 0.0;
    if (s.n_rot > 1) {
        ra = (D_TWO_PI * ir_x) / s.n_rot;
        rb = (D_TWO_PI * ir_y) / s.n_rot;
        rc = (D_TWO_PI * ir_z) / s.n_rot;
    }
    Pose pose;
    pose.tx = s.tcx + tox; pose.ty = s.tcy + toy; pose.tz = s.tcz + toz;
    pose.a  = ra;          pose.b  = rb;          pose.c  = rc;
    return pose;
}

// ---------------------------------------------------------------------------
// pack_key: build an unsigned 64-bit key whose UNSIGNED-INTEGER ordering equals
//   the docking preference "lower energy wins; ties -> lower pose index".
//
//   High 32 bits  = an order-preserving image of the pose energy (as float):
//     The IEEE-754 float bit pattern is monotonic for non-negative values, but
//     for negatives the ordering reverses. The classic fix: if the sign bit is
//     set (negative), flip ALL bits; else flip only the sign bit. After this map,
//     smaller float -> smaller uint32, so atomicMin on the key minimises energy.
//   Low 32 bits   = the pose index (poses are < 2^32 in this teaching version).
//
//   Because the index sits in the LOW bits, two poses with the same energy key
//   compare by index -> the lower index wins, exactly like dock_cpu's strict-<
//   tie rule. Thus the GPU and CPU pick the identical winner, deterministically.
//
//   We key on the FLOAT image of the energy (not the full double) only for the
//   REDUCTION/tie-break; the host recomputes the winner's EXACT double energy
//   from its index (dock_gpu), so no precision is lost in the reported result.
// ---------------------------------------------------------------------------
__device__ inline unsigned long long pack_key(double energy, unsigned int idx) {
    float ef = static_cast<float>(energy);
    unsigned int u;
    memcpy(&u, &ef, sizeof(u));                       // reinterpret float bits
    // order-preserving transform (see above)
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | idx;
}

// ---------------------------------------------------------------------------
// dock_kernel: one thread per pose (grid-stride), reduce to the global best.
//   Thread (block, lane) starts at pose p = block*blockDim + thread and strides
//   by the total thread count, scoring each pose it owns and tracking its local
//   best key. The block then reduces:
//     (a) WARP reduction via __shfl_down_sync (no shared memory, no __syncthreads
//         within a warp) -- each warp collapses to its lane-0 minimum key;
//     (b) the per-warp minima go through shared memory and one final warp reduces
//         them to the block minimum;
//     (c) thread 0 folds the block minimum into the global result with a single
//         atomicMin on the packed u64. atomicMin is associative & commutative on
//         integers, so the final value is independent of block/warp ordering ->
//         fully deterministic (PATTERNS.md S3).
//   Memory: grid + ligand in global memory (read-only, __restrict__); a tiny
//   shared array holds one key per warp (<= 8 entries for 256 threads).
// ---------------------------------------------------------------------------
__global__ void dock_kernel(const double* __restrict__ d_grid, GridDims dims,
                            const double* __restrict__ d_lx,
                            const double* __restrict__ d_ly,
                            const double* __restrict__ d_lz,
                            const double* __restrict__ d_w, int n_atoms,
                            SearchSpace space, long long n_poses,
                            unsigned long long* d_best) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    // 0xFFFFFFFFFFFFFFFF is the "worst" key (largest energy image, largest idx):
    // any real pose key compares strictly less, so it is a safe reduction seed.
    unsigned long long local = 0xFFFFFFFFFFFFFFFFull;

    // --- each thread scores all the poses it owns, tracking its local-min key ---
    for (long long p = blockIdx.x * blockDim.x + threadIdx.x; p < n_poses; p += stride) {
        const Pose pose = d_unrank_pose(space, p);
        const double e = score_pose(d_grid, dims, d_lx, d_ly, d_lz, d_w, n_atoms, pose);
        const unsigned long long key = pack_key(e, static_cast<unsigned int>(p));
        if (key < local) local = key;                 // integer compare == energy/idx order
    }

    // --- (a) warp-level reduction: min the key across the 32 lanes ---
    // __shfl_down_sync lets lane L read lane (L+offset)'s value with no shared
    // memory; halving the offset each step folds 32 lanes into lane 0 in 5 steps.
    const unsigned full_mask = 0xFFFFFFFFu;           // all 32 lanes participate
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down_sync(full_mask, local, offset);
        if (other < local) local = other;
    }

    // --- (b) combine the per-warp minima through shared memory ---
    __shared__ unsigned long long warp_min[THREADS_PER_BLOCK / 32];
    const int lane = threadIdx.x & 31;                // lane within the warp
    const int warp = threadIdx.x >> 5;                // warp index within block
    if (lane == 0) warp_min[warp] = local;            // lane 0 holds the warp min
    __syncthreads();                                  // all warp minima are visible

    // The first warp reduces the (<=8) per-warp minima to the block minimum.
    if (warp == 0) {
        const int n_warps = blockDim.x / warpSize;
        unsigned long long v = (lane < n_warps) ? warp_min[lane]
                                                : 0xFFFFFFFFFFFFFFFFull;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(full_mask, v, offset);
            if (other < v) v = other;
        }
        // --- (c) one atomic per block folds into the global best ---
        if (lane == 0) atomicMin(d_best, v);
    }
}

// ---------------------------------------------------------------------------
// unpack_index: pull the pose index back out of a packed key (host side).
//   The index is the low 32 bits (see pack_key).
// ---------------------------------------------------------------------------
static inline long long unpack_index(unsigned long long key) {
    return static_cast<long long>(key & 0xFFFFFFFFull);
}

// ---------------------------------------------------------------------------
// dock_gpu: host wrapper. The canonical CUDA steps, plus the reduction seed and
//   the EXACT-energy recompute:
//     (1) upload grid + ligand (SoA) + the seed best-key;
//     (2) launch dock_kernel (timed with CUDA events -- kernel only);
//     (3) copy the packed best key back, decode the winning pose index;
//     (4) recompute that pose's EXACT double energy on the host via score_pose
//         (the kernel keyed on the float image only for the tie-break, so this
//          restores full precision and makes GPU==CPU verification meaningful);
//     (5) free device memory.
// ---------------------------------------------------------------------------
void dock_gpu(const DockingProblem& prob, double* out_energy,
              long long* out_index, float* kernel_ms) {
    const GridDims&    dims  = prob.dims;
    const Ligand&      lig   = prob.ligand;
    const SearchSpace& space = prob.space;
    const long long n_poses  = space.n_poses();
    const std::size_t grid_bytes = static_cast<std::size_t>(dims.count()) * sizeof(double);
    const std::size_t atom_bytes = static_cast<std::size_t>(lig.n_atoms) * sizeof(double);

    // (1a) device buffers (d_ = device pointer; dereferencing on host would crash)
    double *d_grid = nullptr, *d_lx = nullptr, *d_ly = nullptr, *d_lz = nullptr, *d_w = nullptr;
    unsigned long long* d_best = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_lx, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_ly, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_lz, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_w,  atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(unsigned long long)));

    // (1b) upload inputs H2D
    CUDA_CHECK(cudaMemcpy(d_grid, prob.grid.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lx, lig.x.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ly, lig.y.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lz, lig.z.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,  lig.weight.data(), atom_bytes, cudaMemcpyHostToDevice));
    // seed the global best with the "worst possible" key so any real pose beats it
    const unsigned long long seed = 0xFFFFFFFFFFFFFFFFull;
    CUDA_CHECK(cudaMemcpy(d_best, &seed, sizeof(seed), cudaMemcpyHostToDevice));

    // (2) launch: enough blocks to cover the poses, capped (grid-stride covers
    //     any remainder). Cap keeps the grid modest for the tiny teaching sample.
    long long want = (n_poses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks = (want > 4096) ? 4096 : (want < 1 ? 1 : static_cast<int>(want));
    GpuTimer timer;
    timer.start();
    dock_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_grid, dims, d_lx, d_ly, d_lz, d_w,
                                               lig.n_atoms, space, n_poses, d_best);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("dock_kernel");

    // (3) copy the packed best key back and decode the winning pose index
    unsigned long long best_key = 0;
    CUDA_CHECK(cudaMemcpy(&best_key, d_best, sizeof(best_key), cudaMemcpyDeviceToHost));
    const long long best_idx = unpack_index(best_key);

    // (4) recompute the winner's EXACT double energy on the host (full precision).
    //     unrank_pose + score_pose are the same routines the CPU reference uses,
    //     so this exactly matches dock_cpu's energy for the same index.
    const Pose best_pose = unrank_pose(space, best_idx);
    const double best_e = score_pose(prob.grid.data(), dims, lig.x.data(), lig.y.data(),
                                     lig.z.data(), lig.weight.data(), lig.n_atoms, best_pose);
    *out_index  = best_idx;
    *out_energy = best_e;

    // (5) free device memory (no GPU garbage collector exists)
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_lx));
    CUDA_CHECK(cudaFree(d_ly));
    CUDA_CHECK(cudaFree(d_lz));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_best));
}
