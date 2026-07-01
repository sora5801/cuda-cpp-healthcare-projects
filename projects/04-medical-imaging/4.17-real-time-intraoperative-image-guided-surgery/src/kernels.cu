// ===========================================================================
// src/kernels.cu  --  GPU ICP: correspondence + fixed-point covariance reduce
// ---------------------------------------------------------------------------
// Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
//
// WHAT THIS FILE DOES
//   The GPU twin of icp_cpu(). Each ICP iteration runs ONE kernel that, per
//   MOVING point, (1) transforms it by the current guess, (2) finds its NEAREST
//   fixed point (brute force), and (3) atomicAdds its fixed-point covariance /
//   centroid contribution into 16 device accumulators. The host then reads those
//   back, runs the SAME 3x3 SVD as the CPU (icp.h solve_rigid), composes the
//   incremental transform, and loops. Because the reduction is INTEGER
//   fixed-point (adds commute), the GPU accumulators equal the CPU ones exactly
//   -> the recovered transform matches the CPU reference bit-for-bit.
//
//   Independent nearest-neighbour search + atomic reduction is the SAME pattern
//   as flagship 11.09 (k-means): docs/PATTERNS.md sections 1 and 2.
//
// READ THIS AFTER: kernels.cuh, icp.h, util/cuda_check.cuh, util/timer.cuh.
// Compare with reference_cpu.cpp (the serial twin).
// ===========================================================================
#include "kernels.cuh"
#include "reference_cpu.h"       // rms_error (shared host metric for history[])
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <vector>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide the global-memory latency of the nearest-
// neighbour scan, and it keeps many blocks resident for good occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// The current ICP transform lives in CONSTANT memory. It is read by EVERY
// thread (to move that thread's point) but never changes during a launch, which
// is precisely what constant memory's broadcast cache is optimized for -- one
// fetch is broadcast to all threads in a warp. We upload the fresh transform
// here before each iteration's launch. (Same "one shared read-only object"
// idea as the constant-memory query in flagship 1.12.)
// ---------------------------------------------------------------------------
__constant__ Rigid c_transform;

// ---------------------------------------------------------------------------
// Device-side signed 64-bit atomic add.
//   CUDA ships atomicAdd for `unsigned long long` but not for signed 64-bit.
//   Our covariance and centroid sums can be NEGATIVE (surgical coordinates
//   straddle the origin). Trick: two's-complement addition is BIT-IDENTICAL for
//   signed and unsigned integers, so we reinterpret the signed accumulator as
//   unsigned long long, add the (reinterpreted) signed increment, and the wrap-
//   around yields the correct signed running sum. This keeps the reduction
//   integer/commutative -> deterministic and CPU-matching.
// ---------------------------------------------------------------------------
__device__ inline void atomic_add_i64(long long* addr, long long val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

// ---------------------------------------------------------------------------
// correspond_accumulate_kernel: the parallel CORRESPOND + REDUCE step.
//   Launch config (set in icp_gpu):
//     grid  = ceil(np / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x*blockDim.x + threadIdx.x owns MOVING
//     point P[i]. Guard the ragged last block with `if (i >= np) return;`.
//   Memory: reads P[i] (global), scans all of Q (global, O(nq) per thread),
//     reads the transform from constant memory, and atomicAdds into the global
//     accumulators d_acc. No shared memory -- the win here is the massive thread
//     parallelism over the O(np*nq) correspondence search, not tiling.
//
//   d_acc layout (all long long, so the atomics commute):
//     [0..2]   sumP  : sum of transformed moving points        (POS_SCALE)
//     [3..5]   sumQ  : sum of matched fixed points             (POS_SCALE)
//     [6..14]  sumPQ : sum of outer products p*q^T, row-major  (COV_SCALE)
//     [15]     count : number of pairs (== np here; every point matches)
//   This is exactly AccumFixed's field order (see the pack/unpack in icp_gpu).
// ---------------------------------------------------------------------------
__global__ void correspond_accumulate_kernel(const Vec3* __restrict__ P, int np,
                                             const Vec3* __restrict__ Q, int nq,
                                             long long* __restrict__ d_acc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's moving pt
    if (i >= np) return;                                   // ragged-block guard

    // (1) Transform this moving point by the current ICP guess (constant memory).
    const Vec3 tp = rigid_apply(c_transform, P[i]);

    // (2) CORRESPOND: brute-force nearest fixed point (shared icp.h routine ->
    //     identical tie-breaking to the CPU, so both pick the same match).
    const int j = nearest_index(tp, Q, nq);
    const Vec3 q = Q[j];

    // (3) REDUCE: quantize this pair's contribution to fixed-point and atomicAdd
    //     it into the shared accumulators. We compute the fixed-point terms with
    //     the SAME icp.h helpers the CPU uses (to_fixed), so every integer added
    //     here equals the integer the CPU adds -- guaranteeing bit-identical sums.
    const double pv[3] = { (double)tp.x, (double)tp.y, (double)tp.z };
    const double qv[3] = { (double)q.x,  (double)q.y,  (double)q.z  };
    for (int r = 0; r < 3; ++r) {
        atomic_add_i64(&d_acc[r],     (long long)to_fixed(pv[r], POS_SCALE)); // sumP
        atomic_add_i64(&d_acc[3 + r], (long long)to_fixed(qv[r], POS_SCALE)); // sumQ
        for (int c = 0; c < 3; ++c)
            atomic_add_i64(&d_acc[6 + r * 3 + c],
                           (long long)to_fixed(pv[r] * qv[c], COV_SCALE));    // sumPQ
    }
    // Count this pair. atomicAdd on the count keeps the same commutative style
    // (the value is trivially np, but we tally it to mirror the CPU exactly).
    atomic_add_i64(&d_acc[15], 1LL);
}

// ---------------------------------------------------------------------------
// icp_gpu: the host driver. Uploads P and Q once, then loops `iters` times:
//   upload current transform to constant memory -> zero accumulators ->
//   launch correspond_accumulate_kernel -> copy the 16 accumulators back ->
//   unpack into an AccumFixed -> solve_rigid (host 3x3 SVD) -> compose ->
//   record RMS error. Returns the final transform.
// ---------------------------------------------------------------------------
Rigid icp_gpu(const std::vector<Vec3>& P, const std::vector<Vec3>& Q,
              int iters, std::vector<double>& history, float* kernel_ms) {
    const int np = static_cast<int>(P.size());
    const int nq = static_cast<int>(Q.size());
    history.clear();
    history.reserve(static_cast<std::size_t>(iters));

    // ---- Device buffers ----------------------------------------------------
    // P and Q live on the device for the whole run (uploaded once). d_acc holds
    // the 16 long long accumulators, re-zeroed each iteration.
    Vec3 *d_P = nullptr, *d_Q = nullptr;
    long long* d_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, P.size() * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_Q, Q.size() * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_acc, 16 * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_P, P.data(), P.size() * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(Vec3), cudaMemcpyHostToDevice));

    // Start from the SAME centroid pre-alignment as the CPU reference (icp.h),
    // so the two paths follow an identical iteration sequence. The prealign is a
    // tiny host-side mean over the clouds -- negligible next to the kernels.
    Rigid g = centroid_prealign(P.data(), np, Q.data(), nq);
    const int blocks = (np + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    long long h_acc[16];                              // host mirror of accumulators

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int it = 0; it < iters; ++it) {
        // Upload the fresh transform to constant memory (read by all threads).
        CUDA_CHECK(cudaMemcpyToSymbol(c_transform, &g, sizeof(Rigid)));

        // Zero the accumulators, then run the correspondence/reduction kernel.
        CUDA_CHECK(cudaMemset(d_acc, 0, 16 * sizeof(long long)));
        timer.start();
        correspond_accumulate_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_P, np, d_Q, nq, d_acc);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("correspond_accumulate_kernel");

        // Bring the 16 accumulators back and UNPACK into AccumFixed (same field
        // order as icp.h). The subsequent solve_rigid is identical to the CPU's.
        CUDA_CHECK(cudaMemcpy(h_acc, d_acc, 16 * sizeof(long long), cudaMemcpyDeviceToHost));
        AccumFixed acc;
        for (int r = 0; r < 3; ++r) { acc.sumP[r] = h_acc[r]; acc.sumQ[r] = h_acc[3 + r]; }
        for (int k = 0; k < 9; ++k) acc.sumPQ[k] = h_acc[6 + k];
        acc.count = h_acc[15];

        // ALIGN on the host (tiny 3x3 SVD) and compose onto the running estimate.
        const Rigid inc = solve_rigid(acc);
        g = rigid_compose(inc, g);

        // Record the alignment quality (host rms_error -- shared with the CPU).
        history.push_back(rms_error(P, Q, g));
    }
    *kernel_ms = total_ms;

    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_acc));
    return g;
}
