// ===========================================================================
// src/kernels.cu  --  GPU SART reconstruction (one thread per proton)
// ---------------------------------------------------------------------------
// Project 5.15 : Proton CT & Ion Imaging Reconstruction
//
// WHAT THIS FILE DOES
//   The GPU twin of reconstruct_cpu(). Two kernels + a host driver:
//     * tally_kernel  : one thread per proton -- forward-project along the MLP,
//                       form the WEPL residual, atomicAdd a fixed-point,
//                       length-weighted correction into per-voxel accumulators.
//     * update_kernel : one thread per voxel  -- rsp += relax * num/den.
//   The host driver reconstruct_gpu() loops these for `iters` SART sweeps.
//
//   PARITY: both kernels call the SAME shared physics (pct_physics.h:
//   mlp_point) and the SAME nearest-voxel binning (device_world_to_voxel below,
//   a mirror of world_to_voxel in reference_cpu.cpp), and accumulate in the SAME
//   fixed-point integers with the SAME rounding (llround of a double). So the
//   GPU reconstruction is BIT-IDENTICAL to the CPU one (docs/PATTERNS.md 2 & 3).
//
//   READ THIS AFTER: kernels.cuh (declarations + thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include "pct_physics.h"         // mlp_point, PctGeom, Proton (host+device)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide global-memory latency, plenty of resident
// blocks for occupancy. Used for both the proton grid and the voxel grid.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// device_world_to_voxel : EXACT mirror of world_to_voxel() in reference_cpu.cpp.
//   Nearest-voxel binning of a world point; returns the row-major index or -1.
//   Kept as a tiny __device__ function (not the HD header) because the CPU
//   version lives in reference_cpu.cpp; the logic is identical line-for-line so
//   both sides bin every MLP sample into the same voxel. (Refactoring both into
//   pct_physics.h is a fine exercise; duplicated here to keep that header free of
//   the PctGeom<->index coupling and easy to read.)
// ---------------------------------------------------------------------------
__device__ inline int device_world_to_voxel(const PctGeom& geom, float x, float y) {
    const float vs = geom.voxel_size();
    if (vs <= 0.0f) return -1;
    const int ix = static_cast<int>(floorf((x + geom.half) / vs + 0.5f));
    const int iy = static_cast<int>(floorf((y + geom.half) / vs + 0.5f));
    if (ix < 0 || ix >= geom.n || iy < 0 || iy >= geom.n) return -1;
    return iy * geom.n + ix;
}

// ---------------------------------------------------------------------------
// tally_kernel : thread i owns proton i. Two-pass over the MLP samples, exactly
//   as reconstruct_cpu does per proton:
//     pass 1 -- forward project (sum rsp*seg_len) and count in-grid hits;
//     pass 2 -- scatter length-weighted correction into fixed-point num/den.
//
//   Launch: grid = ceil(n_protons / 256), block = 256.
//   Thread-to-data: i = blockIdx.x*blockDim.x + threadIdx.x -> proton i.
//   Memory: reads protons[i] and rsp[] from global; writes via atomicAdd to the
//     shared int64 accumulators num_fx/den_fx (many protons hit the same voxel).
//   Atomics: atomicAdd on `long long` (unsigned 64-bit hardware add reinterpreted)
//     is supported on sm_60+. Integer adds COMMUTE, so the tally is deterministic
//     regardless of thread order -- the whole reason we use fixed point.
// ---------------------------------------------------------------------------
__global__ void tally_kernel(const Proton* __restrict__ protons, int n_protons,
                             const float* __restrict__ rsp, PctGeom geom,
                             int path_samples,
                             long long* __restrict__ num_fx,
                             long long* __restrict__ den_fx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_protons) return;                    // guard the ragged last block

    const Proton p = protons[i];                   // this thread's proton (to registers)

    // Segment length this proton assigns to each MLP sample.
    const float dx = p.x1 - p.x0, dy = p.y1 - p.y0;
    const float chord_len = sqrtf(dx * dx + dy * dy);
    const float seg_len = chord_len / path_samples;   // cm per sample

    // --- pass 1: forward project + count hits -----------------------------
    float est = 0.0f;
    int   n_hit = 0;
    for (int s = 0; s < path_samples; ++s) {
        const float t = (s + 0.5f) / path_samples;
        float px, py; mlp_point(p, t, &px, &py);      // shared MLP physics
        const int v = device_world_to_voxel(geom, px, py);
        if (v >= 0) { est += rsp[v] * seg_len; ++n_hit; }
    }
    if (n_hit == 0) return;                        // proton missed the grid

    // SART correction: residual / in-grid path length (L1 norm). Same math as
    // reference_cpu.cpp so CPU and GPU agree.
    const float resid = p.wepl - est;              // WEPL residual (cm)
    const float corr  = resid / (n_hit * seg_len); // RSP correction / cm

    // --- pass 2: scatter fixed-point correction ---------------------------
    for (int s = 0; s < path_samples; ++s) {
        const float t = (s + 0.5f) / path_samples;
        float px, py; mlp_point(p, t, &px, &py);
        const int v = device_world_to_voxel(geom, px, py);
        if (v < 0) continue;
        // Round-to-nearest in DOUBLE, matching the host's std::llround exactly.
        const long long num_add = llround(
            static_cast<double>(corr) * static_cast<double>(seg_len) * PCT_FIXED_SCALE);
        const long long den_add = llround(
            static_cast<double>(seg_len) * PCT_FIXED_SCALE);
        // atomicAdd wants (unsigned long long*), reinterpret the signed buffers;
        // two's-complement wrap makes signed/unsigned add bit-identical.
        atomicAdd(reinterpret_cast<unsigned long long*>(&num_fx[v]),
                  static_cast<unsigned long long>(num_add));
        atomicAdd(reinterpret_cast<unsigned long long*>(&den_fx[v]),
                  static_cast<unsigned long long>(den_add));
    }
}

// ---------------------------------------------------------------------------
// update_kernel : thread v owns voxel v. Applies the SART step, mirroring the
//   host update loop exactly (same double math, same relax).
//   Launch: grid = ceil(cells/256), block = 256.
// ---------------------------------------------------------------------------
__global__ void update_kernel(float* __restrict__ rsp, int cells, float relax,
                              const long long* __restrict__ num_fx,
                              const long long* __restrict__ den_fx) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= cells) return;
    if (den_fx[v] == 0) return;                    // untouched voxel this sweep
    const double num = static_cast<double>(num_fx[v]) / PCT_FIXED_SCALE;
    const double den = static_cast<double>(den_fx[v]) / PCT_FIXED_SCALE;
    rsp[v] += static_cast<float>(static_cast<double>(relax) * (num / den));
}

// ---------------------------------------------------------------------------
// reconstruct_gpu : host driver. Upload protons once, then loop the SART sweeps:
//   each sweep zeroes the accumulators, runs tally_kernel then update_kernel.
//   We time only the KERNELS (CUDA events), summing over sweeps -- transfers are
//   discussed separately in THEORY. The RSP image lives on the device across all
//   sweeps (only copied back at the end), so there is no per-sweep PCIe traffic.
// ---------------------------------------------------------------------------
void reconstruct_gpu(const PctProblem& prob, std::vector<float>& result,
                     float* kernel_ms) {
    const PctGeom geom = prob.geom;
    const int cells = geom.n * geom.n;
    const int n_protons = static_cast<int>(prob.protons.size());
    result.assign(static_cast<std::size_t>(cells), 0.0f);

    // --- device buffers ---------------------------------------------------
    Proton*    d_protons = nullptr;   // [n_protons] histories (uploaded once)
    float*     d_rsp     = nullptr;   // [cells] RSP image (lives on device)
    long long* d_num     = nullptr;   // [cells] fixed-point numerator accumulator
    long long* d_den     = nullptr;   // [cells] fixed-point denominator accumulator
    CUDA_CHECK(cudaMalloc(&d_protons, sizeof(Proton) * n_protons));
    CUDA_CHECK(cudaMalloc(&d_rsp,     sizeof(float)     * cells));
    CUDA_CHECK(cudaMalloc(&d_num,     sizeof(long long) * cells));
    CUDA_CHECK(cudaMalloc(&d_den,     sizeof(long long) * cells));

    CUDA_CHECK(cudaMemcpy(d_protons, prob.protons.data(),
                          sizeof(Proton) * n_protons, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rsp, 0, sizeof(float) * cells));   // start from vacuum

    const int proton_blocks = (n_protons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int voxel_blocks  = (cells     + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int it = 0; it < prob.iters; ++it) {
        // Zero the per-sweep accumulators (int64 all-bits-zero == integer 0).
        CUDA_CHECK(cudaMemset(d_num, 0, sizeof(long long) * cells));
        CUDA_CHECK(cudaMemset(d_den, 0, sizeof(long long) * cells));

        timer.start();
        // Tally: one thread per proton scatters its correction.
        tally_kernel<<<proton_blocks, THREADS_PER_BLOCK>>>(
            d_protons, n_protons, d_rsp, geom, prob.path_samples, d_num, d_den);
        // Update: one thread per voxel applies rsp += relax*num/den.
        update_kernel<<<voxel_blocks, THREADS_PER_BLOCK>>>(
            d_rsp, cells, prob.relax, d_num, d_den);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("SART sweep (tally_kernel + update_kernel)");
    }
    *kernel_ms = total_ms;

    // Bring the reconstructed RSP image back to the host.
    CUDA_CHECK(cudaMemcpy(result.data(), d_rsp, sizeof(float) * cells,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_protons));
    CUDA_CHECK(cudaFree(d_rsp));
    CUDA_CHECK(cudaFree(d_num));
    CUDA_CHECK(cudaFree(d_den));
}
