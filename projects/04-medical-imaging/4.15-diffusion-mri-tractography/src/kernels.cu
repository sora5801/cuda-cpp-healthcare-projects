// ===========================================================================
// src/kernels.cu  --  DTI fit kernel + tractography kernel + host wrappers
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// These are the GPU twins of fit_all_voxels_cpu() and trace_streamlines_cpu()
// in reference_cpu.cpp. Because the per-voxel physics (dti_core.h) and per-step
// tractography (tract_core.h) are shared __host__ __device__ code, the GPU and
// CPU produce matching results; main.cu runs both and asserts they agree.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "dti_core.h"            // fit_voxel, VoxelResult, NPARAM, NMEAS (shared)
#include "tract_core.h"          // sample_dir, nearest_dir (shared stepping)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// The OLS pseudo-inverse Minv in CONSTANT memory.
//   * Every voxel-thread multiplies its log-signal vector by the SAME
//     NPARAM x NMEAS matrix and never writes it -> constant memory is ideal: its
//     hardware cache broadcasts one address to a whole warp in a single
//     transaction, instead of NPARAM*NMEAS global loads per thread.
//   * Fixed size (7*13*8 = 728 bytes) sits comfortably in the 64 KB constant
//     bank. Filled by cudaMemcpyToSymbol() in fit_gpu().
// ---------------------------------------------------------------------------
__constant__ double c_Minv[NPARAM * NMEAS];

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (see THEORY "GPU mapping" for the occupancy reasoning).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// fit_kernel: one logical thread per voxel, via a grid-stride loop so a fixed
// grid still covers an arbitrarily large volume.
//   Thread (blockIdx.x, threadIdx.x) handles voxel v = block*blockDim + thread,
//   striding by the total thread count until v >= nvox.
//   Memory: c_Minv from the constant cache; the NMEAS signals of voxel v from
//   global memory; writes one VoxelResult. No shared memory / atomics -- voxels
//   are independent (the whole reason DTI fitting is embarrassingly parallel).
// ---------------------------------------------------------------------------
__global__ void fit_kernel(const double* __restrict__ signal, int nvox,
                           VoxelResult* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nvox; v += stride) {
        const double* sig = signal + static_cast<std::size_t>(v) * NMEAS;   // this voxel's row
        // The ENTIRE per-voxel computation is the shared fit_voxel(): log +
        // fixed matvec (using c_Minv) + analytic 3x3 eigen-decomposition + FA/MD.
        // Identical bits to the CPU reference because it is the same source.
        out[v] = fit_voxel(sig, c_Minv);
    }
}

// ---------------------------------------------------------------------------
// fit_gpu: host wrapper. Uploads Minv to constant memory + signals to global
// memory, launches fit_kernel, times ONLY the kernel (CUDA events), copies the
// VoxelResults back. The five canonical CUDA steps with the operator going to
// constant memory instead of a global buffer.
// ---------------------------------------------------------------------------
void fit_gpu(const DwiVolume& vol, const std::vector<double>& Minv,
             std::vector<VoxelResult>& out, float* kernel_ms) {
    const int nvox = vol.nvox;
    out.resize(nvox);
    const std::size_t sig_bytes = static_cast<std::size_t>(nvox) * NMEAS * sizeof(double);
    const std::size_t out_bytes = static_cast<std::size_t>(nvox) * sizeof(VoxelResult);

    // (a) Upload Minv into the __constant__ symbol (a special copy targeting the
    //     constant bank, not ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_Minv, Minv.data(), NPARAM * NMEAS * sizeof(double)));

    // (b) Allocate + upload the signals, allocate the results.
    double*      d_sig = nullptr;   // [nvox*NMEAS] device, row-major per voxel
    VoxelResult* d_out = nullptr;   // [nvox] device results
    CUDA_CHECK(cudaMalloc(&d_sig, sig_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_sig, vol.signal.data(), sig_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover nvox one-thread-per-voxel, capped so the
    //     grid stays modest; the grid-stride loop handles any remainder.
    int blocks = (nvox + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;
    GpuTimer timer;
    timer.start();
    fit_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_sig, nvox, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("fit_kernel");

    // (d) Copy results back, then (e) free.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sig));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// tract_step_dir: one directional trace (sign = +1 forward or -1 backward),
// writing points into `pts` starting at index `*np`, up to `cap` total points.
// This is the device version of trace_one() in reference_cpu.cpp -- it uses the
// SAME shared stepping (sample_dir/nearest_dir from tract_core.h), so the GPU and
// CPU streamlines match. Returns the number of points written by THIS call.
//   `dir_index(k)` maps the k-th recorded point of a directional trace to a slot
//   index; passing an ascending map traces forward, a descending map traces the
//   backward half straight into REVERSED positions (so no in-place reversal is
//   ever needed -- the bug that a naive reverse-in-place would introduce). `cap`
//   guards the fixed-size slot. Returns the number of points written.
// ---------------------------------------------------------------------------
__device__ inline int tract_step_dir(const VoxelResult* fit, int nx, int ny, int nz,
                                     double sx, double sy, double sz, double sign,
                                     int max_steps, float step, float fa_min, float cos_min,
                                     float* pts, int base, int istep, int cap) {
    double px = sx, py = sy, pz = sz;
    double rdx, rdy, rdz;
    nearest_dir(fit, nx, ny, nz, px, py, pz, rdx, rdy, rdz);   // seed reference dir
    rdx *= sign; rdy *= sign; rdz *= sign;

    int written = 0;
    for (int s = 0; s < max_steps; ++s) {
        if (px < 0 || py < 0 || pz < 0 || px > nx - 1 || py > ny - 1 || pz > nz - 1) break;
        double dx, dy, dz, fa;
        sample_dir(fit, nx, ny, nz, px, py, pz, rdx, rdy, rdz, dx, dy, dz, fa);
        if (fa < fa_min) break;
        const double turn = dx*rdx + dy*rdy + dz*rdz;
        if (turn < cos_min) break;
        const int idx = base + istep * written;   // ascending or descending target
        if (idx < 0 || idx >= cap) break;          // slot full: stop (fixed-size output)
        pts[3*idx+0] = (float)px; pts[3*idx+1] = (float)py; pts[3*idx+2] = (float)pz;
        ++written;
        px += step * dx; py += step * dy; pz += step * dz;
        rdx = dx; rdy = dy; rdz = dz;
    }
    return written;
}

// ---------------------------------------------------------------------------
// tract_kernel: one thread per seed. A fiber has no intrinsic direction, so we
// trace BOTH ways from the seed and stitch the halves into one polyline inside a
// fixed-size slot (cap = 2*max_steps points), then report the compact run
// [start, start+count) so no dynamic device allocation is ever needed.
//
//   LAYOUT (the trick that avoids an in-place reversal, which would be a real bug
//   -- writing a reversed copy over its own unread source corrupts the tail):
//     * The BACKWARD trace writes its k-th point at index (max_steps-1 - k): it
//       fills DOWNWARD from max_steps-1, so the backward points land already
//       ordered "far end -> seed". No reversal step is ever performed.
//     * The FORWARD trace writes its k-th point at index (max_steps-1 + k): it
//       fills UPWARD from max_steps-1. Its 0th point (the seed) lands at exactly
//       index max_steps-1 -- the SAME slot the backward seed used, and the SAME
//       coordinate -- so it harmlessly overwrites the duplicate seed. This is how
//       we "drop the backward half's duplicate seed" to match the CPU reference.
//   Result: the stitched polyline is the CONTIGUOUS run
//       start = max_steps - nback   (backward far-end),
//       count = (nback - 1) + nfwd  (backward minus its seed, plus all forward),
//   with backward points nback-1..1 in [max_steps-nback, max_steps-2] and forward
//   points 0..nfwd-1 in [max_steps-1, max_steps-1+nfwd). This reproduces
//   reference_cpu.cpp::trace_streamlines_cpu vertex-for-vertex.
//   Thread s owns slot s (pts + s*cap*3); fully independent -> no atomics/shared.
// ---------------------------------------------------------------------------
__global__ void tract_kernel(const VoxelResult* __restrict__ fit,
                             int nx, int ny, int nz,
                             const float* __restrict__ seeds, int nseeds,
                             int cap, int max_steps,
                             float step, float fa_min, float cos_min,
                             float* __restrict__ pts, int* __restrict__ starts,
                             int* __restrict__ lengths) {
    const int stride = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < nseeds; s += stride) {
        float* slot = pts + static_cast<std::size_t>(s) * cap * 3;  // this seed's output
        const double sx = seeds[3*s+0], sy = seeds[3*s+1], sz = seeds[3*s+2];

        // Backward half fills downward from max_steps-1 (seed at max_steps-1).
        int nback = tract_step_dir(fit, nx, ny, nz, sx, sy, sz, -1.0,
                                   max_steps, step, fa_min, cos_min,
                                   slot, /*base=*/max_steps - 1, /*istep=*/-1, cap);
        // Forward half fills upward from max_steps-1; its seed overwrites the
        // backward seed at index max_steps-1 (identical coordinate).
        int nfwd  = tract_step_dir(fit, nx, ny, nz, sx, sy, sz, +1.0,
                                   max_steps, step, fa_min, cos_min,
                                   slot, /*base=*/max_steps - 1, /*istep=*/+1, cap);
        // Compact run: drop the backward half's duplicate seed (keep nback-1 of it).
        int back_used = (nback > 0) ? (nback - 1) : 0;
        starts[s]  = (max_steps - 1) - back_used;   // first occupied index
        lengths[s] = back_used + nfwd;              // contiguous point count
    }
}

// ---------------------------------------------------------------------------
// tract_gpu: host wrapper for the tractography kernel. Uploads the fit + seeds,
// launches tract_kernel, times it, and unpacks the fixed-size output into one
// Streamline per seed.
// ---------------------------------------------------------------------------
void tract_gpu(const std::vector<VoxelResult>& fit, const DwiVolume& vol,
               const std::vector<float>& seeds,
               int max_steps, float step, float fa_min, float cos_min,
               std::vector<Streamline>& lines, float* kernel_ms) {
    const int nseeds = static_cast<int>(seeds.size() / 3);
    // Per-seed slot capacity: forward + backward traces both bounded by max_steps,
    // so 2*max_steps points is a safe upper bound.
    const int cap = 2 * max_steps;
    lines.assign(nseeds, Streamline{});

    const std::size_t fit_bytes  = fit.size() * sizeof(VoxelResult);
    const std::size_t seed_bytes = seeds.size() * sizeof(float);
    const std::size_t pts_bytes  = static_cast<std::size_t>(nseeds) * cap * 3 * sizeof(float);
    const std::size_t idx_bytes  = static_cast<std::size_t>(nseeds) * sizeof(int);

    VoxelResult* d_fit = nullptr;
    float* d_seeds = nullptr;
    float* d_pts = nullptr;
    int*   d_start = nullptr;   // first occupied slot index per seed
    int*   d_len = nullptr;     // contiguous point count per seed
    CUDA_CHECK(cudaMalloc(&d_fit, fit_bytes));
    CUDA_CHECK(cudaMalloc(&d_seeds, seed_bytes));
    CUDA_CHECK(cudaMalloc(&d_pts, pts_bytes));
    CUDA_CHECK(cudaMalloc(&d_start, idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_len, idx_bytes));
    CUDA_CHECK(cudaMemcpy(d_fit, fit.data(), fit_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data(), seed_bytes, cudaMemcpyHostToDevice));

    int blocks = (nseeds + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1) blocks = 1;
    GpuTimer timer;
    timer.start();
    tract_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_fit, vol.nx, vol.ny, vol.nz,
                                                d_seeds, nseeds, cap, max_steps,
                                                step, fa_min, cos_min, d_pts, d_start, d_len);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("tract_kernel");

    // Bring back the flat points + per-seed (start,count), then copy the compact
    // contiguous run [start, start+count) out of each slot into a Streamline.
    std::vector<float> h_pts(static_cast<std::size_t>(nseeds) * cap * 3);
    std::vector<int>   h_start(nseeds), h_len(nseeds);
    CUDA_CHECK(cudaMemcpy(h_pts.data(), d_pts, pts_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_start.data(), d_start, idx_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_len.data(), d_len, idx_bytes, cudaMemcpyDeviceToHost));

    for (int s = 0; s < nseeds; ++s) {
        Streamline& L = lines[s];
        L.nsteps = h_len[s];
        const float* slot = &h_pts[static_cast<std::size_t>(s) * cap * 3];
        const float* run  = slot + static_cast<std::size_t>(h_start[s]) * 3;
        L.pts.assign(run, run + static_cast<std::size_t>(L.nsteps) * 3);
    }

    CUDA_CHECK(cudaFree(d_fit));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_len));
}
