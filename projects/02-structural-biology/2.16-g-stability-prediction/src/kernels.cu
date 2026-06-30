// ===========================================================================
// src/kernels.cu  --  ΔΔG saturation-mutagenesis kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.16 : ΔΔG Stability Prediction (reduced-scope teaching version)
//
// This is the GPU twin of ddg_scan_cpu() in reference_cpu.cpp. main.cu runs both
// and asserts they agree to a documented tolerance. The per-mutation math is the
// SHARED ddg_predict() from ddg_model.h, so the only CPU/GPU differences come
// from floating-point rounding in tanhf (see ../THEORY.md "Numerical").
// ===========================================================================
#include "kernels.cuh"
#include "ddg_model.h"           // ddg_predict (compiled __host__ __device__ here)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// Per-residue FEATURES in CONSTANT memory.
//   * Every thread that handles position p reads c_wt[p] and c_buried[p]; the 20
//     threads for a position all read the SAME two values. Constant memory's
//     hardware broadcast cache serves one address to a whole warp in a single
//     transaction -- ideal for this read-only, reused-by-many data.
//   * Sized to MAX_RESIDUES (compile-time) so the buffers are fixed-size and live
//     in the 64 KB constant bank: 4096 ints + 4096 floats = 32 KB total. Filled
//     by cudaMemcpyToSymbol() in ddg_scan_gpu().
// ---------------------------------------------------------------------------
__constant__ int   c_wt[MAX_RESIDUES];       // [L] wild-type AA index per position
__constant__ float c_buried[MAX_RESIDUES];   // [L] burial fraction per position

// Block shape: 32 amino-acid lanes (we only need NUM_AA=20, but 32 = one warp so
// the threadIdx.x dimension aligns to a warp) × 8 positions per block. 32*8 = 256
// threads/block, a good occupancy default on sm_75..sm_89. The 12 unused
// amino-acid lanes (20..31) are masked off by the bounds check in the kernel.
static constexpr int AA_LANES   = 32;   // threadIdx.x spans amino acids (warp-aligned)
static constexpr int POS_PER_BLK = 8;   // threadIdx.y spans positions

// ---------------------------------------------------------------------------
// ddg_scan_kernel: one thread per (position p, amino acid a) cell.
//   Thread mapping:
//     a = blockIdx.x * AA_LANES   + threadIdx.x        (amino-acid index)
//     p = blockIdx.y * POS_PER_BLK + threadIdx.y       (residue position, base)
//   We grid-stride over positions in y so a fixed-height grid covers any L.
//   Memory: c_wt / c_buried from the constant cache; one write to global out[].
//   No shared memory or atomics -- the cells are fully independent (the textbook
//   "embarrassingly parallel" case; see ../THEORY.md "GPU mapping").
// ---------------------------------------------------------------------------
__global__ void ddg_scan_kernel(int L, float* __restrict__ out) {
    // This thread's amino-acid column. Lanes 20..31 of each x-block are spare
    // (warp padding); they must NOT write, so guard a >= NUM_AA below.
    const int a = blockIdx.x * AA_LANES + threadIdx.x;
    if (a >= NUM_AA) return;                       // mask the warp-padding lanes

    // Grid-stride over residue positions: start at this thread's position and
    // step by the total number of position-threads in the grid until p >= L.
    const int p_stride = gridDim.y * POS_PER_BLK;  // total positions covered per step
    for (int p = blockIdx.y * POS_PER_BLK + threadIdx.y; p < L; p += p_stride) {
        const int   wt  = c_wt[p];        // wild-type AA at this position (broadcast)
        const float bur = c_buried[p];    // burial fraction at this position (broadcast)
        // The SHARED model: identical to the CPU reference's call. The (a == wt)
        // cell evaluates to exactly 0 (self-mutation) by construction.
        out[static_cast<std::size_t>(p) * NUM_AA + a] = ddg_predict(wt, a, bur);
    }
}

// ---------------------------------------------------------------------------
// ddg_scan_gpu: the canonical CUDA steps, with the per-residue features going to
// constant memory instead of a global buffer. We time ONLY the kernel (CUDA
// events), not the H2D/D2H copies (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void ddg_scan_gpu(const Protein& prot, std::vector<float>& out, float* kernel_ms) {
    const int L = prot.L;
    out.assign(static_cast<std::size_t>(L) * NUM_AA, 0.0f);
    const std::size_t out_bytes = out.size() * sizeof(float);

    // (a) Upload the per-residue features to the __constant__ symbols. These are
    //     special copies that target the constant bank rather than ordinary
    //     global memory. L is validated against MAX_RESIDUES by the caller.
    CUDA_CHECK(cudaMemcpyToSymbol(c_wt, prot.wt_code.data(),
                                  static_cast<std::size_t>(L) * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_buried, prot.buried.data(),
                                  static_cast<std::size_t>(L) * sizeof(float)));

    // (b) Allocate the output score grid on the device.
    float* d_out = nullptr;   // [L*NUM_AA] device, row-major
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    // (c) Launch over the (amino-acid, position) grid. x covers NUM_AA columns
    //     (rounded up to whole AA_LANES warps); y covers the L positions in
    //     blocks of POS_PER_BLK, capped so the grid stays modest -- the
    //     grid-stride loop handles any remaining positions.
    const dim3 block(AA_LANES, POS_PER_BLK);
    int blocks_x = (NUM_AA + AA_LANES - 1) / AA_LANES;        // = 1 (20 <= 32)
    int blocks_y = (L + POS_PER_BLK - 1) / POS_PER_BLK;       // enough rows for L
    if (blocks_y > 4096) blocks_y = 4096;                     // cap; stride covers rest
    const dim3 grid(blocks_x, blocks_y);

    GpuTimer timer;
    timer.start();
    ddg_scan_kernel<<<grid, block>>>(L, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("ddg_scan_kernel");

    // (d) Copy the score grid back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
