// ===========================================================================
// src/kernels.cu  --  SMLM GPU kernels (localize + atomic render) + host loop
// ---------------------------------------------------------------------------
// Project 4.10 : Super-Resolution Microscopy Reconstruction  (STORM / PALM SMLM)
//
// GPU twin of the CPU reference. Both call the SAME smlm_localize() / render math
// (smlm.h) and reuse render_image()/summarize() from reference_cpu.cpp, so the
// results match bit-for-bit. main.cu runs both and compares. See ../THEORY.md
// "GPU mapping". Comment density is kept high (CLAUDE.md §6.2, ~1:1 in kernels).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <vector>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide the patch's global loads,
// and keeps many blocks resident for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// localize_kernel  --  one thread per interior pixel of ONE frame.
//
//   Launch config (set in smlm_gpu, once per frame):
//     interior grid = (H-2*PATCH_R) x (W-2*PATCH_R) pixels, flattened to 1-D.
//     grid  = ceil(interior / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: linear id `t` -> interior pixel
//       rr = t / IW + PATCH_R ,  cc = t % IW + PATCH_R
//   where IW = W - 2*PATCH_R is the interior width. So each thread owns exactly
//   one candidate pixel and, crucially, writes to output slot `t` -- the slot
//   index IS the scan position, which preserves canonical order for free.
//
//   Memory: reads a 7x7 patch from GLOBAL memory (the frame), writes one
//   Localization + one flag. No shared memory or atomics: fits are independent.
//   (A production kernel would stage the patch in shared memory and assign one
//   WARP per candidate for higher throughput -- see THEORY §4 and Exercise 3.)
//
//   `slot`  : [interior] output localizations (only valid where valid[t]==1).
//   `valid` : [interior] 1 if pixel t was detected + localized, else 0.
// ---------------------------------------------------------------------------
__global__ void localize_kernel(const float* __restrict__ frame, int H, int W,
                                double background, double threshold, int frame_idx,
                                Localization* __restrict__ slot,
                                unsigned char* __restrict__ valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's slot
    const int IW = W - 2 * PATCH_R;                        // interior width
    const int IH = H - 2 * PATCH_R;                        // interior height
    const int interior = IW * IH;
    if (t >= interior) return;                             // guard ragged block

    // Map the flat interior index back to an absolute (row, col) inside the frame.
    const int rr = t / IW + PATCH_R;   // absolute row of this candidate pixel
    const int cc = t % IW + PATCH_R;   // absolute column

    // DETECT: strict local maximum above threshold? (identical test to the CPU)
    if (smlm_is_local_max(frame, H, W, rr, cc, threshold)) {
        // LOCALIZE: the same sub-pixel fit the CPU runs, on this thread's patch.
        slot[t]  = smlm_localize(frame, H, W, rr, cc, background, frame_idx);
        valid[t] = 1;
    } else {
        valid[t] = 0;   // not an emitter: leave slot untouched, mark invalid
    }
}

// ---------------------------------------------------------------------------
// render_kernel  --  one thread per localization; ATOMIC scatter into the image.
//
//   Launch config: grid = ceil(n / THREADS_PER_BLOCK), block = THREADS_PER_BLOCK.
//   Thread `i` owns localization i; it computes the super-resolution bin its
//   (x,y) falls in and atomicAdds its FIXED-POINT photons there. Many emitters
//   can map to the same bin (that pile-up is exactly how the image forms), so the
//   writes collide -> atomicAdd. Integer fixed-point (smlm.h) makes the adds
//   COMMUTE, so the image is order-independent -> deterministic and CPU-matching.
//
//   Memory: reads one Localization from global, one atomic add to global. No
//   shared memory (bins are spread across a large image, so shared-mem privatis-
//   ation buys little here; see Exercise 4).
// ---------------------------------------------------------------------------
__global__ void render_kernel(const Localization* __restrict__ locs, int n,
                              int srH, int srW,
                              unsigned long long* __restrict__ img_fixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Localization L = locs[i];
    // Same bin math as render_image() on the host (truncation = floor for x,y>=0).
    const int sc = static_cast<int>(L.x * UPSAMPLE);
    const int sr = static_cast<int>(L.y * UPSAMPLE);
    if (sr < 0 || sr >= srH || sc < 0 || sc >= srW) return;  // off-grid, skip
    // atomicAdd on unsigned long long: hardware-serialized, order-independent
    // integer add. This is the atomic-reduction lesson of docs/PATTERNS.md §3.
    atomicAdd(&img_fixed[static_cast<std::size_t>(sr) * srW + sc],
              smlm_to_fixed(L.photons));
}

// ---------------------------------------------------------------------------
// smlm_gpu  --  host wrapper orchestrating the whole GPU pipeline.
//   Steps:
//     (1) upload the full frame stack once.
//     (2) for each frame, launch localize_kernel over its interior pixels; copy
//         back the per-slot localizations + valid flags, and COMPACT the valid
//         ones (in slot-index = scan order) into the growing canonical list.
//         Compaction on the host keeps the list order identical to the CPU's --
//         the simplest way to guarantee element-for-element comparability. (A
//         pure-GPU stream-compaction with CUB is Exercise 5.)
//     (3) upload the localization list, launch render_kernel to atomically build
//         the fixed-point super-resolution image, copy it back.
//     (4) summarize + free.
//   Only the kernels are timed (CUDA events), not the PCIe copies (THEORY §7).
// ---------------------------------------------------------------------------
ResultSummary smlm_gpu(const FrameStack& stack,
                       std::vector<Localization>& out_locs,
                       std::vector<unsigned long long>& img_fixed,
                       int& srH, int& srW, float* kernel_ms) {
    const int H = stack.H, W = stack.W;
    const int IW = W - 2 * PATCH_R;
    const int IH = H - 2 * PATCH_R;
    const int interior = IW * IH;                 // candidate slots per frame

    srH = H * UPSAMPLE;
    srW = W * UPSAMPLE;
    out_locs.clear();

    // (1) Upload the whole stack once (frames are re-used only within a frame,
    //     but a single upload keeps the host code simple and the copy off the
    //     timed path).
    float* d_stack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stack, stack.bytes()));
    CUDA_CHECK(cudaMemcpy(d_stack, stack.data.data(), stack.bytes(),
                          cudaMemcpyHostToDevice));

    // Per-frame scratch: the localization slots and the valid flags.
    Localization*  d_slot  = nullptr;
    unsigned char* d_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_slot,  static_cast<std::size_t>(interior) * sizeof(Localization)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<std::size_t>(interior) * sizeof(unsigned char)));

    std::vector<Localization>  h_slot(interior);
    std::vector<unsigned char> h_valid(interior);

    const int loc_grid = (interior + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    float total_ms = 0.0f;

    // (2) Detect + localize each frame; compact valids in scan order.
    for (int f = 0; f < stack.F; ++f) {
        const float* d_frame = d_stack + static_cast<std::size_t>(f) * stack.frame_pixels();
        timer.start();
        localize_kernel<<<loc_grid, THREADS_PER_BLOCK>>>(
            d_frame, H, W, stack.background, stack.threshold, f, d_slot, d_valid);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("localize_kernel");

        CUDA_CHECK(cudaMemcpy(h_slot.data(), d_slot,
                              static_cast<std::size_t>(interior) * sizeof(Localization),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_valid.data(), d_valid,
                              static_cast<std::size_t>(interior) * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
        // Keep valid slots in ascending index order == (row,col) scan order ==
        // the CPU's order. Appended per frame -> whole list is (frame,row,col).
        for (int t = 0; t < interior; ++t)
            if (h_valid[t]) out_locs.push_back(h_slot[t]);
    }

    // (3) Render the localizations into the fixed-point SR image (atomic scatter).
    const std::size_t img_n = static_cast<std::size_t>(srH) * srW;
    img_fixed.assign(img_n, 0ull);
    const int n = static_cast<int>(out_locs.size());
    if (n > 0) {
        Localization* d_locs = nullptr;
        unsigned long long* d_img = nullptr;
        CUDA_CHECK(cudaMalloc(&d_locs, static_cast<std::size_t>(n) * sizeof(Localization)));
        CUDA_CHECK(cudaMalloc(&d_img,  img_n * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(d_locs, out_locs.data(),
                              static_cast<std::size_t>(n) * sizeof(Localization),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_img, 0, img_n * sizeof(unsigned long long)));

        const int render_grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        timer.start();
        render_kernel<<<render_grid, THREADS_PER_BLOCK>>>(d_locs, n, srH, srW, d_img);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("render_kernel");

        CUDA_CHECK(cudaMemcpy(img_fixed.data(), d_img,
                              img_n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_locs));
        CUDA_CHECK(cudaFree(d_img));
    }
    *kernel_ms = total_ms;

    // (4) Free scratch + stack; build the summary (shared with the CPU path).
    CUDA_CHECK(cudaFree(d_slot));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_stack));
    return summarize(out_locs, img_fixed, srH, srW);
}
