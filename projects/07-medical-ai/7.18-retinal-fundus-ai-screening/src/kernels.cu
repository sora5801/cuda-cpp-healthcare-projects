// ===========================================================================
// src/kernels.cu  --  GPU forward pass: tiled conv+ReLU, max-pool, GAP, FC, CAM
// ---------------------------------------------------------------------------
// Project 7.18 : Retinal Fundus AI Screening
//
// WHAT THIS FILE DOES
//   The GPU twin of forward_cpu(). It runs the whole inference pipeline on the
//   device, calling the SAME per-pixel math (cnn_core.h: conv_at, relu,
//   maxpool2x2_at) as the CPU reference, so main.cu can verify the two agree.
//
//   Kernels, in launch order:
//     conv_relu_tiled   -- conv + ReLU, staging the input into SHARED MEMORY
//                          (the headline lesson; the 2-D analog of flagship 7.10)
//     maxpool2x2        -- 2x2 stride-2 down-sample (one thread per output px)
//     global_avg_pool   -- one block per feature map reduces its plane to a mean
//     fc_and_cam        -- the classifier logits + the Grad-CAM-style heatmap
//
//   The two conv layers dominate the cost, so forward_gpu() times only those
//   (CUDA events) and reports the figure to stderr as a teaching artifact.
//
// READ THIS AFTER: kernels.cuh (thread mapping), cnn_core.h (the shared math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <vector>

// ===========================================================================
// KERNEL 1 -- conv_relu_tiled : shared-memory tiled convolution + ReLU
// ---------------------------------------------------------------------------
// Thread-to-data map:
//   * blockIdx.z  selects the OUTPUT channel `oc` (0..C_out-1).
//   * a TILE x TILE block of threads covers a TILE x TILE patch of that map;
//     thread (threadIdx.x, threadIdx.y) in block (blockIdx.x, blockIdx.y) owns
//     output pixel (ox = blockIdx.x*TILE + threadIdx.x,
//                   oy = blockIdx.y*TILE + threadIdx.y).
//
// Why shared memory:
//   Every output pixel reads a 3x3 window of EACH input channel. Neighbours in
//   the tile overlap by 2 of 3 columns/rows, so the naive kernel re-reads each
//   input pixel up to K*K = 9 times from slow global memory. Instead each block
//   loads its (TILE+2*halo) x (TILE+2*halo) patch of one input channel into
//   shared memory ONCE (cooperatively), __syncthreads, then all threads read
//   their 3x3 windows from that on-chip tile. We loop this over input channels,
//   accumulating into a per-thread register `acc`.
//
// Note: we reuse conv_at()'s indexing logic conceptually but re-derive the MAC
//   here against the shared tile (conv_at reads a flat global buffer; the tiled
//   read is the optimization the header promised). The ARITHMETIC per pixel is
//   identical to the CPU path (same weights, same +, same zero padding), so the
//   results match to float round-off.
// ===========================================================================
__global__ void conv_relu_tiled(const float* __restrict__ in, int C_in, int H, int W,
                                 const float* __restrict__ weights,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out /* [C_out*H*W] */) {
    // Shared tile for ONE input channel: (TILE + 2*halo) square.
    const int SH = TILE + 2 * CNN_HALO;          // shared tile side (18 for TILE=16)
    __shared__ float tile[(TILE + 2 * CNN_HALO) * (TILE + 2 * CNN_HALO)];

    const int oc = blockIdx.z;                    // this block's output channel
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int ox = blockIdx.x * TILE + tx;        // output column this thread owns
    const int oy = blockIdx.y * TILE + ty;        // output row    this thread owns

    // Top-left GLOBAL coordinate of the shared tile (including the left/top halo).
    const int tile_x0 = blockIdx.x * TILE - CNN_HALO;
    const int tile_y0 = blockIdx.y * TILE - CNN_HALO;

    float acc = bias[oc];                         // start from the channel bias
    const int wbase = oc * C_in * CNN_KSIZE * CNN_KSIZE;   // weights for this oc

    // Sweep input channels, staging each into shared memory then convolving.
    for (int ic = 0; ic < C_in; ++ic) {
        const int in_ch = ic * H * W;             // start of input channel ic

        // --- Cooperative load of the SH x SH halo tile into shared memory ---
        // Each thread loads one-or-more tile cells in a strided loop so that any
        // TILE (<=SH) block size still fills the whole SH x SH tile. Out-of-image
        // cells are zero (matches the CPU zero-padding in conv_at()).
        for (int sy = ty; sy < SH; sy += TILE) {
            const int gy = tile_y0 + sy;          // global row for this tile cell
            for (int sx = tx; sx < SH; sx += TILE) {
                const int gx = tile_x0 + sx;      // global col for this tile cell
                float v = 0.0f;
                if (gy >= 0 && gy < H && gx >= 0 && gx < W)
                    v = in[in_ch + gy * W + gx];
                tile[sy * SH + sx] = v;
            }
        }
        __syncthreads();                          // tile fully loaded before reads

        // --- Convolve this thread's 3x3 window from the shared tile ---------
        // The thread's window top-left in tile coords is (ty, tx); adding ky/kx
        // walks the 3x3. Because the tile already includes the halo, no bounds
        // test is needed here (the zeros were baked in during the load).
        if (ox < W && oy < H) {
            const int w_ch = wbase + ic * CNN_KSIZE * CNN_KSIZE;
            #pragma unroll
            for (int ky = 0; ky < CNN_KSIZE; ++ky)
                #pragma unroll
                for (int kx = 0; kx < CNN_KSIZE; ++kx)
                    acc += weights[w_ch + ky * CNN_KSIZE + kx]
                         * tile[(ty + ky) * SH + (tx + kx)];
        }
        __syncthreads();                          // reuse `tile` for next channel
    }

    // ReLU + store. relu() is the shared HD helper, identical to the CPU path.
    if (ox < W && oy < H)
        out[oc * H * W + oy * W + ox] = relu(acc);
}

// ===========================================================================
// KERNEL 2 -- maxpool2x2 : 2x2 stride-2 max-pool (one thread per output pixel)
// ---------------------------------------------------------------------------
//   grid.z = channel; a TILE x TILE block covers a TILE x TILE OUTPUT patch.
//   Output size is (H/2) x (W/2). Calls maxpool2x2_at() so CPU/GPU pool alike.
// ===========================================================================
__global__ void maxpool2x2(const float* __restrict__ in, int C, int H, int W,
                            float* __restrict__ out /* [C*(H/2)*(W/2)] */) {
    const int c  = blockIdx.z;
    const int ox = blockIdx.x * TILE + threadIdx.x;   // output column
    const int oy = blockIdx.y * TILE + threadIdx.y;   // output row
    const int Ho = H / 2, Wo = W / 2;
    if (ox < Wo && oy < Ho)
        out[c * Ho * Wo + oy * Wo + ox] = maxpool2x2_at(in, c, H, W, oy, ox);
}

// ===========================================================================
// KERNEL 3 -- global_avg_pool : reduce each feature map plane to its mean
// ---------------------------------------------------------------------------
//   One BLOCK per feature channel (blockIdx.x = c). The block's threads stride
//   over the H*W plane summing into shared memory, then a tree reduction gives
//   the total; thread 0 writes total/(H*W). This is a classic block reduction.
//   `spatial = H*W`; the sum is done in float (matches the CPU's double? -- no:
//   the CPU uses double for GAP; we accept the tiny resulting difference within
//   the documented tolerance -- see THEORY.md section 5).
// ===========================================================================
__global__ void global_avg_pool(const float* __restrict__ in, int spatial,
                                 float* __restrict__ gap /* [C] */) {
    extern __shared__ float sdata[];              // blockDim.x floats
    const int c = blockIdx.x;                     // this block's channel
    const int t = threadIdx.x;
    const int base = c * spatial;

    float partial = 0.0f;                         // this thread's running sum
    for (int i = t; i < spatial; i += blockDim.x)
        partial += in[base + i];
    sdata[t] = partial;
    __syncthreads();

    // Tree reduction in shared memory: fold the array in half each step.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) sdata[t] += sdata[t + stride];
        __syncthreads();
    }
    if (t == 0) gap[c] = sdata[0] / static_cast<float>(spatial);
}

// ===========================================================================
// KERNEL 4 -- fc_and_cam : classifier logits, then the Grad-CAM heatmap
// ---------------------------------------------------------------------------
//   Launched in two tiny passes from forward_gpu():
//     pass A (compute_logits=1): NUM_CLASSES threads, thread k computes
//        logit[k] = fc_b[k] + sum_c fc_w[k,c]*gap[c].
//     pass B (compute_logits=0): after the host picks the winning class `best`,
//        one thread per CAM pixel computes ReLU(sum_c fc_w[best,c]*f2(c,y,x)).
//   Splitting keeps each kernel dead simple and deterministic.
// ===========================================================================
__global__ void fc_logits(const float* __restrict__ gap,
                          const float* __restrict__ fc_w,
                          const float* __restrict__ fc_b,
                          float* __restrict__ logits) {
    const int k = threadIdx.x;                    // one thread per class
    if (k >= CNN_NUM_CLASSES) return;
    float acc = fc_b[k];
    for (int c = 0; c < CNN_C2; ++c)
        acc += fc_w[k * CNN_C2 + c] * gap[c];
    logits[k] = acc;
}

__global__ void cam_kernel(const float* __restrict__ f2, int Hf, int Wf,
                          const float* __restrict__ fc_w, int best,
                          float* __restrict__ cam) {
    const int x = blockIdx.x * TILE + threadIdx.x;
    const int y = blockIdx.y * TILE + threadIdx.y;
    if (x >= Wf || y >= Hf) return;
    const int spatial = Hf * Wf;
    float s = 0.0f;
    for (int c = 0; c < CNN_C2; ++c)
        s += fc_w[best * CNN_C2 + c] * f2[c * spatial + y * Wf + x];
    cam[y * Wf + x] = relu(s);
}

// ===========================================================================
// HOST ORCHESTRATION -- forward_gpu
// ---------------------------------------------------------------------------
// Allocates device buffers for every stage, uploads image + weights, launches
// the kernels in order, brings back logits + CAM, and finishes softmax/argmax
// on the host (5 numbers -- not worth a kernel). Times only the two conv layers.
// ===========================================================================

// Small helper: upload a host vector to a fresh device buffer.
static float* upload(const std::vector<float>& v) {
    float* d = nullptr;
    const std::size_t bytes = v.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, v.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

void forward_gpu(const FundusImage& img, const CnnModel& model,
                 ForwardResult& out, float* conv_ms) {
    const int H0 = img.H, W0 = img.W;
    const int H1 = H0 / 2, W1 = W0 / 2;           // after pool 1
    const int H2 = H1 / 2, W2 = W1 / 2;           // after pool 2

    // --- Upload inputs and weights -----------------------------------------
    float* d_img     = upload(img.data);
    float* d_conv1_w = upload(model.conv1_w);
    float* d_conv1_b = upload(model.conv1_b);
    float* d_conv2_w = upload(model.conv2_w);
    float* d_conv2_b = upload(model.conv2_b);
    float* d_fc_w    = upload(model.fc_w);
    float* d_fc_b    = upload(model.fc_b);

    // --- Intermediate device buffers ---------------------------------------
    float *d_c1 = nullptr, *d_p1 = nullptr, *d_c2 = nullptr, *d_f2 = nullptr;
    float *d_gap = nullptr, *d_logits = nullptr, *d_cam = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c1, (std::size_t)CNN_C1 * H0 * W0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1, (std::size_t)CNN_C1 * H1 * W1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2, (std::size_t)CNN_C2 * H1 * W1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f2, (std::size_t)CNN_C2 * H2 * W2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gap, (std::size_t)CNN_C2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, (std::size_t)CNN_NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cam, (std::size_t)H2 * W2 * sizeof(float)));

    // Block/grid helpers for the tiled 2-D kernels.
    const dim3 block(TILE, TILE);
    auto grid2d = [&](int Wd, int Hd, int Cd) {
        return dim3((Wd + TILE - 1) / TILE, (Hd + TILE - 1) / TILE, Cd);
    };

    // --- Time the two conv+ReLU layers (the dominant cost) -----------------
    GpuTimer timer; timer.start();

    // Layer 1: conv (3->C1) + ReLU  at full resolution.
    conv_relu_tiled<<<grid2d(W0, H0, CNN_C1), block>>>(
        d_img, CNN_C_IN, H0, W0, d_conv1_w, d_conv1_b, d_c1);
    CUDA_CHECK_LAST("conv_relu_tiled (layer1)");
    // Pool 1: 2x2 -> H1 x W1.
    maxpool2x2<<<grid2d(W1, H1, CNN_C1), block>>>(d_c1, CNN_C1, H0, W0, d_p1);
    CUDA_CHECK_LAST("maxpool2x2 (layer1)");

    // Layer 2: conv (C1->C2) + ReLU at H1 x W1.
    conv_relu_tiled<<<grid2d(W1, H1, CNN_C2), block>>>(
        d_p1, CNN_C1, H1, W1, d_conv2_w, d_conv2_b, d_c2);
    CUDA_CHECK_LAST("conv_relu_tiled (layer2)");
    // Pool 2: 2x2 -> H2 x W2 (this is the CAM/feature grid).
    maxpool2x2<<<grid2d(W2, H2, CNN_C2), block>>>(d_c2, CNN_C2, H1, W1, d_f2);
    CUDA_CHECK_LAST("maxpool2x2 (layer2)");

    *conv_ms = timer.stop_ms();                   // GPU-measured conv-layer time

    // --- Global average pool: one block per channel ------------------------
    const int gap_threads = 128;                  // <= spatial; power of two for the tree
    global_avg_pool<<<CNN_C2, gap_threads, gap_threads * sizeof(float)>>>(
        d_f2, H2 * W2, d_gap);
    CUDA_CHECK_LAST("global_avg_pool");

    // --- Classifier logits (NUM_CLASSES threads) ---------------------------
    fc_logits<<<1, CNN_NUM_CLASSES>>>(d_gap, d_fc_w, d_fc_b, d_logits);
    CUDA_CHECK_LAST("fc_logits");

    // Bring logits back; softmax + argmax on the host (5 numbers).
    out.logits.assign(CNN_NUM_CLASSES, 0.0f);
    CUDA_CHECK(cudaMemcpy(out.logits.data(), d_logits,
                          CNN_NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
    softmax(out.logits, out.probs);
    int best = 0;
    for (int k = 1; k < CNN_NUM_CLASSES; ++k)
        if (out.probs[k] > out.probs[best]) best = k;
    out.pred_grade = best;

    // --- CAM for the winning class -----------------------------------------
    cam_kernel<<<grid2d(W2, H2, 1), block>>>(d_f2, H2, W2, d_fc_w, best, d_cam);
    CUDA_CHECK_LAST("cam_kernel");
    out.cam_h = H2; out.cam_w = W2;
    out.cam.assign((std::size_t)H2 * W2, 0.0f);
    CUDA_CHECK(cudaMemcpy(out.cam.data(), d_cam,
                          (std::size_t)H2 * W2 * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Free everything (no GPU garbage collector) ------------------------
    for (float* p : {d_img, d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b, d_fc_w, d_fc_b,
                     d_c1, d_p1, d_c2, d_f2, d_gap, d_logits, d_cam})
        CUDA_CHECK(cudaFree(p));
}
