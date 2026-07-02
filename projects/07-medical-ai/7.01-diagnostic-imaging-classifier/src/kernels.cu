// ===========================================================================
// src/kernels.cu  --  GPU CNN forward pass: conv+pool and dense kernels
// ---------------------------------------------------------------------------
// Project 7.1 : Diagnostic Imaging Classifier   (REDUCED-SCOPE teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of reference_cpu.cpp's forward pass. It defines two kernels
//   (conv+pool, then dense) and the host wrapper classify_gpu() that moves data,
//   launches them, and brings the logits back. main.cu runs this next to the CPU
//   reference and asserts they agree EXACTLY.
//
//   The actual arithmetic (conv_pixel / pool_pixel / dense_logit) is NOT rewritten
//   here -- it is the SAME __host__ __device__ code from reference_cpu.h. This
//   file only supplies: (a) the thread->data index math, and (b) the memory
//   plumbing (constant memory for weights, global memory for images/features).
//   That separation is the whole point: identical math on both sides => the
//   float results are bit-identical, so verification uses tolerance 0.
//
// READ THIS AFTER: kernels.cuh (the mapping idea), reference_cpu.h (the math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstddef>

// ---------------------------------------------------------------------------
// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide latency, plenty of resident blocks for occupancy.
// ---------------------------------------------------------------------------
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY for the network weights.
//   The weights are small (< 1 KB here, constant-memory limit is 64 KB) and are
//   read by every thread but never written during a launch. Constant memory has
//   a per-SM broadcast cache: when all threads in a warp read the SAME address
//   (e.g. the same filter tap), it is served in a single transaction -- ideal.
//   We COPY the host weights into these symbols with cudaMemcpyToSymbol before
//   launching (see classify_gpu). Fixed-size arrays sized from the geometry.
// ---------------------------------------------------------------------------
__constant__ float c_conv_w [CONV_WSZ];              // conv filters
__constant__ float c_conv_b [NUM_F];                 // conv biases
__constant__ float c_dense_w[NUM_CLS * FLAT];        // dense weights
__constant__ float c_dense_b[NUM_CLS];               // dense biases

// ---------------------------------------------------------------------------
// conv_pool_kernel: one thread produces one POOLED feature value.
//   Launch config (set in classify_gpu):
//     total threads = n * FLAT   (FLAT = NUM_F * POOL_H * POOL_W)
//     grid = ceil(total / 256) blocks, block = 256 threads
//   Thread-to-data map: a flat id t is decoded into (i, f, py, px):
//     px = t % POOL_W ; py = (t / POOL_W) % POOL_H ;
//     f  = (t / (POOL_W*POOL_H)) % NUM_F ; i = t / FLAT
//   Memory: reads this image's pixels from GLOBAL memory through conv_pixel;
//   reads conv weights/biases from CONSTANT memory; writes one float to d_feat.
//   No atomics, no shared memory -- every output is independent (a pure gather).
// ---------------------------------------------------------------------------
__global__ void conv_pool_kernel(const float* __restrict__ d_images, int n,
                                 float* __restrict__ d_feat) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's flat id
    int total = n * FLAT;                            // number of pooled features
    if (t >= total) return;                          // guard the ragged last block

    // --- decode the flat id into (image, filter, pooled y, pooled x) -------
    int px = t % POOL_W;
    int py = (t / POOL_W) % POOL_H;
    int f  = (t / (POOL_W * POOL_H)) % NUM_F;
    int i  = t / FLAT;

    // Pointer to THIS image's pixels in the global batch buffer.
    const float* img = d_images + static_cast<std::size_t>(i) * IMG_SIZE;

    // --- CONV + ReLU over the 2x2 pooling window, then MAX-POOL ------------
    // The pooled feature at (py,px) is the max over conv activations at conv
    // coordinates (2*py+dy, 2*px+dx). We compute those four conv_pixel values on
    // the fly (fusing conv+pool) instead of materializing the whole feature map
    // -- fewer global writes, and it keeps the kernel to one output per thread.
    int cy0 = py * POOL;                             // top conv row of the block
    int cx0 = px * POOL;                             // left conv col of the block
    float m = conv_pixel(img, c_conv_w, c_conv_b, f, cy0, cx0);  // seed = (0,0)
    for (int dy = 0; dy < POOL; ++dy)
        for (int dx = 0; dx < POOL; ++dx) {
            float v = conv_pixel(img, c_conv_w, c_conv_b, f, cy0 + dy, cx0 + dx);
            if (v > m) m = v;                        // running max (same as CPU)
        }

    // Write the pooled feature at the flattened index the dense layer expects:
    //   feat[(f*POOL_H + py)*POOL_W + px] within image i.
    int feat_j = (f * POOL_H + py) * POOL_W + px;
    d_feat[static_cast<std::size_t>(i) * FLAT + feat_j] = m;
}

// ---------------------------------------------------------------------------
// dense_kernel: one thread computes one class logit for one image.
//   Launch: total = n * NUM_CLS threads. Thread t -> (image i, class c):
//     c = t % NUM_CLS ; i = t / NUM_CLS.
//   Uses the SHARED dense_logit() (a FLAT-length dot product) reading the dense
//   weights/biases from CONSTANT memory and the features from GLOBAL memory.
// ---------------------------------------------------------------------------
__global__ void dense_kernel(const float* __restrict__ d_feat, int n,
                             float* __restrict__ d_logits) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * NUM_CLS;
    if (t >= total) return;

    int c = t % NUM_CLS;                             // which output class
    int i = t / NUM_CLS;                             // which image

    const float* feat = d_feat + static_cast<std::size_t>(i) * FLAT;
    d_logits[static_cast<std::size_t>(i) * NUM_CLS + c] =
        dense_logit(feat, c_dense_w, c_dense_b, c);
}

// ---------------------------------------------------------------------------
// classify_gpu: host wrapper -- the whole forward pass on the GPU.
//   Steps: (1) copy weights to constant memory; (2) upload images; (3) launch
//   conv_pool_kernel; (4) launch dense_kernel; (5) copy logits back and derive
//   the argmax prediction with the shared helper; (6) free device memory.
//   Only the two kernel launches are timed (CUDA events) -> *kernel_ms reflects
//   compute, not PCIe transfers (discussed in THEORY.md).
// ---------------------------------------------------------------------------
void classify_gpu(const Weights& w, const Dataset& d,
                  std::vector<float>& logits, std::vector<int>& pred,
                  float* kernel_ms) {
    const int n = d.n;
    logits.assign(static_cast<std::size_t>(n) * NUM_CLS, 0.0f);
    pred.assign(static_cast<std::size_t>(n), 0);

    // (1) Weights -> constant memory. cudaMemcpyToSymbol copies from host into
    //     the named __constant__ symbol; sizes must match the array bytes.
    CUDA_CHECK(cudaMemcpyToSymbol(c_conv_w,  w.conv_w.data(),  CONV_WSZ * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_conv_b,  w.conv_b.data(),  NUM_F * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dense_w, w.dense_w.data(), NUM_CLS * FLAT * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dense_b, w.dense_b.data(), NUM_CLS * sizeof(float)));

    // (2) Device buffers: the image batch (input), pooled features (scratch),
    //     class logits (output). d_ prefix marks DEVICE pointers (CLAUDE.md 12).
    const std::size_t img_bytes  = static_cast<std::size_t>(n) * IMG_SIZE * sizeof(float);
    const std::size_t feat_bytes = static_cast<std::size_t>(n) * FLAT     * sizeof(float);
    const std::size_t log_bytes  = static_cast<std::size_t>(n) * NUM_CLS  * sizeof(float);
    float *d_images = nullptr, *d_feat = nullptr, *d_logits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_feat,   feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_logits, log_bytes));
    CUDA_CHECK(cudaMemcpy(d_images, d.images.data(), img_bytes, cudaMemcpyHostToDevice));

    // (3)+(4) Launch both kernels, timing them together. Grids round up so the
    //         ragged last block is covered (guarded inside each kernel).
    const int conv_threads  = n * FLAT;
    const int conv_blocks   = (conv_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int dense_threads = n * NUM_CLS;
    const int dense_blocks  = (dense_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    conv_pool_kernel<<<conv_blocks, THREADS_PER_BLOCK>>>(d_images, n, d_feat);
    dense_kernel    <<<dense_blocks, THREADS_PER_BLOCK>>>(d_feat, n, d_logits);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("conv_pool_kernel / dense_kernel");   // launch + exec errors

    // (5) Logits back to host; derive argmax with the shared helper so the
    //     prediction rule is identical to the CPU path.
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, log_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        pred[i] = argmax2(logits[static_cast<std::size_t>(i) * NUM_CLS + 0],
                          logits[static_cast<std::size_t>(i) * NUM_CLS + 1]);

    // (6) Free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_feat));
    CUDA_CHECK(cudaFree(d_logits));
}
