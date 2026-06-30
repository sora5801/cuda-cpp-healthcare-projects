// ===========================================================================
// src/kernels.cu  --  Voxel-parallel 3D-convolution segmentation kernels
// ---------------------------------------------------------------------------
// Project 4.7 : Medical Image Segmentation (Deep Learning)   [REDUCED SCOPE]
//
// GPU twin of segment_cpu(): the SAME two-layer fully-convolutional head, but
// each output voxel is computed by its own CUDA thread. Both kernels call the
// shared conv3x3x3_at() / relu() from reference_cpu.h, so the GPU label map
// matches the CPU one exactly and the float logits match to ~1e-3.
// main.cu runs both paths and verifies they agree. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstdio>

// ---------------------------------------------------------------------------
// CONSTANT MEMORY for the network weights.
//   The weights are read by EVERY thread but never change during a launch, so
//   constant memory (a small, cached, broadcast-optimized space) is the ideal
//   home: when all threads in a warp read the same tap, the constant cache
//   services them in a single transaction. We size the arrays to this teaching
//   network's fixed shape (see reference_cpu.h):
//     c_w1 : [C_HID * 1     * KVOL]   layer-1 filter weights
//     c_b1 : [C_HID]                  layer-1 biases
//     c_w2 : [N_CLASS * C_HID * KVOL] layer-2 filter weights
//     c_b2 : [N_CLASS]                layer-2 biases
//   The host wrapper fills them with cudaMemcpyToSymbol before the launches.
// ---------------------------------------------------------------------------
__constant__ float c_w1[C_HID * 1 * KVOL];
__constant__ float c_b1[C_HID];
__constant__ float c_w2[N_CLASS * C_HID * KVOL];
__constant__ float c_b2[N_CLASS];

// ---------------------------------------------------------------------------
// seg_layer1_kernel: layer 1 = conv (1 input channel) + ReLU.
//   Launch config (set in segment_gpu):
//     grid  = ceil(D*H*W / SEG_BLOCK) blocks ; block = SEG_BLOCK threads.
//   Thread-to-data map: flat voxel index i = blockIdx.x*blockDim.x + threadIdx.x;
//   decode (z,y,x) from i. Each thread computes ALL C_HID hidden channels for
//   its voxel (a small fixed loop), reading the input neighbourhood from global
//   memory and the filter from constant memory.
//   Memory: global reads of the 3x3x3 input neighbourhood (gather), constant
//   reads of the weights, global write of C_HID hidden values. No atomics, no
//   shared memory -- output voxels are fully independent.
// ---------------------------------------------------------------------------
__global__ void seg_layer1_kernel(const float* __restrict__ d_in, int D, int H, int W,
                                  float* __restrict__ d_hidden) {
    const long long n = (long long)D * H * W;             // voxels per channel
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                                   // guard ragged last block

    // Decode the flat index i back into (z,y,x). Row-major (D,H,W) => x fastest.
    const int x = (int)(i % W);
    const int y = (int)((i / W) % H);
    const int z = (int)(i / ((long long)H * W));

    // One conv per hidden channel, then ReLU. conv3x3x3_at is the SHARED core.
    #pragma unroll
    for (int co = 0; co < C_HID; ++co) {
        const float* w = &c_w1[co * 1 * KVOL];            // filter for channel co
        const float  b = c_b1[co];
        const float a  = conv3x3x3_at(d_in, /*cin=*/1, D, H, W, w, b, z, y, x);
        d_hidden[(long long)co * n + i] = relu(a);        // activation
    }
}

// ---------------------------------------------------------------------------
// seg_layer2_kernel: layer 2 = conv (C_HID input channels) + per-voxel argmax.
//   Same one-thread-per-voxel mapping as layer 1. Each thread computes the
//   N_CLASS logits for its voxel, takes the argmax to get the 0/1 label, and
//   also writes the lesion-class (class 1) logit so the host can run a float
//   tolerance check against the CPU.
//   The argmax uses strict '>' so a tie keeps the LOWER class index -- this
//   matches segment_cpu() exactly, keeping the integer label map deterministic
//   and bit-identical to the CPU (so stdout is reproducible).
// ---------------------------------------------------------------------------
__global__ void seg_layer2_kernel(const float* __restrict__ d_hidden, int D, int H, int W,
                                  int* __restrict__ d_label, float* __restrict__ d_logit1) {
    const long long n = (long long)D * H * W;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int x = (int)(i % W);
    const int y = (int)((i / W) % H);
    const int z = (int)(i / ((long long)H * W));

    int   best_c = 0;
    float best_z = -1e30f;     // running max logit
    float lesion = 0.0f;       // class-1 logit, remembered for the tolerance check
    #pragma unroll
    for (int cls = 0; cls < N_CLASS; ++cls) {
        const float* w = &c_w2[cls * C_HID * KVOL];       // filter for this class
        const float  b = c_b2[cls];
        const float zl = conv3x3x3_at(d_hidden, C_HID, D, H, W, w, b, z, y, x);
        if (cls == 1) lesion = zl;
        if (zl > best_z) { best_z = zl; best_c = cls; }   // strict '>' -> low index on ties
    }
    d_label[i]  = best_c;
    d_logit1[i] = lesion;
}

// ---------------------------------------------------------------------------
// segment_gpu: host wrapper -- the canonical CUDA steps for this two-kernel
// pipeline:
//   (0) upload the fixed weights to constant memory (once per call).
//   (1) allocate device buffers: input volume, hidden maps, label, logit.
//   (2) copy the input volume H2D.
//   (3) launch layer 1, then layer 2 (timed together with CUDA events).
//   (4) copy the label map and lesion logits D2H.
//   (5) free device memory.
// Only step (3) is timed, so the reported figure is kernel compute, not PCIe
// transfer (discussed separately in THEORY). Timing is a teaching artifact.
// ---------------------------------------------------------------------------
void segment_gpu(const Volume& vol, const SegNet& net,
                 std::vector<int>& label, std::vector<float>& logit1,
                 float* kernel_ms) {
    const int D = vol.D, H = vol.H, W = vol.W;
    const long long n = vol.size();
    label.assign(static_cast<std::size_t>(n), 0);
    logit1.assign(static_cast<std::size_t>(n), 0.0f);

    // (0) Upload weights to constant memory. The host SegNet vectors are exactly
    //     the sizes of the __constant__ arrays (validated implicitly by sizeof).
    CUDA_CHECK(cudaMemcpyToSymbol(c_w1, net.w1.data(), net.w1.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b1, net.b1.data(), net.b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w2, net.w2.data(), net.w2.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b2, net.b2.data(), net.b2.size() * sizeof(float)));

    // (1) Device buffers (d_ prefix = device pointer; CLAUDE.md §12).
    float *d_in = nullptr, *d_hidden = nullptr, *d_logit1 = nullptr;
    int   *d_label = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,     n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, (long long)C_HID * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_label,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logit1, n * sizeof(float)));

    // (2) Copy the input intensity volume to the device.
    CUDA_CHECK(cudaMemcpy(d_in, vol.v.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // (3) Launch both layers. Grid covers all n voxels (ceiling division).
    const int block = SEG_BLOCK;
    const int grid  = (int)((n + block - 1) / block);
    GpuTimer timer;
    timer.start();
    seg_layer1_kernel<<<grid, block>>>(d_in, D, H, W, d_hidden);
    seg_layer2_kernel<<<grid, block>>>(d_hidden, D, H, W, d_label, d_logit1);
    *kernel_ms = timer.stop_ms();             // total GPU time for both kernels
    CUDA_CHECK_LAST("seg_layer1/2_kernel");   // catch launch + execution errors

    // (4) Bring the results back to the host.
    CUDA_CHECK(cudaMemcpy(label.data(),  d_label,  n * sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(logit1.data(), d_logit1, n * sizeof(float), cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_label));
    CUDA_CHECK(cudaFree(d_logit1));
}
