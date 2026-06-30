// ===========================================================================
// src/reference_cpu.h  --  Volume model + tiny 3D U-Net-style segmentation head
//                          + the SHARED per-voxel convolution core (CPU/GPU)
// ---------------------------------------------------------------------------
// Project 4.7 : Medical Image Segmentation (Deep Learning)   [REDUCED SCOPE]
//
// WHAT THIS PROJECT COMPUTES
//   Volumetric semantic SEGMENTATION of a 3D medical image: label every voxel
//   of a CT/MRI volume as belonging to an anatomical structure (here: "lesion"
//   foreground vs. background). Production tools (nnU-Net, TotalSegmentator,
//   MONAI, Swin-UNETR) do this with deep encoder-decoder CNNs trained on
//   thousands of labelled volumes. We ship a deliberately small, FULLY
//   DETERMINISTIC teaching version that exercises the exact GPU primitive those
//   networks spend ~90% of their FLOPs on: the **3D convolution**.
//
//   The teaching network is a 2-layer fully-convolutional segmentation head:
//
//        input volume  x[1, D, H, W]                (1 input channel = intensity)
//          |  3x3x3 conv  (C_HID filters) + bias  -> ReLU
//        hidden  h[C_HID, D, H, W]                  (learned feature maps)
//          |  3x3x3 conv  (2 filters)     + bias
//        logits  z[2,    D, H, W]                   (background / lesion scores)
//          |  argmax over the 2 channels
//        label  y[D, H, W] in {0,1}                 (the segmentation mask)
//
//   The two conv layers use FIXED, hand-designed weights (a smoothing stage then
//   a center-surround "blob detector"), so the output is reproducible to the bit
//   in the integer label map -- no training, no randomness. This is the "forward
//   inference (sliding-window patch) pass" the catalog calls out, reduced to one
//   patch = the whole small volume. See THEORY.md "Where this sits in the real
//   world" for what the full nnU-Net pipeline adds (encoder/decoder, skip
//   connections, training, augmentation, sliding-window stitching).
//
// WHY A GPU
//   Each output voxel of a conv layer is an INDEPENDENT dot product of a small
//   weight stencil with a local neighbourhood -- millions of identical, parallel
//   little reductions. That is the canonical 3D-stencil / gather workload cuDNN
//   accelerates; we map one CUDA thread to one output voxel (kernels.cu). A
//   512x512x200 CT through a real 3D U-Net is ~200 GFLOPs per forward pass --
//   minutes on a CPU, < 1 s on a GPU. Our toy volume is tiny so it fits in a
//   slide, but the parallel structure is identical.
//
// THE SHARED __host__ __device__ CORE (PATTERNS.md §2)
//   conv3x3x3_at() computes ONE output voxel of a 3x3x3 convolution. The CPU
//   reference loops it over every voxel; the GPU kernel calls it from one thread
//   per voxel. Identical math on both sides -> the label maps match exactly and
//   the float logits match to ~1e-3. Keep this header CUDA-type-free so the host
//   compiler (cl.exe/g++) can include it too.
//
//   READ THIS AFTER: util/io.hpp. READ BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (PATTERNS.md §2 idiom).
//   * Under nvcc (__CUDACC__ defined) it expands to __host__ __device__, so the
//     SAME inline function is compiled for both the CPU and the GPU.
//   * Under the plain host compiler the decorators do not exist, so it expands
//     to nothing and the function is an ordinary inline C++ function.
//   This is what guarantees CPU/GPU numerical parity for verification.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Network shape constants (small on purpose -- a teaching head, not nnU-Net).
//   C_HID  : number of hidden feature maps produced by conv layer 1.
//   N_CLASS: number of output classes (2 = background / lesion).
//   KR     : convolution radius. A 3x3x3 kernel has radius 1 (taps at -1,0,+1).
static constexpr int C_HID   = 2;   // hidden channels (smooth + center-surround)
static constexpr int N_CLASS = 2;   // background (0) and lesion (1)
static constexpr int KR      = 1;   // 3x3x3 kernel radius
static constexpr int KW      = 2 * KR + 1;          // kernel width per axis = 3
static constexpr int KVOL    = KW * KW * KW;         // taps per filter = 27

// ---------------------------------------------------------------------------
// Volume: a 3D scalar image stored as a flat, row-major (D,H,W) float array.
//   Index of voxel (z,y,x) is  ((z*H) + y)*W + x  -- the same layout the GPU
//   kernel uses, so a voxel maps to the same memory address on both sides.
//   Intensities are normalized to roughly [0,1] (Hounsfield/MRI units rescaled
//   by the data generator); the network is intensity-driven, like a real CT
//   segmenter.
// ---------------------------------------------------------------------------
struct Volume {
    int D = 0, H = 0, W = 0;        // depth (z), height (y), width (x)
    std::vector<float> v;           // [D*H*W] voxel intensities, row-major

    // Flat index helper (host-only; kernels recompute this inline on device).
    int idx(int z, int y, int x) const { return (z * H + y) * W + x; }
    long long size() const { return (long long)D * H * W; }
};

// ---------------------------------------------------------------------------
// SegNet: the fixed weights of the 2-layer segmentation head.
//   Layout (all row-major, contiguous):
//     w1 : [C_HID][1][KVOL]       conv-1 weights (1 input channel)
//     b1 : [C_HID]                conv-1 biases
//     w2 : [N_CLASS][C_HID][KVOL] conv-2 weights (C_HID input channels)
//     b2 : [N_CLASS]              conv-2 biases
//   The weights are built deterministically by make_segnet() (reference_cpu.cpp)
//   -- a Gaussian smoother in layer 1 and a center-surround blob detector in
//   layer 2 -- so no training data or RNG is involved.
// ---------------------------------------------------------------------------
struct SegNet {
    std::vector<float> w1;   // [C_HID * 1     * KVOL]
    std::vector<float> b1;   // [C_HID]
    std::vector<float> w2;   // [N_CLASS * C_HID * KVOL]
    std::vector<float> b2;   // [N_CLASS]
};

// ---------------------------------------------------------------------------
// conv3x3x3_at  --  THE SHARED CORE: one output voxel of a 3x3x3 convolution.
//   Computes  acc = bias + sum_{c,dz,dy,dx} w[c,dz,dy,dx] * in[c, z+dz, y+dy, x+dx]
//   for a single output voxel (z,y,x), summing over `cin` input channels and the
//   27 stencil taps, with ZERO PADDING at the volume border (out-of-range voxels
//   contribute 0 -- the standard "same" convolution boundary).
//
//   Parameters (all device-safe types; no std:: containers so it runs on GPU):
//     in    : [cin * D*H*W] input feature maps, channel-major then row-major.
//     cin   : number of input channels.
//     D,H,W : volume dimensions.
//     w     : [cin * KVOL] weights for THIS output channel (one filter).
//     bias  : scalar bias for THIS output channel.
//     z,y,x : the output voxel coordinate this call computes.
//   Returns the pre-activation (logit) for this output voxel & channel.
//
//   Why a function (not inlined twice): putting the loop here once guarantees
//   the CPU reference and the GPU kernel execute the IDENTICAL ordered sequence
//   of multiply-adds, so their results agree (PATTERNS.md §2). Complexity per
//   call is O(cin * 27) fused multiply-adds; the tap loop order (c, dz, dy, dx)
//   is fixed so the float summation order matches on both sides.
// ---------------------------------------------------------------------------
HD inline float conv3x3x3_at(const float* in, int cin, int D, int H, int W,
                             const float* w, float bias,
                             int z, int y, int x) {
    float acc = bias;                              // start from the channel bias
    const int plane = H * W;                       // voxels per z-slice
    const int chan  = D * plane;                   // voxels per input channel
    // Loop over input channels, then the 3x3x3 neighbourhood (dz,dy,dx).
    for (int c = 0; c < cin; ++c) {
        const float* inc = in + (long long)c * chan;   // base of channel c
        const float* wc  = w  + (long long)c * KVOL;   // weights for channel c
        int t = 0;                                  // running tap index 0..26
        for (int dz = -KR; dz <= KR; ++dz) {
            const int zz = z + dz;                  // neighbour z (may be OOB)
            for (int dy = -KR; dy <= KR; ++dy) {
                const int yy = y + dy;
                for (int dx = -KR; dx <= KR; ++dx, ++t) {
                    const int xx = x + dx;
                    // Zero padding: skip taps whose neighbour falls outside the
                    // volume (their contribution is defined to be 0).
                    if (zz < 0 || zz >= D || yy < 0 || yy >= H || xx < 0 || xx >= W)
                        continue;
                    const float val = inc[(zz * H + yy) * W + xx];
                    acc += wc[t] * val;             // one (fused) multiply-add
                }
            }
        }
    }
    return acc;
}

// relu: the standard non-linearity between conv layers (max(0,x)). Shared so the
// CPU and GPU clamp identically. Trivial, but kept in the core header for parity
// and documentation -- the activation is part of the "one true formula".
HD inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// ---------------------------------------------------------------------------
// Host-side declarations (defined in reference_cpu.cpp; not used on device).
// ---------------------------------------------------------------------------

// Load a volume in the data/README.md text format: "D H W" then D*H*W floats.
Volume load_volume(const std::string& path);

// Build the fixed segmentation-head weights (deterministic; no training).
SegNet make_segnet();

// CPU REFERENCE: run the full 2-layer head on `vol`, writing the integer label
// map (argmax class per voxel, 0/1) into `label` and the lesion-class logits
// into `logit1` (for the float tolerance check). label/logit1 sized to D*H*W.
// This is the trusted baseline the GPU kernels are verified against.
void segment_cpu(const Volume& vol, const SegNet& net,
                 std::vector<int>& label, std::vector<float>& logit1);

// Dice overlap between two binary masks (2*|A&B| / (|A|+|B|)). The standard
// segmentation accuracy metric; 1.0 = perfect overlap. Used to score the
// predicted mask against the known synthetic ground-truth sphere.
double dice(const std::vector<int>& a, const std::vector<int>& b);
