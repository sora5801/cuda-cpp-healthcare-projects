// ===========================================================================
// src/reference_cpu.cpp  --  Volume loader, fixed SegNet weights, serial
//                            2-layer segmentation forward pass, Dice metric
// ---------------------------------------------------------------------------
// Project 4.7 : Medical Image Segmentation (Deep Learning)   [REDUCED SCOPE]
//
// ROLE IN THE PROJECT
//   The trusted, obviously-correct CPU baseline. It runs the SAME two-layer
//   fully-convolutional head as the GPU (sharing conv3x3x3_at / relu from
//   reference_cpu.h), so when the GPU label map matches this one exactly we
//   believe the GPU. Compiled by the host C++ compiler only (no CUDA syntax).
//
//   READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_volume: read the data/sample text format -> a Volume.
//   Format (see data/README.md):  D H W   then D*H*W float intensities.
//   Throws std::runtime_error on any problem so demos fail loudly instead of
//   silently segmenting empty/garbage input.
// ---------------------------------------------------------------------------
Volume load_volume(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);
    Volume vol;
    if (!(in >> vol.D >> vol.H >> vol.W) || vol.D <= 0 || vol.H <= 0 || vol.W <= 0)
        throw std::runtime_error("bad header (expected 'D H W') in " + path);
    const long long n = vol.size();
    vol.v.resize(static_cast<std::size_t>(n));
    for (long long i = 0; i < n; ++i)
        if (!(in >> vol.v[static_cast<std::size_t>(i)]))
            throw std::runtime_error("volume truncated in " + path);
    return vol;
}

// ---------------------------------------------------------------------------
// gaussian27: fill a 27-tap (3x3x3) separable Gaussian smoothing stencil,
//   normalized to sum 1, into w[0..26] in the (dz,dy,dx) order conv3x3x3_at
//   walks. A small sigma blurs the high-frequency noise out of the volume so
//   the second layer sees a clean blob -- this is the "denoise" feature map.
// ---------------------------------------------------------------------------
static void gaussian27(float* w, double sigma) {
    double sum = 0.0;
    int t = 0;
    for (int dz = -KR; dz <= KR; ++dz)
        for (int dy = -KR; dy <= KR; ++dy)
            for (int dx = -KR; dx <= KR; ++dx, ++t) {
                const double r2 = double(dz*dz + dy*dy + dx*dx);
                w[t] = static_cast<float>(std::exp(-0.5 * r2 / (sigma * sigma)));
                sum += w[t];
            }
    for (int k = 0; k < KVOL; ++k) w[k] = static_cast<float>(w[k] / sum);
}

// Lesion intensity threshold. The synthetic lesion is bright (smoothed intensity
// ~0.5..1.0) against ~0.3 tissue (see data/README.md / make_synthetic.py); a
// 3x3x3 box-average above this bar marks a voxel as lesion. 0.46 maximizes Dice
// on the committed sample (~0.96). Tuning it is an Exercise (README §Exercises).
static constexpr float LESION_THRESHOLD = 0.46f;

// ---------------------------------------------------------------------------
// make_segnet: build the FIXED weights of the teaching segmentation head.
//   No training, no RNG -- the weights are designed by hand so the forward pass
//   is a reproducible, interpretable "bright-lesion detector". It is a hand-set
//   instance of the SAME architecture a trained 3D U-Net learns; here the two
//   conv layers implement an explicit denoise-then-threshold rule:
//
//   LAYER 1 (1 input channel -> C_HID=2 hidden channels):
//     * channel 0 = Gaussian smoother (sigma 0.9): denoises the intensity so the
//       noise added in make_synthetic.py does not flip individual voxels. This
//       is the channel layer 2 actually segments on.
//     * channel 1 = identity (center tap = 1): passes the raw intensity through.
//       Unused by the current detector, but present to make the multi-channel
//       conv real and to leave room for the learner's experiments (Exercises).
//     Both biases 0; a ReLU follows (intensities are >=0, so ReLU is a near
//     no-op here, but it is the genuine activation the GPU must also apply, so
//     CPU/GPU stay in lockstep).
//
//   LAYER 2 (C_HID=2 input channels -> N_CLASS=2 output logits):
//     * class 1 (lesion) = a 3x3x3 BOX AVERAGE (uniform 1/27 weights) over the
//       SMOOTHED channel 0, minus the lesion threshold as a bias. So the logit
//       is  (mean intensity in the 27-voxel neighbourhood) - LESION_THRESHOLD,
//       which is positive exactly where a bright region sits. Averaging 27
//       voxels rejects residual noise -- a genuine use of the 3D neighbourhood.
//     * class 0 (background) = all-zero weights and zero bias, so its logit is 0
//       everywhere. argmax then labels a voxel "lesion" iff the class-1 logit
//       exceeds 0, i.e. iff the local mean intensity exceeds the threshold.
//     argmax(class0=0, class1) yields the 0/1 mask.
// ---------------------------------------------------------------------------
SegNet make_segnet() {
    SegNet net;
    net.w1.assign(static_cast<std::size_t>(C_HID) * 1 * KVOL, 0.0f);
    net.b1.assign(C_HID, 0.0f);
    net.w2.assign(static_cast<std::size_t>(N_CLASS) * C_HID * KVOL, 0.0f);
    net.b2.assign(N_CLASS, 0.0f);

    // --- Layer 1, channel 0: Gaussian smoother (denoise) ---
    gaussian27(&net.w1[0 * KVOL], /*sigma=*/0.9);
    // --- Layer 1, channel 1: identity (center tap = 1) ---
    //   The center tap of a 3x3x3 stencil is index 13 (dz=dy=dx=0 -> 1*9+1*3+1).
    net.w1[1 * KVOL + 13] = 1.0f;

    // --- Layer 2, class 1 (lesion): box average over the SMOOTHED channel 0 ---
    //   Uniform 1/27 weights on channel 0 (the smoothed map); zero on channel 1.
    //   logit = mean_{27}(smoothed) + bias, with bias = -threshold so the logit
    //   crosses 0 exactly at the intensity boundary of the bright lesion.
    for (int k = 0; k < KVOL; ++k)
        net.w2[(1 * C_HID + 0) * KVOL + k] = 1.0f / static_cast<float>(KVOL);  // class1 <- chan0
    // (channel 1 weights for class 1 stay 0 -> identity channel is ignored here)
    net.b2[1] = -LESION_THRESHOLD;

    // --- Layer 2, class 0 (background): all-zero filter -> constant logit 0 ---
    //   Background is the default; a voxel is lesion only when class-1 beats 0.
    net.b2[0] = 0.0f;

    return net;
}

// ---------------------------------------------------------------------------
// segment_cpu: the full serial forward pass (the reference the GPU is checked
//   against). Two convolution layers + ReLU + per-voxel argmax.
//   Steps:
//     (1) layer 1: for each hidden channel, conv the 1-channel input -> ReLU,
//         producing hidden[C_HID * D*H*W].
//     (2) layer 2: for each class, conv the C_HID-channel hidden -> logits.
//     (3) argmax over the N_CLASS logits -> integer label per voxel; record the
//         lesion-class logit for the float-tolerance check.
//   Complexity: O(D*H*W * (C_HID*27 + N_CLASS*C_HID*27)) FMAs -- linear in voxels.
// ---------------------------------------------------------------------------
void segment_cpu(const Volume& vol, const SegNet& net,
                 std::vector<int>& label, std::vector<float>& logit1) {
    const int D = vol.D, H = vol.H, W = vol.W;
    const long long n = vol.size();

    // (1) Layer 1 -> hidden feature maps, with ReLU.
    std::vector<float> hidden(static_cast<std::size_t>(C_HID) * n, 0.0f);
    for (int co = 0; co < C_HID; ++co) {
        const float* w = &net.w1[static_cast<std::size_t>(co) * 1 * KVOL];
        const float  b = net.b1[co];
        for (int z = 0; z < D; ++z)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    // 1 input channel for layer 1 (the raw intensity volume).
                    float a = conv3x3x3_at(vol.v.data(), 1, D, H, W, w, b, z, y, x);
                    hidden[static_cast<std::size_t>(co) * n + vol.idx(z, y, x)] = relu(a);
                }
    }

    // (2) Layer 2 -> per-class logits, then (3) argmax -> label map.
    label.assign(static_cast<std::size_t>(n), 0);
    logit1.assign(static_cast<std::size_t>(n), 0.0f);
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int vi = vol.idx(z, y, x);
                int   best_c = 0;
                float best_z = -1e30f;       // running max logit
                float lesion = 0.0f;         // remember class-1 logit
                for (int cls = 0; cls < N_CLASS; ++cls) {
                    const float* w = &net.w2[static_cast<std::size_t>(cls) * C_HID * KVOL];
                    const float  b = net.b2[cls];
                    const float zl = conv3x3x3_at(hidden.data(), C_HID, D, H, W, w, b, z, y, x);
                    if (cls == 1) lesion = zl;
                    // Deterministic argmax: strict '>' so ties keep the lower
                    // class index (matches the GPU kernel exactly).
                    if (zl > best_z) { best_z = zl; best_c = cls; }
                }
                label[static_cast<std::size_t>(vi)]  = best_c;
                logit1[static_cast<std::size_t>(vi)] = lesion;
            }
}

// ---------------------------------------------------------------------------
// dice: Dice similarity coefficient between two binary (0/1) masks of equal
//   length:  Dice = 2|A&B| / (|A| + |B|). Both intersection and the two set
//   sizes are INTEGER counts, so this metric is computed exactly and is the
//   same on any machine -- it is safe to print to deterministic stdout.
//   Returns 1.0 for two empty masks (perfect trivial agreement).
// ---------------------------------------------------------------------------
double dice(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return 0.0;
    long long inter = 0, sa = 0, sb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int ai = a[i] != 0, bi = b[i] != 0;
        inter += (ai & bi);
        sa += ai;
        sb += bi;
    }
    if (sa + sb == 0) return 1.0;
    return (2.0 * static_cast<double>(inter)) / static_cast<double>(sa + sb);
}
