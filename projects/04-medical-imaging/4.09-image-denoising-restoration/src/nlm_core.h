// ===========================================================================
// src/nlm_core.h  --  The ONE shared per-pixel Non-Local-Means math core
// ---------------------------------------------------------------------------
// Project 4.9 : Image Denoising & Restoration  (Non-Local Means, NLM)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2 -- the __host__ __device__ idiom)
//   This file holds the *per-output-pixel physics* of Non-Local Means denoising
//   as a handful of `NLM_HD inline` functions. It is compiled TWICE:
//     * by the host C++ compiler when reference_cpu.cpp includes it, and
//     * by nvcc when kernels.cu includes it (where NLM_HD expands to
//       `__host__ __device__`, so the SAME code runs on the GPU).
//   Because both sides execute byte-for-byte-identical float arithmetic *in the
//   same order*, the CPU reference and the GPU kernel agree to a very tight
//   tolerance -- verification becomes near-exact instead of hand-wavy.
//
//   Keep this header CUDA-*type*-free: no `__global__`, no `dim3`, no
//   `cudaMalloc`. Only plain scalars and raw pointers, so the host compiler is
//   perfectly happy to compile it. (The kernel launch + device memory live in
//   kernels.cu; the serial driver loop lives in reference_cpu.cpp.)
//
// WHAT NON-LOCAL MEANS COMPUTES  (see THEORY.md "The math")
//   Classic local filters (Gaussian, bilateral) average a pixel with its spatial
//   neighbours and therefore blur edges and fine texture. Non-Local Means
//   (Buades, Coll & Morel, 2005) instead says: *a pixel's best evidence for its
//   true value is other pixels whose SURROUNDING PATCH looks like this pixel's
//   patch*, no matter where in the image they sit. Two pixels in different parts
//   of a CT slice that both sit on the same tissue boundary are "neighbours" in
//   patch-space even if they are far apart in (x,y).
//
//   For output pixel p we compute a weighted average of every candidate pixel q
//   inside a search window around p:
//
//       out(p) = ( Σ_q  w(p,q) * in(q) )  /  ( Σ_q  w(p,q) )
//
//   where the weight compares the PATCHES (small square neighbourhoods) centred
//   at p and q:
//
//       d2(p,q) = (1/|patch|) * Σ_{k in patch}  ( in(p+k) - in(q+k) )^2
//       w(p,q)  = exp( -max(d2(p,q) - 2σ², 0) / h² )
//
//   * d2 is the mean squared per-pixel difference of the two patches (units:
//     intensity²). Similar patches -> small d2 -> weight near 1. Dissimilar
//     patches -> large d2 -> weight near 0, so unrelated pixels barely count.
//   * σ is the noise standard deviation. Subtracting 2σ² is the standard NLM
//     bias correction: even two patches of the *same* underlying signal differ
//     by ~2σ² purely because of noise, so we discount that expected amount
//     before penalising. We clamp at 0 so the argument to exp is never positive.
//   * h is the filtering strength (larger h -> flatter weights -> more smoothing;
//     a common choice is h ≈ k·σ). It sets how quickly the weight decays with
//     patch distance.
//
// THREAD-TO-DATA MAPPING (used identically by CPU loop and GPU kernel)
//   Each OUTPUT pixel is fully independent -- it only READS the noisy input and
//   writes its own result -- so on the GPU one thread owns one output pixel
//   (kernels.cu), and on the CPU a double loop owns one output pixel at a time
//   (reference_cpu.cpp). Both call nlm_pixel() below, so the arithmetic matches.
//
// READ THIS AFTER: THEORY.md (the derivation); BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::sqrt, std::fabs  (host side)

// --- The HD (host+device) decorator switch (PATTERNS.md §2) -----------------
// When this header is fed to nvcc, __CUDACC__ is defined and NLM_HD becomes the
// CUDA decorators, so every function below is emitted for BOTH the host and the
// device. When the plain host compiler sees it, the decorators do not exist, so
// NLM_HD expands to nothing and the functions are ordinary inline C++.
#ifdef __CUDACC__
#define NLM_HD __host__ __device__
#else
#define NLM_HD
#endif

// ---------------------------------------------------------------------------
// NlmParams: every scalar the NLM math needs, in ONE struct.
//   Bundling them keeps the CPU call and the kernel launch in lockstep (change
//   a field once, both sides see it) and makes the code read like the equations
//   above. All sizes are in PIXELS; intensities are in normalized [0,1] units.
// ---------------------------------------------------------------------------
struct NlmParams {
    int   width       = 0;      // image width  in pixels (columns)
    int   height      = 0;      // image height in pixels (rows)
    int   patch_radius = 0;     // patch is (2*patch_radius+1)^2 pixels (e.g. r=2 -> 5x5)
    int   search_radius = 0;    // search window is (2*search_radius+1)^2 candidates (e.g. r=5 -> 11x11)
    float sigma       = 0.0f;   // noise std-dev (same [0,1] intensity units as the image)
    float h           = 0.0f;   // filter strength: weight = exp(-max(d2-2σ²,0)/h²)
};

// ---------------------------------------------------------------------------
// clamp_index: reflect an out-of-range coordinate back inside [0, n).
//   Patches near the border reach past the image edge. Rather than special-case
//   the borders (which would make the CPU and GPU code diverge), we MIRROR:
//   index -1 maps to 1, index n maps to n-2, etc. Mirroring is smooth (no hard
//   seam) and, crucially, is written ONCE here so both sides border-handle the
//   same way -> their results stay identical.
//     n : axis length (width or height), n >= 1
//   returns a valid index in [0, n-1]
// ---------------------------------------------------------------------------
NLM_HD inline int clamp_index(int idx, int n) {
    // Fold negatives across 0 and overshoots across n-1. One reflection step is
    // enough here because patch_radius is always < n for our tiny sample.
    if (idx < 0)      idx = -idx;            // -1 -> 1, -2 -> 2, ...
    if (idx >= n)     idx = 2 * (n - 1) - idx; // n -> n-2, n+1 -> n-3, ...
    // Guard the pathological tiny-image case so we never index out of bounds.
    if (idx < 0)      idx = 0;
    if (idx >= n)     idx = n - 1;
    return idx;
}

// ---------------------------------------------------------------------------
// pixel_at: read input(row, col) with mirrored borders.
//   img : row-major [height*width] noisy input, intensities in [0,1]
//   Using this everywhere means every patch read is border-safe and identical
//   on host and device. Returns the (float) intensity.
// ---------------------------------------------------------------------------
NLM_HD inline float pixel_at(const float* img, const NlmParams& p, int row, int col) {
    const int r = clamp_index(row, p.height);
    const int c = clamp_index(col, p.width);
    return img[(size_t)r * p.width + c];      // row-major flatten: idx = row*W + col
}

// ---------------------------------------------------------------------------
// patch_distance2: mean squared difference between the patch centred at (pr,pc)
//   and the patch centred at (qr,qc). This is the O(patch²) inner cost that the
//   GPU parallelises across the many candidate pixels q.
//     returns d2 in intensity² units (>= 0). Identical patches -> 0.
// ---------------------------------------------------------------------------
NLM_HD inline float patch_distance2(const float* img, const NlmParams& p,
                                    int pr, int pc, int qr, int qc) {
    float acc = 0.0f;                          // running sum of squared diffs
    int   count = 0;                           // number of patch pixels summed
    const int R = p.patch_radius;
    // Walk the square patch offset (dr,dc) around each centre in lockstep.
    for (int dr = -R; dr <= R; ++dr) {
        for (int dc = -R; dc <= R; ++dc) {
            const float a = pixel_at(img, p, pr + dr, pc + dc); // patch-p sample
            const float b = pixel_at(img, p, qr + dr, qc + dc); // patch-q sample
            const float diff = a - b;
            acc += diff * diff;               // squared photometric difference
            ++count;
        }
    }
    // Mean over the patch so d2 does not grow with patch size (keeps h tuning
    // independent of patch_radius). count == (2R+1)^2, never zero.
    return acc / (float)count;
}

// ---------------------------------------------------------------------------
// nlm_weight: turn a patch distance into a similarity weight.
//   w = exp( -max(d2 - 2σ², 0) / h² ).  See the header math above for the
//   meaning of the 2σ² noise-bias subtraction and the h² decay scale.
//     returns a weight in (0, 1].
// ---------------------------------------------------------------------------
NLM_HD inline float nlm_weight(float d2, const NlmParams& p) {
    // Discount the squared distance we EXPECT from noise alone (2σ²); clamp so
    // the exponent argument is never positive (which would give w > 1).
    float adj = d2 - 2.0f * p.sigma * p.sigma;
    if (adj < 0.0f) adj = 0.0f;
    const float h2 = p.h * p.h;               // decay scale (h² in the exponent)
    // expf/exp: nvcc picks the device expf for float on the GPU; on the host the
    // <cmath> std::exp(float) overload is used. Both are IEEE-correct here.
    return std::exp(-adj / h2);
}

// ---------------------------------------------------------------------------
// nlm_pixel: THE per-output-pixel computation -- the heart shared by CPU & GPU.
//   Computes out(pr,pc) = Σ_q w(p,q) in(q) / Σ_q w(p,q) over the search window.
//   This function is what one GPU thread runs (kernels.cu) and what the CPU
//   reference calls in its double loop (reference_cpu.cpp) -> exact parity.
//     img : row-major noisy input [height*width], intensities in [0,1]
//     p   : all NLM parameters
//     pr,pc : the output pixel this call is responsible for
//   returns the denoised intensity for (pr,pc).
// ---------------------------------------------------------------------------
NLM_HD inline float nlm_pixel(const float* img, const NlmParams& p, int pr, int pc) {
    float weight_sum = 0.0f;   // Σ_q w(p,q)          -- the normaliser (denominator)
    float value_sum  = 0.0f;   // Σ_q w(p,q)*in(q)    -- the weighted intensity (numerator)
    const int S = p.search_radius;

    // Sweep every candidate pixel q = (qr,qc) in the square search window around
    // the output pixel. Each q contributes its intensity, weighted by how similar
    // its patch is to the output pixel's patch.
    for (int sr = -S; sr <= S; ++sr) {
        for (int sc = -S; sc <= S; ++sc) {
            const int qr = pr + sr;           // candidate row
            const int qc = pc + sc;           // candidate col
            const float d2 = patch_distance2(img, p, pr, pc, qr, qc);
            const float w  = nlm_weight(d2, p);
            weight_sum += w;
            value_sum  += w * pixel_at(img, p, qr, qc);
        }
    }
    // The centre pixel (q == p) always has d2 == 0 -> w == 1, so weight_sum is
    // strictly > 0 and this division is always safe. Fall back defensively.
    return (weight_sum > 0.0f) ? (value_sum / weight_sum) : pixel_at(img, p, pr, pc);
}
