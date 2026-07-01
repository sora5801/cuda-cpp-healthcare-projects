// ===========================================================================
// src/recon_core.h  --  Shared (host + device) building blocks of one
//                       unrolled-reconstruction cascade stage
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHY THIS FILE EXISTS
//   The full project (see ../THEORY.md "Where this sits in the real world") is an
//   END-TO-END TRAINED network -- E2E-VarNet -- with millions of learned weights,
//   trained on TB of multi-coil raw k-space (fastMRI) using cuDNN + tensor cores.
//   That needs a deep-learning framework and a training pipeline; it is out of
//   scope for a single self-contained CUDA C++ demo. So we teach the STRUCTURE
//   that makes learned reconstruction work -- the UNROLLED ITERATION -- with the
//   learned pieces replaced by FIXED, transparent operators. Nothing is trained;
//   every number here is either measured or a fixed constant we can explain.
//
//   The unrolled recon repeats, for a handful of "cascade stages", exactly two
//   operations (this is the skeleton of every deep-cascade / variational net):
//     (1) REGULARIZE in the image domain  : x <- x + lambda * (D(x) - x)
//         D is a denoiser. In a real net D is a small trained CNN; here it is a
//         FIXED 3x3 convolution (a Gaussian-like smoothing "prior"). The *shape*
//         of the compute -- a per-pixel 3x3 stencil -- is exactly what cuDNN's
//         convolution layers accelerate, so the GPU pattern is faithful.
//     (2) DATA CONSISTENCY in the frequency domain : go to k-space, overwrite the
//         SAMPLED frequencies with the measured values, come back. This injects
//         the physics/measurement constraint and is what stops the denoiser from
//         hallucinating. It is the "gradient-descent-on-the-data-term" layer of a
//         variational network, in its simplest projection form.
//
//   The per-pixel stencil math of step (1) lives HERE as ONE __host__ __device__
//   function so the CPU reference (reference_cpu.cpp) and the GPU kernel
//   (kernels.cu) run BYTE-FOR-BYTE identical arithmetic -- the key to exact
//   verification (PATTERNS.md section 2). RECON_HD expands to __host__ __device__
//   under nvcc and to nothing under the plain host compiler, so this header is
//   safe to include from reference_cpu.cpp (compiled by cl.exe/g++).
//
//   Keep CUDA-only constructs OUT of this header (no __global__, no <<<>>>), so
//   the host compiler can include it.
//
// READ THIS AFTER: main.cu (the high-level unroll loop). READ BEFORE: kernels.cu
// (the GPU twin) and reference_cpu.cpp (the CPU twin).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// RECON_HD: the "host+device" decorator switch (the HD-macro idiom).
//   * Under nvcc (__CUDACC__ defined) it becomes "__host__ __device__" so the
//     SAME inline function is compiled for both the CPU and every GPU thread.
//   * Under the plain host compiler the decorators do not exist, so it expands
//     to nothing and the function is an ordinary inline C++ function.
#ifdef __CUDACC__
#define RECON_HD __host__ __device__
#else
#define RECON_HD
#endif

// ---------------------------------------------------------------------------
// The fixed 3x3 "denoiser" prior D.
//   A real learned reconstruction replaces this with a trained CNN. We use a
//   fixed, separable Gaussian-like smoothing kernel (normalized 1-2-1 outer
//   product) because:
//     * it is a genuine low-pass denoiser (removes the high-frequency aliasing
//       that under-sampling injects), so the demo actually reduces error, and
//     * its 3x3 stencil is the exact memory-access pattern of a CNN conv layer,
//       so the GPU mapping we teach transfers directly to cuDNN.
//   Weights (row-major, sum = 16):
//         1 2 1
//         2 4 2
//         1 2 1
//   These are FIXED CONSTANTS, not learned -- see the file header.
// ---------------------------------------------------------------------------
RECON_HD inline float denoise_weight(int ky, int kx) {
    // ky, kx in {-1,0,1}. |ky|+|kx| distance picks the separable 1-2-1 weights:
    //   center (0,0) -> 4 ; edge (one axis 0) -> 2 ; corner -> 1.
    const int a = (ky == 0 ? 2 : 1);   // vertical tap: 2 at center row, else 1
    const int b = (kx == 0 ? 2 : 1);   // horizontal tap: 2 at center col, else 1
    return static_cast<float>(a * b);  // outer product -> {1,2,4} pattern
}

// Sum of all 9 weights (the normalizer). Kept as a function so host and device
// agree exactly; equals 16 for the 1-2-1 outer product above.
RECON_HD inline float denoise_norm() { return 16.0f; }

// Flat row-major index of pixel (y,x) in an (ny x nx) image.
//   Row-major means consecutive x are contiguous in memory -> when GPU threads
//   in a warp cover consecutive x, their global reads coalesce.
RECON_HD inline std::size_t img_idx(int y, int x, int nx) {
    return static_cast<std::size_t>(y) * nx + x;
}

// ---------------------------------------------------------------------------
// denoise_pixel: compute D(x) at ONE output pixel (y,x) -- the shared stencil.
//   This is the whole per-pixel physics of step (1). The CPU reference loops it
//   over all pixels; the GPU kernel runs one thread per pixel. Because both call
//   THIS function, their results match to the bit (modulo FMA reassociation,
//   which we bound with the documented tolerance -- see THEORY "verify").
//
//   Boundary handling: CLAMP (replicate the edge pixel). Out-of-range neighbours
//   reuse the nearest in-range pixel, which avoids darkening the border the way
//   zero-padding would. This choice is identical on host and device.
//
//   Parameters:
//     img : [ny*nx] input image (row-major), read-only.
//     y,x : the output pixel this call computes (0 <= y < ny, 0 <= x < nx).
//     ny,nx : image dimensions in pixels.
//   Returns: the denoised value at (y,x) (a weighted average of its 3x3 window).
// ---------------------------------------------------------------------------
RECON_HD inline float denoise_pixel(const float* img, int y, int x, int ny, int nx) {
    float acc = 0.0f;                       // running weighted sum of neighbours
    // Walk the 3x3 window centered on (y,x). ky,kx are neighbour OFFSETS.
    for (int ky = -1; ky <= 1; ++ky) {
        // Clamp the sampled row into [0, ny-1] (replicate-edge boundary).
        int yy = y + ky;
        if (yy < 0) yy = 0; else if (yy >= ny) yy = ny - 1;
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = x + kx;
            if (xx < 0) xx = 0; else if (xx >= nx) xx = nx - 1;
            // Weighted contribution of this neighbour to the smoothed output.
            acc += denoise_weight(ky, kx) * img[img_idx(yy, xx, nx)];
        }
    }
    return acc / denoise_norm();            // normalize so a flat image is unchanged
}

// ---------------------------------------------------------------------------
// regularize_pixel: one full REGULARIZATION update at pixel (y,x):
//     x_out = x_in + lambda * (D(x_in) - x_in)
//   This is a damped step toward the denoised image (a proximal-gradient move on
//   the regularizer). lambda in [0,1] controls how strongly each stage smooths;
//   a trained net would LEARN this step. We keep it fixed and documented.
//   `img` is the current image; `self` is img[(y,x)] passed in to avoid a second
//   index computation. Runs identically on host and device.
// ---------------------------------------------------------------------------
RECON_HD inline float regularize_pixel(const float* img, float self,
                                       int y, int x, int ny, int nx, float lambda) {
    const float d = denoise_pixel(img, y, x, ny, nx);   // D(x) at this pixel
    return self + lambda * (d - self);                  // damped move toward D(x)
}
