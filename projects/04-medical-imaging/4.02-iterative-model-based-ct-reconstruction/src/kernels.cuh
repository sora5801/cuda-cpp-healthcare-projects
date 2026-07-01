// ===========================================================================
// src/kernels.cuh  --  GPU SIRT interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.2 : Iterative / Model-Based CT Reconstruction
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls sirt_gpu(), which runs the
//   ENTIRE iterative reconstruction on the device (the volume never leaves the
//   GPU between iterations -- only the tiny cos/sin tables and the sinogram are
//   uploaded once). kernels.cu implements the host driver and the four device
//   kernels below. Included only by .cu units (it names __global__ kernels).
//
// THE BIG IDEA -- iterative reconstruction is two adjoint operators in a loop
//   SIRT alternates a FORWARD projection A and a BACKprojection A^T. On the GPU
//   we express EACH as an "each output element is one thread" kernel and keep
//   all buffers resident on the device, so an outer C++ loop just re-launches
//   the kernels `iters` times (PATTERNS.md §7 -- many small launches, launch-
//   bound on tiny inputs). Two mappings appear:
//
//     * FORWARD  (forward_project_kernel): ONE THREAD PER DETECTOR BIN (ray).
//         Thread (k,j) owns sino[k,j] and SUMS the contribution of every image
//         pixel that projects onto its bin. It writes exactly one output and
//         accumulates in a private register -> deterministic, no atomics.
//         (This is the exact transpose of the pixel-gather backprojection: both
//         use interp_stencil, so A^T really is the adjoint of A.)
//
//     * BACK / UPDATE / TV : ONE THREAD PER IMAGE PIXEL.
//         backproject+update owns pixel p, gathers over all angles, and applies
//         the SIRT step; the TV kernel does one edge-preserving smoothing step.
//
//   Determinism: every kernel writes each output from a single thread that sums
//   in a fixed order -> stdout is byte-identical every run (PATTERNS.md §3). No
//   floating-point atomics anywhere.
//
// READ THIS AFTER: ct_geometry.h, util/cuda_check.cuh, util/timer.cuh.
//                  Then read kernels.cu.  reference_cpu.cpp is the CPU twin.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // CTProblem (pure C++, safe inside a .cu)

// ---- Device kernels (documented in full in kernels.cu) -------------------

// FORWARD projection A: sino[k,j] = sum over pixels of image * stencil weight.
//   grid : covers n_angles*n_det rays ; thread owns one (angle,bin) ray.
__global__ void forward_project_kernel(const float* __restrict__ image,
                                       const float* __restrict__ cosv,
                                       const float* __restrict__ sinv,
                                       int n_angles, int n_det, int N,
                                       float ds, float center, float W, float pix,
                                       float* __restrict__ sino_out);

// Row-normalized residual: resid[i] = (b[i] - sim[i]) * row_scale[i].
//   grid : covers n_angles*n_det rays ; one thread per bin (element-wise).
__global__ void residual_kernel(const float* __restrict__ b,
                                const float* __restrict__ sim,
                                const float* __restrict__ row_scale,
                                int n_rays, float* __restrict__ resid);

// BACKproject the residual AND apply the SIRT update in one pass:
//   image[p] = max(0, image[p] + lambda * col_scale[p] * (A^T resid)[p]).
//   grid : 2-D over the N x N image ; thread (px,py) owns one pixel.
__global__ void backproject_update_kernel(const float* __restrict__ resid,
                                          const float* __restrict__ cosv,
                                          const float* __restrict__ sinv,
                                          const float* __restrict__ col_scale,
                                          int n_angles, int n_det, int N,
                                          float ds, float center, float W, float pix,
                                          float lambda, float* __restrict__ image);

// One edge-preserving TOTAL-VARIATION descent step (reads img_in, writes img_out
// so there is no in-place race -- classic double-buffer / ping-pong).
//   grid : 2-D over the N x N image ; thread (px,py) owns one pixel.
__global__ void tv_step_kernel(const float* __restrict__ img_in, int N,
                               float weight, float* __restrict__ img_out);

// ---- Host driver ---------------------------------------------------------
// sirt_gpu: run the WHOLE reconstruction on the device.
//   Uploads sino + trig + SIRT weights ONCE, then loops ct.iters times launching
//   forward -> residual -> backproject/update -> (optional) TV. Copies the final
//   image back to `image` (resized to img*img) and reports the total kernel time
//   summed over all iterations via *kernel_ms. main.cu calls exactly this.
void sirt_gpu(const CTProblem& ct,
              const std::vector<float>& cosv, const std::vector<float>& sinv,
              const std::vector<float>& row_scale, const std::vector<float>& col_scale,
              std::vector<float>& image, float* kernel_ms);
