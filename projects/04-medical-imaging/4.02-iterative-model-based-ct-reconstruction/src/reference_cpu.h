// ===========================================================================
// src/reference_cpu.h  --  CT problem, geometry loader, serial SIRT reference
// ---------------------------------------------------------------------------
// Project 4.2 : Iterative / Model-Based CT Reconstruction
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a 2-D image x from a noisy X-ray sinogram b, NOT by a one-shot
//   analytic inversion (that is Project 4.01, Filtered BackProjection) but by
//   ITERATION. We start from a blank image and repeatedly:
//       1. FORWARD-project the current estimate:  Ax   (what it *would* measure)
//       2. compare to the real data:              r = b - Ax   (the residual)
//       3. BACKproject the residual, normalized:  x += lambda * C A^T R r
//   That is SIRT (Simultaneous Iterative Reconstruction Technique) -- a
//   preconditioned gradient descent on the least-squares data-fit ||Ax-b||^2.
//   Optionally we add a small TOTAL-VARIATION (TV) step each iteration, which
//   nudges the image toward being piecewise-smooth. TV is the "model-based"
//   prior: it encodes our belief that tissue is mostly uniform with sharp
//   edges, and it is what lets iterative reconstruction beat FBP at low dose
//   (30-50% less noise at matched dose -- see the catalog deep dive / THEORY).
//
// WHY A GPU
//   Each SIRT iteration is ONE forward-projection + ONE backprojection -- the
//   exact same gather/scatter that makes FBP GPU-bound -- but we repeat it
//   20-200 times. Clinical volumes (512^3 voxels x ~1000 views) make this
//   hopeless on a CPU and routine on a GPU. We do the small 2-D case so every
//   step is legible; the GPU/CPU speed-up is the teaching artifact.
//
// THIS HEADER is pure C++ (no CUDA). kernels.cu reuses CTProblem and the
// per-ray geometry from ct_geometry.h.  Read ct_geometry.h first.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// One reconstruction problem: the measured sinogram plus all geometry.
//   sino[k*n_det + j] = measured line integral at angle k, detector bin j.
//   Angles are uniform over [0,pi):  theta_k = k * pi / n_angles.
//   Detector bin j sits at signed offset  s_j = (j - (n_det-1)/2) * ds.
//   The reconstructed image is `img` x `img` pixels over [-world_half,+world_half]^2.
//   (Optional) truth[] holds the SYNTHETIC ground-truth image, present only for
//   synthetic samples so we can also report reconstruction error vs. truth --
//   a *scientific* check on top of the CPU-vs-GPU check (PATTERNS.md §4).
struct CTProblem {
    int   n_angles   = 0;     // number of projection angles
    int   n_det      = 0;     // detector bins per projection
    int   img        = 0;     // reconstructed image side length (pixels)
    float ds         = 0.0f;  // detector bin spacing (world units)
    float world_half = 0.0f;  // image spans [-world_half,+world_half] in x and y
    int   iters      = 0;     // number of SIRT iterations to run (from the file)
    float lambda     = 0.0f;  // SIRT relaxation / step size (0<lambda<=1)
    float tv_weight  = 0.0f;  // strength of the TV smoothing step (0 = pure SIRT)
    std::vector<float> sino;   // [n_angles * n_det] measured projections
    std::vector<float> truth;  // [img*img] ground truth if provided (else empty)
};

// Load a CTProblem from the text format documented in data/README.md:
//   line 1 (header): "n_angles n_det ds img world_half iters lambda tv_weight has_truth"
//   then n_angles rows of n_det floats  (the measured sinogram)
//   then (if has_truth==1) img rows of img floats (the ground-truth image).
CTProblem load_ct(const std::string& path);

// Precompute cos/sin of every projection angle ONCE on the host, so the CPU and
// GPU projectors use bit-identical trig (avoids cos vs cosf drift over many
// iterations). cosv,sinv are sized to n_angles.
void compute_trig(int n_angles, std::vector<float>& cosv, std::vector<float>& sinv);

// Precompute the SIRT normalization diagonals R and C ("SIRT weights"):
//   row_scale[k*n_det+j] = 1 / (row sum of A)  -- one per detector bin (ray)
//   col_scale[py*img+px] = 1 / (col sum of A)  -- one per pixel
// These make SIRT a well-scaled fixed-point iteration (each residual bin and
// each pixel update is divided by how many things touch it). Computed once and
// shared by both the CPU reference and the GPU. See THEORY.md §algorithm.
void compute_sirt_weights(const CTProblem& ct,
                          const std::vector<float>& cosv, const std::vector<float>& sinv,
                          std::vector<float>& row_scale, std::vector<float>& col_scale);

// FORWARD projection A: given an image, simulate the sinogram it would produce.
//   sino_out[k*n_det+j] = sum over pixels on ray (k,j) of image, via the shared
//   interpolation stencil. Sized to n_angles*n_det. The serial baseline for the
//   GPU forward kernel.
void forward_project_cpu(const CTProblem& ct, const std::vector<float>& image,
                         const std::vector<float>& cosv, const std::vector<float>& sinv,
                         std::vector<float>& sino_out);

// BACKprojection A^T: smear a sinogram-shaped array back into image space.
//   image_out[py*img+px] = sum over angles of interp(sino_row_k, s). The adjoint
//   of forward_project_cpu (same stencil). Sized to img*img.
void backproject_cpu(const CTProblem& ct, const std::vector<float>& sino,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& image_out);

// The full serial SIRT (+ optional TV) reconstruction -- the TRUSTED answer the
// GPU is checked against. Runs ct.iters iterations from a zero image and writes
// the final reconstruction into `image` (sized img*img).
void reconstruct_sirt_cpu(const CTProblem& ct,
                          const std::vector<float>& cosv, const std::vector<float>& sinv,
                          const std::vector<float>& row_scale, const std::vector<float>& col_scale,
                          std::vector<float>& image);

// Root-mean-square error between two equal-length images (a scalar quality
// number we can print). Used to compare a reconstruction to the ground truth.
double rms_error(const std::vector<float>& a, const std::vector<float>& b);
