// ===========================================================================
// src/reference_cpu.h  --  DBT problem struct + CPU SART reference (the "why")
// ---------------------------------------------------------------------------
// Project 4.14 : Digital Breast Tomosynthesis
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a 2-D attenuation image of a (simplified) compressed breast
//   slice from a SMALL number of X-ray projections taken over a NARROW angular
//   range -- the defining feature of Digital Breast Tomosynthesis (DBT). Because
//   the angular range is limited (~+/-25 deg here), analytical Filtered
//   BackProjection (project 4.01) is unstable, so we use an ITERATIVE algebraic
//   method: SART (Simultaneous Algebraic Reconstruction Technique).
//
//   SART repeats, for a fixed number of sweeps:
//     1. FORWARD-PROJECT the current image estimate -> simulated projections.
//     2. Compute the RESIDUAL = measured projections - simulated projections.
//     3. BACKPROJECT the (normalised) residual and add a relaxed correction to
//        the image estimate. Negatives are clamped (attenuation >= 0).
//   Each sweep drives the estimate's projections closer to what was measured.
//
// WHY A GPU (the catalog "why")
//   Both the forward projection (one line integral per detector ray, sampling
//   the image along the ray) and the backprojection (one correction per pixel,
//   gathering from every angle) are massively parallel *gathers* with no data
//   dependence between output elements. A clinical DBT volume is ~800x700x60 at
//   85 um from ~9-25 projections -- billions of ray-voxel interactions per
//   iteration, and SART runs many iterations. That is hopeless serially and
//   bandwidth-bound-but-fast on a GPU. This teaching build does the 2-D slice so
//   the geometry is legible; THEORY.md explains the 3-D cone-beam extension.
//
//   This header is PURE C++ (no CUDA). kernels.cu reuses DBTProblem and the
//   shared per-ray math in dbt_geometry.h so CPU and GPU stay in lockstep.
//
// READ THIS AFTER: dbt_geometry.h (the shared ray formula). Then kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// DBTProblem: one limited-angle reconstruction problem -- the measured
// projections plus every geometric constant needed to invert them.
//
//   proj[k*n_det + j] = measured line integral at projection angle k, bin j.
//   Angles are a NARROW symmetric wedge: theta_k spans [-half_span, +half_span]
//   radians in n_angles uniform steps (this is what makes it tomosynthesis, not
//   full CT). Detector bin j sits at signed offset s_j = (j-(n_det-1)/2)*ds.
//   The reconstructed image is `img` x `img` pixels over world [-W, W]^2.
// ---------------------------------------------------------------------------
struct DBTProblem {
    int   n_angles   = 0;     // number of projection angles (DBT: ~9-25)
    int   n_det      = 0;     // detector bins per projection
    int   img        = 0;     // reconstructed image side length (pixels)
    int   n_iters    = 0;     // SART sweeps to run (fixed -> deterministic)
    float ds         = 0.0f;  // detector bin spacing (world units)
    float world_half = 0.0f;  // image spans [-world_half, world_half] in x and y
    float half_span  = 0.0f;  // half the angular wedge, RADIANS (e.g. 0.436 = 25 deg)
    float relax      = 0.0f;  // SART relaxation factor lambda (0<lambda<=1)
    std::vector<float> proj;  // [n_angles * n_det] measured projections
};

// ---------------------------------------------------------------------------
// load_dbt: read a DBTProblem from the text format documented in data/README.md
//   header: "<n_angles> <n_det> <ds> <img> <world_half> <half_span> <relax> <n_iters>"
//   then n_angles rows of n_det floats (the measured projections).
// Throws std::runtime_error on a missing file or malformed/short data so the
// demo fails loudly instead of reconstructing garbage.
// ---------------------------------------------------------------------------
DBTProblem load_dbt(const std::string& path);

// ---------------------------------------------------------------------------
// compute_angles: precompute cos/sin of every projection angle ONCE on the host
// so the CPU reference and GPU kernels use bit-identical trig (avoids the
// cos-vs-cosf disagreement that would otherwise break exact verification).
// Angles are the uniform wedge theta_k = -half_span + k*(2*half_span/(n_angles-1)).
// ---------------------------------------------------------------------------
void compute_angles(const DBTProblem& p, std::vector<float>& cosv, std::vector<float>& sinv);

// ---------------------------------------------------------------------------
// n_ray_steps: the number of samples taken ALONG each ray during forward
// projection. Shared by CPU and GPU so both integrate the ray identically.
// Chosen ~ 2*img so the ray is sampled at roughly sub-pixel spacing (Nyquist-ish
// for the image grid). Kept as a free function so there is a single definition.
// ---------------------------------------------------------------------------
int n_ray_steps(const DBTProblem& p);

// ---------------------------------------------------------------------------
// reconstruct_sart_cpu: the trusted serial baseline the GPU is checked against.
//   Runs p.n_iters SART sweeps and writes the final img*img attenuation image.
//   Deterministic: fixed iteration count, fixed ray sampling, no randomness.
//   `image` is resized to img*img. This is the reference for verification.
// ---------------------------------------------------------------------------
void reconstruct_sart_cpu(const DBTProblem& p,
                          const std::vector<float>& cosv,
                          const std::vector<float>& sinv,
                          std::vector<float>& image);
