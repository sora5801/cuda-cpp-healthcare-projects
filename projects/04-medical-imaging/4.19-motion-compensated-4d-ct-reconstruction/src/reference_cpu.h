// ===========================================================================
// src/reference_cpu.h  --  4D-CT problem: load, ramp-filter, CPU reconstruction
// ---------------------------------------------------------------------------
// Project 4.19 : Motion-Compensated 4D-CT Reconstruction (2-D teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A "4D" (3-D + time) CT scan of a breathing chest. Projections are tagged
//   with the breathing PHASE they were acquired in and binned into P phase
//   groups; each group is severely UNDER-SAMPLED (few angles). We reconstruct a
//   single reference-phase image two ways so the learner can compare them:
//     (A) NAIVE 4D-FBP    : ramp-filter, then plain backprojection ignoring
//                           motion. All phases pile onto the same pixels ->
//                           the moving anatomy SMEARS (motion blur).
//     (B) MOTION-COMPENSATED reconstruction : the same backprojection, but each
//                           phase's ray is displaced by that phase's Deformation
//                           Vector Field (DVF) before sampling, so all phases
//                           reconstruct the SAME reference geometry -> sharp.
//   The heavy per-pixel math lives in mc4dct.h and is shared verbatim with the
//   GPU kernel (kernels.cu), so the GPU and CPU images match to the bit.
//
// WHY A GPU
//   Reconstruction is a per-PIXEL GATHER: every output pixel sums an interpolated
//   sample from every (phase, angle) projection -- and, for MCR, also evaluates
//   the DVF per pixel per phase. Pixels are independent, so one GPU thread per
//   pixel maps perfectly (PATTERNS.md: gather, exemplar 4.01). Clinical 4D-CBCT
//   is a 512^3 volume x thousands of projections x an iterated DVF -- hopeless
//   serially, routine on a GPU.
//
//   This header is pure C++ (no CUDA) so both cl.exe and nvcc can include it.
//   It re-uses the Geom struct and physics from mc4dct.h.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu; AFTER: mc4dct.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "mc4dct.h"   // Geom, mc_pixel, dvf_at, phase_motion (pure C++/HD)

// One 4D-CT reconstruction problem: the phase-binned sinogram plus geometry.
//   sino[k*n_det + j] = line integral at GLOBAL projection index k, detector j.
//   Global index k runs phase-major: k = p*n_ang_phase + a  (phase p, angle a).
//   The angles are spread over [0, pi) across ALL phases so the union is a full
//   half-turn -- interleaved so each phase individually is under-sampled.
struct FourDCTProblem {
    Geom geom{};                 // all geometry (img, n_det, n_phases, ...)
    std::vector<float> sino;     // [n_phases*n_ang_phase * n_det] raw projections
    int total_angles() const { return geom.n_phases * geom.n_ang_phase; }
};

// Load a FourDCTProblem from the text format in data/README.md:
//   header: "<img> <n_det> <n_phases> <n_ang_phase> <ds> <world_half> <amp>"
//   then (n_phases*n_ang_phase) rows of n_det floats (the sinogram), phase-major.
FourDCTProblem load_4dct(const std::string& path);

// Precompute cos/sin of EVERY global projection angle once on the host, so the
// CPU and GPU backprojections use bit-identical trig. The angle of global index
// k is theta_k = k * pi / total_angles (a uniform half-turn across all phases).
void compute_trig(const FourDCTProblem& prob,
                  std::vector<float>& cosv, std::vector<float>& sinv);

// Ram-Lak ramp filter applied to EACH projection row (the "Filtered" in FBP).
// Shared by both reconstructions so the only difference between them is motion
// compensation, not filtering. `filtered` is sized to the sinogram.
void ramp_filter(const FourDCTProblem& prob, std::vector<float>& filtered);

// CPU reference reconstruction. motion_comp = 0 -> naive 4D-FBP (blurred);
// motion_comp = 1 -> motion-compensated (sharp). Fills `image` (img*img). This
// is the trusted baseline the GPU kernel is checked against; it simply loops
// mc_pixel() from mc4dct.h over every pixel.
void reconstruct_cpu(const FourDCTProblem& prob, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     int motion_comp, std::vector<float>& image);

// Sharpness metric (secondary): mean squared gradient magnitude of the image
// (higher = crisper edges). Reported for both reconstructions as extra context.
double image_sharpness(const std::vector<float>& image, int img);

// Peak recovery metric (HEADLINE): the maximum reconstructed pixel value.
//   A moving nodule's energy is spread across phases by motion, so its NAIVE
//   reconstructed peak sits well below its true density; motion compensation
//   re-concentrates that energy, raising the peak back toward the true value.
//   Comparing the naive vs MCR peak (and the MCR peak vs the known density) is
//   the clearest, most quantitative "does motion compensation work?" number.
//   Deterministic (a max over a fixed array). Also returns the peak's (px,py).
float image_peak(const std::vector<float>& image, int img, int* peak_px, int* peak_py);
