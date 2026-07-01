// ===========================================================================
// src/reference_cpu.h  --  PET data model + serial MLEM reference
// ---------------------------------------------------------------------------
// Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a 2-D tracer-concentration image x from noisy coincidence COUNTS
//   y (a sinogram) by MAXIMUM-LIKELIHOOD EXPECTATION-MAXIMIZATION (MLEM). PET
//   counts are Poisson-distributed, so we do NOT invert with a linear filter
//   (that is FBP, project 4.01); instead we iteratively find the image whose
//   forward projection best explains the observed counts under the Poisson
//   likelihood. The classic multiplicative MLEM update (Shepp-Vardi 1982) is:
//
//       x_j^{n+1} = ( x_j^n / s_j ) * SUM_i A_ij * ( y_i / (A x^n)_i )
//
//   where A is the system matrix (pixel j -> LOR i), (A x)_i is the forward
//   projection (expected counts on LOR i), y_i the measured counts, and
//   s_j = SUM_i A_ij the SENSITIVITY of pixel j (its total detection
//   probability). Each iteration is: forward project, form the ratio y/(Ax),
//   back-project the ratio, multiply by x/s. See THEORY.md for the derivation.
//
// WHY A GPU  (docs/PATTERNS.md: the projection GATHER pattern, like 4.01)
//   Every iteration needs a full forward projection (one sum per LOR over the
//   pixels on its ray) AND a full back projection (one sum per pixel over all
//   LORs through it). Clinical scanners have ~10^9 sinogram elements and run tens
//   of iterations -> billions of ray/pixel touches per reconstruction. Both
//   projections are GATHERS of independent outputs, so each maps onto one GPU
//   thread with no atomics. That is exactly what kernels.cu does.
//
//   This header is PURE C++ (no CUDA) so the host compiler can build the
//   reference. The projection GEOMETRY it uses lives in the shared HD header
//   pet_geometry.h, which kernels.cu also includes -> CPU and GPU agree.
//
// READ THIS AFTER: pet_geometry.h.  READ BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pet_geometry.h"   // PetGeom, make_geom, projection helpers (HD, pure here)

// ---------------------------------------------------------------------------
// PetProblem: one reconstruction problem -- the geometry, the measured counts,
//   and the host-precomputed trig tables.
//     geom          : image/sinogram dimensions + spacing (see pet_geometry.h)
//     counts[k*D+j] : measured coincidence counts on LOR (angle k, bin j) = y_i
//     cosv[k],sinv[k]: cos/sin of angle theta_k, precomputed ONCE so CPU and GPU
//                     use identical trig (avoids cos vs cosf last-bit drift).
//   The image x (N*N floats) is not stored here; it is produced by the solver.
// ---------------------------------------------------------------------------
struct PetProblem {
    PetGeom            geom;    // dimensions + spacing
    std::vector<float> counts;  // [K*D] measured sinogram y (nonneg counts)
    std::vector<float> cosv;    // [K] cos(theta_k)
    std::vector<float> sinv;    // [K] sin(theta_k)
};

// Load a PetProblem from the text format documented in data/README.md:
//   header : "<K> <D> <ds> <N> <W> <iters>"   (iters is advisory; see main.cu)
//   then     K rows of D floats -> the measured sinogram (counts).
// Throws std::runtime_error on any malformed input so demos fail loudly.
// The advisory iteration count is written through `iters_out` (0 if absent).
PetProblem load_pet(const std::string& path, int& iters_out);

// Precompute cos/sin of every projection angle theta_k = k*pi/K into the problem.
//   Done once on the host; both projectors read these tables (CPU/GPU parity).
void compute_trig(PetProblem& p);

// forward_project_cpu: y_hat = A x. For each LOR (k,j) sum the image along the
//   ray, using the shared split_bin() interpolation. Output sized [K*D].
//   (This is the (A x)_i in the MLEM update.)
void forward_project_cpu(const PetProblem& p, const std::vector<float>& image,
                         std::vector<float>& sino);

// backproject_cpu: img = A^T r. For each pixel, sum the sinogram values r on the
//   LORs it lies on (same split_bin weights) -> exact transpose of forward.
//   Used both for the sensitivity image and for back-projecting the ratio.
//   Output sized [N*N].
void backproject_cpu(const PetProblem& p, const std::vector<float>& sino,
                     std::vector<float>& image);

// sensitivity_cpu: s_j = SUM_i A_ij = A^T 1  (back-project a sinogram of ones).
//   The per-pixel normalizer in the MLEM update; computed once before iterating.
void sensitivity_cpu(const PetProblem& p, std::vector<float>& sens);

// mlem_cpu: run `iters` MLEM iterations from a uniform positive image and return
//   the reconstructed image (size N*N). `sens` is the precomputed sensitivity.
//   This is the trusted serial baseline the GPU result is verified against.
void mlem_cpu(const PetProblem& p, const std::vector<float>& sens, int iters,
              std::vector<float>& image);
