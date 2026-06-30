// ===========================================================================
// src/reference_cpu.h  --  Data model + shared physics + CPU reference
// ---------------------------------------------------------------------------
// Project 2.3 : Cryo-EM Single-Particle Reconstruction  (reduced-scope, 2D)
//
// WHY A PURE-C++ HEADER (with a __host__ __device__ core)
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA-ONLY syntax (no __global__, no <<<>>>). So the shared DATA MODEL (image
//   sizes, the Dataset container, the file loader) and the PER-ELEMENT PHYSICS
//   that BOTH the CPU reference and the GPU kernels must run *identically* live
//   here, in `__host__ __device__` inline functions (the "HD" idiom -- see
//   docs/PATTERNS.md §2). Running the exact same arithmetic on both sides is
//   what makes verification EXACT instead of approximate.
//
//   `#ifdef __CUDACC__` lets the SAME header be read two ways:
//     * by nvcc (compiling kernels.cu/main.cu): CRYO_HD expands to the real
//       `__host__ __device__` decorators, so these functions run on the GPU.
//     * by cl.exe/g++ (compiling reference_cpu.cpp): CRYO_HD expands to nothing,
//       so they are ordinary host functions. No CUDA tokens leak into host code.
//
// THE SCIENCE IN ONE PARAGRAPH (full derivation in ../THEORY.md)
//   In single-particle cryo-EM, thousands of copies of the SAME molecule are
//   flash-frozen in random ORIENTATIONS and imaged by an electron microscope.
//   Each image is (to first order) a 2D PROJECTION of the molecule's 3D density
//   along the beam. The job: recover the density from many projections whose
//   orientations are UNKNOWN and whose images are buried in noise. The walltime
//   bottleneck (catalog deep-dive) is PROJECTION MATCHING: comparing every one
//   of N particles against every one of M reference projections -- an O(N*M)
//   cross-correlation sweep. That is the step we put on the GPU.
//
//   We teach the idea in a tractable 2D world: the "molecule" is a 2D image; a
//   "projection" is the 1D sum of that image along parallel rays at some angle
//   (a Radon transform / sinogram column). Reference projections at M known
//   angles form our template bank; each noisy particle is a 1D projection at an
//   unknown angle. We (1) MATCH each particle to its best-fitting reference
//   angle (the E-step), then (2) BACK-PROJECT the matched 1D profiles into a 2D
//   density (the M-step) -- exactly the projection-match + back-project loop at
//   the heart of RELION/cryoSPARC, minus the Bayesian weighting and CTF.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>      // std::sin, std::cos, std::floor, std::sqrt
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The HD ("host+device") macro. See the file header for why this exists.
//   * Under nvcc (__CUDACC__ defined) -> real CUDA decorators -> runs on GPU.
//   * Under a plain host compiler      -> empty -> ordinary host function.
// Keep ONLY math in HD functions (no CUDA types, no __global__) so the host
// compiler can include this header unchanged.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CRYO_HD __host__ __device__
#else
#define CRYO_HD
#endif

// ---------------------------------------------------------------------------
// Geometry constants (COMPILE-TIME so loops unroll and buffers are fixed-size).
//   IMG_SIZE  : the molecule image is IMG_SIZE x IMG_SIZE pixels (square).
//   PROJ_LEN  : a 1D projection has PROJ_LEN samples. We use PROJ_LEN == IMG_SIZE
//               so a ray crossing the image contributes one sample per row.
//   N_ANGLES  : number of reference projection angles M (the template bank). The
//               angles are evenly spaced over [0, pi): projection at theta and
//               theta+pi are mirror images, so half-circle coverage suffices.
// These mirror the catalog's "N particles x M reference projections" with
// M = N_ANGLES. Chosen small so the committed sample runs in milliseconds.
// ---------------------------------------------------------------------------
constexpr int IMG_SIZE = 64;            // density grid is 64 x 64 pixels
constexpr int PROJ_LEN = IMG_SIZE;      // one projection sample per image row
constexpr int N_ANGLES = 60;            // M reference angles over [0, pi)

// Pi as a double; the device path uses the same value so trig matches the host.
constexpr double CRYO_PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// ref_angle(a): the physical angle (radians) of reference projection index a.
//   Evenly spaced: theta_a = a * pi / N_ANGLES, a in [0, N_ANGLES). This is the
//   ONE definition of "what angle does template a represent", shared by the
//   sampler (make_synthetic / loader) and both compute paths, so there is never
//   a half-pixel disagreement about geometry.
// ---------------------------------------------------------------------------
CRYO_HD inline double ref_angle(int a) {
    return static_cast<double>(a) * CRYO_PI / static_cast<double>(N_ANGLES);
}

// ---------------------------------------------------------------------------
// sample_at: bounds-checked pixel read (zero outside the grid). A free function
//   (not a device lambda capture, which is not portable across all toolchains)
//   so project_sample can reuse it on host and device alike.
//     img : [IMG_SIZE*IMG_SIZE] row-major density.
//     xi,yi : integer pixel coords; out-of-range -> 0 (zero-padded support).
// ---------------------------------------------------------------------------
CRYO_HD inline double sample_at(const float* img, int xi, int yi) {
    if (xi < 0 || xi >= IMG_SIZE || yi < 0 || yi >= IMG_SIZE) return 0.0;
    return static_cast<double>(img[yi * IMG_SIZE + xi]);
}

// ---------------------------------------------------------------------------
// project_sample(img, theta, s): the value of the 1D projection of image `img`
//   at projection coordinate s (0..PROJ_LEN-1), for view angle `theta`.
//
//   GEOMETRY (the Radon transform, discretized). Place the origin at the image
//   centre. The detector line is perpendicular to the ray direction. Sample s
//   maps to a signed offset t = s - centre along the detector. A point at
//   detector offset t, walked a distance u along the ray, has image coordinates
//       x =  t*cos(theta) - u*sin(theta)
//       y =  t*sin(theta) + u*cos(theta)
//   The projection value is the LINE INTEGRAL over u: sum of img along that ray.
//   We integrate u over the same range as the detector and bilinearly sample
//   img at each (x,y) -- the standard parallel-beam forward projector, identical
//   in spirit to the back-projector used in CT (project 4.01), run forwards.
//
//   This function is the per-(angle,sample) physics. Using it for BOTH the CPU
//   reference and the GPU kernel guarantees the synthetic templates, the CPU
//   match, and the GPU match all see byte-identical projection values.
//
//   img  : [IMG_SIZE*IMG_SIZE] row-major density (img[y*IMG_SIZE + x]).
//   theta: view angle in radians.
//   s    : detector sample index in [0, PROJ_LEN).
//   returns: the (un-normalized) line integral for this sample.
// ---------------------------------------------------------------------------
CRYO_HD inline float project_sample(const float* img, double theta, int s) {
    const double centre = (IMG_SIZE - 1) * 0.5;   // image/detector centre (pixels)
    const double t      = static_cast<double>(s) - centre;  // signed detector offset
    const double ct = cos(theta);   // device: cos(double) is fine; host: std::cos
    const double st = sin(theta);

    double acc = 0.0;   // running line integral along the ray
    // Walk the ray across the full image extent, one step per row, and bilinearly
    // sample the density. PROJ_LEN steps keep the cost O(IMG_SIZE) per sample.
    for (int k = 0; k < PROJ_LEN; ++k) {
        const double u = static_cast<double>(k) - centre;     // signed distance along ray
        const double x = centre + (t * ct - u * st);          // image x (continuous)
        const double y = centre + (t * st + u * ct);          // image y (continuous)

        // Bilinear interpolation. floor gives the lower grid corner; fx,fy are
        // the fractional offsets used as blend weights. Samples outside the grid
        // contribute 0 (the molecule is compactly supported -> zero padding).
        const int x0 = static_cast<int>(floor(x));
        const int y0 = static_cast<int>(floor(y));
        const double fx = x - x0;
        const double fy = y - y0;
        // Four neighbouring pixels, each guarded against going out of bounds.
        const double v00 = sample_at(img, x0,     y0);
        const double v10 = sample_at(img, x0 + 1, y0);
        const double v01 = sample_at(img, x0,     y0 + 1);
        const double v11 = sample_at(img, x0 + 1, y0 + 1);
        // Standard 2D bilinear blend of the four corners.
        const double top = v00 * (1.0 - fx) + v10 * fx;
        const double bot = v01 * (1.0 - fx) + v11 * fx;
        acc += top * (1.0 - fy) + bot * fy;
    }
    return static_cast<float>(acc);
}

// ---------------------------------------------------------------------------
// ncc_score(a, b, len): the (mean-subtracted) cross-correlation similarity of
//   two length-`len` 1D profiles. This is the SCORE projection matching uses to
//   decide "which reference angle does this particle look most like".
//
//   We use the centred, scale-free Pearson correlation:
//       ncc = sum((a-mean_a)*(b-mean_b)) / sqrt(sum((a-mean_a)^2)*sum((b-mean_b)^2))
//   in [-1, 1]. Subtracting the mean removes a constant additive offset; dividing
//   by the norms removes a multiplicative scale -- so a particle matches the
//   reference of the same SHAPE regardless of overall brightness/contrast, which
//   is exactly what we want when contrast varies particle-to-particle. (Real
//   RELION uses a probabilistic cross-correlation under a noise model; THEORY
//   §"real world" explains the upgrade.)
//
//   IMPORTANT for determinism: the additions happen in a FIXED order (k=0..len-1)
//   and in the SAME order on host and device, and we use the SAME float math, so
//   CPU and GPU compute bit-identical scores -> identical argmax -> EXACT match.
// ---------------------------------------------------------------------------
CRYO_HD inline float ncc_score(const float* a, const float* b, int len) {
    // Pass 1: means. (Two passes keep the formula readable; len is tiny.)
    float mean_a = 0.0f, mean_b = 0.0f;
    for (int k = 0; k < len; ++k) { mean_a += a[k]; mean_b += b[k]; }
    mean_a /= static_cast<float>(len);
    mean_b /= static_cast<float>(len);

    // Pass 2: centred dot product and the two centred norms.
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int k = 0; k < len; ++k) {
        const float da = a[k] - mean_a;
        const float db = b[k] - mean_b;
        dot += da * db;     // numerator
        na  += da * da;     // |a-mean_a|^2
        nb  += db * db;     // |b-mean_b|^2
    }
    const float denom = sqrtf(na * nb);          // sqrtf: same on host & device
    // A flat (zero-variance) profile has no shape to match -> score 0, not NaN.
    return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

// ---------------------------------------------------------------------------
// profile_lerp(prof, t): bilinearly (here: linearly) sample a 1D projection
//   profile at continuous detector coordinate t. Used by back-projection.
//     prof : [PROJ_LEN] one particle's projection values.
//     t    : continuous detector coordinate; out-of-range -> 0.
//   Linear interpolation between the two bracketing samples; this is the 1D
//   analogue of the bilinear gather in project_sample.
// ---------------------------------------------------------------------------
CRYO_HD inline float profile_lerp(const float* prof, double t) {
    if (t < 0.0 || t > static_cast<double>(PROJ_LEN - 1)) return 0.0f;
    const int    s0 = static_cast<int>(floor(t));   // lower detector sample
    const int    s1 = (s0 + 1 < PROJ_LEN) ? s0 + 1 : s0;
    const double f  = t - s0;                        // fractional position
    return static_cast<float>(prof[s0] * (1.0 - f) + prof[s1] * f);
}

// ---------------------------------------------------------------------------
// backproject_pixel(particles, assign, ref_thetas, n, px, py): the back-
//   projected density value at output pixel (px, py).
//
//   THE M-STEP (simple back-projection). Each particle, once assigned an angle,
//   "smears" its 1D profile back across the image along that view direction.
//   The value deposited at image point (px,py) by particle i is the profile of
//   particle i sampled at the detector coordinate that (px,py) projects to under
//   particle i's assigned angle:
//       t_i = (px-c)*cos(theta_i) + (py-c)*sin(theta_i) + c
//   Summing over all particles and dividing by N gives the reconstructed
//   density. We compute it PER OUTPUT PIXEL (a gather, like CT back-projection
//   in project 4.01) -- so there are NO atomics and the per-pixel sum runs in a
//   FIXED particle order (i=0..N-1), making CPU and GPU bit-identical.
//
//   Mathematically this is the (unfiltered) inverse Radon transform; a ramp
//   filter would sharpen it (see THEORY §"real world" and project 4.01). We keep
//   it unfiltered so the teaching code is one clean accumulation.
//
//     particles  : [n*PROJ_LEN] all particle profiles, row-major.
//     assign     : [n] each particle's assigned reference-angle index.
//     ref_thetas : [N_ANGLES] precomputed ref_angle(a) (so we don't recompute
//                  trig per pixel; identical values on host and device).
//     n          : number of particles.
//     px, py     : output pixel coordinates in [0, IMG_SIZE).
//   returns: the back-projected density at (px,py) (mean over particles).
// ---------------------------------------------------------------------------
CRYO_HD inline float backproject_pixel(const float* particles, const int* assign,
                                       const double* ref_thetas, int n,
                                       int px, int py) {
    const double centre = (IMG_SIZE - 1) * 0.5;
    const double dx = static_cast<double>(px) - centre;   // pixel offset from centre
    const double dy = static_cast<double>(py) - centre;
    float acc = 0.0f;   // running sum of contributions from all particles
    for (int i = 0; i < n; ++i) {
        const double th = ref_thetas[assign[i]];          // this particle's view angle
        // Detector coordinate that (px,py) maps to under angle th (note the
        // sign convention matches project_sample's forward geometry).
        const double t = centre + dx * cos(th) + dy * sin(th);
        acc += profile_lerp(particles + static_cast<long long>(i) * PROJ_LEN, t);
    }
    return acc / static_cast<float>(n);   // average -> independent of particle count
}

// ---------------------------------------------------------------------------
// The loaded problem instance.
//   n_particles : number of observed particle images N.
//   true_img    : [IMG_SIZE*IMG_SIZE] the ground-truth density (synthetic, so we
//                 can score the reconstruction). NOT used by the matcher.
//   refs        : [N_ANGLES*PROJ_LEN] the reference projection bank (template a
//                 occupies refs[a*PROJ_LEN .. a*PROJ_LEN+PROJ_LEN-1]).
//   particles   : [n_particles*PROJ_LEN] the observed 1D projections, row-major.
//   true_angle  : [n_particles] the angle index each particle was generated at
//                 (synthetic ground truth, used only to REPORT recovery accuracy
//                 -- the matcher never sees it).
// ---------------------------------------------------------------------------
struct Dataset {
    int n_particles = 0;
    std::vector<float> true_img;     // [IMG_SIZE*IMG_SIZE]
    std::vector<float> refs;         // [N_ANGLES*PROJ_LEN]
    std::vector<float> particles;    // [n_particles*PROJ_LEN]
    std::vector<int>   true_angle;   // [n_particles]
};

// Load a dataset from the text format documented in data/README.md. Throws
// std::runtime_error on a missing file or a geometry mismatch.
Dataset load_dataset(const std::string& path);

// ---------------------------------------------------------------------------
// CPU REFERENCE (the trusted, obviously-correct baseline the GPU is checked
// against). Two outputs, matching the two GPU kernels:
//   assign[i]   = argmax over a of ncc_score(particle_i, ref_a)  (the E-step)
//   recon[p]    = back-projection of all particles into the density (the M-step)
// `recon` is [IMG_SIZE*IMG_SIZE]. Both are filled deterministically.
// ---------------------------------------------------------------------------
void match_cpu(const Dataset& ds, std::vector<int>& assign, std::vector<float>& best_score);
void reconstruct_cpu(const Dataset& ds, const std::vector<int>& assign,
                     std::vector<float>& recon);
