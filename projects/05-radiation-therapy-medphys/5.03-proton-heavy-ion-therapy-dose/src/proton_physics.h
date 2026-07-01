// ===========================================================================
// src/proton_physics.h  --  Shared (host + device) proton pencil-beam physics
// ---------------------------------------------------------------------------
// Project 5.3 : Proton & Heavy-Ion Therapy Dose
//
// WHY THIS HEADER IS SHARED (the HD-core idiom, see docs/PATTERNS.md §2)
//   The whole verification strategy is "CPU reference == GPU kernel". That only
//   holds if BOTH sides evaluate the *identical* dose formula, operation for
//   operation. So the per-(voxel, spot) physics lives here, in ONE header,
//   included by reference_cpu.cpp (plain host compiler) AND kernels.cu / main.cu
//   (nvcc). The PB_HD macro expands to `__host__ __device__` under nvcc and to
//   nothing under the host compiler, so the same inline functions compile in
//   both worlds and produce bit-comparable results. Keep this header free of any
//   CUDA-only constructs (no __global__, no <cuda_runtime.h>) so cl.exe/g++ can
//   compile it for the CPU reference.
//
// THE PHYSICS WE MODEL (a deliberately reduced TEACHING model -- THEORY.md has
// the full clinical picture and the simplifications spelled out)
//   A pencil-beam-scanning (PBS) proton plan is a list of "spots". Each spot is
//   a thin proton pencil beam that enters the patient along +z at a lateral
//   position (x0, y0), carries a weight w (proportional to the number of protons
//   / monitor units for that spot), and has an energy that fixes its RANGE R --
//   the depth at which the protons stop. The dose a single spot deposits at a
//   voxel (x, y, z) FACTORISES into two independent parts:
//
//     dose_spot(x,y,z) = w * IDD(z; R) * Lateral(x-x0, y-y0; sigma(z))
//
//     * IDD(z; R)  -- the "integral depth dose": energy deposited per unit depth
//       by the beam as a whole, as a function of depth z. Its hallmark is the
//       BRAGG PEAK: near-flat "plateau" in the entrance region, a sharp rise to
//       a peak just proximal to the range R, then a fast fall to ~zero DISTAL to
//       R. This distal fall-off is exactly what lets proton therapy spare tissue
//       behind the target -- the single most important fact in the whole domain.
//       We use a smooth analytic surrogate (Bortfeld-style) built from the
//       residual range (R - z); see idd_bragg() below.
//
//     * Lateral(dx,dy; sigma) -- an in-plane 2-D Gaussian of width sigma. The
//       beam is narrow at entry and spreads with depth due to multiple Coulomb
//       scattering, so sigma GROWS with z. We model sigma(z) linearly.
//
//   The full 3-D dose is the SUM of dose_spot over every spot -- a convolution
//   of pencil-beam kernels with the spot map. That sum is what the GPU does in
//   parallel (one thread per voxel accumulates over all spots).
//
// UNITS (documented once, used everywhere)
//   lengths (x,y,z,R,sigma) : centimetres (cm)
//   weight w                : arbitrary "MU-like" units (dimensionless here)
//   dose                    : arbitrary units proportional to Gray (a TEACHING
//                             scale -- NOT calibrated to clinical Gy; §Limitations)
//
// READ THIS AFTER: nothing (start here); it is the physics core the rest builds
// on. Then reference_cpu.h, kernels.cuh.  Not for clinical use (CLAUDE.md §1).
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::pow, std::sqrt (host); nvcc maps to device intrinsics

// PB_HD = "__host__ __device__" under nvcc, empty under the host compiler.
// This is the single line that lets one function body serve both the CPU
// reference and the GPU kernel (docs/PATTERNS.md §2).
#ifdef __CUDACC__
#define PB_HD __host__ __device__
#else
#define PB_HD
#endif

// ---------------------------------------------------------------------------
// One PBS spot. Plain-old-data so it copies trivially to the device (we can
// pass a Spot* array straight to cudaMemcpy). Everything is `float` because
// clinical dose engines run in single precision on the GPU for speed/memory,
// and FP32 is plenty for a teaching model -- we simply verify CPU and GPU in
// the SAME precision so the comparison is apples-to-apples (THEORY.md §numerics).
// ---------------------------------------------------------------------------
struct Spot {
    float x0;      // lateral x position of the pencil-beam axis (cm)
    float y0;      // lateral y position of the pencil-beam axis (cm)
    float range;   // beam range R: depth where protons stop (cm). Sets the Bragg-peak depth.
    float weight;  // spot weight w (MU-like); scales this spot's whole contribution
};

// ---------------------------------------------------------------------------
// Geometry of the dose grid we score onto. A regular Cartesian box of
// nx*ny*nz voxels, each of edge `dx` cm, with the grid origin at (ox,oy,oz).
// Voxel (i,j,k) has its CENTRE at (ox + (i+0.5)*dx, oy + (j+0.5)*dx,
// oz + (k+0.5)*dx). Storing the origin + spacing (instead of per-voxel
// coordinates) keeps the struct tiny and lets every thread compute its own
// voxel centre from its linear index -- no coordinate lookup table needed.
// ---------------------------------------------------------------------------
struct Grid {
    int   nx, ny, nz;   // voxel counts along x, y, z (z is the beam/depth axis)
    float dx;           // isotropic voxel edge length (cm)
    float ox, oy, oz;   // world coordinates of the grid's (0,0,0) corner (cm)
};

// Physical constants of the beam model, shared by CPU and GPU so both agree.
// Grouped in a struct (rather than #defines) so they are typed and can be
// passed by value into the kernel.
struct BeamModel {
    float sigma0;      // lateral spread at entrance z=0 (cm): finite spot size at skin
    float sigma_grow;  // growth of sigma per cm of depth (cm/cm): multiple-scattering spread
    float peak_width;  // width parameter of the Bragg peak (cm): larger => broader, lower peak
    float p_exp;       // Bragg exponent p (~1.77 for the Bortfeld model); shapes the rise
};

// A single, canonical default beam model. Defined `inline` so this header can be
// included in multiple translation units (main.cu, kernels.cu, reference_cpu.cpp)
// without a duplicate-symbol link error. These numbers are illustrative teaching
// values, not a fit to a real machine (THEORY.md §real world).
inline BeamModel default_beam_model() {
    BeamModel m;
    m.sigma0     = 0.30f;   // 3 mm spot at the surface
    m.sigma_grow = 0.020f;  // sigma widens ~0.2 mm per cm depth (gentle scattering)
    m.peak_width = 0.60f;   // 6 mm characteristic Bragg-peak width
    m.p_exp      = 1.77f;   // classic Bortfeld exponent
    return m;
}

// ---------------------------------------------------------------------------
// sigma_at_depth: lateral Gaussian width sigma(z) of a pencil beam at depth z.
//   Modelled linearly: sigma(z) = sigma0 + sigma_grow * z (clamped to >0). The
//   real growth is closer to sqrt of a scattering integral, but linear captures
//   the teaching point -- deeper beams are wider -- with a stable, monotone form.
//   Units: z in cm -> sigma in cm.
// ---------------------------------------------------------------------------
PB_HD inline float sigma_at_depth(const BeamModel& m, float z) {
    float s = m.sigma0 + m.sigma_grow * z;   // linear broadening with depth
    return (s > 1e-4f) ? s : 1e-4f;          // guard: never zero (avoids div-by-0 below)
}

// ---------------------------------------------------------------------------
// lateral_gaussian: normalised 2-D Gaussian of the in-plane offset (dx,dy) from
// the beam axis, with width sigma. This spreads each spot's dose laterally.
//   value = 1/(2*pi*sigma^2) * exp( -(dx^2+dy^2) / (2 sigma^2) )
//   The 1/(2*pi*sigma^2) prefactor conserves the in-plane integral (so widening
//   the beam lowers the central dose rather than adding energy) -- a real
//   physical property worth teaching. Units: dx,dy,sigma in cm -> 1/cm^2.
// ---------------------------------------------------------------------------
PB_HD inline float lateral_gaussian(float dx, float dy, float sigma) {
    const float inv_two_sig2 = 1.0f / (2.0f * sigma * sigma);   // 1/(2 sigma^2)
    const float r2 = dx * dx + dy * dy;                          // squared radial offset
    const float norm = inv_two_sig2 / 3.14159265358979323846f;  // = 1/(2*pi*sigma^2)
    return norm * expf(-r2 * inv_two_sig2);
}

// ---------------------------------------------------------------------------
// idd_bragg: the integral depth dose IDD(z; R) -- the depth term carrying the
// Bragg peak. We use a smooth analytic surrogate inspired by Bortfeld's closed
// form, expressed through the RESIDUAL RANGE  u = R - z  (how far the protons
// still have to travel before stopping):
//
//     u <= 0            -> 0              (distal to the range: protons stopped)
//     0 < u             -> plateau + peak
//
//   * PLATEAU: a small, near-constant entrance dose ~ proportional to the
//     stopping power far upstream. We model it as a gentle 1/(u + width) term so
//     the entrance dose is finite and rises slowly as the beam slows.
//   * PEAK: the sharp Bragg maximum near u -> 0. Bortfeld's solution behaves like
//     u^(1/p - 1) near the stop point; we regularise the singularity with the
//     peak_width w so the peak is finite (height ~ w^(1/p - 1)) and smooth:
//         peak(u) = (u + w)^(1/p - 1)
//     Since 1/p - 1 < 0 (p~1.77 -> exponent ~ -0.435), this term is LARGE for
//     small u (just before the stop) and SMALL for large u (near entrance) --
//     exactly the Bragg shape: low plateau, sharp distal peak, then a hard zero.
//
//   The result is multiplied by a modest plateau baseline so the entrance region
//   is nonzero (real beams deposit dose all along the track, not only at the
//   peak). The absolute scale is arbitrary teaching units (see file header).
//   Units: z,R in cm -> dimensionless depth-dose weight.
//
//   WHY REGULARISE INSTEAD OF USING THE EXACT SINGULAR FORM: the true Bortfeld
//   IDD contains a parabolic-cylinder function and a genuine integrable
//   singularity at u=0. Evaluating that identically on host and device (to get a
//   bit-comparable CPU/GPU result) is fiddly and off-topic for the GPU lesson.
//   The regularised (u+w) form keeps the *shape* and the teaching value while
//   being a couple of cheap flops that host and device evaluate identically.
// ---------------------------------------------------------------------------
PB_HD inline float idd_bragg(const BeamModel& m, float z, float R) {
    const float u = R - z;                 // residual range (cm): distance left to travel
    if (u <= 0.0f) return 0.0f;            // distal to the Bragg peak -> zero dose (the key sparing effect)

    const float w = m.peak_width;          // regularisation width (cm)
    // Peak term: (u+w)^(1/p - 1). exponent is negative -> blows up (finitely) as u->0.
    const float peak = powf(u + w, 1.0f / m.p_exp - 1.0f);
    // Plateau term: a small, slowly varying entrance contribution so the track
    // upstream of the peak carries dose too. Bounded and smooth.
    const float plateau = 0.20f / (u + w);
    return peak + plateau;                 // full depth-dose weight at this depth
}

// ---------------------------------------------------------------------------
// dose_from_spot: THE per-(voxel, spot) kernel of the whole engine. Returns the
// dose one spot deposits at one voxel centre (vx,vy,vz). This is the single
// function the GPU thread and the CPU reference both call in their inner loop --
// so if this is identical, the two results are identical.
//
//   dose = w * IDD(vz - z_entry ; R) * Lateral(vx - x0, vy - y0 ; sigma(depth))
//
//   * depth is measured from the patient surface, which we take as the grid's
//     z-origin oz (the beam enters at z = oz). So the physical depth of voxel
//     centre vz is  depth = vz - z_entry.
//   * sigma is evaluated at that depth (deeper -> wider beam).
//   Units: returns arbitrary dose units (see header). All inputs in cm / MU.
// ---------------------------------------------------------------------------
PB_HD inline float dose_from_spot(const BeamModel& m, const Spot& s,
                                  float vx, float vy, float vz, float z_entry) {
    const float depth = vz - z_entry;                  // depth into patient (cm)
    if (depth < 0.0f) return 0.0f;                      // voxel is upstream of the surface
    const float sigma = sigma_at_depth(m, depth);       // lateral width at this depth
    const float lat   = lateral_gaussian(vx - s.x0, vy - s.y0, sigma);
    const float idd   = idd_bragg(m, depth, s.range);   // depth term (Bragg peak)
    return s.weight * idd * lat;                         // combine: weight x depth x lateral
}
