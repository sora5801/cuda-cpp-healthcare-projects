// ===========================================================================
// src/dect.h  --  Shared (host + device) dual-energy CT physics + Newton solver
// ---------------------------------------------------------------------------
// Project 4.20 : Dual-Energy / Spectral CT Reconstruction
//
// WHAT THIS FILE IS
//   The SINGLE source of truth for the per-bin physics of dual-energy CT (DECT)
//   material decomposition. Everything in here is `__host__ __device__` (via the
//   DECT_HD macro) so the CPU reference (reference_cpu.cpp) and the GPU kernel
//   (kernels.cu) run BYTE-FOR-BYTE-IDENTICAL math. That is the whole trick that
//   lets us verify GPU==CPU to machine precision instead of "close enough"
//   (PATTERNS.md §2, the shared __host__ __device__ core idiom).
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>),
//   because the host C++ compiler (cl.exe) must also be able to include it when
//   it compiles reference_cpu.cpp.  DECT_HD expands to `__host__ __device__`
//   under nvcc and to NOTHING under the host compiler.
//
// ----------------------------------------------------------------------------
// THE SCIENCE (see ../THEORY.md for the full derivation)
//
//   A CT scanner measures how much an X-ray beam is attenuated along each line
//   (ray) through the patient. For a MONOCHROMATIC beam of energy E the measured
//   line integral is the classic Beer-Lambert law:
//
//       m(E) = -ln( I / I0 ) = integral_along_ray  mu(E, x)  dx           (1)
//
//   where mu(E,x) is the linear attenuation coefficient [1/cm] at energy E and
//   position x.  In the BASIS-MATERIAL model we write the attenuation of ANY
//   tissue as a combination of two reference materials (here: soft tissue /
//   "water-like" and a contrast/bone material / "iodine-like"):
//
//       mu(E, x) = c1(x) * mu1(E)  +  c2(x) * mu2(E)                      (2)
//
//   mu1, mu2 are the (known) mass-ish attenuation curves of the two basis
//   materials; c1, c2 are the local amounts.  Substituting (2) into (1), the
//   ENERGY integral separates from the SPATIAL integral, so a single ray is
//   fully described by two numbers -- the basis-material PATH LENGTHS:
//
//       t1 = integral c1 dx   [cm of water-equivalent along this ray]
//       t2 = integral c2 dx   [cm of iodine-equivalent along this ray]
//
//   Real scanners are POLYCHROMATIC: the tube emits a whole spectrum S_e(E)
//   (subscript e = spectrum index; e=lo for ~80 kVp, e=hi for ~140 kVp). The
//   detector integrates over energy, so the measured log-attenuation for
//   spectrum e as a function of the path lengths (t1,t2) is the forward model:
//
//                        integral S_e(E) * exp( -(t1*mu1(E) + t2*mu2(E)) ) dE
//       f_e(t1,t2) = -ln --------------------------------------------------- (3)
//                                    integral S_e(E) dE
//
//   This nonlinearity (the log of an energy average of exponentials) is exactly
//   BEAM HARDENING: low-energy photons are absorbed preferentially, so the
//   effective attenuation is not linear in path length.
//
//   MATERIAL DECOMPOSITION = the inverse problem. Given the two MEASURED
//   log-attenuations (m_lo, m_hi) at a sinogram bin, find the path lengths
//   (t1,t2) that reproduce them:
//
//       f_lo(t1,t2) = m_lo
//       f_hi(t1,t2) = m_hi                                                (4)
//
//   Two equations, two unknowns, but NONLINEAR -> solve with Newton's method
//   (below). There are ~10^8 sinogram bins in a real scan and every bin is an
//   INDEPENDENT 2x2 solve -> one GPU thread per bin. That is this project.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The GPU mapping is in THEORY.
// ===========================================================================
#pragma once

// DECT_HD: mark a function callable from BOTH the CPU (host) and a kernel
// (device). Under nvcc __CUDACC__ is defined and we emit both code paths; under
// the plain host compiler the decorators do not exist, so we expand to nothing.
#ifdef __CUDACC__
#define DECT_HD __host__ __device__
#else
#define DECT_HD
#endif

#include <cmath>     // std::exp, std::log, std::fabs, std::fmax
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// NUM_ENERGIES: how many discrete energy samples we quadrature the spectra on.
//   The forward model (3) is an integral over photon energy; we approximate it
//   by a simple sum over NUM_ENERGIES equally-spaced energy bins (rectangle /
//   midpoint rule -- fine for teaching, and identical on host and device). More
//   bins -> a finer spectrum, at linear cost. 24 keeps the sample tiny while
//   still showing real beam-hardening curvature.
// ---------------------------------------------------------------------------
static constexpr int NUM_ENERGIES = 24;

// ---------------------------------------------------------------------------
// SpectralModel: the fixed, scanner-side physics that is the SAME for every
// sinogram bin. We pass it by const-reference to the per-bin solver.
//   * energy_keV[k]  : the k-th sampled photon energy [keV] (for reference/plots)
//   * w_lo[k], w_hi[k]: NORMALISED spectral weights of the low- and high-kVp
//                       spectra at energy k (each array already sums to 1, so the
//                       "integral S_e dE" denominator in (3) is 1 and drops out).
//   * mu1[k], mu2[k] : linear attenuation coefficients [1/cm] of basis material
//                       1 (soft tissue / water-like) and 2 (iodine/bone-like) at
//                       energy k. These encode the ENERGY DEPENDENCE that makes
//                       the two spectra see different contrast -- the entire
//                       reason dual-energy works.
//   Everything is double precision: the decomposition is ill-conditioned (the
//   two materials' attenuation curves are only modestly different), so we do not
//   want to lose bits to FP32 round-off. See THEORY "Numerical considerations".
// ---------------------------------------------------------------------------
struct SpectralModel {
    double energy_keV[NUM_ENERGIES];
    double w_lo[NUM_ENERGIES];
    double w_hi[NUM_ENERGIES];
    double mu1[NUM_ENERGIES];
    double mu2[NUM_ENERGIES];
};

// ---------------------------------------------------------------------------
// forward_one_spectrum: evaluate the polychromatic forward model f_e(t1,t2)
// from equation (3) for ONE spectrum, and ALSO return its partial derivatives
// wrt t1 and t2 (needed to build the Newton Jacobian -- computing them here in
// the same loop is cheaper and keeps host/device identical).
//
//   Inputs:
//     w[k]      : normalised spectral weight at energy k (w_lo or w_hi)
//     mu1,mu2   : the two attenuation curves (arrays of length NUM_ENERGIES)
//     t1,t2     : trial basis-material path lengths [cm]
//   Outputs (by reference):
//     f         : f_e(t1,t2)          = -ln( sum_k w[k] * exp(-p_k) )
//     df_dt1    : d f_e / d t1
//     df_dt2    : d f_e / d t2
//   where p_k = t1*mu1[k] + t2*mu2[k] is the monochromatic attenuation at k.
//
//   DERIVATION of the partials (used verbatim below): let
//       T   = sum_k w[k] * exp(-p_k)             (the "transmission")
//       f   = -ln(T)
//   then  dT/dt1 = sum_k w[k] * exp(-p_k) * (-mu1[k])
//   and   df/dt1 = -(1/T) * dT/dt1 = ( sum_k w[k]*exp(-p_k)*mu1[k] ) / T.
//   (Same with mu2 for t2.) So df/dt is a SPECTRUM-WEIGHTED MEAN of mu -- the
//   "effective attenuation coefficient", which drifts with path length. That
//   drift is beam hardening, and it is why f is nonlinear and we need Newton.
// ---------------------------------------------------------------------------
DECT_HD inline void forward_one_spectrum(const double* w,
                                         const double* mu1, const double* mu2,
                                         double t1, double t2,
                                         double& f, double& df_dt1, double& df_dt2) {
    double T   = 0.0;   // transmission  = sum_k w_k exp(-p_k)
    double num1 = 0.0;  // sum_k w_k exp(-p_k) mu1_k   (numerator of df/dt1 * T)
    double num2 = 0.0;  // sum_k w_k exp(-p_k) mu2_k   (numerator of df/dt2 * T)
    // One pass over the sampled spectrum. This loop is the hot inner work: on the
    // GPU each thread runs it a handful of times (once per Newton iteration).
    for (int k = 0; k < NUM_ENERGIES; ++k) {
        const double p_k = t1 * mu1[k] + t2 * mu2[k];   // monochromatic optical depth
        const double e_k = std::exp(-p_k) * w[k];       // weighted transmitted fraction
        T    += e_k;
        num1 += e_k * mu1[k];
        num2 += e_k * mu2[k];
    }
    // Guard against a pathological all-absorbed ray (T -> 0) so the log/division
    // stay finite; in the sample data T is comfortably > 0.
    const double Tsafe = (T > 1e-300) ? T : 1e-300;
    f      = -std::log(Tsafe);      // equation (3)
    df_dt1 = num1 / Tsafe;          // effective mu1 for this spectrum & path
    df_dt2 = num2 / Tsafe;          // effective mu2 for this spectrum & path
}

// ---------------------------------------------------------------------------
// DecompResult: the outcome of decomposing ONE sinogram bin.
//   * t1, t2   : recovered basis-material path lengths [cm]
//   * iters    : how many Newton iterations were actually taken (<= max)
//   * residual : final ||(f_lo - m_lo, f_hi - m_hi)||_inf  (should be ~tol)
// ---------------------------------------------------------------------------
struct DecompResult {
    double t1;
    double t2;
    int    iters;
    double residual;
};

// ---------------------------------------------------------------------------
// decompose_bin: the CORE of the whole project. Solve the 2x2 nonlinear system
// (4) for one sinogram bin by NEWTON'S METHOD, entirely in registers.
//
//   Newton for a vector system g(x)=0 with x=(t1,t2):
//       x_{n+1} = x_n - J(x_n)^{-1} g(x_n)
//   where g = (f_lo - m_lo, f_hi - m_hi) and J is the 2x2 Jacobian of (f_lo,f_hi)
//   wrt (t1,t2). We assemble J from forward_one_spectrum()'s analytic partials,
//   invert the 2x2 in closed form (cheap, exact), and step. Convergence is
//   quadratic near the root; 5-10 iterations reach machine precision here.
//
//   Inputs:
//     m_lo, m_hi : the two MEASURED log-attenuations at this bin (the data)
//     sm         : the shared spectral model (attenuation curves + spectra)
//     t1_init,   : initial guess for the path lengths. A decent guess makes
//     t2_init      Newton converge in a few steps; we pass the LINEAR-inverse
//                  solution from the caller (see kernels.cu / reference_cpu.cpp).
//     max_iter   : hard cap on iterations (safety; typically not reached)
//     tol        : stop when the max-abs residual drops below this
//   Returns: the recovered (t1,t2), iteration count, and final residual.
//
//   This function is `__host__ __device__` so the CPU loop over bins and the GPU
//   one-thread-per-bin kernel call the IDENTICAL routine -> exact agreement.
// ---------------------------------------------------------------------------
DECT_HD inline DecompResult decompose_bin(double m_lo, double m_hi,
                                          const SpectralModel& sm,
                                          double t1_init, double t2_init,
                                          int max_iter, double tol) {
    double t1 = t1_init;
    double t2 = t2_init;
    int it = 0;
    double res = 0.0;

    for (; it < max_iter; ++it) {
        // Evaluate both spectra's forward model + partials at the current guess.
        double f_lo, dlo_dt1, dlo_dt2;
        double f_hi, dhi_dt1, dhi_dt2;
        forward_one_spectrum(sm.w_lo, sm.mu1, sm.mu2, t1, t2, f_lo, dlo_dt1, dlo_dt2);
        forward_one_spectrum(sm.w_hi, sm.mu1, sm.mu2, t1, t2, f_hi, dhi_dt1, dhi_dt2);

        // Residual g = f(x) - measured. We want g = 0.
        const double g0 = f_lo - m_lo;
        const double g1 = f_hi - m_hi;

        // Max-abs residual: our convergence yardstick (matches THEORY "verify").
        res = std::fmax(std::fabs(g0), std::fabs(g1));
        if (res < tol) break;   // converged -> stop early (fewer iters is fine)

        // 2x2 Jacobian  J = [ df_lo/dt1  df_lo/dt2 ; df_hi/dt1  df_hi/dt2 ].
        const double J00 = dlo_dt1, J01 = dlo_dt2;
        const double J10 = dhi_dt1, J11 = dhi_dt2;

        // Closed-form 2x2 inverse: J^{-1} = (1/det) [ J11 -J01 ; -J10 J00 ].
        // det small -> the two spectra barely disagree -> ill-conditioned; we
        // floor |det| so the step stays finite (THEORY "Numerical considerations").
        double det = J00 * J11 - J01 * J10;
        if (std::fabs(det) < 1e-12) det = (det < 0.0 ? -1e-12 : 1e-12);
        const double inv_det = 1.0 / det;

        // Newton step  delta = J^{-1} g ; update x <- x - delta.
        const double d1 = inv_det * ( J11 * g0 - J01 * g1);
        const double d2 = inv_det * (-J10 * g0 + J00 * g1);
        t1 -= d1;
        t2 -= d2;

        // Path lengths are physically non-negative. Clamp at 0 to keep Newton on
        // the physical branch (a common, documented safeguard in DECT solvers).
        if (t1 < 0.0) t1 = 0.0;
        if (t2 < 0.0) t2 = 0.0;
    }

    DecompResult r;
    r.t1 = t1; r.t2 = t2; r.iters = it; r.residual = res;
    return r;
}

// ---------------------------------------------------------------------------
// virtual_mono_mu: given recovered path lengths (t1,t2) and a chosen VIRTUAL
// MONOENERGETIC energy index k, return the monochromatic log-attenuation that a
// single-energy scan at that energy would have measured:
//       vmi = t1*mu1[k] + t2*mu2[k]          (linear -- no beam hardening!)
//   This is the clinically useful payoff of DECT: "virtual monoenergetic
//   imaging" (VMI) lets a radiologist synthesize an image at any keV, e.g. a low
//   keV to boost iodine contrast or a high keV to suppress metal artifacts. We
//   report one VMI value in the demo to show the decomposition is usable, not
//   just numerically correct.
// ---------------------------------------------------------------------------
DECT_HD inline double virtual_mono_mu(double t1, double t2,
                                      const SpectralModel& sm, int k) {
    return t1 * sm.mu1[k] + t2 * sm.mu2[k];
}
