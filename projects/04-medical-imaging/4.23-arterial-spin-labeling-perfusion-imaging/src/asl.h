// ===========================================================================
// src/asl.h  --  Shared (host + device) ASL Buxton model + per-voxel fit
// ---------------------------------------------------------------------------
// Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
//
// WHAT THIS PROJECT COMPUTES
//   Arterial Spin Labeling (ASL) is a non-contrast MRI method that measures
//   PERFUSION -- how much arterial blood is delivered to tissue per minute.
//   Water protons in arterial blood are magnetically "labeled" (their spins are
//   inverted) just below the brain; after a post-labeling delay (PLD) the blood
//   has flowed into the tissue and slightly changes the MR signal. Subtracting a
//   "control" image (no labeling) from a "label" image gives the perfusion-
//   weighted difference signal, delta-M -- only ~0.5-1% of the raw signal.
//
//   In MULTI-DELAY ASL we acquire delta-M at several PLDs, giving each voxel a
//   short inflow CURVE delta-M(PLD). The general kinetic (Buxton) model predicts
//   that curve from two physiological unknowns:
//       f    = cerebral blood flow  (CBF), in mL blood / (100 g tissue) / min
//       ATT  = arterial transit time, seconds  (how long labeled blood takes to
//              arrive at this voxel)
//   Recovering (f, ATT) per voxel is a small NONLINEAR LEAST-SQUARES fit. Every
//   voxel is INDEPENDENT, so this is the textbook "same model, many parameter
//   sets" GPU pattern (docs/PATTERNS.md rows 8 & 1): one CUDA thread fits one
//   voxel by Gauss-Newton iteration.
//
//   THE SHARED-CORE IDIOM (docs/PATTERNS.md §2). The forward model, its analytic
//   derivatives (the Jacobian), and the Gauss-Newton solver all live here as
//   `__host__ __device__` inline functions. The CPU reference (reference_cpu.cpp)
//   and the GPU kernel (kernels.cu) therefore run BYTE-FOR-BYTE identical double-
//   precision math, so verification is essentially exact, not approximate.
//
//   Teaching scope. We fit the standard single-compartment pCASL model with a
//   fixed (assumed-known) blood T1 and labeling efficiency, exactly as the
//   simplest oxford_asl / BASIL model does before its Bayesian layer. THEORY.md
//   §"real world" explains what full BASIL adds (variational Bayes priors,
//   dispersion, partial-volume correction).
//
//   Keep CUDA-only constructs OUT of this header (no __global__, no <<<>>>), so
//   the plain host compiler can include it for reference_cpu.cpp.
//
// READ THIS AFTER: util/cuda_check.cuh; BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cmath>     // exp, fabs (host: std::; device: the CUDA math builtins)

// HD-macro idiom: on the NVCC device pass __CUDACC__ is defined and we decorate
// every shared function with __host__ __device__ so it compiles for BOTH the CPU
// and the GPU. On the plain host compiler the decorators do not exist, so we
// expand ASL_HD to nothing. One source of truth for the physics -> identical
// results on both sides (docs/PATTERNS.md §2).
#ifdef __CUDACC__
#define ASL_HD __host__ __device__
#else
#define ASL_HD
#endif

// ---------------------------------------------------------------------------
// Fixed acquisition / physiological constants (SI-ish units in comments).
// These are ASSUMED KNOWN during the fit (as in the standard BASIL model): only
// f and ATT are estimated per voxel. Values are the widely-used ASL consensus
// defaults (Alsop et al. 2015, "recommended implementation of ASL").
// ---------------------------------------------------------------------------
struct AslConstants {
    double T1_blood;   // longitudinal relaxation time of arterial blood (s) ~1.65 @3T
    double T1_tissue;  // longitudinal relaxation time of tissue (s) ~1.30 @3T
    double alpha;      // labeling (inversion) efficiency, dimensionless (pCASL ~0.85)
    double lambda;     // blood-brain partition coefficient of water (mL/g) ~0.90
    double tau;        // labeling duration (s), pCASL bolus length ~1.8
    double M0;         // equilibrium magnetization of blood (arbitrary MR units)
};

// Sensible consensus defaults (see AslConstants field comments for provenance).
// Returned by value so both host and device get their own copy in registers.
ASL_HD inline AslConstants asl_default_constants() {
    AslConstants c;
    c.T1_blood  = 1.65;   // s   (3 T arterial blood)
    c.T1_tissue = 1.30;   // s   (3 T grey matter)
    c.alpha     = 0.85;   // pCASL inversion efficiency
    c.lambda    = 0.90;   // mL/g water partition coefficient
    c.tau       = 1.80;   // s   labeling duration
    c.M0        = 1.0;    // MR units (delta-M is expressed relative to this)
    return c;
}

// ---------------------------------------------------------------------------
// UNIT NOTE on f (CBF).
//   Physiologists quote CBF in mL/100g/min. The Buxton kinetic equation wants a
//   flow in "per second" (1/s) form: f_SI = f_phys / 6000, because
//       1 mL/100g/min = (1/100) mL/g / (60 s) = 1/6000  per second  (rho ~ 1 g/mL).
//   We carry f in the PHYSIOLOGICAL unit (mL/100g/min) everywhere the learner
//   sees it, and convert to per-second only inside the model. 6000 = 100 * 60.
// ---------------------------------------------------------------------------
ASL_HD inline double asl_cbf_to_per_second(double f_phys) {
    return f_phys / 6000.0;   // mL/100g/min  ->  1/s
}

// ---------------------------------------------------------------------------
// asl_buxton  --  the general kinetic (Buxton 1998) forward model for pCASL.
//
//   Predicts the perfusion-weighted difference signal delta-M at one post-
//   labeling delay `pld` (seconds), given flow `f_phys` (mL/100g/min) and
//   arterial transit time `att` (s). This is the standard casl/pcasl kinetic
//   curve used by oxford_asl. Three regimes in the PLD timeline:
//
//     (A) pld < att            : labeled blood has NOT yet arrived -> signal 0.
//     (B) att <= pld < att+tau : blood is still arriving (bolus in transit).
//     (C) pld >= att+tau       : the whole labeled bolus has arrived; signal
//                                now only decays with blood T1.
//
//   The closed-form (single-compartment, blood-T1 decay) expressions are:
//       q = 2 * alpha * M0 * (f/lambda) * T1b
//     (B)  delta-M = q * exp(-att/T1b) * (1 - exp(-(pld-att)/T1b))
//     (C)  delta-M = q * exp(-att/T1b) * exp(-(pld-att-tau)/T1b) * (1 - exp(-tau/T1b))
//   where f here is the per-second flow (converted from f_phys) and T1b=T1_blood.
//
//   WHY THIS SHAPE: the label is a finite bolus of duration tau. During (B) more
//   and more of it has flowed in (the (1-exp) build-up); once fully delivered (C)
//   the magnetization simply relaxes back with the blood T1. Perfusion f scales
//   the whole curve's amplitude; the transit time att shifts it in time. That is
//   exactly why multi-delay ASL can separate the two: amplitude <-> f, onset <-> att.
//
//   Returns delta-M in the same MR units as M0.
// ---------------------------------------------------------------------------
ASL_HD inline double asl_buxton(double pld, double f_phys, double att,
                                const AslConstants& c) {
    const double f = asl_cbf_to_per_second(f_phys);   // 1/s
    const double T1b = c.T1_blood;

    // Regime (A): before the labeled blood arrives, there is no perfusion signal.
    if (pld < att) return 0.0;

    // Common amplitude term q (the "fully-delivered, no-decay" magnetization).
    const double q = 2.0 * c.alpha * c.M0 * (f / c.lambda) * T1b;

    // exp(-att/T1b): the label already lost some magnetization decaying in the
    // artery during the transit; this factor is shared by regimes (B) and (C).
    const double decay_transit = exp(-att / T1b);

    if (pld < att + c.tau) {
        // Regime (B): bolus still arriving. Build-up term (1 - exp(-(pld-att)/T1b)).
        return q * decay_transit * (1.0 - exp(-(pld - att) / T1b));
    } else {
        // Regime (C): full bolus delivered at t=att+tau; decay for the remaining
        // (pld-att-tau) seconds, times the full-bolus factor (1 - exp(-tau/T1b)).
        return q * decay_transit
                 * exp(-(pld - att - c.tau) / T1b)
                 * (1.0 - exp(-c.tau / T1b));
    }
}

// ---------------------------------------------------------------------------
// asl_buxton_grad  --  analytic partial derivatives of delta-M w.r.t. (f, att).
//
//   Gauss-Newton needs the Jacobian J = [d(deltaM)/df, d(deltaM)/datt] at each
//   PLD. We derive them in closed form (cheaper and more stable than finite
//   differences, and it keeps CPU/GPU bit-identical):
//
//   * d/df is trivial: delta-M is LINEAR in f (f only enters through q, which is
//     proportional to f). So d(deltaM)/df = deltaM / f_phys   (for f_phys>0).
//     (We compute it as model/f_phys rather than re-deriving q, to guarantee the
//      derivative is exactly consistent with the value.)
//
//   * d/datt is the interesting one; att appears in the decay_transit factor and
//     in the (pld-att) build-up / (pld-att-tau) decay exponents. Differentiating
//     the regime expressions:
//       (B) dM/datt = q * (-1/T1b) e^{-att/T1b}(1-e^{-(pld-att)/T1b})
//                     + q e^{-att/T1b} * (-(1/T1b) e^{-(pld-att)/T1b})
//                   = -(1/T1b) * [ M_B  +  q e^{-att/T1b} e^{-(pld-att)/T1b} ]
//       (C) the (pld-att-tau) exponent's att-dependence CANCELS the e^{-att/T1b}
//           att-dependence in regime C, leaving dM/datt = 0 to first order except
//           through decay_transit; carefully: in (C) the exponent is
//           -(att)/T1b + -(pld-att-tau)/T1b = -(pld-tau)/T1b, which does NOT
//           depend on att -> so dM/datt = 0 in regime (C).
//     Below the derivative outputs are returned via reference parameters.
//
//   Returns nothing; writes d_df and d_datt. `pld < att` gives a flat zero region
//   (both derivatives 0) -- Gauss-Newton simply gets no gradient from those PLDs,
//   which is correct: a too-large att predicts no signal there.
// ---------------------------------------------------------------------------
ASL_HD inline void asl_buxton_grad(double pld, double f_phys, double att,
                                   const AslConstants& c,
                                   double& d_df, double& d_datt) {
    const double T1b = c.T1_blood;
    const double model = asl_buxton(pld, f_phys, att, c);

    if (pld < att) {                 // regime (A): flat zero, no gradient
        d_df = 0.0; d_datt = 0.0; return;
    }

    // d/df: model is linear in f_phys, so the slope is model / f_phys.
    d_df = (f_phys > 0.0) ? (model / f_phys) : 0.0;

    if (pld < att + c.tau) {
        // regime (B): see derivation above.
        const double f = asl_cbf_to_per_second(f_phys);
        const double q = 2.0 * c.alpha * c.M0 * (f / c.lambda) * T1b;
        const double e_att   = exp(-att / T1b);
        const double e_build = exp(-(pld - att) / T1b);
        d_datt = -(1.0 / T1b) * (model + q * e_att * e_build);
    } else {
        // regime (C): the att-dependence cancels (exponent -> -(pld-tau)/T1b),
        // so to first order delta-M does not change with att here.
        d_datt = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Per-voxel fit configuration and result structs (shared host/device).
// ---------------------------------------------------------------------------

// The whole dataset in flat, GPU-friendly form (Structure-of-Arrays).
//   pld[j]          : the j-th post-labeling delay (s), shared by all voxels.
//   signal[v*n + j] : measured delta-M for voxel v at PLD j (row-major).
// Storing PLDs once (not per voxel) mirrors real ASL: the same delay schedule is
// used for every voxel. The pointers live in host OR device memory depending on
// who calls; the fit function itself only dereferences them, never allocates.
struct AslDataset {
    int    n_voxels;    // number of voxels to fit
    int    n_plds;      // number of post-labeling delays per voxel
    const double* pld;      // [n_plds]           the delay schedule (s)
    const double* signal;   // [n_voxels*n_plds]  measured delta-M, row-major
    AslConstants consts;    // fixed acquisition constants (see above)
    // Gauss-Newton controls (kept in the struct so CPU & GPU use identical values)
    int    max_iters;   // Gauss-Newton iteration cap (small; converges fast)
    double f_init;      // initial CBF guess (mL/100g/min)
    double att_init;    // initial ATT guess (s)
};

// One voxel's fitted physiology + fit quality.
struct AslFit {
    double cbf;    // estimated cerebral blood flow  (mL/100g/min)
    double att;    // estimated arterial transit time (s)
    double sse;    // sum of squared residuals at the solution (fit quality)
    int    iters;  // Levenberg-Marquardt iterations actually taken (<= max_iters)
};

// ---------------------------------------------------------------------------
// asl_sse  --  sum of squared residuals of the Buxton model at (f, att).
//   SSE(f,att) = sum_j ( model(pld_j; f,att) - signal[j] )^2. The objective the
//   fit minimizes; also used by Levenberg-Marquardt to accept/reject a trial step.
//   Shared host/device so a trial evaluation is identical on CPU and GPU.
// ---------------------------------------------------------------------------
ASL_HD inline double asl_sse(const double* pld, const double* sig, int n_plds,
                             double f, double att, const AslConstants& c) {
    double s = 0.0;
    for (int j = 0; j < n_plds; ++j) {
        const double r = asl_buxton(pld[j], f, att, c) - sig[j];   // residual
        s += r * r;                                                // squared, summed
    }
    return s;
}

// ---------------------------------------------------------------------------
// asl_fit_voxel  --  Levenberg-Marquardt nonlinear least-squares fit for ONE voxel.
//
//   Minimizes  SSE(f,att) = sum_j ( model(pld_j; f,att) - signal[j] )^2  over the
//   two unknowns (f, att). At each step we linearize the residual and form the
//   2x2 GAUSS-NEWTON normal equations, then damp them the LEVENBERG-MARQUARDT way:
//
//     r_j   = model_j - signal_j                (residual at PLD j)
//     J_j   = [ dmodel_j/df , dmodel_j/datt ]   (1x2 row, from asl_buxton_grad)
//     A = J^T J = [[Sff, Sfa],[Sfa, Saa]]       (2x2, symmetric, accumulated)
//     g = J^T r = [gf, ga]                      (2x1 gradient of 1/2 SSE)
//     solve (A + lambda*diag(A)) d = -g         (Marquardt-scaled damping)
//
//   WHY MARQUARDT SCALING (lambda*diag(A), not lambda*I). Our two parameters live
//   on VERY different scales: delta-M is linear in CBF (~tens) but nonlinear in
//   ATT (~1 s), so the Jacobian columns differ by ~1000x and A is badly scaled.
//   Adding lambda*I would swamp the small CBF-curvature entry (Sff) and cripple the
//   CBF step. Damping by lambda*diag(A) instead is scale-invariant -- it behaves the
//   same whether CBF is measured in mL/100g/min or per-second (see THEORY §numerics).
//
//   WHY ADAPTIVE lambda (accept/reject). lambda interpolates between Gauss-Newton
//   (lambda->0, fast near the minimum) and gradient descent (lambda large, safe far
//   away). We TRY a step; if it lowers SSE we accept it and shrink lambda (trust
//   Gauss-Newton more); if it does not, we reject it and grow lambda (take a
//   smaller, safer step). This is the standard robust NLLS loop and is exactly the
//   kind of per-voxel optimizer oxford_asl/BASIL runs (before its Bayesian layer).
//
//   Determinism: the accumulation order over j is fixed (ascending loop), the 2x2
//   solve is closed-form, the accept/reject test is a plain comparison, and every-
//   thing is double precision -> the CPU and GPU produce identical AslFit values
//   (verified in main.cu). No atomics, no reductions across threads.
//
//   Complexity: O(iters * n_plds) per voxel; trivial per voxel, but a whole-brain
//   map is ~10^5-10^6 voxels -> the parallelism is across voxels, one thread each.
// ---------------------------------------------------------------------------
ASL_HD inline AslFit asl_fit_voxel(const AslDataset& ds, int v) {
    const AslConstants& c = ds.consts;
    const double* pld = ds.pld;
    const double* sig = ds.signal + (size_t)v * ds.n_plds;  // this voxel's curve

    // Current estimate (start from the shared initial guess).
    double f   = ds.f_init;      // CBF   (mL/100g/min)
    double att = ds.att_init;    // ATT   (s)
    double sse = asl_sse(pld, sig, ds.n_plds, f, att, c);   // objective at start

    // Levenberg-Marquardt damping factor. Start mild; multiply/divide by `nu` on
    // reject/accept. These constants are the textbook defaults (Marquardt 1963).
    double lambda_lm = 1e-3;
    const double nu = 10.0;             // lambda up/down factor per reject/accept

    int it = 0;
    for (; it < ds.max_iters; ++it) {
        // --- Accumulate the 2x2 normal system A = J^T J and gradient g = J^T r ---
        double Sff = 0.0, Sfa = 0.0, Saa = 0.0;   // A entries (Sfa is the off-diag)
        double gf  = 0.0, ga  = 0.0;              // J^T r entries
        for (int j = 0; j < ds.n_plds; ++j) {
            const double r = asl_buxton(pld[j], f, att, c) - sig[j];  // residual
            double df_, datt_;
            asl_buxton_grad(pld[j], f, att, c, df_, datt_);
            // Fixed ascending accumulation order -> bit-identical on CPU and GPU.
            Sff += df_   * df_;
            Sfa += df_   * datt_;
            Saa += datt_ * datt_;
            gf  += df_   * r;
            ga  += datt_ * r;
        }

        // Converged if the gradient is essentially zero (a stationary point of SSE).
        if (fabs(gf) < 1e-14 && fabs(ga) < 1e-14) { ++it; break; }

        // --- Inner LM loop: grow lambda until a trial step reduces SSE ---
        bool accepted = false;
        for (int tries = 0; tries < 30; ++tries) {
            // Marquardt-scaled damped system: add lambda*diag(A) to the diagonal.
            // Guard the diagonal with a tiny floor so a zero-curvature direction
            // (e.g. att unconstrained because all PLDs are past att+tau) still gets
            // a finite, invertible damping term.
            const double A00 = Sff + lambda_lm * (Sff > 0.0 ? Sff : 1e-12);
            const double A11 = Saa + lambda_lm * (Saa > 0.0 ? Saa : 1e-12);
            const double A01 = Sfa;
            const double det = A00 * A11 - A01 * A01;   // 2x2 determinant
            if (fabs(det) < 1e-300) { lambda_lm *= nu; continue; }  // singular: damp more

            // Closed-form 2x2 solve of (A+damp) d = -g:
            //   A^{-1} = 1/det [[A11,-A01],[-A01,A00]],  d = -A^{-1} g.
            const double d_f   = -( A11 * gf  - A01 * ga) / det;
            const double d_att = -(-A01 * gf  + A00 * ga) / det;

            // Trial estimate, clamped to the physical domain (CBF>=0, ATT in [0,5]s).
            // Clamps live in the shared core so CPU and GPU clamp identically; they
            // mirror BASIL's non-negativity constraint on flow.
            double f_new   = f   + d_f;
            double att_new = att + d_att;
            if (f_new   < 0.0) f_new   = 0.0;
            if (att_new < 0.0) att_new = 0.0;
            if (att_new > 5.0) att_new = 5.0;

            const double sse_new = asl_sse(pld, sig, ds.n_plds, f_new, att_new, c);
            if (sse_new < sse) {
                // ACCEPT: the step improved the fit. Move, trust GN more (shrink lambda).
                f = f_new; att = att_new;
                // Converged when the objective barely changed (relative to its size).
                const bool converged = (sse - sse_new) <= 1e-15 * (1.0 + sse);
                sse = sse_new;
                lambda_lm /= nu;
                accepted = true;
                if (converged) { ++it; return AslFit{f, att, sse, it}; }
                break;
            }
            // REJECT: worse (or equal) -> take a smaller, safer step (grow lambda).
            lambda_lm *= nu;
        }
        if (!accepted) break;   // could not improve even with heavy damping -> stop
    }

    return AslFit{f, att, sse, it};
}
