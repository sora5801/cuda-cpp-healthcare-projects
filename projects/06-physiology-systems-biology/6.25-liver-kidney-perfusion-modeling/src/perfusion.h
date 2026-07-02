// ===========================================================================
// src/perfusion.h  --  Shared (host + device) liver-sinusoid transport physics
// ---------------------------------------------------------------------------
// Project 6.25 : Liver & Kidney Perfusion Modeling
//
// WHAT THIS PROJECT COMPUTES
//   A liver LOBULE is a repeating hexagonal functional unit. Blood enters at the
//   PERIPORTAL edge (zone 1: oxygen-rich), flows down thousands of narrow
//   capillaries called SINUSOIDS lined with hepatocytes, and leaves at the
//   CENTRILOBULAR vein (zone 3: oxygen-poor). As blood streams past, the
//   hepatocytes CLEAR a drug/toxin from it via a saturable enzyme reaction.
//
//   We model ONE sinusoid as a 1-D plug-flow tube of length L. A drug at
//   concentration C(x) is carried downstream by blood at velocity v (CONVECTION)
//   and simultaneously consumed by the wall enzymes at a Michaelis-Menten rate
//   (REACTION). At steady state the concentration profile obeys the ODE
//
//       v * dC/dx = -R(C, x)                      (convection-reaction balance)
//       R(C, x)   =  Vmax(x) * C / (Km + C)       (Michaelis-Menten clearance)
//
//   The key physiology is ZONATION: the metabolic capacity Vmax is NOT uniform.
//   Different enzymes are expressed periportally vs. centrilobularly (Human
//   Protein Atlas liver data show this gradient). We model Vmax(x) as a linear
//   ramp from Vmax_pp at the inlet (x=0) to Vmax_cl at the outlet (x=L).
//
//   A whole lobule = MANY sinusoids in parallel, each with a slightly different
//   inlet blood velocity (perfusion is heterogeneous). Each sinusoid is an
//   INDEPENDENT 1-D ODE solve -> we give each sinusoid its OWN GPU THREAD
//   (PATTERNS.md section 1: "the same ODE for many parameter sets -> ensemble").
//
//   The derivative AND the RK4 spatial step live here as __host__ __device__
//   inline functions so the CPU reference and the GPU kernel integrate
//   IDENTICALLY -> their results match to round-off. PERF_HD expands to
//   __host__ __device__ under nvcc, and to nothing under the host compiler
//   (PATTERNS.md section 2, the HD-macro idiom).
//
//   >>> Educational, NOT for clinical use. All data are SYNTHETIC. <<<
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// The HD-macro idiom (PATTERNS.md section 2): under nvcc (__CUDACC__ defined) the
// functions below compile for BOTH the host and the device from a single source,
// guaranteeing the CPU reference and the GPU kernel run byte-identical math. Under
// the plain host compiler these decorators do not exist, so we erase them.
#ifdef __CUDACC__
#define PERF_HD __host__ __device__
#else
#define PERF_HD
#endif

// ---------------------------------------------------------------------------
// SinusoidParams : the fixed physical constants shared by every sinusoid in a
//   lobule. Concentrations are in micromolar (uM), lengths in millimetres (mm),
//   velocities in mm/s, and Vmax in uM/s. These units are internally consistent
//   so v*dC/dx [uM/s] balances R [uM/s]. Kept POD so it copies cleanly to the
//   device by value (the kernel takes it as a launch argument).
// ---------------------------------------------------------------------------
struct SinusoidParams {
    double L = 0.0;         // sinusoid length (mm) from periportal (x=0) to centrilobular (x=L)
    double C_in = 0.0;      // inlet drug concentration (uM) entering at x=0
    double Km = 0.0;        // Michaelis constant (uM): C at which the enzyme runs at half Vmax
    double Vmax_pp = 0.0;   // maximal clearance rate at the PERIPORTAL end   (uM/s)
    double Vmax_cl = 0.0;   // maximal clearance rate at the CENTRILOBULAR end (uM/s)
    int    nseg = 0;        // number of spatial RK4 steps along the sinusoid (grid resolution)
};

// ---------------------------------------------------------------------------
// zonal_vmax: the zonation model. Vmax varies LINEARLY along the sinusoid from
//   the periportal value at x=0 to the centrilobular value at x=L. `xfrac` is the
//   fractional position x/L in [0,1]. This one function encodes the entire
//   "oxygen-zone-specific metabolism" idea from the catalog deep-dive; swapping it
//   for a nonlinear (e.g. oxygen-driven) profile is left as an exercise.
// ---------------------------------------------------------------------------
PERF_HD inline double zonal_vmax(const SinusoidParams& p, double xfrac) {
    return p.Vmax_pp + (p.Vmax_cl - p.Vmax_pp) * xfrac;   // linear periportal->centrilobular ramp
}

// ---------------------------------------------------------------------------
// mm_rate: the Michaelis-Menten clearance rate R(C,x) = Vmax(x)*C/(Km+C) [uM/s].
//   * When C << Km the reaction is FIRST-ORDER (R ~ (Vmax/Km)*C, unsaturated).
//   * When C >> Km it SATURATES at Vmax (zero-order, enzymes maxed out).
//   This saturation is exactly why the GPU is useful: the ODE is nonlinear, so
//   there is no closed-form profile and we must integrate numerically per
//   sinusoid. `C` is clamped at 0 so a tiny negative round-off cannot create a
//   spurious negative rate.
// ---------------------------------------------------------------------------
PERF_HD inline double mm_rate(const SinusoidParams& p, double C, double xfrac) {
    double c = (C > 0.0) ? C : 0.0;                 // physical guard: concentration >= 0
    double vmax = zonal_vmax(p, xfrac);             // local (zone-dependent) capacity
    return vmax * c / (p.Km + c);                   // saturable Michaelis-Menten kinetics
}

// ---------------------------------------------------------------------------
// dCdx: the right-hand side of the steady-state transport ODE, dC/dx.
//   Rearranging v*dC/dx = -R(C,x) gives dC/dx = -R(C,x)/v. `v` is this sinusoid's
//   blood velocity (mm/s); faster flow spends less residence time per unit length,
//   so the concentration drops more slowly with x. This is the function RK4
//   evaluates four times per step.
// ---------------------------------------------------------------------------
PERF_HD inline double dCdx(const SinusoidParams& p, double C, double xfrac, double v) {
    return -mm_rate(p, C, xfrac) / v;               // convection carries the reaction loss downstream
}

// ---------------------------------------------------------------------------
// SinusoidResult : the per-sinusoid summary the analysis reports.
//   extraction_ratio E = (C_in - C_out)/C_in is the clinically meaningful
//   quantity -- the fraction of drug removed in one pass (the basis of hepatic
//   clearance). C_out is the venous (centrilobular) concentration.
// ---------------------------------------------------------------------------
struct SinusoidResult {
    double C_out = 0.0;              // outlet (centrilobular) concentration (uM)
    double extraction_ratio = 0.0;  // E = (C_in - C_out)/C_in, dimensionless in [0,1]
};

// ---------------------------------------------------------------------------
// integrate_sinusoid: march the concentration from the inlet (x=0, C=C_in) to
//   the outlet (x=L) with classical 4th-order Runge-Kutta in SPACE. RK4 evaluates
//   dC/dx at four sub-points per step and combines them with the 1/6,2/6,2/6,1/6
//   weights; its O(h^4) accuracy lets a coarse grid resolve the smooth profile.
//   We integrate in the fractional coordinate xfrac in [0,1] with step h=1/nseg,
//   so the physical step is dx = L*h and dC = (dC/dx)*dx = (dC/dx)*L*h. Shared by
//   the CPU reference (loops this) and the GPU kernel (one thread calls it once).
//
//   `v` is THIS sinusoid's blood velocity -- the per-thread ensemble parameter.
// ---------------------------------------------------------------------------
PERF_HD inline SinusoidResult integrate_sinusoid(const SinusoidParams& p, double v) {
    double C = p.C_in;                    // concentration marched downstream (uM)
    const double h = 1.0 / p.nseg;        // step in fractional position xfrac
    const double Lh = p.L * h;            // physical step dx = L * h (mm)

    for (int s = 0; s < p.nseg; ++s) {
        double xf = static_cast<double>(s) / p.nseg;   // fractional position at step start
        // RK4 slopes in dC/dx; multiply by dx=Lh at the end to get the increment.
        double k1 = dCdx(p, C,               xf,            v);
        double k2 = dCdx(p, C + 0.5*Lh*k1,   xf + 0.5*h,    v);
        double k3 = dCdx(p, C + 0.5*Lh*k2,   xf + 0.5*h,    v);
        double k4 = dCdx(p, C +     Lh*k3,   xf +     h,    v);
        C += (Lh / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        if (C < 0.0) C = 0.0;             // physical floor: a drug cannot go negative
    }

    SinusoidResult out;
    out.C_out = C;
    out.extraction_ratio = (p.C_in > 0.0) ? (p.C_in - C) / p.C_in : 0.0;  // fraction cleared in one pass
    return out;
}
