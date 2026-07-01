// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference (the "twin fit" loop)
// ---------------------------------------------------------------------------
// Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
//
// THE INFERENCE STEP, IN MINIATURE
//   Building a digital twin means running MANY forward heart simulations while
//   varying a physiological parameter until the model's output matches a
//   patient's measurement. Here the swept parameter is CONTRACTILITY (the peak
//   systolic elastance E_max, mmHg/mL) -- the single most clinically important
//   knob of pump strength -- and the target measurement is STROKE VOLUME (mL).
//
//   So the ensemble is a 1-D sweep of `n` hearts with E_max from emax_lo to
//   emax_hi. main.cu then reports the member whose stroke volume is closest to
//   the target -- a 1-parameter stand-in for the ensemble-Kalman / adjoint
//   calibration a real twin uses. Every member is an INDEPENDENT forward solve,
//   which is exactly the batched-ensemble GPU pattern (PATTERNS.md section 1,
//   "the same ODE for many parameter sets" -> ensemble RK4, thread per member).
//
//   The config + the (index -> E_max) mapping live here and are marked HEART_HD
//   so the GPU kernel reuses them unchanged. The physics itself is in heart.h.
//
// READ THIS AFTER: heart.h ; BEFORE: kernels.cuh, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "heart.h"   // HEART_HD, HeartParams, TwinResult, simulate_heart

// ---------------------------------------------------------------------------
// EnsembleConfig -- one whole ensemble job: the shared integration settings,
//   the baseline heart parameters, the contractility sweep, and the clinical
//   target the twin is being fitted to.
// ---------------------------------------------------------------------------
struct EnsembleConfig {
    HeartParams base;          // baseline physiology (all members share it...)
    int    n        = 0;       // ...except E_max, which is swept over n members
    double emax_lo  = 0.0;     // lowest  contractility in the sweep [mmHg/mL]
    double emax_hi  = 0.0;     // highest contractility in the sweep [mmHg/mL]
    double dt_ms    = 0.0;     // RK4 timestep [ms]
    int    beats    = 0;       // cardiac cycles to simulate (transient + measure)
    double target_sv = 0.0;    // clinical target stroke volume to fit [mL]
};

// Number of ensemble members (virtual hearts).
HEART_HD inline int ensemble_size(const EnsembleConfig& c) { return c.n; }

// ---------------------------------------------------------------------------
// member_params -- build the HeartParams for ensemble member `idx`.
//   All members copy the baseline physiology; only E_max is overwritten with
//   the idx-th value on the linear sweep emax_lo .. emax_hi. Returned by value
//   (a HeartParams is small) so a kernel thread can hold "its" heart in regs.
// ---------------------------------------------------------------------------
HEART_HD inline HeartParams member_params(const EnsembleConfig& c, int idx) {
    HeartParams p = c.base;                                   // copy the baseline
    const double frac = (c.n > 1) ? (double)idx / (double)(c.n - 1) : 0.0;
    p.E_max = c.emax_lo + (c.emax_hi - c.emax_lo) * frac;     // this member's contractility
    return p;
}

// ---------------------------------------------------------------------------
// load_ensemble -- read an EnsembleConfig from the tiny text sample.
//   Format (whitespace-separated, see data/README.md):
//     n emax_lo emax_hi dt_ms beats target_sv bcl_ms E_min V0 Rp C_art
//   Only the parameters the demo actually varies are exposed; the rest of the
//   physiology uses the documented HeartParams defaults. Throws on a bad file
//   so the demo fails loudly rather than running on garbage.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path);

// ---------------------------------------------------------------------------
// integrate_cpu -- the trusted serial baseline: simulate every member on the
//   CPU (a plain loop) and fill `results` (sized to n). The GPU kernel must
//   reproduce these numbers to round-off. Shares simulate_heart() from heart.h.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<TwinResult>& results);
