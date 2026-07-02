// ===========================================================================
// src/windkessel.h  --  Shared (host + device) physiological model + integrator
// ---------------------------------------------------------------------------
// Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
//
// WHAT LIVES HERE
//   The *per-element physics* that BOTH the CPU reference and the GPU kernel must
//   run byte-for-byte identically. Following the "shared __host__ __device__ core"
//   idiom (docs/PATTERNS.md §2), we put the ODE right-hand side, one RK4 step, and
//   the per-ensemble-member forecast in ONE header. WK_HD expands to
//   `__host__ __device__` under nvcc and to nothing under the plain host compiler,
//   so reference_cpu.cpp (cl.exe) and kernels.cu (nvcc) share the *same math* ->
//   the ensemble forecast is identical on both, and verification is exact.
//
// THE PHYSIOLOGICAL MODEL: the two-element ("RC") WINDKESSEL
//   A classic lumped-parameter model of the systemic arterial system. The aorta +
//   large arteries act like an elastic reservoir (a CAPACITOR of compliance C)
//   draining through the peripheral vasculature (a RESISTOR R). With a known
//   ventricular inflow Q(t) (mL/s) into the reservoir, the aortic pressure P (mmHg)
//   obeys a single first-order ODE:
//
//       C dP/dt = Q(t) - P / R
//
//   * C  = total arterial compliance      (mL/mmHg)  -- "how stretchy" the arteries are
//   * R  = total peripheral resistance     (mmHg*s/mL) -- "how hard" it is to drain
//   * RC = the diastolic pressure-decay time constant (s): during diastole Q=0 and
//          P(t) = P0 * exp(-t/RC), the textbook Windkessel exponential runoff.
//
//   THE INVERSE PROBLEM this project solves: given a NOISY measured aortic-pressure
//   waveform P_obs(t) and the (known) inflow Q(t), estimate the patient-specific
//   (R, C). That is *parameter estimation / data assimilation* -- the catalog task.
//
// WHY augment the STATE with the parameters?
//   The Ensemble Kalman Filter (EnKF) estimates a hidden STATE from observations.
//   We make the unknown PARAMETERS part of the state by augmenting:
//       x = [ P , theta_R , theta_C ]
//   where theta_R = log(R), theta_C = log(C). We carry the parameters in LOG space
//   for two reasons: (1) it guarantees R,C stay strictly positive no matter what
//   the linear Kalman update does, and (2) compliance/resistance vary
//   multiplicatively across patients, so log is the natural scale. The parameters
//   have trivial dynamics (dtheta/dt = 0) -- they only change in the *analysis*
//   step when observations pull them. This "joint state-parameter estimation" is
//   the standard EnKF recipe for calibrating ODE models. See ../THEORY.md §2-3.
//
// READ THIS AFTER: util/cuda_check.cuh ; BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>   // std::exp (host); nvcc maps exp() to the device intrinsic too

// HD-macro idiom (docs/PATTERNS.md §2): decorate shared functions so the SAME
// source compiles for host and device. Keep CUDA-only constructs (no __global__,
// no <<<>>>) out of this header so the host compiler can include it unchanged.
#ifdef __CUDACC__
#define WK_HD __host__ __device__
#else
#define WK_HD
#endif

// Number of augmented state variables: [ P, log R, log C ]. A compile-time
// constant so both host and device size their per-member arrays identically.
#define WK_NSTATE 3

// ---------------------------------------------------------------------------
// wk_inflow: the KNOWN ventricular inflow Q(t) into the arterial reservoir.
//   We model one cardiac cycle of period `T` (s) as a half-sine ejection during
//   systole (the first `t_sys` seconds) and zero during diastole (valve shut):
//
//       Q(t) = Q_peak * sin(pi * tau / t_sys)   for 0 <= tau < t_sys   (systole)
//              0                                 otherwise             (diastole)
//
//   where tau = t mod T is the phase within the current beat. This is a smooth,
//   physiologically-shaped stand-in for a measured aortic flow probe. Q is treated
//   as a *known input* (measured separately), NOT something we estimate -- exactly
//   the setup in cardiovascular parameter-ID studies.
//
//   Parameters (with units so the arithmetic is auditable):
//     t        : absolute time (s)
//     T        : cardiac cycle length (s)          e.g. 0.8 s  (75 bpm)
//     t_sys    : systolic ejection duration (s)    e.g. 0.3 s
//     Q_peak   : peak inflow (mL/s)                e.g. 500 mL/s
//   Returns Q(t) in mL/s.
// ---------------------------------------------------------------------------
WK_HD inline double wk_inflow(double t, double T, double t_sys, double Q_peak) {
    // Reduce to the phase within the current beat. fmod is available on both host
    // and device; the result is in [0, T).
    double tau = t - T * std::floor(t / T);
    if (tau < t_sys) {
        // A half-sine bolus during systole. sin(pi * tau/t_sys) rises from 0, peaks
        // at tau = t_sys/2, and returns to 0 at end-systole -- a clean ejection.
        return Q_peak * std::sin(3.14159265358979323846 * tau / t_sys);
    }
    return 0.0;   // diastole: aortic valve closed, no inflow
}

// ---------------------------------------------------------------------------
// wk_deriv: right-hand side of the augmented ODE, dx/dt.
//   State x = [P, theta_R, theta_C] with R = exp(theta_R), C = exp(theta_C).
//   Pressure evolves by the Windkessel equation; the parameters are static
//   (their derivative is exactly zero -- they are constants of the forecast and
//   only move during the EnKF analysis).
//
//     dP/dt      = ( Q(t) - P/R ) / C
//     dtheta_R/dt = 0
//     dtheta_C/dt = 0
//
//   dx is written in place (length WK_NSTATE). Inflow parameters (T,t_sys,Q_peak)
//   are passed through so the derivative is a pure function of (x, t, inputs).
// ---------------------------------------------------------------------------
WK_HD inline void wk_deriv(const double* x, double t,
                           double T, double t_sys, double Q_peak,
                           double* dx) {
    const double P = x[0];                 // aortic pressure (mmHg)
    const double R = std::exp(x[1]);       // recover R from log-parameter (mmHg*s/mL)
    const double C = std::exp(x[2]);       // recover C from log-parameter (mL/mmHg)
    const double Q = wk_inflow(t, T, t_sys, Q_peak);   // known inflow (mL/s)

    dx[0] = (Q - P / R) / C;   // Windkessel: net volume rate / compliance -> dP/dt
    dx[1] = 0.0;               // log R is a static parameter during the forecast
    dx[2] = 0.0;               // log C is a static parameter during the forecast
}

// ---------------------------------------------------------------------------
// wk_rk4_step: one classical 4th-order Runge-Kutta step of size `dt`.
//   RK4 samples the derivative at four points and combines them for O(dt^4) local
//   error -- accurate and stable for this smooth, non-stiff pressure ODE. We keep
//   the whole augmented state (length WK_NSTATE) in tiny stack arrays so the exact
//   same code runs in a GPU thread's registers/local memory and on the CPU.
//
//   Advances x in place by dt; `t` is the time at the START of the step.
// ---------------------------------------------------------------------------
WK_HD inline void wk_rk4_step(double* x, double t, double dt,
                              double T, double t_sys, double Q_peak) {
    double k1[WK_NSTATE], k2[WK_NSTATE], k3[WK_NSTATE], k4[WK_NSTATE];
    double tmp[WK_NSTATE];

    wk_deriv(x, t, T, t_sys, Q_peak, k1);
    for (int i = 0; i < WK_NSTATE; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
    wk_deriv(tmp, t + 0.5 * dt, T, t_sys, Q_peak, k2);
    for (int i = 0; i < WK_NSTATE; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
    wk_deriv(tmp, t + 0.5 * dt, T, t_sys, Q_peak, k3);
    for (int i = 0; i < WK_NSTATE; ++i) tmp[i] = x[i] + dt * k3[i];
    wk_deriv(tmp, t + dt, T, t_sys, Q_peak, k4);

    for (int i = 0; i < WK_NSTATE; ++i)
        x[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

// ---------------------------------------------------------------------------
// wk_forecast_member: advance ONE ensemble member's augmented state forward over
//   an assimilation window of `nsub` RK4 sub-steps of size `dt`, starting at time
//   `t0`. This is the "forecast" (a.k.a. prediction) half of one EnKF cycle: each
//   member integrates the SAME ODE independently, so it maps to one GPU thread.
//
//   Because the parameters have zero derivative, only x[0] (pressure) actually
//   moves here; the member's (R,C) are held fixed through the window and then
//   corrected in the host analysis step. Keeping the forecast in this shared
//   function is what makes the GPU kernel and CPU reference agree exactly.
//
//   x     : in/out augmented state [P, logR, logC] for this member
//   t0    : window start time (s)
//   dt    : RK4 sub-step (s)
//   nsub  : number of sub-steps in the window (window length = nsub*dt)
//   T,t_sys,Q_peak : inflow-waveform inputs (shared by all members)
// ---------------------------------------------------------------------------
WK_HD inline void wk_forecast_member(double* x, double t0, double dt, int nsub,
                                     double T, double t_sys, double Q_peak) {
    double t = t0;
    for (int s = 0; s < nsub; ++s) {
        wk_rk4_step(x, t, dt, T, t_sys, Q_peak);
        t += dt;
    }
}
