// ===========================================================================
// src/flash.h  --  Shared (host + device) FLASH radiation-chemistry ODE core
// ---------------------------------------------------------------------------
// Project 5.12 : FLASH Radiotherapy GPU Modeling
//
// WHAT THIS PROJECT COMPUTES (a REDUCED-SCOPE, TEACHING model of the FLASH effect)
//   FLASH radiotherapy (FLASH-RT) delivers a whole treatment dose in a few
//   millisecond pulses at ULTRA-HIGH DOSE RATE (UHDR, > 40 Gy/s), and -- in
//   animal studies -- spares normal tissue while still controlling the tumour.
//   One leading mechanistic hypothesis is TRANSIENT OXYGEN DEPLETION plus
//   altered radical chemistry: the radiation splits water into reactive
//   radicals; those radicals either recombine harmlessly (radical-radical) or
//   react with molecular oxygen to "fix" (make permanent) DNA damage
//   (the classic "oxygen fixation hypothesis"). At UHDR the radicals appear
//   almost instantaneously and locally consume the available O2 faster than the
//   vasculature can resupply it, so LESS oxygen is present to fix damage -> less
//   biological damage in (already fairly oxygenated) normal tissue.
//
//   We model, PER TISSUE VOXEL, the coupled chemistry AFTER a pulse train, as a
//   small system of ordinary differential equations (ODEs) in two lumped
//   concentrations:
//
//       R    = reactive-radical concentration           [micromol/L = uM]
//       O2   = dissolved molecular-oxygen concentration [uM]  (pO2 in mmHg maps
//              to uM; see po2_mmHg_to_uM below)
//
//   Radicals are injected in short bursts (the pulses); between and after the
//   pulses the chemistry relaxes. The ensemble sweeps two axes that matter
//   clinically:
//       * initial oxygen tension pO2 (hypoxic tumour core .. oxygenated normal
//         tissue), and
//       * delivery mode (CONVENTIONAL low dose-rate vs FLASH UHDR), which
//         changes ONLY how the same total dose is packed in time.
//
//   From each voxel's trajectory we score the OXYGEN-FIXED DAMAGE (the integral
//   of the radical-oxygen fixation reaction over time) and form an
//   OXYGEN ENHANCEMENT RATIO (OER)-like number. Comparing FLASH vs CONVENTIONAL
//   at the same pO2 reproduces the qualitative FLASH signature: a normal-tissue
//   SPARING that shrinks toward zero in already-hypoxic tissue.
//
//   >>> THIS IS EDUCATIONAL, NOT CLINICAL. The rate constants are illustrative
//   >>> lumped values chosen so the demo is interpretable; they are NOT fitted to
//   >>> patient data and MUST NOT be used for any treatment decision. The full
//   >>> problem (Geant4-DNA / MPEXS track-structure Monte Carlo radiolysis with
//   >>> dozens of species and spatial diffusion) is described in THEORY.md.
//
//   The derivative AND the RK4 step live here as __host__ __device__ inline
//   functions so the CPU reference and the GPU kernel integrate IDENTICALLY ->
//   their results match to round-off. FLASH_HD expands to __host__ __device__
//   under nvcc, and to nothing under the plain host compiler.
//
// READ THIS AFTER: README.md; alongside THEORY.md (the derivation + numerics).
// READ THIS BEFORE: reference_cpu.h, kernels.cuh (they build on this core).
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// The host/device portability macro (PATTERNS.md §2, the "HD" idiom).
//   * Under nvcc (__CUDACC__ defined) every function below is compiled for BOTH
//     the CPU and the GPU, so the exact same machine formula runs on each side.
//   * Under the plain host compiler (cl.exe building reference_cpu.cpp) the
//     decorators do not exist, so we erase them.
//   Keep this header free of CUDA-only types (no __global__, no cudaXxx) so the
//   host compiler can include it.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define FLASH_HD __host__ __device__
#else
#define FLASH_HD
#endif

// ---------------------------------------------------------------------------
// Physical/chemical constants for the lumped model (ILLUSTRATIVE teaching
// values, not fitted). Units are documented on every line. These are shared by
// CPU and GPU so both integrate the identical system.
// ---------------------------------------------------------------------------

// Convert oxygen partial pressure (pO2, clinically reported in mmHg) to a
// dissolved-O2 concentration in micromol/L. At body temperature the solubility
// of O2 in water/tissue is ~1.4 uM per mmHg (Henry's law, order-of-magnitude).
// We keep this simple linear map so pO2 inputs read in familiar mmHg.
FLASH_HD inline double po2_mmHg_to_uM(double po2_mmHg) {
    return 1.4 * po2_mmHg;                 // [uM] = 1.4 [uM/mmHg] * [mmHg]
}

// The oxygen enhancement ratio (OER) as a function of local O2 -- the second
// pillar of the model (the radiobiology). The classic Alper-Howard-Flanders
// picture says radiation is MORE damaging when oxygen is present ("oxygen fixes
// the damage"): OER rises smoothly from 1 in anoxia to a maximum m (~3) when
// fully oxygenated, with a half-maximum at O2 = K. The normalized form is
//     OER(O2) = (m*O2 + K) / (O2 + K)
//   * O2 = 0   ->  K/K = 1           (anoxic: least radiosensitive)
//   * O2 = K   ->  (m+1)/2           (half-way up the S-curve)
//   * O2 >> K  ->  m                 (fully oxygenated: most radiosensitive)
// This is the S-shaped sensitizer curve whose STEEP mid-region is exactly where
// a small FLASH-induced O2 depletion buys the largest sparing.
FLASH_HD inline double oer(double O2_uM, double m, double K_uM) {
    return (m * O2_uM + K_uM) / (O2_uM + K_uM);
}

// Rate constants of the two lumped radical reactions + the O2 resupply, plus the
// OER parameters. Chosen (ILLUSTRATIVE, not fitted) so that:
//   * radicals consume O2 quickly (k_ro large) AND clear within a CONVENTIONAL
//     inter-pulse gap (k_rr moderate), and the vasculature REFILLS O2 quickly
//     between conventional pulses (k_diff large, recovery time ~1/k_diff ~= 3 ms
//     << the ~40 ms conventional gap) -- so conventional delivery irradiates at
//     near-AMBIENT O2 (the radicals of each pulse see a fully re-oxygenated
//     voxel), while
//   * a FLASH/UHDR pulse train (gap ~= dt, ~10 us << 3 ms recovery) dumps all
//     the radicals before O2 can be resupplied, so they collectively DEPLETE O2
//     and see a much LOWER effective O2 during irradiation.
// Because damage is scored through OER(effective O2 during irradiation), the
// FLASH depletion lowers the effective OER and hence the damage -- the modelled
// FLASH sparing (conv_damage / flash_damage > 1). See THEORY.md for the full
// derivation and the parameter rationale.
struct FlashRates {
    double k_rr;    // radical-radical recombination rate   [1/(uM*s)] (2nd order)
    double k_ro;    // radical-oxygen consumption rate       [1/(uM*s)] (2nd order)
    double k_diff;  // O2 resupply (diffusion from vessels)  [1/s]      (1st order)
    double g_rad;   // radical yield per unit dose           [uM/Gy] (G-value-like)
    double oer_max; // OER plateau m (fully-oxygenated sensitivity, dimensionless)
    double oer_K;   // O2 at OER half-maximum [uM]
};

// A sensible default parameter set for the demo. `static` + `inline` (C++17)
// gives one shared definition usable from both translation units. These values
// produce a clean, interpretable FLASH signature over pO2 = 2..40 mmHg (the
// tumour-hypoxic .. normal-tissue range): FLASH spares tissue at every oxygen
// level (sparing > 1), with the LARGEST relative sparing (~1.15x) at low
// oxygenation, shrinking toward ~1.05x as the tissue becomes well oxygenated
// (where ample O2 remains even after depletion).
FLASH_HD inline FlashRates default_rates() {
    FlashRates r;
    r.k_rr    = 1.0e-1;  // radicals recombine fast enough to clear within a conv gap
    r.k_ro    = 3.0e0;   // radicals consume O2 quickly (drives UHDR depletion)
    r.k_diff  = 3.0e2;   // fast O2 refill: time-constant ~1/300 s ~= 3 ms
    r.g_rad   = 4.0e1;   // lumped radical yield [uM/Gy] (amplified for a visible effect)
    r.oer_max = 3.0;     // maximum OER ~3 (well-oxygenated tissue) -- textbook value
    r.oer_K   = 4.2;     // half-max at ~3 mmHg (3 mmHg * 1.4 uM/mmHg = 4.2 uM)
    return r;
}

// ---------------------------------------------------------------------------
// The state integrated per voxel: the two chemical concentrations, plus two
// running sums that let us form the RADICAL-WEIGHTED AVERAGE OXYGEN -- the
// "effective O2" the DNA sees WHILE damage is being fixed.
//   R  : radical concentration [uM]
//   O2 : oxygen concentration  [uM]
//   wO2 : time-integral of (R * O2)  -- numerator of the weighted O2 average
//   wR  : time-integral of (R)       -- denominator (total radical-exposure)
// We weight by R because oxygen only matters for damage fixation WHERE/WHEN
// radicals are present; averaging O2 weighted by radical concentration captures
// "the O2 the radicals actually experienced". effective_O2 = wO2/wR.
// ---------------------------------------------------------------------------
struct ChemState {
    double R;
    double O2;
    double wO2;   // sum of R*O2 * dt  (radical-weighted O2, numerator)
    double wR;    // sum of R    * dt  (radical exposure,     denominator)
};

// Derivative of the chemistry (the ODE right-hand side). Given the current
// state and the rate constants, write dR/dt, dO2/dt and the two weight fluxes.
//   dR/dt   = -2 k_rr R^2         (two radicals lost per recombination)
//             - k_ro R O2         (radical consumed while consuming an O2)
//   dO2/dt  = -k_ro R O2          (each radical-O2 reaction consumes one O2)
//             + k_diff (O2sup-O2) (linear resupply toward the local supply level)
//   dwO2/dt =  R * O2             (accumulate radical-weighted O2)
//   dwR/dt  =  R                  (accumulate radical exposure)
// The pulse source term (radical injection) is applied SEPARATELY as discrete
// deposits in integrate_voxel(), not here, so this RHS stays a clean autonomous
// system that RK4 can evaluate at sub-steps.
FLASH_HD inline void chem_deriv(const ChemState& s, const FlashRates& k, double O2_supply,
                                double& dR, double& dO2, double& dwO2, double& dwR) {
    const double recomb = k.k_rr * s.R * s.R;    // radical-radical loss    [uM/s]
    const double cons   = k.k_ro * s.R * s.O2;   // radical-oxygen reaction [uM/s]
    dR   = -2.0 * recomb - cons;                 // radicals disappear both ways
    dO2  = -cons + k.k_diff * (O2_supply - s.O2);// consumed by radicals, resupplied by diffusion
    dwO2 = s.R * s.O2;                            // radical-weighted O2 accumulator
    dwR  = s.R;                                   // radical-exposure accumulator
}

// One classical 4th-order Runge-Kutta (RK4) step of size dt advancing the state
// in place. RK4 samples the derivative at four points and combines them for
// O(dt^4) local accuracy -- accurate and stable for this smooth, stiff-ish
// chemistry at the timesteps we use. Writing it out (rather than looping) keeps
// every term visible for the learner.
FLASH_HD inline void chem_rk4_step(ChemState& s, const FlashRates& k, double O2_supply, double dt) {
    double a1, b1, c1, d1;  chem_deriv(s, k, O2_supply, a1, b1, c1, d1);

    ChemState s2{ s.R + 0.5*dt*a1, s.O2 + 0.5*dt*b1, s.wO2 + 0.5*dt*c1, s.wR + 0.5*dt*d1 };
    double a2, b2, c2, d2;  chem_deriv(s2, k, O2_supply, a2, b2, c2, d2);

    ChemState s3{ s.R + 0.5*dt*a2, s.O2 + 0.5*dt*b2, s.wO2 + 0.5*dt*c2, s.wR + 0.5*dt*d2 };
    double a3, b3, c3, d3;  chem_deriv(s3, k, O2_supply, a3, b3, c3, d3);

    ChemState s4{ s.R + dt*a3, s.O2 + dt*b3, s.wO2 + dt*c3, s.wR + dt*d3 };
    double a4, b4, c4, d4;  chem_deriv(s4, k, O2_supply, a4, b4, c4, d4);

    s.R   += (dt/6.0) * (a1 + 2.0*a2 + 2.0*a3 + a4);
    s.O2  += (dt/6.0) * (b1 + 2.0*b2 + 2.0*b3 + b4);
    s.wO2 += (dt/6.0) * (c1 + 2.0*c2 + 2.0*c3 + c4);
    s.wR  += (dt/6.0) * (d1 + 2.0*d2 + 2.0*d3 + d4);

    // Concentrations cannot go negative physically; clamp tiny RK4 undershoots so
    // a following R^2 term can never blow up. (Both CPU and GPU clamp identically,
    // so this does NOT break bit-for-bit parity.)
    if (s.R  < 0.0) s.R  = 0.0;
    if (s.O2 < 0.0) s.O2 = 0.0;
}

// ---------------------------------------------------------------------------
// Per-voxel job description: everything one ensemble member needs.
//   The delivery is a train of `n_pulses` identical radical deposits (one per
//   beam pulse), separated by `pulse_gap_s` seconds. Total injected radicals =
//   n_pulses * dose_per_pulse * g_rad. FLASH vs CONVENTIONAL differ ONLY in
//   pulse_gap_s (UHDR = tiny gaps -> pulses pile up before O2 can refill;
//   conventional = long gaps -> O2 refills between pulses).
// ---------------------------------------------------------------------------
struct VoxelJob {
    double po2_mmHg;      // initial (and supply) oxygen tension [mmHg]
    double dose_per_pulse;// Gy deposited per pulse
    int    n_pulses;      // number of pulses in the train
    double pulse_gap_s;   // seconds between pulse onsets (delivery-mode knob)
    double dt;            // RK4 timestep [s]
    int    steps_per_gap; // integration sub-steps between consecutive pulses
    int    relax_steps;   // extra steps integrated AFTER the last pulse (relaxation)
    FlashRates k;         // rate constants
};

// Per-voxel result the analysis cares about (all deterministic doubles).
struct VoxelResult {
    double fixed_damage;  // OER-weighted damage = dose * OER(effective O2) [Gy-equiv]
    double min_O2;        // lowest O2 reached during delivery [uM] (depletion depth)
    double eff_O2;        // radical-weighted effective O2 the DNA "saw" [uM]
};

// Integrate ONE voxel through its whole pulse train + relaxation and return its
// summary. This is the single function shared by the CPU reference (looped over
// all voxels) and the GPU kernel (one thread per voxel) -- guaranteeing
// identical arithmetic on both sides.
//
//   Algorithm:
//     O2_supply <- pO2 converted to uM (the level the vasculature refills toward)
//     state <- (R=0, O2=O2_supply, weight sums=0)
//     for each pulse p:
//         inject radicals:  R += dose_per_pulse * g_rad     (instantaneous deposit)
//         integrate steps_per_gap RK4 steps of dt           (chemistry between pulses)
//         track min O2, accumulate radical-weighted O2
//     integrate relax_steps more RK4 steps                  (post-delivery relaxation)
//     effective_O2 <- wO2 / wR                              (radical-weighted mean O2)
//     damage       <- total_dose * OER(effective_O2)        (radiobiological score)
//
//   The FLASH effect is EMERGENT, not hard-coded: with tiny gaps (UHDR) the
//   injected radical spikes overlap and drive O2 low WHILE radicals are present,
//   so the radical-weighted effective O2 is depressed -> a lower OER -> less
//   damage than the SAME dose delivered slowly (where O2 refills between pulses
//   and the radicals see near-ambient O2). See THEORY.md for the full argument.
FLASH_HD inline VoxelResult integrate_voxel(const VoxelJob& j) {
    const double O2_supply = po2_mmHg_to_uM(j.po2_mmHg);   // resupply target [uM]
    ChemState s{ 0.0, O2_supply, 0.0, 0.0 };               // start oxygenated, no radicals
    double min_O2 = s.O2;                                  // track deepest depletion

    const double inject     = j.dose_per_pulse * j.k.g_rad;      // radicals per pulse [uM]
    const double total_dose = j.dose_per_pulse * j.n_pulses;     // Gy delivered overall

    for (int p = 0; p < j.n_pulses; ++p) {
        s.R += inject;                                     // deposit this pulse's radicals
        for (int t = 0; t < j.steps_per_gap; ++t) {
            chem_rk4_step(s, j.k, O2_supply, j.dt);        // relax chemistry for one sub-step
            if (s.O2 < min_O2) min_O2 = s.O2;              // record depletion low-water mark
        }
    }
    // Post-delivery relaxation: let residual radicals finish reacting so the
    // weighted-O2 average includes the tail of radical activity.
    for (int t = 0; t < j.relax_steps; ++t) {
        chem_rk4_step(s, j.k, O2_supply, j.dt);
        if (s.O2 < min_O2) min_O2 = s.O2;
    }

    // Radical-weighted effective O2 (guard the degenerate wR==0 case that would
    // only occur if no radicals were ever injected).
    const double eff_O2 = (s.wR > 0.0) ? (s.wO2 / s.wR) : O2_supply;

    VoxelResult out;
    out.fixed_damage = total_dose * oer(eff_O2, j.k.oer_max, j.k.oer_K);
    out.min_O2       = min_O2;
    out.eff_O2       = eff_O2;
    return out;
}
