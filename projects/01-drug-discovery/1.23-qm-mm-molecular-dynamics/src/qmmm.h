// ===========================================================================
// src/qmmm.h  --  Shared (host + device) QM/MM physics + velocity-Verlet core
// ---------------------------------------------------------------------------
// Project 1.23 : QM/MM Molecular Dynamics   (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS PROJECT COMPUTES (and the honest scope)
//   Real QM/MM (AMBER+QUICK, TeraChem, CP2K) treats a 50-200 atom reactive
//   region with full DFT -- thousands of basis functions and electron-repulsion
//   integrals (ERIs) re-evaluated every femtosecond. That is genuinely
//   research-grade and far beyond a single didactic file. Per CLAUDE.md §13 we
//   ship the SMALLEST version that still teaches the load-bearing ideas, and we
//   describe the full machinery in THEORY.md "Where this sits in the real world".
//
//   The teaching model: a PROTON-TRANSFER reaction (the textbook QM/MM use case
//   -- enzyme catalysis, proton wires). The QM "region" is ONE light particle
//   (the proton) whose electronic structure is a TWO-STATE quantum system:
//     * state |L> = proton bonded to the donor  (left well)
//     * state |R> = proton bonded to the acceptor (right well)
//   The MM region is a classical solvent field that does NOT move here but
//   POLARIZES the QM Hamiltonian -- this is "electrostatic embedding", the
//   single most important QM/MM coupling concept (catalog key algorithm).
//
//   Every MD step, for proton coordinate x we:
//     1. Build a 2x2 QM Hamiltonian H(x, phi_MM)            [build_hamiltonian]
//     2. Diagonalize it ANALYTICALLY (closed form, exact)   [ground_state]
//        -> adiabatic ground-state energy E(x) and gap.
//     3. Add the classical MM potential, get the total force [total_force]
//     4. Advance (x, v) by one velocity-Verlet step          [verlet_step]
//   This is "AIMD in the QM region with Verlet MM": the nuclear motion is
//   classical (Verlet), but the force comes from a QM electronic-structure solve
//   at every step. We just replaced a 10^4-basis DFT solve with an exactly
//   solvable 2x2 model so a learner can see the whole loop end to end.
//
//   ENSEMBLE: we run MANY independent trajectories (a sweep over MM field
//   strength and initial proton displacement) -- each is one GPU thread. This is
//   the ensemble-integration pattern (PATTERNS.md §1; cf. 9.02 SEIR, 13.02 PBPK).
//
//   The per-step physics AND the Verlet integrator live here as __host__
//   __device__ inline functions so the CPU reference and the GPU kernel run
//   BYTE-FOR-BYTE identical math -> results agree to round-off. QMMM_HD expands
//   to __host__ __device__ under nvcc, to nothing under the host compiler.
//   Keep this header free of CUDA-only types (no __global__) so reference_cpu.cpp
//   can include it through the plain C++ compiler.
//
// UNITS (a self-consistent "model" unit system, NOT real atomic units; the point
//   is the algorithm, not a calibrated PES):
//     length      = Angstrom-like reaction coordinate
//     energy      = eV-like model energy
//     mass        = proton mass set to PROTON_MASS below (model value)
//     time        = chosen so the Verlet step is stable
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::fabs, std::exp  (host); device gets the CUDA math

// ---------------------------------------------------------------------------
// QMMM_HD: the host/device parity switch (PATTERNS.md §2). Under nvcc the
// functions compile for BOTH the CPU and the GPU; under cl.exe/g++ the
// decorators simply vanish, so reference_cpu.cpp links cleanly.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define QMMM_HD __host__ __device__
#else
#define QMMM_HD
#endif

// ---------------------------------------------------------------------------
// Physical/model constants (compile-time so the optimizer can fold them).
//   These define the double-well electronic structure and the nuclear dynamics.
//   They are MODEL values chosen to give a clean, interpretable barrier-crossing
//   -- see THEORY.md for how each maps to a real proton-transfer surface.
// ---------------------------------------------------------------------------
namespace qmmm {

// --- Electronic-structure (QM) parameters -------------------------------------
// EPS0     : on-site energy of each diabatic state at the bottom of its well.
// KWELL    : curvature (stiffness) of each diabatic parabola; the diabatic state
//            |L> has energy EPS0 + 0.5*KWELL*(x - X_L)^2, and |R> mirror-image.
// X_L, X_R : the donor/acceptor minima of the two diabatic wells (the proton sits
//            near X_L bonded to the donor, near X_R bonded to the acceptor).
// COUPLING : the electronic coupling (tunneling matrix element) between |L> and
//            |R>. It is the OFF-DIAGONAL of H; it splits the two diabatic curves
//            into adiabatic surfaces and SETS THE BARRIER HEIGHT. The two diabatic
//            parabolas cross at x=0 with energy 0.5*KWELL*(X_R)^2; the adiabatic
//            ground state at the crossing is lowered by COUPLING, so the barrier
//            is roughly 0.5*KWELL*X_R^2 - COUPLING. We choose COUPLING small
//            compared to that crossing energy so a GENUINE double well (a real
//            barrier near x=0) survives -- otherwise the surface collapses to a
//            single well and there is nothing to "transfer" across.
// QPROTON  : the partial charge the MM field couples to (electrostatic embedding).
static constexpr double EPS0     = 0.0;     // model energy zero of each well
static constexpr double KWELL    = 30.0;    // diabatic well curvature  (energy/length^2)
static constexpr double X_L      = -0.6;    // donor-side minimum   (length)
static constexpr double X_R      =  0.6;    // acceptor-side minimum (length)
static constexpr double COUPLING = 2.0;     // electronic coupling beta (energy); barrier ~ 0.5*30*0.36 - 2 = 3.4
static constexpr double QPROTON  = 1.0;     // QM charge seen by the MM field

// --- Nuclear-dynamics (MM/Verlet) parameters ----------------------------------
// PROTON_MASS : model nuclear mass for the moving proton (heavier -> slower).
// MM_K        : a classical harmonic tether from the MM environment that keeps the
//               proton near the reaction axis (a stand-in for the protein scaffold
//               / solvent cage). This is the purely-classical "MM force" added on
//               top of the QM force; it also bounds the motion so a Verlet run with
//               no thermostat conserves total energy well (see THEORY.md §5).
static constexpr double PROTON_MASS = 8.0;  // model mass (heavy enough for a stable dt)
static constexpr double MM_K        = 2.0;  // MM harmonic tether stiffness

// ---------------------------------------------------------------------------
// diabatic_energies: the two UNCOUPLED (diabatic) state energies at proton
//   position x, INCLUDING the electrostatic-embedding shift from the MM field.
//
//   ELECTROSTATIC EMBEDDING (the heart of QM/MM): the MM environment enters the
//   QM Hamiltonian as an external electrostatic potential. We model the MM field
//   as a uniform bias `field` (units energy/length) that lowers the energy of
//   whichever side the field favors -- exactly how a charged residue or solvent
//   dipole tilts a reaction's free-energy surface. The embedding term is
//   `QPROTON * field * x` (charge x potential), added to BOTH diabatic curves.
//
//   This asymmetry is what turns a symmetric double well into a downhill
//   (exergonic) or uphill (endergonic) transfer -- the chemistry the enzyme
//   tunes. We rejected hard-coding the asymmetry into EPS0 because the WHOLE
//   point of QM/MM is that the environment, not the QM region, sets it.
//
//   eL = left  diabatic state energy  (proton on donor)
//   eR = right diabatic state energy  (proton on acceptor)
// ---------------------------------------------------------------------------
QMMM_HD inline void diabatic_energies(double x, double field, double& eL, double& eR) {
    const double embed = QPROTON * field * x;          // electrostatic-embedding shift
    eL = EPS0 + 0.5 * KWELL * (x - X_L) * (x - X_L) + embed;
    eR = EPS0 + 0.5 * KWELL * (x - X_R) * (x - X_R) + embed;
}

// ---------------------------------------------------------------------------
// ground_state_energy: the ADIABATIC ground-state energy E_-(x) -- the lower
//   eigenvalue of the 2x2 QM Hamiltonian
//        H = [ eL        COUPLING ]
//            [ COUPLING  eR       ]
//   For a real symmetric 2x2 matrix the eigenvalues are available IN CLOSED FORM
//   (no iterative diagonalizer, no cuSOLVER needed -- and the closed form is
//   exact, which keeps CPU and GPU bit-comparable):
//        E_+- = (eL+eR)/2  ±  sqrt( ((eL-eR)/2)^2 + COUPLING^2 )
//   The "-" root is the electronic ground state the nuclei move on (Born-
//   Oppenheimer): at each x the electrons relax instantly and we take E_-(x).
//
//   The square-root term is half the ADIABATIC GAP between ground and excited
//   surfaces; we return it via `half_gap` because the minimum gap along a
//   trajectory measures how close the dynamics come to a non-adiabatic
//   (surface-hopping) regime -- a real diagnostic in proton-coupled electron
//   transfer.  half_gap is always >= COUPLING > 0, so the surfaces never touch.
// ---------------------------------------------------------------------------
QMMM_HD inline double ground_state_energy(double x, double field, double& half_gap) {
    double eL, eR;
    diabatic_energies(x, field, eL, eR);
    const double avg  = 0.5 * (eL + eR);
    const double diff = 0.5 * (eL - eR);
    half_gap = std::sqrt(diff * diff + COUPLING * COUPLING);  // >= |COUPLING|
    return avg - half_gap;                                    // lower (ground) root
}

// ---------------------------------------------------------------------------
// qm_force: the force on the proton from the QM ground-state surface,
//   F_QM = -dE_-/dx, computed by an analytic derivative of the closed-form
//   eigenvalue (cheaper and more accurate than a finite difference, and it keeps
//   CPU==GPU exact). Differentiating
//        E_-(x) = avg(x) - sqrt(diff(x)^2 + COUPLING^2)
//   with  avg' = 0.5*(eL'+eR'),  diff' = 0.5*(eL'-eR'):
//        dE_-/dx = avg' - (diff*diff') / sqrt(diff^2 + COUPLING^2)
//   The diabatic derivatives include the embedding term's slope (QPROTON*field):
//        eL' = KWELL*(x - X_L) + QPROTON*field
//        eR' = KWELL*(x - X_R) + QPROTON*field
// ---------------------------------------------------------------------------
QMMM_HD inline double qm_force(double x, double field) {
    double eL, eR;
    diabatic_energies(x, field, eL, eR);
    const double dL = KWELL * (x - X_L) + QPROTON * field;   // d(eL)/dx
    const double dR = KWELL * (x - X_R) + QPROTON * field;   // d(eR)/dx
    const double avg_p  = 0.5 * (dL + dR);
    const double diff   = 0.5 * (eL - eR);
    const double diff_p = 0.5 * (dL - dR);
    const double root   = std::sqrt(diff * diff + COUPLING * COUPLING);
    const double dE     = avg_p - (diff * diff_p) / root;    // dE_-/dx
    return -dE;                                              // force = -gradient
}

// ---------------------------------------------------------------------------
// mm_force: the classical MM contribution. Here it is a weak harmonic tether to
//   the origin (the protein/solvent cage), F_MM = -MM_K * x. In a full QM/MM run
//   this is where the bonded + Lennard-Jones + MM-MM electrostatics of thousands
//   of atoms would live (pmemd.cuda territory); we keep one analytic term so the
//   "MM force added to the QM force" structure is explicit.
// ---------------------------------------------------------------------------
QMMM_HD inline double mm_force(double x) {
    return -MM_K * x;
}

// ---------------------------------------------------------------------------
// total_force: the QM/MM force = QM (electronic) force + MM (classical) force.
//   This single sum is the whole point of QM/MM force evaluation: a quantum
//   force on the reactive coordinate plus a classical force from the environment.
// ---------------------------------------------------------------------------
QMMM_HD inline double total_force(double x, double field) {
    return qm_force(x, field) + mm_force(x);
}

// ---------------------------------------------------------------------------
// total_potential: the QM/MM potential energy = QM ground-state energy + MM
//   harmonic energy. Used for an energy-conservation diagnostic (a Verlet run
//   with no thermostat should conserve total energy to integrator order).
// ---------------------------------------------------------------------------
QMMM_HD inline double total_potential(double x, double field, double& half_gap) {
    const double e_qm = ground_state_energy(x, field, half_gap);
    const double e_mm = 0.5 * MM_K * x * x;
    return e_qm + e_mm;
}

// ---------------------------------------------------------------------------
// verlet_step: ONE velocity-Verlet step advancing (x, v) by dt. Velocity Verlet
//   is the standard MD integrator: it is time-reversible and symplectic, so it
//   conserves energy far better than naive Euler over long runs (THEORY.md §5).
//   The catalog calls this "Verlet MM".
//
//   Given force F = m*a, with a = F/m:
//       x(t+dt) = x + v*dt + 0.5*a*dt^2
//       a_new   = F(x_new)/m                 (the QM solve happens HERE)
//       v(t+dt) = v + 0.5*(a + a_new)*dt
//   Passing `accel` in/out lets the caller reuse the force from the previous step
//   (one QM evaluation per step, not two). On the first step the caller seeds
//   accel = total_force(x0, field)/PROTON_MASS.
// ---------------------------------------------------------------------------
QMMM_HD inline void verlet_step(double& x, double& v, double& accel, double field, double dt) {
    // 1) drift position using the CURRENT acceleration
    x += v * dt + 0.5 * accel * dt * dt;
    // 2) NEW force at the new position -> the per-step QM electronic-structure solve
    const double a_new = total_force(x, field) / PROTON_MASS;
    // 3) kick velocity with the AVERAGE of old and new acceleration (the Verlet trick)
    v += 0.5 * (accel + a_new) * dt;
    // 4) hand the new acceleration back so the next step needs only one force eval
    accel = a_new;
}

// ---------------------------------------------------------------------------
// TrajResult: the per-trajectory summary the analysis reports. Plain old data so
//   it copies trivially between host and device.
// ---------------------------------------------------------------------------
struct TrajResult {
    double final_x;       // proton position at the end of the run
    double final_energy;  // total QM/MM potential energy at the end
    double min_gap;       // smallest adiabatic gap (= 2*half_gap) seen en route
    double frac_product;  // fraction of steps spent on the product (x>0) side
    int    transferred;   // 1 if the proton ended on the acceptor side (x>0), else 0
};

// ---------------------------------------------------------------------------
// integrate_trajectory: run ONE QM/MM trajectory to completion and summarize it.
//   This is the function the CPU reference loops over and the GPU kernel calls
//   from one thread -- the shared "one true" simulation.
//
//   Parameters:
//     x0     : initial proton position (length)
//     v0     : initial proton velocity (length/time)
//     field  : MM electrostatic-embedding field strength (energy/length); this is
//              what the ensemble sweeps -- a more negative field favors transfer.
//     dt     : Verlet timestep
//     steps  : number of Verlet steps
//
//   Returns a TrajResult. The math is identical on CPU and GPU because every
//   operation here is from this header (PATTERNS.md §2).
// ---------------------------------------------------------------------------
QMMM_HD inline TrajResult integrate_trajectory(double x0, double v0, double field,
                                              double dt, int steps) {
    double x = x0;
    double v = v0;
    // Seed the acceleration with the force at the initial configuration so the
    // first verlet_step is correct (velocity Verlet needs a(t) before stepping).
    double accel = total_force(x, field) / qmmm::PROTON_MASS;

    // Initialize the running minimum adiabatic gap with the gap at t=0.
    // ground_state_energy writes half_gap (half the ground/excited splitting);
    // the full gap is 2*half_gap, which is what we track and report.
    double half_gap;
    ground_state_energy(x, field, half_gap);   // fills half_gap at the start point
    double min_gap = 2.0 * half_gap;            // smallest gap seen so far (gap at t=0)
    long   product_steps = 0;                   // count of steps with the proton at x>0

    for (int s = 0; s < steps; ++s) {
        verlet_step(x, v, accel, field, dt);
        // Track the minimum adiabatic gap (closest approach to non-adiabaticity).
        ground_state_energy(x, field, half_gap);
        const double gap = 2.0 * half_gap;
        if (gap < min_gap) min_gap = gap;
        // Count time spent on the product side of the barrier (x > 0).
        if (x > 0.0) ++product_steps;
    }

    TrajResult out;
    out.final_x      = x;
    out.final_energy = total_potential(x, field, half_gap);
    out.min_gap      = min_gap;
    out.frac_product = static_cast<double>(product_steps) / static_cast<double>(steps);
    out.transferred  = (x > 0.0) ? 1 : 0;
    return out;
}

}  // namespace qmmm
