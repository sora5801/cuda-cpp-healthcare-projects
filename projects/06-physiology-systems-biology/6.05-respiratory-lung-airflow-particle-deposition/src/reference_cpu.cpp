// ===========================================================================
// src/reference_cpu.cpp  --  Loader, airway builder, and serial CPU reference
// ---------------------------------------------------------------------------
// Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over particle histories,
//   no parallelism -- so that when the GPU and CPU tallies agree, we believe the
//   GPU. All the per-particle physics lives in lung_physics.h and is SHARED with
//   the GPU kernel, so "agree" here means "bit-for-bit identical".
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, lung_physics.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_problem: parse the one-line, human-friendly sample file into SI units.
//   File format (whitespace separated -- see data/README.md):
//     d_p_microns  rho_p_kg_m3  n_gen  flow_L_per_min  n_particles  seed
//   We accept convenient units (microns, litres/minute) in the file and convert
//   to SI here so the committed sample is easy to read and edit by hand.
// ---------------------------------------------------------------------------
DepositionProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);

    // Read the first NON-COMMENT, non-blank line so the sample can carry a '#'
    // header explaining its own fields (see data/sample/lung_params.txt).
    std::string line;
    bool got = false;
    while (std::getline(in, line)) {
        // Trim leading whitespace to test for a comment marker.
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;   // blank line
        if (line[s] == '#') continue;           // comment line
        line = line.substr(s);
        got = true;
        break;
    }
    if (!got) throw std::runtime_error("no data line found in " + path);

    double d_p_um = 0.0, rho_p = 0.0, flow_lpm = 0.0;
    int n_gen = 0;
    unsigned long long n_particles = 0, seed = 0;
    std::istringstream ls(line);
    if (!(ls >> d_p_um >> rho_p >> n_gen >> flow_lpm >> n_particles >> seed))
        throw std::runtime_error("bad parameters (expected "
            "'d_p_microns rho_p n_gen flow_L_per_min n_particles seed') in " + path);

    // Basic sanity so demos fail loudly rather than producing nonsense.
    if (d_p_um <= 0.0 || rho_p <= 0.0 || n_gen <= 0 || n_gen > lung::MAX_GEN
        || flow_lpm <= 0.0 || n_particles == 0)
        throw std::runtime_error("invalid deposition parameters in " + path);

    DepositionProblem p;
    p.d_p         = d_p_um * 1e-6;              // microns -> metres
    p.rho_p       = rho_p;                      // already SI (kg/m^3)
    p.n_gen       = n_gen;
    p.flow_rate   = flow_lpm * 1e-3 / 60.0;     // L/min -> m^3/s
    p.n_particles = n_particles;
    p.seed        = seed;
    return p;
}

// ---------------------------------------------------------------------------
// build_airway: construct the symmetric bifurcating (Weibel-A) airway tree.
//
//   GEOMETRY (deterministic, pure): starting from the trachea (generation 0)
//   with radius r0 and length L0, each deeper generation is scaled by fixed
//   ratios that approximate Weibel's measured human-lung data:
//       r[g+1] = r[g] * R_RATIO,   L[g+1] = L[g] * L_RATIO.
//   Generation g contains 2^g parallel tubes (the airway bifurcates at each
//   step), so the TOTAL cross-sectional area at generation g is
//       A_tot(g) = 2^g * pi * r[g]^2.
//   Continuity (mass conservation of incompressible air) fixes the mean axial
//   velocity in EACH tube of generation g from the whole-lung flow Q:
//       U[g] = Q / A_tot(g).
//   Because A_tot grows rapidly with depth, U[g] falls sharply -- which is
//   exactly why impaction dominates in the big fast upper airways and diffusion
//   dominates in the slow deep airways (see THEORY.md).
//
//   The constants below are representative adult values; they are approximate on
//   purpose (this is a teaching model, not a patient-specific CT geometry -- the
//   catalog's LIDC-IDRI / COPDGene CT route is described in data/README.md).
// ---------------------------------------------------------------------------
lung::Airway build_airway(const DepositionProblem& prob) {
    lung::Airway aw;
    aw.n_gen = prob.n_gen;

    // Trachea (generation 0) dimensions and the geometric shrink ratios.
    const double r0      = 0.009;   // tracheal radius  ~9 mm                [m]
    const double L0      = 0.120;   // tracheal length  ~12 cm               [m]
    const double R_RATIO = 0.85;    // radius  multiplier per generation
    const double L_RATIO = 0.62;    // length  multiplier per generation

    double r = r0, L = L0;
    for (int g = 0; g < aw.n_gen; ++g) {
        aw.r[g] = r;
        aw.L[g] = L;

        // Total cross-section of all 2^g tubes at this generation. We use
        // ldexp(1.0, g) = 2^g to avoid overflow/precision worries for large g.
        const double n_tubes = std::ldexp(1.0, g);              // 2^g as a double
        const double A_tot   = n_tubes * lung::PI * r * r;      // total area [m^2]
        aw.U[g] = prob.flow_rate / A_tot;                       // per-tube velocity

        r *= R_RATIO;
        L *= L_RATIO;
    }
    return aw;
}

// ---------------------------------------------------------------------------
// deposition_cpu: the serial reference. Track every particle history, one at a
// time, and increment an INTEGER counter for the generation it deposits in
// (index n_gen counts "exhaled"). This mirrors the GPU kernel exactly -- same
// per-particle RNG seeding, same track_particle() -- just serial and with a
// plain ++ instead of atomicAdd. Integer counts commute, so CPU == GPU exactly.
//
//   Complexity: O(n_particles * n_gen) time, O(n_gen) space for the tally.
// ---------------------------------------------------------------------------
void deposition_cpu(const DepositionProblem& prob, const lung::Airway& aw,
                    std::vector<uint64_t>& tally) {
    // n_gen slots for "deposited in generation g" + 1 slot for "exhaled".
    tally.assign(static_cast<std::size_t>(aw.n_gen) + 1, 0ULL);

    // The particle's fixed properties (tau, v_s, D) are the same for every
    // history in this monodisperse aerosol, so compute them once.
    const lung::Particle p = lung::make_particle(prob.d_p, prob.rho_p);

    for (uint64_t i = 0; i < prob.n_particles; ++i) {
        lung::Rng rng = lung::rng_seed(prob.seed, i);   // this particle's stream
        const int g = lung::track_particle(p, aw, rng); // deposited-in generation
        tally[static_cast<std::size_t>(g)] += 1;        // plain integer increment
    }
}
