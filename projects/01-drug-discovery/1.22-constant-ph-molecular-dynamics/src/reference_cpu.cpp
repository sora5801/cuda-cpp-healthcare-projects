// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial constant-pH titration reference
// ---------------------------------------------------------------------------
// Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct: a flat triple loop (pH, replica, chain) with no
//   parallelism and no cleverness, calling the SAME shared physics (cph_core.h
//   run_chain) that the GPU thread calls. When the GPU and CPU integer tallies
//   agree, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: cph_core.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::isnan, std::pow
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// (chain_id, the (pH,replica)->seed packing shared with the GPU, lives in
//  cph_core.h so both sides pack identically.)

// ---------------------------------------------------------------------------
// load_cph_problem: parse the tiny text format documented in data/README.md.
//
//   FORMAT (whitespace/newline separated; '#' starts a comment to end of line):
//     line 1:  n_res  coulomb_k  kT  sweeps  burn_in
//     line 2:  pH_min  pH_max  n_pH  replicas  seed
//     then n_res lines, one per residue:
//              pKa_intrinsic  q_prot  q_deprot  x  y  z
//
//   Throwing std::runtime_error on any problem makes the demo fail loudly with a
//   precise message rather than silently simulating nonsense.
// ---------------------------------------------------------------------------
CphProblem load_cph_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    // Read the whole file, stripping '#' comments, into a single token stream.
    // This keeps the parser order-driven and forgiving of blank/comment lines.
    std::ostringstream cleaned;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);   // drop the comment
        cleaned << line << '\n';
    }
    std::istringstream ts(cleaned.str());

    CphProblem p;
    // --- line 1: system-wide controls ---
    if (!(ts >> p.sys.n_res >> p.sys.coulomb_k >> p.sys.kT
             >> p.sys.sweeps >> p.sys.burn_in))
        throw std::runtime_error("bad header (expected 'n_res coulomb_k kT "
                                 "sweeps burn_in') in " + path);
    if (p.sys.n_res <= 0 || p.sys.n_res > CPH_MAX_RESIDUES)
        throw std::runtime_error("n_res out of range [1," +
                                 std::to_string(CPH_MAX_RESIDUES) + "] in " + path);
    if (p.sys.kT <= 0.0)
        throw std::runtime_error("kT must be > 0 in " + path);
    if (p.sys.sweeps <= 0 || p.sys.burn_in < 0 || p.sys.burn_in >= p.sys.sweeps)
        throw std::runtime_error("need 0 <= burn_in < sweeps in " + path);

    // --- line 2: pH grid + ensemble controls ---
    if (!(ts >> p.pH_min >> p.pH_max >> p.n_pH >> p.replicas >> p.seed))
        throw std::runtime_error("bad pH line (expected 'pH_min pH_max n_pH "
                                 "replicas seed') in " + path);
    if (p.n_pH < 2 || p.pH_max <= p.pH_min)
        throw std::runtime_error("need n_pH>=2 and pH_max>pH_min in " + path);
    if (p.replicas <= 0)
        throw std::runtime_error("replicas must be >= 1 in " + path);

    // --- n_res residue lines ---
    for (int i = 0; i < p.sys.n_res; ++i) {
        Residue& R = p.sys.res[i];
        if (!(ts >> R.pKa_intrinsic >> R.q_prot >> R.q_deprot
                 >> R.x >> R.y >> R.z))
            throw std::runtime_error("bad residue line " + std::to_string(i) +
                " (expected 'pKa q_prot q_deprot x y z') in " + path);
    }
    return p;
}

// ---------------------------------------------------------------------------
// titrate_cpu: the serial reference. For every pH grid point and every replica,
// seed a chain, run the shared Monte Carlo (run_chain), and ADD its per-residue
// protonation counts into the flat output array. Plain integer '+=' here; the
// GPU uses atomicAdd on the same integers -- both are exact and must agree.
//
//   Complexity: O(n_pH * replicas * sweeps * n_res^2). The n_res^2 is the
//   coupling sum inside delta_G_flip; everything else is a count of independent
//   chains -- which is precisely the parallelism the GPU exploits.
// ---------------------------------------------------------------------------
void titrate_cpu(const CphProblem& prob, CphResult& out) {
    const int n_res = prob.sys.n_res;
    // Allocate + zero the [n_pH * n_res] tally. uint64_t so it cannot overflow
    // for any realistic sweeps*replicas, and so it matches the GPU's type.
    out.prot_count.assign(static_cast<std::size_t>(prob.n_pH) * n_res, 0ULL);
    out.tallied_per_pH = 0;

    // Per-chain scratch: counts protonated sweeps for each residue in ONE chain
    // before we fold it into the shared per-pH tally. Keeping the chain's own
    // counts as int mirrors the GPU thread's register-resident accumulation.
    int chain_counts[CPH_MAX_RESIDUES];

    for (int k = 0; k < prob.n_pH; ++k) {
        // The k-th pH on the linear grid (n_pH>=2 guaranteed by the loader).
        const double pH = prob.pH_min +
            (prob.pH_max - prob.pH_min) * k / (prob.n_pH - 1);

        for (int r = 0; r < prob.replicas; ++r) {
            // Seed this chain identically to the GPU (same chain_id packing).
            Rng rng = rng_seed(prob.seed, chain_id(k, r));
            for (int i = 0; i < n_res; ++i) chain_counts[i] = 0;

            const int tallied = run_chain(prob.sys, pH, rng, chain_counts);
            // The denominator is the same for every chain; record it once.
            if (k == 0 && r == 0)
                out.tallied_per_pH =
                    static_cast<uint64_t>(prob.replicas) * tallied;

            // Fold this chain's counts into the shared per-pH tally.
            for (int i = 0; i < n_res; ++i)
                out.prot_count[static_cast<std::size_t>(k) * n_res + i] +=
                    static_cast<uint64_t>(chain_counts[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// estimate_pKa: read a residue's pKa off its titration curve. The curve is the
// fraction protonated f(pH) = prot_count / tallied_per_pH at each grid pH. By
// definition the pKa is the pH where f = 0.5. We scan adjacent grid points for
// the bracket where f crosses 0.5 and linearly interpolate the crossing pH.
//
//   Acids (q goes 0 -> -1 as pH rises) and bases (q goes +1 -> 0) both have f
//   DECREASING with pH, so we look for the first k where f[k] >= 0.5 > f[k+1].
//   Returns NaN if no crossing is on the grid (pKa outside [pH_min,pH_max]).
//
//   This is the scientific readout of the whole simulation: with coupling off it
//   must return (within MC noise) the residue's intrinsic pKa -- the analytic
//   sanity check main.cu prints.
// ---------------------------------------------------------------------------
double estimate_pKa(const CphProblem& prob, const CphResult& res, int residue) {
    const int n_res = prob.sys.n_res;
    const double denom = static_cast<double>(res.tallied_per_pH);
    // fraction protonated at pH index k for this residue
    auto frac = [&](int k) -> double {
        return static_cast<double>(
            res.prot_count[static_cast<std::size_t>(k) * n_res + residue]) / denom;
    };
    for (int k = 0; k + 1 < prob.n_pH; ++k) {
        const double f0 = frac(k), f1 = frac(k + 1);
        // Bracket where the (decreasing) curve passes through 0.5.
        if (f0 >= 0.5 && f1 < 0.5) {
            const double pH0 = prob.pH_min +
                (prob.pH_max - prob.pH_min) * k / (prob.n_pH - 1);
            const double pH1 = prob.pH_min +
                (prob.pH_max - prob.pH_min) * (k + 1) / (prob.n_pH - 1);
            // Linear interpolation: solve f0 + t*(f1-f0) = 0.5 for t in [0,1].
            const double t = (0.5 - f0) / (f1 - f0);
            return pH0 + t * (pH1 - pH0);
        }
    }
    return std::numeric_limits<double>::quiet_NaN();   // never crosses 0.5
}
