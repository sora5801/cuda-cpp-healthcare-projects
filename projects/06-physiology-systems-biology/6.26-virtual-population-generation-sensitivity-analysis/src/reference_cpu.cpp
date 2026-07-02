// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Saltelli evaluation + Sobol post
// ---------------------------------------------------------------------------
// Project 6.26 : Virtual Population Generation & Sensitivity Analysis
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against. Everything here is written to
//   be OBVIOUSLY correct: a single readable loop over the Saltelli evaluations,
//   and a textbook implementation of the Saltelli Sobol estimators. When the GPU
//   raw-output array and Sobol indices match this, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-sample model
//   and sampling live in vpop.h, shared verbatim with kernels.cu.
//
// READ THIS AFTER: reference_cpu.h, vpop.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_vpop: parse the whitespace population-configuration file. The format is
// documented in data/README.md; we read it field-by-field and validate ranges
// so a demo fails LOUDLY on bad input rather than silently producing nonsense.
// ---------------------------------------------------------------------------
VpopParams load_vpop(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open population config file: " + path);

    VpopParams P{};
    // Order matches data/README.md exactly. std::ifstream::operator>> skips
    // whitespace/newlines, so the physical line breaks in the file are cosmetic.
    if (!(in >> P.dose
             >> P.lo[0] >> P.hi[0]      // ka range (1/h)
             >> P.lo[1] >> P.hi[1]      // CL range (L/h)
             >> P.lo[2] >> P.hi[2]      // V  range (L)
             >> P.lo[3] >> P.hi[3]      // F  range (unitless)
             >> P.t_end >> P.steps
             >> P.N >> P.seed)) {
        throw std::runtime_error(
            "bad config (expected: dose, then 4 lo/hi pairs for ka,CL,V,F, "
            "then 't_end steps', then 'N seed') in " + path);
    }

    // Physical-sanity guards: positive dose, well-ordered ranges, positive
    // clearance/volume lower bounds (kel = CL/V must be finite and > 0), and a
    // sensible integration grid and sample count.
    if (P.dose <= 0.0 || P.t_end <= 0.0 || P.steps <= 0 || P.N <= 0)
        throw std::runtime_error("non-physical scalar in config: " + path);
    for (int j = 0; j < VPOP_K; ++j)
        if (!(P.hi[j] > P.lo[j]))
            throw std::runtime_error("parameter range must have hi > lo in: " + path);
    if (P.lo[1] <= 0.0 || P.lo[2] <= 0.0)
        throw std::runtime_error("CL and V lower bounds must be > 0 in: " + path);

    return P;
}

// ---------------------------------------------------------------------------
// evaluate_cpu: run the forward model for every Saltelli global index g. This is
// the serial twin of the GPU kernel -- literally the same vpop_eval() call in a
// plain loop, so the two output arrays are bit-identical up to floating-point
// round-off (and in practice identical: the same ops in the same order).
//   Complexity: O(N*(k+2) * steps). Each evaluation is independent -> one GPU
//   thread per g in kernels.cu.
// ---------------------------------------------------------------------------
void evaluate_cpu(const VpopParams& P, std::vector<double>& out) {
    const long total = vpop_num_evals(P.N);            // N*(k+2) model runs
    out.assign(static_cast<std::size_t>(total), 0.0);
    for (long g = 0; g < total; ++g)
        out[static_cast<std::size_t>(g)] = vpop_eval(P, g);
}

// ---------------------------------------------------------------------------
// compute_sobol: the Saltelli variance-based estimators.
//
//   Let A = f over block 0, B = f over block 1, and AB_j = f over block (2+j),
//   each an N-vector of model outputs. With f0 = mean over A:
//
//     Var(Y)  = (1/N) sum_i (A_i - f0)^2                         (total variance)
//     Vj      = (1/N) sum_i B_i * (AB_j,i - A_i)                 (first-order)
//     VTj     = (1/2N) sum_i (A_i - AB_j,i)^2                    (total-order)
//     S_j     = Vj  / Var(Y)          ST_j = VTj / Var(Y)
//
//   These are the estimators from Saltelli et al. 2010 (the ones SALib uses).
//   The (A - AB_j) form for total effects is the "Jansen" estimator, which is
//   numerically robust because it never subtracts two large similar means.
//
//   Everything is a serial double-precision reduction: cheap (O(N*k)) and
//   deterministic. main.cu runs this on BOTH the CPU array and the GPU array
//   and requires the resulting indices to agree.
// ---------------------------------------------------------------------------
SobolResult compute_sobol(const VpopParams& P, const std::vector<double>& f) {
    const int N = P.N;
    // Pointers to the start of each block within the flat array. Block b spans
    // rows [b*N, (b+1)*N). This mirrors the layout documented in vpop.h.
    const double* A = f.data() + 0L * N;   // block 0
    const double* B = f.data() + 1L * N;   // block 1
    // AB_j starts at block (2+j): base pointer f.data() + (2+j)*N.

    // ---- population mean and total variance over matrix A -----------------
    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += A[i];
    mean /= (double)N;

    double var = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = A[i] - mean;
        var += d * d;
    }
    var /= (double)N;

    SobolResult R{};
    R.mean = mean;
    R.var  = var;

    // ---- per-parameter first-order (Vj) and total-order (VTj) effects -----
    for (int j = 0; j < VPOP_K; ++j) {
        const double* ABj = f.data() + (long)(2 + j) * N;   // block 2+j
        double Vj = 0.0;      // first-order numerator
        double VTj = 0.0;     // total-order numerator (Jansen form)
        for (int i = 0; i < N; ++i) {
            Vj  += B[i] * (ABj[i] - A[i]);
            const double d = A[i] - ABj[i];
            VTj += d * d;
        }
        Vj  /= (double)N;
        VTj /= (2.0 * (double)N);
        // Normalize by total variance. Guard the degenerate var==0 case (all
        // outputs identical -> no sensitivity is defined; report 0).
        R.S[j]  = (var > 0.0) ? (Vj  / var) : 0.0;
        R.ST[j] = (var > 0.0) ? (VTj / var) : 0.0;
    }
    return R;
}
