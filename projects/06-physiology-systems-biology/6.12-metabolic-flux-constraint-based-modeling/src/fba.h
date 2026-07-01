// ===========================================================================
// src/fba.h  --  Shared (host + device) Flux Balance Analysis LP solver
// ---------------------------------------------------------------------------
// Project 6.12 : Metabolic Flux / Constraint-Based Modeling
//
// WHAT THIS PROJECT COMPUTES
//   Flux Balance Analysis (FBA). A cell's metabolism is a network of reactions.
//   Let v be the vector of reaction fluxes (rates, mmol/gDW/h). The stoichiometry
//   matrix S (metabolites x reactions) says how each reaction produces/consumes
//   each internal metabolite. At metabolic STEADY STATE nothing accumulates, so
//
//       S v = 0                          (mass balance on every internal metabolite)
//
//   Each flux is bounded by thermodynamics / enzyme capacity / nutrient supply:
//
//       lb_j <= v_j <= ub_j
//
//   FBA picks, among all steady-state flux distributions, the one that MAXIMISES
//   a biological objective c^T v (classically the "biomass" pseudo-reaction, a
//   proxy for growth rate). That is a LINEAR PROGRAM (LP):
//
//       maximise  c^T v      subject to   S v = 0,   lb <= v <= ub.
//
//   Solving it yields the predicted growth rate and a flux distribution. Deleting
//   a reaction (a gene knockout) is just setting its bounds to lb=ub=0 and
//   re-solving -- if growth collapses, that gene is ESSENTIAL. Screening ALL
//   single-reaction knockouts = many INDEPENDENT LPs -> perfect GPU work.
//
// WHY THE MATH LIVES HERE (the HD-macro idiom, PATTERNS.md section 2)
//   The LP SOLVER below is written once as `__host__ __device__` inline code, so
//   the CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) run the
//   *identical* arithmetic in *identical* order. Same pivots -> same answer, so
//   verification is exact (integer/deterministic pivot rule) rather than fuzzy.
//   FBA_HD expands to __host__ __device__ under nvcc, and to nothing under the
//   plain host compiler (which has never heard of those keywords).
//
// THE ALGORITHM: a bounded-variable primal SIMPLEX
//   The simplex method walks vertices of the feasible polytope, each step moving
//   to an adjacent vertex that improves the objective, until none does (optimum).
//   We use the *bounded-variable* form because FBA variables have BOTH lower and
//   upper bounds -- textbook simplex assumes 0 <= x <= inf, which does not fit.
//   To turn the equalities S v = 0 into a solvable system with an obvious starting
//   basis, we append one artificial SLACK per metabolite whose bounds are fixed to
//   [0,0]: the system becomes  [S | I] [v; s] = 0  with the slacks basic at value
//   0. That identity basis is our feasible starting vertex.
//
//   Pivot rule: lowest-index entering variable among all that improve (Bland's
//   rule). Bland's rule provably prevents cycling AND is deterministic, which is
//   exactly what we need for reproducible CPU==GPU results. See THEORY.md for the
//   full derivation, complexity, and worked example.
//
//   NOTE ON SCOPE (CLAUDE.md section 13): this is a REDUCED-SCOPE teaching
//   solver -- a dense tableau simplex sized for small didactic networks (a few
//   metabolites, ~a dozen reactions), one LP per GPU thread. Genome-scale models
//   (Recon3D: ~8000 reactions) need sparse interior-point / revised-simplex
//   solvers (COBRApy + HiGHS/Gurobi). The MATH and the PARALLELISM are identical;
//   only the per-LP solver is swapped. THEORY.md section "Where this sits in the
//   real world" explains the production picture.
//
// READ THIS AFTER: (nothing) -- this is the core. Then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// FBA_HD: mark a function callable from BOTH host and device under nvcc, and as
// a plain inline function under the host compiler.
#ifdef __CUDACC__
#define FBA_HD __host__ __device__
#else
#define FBA_HD
#endif

// ---------------------------------------------------------------------------
// Compile-time capacity limits.
//   The dense simplex tableau and all working arrays live in per-thread LOCAL
//   memory (registers spill to local), so they must be FIXED-SIZE. We size them
//   generously for teaching networks. A thread that exceeds these bails out with
//   status STATUS_TOOBIG rather than corrupting memory.
//     FBA_MAX_MET   : max metabolites  (rows of S, = number of equality rows m)
//     FBA_MAX_RXN   : max reactions    (columns of S = structural variables n)
//     FBA_MAX_VARS  : structural + slack columns = n + m
//   With 12 x 24 the tableau is 12 x 36 doubles ~= 3.4 KB/thread -- fine for the
//   modest occupancy this embarrassingly-parallel screen needs.
// ---------------------------------------------------------------------------
#define FBA_MAX_MET   12
#define FBA_MAX_RXN   24
#define FBA_MAX_VARS  (FBA_MAX_RXN + FBA_MAX_MET)

// Solver exit codes (returned in FbaResult.status).
enum FbaStatus {
    FBA_OPTIMAL   = 0,   // found the optimum
    FBA_UNBOUNDED = 1,   // objective can grow without limit (ill-posed model)
    FBA_ITERLIMIT = 2,   // hit the safety iteration cap (should not happen here)
    FBA_TOOBIG    = 3    // model exceeds FBA_MAX_* capacities
};

// ---------------------------------------------------------------------------
// FbaModel: a complete FBA linear program, laid out for value-copy to the GPU.
//   S is stored ROW-MAJOR and DENSE: S[i*nrxn + j] is the stoichiometric
//   coefficient of metabolite i in reaction j (+ produced, - consumed). Dense is
//   wasteful for genome-scale models (which are ~99% zeros) but crystal-clear for
//   teaching; THEORY.md discusses the sparse (CSR) production layout.
//   The struct is a plain-old-data value (no pointers) so it can be passed to a
//   kernel BY VALUE -- exactly how 9.02 passes its EnsembleConfig.
// ---------------------------------------------------------------------------
struct FbaModel {
    int    nmet;                          // number of metabolites (equality rows)
    int    nrxn;                          // number of reactions (structural vars)
    double S[FBA_MAX_MET * FBA_MAX_RXN];  // dense row-major stoichiometry matrix
    double lb[FBA_MAX_RXN];               // per-reaction lower flux bound
    double ub[FBA_MAX_RXN];               // per-reaction upper flux bound
    double c[FBA_MAX_RXN];                // objective coefficients (biomass = 1)
};

// Result of one LP solve: the optimal objective and how it terminated.
//   We return only the scalar objective (predicted growth) because that is the
//   headline the knockout screen ranks on; the full flux vector is available in
//   the solver but omitted from the result to keep the device->host copy tiny.
struct FbaResult {
    double objective;   // optimal c^T v  (predicted biomass / growth rate)
    int    status;      // an FbaStatus value
    int    iters;       // simplex iterations taken (a teaching diagnostic)
};

// ---------------------------------------------------------------------------
// solve_fba: solve ONE FBA linear program with the bounded-variable simplex.
//
//   Inputs : model  -- the LP (S, lb, ub, c). Passed by const ref; on the device
//                      it is a per-thread copy in local memory.
//   Returns: an FbaResult (objective + status + iteration count).
//
//   Everything is self-contained in fixed-size local arrays: no heap, no shared
//   memory, no inter-thread communication -- so one thread solves one LP with no
//   coordination. That is what makes the knockout screen embarrassingly parallel.
//
//   Complexity: each simplex iteration is O(m * ntot) work (price all columns,
//   pivot one row); the number of iterations is small for these tiny models
//   (empirically a handful). See THEORY.md for the worst-case discussion.
// ---------------------------------------------------------------------------
FBA_HD inline FbaResult solve_fba(const FbaModel& model) {
    const int m    = model.nmet;              // equality rows
    const int n    = model.nrxn;              // structural variables (reactions)
    const int ntot = n + m;                   // + one slack per metabolite

    FbaResult res;
    res.objective = 0.0; res.status = FBA_OPTIMAL; res.iters = 0;

    // Guard against models that would overflow our fixed local arrays.
    if (m > FBA_MAX_MET || n > FBA_MAX_RXN || ntot > FBA_MAX_VARS || m <= 0 || n <= 0) {
        res.status = FBA_TOOBIG;
        return res;
    }

    // ---- Working storage (all per-thread local memory) --------------------
    // T   : the dense simplex tableau, T = B^{-1} A, kept updated by pivoting.
    //       Row-major [m x ntot]. Starts equal to A = [S | I] because the initial
    //       basis B is the identity (the slack columns).
    // L,U : lower/upper bound of every variable (structural then slack).
    // cost: objective coefficient of every variable (slacks contribute 0).
    // basis[i]  : which variable is basic in tableau row i.
    // inbasis[j]: is variable j currently basic?
    // atUpper[j]: for a NONbasic variable, is it resting at its upper bound
    //             (true) or its lower bound (false)?  Basic vars: ignored.
    // xB[i]     : current value of the basic variable in row i.
    double T[FBA_MAX_MET * FBA_MAX_VARS];
    double L[FBA_MAX_VARS], U[FBA_MAX_VARS], cost[FBA_MAX_VARS];
    int    basis[FBA_MAX_MET];
    bool   inbasis[FBA_MAX_VARS];
    bool   atUpper[FBA_MAX_VARS];
    double xB[FBA_MAX_MET];

    // ---- Build the augmented system [S | I] and bounds ---------------------
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            T[i * ntot + j] = model.S[i * n + j];     // structural columns = S
        for (int j = 0; j < m; ++j)
            T[i * ntot + (n + j)] = (i == j) ? 1.0 : 0.0;  // slack columns = I
    }
    for (int j = 0; j < n; ++j) { L[j] = model.lb[j]; U[j] = model.ub[j]; cost[j] = model.c[j]; }
    for (int j = 0; j < m; ++j) { L[n + j] = 0.0; U[n + j] = 0.0; cost[n + j] = 0.0; } // slacks fixed to 0

    for (int j = 0; j < ntot; ++j) { inbasis[j] = false; atUpper[j] = false; }
    for (int i = 0; i < m; ++i) { basis[i] = n + i; inbasis[n + i] = true; } // slacks basic

    // ---- Initialise basic values so that T x = 0 (RHS is the zero vector) ---
    // For nonbasic variable j its value is nbval(j) = (atUpper ? U : L). The RHS
    // contribution of the nonbasics must be cancelled by the basics:
    //   xB[i] = - sum_{nonbasic j} T[i][j] * nbval(j).
    // (All our nonbasic structural vars start at their lower bound = 0 here, but
    // we compute it generally so re-solves with nonzero lb behave correctly.)
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < ntot; ++j)
            if (!inbasis[j]) {
                double vj = atUpper[j] ? U[j] : L[j];
                s += T[i * ntot + j] * vj;
            }
        xB[i] = -s;
    }

    // ---- Simplex iterations ------------------------------------------------
    const int    MAX_ITERS = 10000;   // generous safety cap; real count is tiny
    const double EPS        = 1e-9;    // reduced-cost / pivot significance threshold
    const double INF        = 1e30;    // "no finite ratio" sentinel

    int it = 0;
    for (; it < MAX_ITERS; ++it) {
        // --- Pricing: reduced cost d_j = cost_j - cost_B^T (B^{-1} A)_j -------
        // cost_B^T times column j of the tableau. For a MAXIMISATION problem a
        // nonbasic var at its LOWER bound improves if d_j > 0 (raise it); at its
        // UPPER bound it improves if d_j < 0 (lower it). Bland's rule: take the
        // LOWEST-index improving variable (deterministic, anti-cycling).
        int    enter = -1;      // entering variable index (lowest improving)
        int    enterDir = 0;    // +1 = increase from lower, -1 = decrease from upper
        for (int j = 0; j < ntot; ++j) {
            if (inbasis[j]) continue;
            double dj = cost[j];
            for (int i = 0; i < m; ++i) dj -= cost[basis[i]] * T[i * ntot + j];
            if (!atUpper[j] && dj > EPS)      { enter = j; enterDir = +1; break; }
            else if (atUpper[j] && dj < -EPS) { enter = j; enterDir = -1; break; }
        }
        if (enter < 0) break;   // no improving column -> current vertex is OPTIMAL

        // --- Ratio test: how far can the entering variable move? -------------
        // Moving x_enter by t (in direction enterDir) changes each basic var by
        //   xB[i] -= T[i][enter] * t * enterDir.
        // The move stops when (a) the entering var reaches its OTHER bound
        // (t = U-L, a "bound flip" -- no basis change), or (b) some basic var
        // hits one of ITS bounds first (a pivot). We take the smallest such t.
        double tMax = U[enter] - L[enter];   // (a) entering var's own bound-flip limit
        int    leave = -1;                    // row whose basic var leaves (or -1 = flip)
        bool   leaveToUpper = false;          // does the leaving var exit at its upper bound?
        for (int i = 0; i < m; ++i) {
            double a  = T[i * ntot + enter] * enterDir;   // signed rate xB[i] falls
            int    bi = basis[i];
            if (a > 1e-12) {                              // xB[i] decreasing -> lower bound
                double t = (xB[i] - L[bi]) / a;
                if (t < tMax - 1e-12) { tMax = t; leave = i; leaveToUpper = false; }
            } else if (a < -1e-12) {                      // xB[i] increasing -> upper bound
                double t = (U[bi] - xB[i]) / (-a);
                if (t < tMax - 1e-12) { tMax = t; leave = i; leaveToUpper = true; }
            }
        }
        if (tMax >= INF) { res.status = FBA_UNBOUNDED; break; } // no bound stops it

        // Apply the step to every basic variable's value.
        double t = tMax * enterDir;
        for (int i = 0; i < m; ++i) xB[i] -= T[i * ntot + enter] * t;

        if (leave < 0) {
            // (a) BOUND FLIP: the entering var just moved to its other bound and
            // stays nonbasic; no pivot, no basis change. Flip its bound flag and
            // recompute basic values from scratch (cheap, and avoids drift).
            atUpper[enter] = !atUpper[enter];
            for (int i = 0; i < m; ++i) {
                double s = 0.0;
                for (int j = 0; j < ntot; ++j)
                    if (!inbasis[j]) { double vj = atUpper[j] ? U[j] : L[j]; s += T[i * ntot + j] * vj; }
                xB[i] = -s;
            }
        } else {
            // (b) PIVOT: entering var becomes basic in row `leave`; the old basic
            // var of that row leaves to the bound it hit. Gauss-Jordan eliminate
            // the entering column so the tableau stays T = B^{-1} A.
            int    leaving = basis[leave];
            double piv     = T[leave * ntot + enter];
            for (int j = 0; j < ntot; ++j) T[leave * ntot + j] /= piv;   // normalise pivot row
            for (int i = 0; i < m; ++i) {
                if (i == leave) continue;
                double f = T[i * ntot + enter];
                if (f != 0.0)
                    for (int j = 0; j < ntot; ++j)
                        T[i * ntot + j] -= f * T[leave * ntot + j];
            }
            inbasis[leaving] = false;
            inbasis[enter]   = true;
            atUpper[leaving] = leaveToUpper;   // leaving var rests at the bound it hit
            basis[leave]     = enter;
            // Recompute basic values consistently after the basis change.
            for (int i = 0; i < m; ++i) {
                double s = 0.0;
                for (int j = 0; j < ntot; ++j)
                    if (!inbasis[j]) { double vj = atUpper[j] ? U[j] : L[j]; s += T[i * ntot + j] * vj; }
                xB[i] = -s;
            }
        }
    }
    if (it >= MAX_ITERS && res.status == FBA_OPTIMAL) res.status = FBA_ITERLIMIT;
    res.iters = it;

    // ---- Read out the objective c^T v -------------------------------------
    // Reconstruct each structural variable's value: nonbasic = its bound, basic =
    // xB of its row. Only the first n (structural) variables carry objective cost.
    double obj = 0.0;
    for (int j = 0; j < n; ++j) {
        double vj;
        if (inbasis[j]) {
            vj = 0.0;
            for (int i = 0; i < m; ++i) if (basis[i] == j) { vj = xB[i]; break; }
        } else {
            vj = atUpper[j] ? U[j] : L[j];
        }
        obj += model.c[j] * vj;
    }
    res.objective = obj;
    return res;
}

// ---------------------------------------------------------------------------
// solve_knockout: solve the FBA LP for the model with reaction `ko` DELETED.
//   A gene/reaction knockout is modelled by clamping that reaction's flux to
//   zero (lb = ub = 0) and re-optimising. ko < 0 means "wild type" (no deletion).
//   This is the per-item job the knockout-screen kernel runs, one thread per ko.
//   Returns the optimal objective (predicted growth of the mutant).
// ---------------------------------------------------------------------------
FBA_HD inline FbaResult solve_knockout(const FbaModel& base, int ko) {
    FbaModel m = base;                 // local per-thread copy we may perturb
    if (ko >= 0 && ko < m.nrxn) {      // clamp the deleted reaction's flux to 0
        m.lb[ko] = 0.0;
        m.ub[ko] = 0.0;
    }
    return solve_fba(m);
}
