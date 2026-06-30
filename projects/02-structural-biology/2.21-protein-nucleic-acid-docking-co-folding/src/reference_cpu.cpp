// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable triple loop, no parallelism,
//   no cleverness -- so that when the GPU and CPU agree (here, EXACTLY, because
//   the scoring is pure integer arithmetic shared via docking_core.h), we
//   believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
//   Three things live in this file:
//     1. cube_rotations() -- builds the 24-orientation search set.
//     2. load_problem()   -- parses the text data format.
//     3. dock_cpu()       -- the reference exhaustive pose search.
//
// READ THIS AFTER: reference_cpu.h, docking_core.h.
// Compare dock_cpu() against dock_gpu() in kernels.cu (its GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// matmul3: multiply two 3x3 integer matrices (row-major). Used only to GENERATE
//   the cube-group rotation set by composing the three face rotations; it is a
//   build-time helper, not on any hot path.
// ---------------------------------------------------------------------------
static Rot3 matmul3(const Rot3& A, const Rot3& B) {
    Rot3 C{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            int s = 0;
            for (int k = 0; k < 3; ++k) s += A.m[r * 3 + k] * B.m[k * 3 + c];
            C.m[r * 3 + c] = s;
        }
    return C;
}

// rot_equal: are two integer rotation matrices identical? (For de-duplication
//   while enumerating the group.)
static bool rot_equal(const Rot3& A, const Rot3& B) {
    for (int i = 0; i < 9; ++i) if (A.m[i] != B.m[i]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// cube_rotations: the 24 proper (det = +1) rotations of a cube -- the canonical
//   coarse orientation set. We GENERATE the group rather than hand-type 24
//   matrices: start from the identity and three 90-degree face rotations
//   (about x, y, z), then repeatedly multiply known elements by the generators
//   until the set stops growing (a tiny closure / breadth-first fill). This is
//   the cleanest "no magic constants" way to get exactly the 24 elements.
//
//   Why the cube group? Its matrices have entries in {-1,0,+1}, so rotating an
//   integer atom coordinate yields an exact integer -- no trig, no rounding,
//   so CPU and GPU stay bit-identical (docking_core.h rationale). A production
//   docker samples SO(3) far more finely (thousands of orientations); we trade
//   resolution for exactness and clarity -- see THEORY "real world".
//
//   rots[0] is guaranteed to be the identity (so pose 0 is "ligand unmoved").
// ---------------------------------------------------------------------------
std::vector<Rot3> cube_rotations() {
    // Identity and the three 90-degree right-hand rotations about each axis.
    const Rot3 I   { 1,0,0,  0,1,0,  0,0,1 };
    const Rot3 Rx  { 1,0,0,  0,0,-1, 0,1,0 };   // rotate +90 about x
    const Rot3 Ry  { 0,0,1,  0,1,0,  -1,0,0 };  // rotate +90 about y
    const Rot3 Rz  { 0,-1,0, 1,0,0,  0,0,1 };   // rotate +90 about z
    const std::array<Rot3, 3> gens{ Rx, Ry, Rz };

    std::vector<Rot3> group{ I };               // start with the identity
    // Closure fill: keep multiplying every known element by every generator and
    // add any new matrices, until a full sweep adds nothing. The cube rotation
    // group has order 24, so this terminates quickly.
    bool grew = true;
    while (grew) {
        grew = false;
        const std::size_t cur = group.size();
        for (std::size_t i = 0; i < cur; ++i) {
            for (const Rot3& g : gens) {
                const Rot3 cand = matmul3(group[i], g);
                bool seen = false;
                for (const Rot3& e : group) if (rot_equal(e, cand)) { seen = true; break; }
                if (!seen) { group.push_back(cand); grew = true; }
            }
        }
    }
    return group;   // exactly 24 elements, group[0] == identity
}

// ---------------------------------------------------------------------------
// load_problem: parse the documented text format (data/README.md).
//   The format, all integers (fixed-point milli-Angstrom), is:
//
//     # comment lines (starting with '#') and blank lines are ignored
//     Np Nl                              <- atom counts
//     tx0 ty0 tz0 step nx ny nz          <- translational pose grid
//     clash_r2 contact_r2 clash_pen contact_w elec_w   <- scoring params
//     x y z charge        (Np protein atom lines)
//     ...
//     x y z charge        (Nl ligand  atom lines)
//
//   We throw std::runtime_error on any structural problem so a malformed file
//   fails loudly in the demo rather than silently scoring garbage.
// ---------------------------------------------------------------------------
DockingProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open data file: " + path);

    // next_line: return the next non-comment, non-blank line, or throw on EOF.
    auto next_line = [&](std::string& line) {
        while (std::getline(in, line)) {
            // Strip a trailing '\r' so files authored on Windows parse on Linux.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // Skip pure-comment / blank lines.
            std::size_t first = line.find_first_not_of(" \t");
            if (first == std::string::npos) continue;     // blank
            if (line[first] == '#') continue;             // comment
            return true;
        }
        return false;
    };

    DockingProblem prob;
    std::string line;

    // (1) atom counts -----------------------------------------------------
    if (!next_line(line)) throw std::runtime_error("missing 'Np Nl' header line");
    int Np = 0, Nl = 0;
    { std::istringstream ss(line); if (!(ss >> Np >> Nl) || Np <= 0 || Nl <= 0)
          throw std::runtime_error("bad 'Np Nl' line: " + line); }

    // (2) pose grid -------------------------------------------------------
    if (!next_line(line)) throw std::runtime_error("missing pose-grid line");
    { std::istringstream ss(line);
      if (!(ss >> prob.grid.tx0 >> prob.grid.ty0 >> prob.grid.tz0
               >> prob.grid.step >> prob.grid.nx >> prob.grid.ny >> prob.grid.nz))
          throw std::runtime_error("bad pose-grid line: " + line);
      if (prob.grid.nx <= 0 || prob.grid.ny <= 0 || prob.grid.nz <= 0 || prob.grid.step <= 0)
          throw std::runtime_error("pose-grid counts/step must be positive"); }

    // (3) scoring params --------------------------------------------------
    if (!next_line(line)) throw std::runtime_error("missing scoring-params line");
    { std::istringstream ss(line);
      if (!(ss >> prob.params.clash_r2 >> prob.params.contact_r2
               >> prob.params.clash_pen >> prob.params.contact_w >> prob.params.elec_w))
          throw std::runtime_error("bad scoring-params line: " + line);
      if (prob.params.contact_r2 < prob.params.clash_r2)
          throw std::runtime_error("contact_r2 must be >= clash_r2"); }

    // (4) protein atoms ---------------------------------------------------
    prob.protein.reserve(Np);
    for (int i = 0; i < Np; ++i) {
        if (!next_line(line)) throw std::runtime_error("missing protein atom line");
        Atom a{}; std::istringstream ss(line);
        if (!(ss >> a.x >> a.y >> a.z >> a.charge))
            throw std::runtime_error("bad protein atom line: " + line);
        prob.protein.push_back(a);
    }

    // (5) ligand atoms ----------------------------------------------------
    prob.ligand.reserve(Nl);
    for (int i = 0; i < Nl; ++i) {
        if (!next_line(line)) throw std::runtime_error("missing ligand atom line");
        Atom a{}; std::istringstream ss(line);
        if (!(ss >> a.x >> a.y >> a.z >> a.charge))
            throw std::runtime_error("bad ligand atom line: " + line);
        prob.ligand.push_back(a);
    }

    // Orientation set is not in the file -- it is the fixed cube group.
    prob.rots = cube_rotations();
    return prob;
}

// ---------------------------------------------------------------------------
// dock_cpu: the reference exhaustive search.
//   For every flat pose index p in [0, n_poses): decode it to a (rotation,
//   translation), score it with the SHARED score_pose() core, and store the
//   int64 result. One readable serial loop -- O(n_poses * Np * Nl).
//
//   Because score_pose() is the identical HD function the GPU calls, and all
//   arithmetic is integer, scores[p] here equals scores[p] on the GPU EXACTLY.
//   main.cu asserts that equality with tolerance 0 (PATTERNS.md sec 4: exact).
// ---------------------------------------------------------------------------
void dock_cpu(const DockingProblem& prob, std::vector<int64_t>& scores) {
    const long long N = prob.n_poses();
    scores.assign((std::size_t)N, 0);

    const Atom*  pro = prob.protein.data();
    const Atom*  lig = prob.ligand.data();
    const int    Np  = prob.Np();
    const int    Nl  = prob.Nl();
    const int    nr  = prob.n_rot();

    for (long long p = 0; p < N; ++p) {
        int32_t tx, ty, tz;
        const int r = decode_pose(p, prob.grid, nr, tx, ty, tz);   // same as GPU
        scores[(std::size_t)p] =
            score_pose(pro, Np, lig, Nl, prob.rots[r], tx, ty, tz, prob.params);
    }
}
