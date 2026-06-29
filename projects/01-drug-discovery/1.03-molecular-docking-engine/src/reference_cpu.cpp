// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial docking baseline + data loader
// ---------------------------------------------------------------------------
// Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   (1) load_problem(): parse the tiny text dataset (data/README.md format) into
//       a DockingProblem (energy grid + ligand + pose search space).
//   (2) unrank_pose(): turn a flat pose index into a concrete Pose. SHARED by the
//       CPU loop here and (a synced copy in) the GPU kernel, so both enumerate
//       the same poses in the same order.
//   (3) dock_cpu(): the obviously-correct serial search -- score EVERY pose with
//       the shared docking_core.h::score_pose() and keep the best. This is the
//       oracle the GPU result is verified against and the timing baseline.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, docking_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>        // M_PI is not standard on MSVC, so we define TWO_PI below
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// 2*pi as a literal so the rotation sweep is identical on host and device
// (MSVC does not define M_PI without _USE_MATH_DEFINES; a literal sidesteps that
//  and guarantees the CPU/GPU pose enumeration uses the byte-identical constant).
static constexpr double TWO_PI = 6.283185307179586476925286766559;

// ---------------------------------------------------------------------------
// unrank_pose: flat index p -> Pose (see reference_cpu.h for the contract).
//   We peel sub-indices off p in mixed-radix order, lowest axis first:
//       it_x (radix n_trans), it_y, it_z, ir_x (radix n_rot), ir_y, ir_z.
//   Each translation sub-index maps to an evenly spaced offset in
//   [-trans_range, +trans_range]; each rotation sub-index to an angle in
//   [0, 2pi). With n_trans==1 we place the ligand exactly at the pocket centre
//   (offset 0); with n_rot==1 the angle is 0. The arithmetic is pure
//   integer/double so the identical body runs on the GPU (kernels.cu).
// ---------------------------------------------------------------------------
Pose unrank_pose(const SearchSpace& s, long long p) {
    // --- decode the six sub-indices (mixed radix) ---
    const long long nt = s.n_trans;     // radix for the three translation axes
    const long long nr = s.n_rot;       // radix for the three rotation axes
    const int it_x = static_cast<int>(p % nt); p /= nt;
    const int it_y = static_cast<int>(p % nt); p /= nt;
    const int it_z = static_cast<int>(p % nt); p /= nt;
    const int ir_x = static_cast<int>(p % nr); p /= nr;
    const int ir_y = static_cast<int>(p % nr); p /= nr;
    const int ir_z = static_cast<int>(p % nr); p /= nr;

    // --- map sub-indices to physical translation offsets (Angstrom) ---
    // For n>1, sample i in {0..n-1} maps to lo + i*(hi-lo)/(n-1) spanning the
    // full closed range [-R, +R]; for n==1 the single sample is the midpoint 0.
    auto trans_offset = [&](int i) -> double {
        if (s.n_trans <= 1) return 0.0;
        const double step = (2.0 * s.trans_range) / (s.n_trans - 1);
        return -s.trans_range + i * step;
    };
    // --- map sub-indices to rotation angles (radians), half-open [0, 2pi) ---
    auto rot_angle = [&](int i) -> double {
        if (s.n_rot <= 1) return 0.0;
        return (TWO_PI * i) / s.n_rot;     // 0, 2pi/n, ... (excludes 2pi == 0)
    };

    Pose pose;
    pose.tx = s.tcx + trans_offset(it_x);
    pose.ty = s.tcy + trans_offset(it_y);
    pose.tz = s.tcz + trans_offset(it_z);
    pose.a  = rot_angle(ir_x);
    pose.b  = rot_angle(ir_y);
    pose.c  = rot_angle(ir_z);
    return pose;
}

// ---------------------------------------------------------------------------
// A small helper that reads the next whitespace-separated token as a double,
// throwing a clear error (with the field name) if the stream is exhausted or the
// token is not a number. Keeps load_problem() readable.
// ---------------------------------------------------------------------------
static double read_double(std::istream& in, const char* field, const std::string& path) {
    double v;
    if (!(in >> v))
        throw std::runtime_error("bad/missing '" + std::string(field) + "' in " + path);
    return v;
}
static int read_int(std::istream& in, const char* field, const std::string& path) {
    int v;
    if (!(in >> v))
        throw std::runtime_error("bad/missing '" + std::string(field) + "' in " + path);
    return v;
}

// ---------------------------------------------------------------------------
// load_problem: parse the dataset. Lines starting with '#' are comments. The
//   format (full spec + a worked example in data/README.md):
//
//     GRID  nx ny nz  ox oy oz  spacing
//     <nx*ny*nz energies, whitespace-separated, x fastest then y then z>
//     LIGAND  n_atoms
//     <n_atoms lines:  x y z weight>      (ligand-local coordinates, Angstrom)
//     SEARCH  n_trans n_rot trans_range  tcx tcy tcz
//
//   We strip comments by scanning into a stringstream so '#' lines never reach
//   the numeric parser. Throws on any structural problem so the demo cannot run
//   on a half-read file.
// ---------------------------------------------------------------------------
DockingProblem load_problem(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open docking input file: " + path);

    // Pre-pass: copy non-comment, non-blank content into a token stream. This
    // lets the dataset carry explanatory '#' comments (used heavily in the
    // sample) without complicating the field-by-field parse below.
    std::ostringstream cleaned;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);    // drop trailing comment
        cleaned << line << ' ';
    }
    std::istringstream in(cleaned.str());

    DockingProblem prob;
    std::string tag;

    // ---- GRID block ----
    if (!(in >> tag) || tag != "GRID")
        throw std::runtime_error("expected 'GRID' tag at start of " + path);
    prob.dims.nx = read_int(in, "nx", path);
    prob.dims.ny = read_int(in, "ny", path);
    prob.dims.nz = read_int(in, "nz", path);
    prob.dims.ox = read_double(in, "ox", path);
    prob.dims.oy = read_double(in, "oy", path);
    prob.dims.oz = read_double(in, "oz", path);
    prob.dims.spacing = read_double(in, "spacing", path);
    if (prob.dims.nx <= 0 || prob.dims.ny <= 0 || prob.dims.nz <= 0)
        throw std::runtime_error("non-positive grid dimension in " + path);
    if (prob.dims.spacing <= 0.0)
        throw std::runtime_error("non-positive grid spacing in " + path);

    const long long ncells = prob.dims.count();
    prob.grid.resize(static_cast<std::size_t>(ncells));
    for (long long i = 0; i < ncells; ++i)
        prob.grid[static_cast<std::size_t>(i)] = read_double(in, "grid energy", path);

    // ---- LIGAND block ----
    if (!(in >> tag) || tag != "LIGAND")
        throw std::runtime_error("expected 'LIGAND' tag in " + path);
    const int na = read_int(in, "n_atoms", path);
    if (na <= 0) throw std::runtime_error("non-positive n_atoms in " + path);
    prob.ligand.n_atoms = na;
    prob.ligand.x.resize(na); prob.ligand.y.resize(na);
    prob.ligand.z.resize(na); prob.ligand.weight.resize(na);
    for (int k = 0; k < na; ++k) {
        prob.ligand.x[k]      = read_double(in, "atom x", path);
        prob.ligand.y[k]      = read_double(in, "atom y", path);
        prob.ligand.z[k]      = read_double(in, "atom z", path);
        prob.ligand.weight[k] = read_double(in, "atom weight", path);
    }

    // ---- SEARCH block ----
    if (!(in >> tag) || tag != "SEARCH")
        throw std::runtime_error("expected 'SEARCH' tag in " + path);
    prob.space.n_trans     = read_int(in, "n_trans", path);
    prob.space.n_rot       = read_int(in, "n_rot", path);
    prob.space.trans_range = read_double(in, "trans_range", path);
    prob.space.tcx         = read_double(in, "tcx", path);
    prob.space.tcy         = read_double(in, "tcy", path);
    prob.space.tcz         = read_double(in, "tcz", path);
    if (prob.space.n_trans <= 0 || prob.space.n_rot <= 0)
        throw std::runtime_error("non-positive search resolution in " + path);

    return prob;
}

// ---------------------------------------------------------------------------
// dock_cpu: exhaustively score every pose and keep the best (lowest energy).
//   This is intentionally a single flat loop with no parallelism: it is the
//   reference whose result the GPU must reproduce. Determinism / tie rule: we
//   only replace the incumbent on a STRICTLY lower energy, so among exactly-equal
//   energies the FIRST (lowest index) pose wins -- the same rule the GPU's
//   index-carrying reduction uses, guaranteeing identical winners.
//   Complexity: O(n_poses * n_atoms). For the committed sample this is small;
//   the point is that all n_poses scorings are independent -> the GPU does them
//   at once (kernels.cu).
// ---------------------------------------------------------------------------
void dock_cpu(const DockingProblem& prob, double* out_energy, long long* out_index) {
    const long long n_poses = prob.space.n_poses();
    double    best_e   = std::numeric_limits<double>::infinity();
    long long best_idx = 0;

    for (long long p = 0; p < n_poses; ++p) {
        const Pose pose = unrank_pose(prob.space, p);          // same as GPU
        const double e = score_pose(prob.grid.data(), prob.dims,
                                    prob.ligand.x.data(), prob.ligand.y.data(),
                                    prob.ligand.z.data(), prob.ligand.weight.data(),
                                    prob.ligand.n_atoms, pose);
        if (e < best_e) { best_e = e; best_idx = p; }          // strict -> tie=lowest idx
    }
    *out_energy = best_e;
    *out_index  = best_idx;
}
