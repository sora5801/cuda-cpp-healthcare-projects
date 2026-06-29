// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial covalent-docking reference
// ---------------------------------------------------------------------------
// Project 1.28 : Covalent Docking
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- a single readable loop over all conformations, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. It also loads the docking problem from disk for both paths.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-conformation
//   PHYSICS it calls (score_conformation) lives in docking.h, the same header
//   the GPU kernel includes -> identical math, exact agreement.
//
// READ THIS AFTER: reference_cpu.h, docking.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// read_token line helper: pull the next whitespace-separated number from a
// stream, throwing a descriptive error if the file ends early. Keeping this in
// one place makes the loader below short and uniform.
// ---------------------------------------------------------------------------
namespace {
double next_double(std::istream& in, const std::string& path, const char* what) {
    double v;
    if (!(in >> v))
        throw std::runtime_error("covalent-docking: ran out of data reading '" +
                                 std::string(what) + "' in " + path);
    return v;
}
}  // namespace

// ---------------------------------------------------------------------------
// load_problem: parse the synthetic sample (see data/README.md for the exact
// field order). The file is a flat list of numbers; we read them in a fixed
// order into the DockProblem. Comments in the sample file (lines starting '#')
// are stripped first so the dataset can document itself inline.
// ---------------------------------------------------------------------------
DockProblem load_problem(const std::string& path) {
    std::ifstream raw(path);
    if (!raw) throw std::runtime_error("covalent-docking: cannot open " + path);

    // Strip '#' comment lines into a clean number-only stream. This lets the
    // committed sample carry human annotations without confusing the parser.
    std::ostringstream cleaned;
    std::string line;
    while (std::getline(raw, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        cleaned << line << '\n';
    }
    std::istringstream in(cleaned.str());

    DockProblem p{};

    // Anchor (warhead) and cysteine S-gamma positions (3 doubles each).
    p.anchor = Vec3{ next_double(in, path, "anchor.x"),
                     next_double(in, path, "anchor.y"),
                     next_double(in, path, "anchor.z") };
    p.sg     = Vec3{ next_double(in, path, "sg.x"),
                     next_double(in, path, "sg.y"),
                     next_double(in, path, "sg.z") };

    // Covalent constraint parameters.
    p.bond_len_ideal = next_double(in, path, "bond_len_ideal");
    p.angle_ideal    = next_double(in, path, "angle_ideal");
    p.k_bond         = next_double(in, path, "k_bond");
    p.k_angle        = next_double(in, path, "k_angle");

    // Ligand-chain template.
    p.seg_len    = next_double(in, path, "seg_len");
    p.bond_angle = next_double(in, path, "bond_angle");
    p.first_dir  = Vec3{ next_double(in, path, "first_dir.x"),
                         next_double(in, path, "first_dir.y"),
                         next_double(in, path, "first_dir.z") };

    // Ligand atom nonbonded parameters (shared by all ligand atoms here).
    p.lig_sigma   = next_double(in, path, "lig_sigma");
    p.lig_epsilon = next_double(in, path, "lig_epsilon");
    p.lig_charge  = next_double(in, path, "lig_charge");

    // The fixed pocket atoms: each is pos(3) + sigma + epsilon + charge.
    for (int q = 0; q < N_POCKET; ++q) {
        p.pocket[q].pos = Vec3{ next_double(in, path, "pocket.pos.x"),
                                next_double(in, path, "pocket.pos.y"),
                                next_double(in, path, "pocket.pos.z") };
        p.pocket[q].sigma   = next_double(in, path, "pocket.sigma");
        p.pocket[q].epsilon = next_double(in, path, "pocket.epsilon");
        p.pocket[q].charge  = next_double(in, path, "pocket.charge");
    }

    return p;
}

// ---------------------------------------------------------------------------
// score_all_cpu: the serial baseline. Loop over EVERY conformation id and store
// its energy. Each iteration is independent (it only reads the shared, const
// problem `p`) -> this is exactly the loop the GPU parallelizes one-thread-per-
// id in kernels.cu.
//   Complexity: O(M * N_LIG_ATOMS * N_POCKET) where M = n_conformations() is
//   exponential in the number of torsions. O(M) extra space for the energies.
// ---------------------------------------------------------------------------
void score_all_cpu(const DockProblem& p, std::vector<double>& energies) {
    const long long M = n_conformations();
    energies.assign(static_cast<std::size_t>(M), 0.0);
    for (long long id = 0; id < M; ++id) {
        // score_conformation() is the shared __host__ __device__ core; calling
        // it here (host) and in the kernel (device) is what guarantees parity.
        energies[static_cast<std::size_t>(id)] = score_conformation(p, id);
    }
}

// ---------------------------------------------------------------------------
// argmin_energy: deterministic reduction to the best pose. We scan in ascending
// id and only REPLACE the incumbent on a STRICTLY lower energy, so among equal-
// energy poses the smallest id wins -- a stable, order-independent tie-break.
// Both the CPU and GPU paths feed their (identical) energy arrays through this
// same function, so the reported docked pose is the same either way.
// ---------------------------------------------------------------------------
DockResult argmin_energy(const std::vector<double>& energies) {
    DockResult best{0, energies.empty() ? 0.0 : energies[0]};
    for (std::size_t id = 1; id < energies.size(); ++id) {
        if (energies[id] < best.best_energy) {
            best.best_energy = energies[id];
            best.best_id = static_cast<long long>(id);
        }
    }
    return best;
}
