// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial bone-remodeling reference
// ---------------------------------------------------------------------------
// Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against. It is deliberately written as
//   plain nested loops -- no parallelism, no cleverness -- so that when the GPU
//   result agrees with it, we trust the GPU. Compiled by the host C++ compiler
//   only (no CUDA here). The per-voxel physics is the shared bone_remodel.h, so
//   this reference and kernels.cu compute identical values.
//
// READ THIS AFTER: reference_cpu.h, bone_remodel.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_bone : read the 10 whitespace-separated parameters in fixed order.
// ---------------------------------------------------------------------------
BoneParams load_bone(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open bone parameter file: " + path);

    BoneParams p;
    // Read every field in the documented order (data/README.md). If any read
    // fails, operator>> leaves the stream in a failed state and the check below
    // fires -- so a truncated file is reported, not silently half-parsed.
    if (!(in >> p.nx >> p.ny >> p.remodel_steps >> p.relax_iters
             >> p.load >> p.load_x0 >> p.load_x1
             >> p.setpoint >> p.lazy >> p.rate
             >> p.rho_min >> p.rho_init)) {
        throw std::runtime_error(
            "bad parameters (expected 'nx ny remodel_steps relax_iters load "
            "load_x0 load_x1 setpoint lazy rate rho_min rho_init') in " + path);
    }
    // Basic sanity so downstream code and the S/rho division stay well-defined.
    if (p.nx <= 0 || p.ny <= 0 || p.remodel_steps < 0 || p.relax_iters < 0)
        throw std::runtime_error("invalid grid/iteration counts in " + path);
    if (p.load_x0 < 0 || p.load_x1 >= p.nx || p.load_x0 > p.load_x1)
        throw std::runtime_error("need 0 <= load_x0 <= load_x1 < nx in " + path);
    if (p.rho_min <= 0.0 || p.rho_init < p.rho_min || p.rho_init > 1.0)
        throw std::runtime_error("need 0 < rho_min <= rho_init <= 1 in " + path);
    if (p.rate < 0.0 || p.lazy < 0.0 || p.load < 0.0)
        throw std::runtime_error("rate, lazy, and load must be non-negative in " + path);
    return p;
}

// ---------------------------------------------------------------------------
// bone_cpu : the reference remodeling loop (see reference_cpu.h for contract).
// ---------------------------------------------------------------------------
void bone_cpu(const BoneParams& p,
              std::vector<double>& rho_final,
              std::vector<double>& S_final) {
    const std::size_t N = static_cast<std::size_t>(p.nx) * p.ny;   // voxel count

    // Density field, initialized to a uniform "blank" bone. Two stimulus buffers
    // (Sa, Sb) so the Jacobi relaxation can ping-pong (read one, write the other).
    std::vector<double> rho(N, p.rho_init);
    std::vector<double> Sa(N, 0.0);   // mechanical-stimulus field, buffer A
    std::vector<double> Sb(N, 0.0);   // mechanical-stimulus field, buffer B

    // ---- Outer loop: remodeling steps (each ~ "one month" of biology) -------
    for (int step = 0; step < p.remodel_steps; ++step) {

        // (1) Relax the stimulus field toward equilibrium for the CURRENT
        //     density, using `relax_iters` Jacobi sweeps. We start each step
        //     from the previous field (warm start) so few sweeps suffice.
        double* Sin  = Sa.data();
        double* Sout = Sb.data();
        for (int it = 0; it < p.relax_iters; ++it) {
            for (int y = 0; y < p.ny; ++y)
                for (int x = 0; x < p.nx; ++x)
                    Sout[bone_idx(x, y, p.nx)] =
                        bone_relax_point(x, y, p.nx, p.ny, p.load,
                                         p.load_x0, p.load_x1, Sin, rho.data());
            std::swap(Sin, Sout);     // ping-pong: last-written becomes next input
        }
        // After the swap, `Sin` holds the freshest stimulus field. Keep both
        // Sa/Sb coherent by copying the freshest one back into Sa (so the next
        // step warm-starts from Sa regardless of the sweep parity).
        if (Sin != Sa.data()) Sa = Sb;   // Sin == Sb.data() -> copy Sb into Sa

        // (2) Apply the mechanostat: update every voxel's density from the
        //     settled stimulus field Sa. Independent per voxel -> a plain loop
        //     here, one GPU thread each in kernels.cu.
        std::vector<double> rho_next(N);
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                rho_next[bone_idx(x, y, p.nx)] =
                    bone_apply_stimulus(x, y, p.nx, p.setpoint, p.lazy,
                                        p.rate, p.rho_min, Sa.data(), rho.data());
        rho.swap(rho_next);   // adopt the remodeled density for the next step
    }

    // Hand back the final density and the last settled stimulus field.
    rho_final = rho;
    S_final   = Sa;
}

// ---------------------------------------------------------------------------
// bone_summary : deterministic scalar reductions used by the report.
// ---------------------------------------------------------------------------
void bone_summary(const BoneParams& p, const std::vector<double>& rho,
                  double& total_mass, std::vector<double>& col_mass) {
    total_mass = 0.0;
    col_mass.assign(static_cast<std::size_t>(p.nx), 0.0);
    for (int y = 0; y < p.ny; ++y) {
        for (int x = 0; x < p.nx; ++x) {
            const double r = rho[bone_idx(x, y, p.nx)];
            total_mass += r;
            col_mass[static_cast<std::size_t>(x)] += r;
        }
    }
}
