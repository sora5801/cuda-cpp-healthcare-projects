// ===========================================================================
// src/reference_cpu.cpp  --  Loader, mesh init, serial PBD reference
// ---------------------------------------------------------------------------
// Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
// Compiled by the host compiler only. Physics lives in pbd.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

PbdParams load_pbd(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open PBD parameter file: " + path);
    PbdParams P{};
    if (!(in >> P.R >> P.C >> P.spacing >> P.dt >> P.gravity
             >> P.stiffness >> P.omega >> P.iters >> P.steps))
        throw std::runtime_error("bad parameters (expected "
            "'R C spacing dt gravity stiffness omega iters steps') in " + path);
    if (P.R <= 1 || P.C <= 1 || P.steps < 0 || P.iters < 0 || P.dt <= 0)
        throw std::runtime_error("invalid PBD parameters in " + path);
    return P;
}

void init_mesh(const PbdParams& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
               std::vector<double>& w) {
    const int N = P.R * P.C;
    x.resize(N);
    v.assign(N, Vec3{0.0, 0.0, 0.0});
    w.resize(N);
    for (int r = 0; r < P.R; ++r) {
        for (int c = 0; c < P.C; ++c) {
            const int i = r * P.C + c;
            // Flat sheet in the x-z plane; the top edge (r==0) is pinned.
            x[i] = Vec3{c * P.spacing, 0.0, r * P.spacing};
            w[i] = (r == 0) ? 0.0 : 1.0;   // inverse mass: 0 = immovable (pinned)
        }
    }
}

void simulate_cpu(const PbdParams& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
                  const std::vector<double>& w) {
    const int N = P.R * P.C;
    std::vector<Vec3> pa(N), pb(N);

    for (int step = 0; step < P.steps; ++step) {
        // 1) Predict positions under gravity into pa.
        for (int i = 0; i < N; ++i)
            pa[i] = pbd_predict(x[i], v[i], w[i], P.dt, P.gravity);

        // 2) Jacobi constraint projection: read `src`, write `dst`, swap.
        Vec3* src = pa.data();
        Vec3* dst = pb.data();
        for (int it = 0; it < P.iters; ++it) {
            for (int r = 0; r < P.R; ++r)
                for (int c = 0; c < P.C; ++c) {
                    const int i = r * P.C + c;
                    dst[i] = src[i] + pbd_correction(r, c, P, src, w.data());
                }
            Vec3* tmp = src; src = dst; dst = tmp;
        }

        // 3) Velocity update from the position change; commit x = final position.
        for (int i = 0; i < N; ++i) {
            v[i] = pbd_new_velocity(src[i], x[i], w[i], P.dt);
            x[i] = src[i];
        }
    }
}
