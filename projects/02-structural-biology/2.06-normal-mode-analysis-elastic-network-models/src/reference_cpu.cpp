// ===========================================================================
// src/reference_cpu.cpp  --  Loader, ANM Hessian, Jacobi eigensolver, mobility
// ---------------------------------------------------------------------------
// Project 2.06 : Normal Mode Analysis / Elastic Network Models
// Compiled by the host compiler only. See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

Protein load_protein(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open protein file: " + path);
    Protein p;
    if (!(in >> p.N >> p.cutoff) || p.N <= 1 || p.cutoff <= 0)
        throw std::runtime_error("bad header (expected 'N cutoff') in " + path);
    p.coords.resize(static_cast<std::size_t>(3) * p.N);
    for (std::size_t i = 0; i < p.coords.size(); ++i)
        if (!(in >> p.coords[i])) throw std::runtime_error("coordinates truncated in " + path);
    return p;
}

void build_hessian(const Protein& p, double gamma, std::vector<double>& H) {
    const int N = p.N, n = 3 * N;
    const double rc2 = p.cutoff * p.cutoff;
    H.assign(static_cast<std::size_t>(n) * n, 0.0);

    // For every ordered pair (i,j) within cutoff, set the off-diagonal 3x3 block
    // -(gamma/d^2)*ΔΔ^T and subtract it from the i-th diagonal block. Visiting
    // both (i,j) and (j,i) keeps H symmetric.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            const double dx = p.coords[3 * j + 0] - p.coords[3 * i + 0];
            const double dy = p.coords[3 * j + 1] - p.coords[3 * i + 1];
            const double dz = p.coords[3 * j + 2] - p.coords[3 * i + 2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > rc2 || d2 == 0.0) continue;
            const double delta[3] = {dx, dy, dz};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    const double off = -gamma * delta[a] * delta[b] / d2;
                    H[static_cast<std::size_t>(3 * i + a) * n + (3 * j + b)] = off;        // block (i,j)
                    H[static_cast<std::size_t>(3 * i + a) * n + (3 * i + b)] -= off;        // block (i,i)
                }
        }
    }
}

void jacobi_eigenvalues(std::vector<double> A, int n, std::vector<double>& eig) {
    // Cyclic Jacobi: repeatedly apply Givens rotations that zero the largest
    // off-diagonal entries; the diagonal converges to the eigenvalues. Clear and
    // obviously correct for symmetric matrices -- the trusted reference.
    const int MAX_SWEEPS = 100;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += A[static_cast<std::size_t>(p) * n + q] * A[static_cast<std::size_t>(p) * n + q];
        if (off < 1e-22) break;                       // converged: off-diagonals ~0

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[static_cast<std::size_t>(p) * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = A[static_cast<std::size_t>(p) * n + p];
                const double aqq = A[static_cast<std::size_t>(q) * n + q];
                // Rotation angle that diagonalizes the 2x2 (p,q) sub-block.
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                // Apply J^T A J: rotate columns p,q then rows p,q.
                for (int k = 0; k < n; ++k) {
                    const double akp = A[static_cast<std::size_t>(k) * n + p];
                    const double akq = A[static_cast<std::size_t>(k) * n + q];
                    A[static_cast<std::size_t>(k) * n + p] = c * akp - s * akq;
                    A[static_cast<std::size_t>(k) * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[static_cast<std::size_t>(p) * n + k];
                    const double aqk = A[static_cast<std::size_t>(q) * n + k];
                    A[static_cast<std::size_t>(p) * n + k] = c * apk - s * aqk;
                    A[static_cast<std::size_t>(q) * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    eig.resize(n);
    for (int i = 0; i < n; ++i) eig[i] = A[static_cast<std::size_t>(i) * n + i];
    std::sort(eig.begin(), eig.end());
}

void mobility(const std::vector<double>& eig, const std::vector<double>& evec,
              int N, double thr, std::vector<double>& mob) {
    const int n = 3 * N;
    mob.assign(N, 0.0);
    // Sum over non-trivial modes: a low-frequency (small eigenvalue) mode
    // contributes a lot of motion (1/eig), weighted by the residue's amplitude
    // in that mode. Eigenvectors are column-major: v_k[row] = evec[k*n + row].
    for (int k = 0; k < n; ++k) {
        if (eig[k] <= thr) continue;                  // skip the ~zero rigid-body modes
        const double w = 1.0 / eig[k];
        const double* vk = &evec[static_cast<std::size_t>(k) * n];
        for (int i = 0; i < N; ++i) {
            const double vx = vk[3 * i + 0], vy = vk[3 * i + 1], vz = vk[3 * i + 2];
            mob[i] += w * (vx * vx + vy * vy + vz * vz);
        }
    }
}
