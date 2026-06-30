// ===========================================================================
// src/reference_cpu.cpp  --  Loader, grid builder, serial PBE reference, SASA
// ---------------------------------------------------------------------------
// Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
//
// Compiled by the host compiler only. The per-cell relaxation physics lives in
// pbe.h and is shared verbatim with the GPU kernel (kernels.cu), so the serial
// solver here and the parallel solver there converge to the same field. This
// file is the TRUSTED baseline: small, plain, readable, heavily commented.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::min, std::max
#include <cmath>       // std::sqrt, std::floor
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_atoms: parse the tiny .pqr-style text file (format in data/README.md).
//   Header line:  natoms n h eps_in eps_out kappa2 iters
//   Body:         natoms rows of  "x y z q radius".
//   We fill P_out with the grid numerics; charge_to_phi (the unit-folding
//   factor) is derived later in build_problem once h is known.
// ---------------------------------------------------------------------------
std::vector<Atom> load_atoms(const std::string& path, GridParams& P_out) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open atom file: " + path);

    int natoms = 0;
    // Read the header: how many atoms, and the grid/physics knobs.
    if (!(in >> natoms >> P_out.n >> P_out.h >> P_out.eps_in
             >> P_out.eps_out >> P_out.kappa2 >> P_out.iters))
        throw std::runtime_error("bad header (expected "
            "'natoms n h eps_in eps_out kappa2 iters') in " + path);

    if (natoms <= 0 || P_out.n < 8 || P_out.h <= 0.0 || P_out.iters < 0
        || P_out.eps_in <= 0.0 || P_out.eps_out <= 0.0 || P_out.kappa2 < 0.0)
        throw std::runtime_error("invalid PBE parameters in " + path);

    std::vector<Atom> atoms;
    atoms.reserve(natoms);
    for (int i = 0; i < natoms; ++i) {
        Atom a{};
        if (!(in >> a.x >> a.y >> a.z >> a.q >> a.radius))
            throw std::runtime_error("truncated atom row in " + path);
        if (a.radius <= 0.0)
            throw std::runtime_error("non-positive atomic radius in " + path);
        atoms.push_back(a);
    }
    return atoms;
}

// ---------------------------------------------------------------------------
// build_problem: turn the atom list into the three grids the solver needs.
//
//   Step 1 -- place the grid. We centre an n^3 cube of spacing h on the atoms'
//   bounding-box centre, so the molecule sits in the middle with a solvent
//   margin out to the grounded boundary (this margin is why the grounded-box
//   boundary condition is acceptable here; THEORY discusses the better
//   Debye-Huckel boundary).
//
//   Step 2 -- eps(r) and kappa^2(r). A cell whose centre lies within ANY atom's
//   radius is "protein interior": low dielectric, no mobile ions (kappa^2 = 0).
//   Every other cell is "solvent": high dielectric eps_out and screening
//   kappa^2 = P.kappa2. This sharp two-value map is the simplest dielectric
//   model (van der Waals surface); APBS smooths it (THEORY "real world").
//
//   Step 3 -- rho(r). Each atom's point charge is deposited on its NEAREST grid
//   cell (nearest-grid-point assignment) and multiplied by charge_to_phi, the
//   factor that folds 4*pi/(eps0) and the grid geometry into reduced potential
//   units (kT/e). We define charge_to_phi = 4*pi / h here (a teaching-scale
//   constant, documented in THEORY) so phi comes out O(1) and easy to read; the
//   ABSOLUTE scale is illustrative, but it is IDENTICAL on CPU and GPU, which is
//   all the verification needs.
// ---------------------------------------------------------------------------
PbeProblem build_problem(const std::vector<Atom>& atoms, const GridParams& Pin) {
    PbeProblem prob;
    prob.P = Pin;
    const int n = prob.P.n;
    const double h = prob.P.h;
    const size_t N = static_cast<size_t>(n) * n * n;

    // Unit-folding factor (see header comment). Folded once, reused everywhere.
    const double PI = 3.14159265358979323846;
    prob.P.charge_to_phi = 4.0 * PI / h;

    // --- Step 1: centre the grid on the atoms' bounding box -----------------
    double lo_x = atoms[0].x, lo_y = atoms[0].y, lo_z = atoms[0].z;
    double hi_x = lo_x, hi_y = lo_y, hi_z = lo_z;
    for (const Atom& a : atoms) {
        lo_x = std::min(lo_x, a.x); hi_x = std::max(hi_x, a.x);
        lo_y = std::min(lo_y, a.y); hi_y = std::max(hi_y, a.y);
        lo_z = std::min(lo_z, a.z); hi_z = std::max(hi_z, a.z);
    }
    const double cx = 0.5 * (lo_x + hi_x);   // bounding-box centre
    const double cy = 0.5 * (lo_y + hi_y);
    const double cz = 0.5 * (lo_z + hi_z);
    // Origin = centre minus half the grid extent, so the box is centred.
    const double half = 0.5 * (n - 1) * h;
    prob.origin_x = cx - half;
    prob.origin_y = cy - half;
    prob.origin_z = cz - half;

    // --- Step 2: dielectric + screening maps (default = solvent) -----------
    prob.eps.assign(N, prob.P.eps_out);
    prob.kappa2.assign(N, prob.P.kappa2);
    prob.rho.assign(N, 0.0);

    // Mark protein-interior cells. We only need to scan the bounding box of
    // each atom's sphere in grid coordinates (not the whole grid) -- a small,
    // readable optimization that keeps setup O(atoms * sphere-volume).
    for (const Atom& a : atoms) {
        const double r2 = a.radius * a.radius;
        // grid-coordinate window covering this atom's sphere
        const int gx0 = std::max(0, (int)std::floor((a.x - a.radius - prob.origin_x) / h));
        const int gx1 = std::min(n - 1, (int)std::floor((a.x + a.radius - prob.origin_x) / h));
        const int gy0 = std::max(0, (int)std::floor((a.y - a.radius - prob.origin_y) / h));
        const int gy1 = std::min(n - 1, (int)std::floor((a.y + a.radius - prob.origin_y) / h));
        const int gz0 = std::max(0, (int)std::floor((a.z - a.radius - prob.origin_z) / h));
        const int gz1 = std::min(n - 1, (int)std::floor((a.z + a.radius - prob.origin_z) / h));
        for (int gz = gz0; gz <= gz1; ++gz)
        for (int gy = gy0; gy <= gy1; ++gy)
        for (int gx = gx0; gx <= gx1; ++gx) {
            const double wx = prob.origin_x + gx * h;   // world coord of cell
            const double wy = prob.origin_y + gy * h;
            const double wz = prob.origin_z + gz * h;
            const double dx = wx - a.x, dy = wy - a.y, dz = wz - a.z;
            if (dx * dx + dy * dy + dz * dz <= r2) {
                const int c = pbe_idx(gx, gy, gz, n);
                prob.eps[c] = prob.P.eps_in;   // low dielectric inside protein
                prob.kappa2[c] = 0.0;          // no mobile ions inside protein
            }
        }
    }

    // --- Step 3: deposit each atom's charge on its nearest grid cell --------
    for (const Atom& a : atoms) {
        const int gx = (int)std::floor((a.x - prob.origin_x) / h + 0.5);
        const int gy = (int)std::floor((a.y - prob.origin_y) / h + 0.5);
        const int gz = (int)std::floor((a.z - prob.origin_z) / h + 0.5);
        // Keep charges strictly interior so they are never on the grounded face.
        if (gx < 1 || gx > n - 2 || gy < 1 || gy > n - 2 || gz < 1 || gz > n - 2)
            continue;
        const int c = pbe_idx(gx, gy, gz, n);
        prob.rho[c] += a.q * prob.P.charge_to_phi;
    }

    return prob;
}

// ---------------------------------------------------------------------------
// solve_cpu: the serial red-black Gauss-Seidel reference.
//
//   Red-black ordering colours cell (x,y,z) by the parity of (x+y+z). One
//   "sweep" updates all RED cells (parity 0) using their neighbours, then all
//   BLACK cells (parity 1). Because every neighbour of a red cell is black and
//   vice-versa, within a colour no cell depends on another of the same colour:
//   the colour update is order-independent, which is what lets the GPU do all
//   same-colour cells in parallel and STILL match this serial loop bit-for-bit.
//
//   We do exactly that ordering here (not a plain lexicographic sweep) so the
//   CPU reference and the GPU kernel perform identical arithmetic in identical
//   order, making verification near-exact (THEORY "How we verify"). Boundary
//   cells (the outer shell) are never updated; they stay at phi = 0 (grounded
//   box). The per-cell formula is the shared pbe_relax_cell() in pbe.h.
// ---------------------------------------------------------------------------
void solve_cpu(const PbeProblem& prob, std::vector<double>& phi) {
    const GridParams& P = prob.P;
    const int n = P.n;
    const double* eps = prob.eps.data();
    const double* kappa2 = prob.kappa2.data();
    const double* rho = prob.rho.data();

    for (int it = 0; it < P.iters; ++it) {
        // color = 0 (red) then color = 1 (black)
        for (int color = 0; color < 2; ++color) {
            for (int z = 1; z < n - 1; ++z)
            for (int y = 1; y < n - 1; ++y)
            for (int x = 1; x < n - 1; ++x) {
                if (((x + y + z) & 1) != color) continue;   // skip other colour
                const int c = pbe_idx(x, y, z, n);
                phi[c] = pbe_relax_cell(x, y, z, P, phi.data(), rho, eps, kappa2);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compute_sasa: the geometric "surface" half of the project title.
//
//   Shrake-Rupley: around each atom, sample `sphere_points` points on a sphere
//   of radius (atom_radius + probe_radius) -- the locus a water-sized probe's
//   centre can reach. A sample point is "exposed" if it lies outside every
//   OTHER atom's expanded sphere; the fraction exposed times the sphere area is
//   that atom's accessible area, and the sum is the molecular SASA.
//
//   We distribute the sample points with the deterministic golden-spiral
//   (Fibonacci sphere) so the result is reproducible and main.cu can print it
//   as a fixed scalar. This is a textbook O(atoms^2 * points) method -- fine for
//   the tiny teaching molecule; production codes use neighbour grids.
// ---------------------------------------------------------------------------
double compute_sasa(const std::vector<Atom>& atoms, double probe_radius, int sphere_points) {
    const double PI = 3.14159265358979323846;
    const double golden = PI * (3.0 - std::sqrt(5.0));   // golden angle (rad)
    double total = 0.0;

    for (size_t i = 0; i < atoms.size(); ++i) {
        const Atom& ai = atoms[i];
        const double Ri = ai.radius + probe_radius;   // expanded radius
        int exposed = 0;
        for (int s = 0; s < sphere_points; ++s) {
            // Fibonacci-sphere sample direction (deterministic, near-uniform).
            const double zt = 1.0 - 2.0 * (s + 0.5) / sphere_points;  // [-1,1]
            const double rr = std::sqrt(std::max(0.0, 1.0 - zt * zt));
            const double phi_ang = golden * s;
            const double px = ai.x + Ri * rr * std::cos(phi_ang);
            const double py = ai.y + Ri * rr * std::sin(phi_ang);
            const double pz = ai.z + Ri * zt;
            // Exposed unless it falls inside another atom's expanded sphere.
            bool buried = false;
            for (size_t j = 0; j < atoms.size(); ++j) {
                if (j == i) continue;
                const Atom& aj = atoms[j];
                const double Rj = aj.radius + probe_radius;
                const double dx = px - aj.x, dy = py - aj.y, dz = pz - aj.z;
                if (dx * dx + dy * dy + dz * dz < Rj * Rj) { buried = true; break; }
            }
            if (!buried) ++exposed;
        }
        const double area_i = 4.0 * PI * Ri * Ri;       // full sphere area
        total += area_i * (double)exposed / (double)sphere_points;
    }
    return total;
}
