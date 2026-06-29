// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial Shrake-Rupley baseline + loader
// ---------------------------------------------------------------------------
// Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
//
// ROLE
//   (1) vdw_radius()    : element letter -> van der Waals radius (Angstrom).
//   (2) load_molecule() : parse the tiny text dataset (data/README.md format).
//   (3) sasa_cpu()      : the obviously-correct serial SASA the GPU is verified
//                         against. It just loops the SHARED per-atom functions
//                         from sasa_core.h -- no cleverness on purpose. If the
//                         GPU (which calls the same functions) agrees, we trust it.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h; the
//   per-atom math lives in sasa_core.h (shared __host__ __device__).
//
// READ THIS AFTER: sasa_core.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cstdio>      // std::fprintf (warn on unknown elements)
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream (parse one atom line)
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// vdw_radius: Bondi (1964) van der Waals radii in Angstrom, keyed by the
//   one-letter element symbol. These are the same defaults FreeSASA uses for its
//   simplest model, so our numbers are comparable to a real tool. We only need
//   the handful of elements that appear in proteins/ligands; anything else maps
//   to carbon (1.70) and is logged, so a typo in the input cannot crash the demo.
// ---------------------------------------------------------------------------
double vdw_radius(char element) {
    switch (element) {
        case 'H': return 1.20;   // hydrogen
        case 'C': return 1.70;   // carbon
        case 'N': return 1.55;   // nitrogen
        case 'O': return 1.52;   // oxygen
        case 'S': return 1.80;   // sulfur
        case 'P': return 1.80;   // phosphorus
        case 'F': return 1.47;   // fluorine
        default:
            std::fprintf(stderr,
                "[load] warning: unknown element '%c' -> using carbon radius 1.70 A\n",
                element);
            return 1.70;
    }
}

// ---------------------------------------------------------------------------
// load_molecule: read the simple atom-list text format (see data/README.md).
//   The format is deliberately trivial so the file is human-readable and the
//   loader is short. We skip blank lines and '#' comments, read the declared
//   atom count, then read exactly that many "<element> <x> <y> <z>" rows.
// ---------------------------------------------------------------------------
Molecule load_molecule(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open molecule file: " + path);

    // Helper: pull the next NON-comment, NON-blank line into `line`.
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // Trim leading whitespace to detect a '#' comment after indentation.
            std::size_t p = line.find_first_not_of(" \t\r\n");
            if (p == std::string::npos) continue;     // blank line
            if (line[p] == '#') continue;             // comment line
            return true;
        }
        return false;
    };

    std::string line;
    if (!next_line(line)) throw std::runtime_error("empty molecule file: " + path);

    int n = 0;
    {
        std::istringstream hs(line);
        if (!(hs >> n) || n <= 0)
            throw std::runtime_error("bad atom count (expected a positive integer) in " + path);
    }

    Molecule mol;
    mol.n = n;
    mol.atoms.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!next_line(line))
            throw std::runtime_error("unexpected end of data (need " + std::to_string(n) +
                                     " atoms) in " + path);
        std::istringstream as(line);
        std::string elem;
        double x, y, z;
        if (!(as >> elem >> x >> y >> z))
            throw std::runtime_error("malformed atom line: \"" + line + "\" in " + path);
        Atom& a = mol.atoms[static_cast<std::size_t>(i)];
        a.x = x;
        a.y = y;
        a.z = z;
        // The element column may be a full symbol ("CA", "Cl"); we key the radius
        // on the FIRST letter, uppercased, which is correct for the light protein
        // elements here. THEORY notes the limits of this simplification.
        char first = elem.empty() ? 'C' : elem[0];
        if (first >= 'a' && first <= 'z') first = static_cast<char>(first - ('a' - 'A'));
        a.r = vdw_radius(first);
    }
    return mol;
}

// ---------------------------------------------------------------------------
// sasa_cpu: the serial reference. For each atom we (1) count its exposed test
//   points with the SHARED count_exposed_points(), then (2) turn that integer
//   into an area with the SHARED atom_sasa(). Because these are the exact same
//   functions the kernel calls (sasa_core.h), the integer counts match BIT-FOR-
//   BIT and the per-atom areas match to the last ULP. Complexity: O(n^2 * P)
//   where P = N_SPHERE_POINTS (every atom tests every point against every other
//   atom) -- this all-pairs cost is exactly what the GPU parallelizes.
// ---------------------------------------------------------------------------
void sasa_cpu(const Molecule& mol,
              std::vector<int>& exposed,
              std::vector<double>& sasa) {
    exposed.assign(static_cast<std::size_t>(mol.n), 0);
    sasa.assign(static_cast<std::size_t>(mol.n), 0.0);
    const Atom* atoms = mol.atoms.data();
    for (int i = 0; i < mol.n; ++i) {
        // (1) EXACT integer count of solvent-accessible points (order-independent).
        const int e = count_exposed_points(i, atoms, mol.n, PROBE_RADIUS);
        // (2) Derive the area from that count and atom i's inflated radius.
        const double surf_r = atoms[i].r + PROBE_RADIUS;
        exposed[static_cast<std::size_t>(i)] = e;
        sasa[static_cast<std::size_t>(i)]    = atom_sasa(e, surf_r);
    }
}
