# Data — 2.5 Coarse-Grained / MARTINI Simulation

## Committed sample (`sample/cg_system.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** CG system (`scripts/make_synthetic.py`) — no patient data |
| License | Public domain (CC0) — it is synthetic |
| Size | < 2 KB (16 beads) |
| Setup | 8 apolar "C" beads + 8 polar "P" beads in a 6 nm periodic box, at rest |

The "data" is a small **coarse-grained molecular system**: a list of beads, each
with a position, a velocity, and a MARTINI-like *type*. The simulation
(`src/`) advances it with velocity-Verlet under a Lennard-Jones force field.

### File format

```
line 1 :  n box dt steps rcut mass sigma epsCC epsCP epsPP
n lines:  x y z vx vy vz type
```

| Field | Meaning | Units |
|---|---|---|
| `n` | number of beads | — |
| `box` | cubic box edge (periodic in x, y, z) | nm |
| `dt` | integration timestep | reduced |
| `steps` | number of velocity-Verlet steps | — |
| `rcut` | non-bonded cutoff (pairs beyond `rcut` are ignored) | nm |
| `mass` | bead mass (shared by all beads here) | reduced |
| `sigma` | Lennard-Jones contact distance | nm |
| `epsCC`, `epsCP`, `epsPP` | LJ well depths for each type pair (the MARTINI interaction matrix) | kJ/mol-ish |
| `x y z` | bead position | nm |
| `vx vy vz` | bead velocity | nm / time-unit |
| `type` | `0` = C (apolar / lipid-tail-like), `1` = P (polar / water-like) | — |

Default: `16 6 0.005 200 2.5 1 0.47 4 1 4`. Because `epsCC = epsPP = 4 > epsCP = 1`
("like likes like"), the C and P groups stay demixed and tighten into clusters —
the same physics that drives lipid self-assembly in real MARTINI runs.

Regenerate or resize (no download needed):

```bash
python scripts/make_synthetic.py                       # the committed 16-bead box
python scripts/make_synthetic.py --per-side 3 --steps 400   # 54 beads, longer run
```

## "Full dataset" / realistic MARTINI systems

Real MARTINI simulations use membrane systems built by dedicated tools and run in
GROMACS. None of these are needed for the demo, and none are redistributed here:

- **CHARMM-GUI Martini Maker** (<https://charmm-gui.org>) — builds membranes /
  membrane-protein systems as GROMACS `.gro` + `.itp` files (registration required).
- **MARTINI force-field files** (<https://cgmartini.nl>) — the official bead types
  and interaction matrix (the `eps` table this toy hard-codes for two types).
- **`insane.py`** (<https://github.com/Tsjerk/Insane>) and **TS2CG**
  (<https://github.com/weria-pezeshkian/TS2CG>) — assemble bilayers / vesicles.
- **EMDB** (<https://www.ebi.ac.uk/emdb/>) — cryo-EM maps used to validate large
  CG assemblies (e.g. viral capsids).

`scripts/download_data.ps1` / `.sh` print these pointers and do **not** bypass any
registration (CLAUDE.md §8).

## Provenance & honesty

The system is a **synthetic two-type bead box**, not a real lipid membrane, and the
force field is a single shared `sigma` with a 2×2 `eps` matrix — a deliberate
teaching reduction of MARTINI 3's ~800-level interaction table. It demonstrates the
non-bonded CG-MD GPU pattern; it is **not** a validated molecular model and **not
for any clinical or scientific production use**.
