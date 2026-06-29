# Data — 10.02 Real-Time Soft-Tissue Deformation (PBD)

## Committed sample (`sample/cloth_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** simulation parameters (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) |
| Setup | 24×24 particle grid, top row pinned, draping under gravity via PBD |

The "data" is the **simulation setup**; the mesh (particle positions, pinned top
row, distance constraints) is built deterministically from these parameters.

### File format (one line)

```
R  C  spacing  dt  gravity  stiffness  omega  iters  steps
```

| Field | Meaning |
|---|---|
| `R`, `C` | grid rows × columns of particles |
| `spacing` | initial rest spacing (and structural-spring rest length) |
| `dt` | timestep |
| `gravity` | gravitational acceleration (applied in −y) |
| `stiffness` | constraint stiffness in [0,1] |
| `omega` | Jacobi relaxation factor (~1.0) |
| `iters` | constraint-projection iterations per step |
| `steps` | number of timesteps |

Default: `24 24 1 0.02 10 1 1 20 300` → a sheet that drapes ~12.6 units.

## "Full dataset" / realistic meshes

Real surgical simulators deform **organ meshes** (tens to hundreds of thousands
of tetrahedral/surface elements) segmented from patient CT/MRI:

- **SOFA** (<https://github.com/sofa-framework/sofa>) — physics engine, GPU PBD + haptics.
- **iMSTK** (<https://github.com/Kitware/iMSTK>) — interactive medical simulation toolkit (CUDA).
- **NVIDIA FleX** (<https://github.com/NVIDIAGameWorks/FleX>) — GPU PBD particle solver.

Bigger mesh: `python scripts/make_synthetic.py --R 128 --C 128 --steps 600`.

## Provenance & honesty

The mesh is a **synthetic grid sheet**, not a patient organ, and the material
model is a simple distance-constraint network. It demonstrates the PBD/GPU
pattern; it is **not** a validated biomechanical model and not for clinical use.
