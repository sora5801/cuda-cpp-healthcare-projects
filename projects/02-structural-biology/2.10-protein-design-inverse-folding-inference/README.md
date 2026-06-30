# 2.10 — Protein Design / Inverse Folding Inference

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope teaching model**, not a real protein-design tool — see
> "Limitations & honesty" below and [THEORY.md](THEORY.md) §7._

## Summary

**Inverse folding** is the reverse of structure prediction: instead of "given a
sequence, what shape does it fold into?", it asks "**given a fixed backbone
shape, what amino-acid sequence will fold into it?**" — the core step in
computational protein design. This project takes a protein backbone (a chain of
Cα atom coordinates) and designs a sequence for it on the GPU, then measures how
many positions match the protein's native sequence ("recovery"). It is a
deliberately small, transparent stand-in for **ProteinMPNN**: it captures the one
dominant signal real models learn — hydrophobic residues belong in the buried
core, polar residues on the exposed surface — and uses it to teach the GPU
pattern of *scoring every residue independently in parallel*.

## What this computes & why the GPU helps

The computation is two passes over the `L` residues:

1. **Burial** — for each residue, count how many other Cα atoms lie within 10 Å.
   This is an **all-pairs `O(L²)`** computation and the analog of *message passing
   over the protein contact graph* in a real graph neural network.
2. **Design** — for each residue, score all 20 amino acids by how well their
   preferred burial matches this position's burial, and pick the best (argmax).

**The parallel bottleneck:** the `O(L²)` burial pass dominates the runtime, and
every residue's burial (and every residue's 20-way argmax) is **independent of the
others**. That independence is what the GPU exploits — one thread per residue —
and the burial kernel additionally **tiles the coordinates through shared memory**
so each global load is reused across the whole block. At real protein and library
scale (thousands of residues × thousands of backbones) this is where the GPU's
throughput wins; on the tiny demo input it is dominated by launch overhead (the
timing line says so honestly).

## The algorithm in brief

- **Contact-based burial:** `n_i = #{ j≠i : ‖r_i − r_j‖² ≤ (10 Å)² }` (squared
  distances, no sqrt).
- **Quadratic-well scoring:** `score(a,i) = −(n_i − b_a)²`, where `b_a` is amino
  acid `a`'s preferred burial (hydrophobic → high, polar → low).
- **Per-position argmax decode:** `design_i = argmax_a score(a,i)` (the
  temperature-0 limit of ProteinMPNN's autoregressive sampler), ties broken by
  lowest index for determinism.
- **Native sequence recovery:** the fraction of positions where the design equals
  the native residue — the headline metric ProteinMPNN reports (~50% on real
  proteins).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-design-inverse-folding-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-design-inverse-folding-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-design-inverse-folding-inference.sln /p:Configuration=Release /p:Platform=x64
```

The project links only the CUDA runtime (`cudart`); no extra CUDA library is
needed because both kernels are hand-rolled (that is the lesson).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the designed sequence
and native recovery, shows the exact GPU-vs-CPU agreement check, and prints a
timing line.

## Data

- **Sample (committed):** `data/sample/backbone_sample.txt` — a tiny **synthetic**
  60-residue backbone so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain real
  backbones; `scripts/make_synthetic.py` regenerates (or enlarges) the sample.
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  synthetic and labeled synthetic everywhere.

Catalog dataset notes: CATH protein structure database — 500k+ domain structures
(<https://www.cathdb.info>); PDB training set for ProteinMPNN
(<https://www.rcsb.org>); ProteinGym benchmark — mutational fitness
(<https://github.com/OATML-Markslab/ProteinGym>); CAMEO validation
(<https://www.cameo3d.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
2.10 -- Protein Design / Inverse Folding Inference
inverse folding (reduced-scope teaching model): design a sequence for a fixed backbone
residues L = 60   buried (>=16 contacts) = 28   exposed = 32
native   : TFFFFFFFFFFFFFNDFFFFHFFFFFFFWPPGSSKGPPNRTRDDDRRRERRRRREEEEEG
designed : FFFFFFFFFFFFFFFFFFFFFFFFFFFFNPPGSSSGPPNRRRDDDRRRERRRRREEEEEE
native sequence recovery: 87%
RESULT: PASS (GPU design matches CPU reference exactly)
```

The program computes the design on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree **exactly** —
the burial counts, designed residues, and scores are all integers computed by the
*same* shared scoring function, so there is no floating-point tolerance to
fudge. That exact agreement is the correctness guarantee. Note how the core
(buried) positions are designed as hydrophobic Phe (`F`) and the surface positions
as charged Arg/Asp/Glu (`R`/`D`/`E`) — exactly the hydrophobic-core rule at work.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the backbone, runs CPU + GPU, verifies
   exact agreement, prints the design and recovery.
2. [`src/inverse_folding.h`](src/inverse_folding.h) — the shared
   `__host__ __device__` scoring core (the one formula both CPU and GPU use).
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the data model, the file loader, and the trusted serial baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the two-kernel idea.
5. [`src/kernels.cu`](src/kernels.cu) — the burial kernel (shared-memory tiling)
   and the design kernel (per-residue argmax), plus the host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **ProteinMPNN** (<https://github.com/dauparas/ProteinMPNN>) — the official GPU
  inverse-folding model (Baker Lab); study its message-passing encoder and
  autoregressive decoder, which this project is a tiny shadow of.
- **LigandMPNN** (<https://github.com/dauparas/LigandMPNN>) — inverse folding with
  small-molecule/ligand context; learn how the design graph gains extra nodes.
- **ESM-IF1** (<https://github.com/facebookresearch/esm>) — an alternative
  inverse-folding model trained on millions of *predicted* structures.
- **RFdiffusion** (<https://github.com/RosettaCommons/RFdiffusion>) — generates the
  backbones that ProteinMPNN then designs sequences for; the upstream half of the
  de-novo design pipeline.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

"Score N independent items" (`docs/PATTERNS.md` §1, exemplar `1.12` Tanimoto):
**one thread per residue**, in two kernels. Kernel 1 (burial) is an all-pairs
gather that **stages coordinates through shared memory** so each global load is
reused block-wide — the same tiling trick as a tiled matrix multiply or N-body
force kernel. Kernel 2 (design) is a private 20-way argmax per thread with no
shared memory or atomics. The per-residue physics lives in **one shared
`__host__ __device__` header** (`inverse_folding.h`) so the CPU and GPU compute
byte-identical integer scores → **exact** verification (PATTERNS.md §2, §4).

## Exercises

1. **Temperature sampling.** Replace the argmax in `design_kernel` with
   temperature-controlled sampling (softmax over `score(a,i)/T` + a per-thread
   cuRAND draw). Watch recovery fall but sequence diversity rise — exactly the
   trade-off ProteinMPNN exposes. (Keep the deterministic argmax path for the
   verified demo.)
2. **A real distance graph.** The burial kernel re-reads coordinates `O(L²)`
   times. Build an explicit k-nearest-neighbour list first (a spatial grid / cell
   list), then have the design step gather only over real neighbours — the
   structure of an actual GNN message-passing layer.
3. **Richer geometry.** Add a virtual Cβ per residue (from the N–Cα–C frame) and
   make the score depend on Cβ–Cβ contacts, which encode side-chain direction —
   closer to what ProteinMPNN's edge features carry.
4. **Pairwise coupling.** Add a term that rewards compatible *neighbouring* residue
   choices (e.g. charge complementarity), turning the independent per-position
   argmax into a coupled optimisation — the first step toward autoregressive
   decoding.
5. **Scale it.** Use `make_synthetic.py --shells 40 --per 12` to build a large
   backbone and watch the GPU-vs-CPU timing gap on the `O(L²)` burial pass open up.

## Limitations & honesty

- **This is a reduced-scope teaching model, not ProteinMPNN.** Real inverse
  folding uses a *trained* graph neural network with learned geometric edge
  features and autoregressive, coupled decoding. We use a hand-written quadratic
  energy on a single geometric scalar (neighbour count) and decode each position
  independently. See [THEORY.md](THEORY.md) §7 for the full gap.
- **The sample is synthetic.** `data/sample/backbone_sample.txt` is a generated
  toy "protein" (concentric shells), not a real PDB structure, engineered so the
  burial signal is clean and the recovery number is interpretable. It is labeled
  synthetic everywhere.
- **Recovery is not a quality claim.** The 87% recovery on the synthetic input
  reflects how the sample was built (natives chosen to mostly fit burial, then 25%
  mutated). On real proteins this simple model would recover far less than
  ProteinMPNN's ~50%. The number teaches the *metric*, not a design capability.
- **No clinical use.** Nothing here designs a real therapeutic protein.
