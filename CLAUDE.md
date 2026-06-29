# CLAUDE.md — Repository Contract & Working Agreement

> This file is auto-loaded by Claude Code at the start of every session. It is the **single source of
> truth** for how work happens in this repository. Read it fully before doing anything. When in doubt,
> follow this document over any habit or default. If a rule here ever blocks progress, stop and surface
> the conflict in a `push-note` rather than silently working around it.

---

## 1. Mission & philosophy

This repository, **`cuda-cpp-healthcare-projects`**, is a **didactic study collection**: ~301 self-contained
CUDA C++ projects spanning drug discovery, structural biology, genomics, medical imaging, radiation
physics, physiology, medical AI, neuroscience, epidemiology, biomechanics, biotechnology, omics,
pharmacology, and emerging frontiers.

The reader (the repository owner) is using this as **study material**. That single fact drives every
decision:

- **Teaching beats cleverness.** A slower kernel that a learner can follow is better than a fast one they
  cannot. When forced to choose, choose the version that teaches, and *explain the faster version in
  comments*.
- **Every artifact must explain itself.** Code, build files, data scripts, and demos are all teaching
  surfaces. A file with sparse comments is considered **unfinished**, no matter how well it runs.
- **Nothing is a black box.** If a project uses a library kernel (cuBLAS, cuFFT, Thrust…), the surrounding
  comments must explain *what that call computes, why it is used here, and what it would take to write by
  hand*.
- **Reproducibility is sacred.** Anyone should be able to clone, open the Visual Studio solution, build,
  run the demo, and see the documented result — on a normal Windows + NVIDIA machine.

> **Not for clinical use.** Everything here is educational. No project output may be used for diagnosis,
> treatment, or any real medical decision. Datasets that touch real patients are handled under §8.

---

## 2. Source-of-truth inputs

Two files at the repo root define *what* to build. They were generated before this repo existed and must
be treated as read-only references (do not rewrite them; if something is wrong, note it in a push-note):

| File | Role |
|------|------|
| `CUDA_CPP_Healthcare_Projects.xlsx` | The **catalog**. One row per project. Columns: `Section`, `Domain`, `ID`, `Project`, `Difficulty`, `Maturity`, `Deep Dive`, `Key Algorithms`, `Datasets`, `Starter Repos / Tools`, `CUDA Libraries & GPU Pattern`. The `By Domain` tab has counts; `README` tab has the legend. |
| `CUDA_CPP_Healthcare_Projects_DeepDive.md` | The **prose reference**. The same 301 projects in long form. Each entry has Deep dive / Key algorithms / Datasets / Starter repos / CUDA pattern. |

**How the catalog maps into each project** (do this mapping for every project):

- `Project` + `ID` → folder name and README title.
- `Deep Dive` → the README "What this computes & why" section and the opening of `THEORY.md`.
- `Key Algorithms` → the algorithms `THEORY.md` must explain and the code must implement.
- `Datasets` → `data/README.md` provenance + `scripts/download_data.*`.
- `Starter Repos / Tools` → README "Prior art & further reading" (study these, do **not** copy-paste code
  wholesale; reimplement didactically and credit the source).
- `CUDA Libraries & GPU Pattern` → the implementation approach and the "GPU mapping" section of `THEORY.md`.

A helper, `tools/catalog.py`, parses the xlsx (via `openpyxl`) and the MD into a single `catalog.json` so
agents never have to parse Office files by hand. Treat `catalog.json` as the machine-readable catalog.

---

## 3. Repository layout

```
cuda-cpp-healthcare-projects/
├── CLAUDE.md                              # this file (the contract)
├── README.md                             # front door: what the repo is, how to use it, index
├── LICENSE                               # MIT (code) — see §8 for data licensing
├── .gitignore                            # ignores build artifacts, large data, secrets
├── CHANGELOG.md                          # concise index of every push, links into /push-notes
├── CUDA_CPP_Healthcare_Projects.xlsx     # catalog (source of truth)
├── CUDA_CPP_Healthcare_Projects_DeepDive.md
├── catalog.json                          # generated machine-readable catalog
├── docs/
│   ├── COMMENTING_STANDARD.md            # the full commenting rubric (canonical copy)
│   ├── BUILD_GUIDE.md                    # installing CUDA + VS, building any project
│   ├── PROJECT_TEMPLATE/                 # the canonical empty project (copied to start each one)
│   └── STATUS.md                         # generated dashboard: per-project todo/in-progress/done
├── push-notes/                           # one didactic note per push (§7)
│   └── 2026-06-28-00-bootstrap.md
├── tools/
│   ├── catalog.py                        # xlsx+md -> catalog.json
│   ├── scaffold.py                       # catalog.json -> all 301 project skeletons
│   ├── verify_project.py                 # checks a project meets the Definition of Done
│   ├── status.py                         # (re)generates docs/STATUS.md work-queue dashboard
│   └── new_pushnote.py                   # generates a dated push-note stub
├── showcase/                             # top-level demo that ties everything together (§6.3)
│   ├── showcase.sln
│   └── ...
└── projects/
    ├── 01-drug-discovery/
    │   ├── 1.01-molecular-dynamics-engine/
    │   ├── 1.12-molecular-fingerprint-similarity-search/
    │   └── ...
    ├── 02-structural-biology/
    ├── 03-genomics/
    ├── 04-medical-imaging/
    ├── 05-radiation-therapy-medphys/
    ├── 06-physiology-systems-biology/
    ├── 07-medical-ai/
    ├── 08-neuroscience-bci/
    ├── 09-epidemiology-public-health/
    ├── 10-biomechanics-devices/
    ├── 11-biotech-synthbio/
    ├── 12-omics-data-processing/
    ├── 13-pharmacology-quant/
    └── 14-emerging-frontiers/
```

### Domain folder slugs (section number → slug)

```
01-drug-discovery               08-neuroscience-bci
02-structural-biology           09-epidemiology-public-health
03-genomics                     10-biomechanics-devices
04-medical-imaging              11-biotech-synthbio
05-radiation-therapy-medphys    12-omics-data-processing
06-physiology-systems-biology   13-pharmacology-quant
07-medical-ai                   14-emerging-frontiers
```

### Project folder naming

`projects/<domain-slug>/<ID>-<project-slug>/` where:

- `<ID>` is the catalog ID with the minor number **zero-padded to two digits** so folders sort correctly
  (`1.1` → `1.01`, `3.10` → `3.10`). This keeps `1.01 … 1.35` in numeric order.
- `<project-slug>` is the `Project` name lowercased, ASCII, spaces/`/`→`-`, no punctuation
  (e.g., "Quantum Chemistry / DFT on GPU" → `quantum-chemistry-dft`).

`tools/scaffold.py` generates these names deterministically — always use it rather than inventing names.

---

## 4. The standard project layout (Definition of "a project exists")

Every project folder MUST contain exactly this structure. `docs/PROJECT_TEMPLATE/` is the canonical copy;
`scaffold.py` stamps it out with catalog fields pre-filled.

```
<ID>-<slug>/
├── README.md            # the learner's entry point (see §4.1)
├── THEORY.md            # the deep didactic explanation (the "why") (see §4.2)
├── src/                 # implementation: .cu / .cpp / .cuh / .h — maximally commented
│   ├── main.cu          # entry point: parses args, loads data, runs, prints/saves result
│   ├── kernels.cu       # the GPU kernels (one teaching-focused kernel per concept)
│   ├── kernels.cuh      # kernel declarations + extensive header comments
│   ├── reference_cpu.cpp# a plain-C++ reference implementation used to VERIFY the GPU result
│   └── util/            # timing, error-checking macros, I/O helpers (shared style, copied per project)
├── data/
│   ├── sample/          # a TINY committed sample so the demo runs with zero downloads
│   └── README.md        # provenance, license, size, checksum, what each field means
├── scripts/
│   ├── download_data.ps1 / .sh   # fetch the full dataset (documented, idempotent)
│   └── make_synthetic.py         # generate synthetic data when no public set is usable
├── demo/
│   ├── run_demo.ps1 / .sh        # one command: build (if needed) + run on sample + show result
│   ├── expected_output.txt       # what the learner should see (used by verify)
│   └── README.md                 # what the demo demonstrates, annotated
├── build/
│   ├── <slug>.sln                # Visual Studio 2026 solution (v145 toolset)
│   ├── <slug>.vcxproj            # CUDA project (see §5)
│   └── <slug>.vcxproj.filters
├── CMakeLists.txt       # OPTIONAL cross-platform build (nice-to-have; VS is the required one)
└── .gitignore           # ignores x64/, *.obj, *.exe, downloaded data, etc.
```

> A project is **not done** until *all* of the above exist, the VS build succeeds, the demo runs and
> matches `expected_output.txt` (within documented tolerance), and the comment density passes
> `verify_project.py`. See §9.

### 4.1 `README.md` (per project) — required sections

1. **Title** — `# <ID> — <Project name>` and the difficulty/maturity badges.
2. **One-paragraph summary** — what it does, in plain language.
3. **What this computes & why the GPU helps** — from the catalog `Deep Dive`; name the bottleneck that is
   parallelized.
4. **The algorithm in brief** — bullet list of the key algorithms (link to `THEORY.md` for depth).
5. **Build** — exact steps (open `build/<slug>.sln` in VS 2022, select `Release|x64`, Build). Link
   `docs/BUILD_GUIDE.md`.
6. **Run the demo** — the single command in `demo/`.
7. **Data** — what the sample is, how to get the full dataset (`scripts/download_data.*`), licensing.
8. **Expected output** — what success looks like; how the GPU result is checked against the CPU reference.
9. **Code tour** — a short guided reading order through `src/` ("start in `main.cu`, then `kernels.cu`…").
10. **Prior art & further reading** — the catalog `Starter Repos / Tools`, with one line each on what to
    learn from them.
11. **Exercises** — 3–5 "try this next" extensions for the learner (this is study material — leave them
    something to do).
12. **Limitations & honesty** — what is simplified, what is synthetic, what would differ in production.

### 4.2 `THEORY.md` (per project) — the deep dive

This is where the teaching lives. Expected contents:

- **The science**: the biology/medicine/physics being modeled, enough to understand the problem.
- **The math**: the governing equations / formal problem statement, with notation defined.
- **The algorithm**: step-by-step, with complexity analysis (serial vs. parallel).
- **The GPU mapping**: how the algorithm becomes threads/blocks/grids; the memory hierarchy used (global /
  shared / registers / texture / constant) and *why*; occupancy and bandwidth considerations; which CUDA
  library does what.
- **Numerical considerations**: precision (FP32/FP64), stability, race conditions, atomics, determinism.
- **How we verify correctness**: the CPU reference, the tolerance, edge cases.
- **Where this sits in the real world**: how production tools (named in the catalog) do it differently.

Write `THEORY.md` as if explaining to a sharp student who knows C++ but is new to CUDA and new to the
domain. Diagrams in Mermaid/ASCII are welcome.

---

## 5. Build standard (CUDA + Visual Studio)

> **Toolchain ratification (2026-06-28).** This contract was originally written for CUDA 12.x + Visual
> Studio 2022. The owner's machine ships **CUDA Toolkit 13.3 + Visual Studio 2026 (Community, v145
> toolset)** instead, and the owner explicitly ratified adopting that installed toolchain as this repo's
> build standard. Both were verified working before scaffolding: `nvcc` compiles for the local GPU's
> `sm_75`, and the `CUDA 13.3` MSBuild integration is installed for VS 2026 (`v180\BuildCustomizations`).
> The bullets below reflect the ratified standard. (CUDA 13 dropped Maxwell/Pascal/Volta; `sm_75` Turing is
> the floor — which matches the arch list below.)

**Target toolchain (decided for this repo):**

- **Visual Studio 2026** (Community is fine; `v145` platform toolset) with the *Desktop development with
  C++* workload. _(Originally specified as VS 2022 / `v143`; ratified to VS 2026 above.)_
- **CUDA Toolkit 13.3** with the Visual Studio integration installed (the `CUDA 13.3.props/.targets`
  build customization). _(Originally specified as CUDA 12.x; ratified to 13.3 above.)_ Each `.vcxproj`
  imports `CUDA 13.3.props/.targets`; to retarget another CUDA version, change those two filenames.
- **Auto-detect, multi-architecture builds.** Every `.vcxproj` compiles for a fat set of real
  architectures so the binary runs on most machines, plus PTX for forward compatibility. Use:
  `code generation = compute_75,sm_75;compute_86,sm_86;compute_89,sm_89` and add
  `compute_89,compute_89` (PTX) as the last entry for JIT on newer cards. `BUILD_GUIDE.md` documents how to
  detect the local GPU's compute capability (`nvidia-smi`, or the `deviceQuery` sample) and narrow this
  list for faster local builds.
- **Configurations:** ship both `Debug|x64` and `Release|x64`. Release uses `-O3` host opt and `--use_fast_math`
  **only** where the project explicitly tolerates it (document it). Debug enables `-G` device debug and
  `-lineinfo` so the learner can step through kernels in Nsight.

**Every project's VS solution must:**

1. Build out-of-the-box on a clean machine that has VS 2026 + CUDA 13.3 (the ratified standard), with
   **no manual path edits**. Use the CUDA `.props`/`.targets` integration and `$(CUDA_PATH)`; never
   hardcode absolute paths.
2. Link only what it uses; if it uses cuBLAS/cuFFT/cuRAND/cuSOLVER/Thrust, add the library in the project
   and **comment in the `.vcxproj`** (yes, MSBuild XML supports `<!-- -->` comments) why each is linked.
3. Produce a single runnable `.exe` whose output matches `demo/expected_output.txt`.

**CPU reference path:** every project includes `reference_cpu.cpp` — a small, plain, heavily-commented CPU
implementation of the same computation. It exists for two reasons: (a) it is the teaching baseline that
makes the GPU speed-up legible, and (b) the demo runs both and asserts they agree within tolerance. Where a
project genuinely cannot run without a GPU, still provide the CPU reference for a reduced problem size.

**Optional `CMakeLists.txt`:** provide a CMake build too where it is low-cost (it helps Linux learners and
CI). The **VS solution is the required deliverable**; CMake is a bonus, never a substitute.

`docs/BUILD_GUIDE.md` is the canonical, screenshot-light, copy-paste-friendly guide for installing the
toolchain and building any project. Keep it current; if a build step changes, update it in the same push.

---

## 6. Commenting standard — the heart of this repository

> The owner asked for **"as much comment as possible, explaining what each function does, what each
> variable is for, what the logic and thought process is, how everything ties together."** Take this
> literally. Over-comment on purpose. The canonical, full rubric lives in `docs/COMMENTING_STANDARD.md`;
> this section is the binding summary.

### 6.1 Rules

1. **File header block** at the top of every source file: what this file is, its role in the project, the
   key idea, inputs/outputs, and a "read this after / before" pointer to other files. Reference the catalog
   ID so the file is traceable to the deep-dive entry.
2. **Every function** gets a doc-comment block: purpose, each parameter (units! ranges! ownership!), return
   value, side effects, complexity, and *why it exists*. For kernels, additionally document the **launch
   configuration** (grid/block dims and the reasoning), which memory spaces are touched, whether it uses
   atomics/shared memory, and the thread-to-data mapping (e.g., "thread `(bx,tx)` owns output element
   `i = bx*blockDim.x + tx`").
3. **Every non-trivial variable** gets an inline note on first use: what it represents, its units, and why
   it has the type/size it does. Especially flag indices, strides, padded sizes, and anything in device
   memory.
4. **Narrate the thought process.** Before a block of logic, write a short comment explaining the *intent*
   and the alternative you rejected ("We tile into shared memory here because the naive version re-reads
   global memory N times; see THEORY.md §GPU-mapping"). Comments should answer **why**, not just restate
   the code.
5. **Tie it together.** Where a function hands off to another, say so ("result feeds `backproject()` in
   kernels.cu"). Cross-reference README/THEORY sections by name.
6. **Explain library calls.** Any cuBLAS/cuFFT/Thrust/etc. call gets 2–4 lines: what it computes
   mathematically, why we use it instead of hand-rolling, and the shape/layout of its inputs/outputs.
7. **CUDA error checking is always visible and explained.** Wrap API calls in a `CUDA_CHECK(...)` macro
   (defined and commented once in `src/util/`), and comment what class of failure each guarded call can hit.
8. **No commented-out dead code.** Comments teach; they do not store graveyards. Delete dead code; explain
   decisions in prose.

### 6.2 Density target & illustrative example

Aim for a **comment-to-code ratio that a stranger could learn from** — in practice often **≥ 1:1** by line
in kernel files. `verify_project.py` enforces a floor (configurable, default ~0.4 non-trivial-comment lines
per code line in `src/`) — but the floor is a safety net, not the goal. The goal is comprehension.

A taste of the expected density (abbreviated):

```cpp
// ---------------------------------------------------------------------------
// kernels.cu  —  Tanimoto similarity over binary molecular fingerprints
// Project 1.12 (see ../THEORY.md and the catalog deep-dive for the "why").
//
// Big idea: a fingerprint is a fixed-length bit string. Tanimoto similarity of
// two fingerprints A,B is popcount(A & B) / popcount(A | B). We compare ONE
// query against MILLIONS of library fingerprints — each comparison is fully
// independent, so we give each library molecule its own GPU thread.
// ---------------------------------------------------------------------------

// Each fingerprint is FP_WORDS 64-bit words (e.g., 16 words = 1024 bits).
// We keep the query in CONSTANT memory: it is read by every thread but never
// changes during the launch, so constant memory's broadcast cache is ideal.
__constant__ uint64_t c_query[FP_WORDS];   // the single query fingerprint

// Kernel: one thread computes one Tanimoto score.
//   grid  : enough blocks to cover `n` library molecules
//   block : 256 threads (a good occupancy default on sm_75..sm_89)
//   thread (blockIdx.x, threadIdx.x) -> library molecule index `i`
__global__ void tanimoto(const uint64_t* __restrict__ lib, // [n * FP_WORDS] library fps, row-major
                         int n,                             // number of library molecules
                         float* __restrict__ out)           // [n] output similarities
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's molecule
    if (i >= n) return;                             // guard the ragged last block

    int inter = 0, uni = 0;   // running popcounts of (A&B) and (A|B)
    // Walk the fingerprint word by word; __popcll is the 64-bit population count
    // intrinsic (counts set bits) — one hardware instruction per word.
    #pragma unroll
    for (int w = 0; w < FP_WORDS; ++w) {
        uint64_t a = c_query[i? 0 : 0 ? 0 : w]; // (see THEORY: query indexed by word w)
        uint64_t b = lib[(size_t)i * FP_WORDS + w];
        inter += __popcll(a & b);   // bits set in BOTH -> intersection
        uni   += __popcll(a | b);   // bits set in EITHER -> union
    }
    out[i] = uni ? (float)inter / (float)uni : 0.0f; // avoid divide-by-zero
}
```

(The real file would fix the query indexing and expand the comments further — this is only a flavor.)

### 6.3 Demos

- **Per project:** `demo/run_demo.*` is a single command that builds if needed, runs on `data/sample/`, and
  prints a clearly-labeled result plus the GPU-vs-CPU agreement check and a timing line. `demo/README.md`
  annotates what the learner is seeing.
- **Top-level showcase:** `showcase/` is a project of its own that ties the collection together — at
  minimum a menu/CLI that lists domains and launches any project's demo, and a `SHOWCASE.md` tour. It is
  the "demo file showcasing everything" the owner asked for. Build it as a VS solution like any project.

---

## 7. Git & GitHub workflow

**Remote:** a **public** GitHub repository named **`cuda-cpp-healthcare-projects`** under the owner's
account. It is created once during bootstrap with the GitHub CLI (`gh repo create cuda-cpp-healthcare-projects
--public --source=. --remote=origin`). The owner is authenticated in their own environment; **never handle
their credentials or tokens** — if `gh` is not authenticated, stop and ask the owner to run `gh auth login`.

**Commit conventions:**

- Conventional-commit style: `feat(3.01): smith-waterman GPU kernel + demo`, `docs(build): …`,
  `chore(scaffold): …`, `fix(4.01): ramp filter off-by-one`.
- Small, coherent commits. One project's milestone per commit where practical.
- Never commit secrets, tokens, or large/raw datasets (see `.gitignore` and §8).

**Branching:** trunk-based on `main` is fine for a solo study repo. If running many parallel agents
(Ultracode), each agent works on a short-lived branch `proj/<ID>-<slug>` and opens a PR that the lead agent
fast-forward-merges after `verify_project.py` passes — this keeps concurrent work from colliding (each
branch only touches its own project folder; see §10).

### 7.1 Push-notes (REQUIRED on every push)

> The owner asked: **"every time something new gets pushed onto GitHub, create a .md explaining what was
> added."** This is mandatory and load-bearing.

For **every push to `origin/main`**, add a file under `push-notes/` named
`YYYY-MM-DD-NN-short-title.md` (NN = that day's push counter, zero-padded). Generate the stub with
`tools/new_pushnote.py`. Each push-note must contain:

1. **Summary** — one paragraph: what this push adds and why it matters to the learner.
2. **What changed** — the new/edited projects and files, grouped and linked (relative paths).
3. **For each new project** — a 3–5 sentence didactic blurb: the concept it teaches, the CUDA pattern, and
   the single most interesting thing to look at.
4. **How to build & run the new material** — exact commands.
5. **What to study here** — a suggested reading path and 1–2 exercises.
6. **Verification** — what was checked (build passed? demo matched expected? on what GPU/arch?).
7. **Known limitations / TODOs** — honest notes.
8. **Next push preview** — what is planned next.

Then prepend a one-line entry to root `CHANGELOG.md` linking the new push-note. The push-note is written
**before** the push and included **in** that push (so the repo always explains its own latest state).

---

## 8. Datasets, licensing & safety

**Handling (decided for this repo): download-scripts + tiny committed samples.**

- Commit only a **tiny, clearly-synthetic-or-public sample** under `data/sample/` so demos run offline.
- Put the real fetch in `scripts/download_data.*`: idempotent, documented, with the source URL, expected
  size, and a checksum. If a dataset needs credentials/registration (e.g., MIMIC, UK Biobank), the script
  must **not** attempt to bypass that — it prints instructions and links, and `make_synthetic.py` provides a
  stand-in so the project still runs.
- `data/README.md` records provenance, license, and per-field meaning. **Respect every license.** If a
  license forbids redistribution, do not commit the data — sample must be synthetic.
- `.gitignore` excludes downloaded/large data and build artifacts. If a genuinely necessary committed asset
  exceeds ~50 MB, use **Git LFS** and note it; prefer to avoid it.
- **Never fabricate results.** Synthetic data is fine and encouraged, but it must be labeled synthetic
  everywhere it appears, and demos must not imply clinical validity.

**Safety guardrails baked into the work:**

- Educational framing in every README; no diagnostic/therapeutic claims.
- No project should output anything presented as real medical advice.
- Keep patient-derived data out of the repo unless its license explicitly allows redistribution.

---

## 9. Definition of Done & verification gates

A project may be marked **done** (and pushed) only when **all** of these pass:

- [ ] Folder matches the §4 standard layout exactly (run `tools/verify_project.py <path>`).
- [ ] `README.md` has all required sections (§4.1); `THEORY.md` covers science→math→algorithm→GPU mapping.
- [ ] `src/` compiles via the VS solution in `Release|x64` **and** `Debug|x64` with zero errors and zero
      new warnings (treat warnings as defects to explain or fix).
- [ ] `reference_cpu.cpp` exists; the demo runs GPU + CPU and asserts agreement within documented tolerance.
- [ ] `demo/run_demo.*` runs on `data/sample/` and matches `demo/expected_output.txt`.
- [ ] Commenting passes the density floor and, more importantly, a human could learn from it (spot-read).
- [ ] `data/` sample present, `scripts/download_data.*` documented; licenses respected.
- [ ] `docs/STATUS.md` updated (project → done); a push-note written; committed and pushed.

**`tools/verify_project.py`** automates the structural checks (files present, README sections present,
comment-density heuristic, expected_output present). It prints a checklist with pass/fail. Do not mark a
task complete in the task list while any gate fails — keep it in_progress and write a push-note explaining
the blocker.

**CI (optional but recommended):** a GitHub Actions workflow that, on push, (a) runs `verify_project.py`
across all projects and (b) **compiles** changed CUDA projects with the toolkit installed. Note: GitHub's
hosted runners have **no NVIDIA GPU**, so CI can *compile* CUDA but cannot *run* kernels — running/demoing
is a **local** step. Document this clearly; do not let a green "build" badge imply the kernels were executed
in CI. A self-hosted GPU runner can be added later for true run-tests.

---

## 10. Multi-agent orchestration (Ultracode)

This repo is built by many agents working in parallel. The cardinal rule that makes parallelism safe:

> **One agent owns one project folder at a time. Agents never edit files outside their own
> `projects/<…>/` folder** (except via the lead — see below). This guarantees no two agents touch the same
> file, so concurrent work never conflicts.

**Roles:**

- **Lead/integrator (one):** owns all shared/root files — `CLAUDE.md`, `README.md`, `CHANGELOG.md`,
  `docs/`, `tools/`, `catalog.json`, `docs/STATUS.md`, `.gitignore`, CI, and the GitHub remote. The lead
  runs bootstrap (§11 Phase 0), assigns projects, merges branches, writes the push-note for each push, and
  pushes. Workers do **not** push to `main` directly.
- **Workers (many):** each claims one project from `docs/STATUS.md` (set it `in-progress` with the agent's
  name), builds it to the Definition of Done on a `proj/<ID>-<slug>` branch, runs `verify_project.py`, then
  hands back to the lead for merge. Then claims the next unclaimed project.

**Claiming protocol (prevents double-work):** `docs/STATUS.md` (generated from `catalog.json`) is the work
queue. A worker claims the lowest-priority-ranked `todo` item by editing only its own status row on its
branch; the lead resolves the rare claim race at merge time by priority/timestamp. Keep batches modest
(e.g., 8–16 workers in flight) so review and merges stay tractable.

**Integration checkpoints:** after each batch, the lead (a) merges all green branches, (b) runs the
full `verify_project.py` sweep + a build of changed projects, (c) writes one push-note covering the batch,
(d) updates `STATUS.md` and `CHANGELOG.md`, (e) pushes. A red project stays on its branch with a TODO note;
it is not merged until green.

**Consistency:** every worker follows this file and `docs/COMMENTING_STANDARD.md` verbatim, and starts from
`docs/PROJECT_TEMPLATE/`. Shared utility code (`src/util/` error-check + timing macros) is **copied** into
each project from the template, not symlinked, so each project stays self-contained and individually
buildable — a deliberate, documented duplication for didactic independence.

---

## 11. Rollout plan (phased)

**Phase 0 — Bootstrap (lead, once).** Scaffold the repo: write root files, `docs/`, `tools/`, run
`catalog.py` → `catalog.json`, run `scaffold.py` to stamp **all 301** project skeletons (each with a
catalog-prefilled README stub + TODO markers), generate `docs/STATUS.md`, init git, create the **public**
GitHub repo, first commit + push, and write `push-notes/<date>-00-bootstrap.md`.

**Phase 1 — Flagships (one polished project per domain, 14 total).** Build these *completely* first so the
owner has best-in-class study material in every domain quickly, and so the template/standards get
battle-tested before scaling. Suggested flagships (swap for a more tractable sibling in the same domain if
needed — prefer 🟢 Established with a clean demo):

| Domain | Suggested flagship |
|--------|--------------------|
| 01 Drug discovery | `1.12` Molecular fingerprint similarity search (Tanimoto) |
| 02 Structural biology | `2.06` Normal Mode Analysis / Elastic Network Model |
| 03 Genomics | `3.01` Smith-Waterman / Needleman-Wunsch alignment |
| 04 Medical imaging | `4.01` CT filtered backprojection (FDK) |
| 05 Radiation / med-phys | `5.01` Monte Carlo dose (simplified slab geometry) |
| 06 Physiology | `6.04` Lattice-Boltzmann blood/airflow solver |
| 07 Medical AI | `7.10` Physiological signal/waveform analysis (1-D conv on GPU) |
| 08 Neuroscience / BCI | `8.03` EEG/MEG spectral processing (cuFFT) |
| 09 Epidemiology | `9.02` Compartmental / metapopulation ODE ensembles |
| 10 Biomechanics | `10.02` Real-time soft-tissue deformation (mass-spring / PBD) |
| 11 Biotech / synbio | `11.09` Flow-cytometry clustering (GPU k-means) |
| 12 Omics | `12.01` Mass-spec proteomics spectral search |
| 13 Pharmacology | `13.02` PBPK at scale (ODE ensemble over virtual patients) |
| 14 Emerging frontiers | `14.02` Spatial reaction-diffusion (stencil) |

Push after each flagship (or in small batches) with a push-note. After Phase 1, reassess the standards and
update the template/docs if the flagships surfaced improvements.

**Phase 2 — Batched build-out (remaining ~287).** Work domain by domain, **easiest-first within a domain**
(🟢 → 🟡 → 🔴), many workers in parallel per §10. Each project to full Definition of Done. Push per batch
with a push-note. Keep `docs/STATUS.md` and `CHANGELOG.md` current.

**Phase 3 — Showcase & polish.** Build `showcase/` (the everything-demo), a top-level `README.md` index
with badges and a domain map, optional CI, and a final pass for cross-links and consistency. Final push +
summary push-note.

**Priority signal:** within a domain, rank `todo` by Difficulty (Beginner first) then catalog ID. This
front-loads quick didactic wins and defers the 🔴 frontier projects (which may legitimately ship as
"reduced-scope teaching versions" with the full version described in `THEORY.md`).

---

## 12. Conventions quick-reference

- **Language/standard:** CUDA C++ targeting C++17 host code; `.cu`/`.cuh` for device, `.cpp`/`.h` for host.
- **Style:** clear names over short ones; `snake_case` for functions/variables, `PascalCase` for types,
  `UPPER_CASE` for macros/constants; `d_`/`h_` prefixes for device/host pointers; document units in names
  or comments (`dt_seconds`, `dose_gray`).
- **Errors:** every CUDA API/kernel-launch checked via the commented `CUDA_CHECK` macro;
  `cudaGetLastError()` after launches.
- **Timing:** use CUDA events for kernel timing; print a clearly-labeled ms figure and (where meaningful) a
  GPU-vs-CPU speed-up — *as a teaching artifact, never a benchmark claim*.
- **Determinism:** prefer deterministic reductions in teaching code; if using atomics that reorder, say so
  and explain the float-summation caveat.
- **Markdown:** every project README/THEORY uses the same heading order; keep relative links working.
- **No black boxes, no fabricated data, no clinical claims, no committed secrets.** (Repeat of the load-
  bearing rules.)

## 13. When you (Claude Code) are unsure

- If the catalog entry is ambiguous, implement the **simplest correct teaching version**, and document the
  fuller version in `THEORY.md` under "Where this sits in the real world."
- If a build/tooling assumption fails on the owner's machine, **stop and ask** rather than guessing — and
  capture the fix in `docs/BUILD_GUIDE.md`.
- If a dataset cannot be obtained legally, **switch to synthetic** and label it.
- Never silently skip a Definition-of-Done gate. Surface it.

*End of contract. Build to teach.*


