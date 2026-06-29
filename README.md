# CUDA C++ for Health, Medicine & the Life Sciences

![projects](https://img.shields.io/badge/projects-301-blue) ![domains](https://img.shields.io/badge/domains-14-green) ![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900) ![VS](https://img.shields.io/badge/Visual%20Studio-2026%20(v145)-5C2D91) ![license](https://img.shields.io/badge/code-MIT-lightgrey)

A **didactic study collection** of ~**301 self-contained CUDA C++ projects** spanning drug discovery,
structural biology, genomics, medical imaging, radiation physics, physiology, medical AI, neuroscience,
epidemiology, biomechanics, biotechnology, omics, pharmacology, and emerging frontiers. Each project is
built to **teach**: every file is heavily commented, every project has a `THEORY.md`, a Visual Studio build,
a CPU reference, and a one-command demo that verifies the GPU result against the CPU.

> ⚠️ **Not for clinical use.** Everything here is **educational only**. No project output may be used for
> diagnosis, treatment, or any real medical decision. See [CLAUDE.md §8](CLAUDE.md) for data & safety rules.

## Why this exists

Teaching beats cleverness. A slower kernel a learner can follow is better than a fast one they can't. Every
artifact explains itself; nothing is a black box (library calls are explained, not hidden); and every
project is reproducible — clone, open the `.sln`, build, run the demo, see the documented result.

## Toolchain

Validated on **Windows 11 + NVIDIA RTX 2080 (sm_75)** with:

- **CUDA Toolkit 13.3** · **Visual Studio 2026** (Community, `v145` toolset)
- Multi-arch builds: `sm_75` (Turing) / `sm_86` (Ampere) / `sm_89` (Ada) + PTX for newer cards.

Full install + build instructions: **[docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md)**.

## Quick start

```powershell
# 1. Pick a project (e.g. the drug-discovery flagship) and open its solution:
#    projects/01-drug-discovery/1.12-molecular-fingerprint-similarity-search/build/<slug>.sln
#    -> set Release|x64 -> Build (Ctrl+Shift+B)

# 2. Run its demo (builds if needed, runs on the committed sample, verifies GPU==CPU):
cd projects/01-drug-discovery/1.12-molecular-fingerprint-similarity-search
./demo/run_demo.ps1
```

A green `PASS` means the GPU result matched the CPU reference within tolerance.

## How to use this repo as a learner

1. Read this README and **[docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md)**.
2. Open the **work-queue dashboard** — **[docs/STATUS.md](docs/STATUS.md)** — to see what's built (`done`)
   vs. scaffolded (`todo`).
3. Start with a **flagship** (one polished project per domain — see below). In each project, follow the
   `README.md` **"Code tour"**, then `THEORY.md` for the science → math → algorithm → GPU-mapping story.
4. Do the **Exercises** at the bottom of each project README.

### Flagships (start here — one per domain)

| Domain | Flagship |
|---|---|
| 01 Drug discovery | `1.12` Molecular fingerprint similarity (Tanimoto) |
| 02 Structural biology | `2.06` Normal Mode Analysis / Elastic Network Model |
| 03 Genomics | `3.01` Smith-Waterman / Needleman-Wunsch alignment |
| 04 Medical imaging | `4.01` CT filtered backprojection (FDK) |
| 05 Radiation / med-phys | `5.01` Monte Carlo dose (slab geometry) |
| 06 Physiology | `6.04` Lattice-Boltzmann blood/airflow solver |
| 07 Medical AI | `7.10` Physiological signal/waveform analysis (1-D conv) |
| 08 Neuroscience / BCI | `8.03` EEG/MEG spectral processing (cuFFT) |
| 09 Epidemiology | `9.02` Compartmental / metapopulation ODE ensembles |
| 10 Biomechanics | `10.02` Real-time soft-tissue deformation (mass-spring/PBD) |
| 11 Biotech / synbio | `11.09` Flow-cytometry clustering (GPU k-means) |
| 12 Omics | `12.01` Mass-spec proteomics spectral search |
| 13 Pharmacology | `13.02` PBPK at scale (ODE ensemble over virtual patients) |
| 14 Emerging frontiers | `14.02` Spatial reaction-diffusion (stencil) |

## Domain map

| # | Domain | Projects | Folder |
|---|---|---:|---|
| 1 | Drug Discovery & Molecular Design | 35 | [`projects/01-drug-discovery`](projects/01-drug-discovery) |
| 2 | Structural Biology & Protein Science | 35 | [`projects/02-structural-biology`](projects/02-structural-biology) |
| 3 | Genomics, Sequencing & Bioinformatics | 30 | [`projects/03-genomics`](projects/03-genomics) |
| 4 | Medical Imaging & Image Reconstruction | 33 | [`projects/04-medical-imaging`](projects/04-medical-imaging) |
| 5 | Radiation Therapy & Medical Physics | 15 | [`projects/05-radiation-therapy-medphys`](projects/05-radiation-therapy-medphys) |
| 6 | Computational Physiology & Systems Biology | 27 | [`projects/06-physiology-systems-biology`](projects/06-physiology-systems-biology) |
| 7 | Medical AI & Clinical Deep Learning | 19 | [`projects/07-medical-ai`](projects/07-medical-ai) |
| 8 | Neuroscience & Brain-Computer Interfaces | 16 | [`projects/08-neuroscience-bci`](projects/08-neuroscience-bci) |
| 9 | Epidemiology & Public Health | 10 | [`projects/09-epidemiology-public-health`](projects/09-epidemiology-public-health) |
| 10 | Biomechanics, Biomedical Devices & Surgery | 17 | [`projects/10-biomechanics-devices`](projects/10-biomechanics-devices) |
| 11 | Biotechnology, Bioprocess & Synthetic Biology | 12 | [`projects/11-biotech-synthbio`](projects/11-biotech-synthbio) |
| 12 | Analytical & Omics Data Processing | 17 | [`projects/12-omics-data-processing`](projects/12-omics-data-processing) |
| 13 | Pharmacology & Clinical Quantitative Modeling | 19 | [`projects/13-pharmacology-quant`](projects/13-pharmacology-quant) |
| 14 | Emerging, Theoretical & Grand-Challenge Frontiers | 16 | [`projects/14-emerging-frontiers`](projects/14-emerging-frontiers) |
| | **Total** | **301** | |

## Repository layout

```
cuda-cpp-healthcare-projects/
├── README.md                 ← you are here
├── CLAUDE.md                 ← the repository contract & working agreement
├── catalog.json              ← generated machine-readable catalog (301 records)
├── CUDA_CPP_Healthcare_Projects.xlsx          ← the catalog (source of truth)
├── CUDA_CPP_Healthcare_Projects_DeepDive.md   ← long-form prose reference
├── docs/
│   ├── BUILD_GUIDE.md        ← install toolchain, build any project
│   ├── COMMENTING_STANDARD.md← the (deliberately heavy) commenting rubric
│   ├── STATUS.md             ← generated work-queue dashboard
│   └── PROJECT_TEMPLATE/     ← the canonical empty project (copied per project)
├── tools/                    ← catalog.py, scaffold.py, verify_project.py, status.py, new_pushnote.py
├── push-notes/               ← one didactic note per push (what changed & why)
├── projects/                 ← 01-drug-discovery … 14-emerging-frontiers
└── showcase/                 ← top-level demo tying the collection together (Phase 3)
```

## The catalog & tooling

- **[`CUDA_CPP_Healthcare_Projects.xlsx`](CUDA_CPP_Healthcare_Projects.xlsx)** and
  **[`CUDA_CPP_Healthcare_Projects_DeepDive.md`](CUDA_CPP_Healthcare_Projects_DeepDive.md)** are the
  human-facing source of truth (one row / long-form entry per project).
- **`tools/catalog.py`** flattens both into **`catalog.json`** (consumed by all other tools).
- **`tools/scaffold.py`** stamps the 301 project skeletons from `docs/PROJECT_TEMPLATE/`.
- **`tools/verify_project.py`** checks a project against the Definition of Done (`CLAUDE.md §9`).
- **`tools/status.py`** regenerates `docs/STATUS.md`; **`tools/new_pushnote.py`** stamps push-notes.

## Contributing standards

Read **[CLAUDE.md](CLAUDE.md)** (the contract) and **[docs/COMMENTING_STANDARD.md](docs/COMMENTING_STANDARD.md)**.
Every project starts from `docs/PROJECT_TEMPLATE/`, must build clean in `Debug|x64` **and** `Release|x64`,
ship a CPU reference + a demo that matches `expected_output.txt`, and pass `verify_project.py`.

## License

Code is **MIT** ([LICENSE](LICENSE)). Datasets are governed by their own licenses (see each project's
`data/README.md`); nothing redistribution-restricted is committed. **Educational use only.**
