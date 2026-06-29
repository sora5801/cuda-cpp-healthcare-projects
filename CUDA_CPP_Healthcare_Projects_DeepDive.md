# CUDA C++ Projects for Health, Medicine & the Life Sciences — Deep-Dive Edition
An exhaustive, deeply-detailed catalog of **301 GPU-accelerated project ideas** across drug discovery, biotechnology, biomedical engineering, medicine, pharmacology, genomics, imaging, physiology, and beyond. Every entry includes a technical deep dive, the **specific algorithms** involved, real **datasets** (with access links), and **starter repositories/tools** to build on, plus the relevant CUDA libraries and GPU parallelization pattern.
> **Companion spreadsheet:** a sortable/filterable `.xlsx` version accompanies this document — one row per project with columns for domain, difficulty, maturity, algorithms, datasets, repos, and CUDA libraries. Use it to sort by difficulty or filter by domain.

## Legend
**Difficulty:** 🟢 Beginner-friendly · 🟡 Intermediate · 🔴 Advanced  
**Maturity:** *Established* (mature tooling) · *Active R&D* (research-grade, partial tooling) · *Frontier/Theoretical* (open problems where the GPU formulation itself is a contribution)
> **A note on links:** dataset and repository URLs were verified during research, but the field moves fast — a few may have moved. Entries marked *(verify URL)* were flagged as needing a quick confirmation. Always check the latest project page before committing to a tool.

## At a glance
- **Total projects:** 301
- **By difficulty:** 🟢 67 beginner · 🟡 206 intermediate · 🔴 28 advanced
- **By maturity:** 67 established · 206 active R&D · 28 frontier/theoretical

## Table of contents
1. **Drug Discovery & Molecular Design** — 35 projects
2. **Structural Biology & Protein Science** — 35 projects
3. **Genomics, Sequencing & Bioinformatics** — 30 projects
4. **Medical Imaging & Image Reconstruction** — 33 projects
5. **Radiation Therapy & Medical Physics** — 15 projects
6. **Computational Physiology & Systems Biology** — 27 projects
7. **Medical AI & Clinical Deep Learning** — 19 projects
8. **Neuroscience & Brain-Computer Interfaces** — 16 projects
9. **Epidemiology & Public Health** — 10 projects
10. **Biomechanics, Biomedical Devices & Surgery** — 17 projects
11. **Biotechnology, Bioprocess & Synthetic Biology** — 12 projects
12. **Analytical & Omics Data Processing** — 17 projects
13. **Pharmacology & Clinical Quantitative Modeling** — 19 projects
14. **Emerging, Theoretical & Grand-Challenge Frontiers** — 16 projects

---

## 1. Drug Discovery & Molecular Design

### 1.1 Molecular Dynamics Engine 🟢 · Established

- **Deep dive:** Classical MD simulates the time evolution of every atom in a biomolecular system by integrating Newton's equations of motion using empirical force fields (AMBER, CHARMM, GROMOS). Each timestep requires evaluating bonded interactions (bonds, angles, dihedrals) and non-bonded interactions (Lennard-Jones + electrostatics) for millions of atom pairs. GPUs accelerate the embarrassingly parallel pairwise force evaluation, reducing a day of CPU work to minutes on modern A100-class cards. The critical bottleneck — neighbor-list construction and PME reciprocal-space summation — maps cleanly onto CUDA threadblocks. Multi-GPU scaling via domain decomposition allows systems of 10–100 million atoms to be simulated in production.
- **Key algorithms:** Verlet/leapfrog integrator, LINCS/SHAKE bond constraint solvers, Particle-Mesh Ewald (PME) electrostatics, Lennard-Jones cutoff with long-range dispersion correction, Berendsen/Parrinello-Rahman barostat, velocity rescaling/Nosé-Hoover thermostat.
- **Datasets:** CHARMM36m force-field parameter set — comprehensive parameters for proteins, lipids, nucleic acids and carbohydrates (https://mackerell.umaryland.edu/charmm_ff.shtml); AMBER ff19SB — protein force field with improved backbone torsion potentials (https://ambermd.org); GPCRmd database — curated MD trajectories of GPCR proteins (https://gpcrmd.org); MoDEL — molecular dynamics extended library of protein simulations (https://mmb.irbbarcelona.org/MoDEL/).
- **Starter repos/tools:** GROMACS (https://github.com/gromacs/gromacs) — production-grade GPU-accelerated MD engine with CUDA/HIP/SYCL backends; OpenMM (https://github.com/openmm/openmm) — Python-scriptable MD toolkit with CUDA, OpenCL, and CPU platforms; NAMD (https://www.ks.uiuc.edu/Research/namd/) — scalable MD with multi-GPU support via CUDA; AMBER pmemd.cuda (https://ambermd.org/GPUSupport.php) — highly optimized GPU MD engine for AMBER force fields.
- **CUDA libraries & GPU pattern:** cuFFT for PME reciprocal sum, custom CUDA kernels for pairwise force evaluation, thrust for sorted neighbor list, NCCL for multi-GPU halo exchange; pattern is data-parallel threadblocks over atom pairs with shared-memory reductions.

---

### 1.2 Particle-Mesh Ewald Electrostatics 🟢 · Established

- **Deep dive:** Long-range electrostatics in periodic MD systems cannot be truncated without severe artifacts; PME splits the Coulomb sum into a short-range real-space part (evaluated with cutoff) and a smooth long-range reciprocal-space part evaluated on a 3D grid via FFT. The GPU acceleration opportunity is two-fold: the charge spreading (particle-to-mesh) and force interpolation (mesh-to-particle) steps are data-parallel over atoms, while the 3D FFT is handled by cuFFT. PME scales as O(N log N) and dominates walltime for large biological systems. Achieving double-precision accuracy at float throughput is the main engineering challenge.
- **Key algorithms:** Ewald summation, B-spline charge interpolation (order 4–6), 3D FFT on GPU, real-space erfc damping, smooth PME (SPME), Particle-Particle Particle-Mesh (P3M).
- **Datasets:** CHARMM-GUI solvation benchmark sets — pre-built periodic protein-water boxes (https://charmm-gui.org); D. E. Shaw Research Anton trajectories — ms-scale trajectory archives (available via DE Shaw); ion channel benchmark systems (MemProtMD, https://memprotmd.bioch.ox.ac.uk).
- **Starter repos/tools:** GROMACS CUDA PME (https://github.com/gromacs/gromacs) — reference GPU PME implementation; NAMD GPU PME (https://www.ks.uiuc.edu/Research/namd/) — tiled domain-decomposed PME; OpenMM PME plugin (https://github.com/openmm/openmm) — Python-accessible PME with mixed-precision; cuFFT (https://developer.nvidia.com/cufft) — NVIDIA's FFT library used internally by all above.
- **CUDA libraries & GPU pattern:** cuFFT for 3D FFT; custom CUDA kernels for B-spline charge spreading (atom-parallel) and gradient interpolation; shared-memory tiling to minimize global memory traffic; atomics for scatter-add accumulation on the charge grid.

---

### 1.3 Molecular Docking Engine 🟢 · Established

- **Deep dive:** Molecular docking predicts the preferred binding pose and score of a small molecule within a protein binding pocket by sampling ligand conformations (translations, rotations, torsions) and scoring each with an empirical or knowledge-based energy function. The scoring function evaluation for each sampled pose is independent, creating massive data parallelism — thousands of poses per ligand, millions of ligands per campaign. AutoDock-GPU achieves >1000× speedup over single-CPU AutoDock4 by running the Lamarckian genetic algorithm (LGA) in parallel across GPU threads, each evaluating a distinct pose. The bottleneck is the grid-based force-field energy lookup, which benefits from GPU texture-cache acceleration.
- **Key algorithms:** Lamarckian Genetic Algorithm (LGA), gradient-based local search (BFGS), grid-based energy evaluation (electrostatics + vdW precalculated on 3D grids), scoring functions (AutoDock4, Vina, Vinardo, AD4).
- **Datasets:** DUD-E — directory of useful decoys enhanced, 102 targets with actives and decoys (https://dude.docking.org); ChEMBL — bioactivity database with >2M compounds (https://www.ebi.ac.uk/chembl/); PDB-bind — curated protein-ligand complexes with binding affinities (http://www.pdbbind.org.cn); CASF benchmark — comparative assessment of scoring functions (http://www.pdbbind.org.cn/casf.php).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — CUDA/OpenCL GPU docking with LGA parallelism; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — GPU-accelerated batch docking with >2000× speedup on V100; GNINA (https://github.com/gnina/gnina) — CNN-scored docking fork of smina; Vina-GPU 2.1 (https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1) — GPU-accelerated AutoDock Vina with RILC-BFGS.
- **CUDA libraries & GPU pattern:** Texture memory for 3D grid lookups, CUDA threadblocks each running one GA individual per ligand, warp-level reduction for fitness evaluation; grid-strided loops for pose batch processing.

---

### 1.4 Ultra-Large Virtual Screening 🟢 · Established

- **Deep dive:** Modern make-on-demand chemical libraries (Enamine REAL: >6 billion compounds, ZINC: ~2 billion) make exhaustive docking computationally prohibitive with CPU resources. GPU-accelerated docking allows screening of billions of compounds by batching thousands of ligands simultaneously on a single GPU. Additionally, ML surrogate models trained on docked subsets (active learning / Bayesian optimization) dramatically reduce the number of full docking evaluations required. Specialized tools like HASTEN and REINVENT combine GPU docking with ML to achieve 90% recall of true top-1000 hits after evaluating only 1% of the library. The Summit supercomputer campaign against COVID-19 targets docked >1 billion compounds using AutoDock-GPU.
- **Key algorithms:** GPU-batched LGA/BFGS docking, Bayesian active learning, surrogate-model filtering (random forest, GNN), pharmacophore pre-filtering, shape screening pre-filter, Lipinski/ADMET filter cascades.
- **Datasets:** Enamine REAL library — >6B synthesizable compounds (https://enamine.net/compound-collections/real-compounds); ZINC20 — free virtual screening database (https://zinc20.docking.org); ChEMBL — bioactivity reference (https://www.ebi.ac.uk/chembl/); ExCAPE-DB — large-scale public chemogenomics dataset (https://solr.ideaconsult.net/search/excape/).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — core CUDA docking engine used in billion-compound campaigns; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — high-throughput GPU docking with batch input; DiffDock (https://github.com/gcorso/DiffDock) — diffusion model for blind docking of large libraries; gpusimilarity (https://github.com/schrodinger/gpusimilarity) — GPU fingerprint similarity for rapid pre-screening.
- **CUDA libraries & GPU pattern:** Texture memory for grid lookups; warp-parallel GA evaluation; multiple ligands co-resident in GPU memory; NVLink multi-GPU for campaign-scale throughput; thrust for top-K selection.

---

### 1.5 Free Energy Perturbation / Thermodynamic Integration 🟢 · Established

- **Deep dive:** FEP and TI compute binding free energy differences (ΔΔG) between two ligands by running MD along an alchemical λ-pathway that slowly transforms one molecule into another. Each λ-window requires independent GPU MD trajectories; the collection of windows is trivially parallel across GPUs. The critical computational cost is the length and number of λ-windows required for convergence (typically 12–24 windows × 2–5 ns each). GPU-accelerated pmemd.cuda and NAMD-FEP achieve >10× speedup over CPU, reducing multi-day calculations to hours on a single A100. Relative FEP (RBFE) is now a standard tool in lead optimization pipelines at major pharmaceutical companies.
- **Key algorithms:** Alchemical λ-coupling, soft-core potentials (Beutler/Zacharias), multi-state Bennett acceptance ratio (MBAR), thermodynamic integration quadrature, replica exchange with solute tempering (REST2), overlap matrix assessment.
- **Datasets:** Merck FEP benchmark set — 8 targets with experimental ΔΔG (available via OpenFE; https://github.com/OpenFreeEnergy/openfe); FEP+ validation set (Schrodinger, verify URL); PDB-bind — experimental binding affinities (http://www.pdbbind.org.cn); ChEMBL activity data for target families (https://www.ebi.ac.uk/chembl/).
- **Starter repos/tools:** OpenFE (https://github.com/OpenFreeEnergy/openfe) — open FEP toolkit supporting GROMACS and OpenMM backends; GROMACS FEP (https://github.com/gromacs/gromacs) — GPU-accelerated FEP with MBAR post-processing via alchemlyb; OpenMMTools (https://github.com/choderalab/openmmtools) — alchemical replica exchange on GPU via OpenMM; AMBER pmemd.cuda TI (https://ambermd.org/GPUSupport.php) — softcore TI on NVIDIA GPUs.
- **CUDA libraries & GPU pattern:** Full MD engine on GPU (cuFFT PME + custom force kernels); embarrassingly parallel λ-window array across multiple GPUs; NCCL for REMD communication; CPU post-processing via alchemlyb/MBAR.

---

### 1.6 Enhanced Sampling — Metadynamics & Replica Exchange 🟢 · Established

- **Deep dive:** Standard MD cannot cross large free energy barriers on accessible timescales. Enhanced sampling methods accelerate conformational exploration by adding history-dependent bias potentials (metadynamics) or by running multiple copies at different temperatures/Hamiltonians (REMD/HREX). PLUMED plugs into GROMACS, NAMD, OpenMM, and LAMMPS to implement CVs and bias on the fly. GPU MD trajectories feed the bias engine with minimal overhead. Well-tempered metadynamics ensures convergence of the free energy surface (FES) and is widely used for drug binding pathway elucidation. GPU-MetaD (2025) achieves full-lifecycle GPU acceleration for ML potential metadynamics with systems up to 1.3M atoms.
- **Key algorithms:** Well-tempered metadynamics, funnel metadynamics, Hamiltonian replica exchange (HREX), temperature REMD (T-REMD), replica exchange with solute tempering (REST2), collective variable (CV) on-the-fly evaluation, free energy surface estimation via reweighting.
- **Datasets:** PLUMED-NEST — repository of published metadynamics/enhanced sampling input files (https://www.plumed-nest.org); GPCRmd trajectory archive (https://gpcrmd.org); D. E. Shaw millisecond MD datasets (available via RCSB); benchmark FES for alanine dipeptide / chignolin (commonly used test systems).
- **Starter repos/tools:** PLUMED (https://github.com/plumed/plumed2) — plugin for collective variables and enhanced sampling, GPU-compatible via host MD engine; GROMACS + PLUMED (https://github.com/gromacs/gromacs) — standard GPU metadynamics stack; OpenPathSampling (https://github.com/openpathsampling/openpathsampling) — transition path sampling framework; HTMD (https://github.com/Acellera/htmd) — high-throughput MD with adaptive sampling on GPU clusters.
- **CUDA libraries & GPU pattern:** Bias potential evaluation on CPU via PLUMED with negligible overhead; full MD force/integration on GPU; multi-walker metadynamics uses MPI + NCCL across GPUs; GPU kernels for on-the-fly CV computation in GPU-MetaD.

---

### 1.7 Quantum Chemistry / DFT 🟢 · Established

- **Deep dive:** Density Functional Theory (DFT) calculates electronic structure by solving the Kohn-Sham equations self-consistently on a basis set (plane waves or Gaussians). The dominant cost is the construction of the Fock/Kohn-Sham matrix via electron repulsion integrals (ERIs) — an O(N^4) bottleneck that GPUs reduce substantially by computing integrals in batches. TeraChem pioneered GPU-accelerated DFT and can achieve 100× speedup over single-CPU codes. Applications in drug discovery include geometry optimization of drug fragments, calculation of electrostatic potential maps for pharmacophore generation, and QM-derived force field parameterization.
- **Key algorithms:** Kohn-Sham SCF, B3LYP/ωB97X-D exchange-correlation functionals, resolution-of-identity (RI) approximation for ERIs, DIIS convergence acceleration, plane-wave pseudopotential (PW-PP), linear-scaling DFT.
- **Datasets:** QM9 — DFT-computed properties of 134k organic molecules (https://doi.org/10.6084/m9.figshare.978904); ANI-1ccx — CCSD(T)-level energies for diverse organic molecules (https://github.com/isayev/ANI1ccx_dataset); PubChemQC — DFT calculations for ~3M PubChem molecules (http://pubchemqc.riken.jp); CSD — Cambridge Structural Database for crystal structures (https://www.ccdc.cam.ac.uk).
- **Starter repos/tools:** TeraChem (https://www.petachem.com) — GPU-native DFT, commercial but widely cited; PySCF (https://github.com/pyscf/pyscf) — pure Python quantum chemistry with GPU4PySCF extension; CP2K (https://github.com/cp2k/cp2k) — GPU-accelerated mixed Gaussian/plane-wave DFT; NWChem (https://github.com/nwchemgit/nwchem) — parallel quantum chemistry with GPU-accelerated modules.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for ERI computation (two-electron integrals in shared memory); cuBLAS for matrix diagonalization; cuFFT for plane-wave FFT; warp-level parallelism over shell pairs.

---

### 1.8 Semi-Empirical & Tight-Binding Quantum Methods 🟢 · Established

- **Deep dive:** Semi-empirical methods (PM7, GFN2-xTB) approximate quantum mechanics at 100–10000× lower cost than DFT by parameterizing integral expressions with empirical data. They bridge the gap between force fields and full DFT, enabling geometry optimization and reactivity screening of drug-like molecules at scale. GPU implementations parallelize the sparse Hamiltonian construction and diagonalization over molecule batches — thousands of small molecules can be optimized simultaneously on one GPU. XTB is critical for conformer ranking, tautomer enumeration, and QM-based ADMET calculation in modern drug discovery pipelines.
- **Key algorithms:** MNDO/AM1/PM6/PM7 Hamiltonians, GFN1/GFN2-xTB (extended tight-binding), DFTB+ (density functional tight binding), diagonalization via cuSOLVER, GPU-batched molecular calculations.
- **Datasets:** ANI-1 — 20M DFT energy calculations on 57k molecules (https://github.com/isayev/ANI1); QM9 (https://doi.org/10.6084/m9.figshare.978904); GMTKN55 — benchmark thermochemistry and kinetics set (https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn); COMPAS — computational database of polycyclic aromatic systems (verify URL).
- **Starter repos/tools:** xtb (https://github.com/grimme-lab/xtb) — GFN2-xTB reference implementation (CPU-only but used as GPU backend reference); DFTB+ (https://github.com/dftbplus/dftbplus) — GPU-accelerated DFTB via ELPA library; GFN-FF / xTB-IFF (https://github.com/grimme-lab) — force field from tight binding; TBLite (https://github.com/tblite/tblite) — lightweight tight-binding library.
- **CUDA libraries & GPU pattern:** cuSOLVER for batch matrix diagonalization; cuBLAS for Hamiltonian density-matrix products; custom CUDA kernels for two-center integral batches; stream concurrency to overlap compute and data transfer for molecule batches.

---

### 1.9 ML Interatomic Potentials (Neural Network Potentials) 🟢 · Established

- **Deep dive:** Neural network potentials (NNPs) learn the potential energy surface from ab initio data, reproducing DFT accuracy at near-classical MD speed. Architectures range from atom-centered symmetry functions (ANI) to equivariant message-passing networks (NequIP, MACE, SchNet). GPU acceleration is essential: each forward pass involves neighborhood construction, message passing over all atomic pairs within a cutoff, and backpropagation for forces. On an A100, a 500-atom protein+ligand system runs at ~10 ns/day — 1000× slower than classical FF but 100× faster than DFT, enabling reactive drug-target simulations previously impossible.
- **Key algorithms:** Atom-centered symmetry functions (ACSF/BEHLER), equivariant neural networks (E(3)-equivariant / SE(3)), message-passing neural networks (MPNN/SchNet/DimeNet), MACE (multi-ACE), NequIP, neural achitecture via PyTorch Geometric.
- **Datasets:** ANI-1ccx — CCSD(T) energies on 500k conformers of drug-like molecules (https://github.com/isayev/ANI1ccx_dataset); SPICE — quantum chemistry dataset for ML potentials covering drug-like molecules and proteins (https://github.com/openmm/spice-dataset); rMD17 — revised MD17 benchmark (https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038); OE62 — 62k organic molecules with DFT energetics (verify URL).
- **Starter repos/tools:** TorchANI (https://github.com/aiqm/torchani) — PyTorch ANI NNP with CUDA acceleration and OpenMM integration; TorchMD-Net (https://github.com/torchmd/torchmd-net) — equivariant NNPs with GPU-optimized neighbor list; MACE (https://github.com/ACEsuit/mace) — fast equivariant NNP with GPU kernels; NequIP (https://github.com/mir-group/nequip) — E(3)-equivariant network for accurate NNPs.
- **CUDA libraries & GPU pattern:** PyTorch CUDA autograd for force computation via backpropagation; custom CUDA kernels for neighbor list construction with periodic boundaries; torch.compile/TorchScript for inference optimization; multi-GPU via DDP for training.

---

### 1.10 De Novo Generative Molecular Design 🟡 · Active R&D

- **Deep dive:** Generative models learn the distribution of drug-like molecules and sample novel structures optimized for multiple properties (potency, selectivity, ADMET, synthesizability). GPU training is mandatory: large transformer/RNN/diffusion models over SMILES strings or 3D molecular graphs require days on multi-GPU nodes. At inference, reinforcement learning (RL) fine-tuning generates thousands of candidate molecules per GPU-second, enabling goal-directed optimization. REINVENT4 combines RL with curriculum learning on SMILES; diffusion-based methods (DiffSBDD, TargetDiff) generate molecules directly in 3D protein binding pockets.
- **Key algorithms:** Variational autoencoders (VAE), transformer language models on SMILES/SELFIES, graph generative models, denoising diffusion probabilistic models (DDPM), reinforcement learning with REINFORCE/PPO, scoring functions (docking, QED, SA score).
- **Datasets:** ChEMBL — 2M+ bioactive molecules (https://www.ebi.ac.uk/chembl/); ZINC20 — 1.4B purchasable compounds (https://zinc20.docking.org); GuacaMol benchmark — distribution learning and goal-directed generation benchmarks (https://github.com/BenevolentAI/guacamol); MOSES — molecular generation benchmarks (https://github.com/molecularsets/moses).
- **Starter repos/tools:** REINVENT4 (https://github.com/MolecularAI/REINVENT4) — production SMILES generative model with RL, Apache 2.0 license; DiffSBDD (https://github.com/arneschneuing/DiffSBDD) — 3D structure-based diffusion design; DiffDock (https://github.com/gcorso/DiffDock) — diffusion model for pose generation used in SBDD pipelines; DeepChem (https://github.com/deepchem/deepchem) — broad ML drug discovery toolkit including generative models.
- **CUDA libraries & GPU pattern:** cuDNN for transformer/RNN layers; custom CUDA scatter/gather for molecular graph message passing; multi-GPU DDP training; FP16 mixed precision via torch.amp; GPU-batched scoring function evaluation during RL rollouts.

---

### 1.11 QSAR / Property Prediction 🟢 · Established

- **Deep dive:** Quantitative structure-activity relationship (QSAR) models predict biological activity from molecular descriptors or learned representations. Modern approaches use message-passing neural networks (MPNNs) over molecular graphs, enabling GPU-batched training on millions of labeled datapoints. The bottleneck shifts from feature computation to batch normalization and message aggregation over irregular graph structures — handled by PyTorch Geometric or DGL with CUDA backends. GPU-accelerated QSAR models at pharmaceutical companies screen hundreds of millions of virtual compounds per hour for ADMET and activity filters.
- **Key algorithms:** Directed message-passing (D-MPNN / Chemprop), graph convolutional networks (GCN), graph attention networks (GAT), transformer on molecular graphs (Uni-Mol), random forest / XGBoost on Morgan fingerprints, uncertainty quantification (ensemble, MCDropout).
- **Datasets:** MoleculeNet — curated ML benchmark for 17+ molecular datasets (https://moleculenet.org); ChEMBL bioactivity data (https://www.ebi.ac.uk/chembl/); TDC (Therapeutics Data Commons) — 66 tasks for drug discovery ML (https://tdcommons.ai); PCBA (PubChem BioAssay) — 128 bioassays on 440k compounds (https://moleculenet.org).
- **Starter repos/tools:** Chemprop (https://github.com/chemprop/chemprop) — D-MPNN for molecular property prediction, GPU training; Uni-Mol (https://github.com/deepmodeling/Uni-Mol) — 3D molecular transformer pre-trained on 209M conformers; DeepChem (https://github.com/deepchem/deepchem) — broad GPU-accelerated ML chemistry toolkit; DGL-LifeSci (https://github.com/awslabs/dgl-lifesci) — graph neural networks for life science on GPU.
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA sparse tensor ops for graph batching; cuDNN for feedforward layers; FP16 mixed precision; GPU-accelerated descriptor generation via RDKit CUDA extensions (verify URL).

---

### 1.12 Molecular Fingerprint Similarity Search 🟢 · Established

- **Deep dive:** Tanimoto similarity between Morgan/ECFP bit-vectors is the standard metric for chemical similarity searching. Brute-force comparison of a query against a library of 100M compounds requires 10^10 bit-AND/popcount operations — ideally suited for GPU SIMD. Each 2048-bit fingerprint fits in 32 uint64 words; a GPU thread evaluates one query-vs-library pair in ~5 ns. Schrodinger's gpusimilarity loads an entire library into GPU memory and achieves sub-second retrieval on billion-compound libraries. The GPU pattern is embarrassingly data-parallel with a final topK reduction.
- **Key algorithms:** Tanimoto coefficient (Jaccard on bit-vectors), Morgan/ECFP fingerprints (radius 2–3), TopK reduction, LSH-based approximate search, Faiss IVF for high-dimensional vectors.
- **Datasets:** ChEMBL (https://www.ebi.ac.uk/chembl/); ZINC20 (https://zinc20.docking.org); PubChem Compound — 115M+ compounds (https://pubchem.ncbi.nlm.nih.gov); Enamine REAL (https://enamine.net).
- **Starter repos/tools:** gpusimilarity (https://github.com/schrodinger/gpusimilarity) — CUDA/Thrust brute-force fingerprint search; FPSim2 (https://github.com/chembl/FPSim2) — fast similarity search using PyTables and GPU-accelerated popcount; RDKit (https://github.com/rdkit/rdkit) — cheminformatics toolkit with Morgan fingerprint generation; Faiss (https://github.com/facebookresearch/faiss) — GPU-accelerated ANN search applicable to molecular embeddings.
- **CUDA libraries & GPU pattern:** Thrust device_vector for library storage; custom CUDA kernels with __popcll() for bit-count; warp-shuffle reduction for partial Tanimoto sums; GPU topK using cub::DeviceRadixSort; texture memory for fingerprint cache.

---

### 1.13 Pharmacophore & 3D Shape Screening 🟢 · Established

- **Deep dive:** Pharmacophore and shape-based screening compares 3D query features (hydrogen bond donors/acceptors, hydrophobic regions, ionizable groups, molecular shape) against library conformers, capturing complementarity not encoded in 2D fingerprints. ROCS (OpenEye) uses a volumetric Gaussian overlap function (ShapeTanimoto + ColorTanimoto) that is differentiable and GPU-friendly. Screening billions of conformers requires GPU-parallel overlap computation across independent molecule pairs. This is a key pre-filtering step before docking in virtual screening pipelines.
- **Key algorithms:** Gaussian volume overlap (Tversky/Tanimoto), Fast Overlay of Chemical Structures (FOCS), pharmacophore feature matching (HBD/HBA/hydrophobic/aromatic), conformer ensemble generation, rigid body alignment (quaternion-based).
- **Datasets:** ZINC20 conformer libraries (https://zinc20.docking.org); DUD-E (https://dude.docking.org); Enamine REAL conformer sets (https://enamine.net); Directory of Useful Decoys-Enhanced including 3D conformers (verify URL).
- **Starter repos/tools:** ROCS (OpenEye/Cadence) — commercial GPU 3D shape screening (https://www.eyesopen.com/rocs); Open3DQSAR (https://open3dqsar.sourceforge.io) — open 3D-QSAR tool; RDKit shape tools (https://github.com/rdkit/rdkit) — open Gaussian overlap via PyTorch extension (verify URL); Pharmer (https://github.com/dkoes/pharmer) — open pharmacophore search tool.
- **CUDA libraries & GPU pattern:** Warp-parallel Gaussian overlap evaluation over conformer pairs; texture memory for pre-computed atom volumes; GPU-batched rigid-body alignment using quaternion representation; cuBLAS for rotation matrix applications.

---

### 1.14 Conformer Ensemble Generation 🟢 · Established

- **Deep dive:** Drug-like molecules are flexible; binding-relevant conformers must be generated before 3D screening or docking. RDKit ETKDG embeds molecules in 3D using experimental torsion knowledge (ETKDGv3) and distance geometry; generation of thousands of conformers per molecule for a library of millions is a CPU bottleneck. GPU acceleration is achieved by batching conformer embedding across many molecules simultaneously. Alternatively, ML-based conformer predictors (OMEGA-ML, GeoMol, TorsionalDiffusion) use GPU neural networks trained on crystallographic torsion distributions.
- **Key algorithms:** Experimental torsion-angle knowledge distance geometry (ETKDG), MMFF94/UFF energy minimization, breadth-first conformer pruning (RMSD clustering), torsional diffusion (ML), graph neural network conformer prediction.
- **Datasets:** GEOM — 37M conformers of drug-like molecules with DFT energies (https://github.com/learningmatter-mit/geom); CSD torsion library (https://www.ccdc.cam.ac.uk); COD (Crystallography Open Database) — crystal structures for torsion validation (https://www.crystallography.net); PDB small molecule conformations (https://www.rcsb.org).
- **Starter repos/tools:** RDKit ETKDG (https://github.com/rdkit/rdkit) — standard conformer engine, GPU-batched via RDKit-GPU (verify URL); TorsionalDiffusion (https://github.com/gcorso/torsional-diffusion) — GPU diffusion model for conformer sampling; GeoMol (https://github.com/PattanaikL/GeoMol) — ML conformer prediction; Frog2 / OMEGA (OpenEye, commercial) — fast conformer generators.
- **CUDA libraries & GPU pattern:** Batched SVD/distance geometry on GPU via cuSOLVER; custom CUDA kernels for pairwise RMSD computation; GPU-parallel MMFF energy minimization via molecular gradient descent; PyTorch-based diffusion inference with CUDA tensors.

---

### 1.15 Protein-Ligand Binding Affinity Scoring (ML) 🟡 · Active R&D

- **Deep dive:** End-to-end ML scoring functions learn protein-ligand interaction energy surrogates directly from structural data, bypassing physics-based force fields. Models range from 3D-CNNs over voxelized complexes to equivariant GNNs over atom graphs to transformer co-folding models (NeuralPLexer3). GPU inference enables rapid rescoring of millions of docking poses in virtual screening — a 3D-CNN scores a pose in ~1 ms on a GPU vs. >1 s for FEP. The fundamental challenge is generalization across chemical space and protein families.
- **Key algorithms:** 3D-CNN on atomic density grids, equivariant graph neural networks (SchNet/DimeNet++), attention-based protein-ligand co-attention, diffusion-based co-folding (NeuralPLexer), Random Forest on PLEC/ECIF interaction fingerprints.
- **Datasets:** PDB-bind v2020 — 19,443 protein-ligand complexes with Kd/Ki (http://www.pdbbind.org.cn); CASF-2016 benchmark (http://www.pdbbind.org.cn/casf.php); ChEMBL activity data (https://www.ebi.ac.uk/chembl/); BindingDB — 2.8M measured binding affinities (https://www.bindingdb.org).
- **Starter repos/tools:** NeuralPLexer (https://github.com/zrqiao/NeuralPLexer) — state-specific co-folding with binding affinity, requires CUDA; GNINA (https://github.com/gnina/gnina) — CNN rescoring in docking pipeline; DiffDock (https://github.com/gcorso/DiffDock) — generative docking with affinity proxy; DeepChem (https://github.com/deepchem/deepchem) — includes AtomicConvolutions and MPNN-based scoring.
- **CUDA libraries & GPU pattern:** cuDNN for 3D-CNN layers; PyTorch Geometric CUDA kernels for equivariant message passing; FP16 mixed precision for throughput; GPU-parallel batch scoring for post-docking rescoring of millions of poses.

---

### 1.16 ADMET / Toxicity Prediction 🟢 · Established

- **Deep dive:** Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties gate entry into clinical trials; predicting them computationally early in discovery eliminates costly failures. GPU-trained GNN/MPNN models (Chemprop-based) can screen 100M virtual compounds for ADMET in hours. The ADMET-AI platform (2024) uses a Chemprop-RDKit ensemble achieving best-in-class speed. Multi-task learning on heterogeneous assay data (LogP, hERG, Caco-2, microsomal clearance, Ames mutagenicity) benefits from GPU parallelism across tasks and molecules simultaneously.
- **Key algorithms:** Directed message-passing (D-MPNN), multi-task learning, uncertainty quantification (conformal prediction, evidential learning), Tox21 endpoint models, quantum-chemical descriptor augmentation.
- **Datasets:** Tox21 — 12 toxicity endpoints, 8k compounds (https://tripod.nih.gov/tox21/); TDC ADMET benchmark group (https://tdcommons.ai/benchmark/admet_group/overview/); ClinTox — FDA-approved and failed drugs (https://moleculenet.org); DILI (drug-induced liver injury) databases (verify URL).
- **Starter repos/tools:** Chemprop (https://github.com/chemprop/chemprop) — D-MPNN backbone for ADMET models; ADMET-AI (https://github.com/swansonk14/admet_ai) — GPU-accelerated ADMET platform; DeepChem (https://github.com/deepchem/deepchem) — includes Tox21 models; pkCSM (https://biosig.lab.uq.edu.au/pkcsm/) — web server using graph signatures (verify GPU support).
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA sparse ops; cuDNN for feedforward/attention layers; multi-task GPU loss aggregation; FP16 training; batched RDKit fingerprint generation.

---

### 1.17 Markov State Models from MD 🟡 · Active R&D

- **Deep dive:** Markov State Models (MSMs) discretize MD conformational space into metastable states and estimate transition probabilities from long or many-short trajectories. Building MSMs requires: (1) featurization of millions of MD frames, (2) dimensionality reduction (tICA/PCA), (3) clustering (k-means/mini-batch k-means), and (4) transition matrix estimation. Steps 1–3 are GPU-acceleratable via cuML or custom CUDA kernels. The payoff is extraction of thermodynamics and kinetics (kon, koff, binding pathways) from aggregated μs-ms of GPU MD.
- **Key algorithms:** Time-lagged independent component analysis (tICA), mini-batch k-means clustering, transition matrix MLE/Bayesian, PCCA+ for state coarse-graining, Chapman-Kolmogorov test, variational approach to Markov processes (VAMP).
- **Datasets:** MDCATH — 5 μs MD trajectories for 272 proteins (https://huggingface.co/datasets/compsciencelab/mdcath); Fast-folder benchmark trajectories (chignolin, Trp-cage, Villin — publicly shared by Piana/Shaw); GPCRmd (https://gpcrmd.org); D. E. Shaw millisecond trajectories (accessible via RCSB deposition).
- **Starter repos/tools:** PyEMMA (https://github.com/markovmodel/PyEMMA) — MSM construction with CUDA-accelerated k-means; MSMBuilder (https://github.com/msmbuilder/msmbuilder) — statistical models for biomolecular dynamics; deeptime (https://github.com/deeptime-ml/deeptime) — VAMPnets and modern MSM tools on GPU; cuML (https://github.com/rapidsai/cuml) — GPU-accelerated k-means and PCA via RAPIDS.
- **CUDA libraries & GPU pattern:** cuML k-means for GPU clustering; custom CUDA kernels for pairwise RMSD featurization; cuBLAS for tICA covariance matrix; GPU-parallel trajectory loading via RAPIDS cuDF.

---

### 1.18 Fragment / Combinatorial Library Enumeration 🟡 · Active R&D

- **Deep dive:** Fragment-based drug discovery and combinatorial library design require enumerating billions of reaction products from building blocks in silico. A single Enamine REAL-like library contains >6B compounds from ~160 reactions and >130k building blocks. GPU acceleration is applied to (i) SMILES enumeration via GPU-parallel reaction SMARTS matching, (ii) property calculation (MW, cLogP, TPSA) for billions of products, and (iii) diversity filtering via GPU fingerprint clustering. The V-Synthes/SyntheMol approach uses GPU ML to navigate the synthetic graph without explicit enumeration.
- **Key algorithms:** SMARTS reaction transforms, combinatorial product SMILES generation, Lipinski/Veber/PAINS filtering, GPU-parallel property calculation, GPU k-means diversity filtering, synthon-based virtual library navigation.
- **Datasets:** Enamine building block catalog (https://enamine.net/building-blocks); REAL Space library (https://enamine.net); ChemSpace — commercial building blocks (https://chem-space.com); Sigma-Aldrich building block list (verify URL).
- **Starter repos/tools:** RDKit (https://github.com/rdkit/rdkit) — reaction SMARTS and virtual library tools; SyntheMol (https://github.com/swansonk14/SyntheMol) — GPU ML combinatorial synthesis design; ASKCOS (https://github.com/ASKCOS/ASKCOS) — reaction condition prediction platform; SpaceLight/FastROCS (OpenEye, commercial) — GPU shape screening of virtual libraries.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for parallel SMARTS matching over SMILES bytes; GPU-parallel Morgan fingerprint computation; cuML GPU k-means for diversity selection; Thrust for parallel filtering and compaction.

---

### 1.19 Network / Polypharmacology Modeling 🟡 · Active R&D

- **Deep dive:** Polypharmacology recognizes that drugs interact with multiple targets, creating complex biological networks. GPU-accelerated graph neural networks on drug-target interaction (DTI) networks, protein-protein interaction (PPI) networks, and disease-gene networks enable systems-level prediction of off-target effects, drug combinations, and drug repurposing. Large-scale heterogeneous graph training (heterogeneous GNN, knowledge graph embeddings) with millions of nodes requires GPU memory and compute. GPU-parallel network perturbation simulations assist target identification.
- **Key algorithms:** Heterogeneous graph neural networks, knowledge graph embedding (TransE, RotatE), drug-target interaction prediction (DeepDTA, GraphDTA), network diffusion, community detection, drug combination synergy prediction (DeepSynergy).
- **Datasets:** STRING PPI network — 11.8B protein interactions (https://string-db.org); DrugBank — FDA-approved drugs and targets (https://go.drugbank.com); STITCH — drug-protein interactions (http://stitch.embl.de); DrugComb — drug combination synergy data (https://drugcomb.fimm.fi).
- **Starter repos/tools:** PyTorch Geometric (https://github.com/pyg-team/pytorch_geometric) — GPU heterogeneous graph learning; DGL (https://github.com/dmlc/dgl) — GPU graph learning for DTI networks; DeepPurpose (https://github.com/kexinhuang12345/DeepPurpose) — drug-target interaction prediction toolkit; HeteroMed (verify URL) — heterogeneous medical knowledge graphs.
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA sparse tensor operations for heterogeneous GNN; cuSPARSE for adjacency matrix products; GPU-batched negative sampling; FP16 embedding tables for large entity sets.

---

### 1.20 Reaction Yield / Retrosynthesis Scoring 🟡 · Active R&D

- **Deep dive:** Computational retrosynthesis decomposes a target molecule into commercially available building blocks via known reactions, enabling synthesizability assessment in generative design. GPU-trained transformer models (Molecular Transformer, Chemformer) predict reaction products and retrosynthetic routes over large training sets of reaction SMILES. GPU inference scores millions of candidate synthetic routes per second. Reaction yield prediction (using GNN or transformer on reaction SMILES + conditions) guides experimental prioritization. Integration with generative models creates closed-loop synthesis-aware drug design.
- **Key algorithms:** Transformer on augmented SMILES (reaction SMILES), sequence-to-sequence models, Monte Carlo tree search (MCTS) for retrosynthesis planning, graph-to-graph transformations, reaction center prediction, graph neural network on reaction graphs.
- **Datasets:** USPTO-50k — 50k atom-mapped reactions (https://github.com/connorcoley/rexgen_direct); Reaxys/CAS reaction databases (commercial); Open Reaction Database (ORD) — open-access reaction data (https://open-reaction-database.org); USPTO-MIT — 479k reactions (https://github.com/wengong-jin/nips17-rexgen).
- **Starter repos/tools:** Molecular Transformer (https://github.com/pschwllr/MolecularTransformer) — GPU transformer for reaction prediction; AiZynthFinder (https://github.com/MolecularAI/aizynthfinder) — GPU-accelerated retrosynthesis planning; ASKCOS (https://github.com/ASKCOS/ASKCOS) — synthesis planning platform; Chemformer (https://github.com/MolecularAI/Chemformer) — pre-trained BART-based reaction model.
- **CUDA libraries & GPU pattern:** cuDNN transformer attention kernels; FP16 mixed precision for large SMILES vocabularies; GPU-batched beam search decoding; MCTS rollouts in parallel on GPU with batched transformer scoring.

---

### 1.21 Polarizable / AMOEBA Force Field MD 🟡 · Active R&D

- **Deep dive:** Classical fixed-charge force fields miss polarization effects crucial for accurate binding free energies and ionic interactions. The AMOEBA force field includes point multipoles (up to quadrupoles) and induced dipoles solved self-consistently at each MD step via an iterative solver (conjugate gradient). This increases cost ~10× over AMBER but GPU implementation in Tinker-HP achieves >200-fold speedup over single-CPU, making microsecond AMOEBA simulations of large proteins feasible. Applications include protein-ligand FEP with AMOEBA and pKa prediction in complex electrostatic environments.
- **Key algorithms:** Induced dipole iteration (conjugate gradient), Ewald summation for multipoles (PME-multipole), AMOEBA water model, HIPPO force field, PIMD with polarizable FF.
- **Datasets:** AMOEBA protein force field parameter files (https://github.com/TinkerTools/tinker); WaterMap/hydration site datasets (Schrodinger, verify URL); BindingDB experimental affinities (https://www.bindingdb.org); NIST thermophysical properties (https://webbook.nist.gov).
- **Starter repos/tools:** Tinker-HP (https://github.com/TinkerTools/tinker-hp) — massively parallel GPU AMOEBA MD; OpenMM AMOEBA plugin (https://github.com/openmm/openmm) — AMOEBA on CUDA; Tinker9 (https://github.com/TinkerTools/tinker9) — GPU-native Tinker rewrite; AMOEBA+ FF parameters (https://github.com/TinkerTools/poltype2).
- **CUDA libraries & GPU pattern:** Custom CUDA conjugate-gradient solver for induced dipoles; cuFFT for multipole PME; warp-synchronous reduction for energy accumulation; multi-GPU via MPI domain decomposition with NCCL.

---

### 1.22 Constant-pH Molecular Dynamics 🟡 · Active R&D

- **Deep dive:** Biomolecular simulations normally fix protonation states, ignoring pH-dependent conformational changes critical for drug design (e.g., histidine flips, aspartate protonation near binding sites). Continuous constant-pH MD (CpHMD) in AMBER22 pmemd.cuda couples proton titration MC moves to GPU MD, sampling both conformation and protonation simultaneously. A 400-residue protein at single-pH takes ~1 hour on an RTX 2080 — >1000× faster than CPU. Applications include pKa prediction, pH-dependent drug binding, and ion channel gating.
- **Key algorithms:** Continuous CpH titration (GB or PME-explicit solvent), Metropolis MC protonation moves, replica exchange CpHMD (REX-CpHMD), free energy estimation of pKa shifts, AMBER ff14SB/ff19SB titration parameters.
- **Datasets:** pKa databases: PKAD (https://compbio.clemson.edu/pkad/), PHMD reference pKa values; Benchmark pKa sets for Asp/Glu/His/Cys/Lys residues; DrugBank compounds with ionizable groups (https://go.drugbank.com).
- **Starter repos/tools:** AMBER pmemd.cuda CpHMD (https://ambermd.org/GPUSupport.php) — GPU constant-pH MD; CHARMM CpHMD (https://www.charmm.org) — GBSW implicit solvent titration; OpenMM constant-pH (https://github.com/openmm/openmm) — Python CpH framework; PropKa (https://github.com/jensengroup/propka) — fast pKa prediction for system setup.
- **CUDA libraries & GPU pattern:** Full MD on GPU (pmemd.cuda); MC protonation moves evaluated via energy difference on GPU; replica exchange across pH replicas using NCCL/MPI; trajectory analysis on GPU via cuML.

---

### 1.23 QM/MM Molecular Dynamics 🟡 · Active R&D

- **Deep dive:** Hybrid quantum mechanics/molecular mechanics (QM/MM) partitions a system into a reactive QM region (drug + key residues, 50–200 atoms) treated at DFT/semi-empirical level and a larger MM region. GPU acceleration applies to both the QM Hamiltonian (via TeraChem/GPU-DFT) and the MM dynamics (via AMBER/GROMACS). The critical bottleneck is the QM/MM electrostatic coupling and QM Hamiltonian evaluation at every MD step. Open-source GPU QM/MM is available via AMBER+QUICK (GPU-accelerated DFT engine). Applications include enzyme catalysis mechanism, covalent drug reactivity, and proton transfer pathways.
- **Key algorithms:** ONIOM/link-atom QM/MM coupling, electrostatic embedding, DFT-based QM region (B3LYP/PBE), GFN2-xTB semi-empirical QM, AIMD in QM region with Verlet MM, adaptive QM/MM for large reactive systems.
- **Datasets:** QM/MM benchmark from SAMPL challenges (verify URL); enzyme reaction databases (BRENDA, https://www.brenda-enzymes.org); crystal structures of enzyme-drug complexes from PDB (https://www.rcsb.org); RCSB ligand validation data (https://www.rcsb.org).
- **Starter repos/tools:** AMBER+QUICK (https://github.com/merzlab/QUICK) — GPU-accelerated DFT for QM/MM with AMBER; TeraChem-TCPB (https://www.petachem.com) — GPU DFT server for QM/MM with NAMD/AMBER; OpenMM+PySCF QM/MM (https://github.com/openmm/openmm) — Python QM/MM interface; cp2k (https://github.com/cp2k/cp2k) — GPU-accelerated QM/MM for periodic systems.
- **CUDA libraries & GPU pattern:** GPU ERI computation for QM Hamiltonian via TeraChem/QUICK CUDA kernels; MM region on GPU (pmemd.cuda); asynchronous GPU-CPU communication for QM/MM coupling; CUDA streams for overlapping QM and MM compute.

---

### 1.24 Umbrella Sampling / WHAM Free Energy Profiles 🟢 · Established

- **Deep dive:** Umbrella sampling applies harmonic restraints along a reaction coordinate (e.g., ligand unbinding distance, pore radius) to force sampling at energy barriers. Multiple windows run simultaneously (embarrassingly parallel across GPUs), each an independent GPU MD simulation. WHAM (Weighted Histogram Analysis Method) or MBAR post-processes window histograms into a potential of mean force (PMF). GPU MD enables each window to generate nanoseconds of biased trajectory in minutes, enabling convergence that was previously impractical. Applications include permeation barriers in ion channels and drug binding/unbinding free energy profiles.
- **Key algorithms:** Harmonic bias potentials, WHAM self-consistent iteration, MBAR (multistate BAR), steered MD + Jarzynski equality, metadynamics PMF (alternative), local elevation/flooding.
- **Datasets:** Ion channel permeation benchmark sets; SAMPL binding free energy challenges (https://github.com/samplchallenges/SAMPL); BindingDB (https://www.bindingdb.org); GROMACS umbrella sampling tutorials (https://tutorials.gromacs.org).
- **Starter repos/tools:** GROMACS gmx wham (https://github.com/gromacs/gromacs) — built-in WHAM post-processing; OpenMM umbrella sampling (https://github.com/openmm/openmm-cookbook) — Python harmonic restraints; alchemlyb (https://github.com/alchemistry/alchemlyb) — MBAR/WHAM post-processing; PLUMED (https://github.com/plumed/plumed2) — collective variables + restraints.
- **CUDA libraries & GPU pattern:** Full MD per window on GPU; MPI + NCCL to launch window array; WHAM iteration on CPU via numpy; GPU-parallel histogram accumulation using atomicAdd; shared-memory reductions for collective variable forces.

---

### 1.25 Gaussian-Accelerated MD (GaMD) 🟡 · Active R&D

- **Deep dive:** GaMD adds a Gaussian-distributed boost potential to the total potential energy without predefined collective variables, enabling unconstrained enhanced sampling. Implemented in AMBER16+ GPU (pmemd.cuda) and NAMD, GaMD can reveal drug binding pathways, allosteric mechanisms, and protein folding on simulation timescales of microseconds. Unlike steered MD, no reaction coordinate is needed — GaMD monitors the system's total potential and boosts when it falls below a threshold. The boost potential statistics are used for free energy reweighting via cumulant expansion.
- **Key algorithms:** Gaussian boost potential with variance threshold, dual-boost GaMD (dihedral + total), free energy reweighting via cumulant expansion to 2nd order, principal component analysis of boosted trajectories, ligand GaMD (LiGaMD).
- **Datasets:** AMBER GaMD tutorials (https://www.med.unc.edu/pharm/miaolab/resources/gamd/); GPCRmd (https://gpcrmd.org); D. E. Shaw Research benchmark systems; PDB structures of drug targets (https://www.rcsb.org).
- **Starter repos/tools:** AMBER pmemd.cuda GaMD (https://ambermd.org) — reference GPU GaMD implementation; NAMD GaMD (https://www.ks.uiuc.edu/Research/namd/) — GaMD in NAMD for GPU simulations; GaMD analysis scripts (https://github.com/MiaoLab20/GaMD) — post-processing and reweighting tools; OpenMM GaMD plugin (verify URL).
- **CUDA libraries & GPU pattern:** Full GPU MD with real-time boost potential evaluation; CUDA kernels for on-the-fly potential monitoring and bias application; memory-efficient running statistics for Gaussian parameters; multi-GPU replica runs.

---

### 1.26 Steered Molecular Dynamics (SMD) 🟡 · Active R&D

- **Deep dive:** SMD applies external forces or velocity constraints to pull a molecule along a predefined coordinate (e.g., unbinding a ligand from a pocket), enabling calculation of work profiles and estimation of free energies via Jarzynski's equality. GPU MD allows many independent SMD trajectories to be run simultaneously, improving statistical convergence of Jarzynski estimates. Applications include estimation of drug residence time, rupture force of protein-ligand bonds, and domain opening mechanisms. NAMD pioneered GPU SMD; OpenMM provides Python-scriptable SMD via external forces.
- **Key algorithms:** Constant-velocity SMD (harmonic spring), constant-force SMD, Jarzynski equality for ΔG, fluctuation theorems, non-equilibrium work analysis, umbrella integration (follow-up).
- **Datasets:** NAMD SMD tutorials (https://www.ks.uiuc.edu/Training/Tutorials/); BindingDB residence time data (https://www.bindingdb.org); PDB force-probe simulation benchmark cases; published SMD studies on ion channels and motor proteins.
- **Starter repos/tools:** NAMD (https://www.ks.uiuc.edu/Research/namd/) — production GPU SMD; GROMACS pull code (https://github.com/gromacs/gromacs) — GPU SMD via pull-coord; OpenMM CustomExternalForce (https://github.com/openmm/openmm) — Python SMD; alchemlyb (https://github.com/alchemistry/alchemlyb) — Jarzynski post-processing.
- **CUDA libraries & GPU pattern:** Full GPU MD; custom CUDA force kernel for harmonic spring SMD; CUDA streams for multiple independent pulling trajectories; GPU memory for storing work accumulated along path.

---

### 1.27 MM-GBSA / MM-PBSA Rescoring 🟢 · Established

- **Deep dive:** MM-GB(PB)SA computes binding free energies as the MM interaction energy plus solvation free energy (implicit solvent GB or PB), minus entropic terms, from snapshots along an MD trajectory. It is the standard high-throughput rescoring step after docking, offering >10× better accuracy than scoring functions with ~1000× less cost than FEP. GPU-accelerated MD (pmemd.cuda) generates the required trajectory snapshots rapidly; gmx_MMPBSA post-processes GROMACS trajectories. The solvation GB/PB solvers can also be GPU-accelerated.
- **Key algorithms:** Molecular mechanics energy decomposition, Generalized Born (GB) implicit solvent, Poisson-Boltzmann (PB) numerical solver, normal-mode / quasi-harmonic entropy estimation, interaction entropy method, per-residue energy decomposition.
- **Datasets:** PDB-bind (http://www.pdbbind.org.cn); CASF-2016 (http://www.pdbbind.org.cn/casf.php); ChEMBL activity data (https://www.ebi.ac.uk/chembl/); AMBER MM-GBSA tutorial datasets (https://ambermd.org/tutorials/).
- **Starter repos/tools:** AMBER MMPBSA.py (https://ambermd.org/AmberTools.php) — reference MM-GBSA/PBSA implementation; gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — GROMACS compatibility layer; NAMD MMPBSA (https://www.ks.uiuc.edu/Research/namd/) — NAMD-based MM-PBSA; OpenMM MMGBSA (verify URL) — Python MM-GBSA workflow.
- **CUDA libraries & GPU pattern:** GPU MD for trajectory generation (pmemd.cuda); CPU MMPBSA.py for post-processing (GPU PB solver possible via custom CUDA); GPU-parallel evaluation of snapshots via embarrassingly parallel CUDA stream array.

---

### 1.28 Covalent Docking 🟡 · Active R&D

- **Deep dive:** Covalent inhibitors form a permanent or semi-permanent bond with a nucleophilic residue (usually Cys, Ser, Lys, Tyr). Docking them requires two-stage sampling: (1) non-covalent pre-reaction pose generation (as in standard docking) and (2) covalent bond geometry enforcement with post-reaction scoring. GPU acceleration helps explore the expanded conformational space after covalent bond formation. Methods include CovDock (Schrodinger), AutoDock-GPU covalent option, and emerging DL methods (CovDocker, 2025). EGFR/BTK/KRAS(G12C) covalent drug programs drive industrial interest.
- **Key algorithms:** Two-stage covalent docking protocol, warhead reactive group enumeration, covalent bond geometry constraint, MM-GBSA rescoring of covalent complexes, covalent pharmacophore matching.
- **Datasets:** CovDocker benchmark (2025, verify URL); ChEMBL covalent inhibitor set (https://www.ebi.ac.uk/chembl/); PDB covalent complex structures (https://www.rcsb.org); BindingDB covalent entries (https://www.bindingdb.org).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — supports covalent docking mode; GNINA (https://github.com/gnina/gnina) — CNN-scored docking with covalent options; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — GPU docking extendable to covalent; CovDocker (arxiv 2506.21085, verify GitHub URL) — DL covalent docking benchmark.
- **CUDA libraries & GPU pattern:** Same as standard docking GPU pattern; additional CUDA kernel for covalent bond constraint penalty; GPU-parallel conformational sampling of warhead + linker degrees of freedom.

---

### 1.29 Kinase Selectivity Panel Scoring 🟡 · Active R&D

- **Deep dive:** Kinases share highly similar binding pockets, making selectivity a central challenge in kinase drug discovery. A GPU MD + ML pipeline can score a compound across 500+ kinase structures simultaneously: (1) GPU-parallel docking against all kinase homology models, (2) ML scoring using kinase-specific fingerprints (KLIFS features), (3) MM-GBSA rescoring. GPU acceleration allows a compound to be profiled against the entire kinome in minutes rather than days. Selectivity fingerprinting using ensemble docking with GPU makes this tractable.
- **Key algorithms:** Ensemble docking, kinase-ligand interaction fingerprints (KLIFS/IFP), selectivity scoring (SFP), homology model generation, structural kinome alignment, ML kinase activity prediction (KinaseML).
- **Datasets:** KLIFS — kinase-ligand interaction fingerprinting database (https://klifs.net); KinomeScan — 468-kinase selectivity data (verify URL); ChEMBL kinase activity data (https://www.ebi.ac.uk/chembl/); DTC drug-target commons kinase panel (https://dtcommons.ai).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — GPU docking against kinase panels; KLIFS Python API (https://github.com/volkamerlab/kissim) — kinase structural fingerprints; KinoML (https://github.com/openkinome/kinoml) — ML for kinase drug discovery; HTMD (https://github.com/Acellera/htmd) — GPU-based kinome docking workflows.
- **CUDA libraries & GPU pattern:** GPU-parallel docking against kinase model array; GPU-batched IFP featurization; cuML for kinase activity ML training; Thrust for topK selectivity ranking.

---

### 1.30 Trajectory RMSD, Clustering & Contact Analysis 🟢 · Established

- **Deep dive:** Post-MD analysis of multi-microsecond trajectories generates terabytes of coordinate data requiring GPU-accelerated analytics. RMSD calculation requires aligning every frame to a reference (Kabsch algorithm: SVD of 3×3 matrices — trivially parallelized over frames). Pairwise RMSD for clustering requires O(N²) comparisons of millions of frames. H-bond network analysis and contact map generation are similarly parallelizable. MDTraj and cuML enable GPU-accelerated trajectory analysis with RAPIDS. The bottleneck is I/O bandwidth from trajectory files.
- **Key algorithms:** Kabsch RMSD algorithm (SVD), GROMOS/DBSCAN/k-medoids clustering, contact map calculation (distance cutoff), H-bond donor-acceptor angle+distance criteria, radial distribution function (RDF), NMR order parameter S².
- **Datasets:** MDCATH trajectory dataset (https://huggingface.co/datasets/compsciencelab/mdcath); PDB trajectory depositions; GPCRmd (https://gpcrmd.org); MDDB (https://www.mddbr.eu) — molecular dynamics database.
- **Starter repos/tools:** MDTraj (https://github.com/mdtraj/mdtraj) — GPU-accelerated RMSD and trajectory analysis; RAPIDS cuML (https://github.com/rapidsai/cuml) — GPU clustering for MSM construction; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — trajectory analysis with GPU support; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated adaptive MD analysis.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for batched 3×3 SVD (Kabsch rotation); GPU pairwise distance matrix via cuBLAS (outer product formulation); atomic contact map via GPU distance thresholding; RAPIDS cuDF for trajectory frame I/O.

---

### 1.31 Solvent-Accessible Surface Area (SASA) on GPU 🟢 · Established

- **Deep dive:** SASA measures the protein or ligand surface area accessible to solvent, used in solvation energy estimation, buried surface analysis for protein-protein interfaces, and GB implicit solvent models. Shrake-Rupley algorithm uses a grid of test points per atom — embarrassingly parallel over atoms. GPU implementation computes neighbor lists, tests points for burial, and accumulates per-atom SASA in parallel. GPU-SASA is critical in MM-GBSA workflows where SASA must be evaluated for every trajectory snapshot.
- **Key algorithms:** Shrake-Rupley point grid algorithm, LCPO (linear combination of pairwise overlaps), numerical surface integration, buried SASA for interface area, hydrophobic patch detection.
- **Datasets:** PDB protein structures (https://www.rcsb.org); ASA benchmark set for validation (verify URL); MD trajectory ensembles for SASA time series.
- **Starter repos/tools:** FreeSASA (https://github.com/mittinatten/freesasa) — fast open SASA library; AMBER SASA via pmemd (https://ambermd.org); MDTraj SASA (https://github.com/mdtraj/mdtraj) — GPU-friendly SASA computation; Biopython SASA (https://github.com/biopython/biopython) — Python SASA utilities.
- **CUDA libraries & GPU pattern:** CUDA threadblocks over atoms; shared memory for neighbor coordinates; warp reduction for test-point counting; GPU-parallel shrake-rupley with texture memory for neighbor lookup.

---

### 1.32 Alchemical Hydration Free Energy (ΔGsolv) 🟡 · Active R&D

- **Deep dive:** Absolute solvation free energies (ΔGhyd for water, ΔGsolv for organic solvents) are foundational to ADMET modeling (LogP, LogS, membrane permeability). Alchemical calculation via thermodynamic integration or FEP decouples solute-solvent interactions over λ-windows, yielding ΔGsolv directly from GPU MD simulations. Compared to QSAR models, GPU alchemical ΔGsolv achieves sub-kcal/mol accuracy on drug-like molecules. FreeSolv benchmark provides 643 experimental hydration free energies for validation.
- **Key algorithms:** Alchemical decoupling (electrostatics then LJ), soft-core potentials, MBAR/TI post-processing, absolute binding free energy (ABFE), double decoupling, Bennet acceptance ratio (BAR).
- **Datasets:** FreeSolv — 643 experimental hydration free energies (https://github.com/MobleyLab/FreeSolv); MNSol — Minnesota solvation database (https://comp.chem.umn.edu/mnsol/); SAMPL hydration challenges (https://github.com/samplchallenges/SAMPL); NIST ThermoML hydration data (https://trc.nist.gov).
- **Starter repos/tools:** OpenFE (https://github.com/OpenFreeEnergy/openfe) — open alchemical FE toolkit; alchemtest (https://github.com/alchemistry/alchemtest) — test systems for alchemical codes; GROMACS + alchemlyb (https://github.com/gromacs/gromacs) — GPU FEP pipeline; AMBER FEP (https://ambermd.org) — pmemd.cuda alchemical decoupling.
- **CUDA libraries & GPU pattern:** Full GPU MD (cuFFT PME + custom force kernels); parallel λ-window MD across GPU array; MBAR post-processing via pymbar; GPU evaluation of soft-core potential perturbations.

---

### 1.33 Interaction Fingerprinting & Binding-Mode Clustering 🟡 · Active R&D

- **Deep dive:** Protein-ligand interaction fingerprints (IFPs) encode which residues form HBs, hydrophobic contacts, π-stacking, salt bridges, and halogen bonds with a ligand. IFPs enable rapid clustering of thousands of docking poses or MD trajectory frames into distinct binding modes, analogous to chemical fingerprints but for structural biology. GPU-parallel distance/angle evaluation over millions of frame-residue pairs makes real-time IFP generation from MD trajectories feasible. Applications include binding-mode prediction validation and SAR-IFP correlation for lead optimization.
- **Key algorithms:** PLEC (protein-ligand extended connectivity), PLIF (protein-ligand interaction fingerprint), SIFt (structural interaction fingerprint), Tanimoto IFP similarity, GPU-parallel distance/angle kernels, GPU k-means clustering on IFP bit-vectors.
- **Datasets:** PDB-bind complex structures (http://www.pdbbind.org.cn); KLIFS (https://klifs.net); ChEMBL bioactivity with structures (https://www.ebi.ac.uk/chembl/); BindingDB (https://www.bindingdb.org).
- **Starter repos/tools:** ProLIF (https://github.com/chemosim-lab/ProLIF) — protein-ligand interaction fingerprints from MD trajectories; ODDT (https://github.com/oddt/oddt) — open drug discovery toolkit with IFP; Pharmit (https://pharmit.csb.pitt.edu) — pharmacophore + shape screening; KLIFS Python (https://github.com/volkamerlab/kissim) — kinase IFP features.
- **CUDA libraries & GPU pattern:** CUDA kernels for atom-pair distance/angle evaluation over frame×residue grid; GPU popcount for IFP Tanimoto; cuML GPU k-means on IFP matrix; RAPIDS cuDF for MD frame I/O and selection.

---

### 1.34 Amyloid / Aggregation Propensity Prediction 🟡 · Active R&D

- **Deep dive:** Protein aggregation drives diseases (Alzheimer's, Parkinson's, ALS) and is a major liability in biologic drug development. GPU-accelerated coarse-grained and atomistic MD can directly simulate fibril nucleation and extension, but requires microsecond-to-millisecond timescales accessible only with GPU enhanced sampling. ML aggregation predictors (AGGRESCAN3D, CamSol) train on experimental aggregation rates; GPU-trained GNNs on protein sequence+structure outperform sequence-only models. Amyloid fibril cryo-EM structures from EMDB drive validation.
- **Key algorithms:** β-aggregation propensity scoring, coarse-grained MD of oligomerization (MARTINI), REMD/MetaD of early aggregation, GNN aggregation predictor, solubility prediction neural networks.
- **Datasets:** AmyPro — curated amyloidogenic sequence database (https://amypro.net); FoldAmyloid prediction database (verify URL); ThT fluorescence assay aggregation kinetics datasets; EMDB fibril EM maps (https://www.ebi.ac.uk/emdb/).
- **Starter repos/tools:** AGGRESCAN3D server (https://biocomp.chem.uw.edu.pl/A3D2/) — structure-based aggregation prediction; CamSol (https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolmethod) — solubility prediction; WALTZ-DB 2.0 (verify URL) — aggregation kinetics; GROMACS+PLUMED fibril simulation stack (https://github.com/gromacs/gromacs).
- **CUDA libraries & GPU pattern:** GPU MARTINI CG-MD for large oligomerization systems; metadynamics enhanced sampling via PLUMED on GPU; GPU-trained GNN inference for sequence-based aggregation; CUDA-accelerated contact map tracking during aggregation.

---

### 1.35 QMMM/ML Potential Hybrid MD 🔴 · Frontier/Theoretical

- **Deep dive:** The next frontier beyond QM/MM is using ML potentials trained on QM data to replace the expensive QM region — enabling microsecond reactive MD at QM accuracy. GPU-accelerated equivariant NNPs (MACE, NequIP) can serve as drop-in QM replacements in an MM environment. This hybrid NNP/MM approach runs fully on GPU: the NNP forward pass and MM evaluation occur in overlapping CUDA streams. Challenges include training data coverage for reactive intermediates and accurate long-range electrostatics across the QM-ML/MM boundary.
- **Key algorithms:** NNP/MM coupling, link-atom boundary treatment, active learning for reactive system NNP training, δ-ML correction to DFT, equivariant NNP with long-range electrostatic correction.
- **Datasets:** ANI-1ccx reactive extensions (verify URL); DFT reaction pathway datasets from QM/MM studies; Transition1x — 10M DFT calculations along reaction paths (https://zenodo.org/record/5781475); SPICE dataset (https://github.com/openmm/spice-dataset).
- **Starter repos/tools:** TorchMD-Net (https://github.com/torchmd/torchmd-net) — equivariant NNP with MM coupling; MACE (https://github.com/ACEsuit/mace) — fast NNP for hybrid ML/MM; OpenMM-ML (https://github.com/openmm/openmm-ml) — NNP/MM interface for OpenMM; NNPOps (https://github.com/openmm/NNPOps) — CUDA-optimized NNP primitives.
- **CUDA libraries & GPU pattern:** CUDA MACE kernels for equivariant message passing; OpenMM CUDA platform for MM region; CUDA streams for async NNP+MM; PyTorch autograd for NNP force gradients; cuBLAS for spherical harmonic transforms.

---

---

## 2. Structural Biology & Protein Science

### 2.1 Protein Structure Prediction Inference (AlphaFold-class) 🟢 · Established

- **Deep dive:** AlphaFold2 and its successors (RoseTTAFold, ESMFold, OpenFold, Boltz-1, AlphaFold3) predict atomic-resolution 3D protein structures from amino acid sequences using deep learning. The Evoformer stack processes multiple sequence alignments (MSAs) and pair representations through stacked self-attention and triangle-multiplicative update layers — each requiring enormous GPU memory (an A100 40GB handles ~5000 residues for AF2). GPU inference is mandatory: predicting a 500-residue protein takes ~5 minutes on GPU vs. ~12 hours on CPU. ESMFold bypasses MSA entirely, using a 15B-parameter language model for 10–60× faster prediction.
- **Key algorithms:** Evoformer (MSA row/column attention + triangle updates), Structure Module (invariant point attention, IPA), recycling iterations, template attention, diffusion-based structure generation (AF3), confidence scoring (pLDDT, PAE).
- **Datasets:** AlphaFold Database — 200M+ predicted structures (https://alphafold.ebi.ac.uk/); RCSB PDB — 227k+ experimental structures (https://www.rcsb.org); UniProt/UniRef90 MSA databases (https://www.uniprot.org); CAMEO/CASP15 structure prediction benchmarks (https://www.cameo3d.org).
- **Starter repos/tools:** AlphaFold2 (https://github.com/google-deepmind/alphafold) — official DeepMind implementation; OpenFold (https://github.com/aqlaboratory/openfold) — trainable GPU-friendly PyTorch AF2; ESMFold (https://github.com/facebookresearch/esm) — MSA-free language model structure prediction; Boltz-1 (https://github.com/jwohlwend/boltz) — fully open AF3-level biomolecular complex prediction.
- **CUDA libraries & GPU pattern:** cuDNN multi-head attention for Evoformer; custom CUDA triangle update kernels; FP16/BF16 mixed precision; flash attention (FlashAttention2) for memory-efficient MSA attention; multi-GPU model parallelism for large complexes.

---

### 2.2 Protein-Protein Docking 🟢 · Established

- **Deep dive:** Predicting protein-protein complex structures is critical for understanding signaling pathways, antibody-antigen recognition, and designing PPI inhibitors. Classical docking (ClusPro, ZDOCK) uses FFT-based rigid-body search over rotational/translational degrees of freedom — a 6D correlation function evaluable via GPU FFT on spherical harmonic expansions. DL methods (DiffDock-PP, HelixDock, RoseTTAFold2NA) use equivariant diffusion or MSA-based co-evolution to predict complex structures. GPU enables both the FFT rigid-body search and deep learning inference.
- **Key algorithms:** FFT-based rigid-body docking (spherical harmonics), residue-level contact prediction (coevolution), equivariant diffusion docking, half-sphere exposure (HSE) surface scoring, electrostatic/shape complementarity scoring.
- **Datasets:** Docking Benchmark 5.5 — 230 non-redundant protein complexes (https://zlab.umassmed.edu/benchmark/); SAbDab — structural antibody database (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); PPI4DOCK benchmark (verify URL); PDB protein complexes (https://www.rcsb.org).
- **Starter repos/tools:** ClusPro (https://cluspro.bu.edu) — FFT docking server (GPU-accelerated back end); DiffDock-PP (https://github.com/ketatam/DiffDock-PP) — rigid protein-protein diffusion docking; HADDOCK (https://wenmr.science.uu.nl/haddock2.4/) — data-driven docking with GPU MD refinement; RoseTTAFold (https://github.com/RosettaCommons/RoseTTAFold) — two-track network for complex prediction.
- **CUDA libraries & GPU pattern:** cuFFT for rigid-body FFT correlation in 3D; GPU-parallel spherical harmonic expansion; PyTorch CUDA equivariant GNN layers for DL docking; GPU-batched energy evaluation for docking pose refinement.

---

### 2.3 Cryo-EM Single-Particle Reconstruction 🟢 · Established

- **Deep dive:** Single-particle cryo-EM reconstructs 3D density maps from thousands to millions of 2D projection images of vitrified protein particles in random orientations. The reconstruction pipeline involves CTF estimation, 2D class averaging, 3D ab initio reconstruction, and iterative 3D refinement (Bayesian polishing in RELION, non-uniform refinement in cryoSPARC). GPU acceleration is essential: the O(N·M) projection matching step (N particles × M reference projections) dominates walltime. RELION-3/4 and cryoSPARC achieve 10–100× GPU speedup over CPU. EMDB houses 50,000+ deposited maps.
- **Key algorithms:** Contrast transfer function (CTF) estimation, maximum a posteriori (MAP) 3D refinement, expectation-maximization (E-M) for orientation assignment, Fourier-Bessel reconstruction, Bayesian polishing, heterogeneous 3D classification, non-uniform refinement.
- **Datasets:** EMDB — 50,000+ cryo-EM density maps (https://www.ebi.ac.uk/emdb/); EMPIAR — raw cryo-EM particle images (https://www.ebi.ac.uk/empiar/); RCSB PDB structures with cryo-EM validation (https://www.rcsb.org); CryoDRGN benchmark datasets (https://github.com/ml-struct-bio/cryodrgn).
- **Starter repos/tools:** RELION (https://github.com/3dem/relion) — Bayesian cryo-EM reconstruction with CUDA GPU; cryoSPARC (https://cryosparc.com) — commercial GPU reconstruction platform; cryoDRGN (https://github.com/ml-struct-bio/cryodrgn) — heterogeneous VAE reconstruction with GPU; cisTEM (https://cistem.org) — GPU-accelerated cryo-EM software suite.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for Fourier-slice projection; cuFFT for 3D FFT reconstruction; GPU-batched 2D class averaging; warp-parallel expectation step; multi-GPU domain decomposition for large reconstructions.

---

### 2.4 Cryo-ET Subtomogram Averaging 🟡 · Active R&D

- **Deep dive:** Cryo-electron tomography (cryo-ET) images entire cells or organelles, and subtomogram averaging (STA) extracts repeating structural units from noisy 3D tomograms by aligning and averaging thousands of subtomograms. GPU acceleration applies to: (1) tomogram reconstruction from tilt series (weighted back-projection or SART), (2) template matching for particle picking, and (3) subtomogram alignment (cross-correlation in Fourier space). RELION-4 extended STA; the IsoNet neural network corrects missing wedge artifacts with GPU inference.
- **Key algorithms:** Weighted back-projection (WBP), SART tomographic reconstruction, subtomogram cross-correlation alignment, missing wedge compensation, constrained correlation, geometric 3D class averaging, equivariant NN for missing wedge correction.
- **Datasets:** EMDB STA deposits (https://www.ebi.ac.uk/emdb/); EMPIAR-10064 and related datasets (https://www.ebi.ac.uk/empiar/); SHREC subtomogram challenge datasets (verify URL); CryoDRGN-ET benchmark (https://github.com/ml-struct-bio/cryodrgn).
- **Starter repos/tools:** RELION-4 STA (https://github.com/3dem/relion) — Bayesian subtomogram averaging with CUDA; IsoNet (https://github.com/IsoNet-cryoET/IsoNet) — GPU deep learning missing wedge correction; dynamo (https://wiki.dynamo.biozentrum.unibas.ch) — subtomogram averaging with GPU; IMOD (https://bio3d.colorado.edu/imod/) — tomogram reconstruction toolkit.
- **CUDA libraries & GPU pattern:** cuFFT for 3D Fourier-space cross-correlation; custom back-projection CUDA kernels; GPU-parallel template matching over tomogram volume; PyTorch CUDA for IsoNet missing-wedge CNN.

---

### 2.5 Coarse-Grained / MARTINI Simulation 🟢 · Established

- **Deep dive:** Coarse-grained (CG) force fields like MARTINI map ~4 heavy atoms to a single interaction site, enabling microsecond-to-millisecond simulations of large membrane systems (entire plasma membranes with 63 lipid species, viral capsids, ribosomes). MARTINI3 CG-MD runs in GROMACS with full GPU acceleration, gaining ~100-fold timescale extension over all-atom MD. Membrane protein insertion, lipid scrambling, and vesicle formation are accessible only at CG resolution. The GPU bottleneck is non-bonded CG pair interactions; the coarser grid makes PME and neighbor lists faster than all-atom.
- **Key algorithms:** MARTINI3 force field, Lennard-Jones + shifted electrostatics for CG beads, elastic network overlay (Gō-MARTINI) for protein secondary structure, CG-to-AA backmapping, PME for long-range CG electrostatics.
- **Datasets:** CHARMM-GUI MARTINI membrane builder outputs (https://charmm-gui.org); lipid parameter database (https://cgmartini.nl); membrane-active peptide aggregation benchmarks; EMDB viral capsid reference maps for validation.
- **Starter repos/tools:** GROMACS+MARTINI3 (https://github.com/gromacs/gromacs) — production GPU CG-MD; MARTINI force field files (https://cgmartini.nl) — official parameter repository; TS2CG (https://github.com/weria-pezeshkian/TS2CG) — triangulated surface to CG membrane builder; insane.py (https://github.com/Tsjerk/Insane) — membrane assembly tool for MARTINI.
- **CUDA libraries & GPU pattern:** CUDA kernels for CG non-bonded pair evaluation; cuFFT for CG PME; neighbor list construction with larger cutoffs (1.1–1.2 nm vs 0.9 nm AA); GPU memory efficiency improved by reduced atom count (~4× vs AA).

---

### 2.6 Normal Mode Analysis / Elastic Network Models 🟢 · Established

- **Deep dive:** Normal Mode Analysis (NMA) computes the low-frequency vibrational modes of a protein structure, revealing collective motions (domain movements, breathing modes) relevant to allostery and function. The bottleneck is diagonalization of the 3N×3N Hessian matrix (N = atom count) — an O(N³) dense eigenvalue problem. For large proteins (N > 50,000 atoms) this is intractable on CPU. Elastic Network Models (ENMs: ANM, GNM) use simplified Hookean springs between Cα atoms, reducing the matrix but still benefiting from GPU cuSOLVER for eigendecomposition and CUDA-accelerated matrix-vector products (Lanczos iteration for sparse NMA).
- **Key algorithms:** Anisotropic network model (ANM), Gaussian network model (GNM), Hessian matrix construction (pairwise spring contacts), Lanczos/ARPACK for sparse eigendecomposition, overlap with experimental B-factors/conformational changes, RMSF from mode summation.
- **Datasets:** PDB protein structures (https://www.rcsb.org); ProDy structural dynamics dataset (https://github.com/prody/ProDy); MoDEL MD database for NMA validation (https://mmb.irbbarcelona.org/MoDEL/); flexnMR NMR flexibility benchmark (verify URL).
- **Starter repos/tools:** ProDy (https://github.com/prody/ProDy) — Python NMA/ENM with GPU support via PyTorch; iModS server (https://imods.iqfr.csic.es) — NMA-based motion analysis; Bio3D R package (https://thegrantlab.org/bio3d/) — NMA in R; ElNemo (https://www.sciences.univ-nantes.fr/elnemo/) — elastic network modes server.
- **CUDA libraries & GPU pattern:** cuSOLVER dense dsyevd for moderate-sized Hessians; cuSPARSE for sparse ANM matrix-vector products; custom CUDA Lanczos iteration for large sparse NMA; cuBLAS for B-factor RMSF accumulation.

---

### 2.7 Monte Carlo Protein Structure Sampling 🟡 · Active R&D

- **Deep dive:** Monte Carlo (MC) methods sample protein conformational space by proposing random moves (backbone/sidechain dihedral rotations, rigid-body domain motions) and accepting/rejecting via Metropolis criterion. GPU acceleration is applied to (i) batch scoring of many independent MC walkers in parallel and (ii) GPU-accelerated energy evaluation for each trial move. Rosetta's protein design/folding MC engine has been partially GPU-accelerated. Parallel tempering MC scales to GPU arrays via independent temperature replicas. Applications include loop modeling, sidechain packing, and protein-ligand pose sampling.
- **Key algorithms:** Metropolis-Hastings MC, parallel tempering, fragment-based backbone moves (Rosetta), rotamer library sidechain packing (Dunbrack), basin hopping, simulated annealing, energy function evaluation (Rosetta or AMBER).
- **Datasets:** CASP protein structure benchmarks (https://predictioncenter.org); PDB structures for folding benchmarks (https://www.rcsb.org); Dunbrack rotamer library (https://dunbrack.fccc.edu/bbdep2010/); CAMEO continuous benchmarking (https://www.cameo3d.org).
- **Starter repos/tools:** Rosetta (https://github.com/RosettaCommons/rosetta) — protein MC sampling (GPU extensions experimental); FoldX (https://foldxsuite.crg.eu) — fast energy evaluation for MC design; OpenMM MC (https://github.com/openmm/openmm) — Python MC on GPU via custom integrators; ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU sequence design complementary to MC backbone sampling.
- **CUDA libraries & GPU pattern:** GPU-parallel scoring of independent MC replica arrays; CUDA kernels for energy evaluation (Lennard-Jones + torsion); cuRAND for GPU random number generation; warp-level acceptance ratio evaluation.

---

### 2.8 GPU Molecular Visualization & Ray Tracing 🟢 · Established

- **Deep dive:** Interactive visualization of molecular dynamics trajectories, cryo-EM density maps, and protein structures requires real-time rendering of millions of atoms with surface representations (VDW spheres, solvent-accessible surface, cartoon ribbons). VMD uses CUDA for GPU ray tracing (Tachyon/RTX OptiX), generating photorealistic images of molecular systems with ambient occlusion and shadows. Interactive manipulation of multi-million-atom systems at >30 fps is achievable on RTX GPUs. NVIDIA IndeX provides volume rendering of cryo-EM maps directly on GPU.
- **Key algorithms:** GPU ray tracing (OptiX/Embree), CUDA ambient occlusion, isosurface extraction (marching cubes on GPU), molecular surface triangulation, GPU-accelerated MSMS algorithm, volume rendering via GPU compositing, instanced rendering for periodic systems.
- **Datasets:** EMDB cryo-EM maps (https://www.ebi.ac.uk/emdb/); RCSB PDB molecular structures (https://www.rcsb.org); GPCRmd MD trajectories (https://gpcrmd.org); CHARMM-GUI example systems (https://charmm-gui.org).
- **Starter repos/tools:** VMD (https://www.ks.uiuc.edu/Research/vmd/) — GPU-accelerated molecular visualization with CUDA/OptiX ray tracing; PyMOL (https://github.com/schrodinger/pymol-open-source) — GPU-rendered molecular graphics; OVITO (https://www.ovito.org) — GPU-enabled scientific visualization for MD; Mol* (https://github.com/molstar/molstar) — WebGL-accelerated online viewer.
- **CUDA libraries & GPU pattern:** NVIDIA OptiX for hardware ray tracing; custom CUDA marching cubes for isosurface; CUDA sphere/cylinder instanced rendering; GPU-parallel surface normal computation; cuFFT for density map smoothing.

---

### 2.9 Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics 🟢 · Established

- **Deep dive:** Continuum electrostatics models (Poisson-Boltzmann equation, PBE) compute the electrostatic potential of a protein in ionic solvent by solving a partial differential equation on a 3D grid. This enables calculation of protein pKa values, electrostatic binding contributions, and zeta potentials for colloidal drug carriers. GPU-accelerated PBE solvers (APBS, DelPhi-GPU) discretize the molecule onto a Eulerian grid and solve via Gauss-Seidel iteration or multigrid methods on GPU. The bottleneck is the 3D finite-difference PBE solve — parallelized via coloring (red-black ordering) on GPU threads.
- **Key algorithms:** Linearized Poisson-Boltzmann equation (LPBE), non-linear PBE, finite difference discretization (3D grid), red-black Gauss-Seidel iteration, multigrid preconditioning, generalized Born (GB) analytic approximation, SASA computation.
- **Datasets:** pKDBD — database of protein pKa values (verify URL); BindingMOAD — protein-ligand electrostatic data (https://bindingmoad.org); RCSB PDB structural data (https://www.rcsb.org); APBS validation benchmark (https://github.com/Electrostatics/apbs).
- **Starter repos/tools:** APBS (https://github.com/Electrostatics/apbs) — Poisson-Boltzmann solver with GPU acceleration; DelPhi (http://compbio.clemson.edu/delphi) — PB electrostatics with GPU solver; OpenMM GB force (https://github.com/openmm/openmm) — GPU Generalized Born; PDB2PQR (https://github.com/Electrostatics/pdb2pqr) — structure preparation for PBE.
- **CUDA libraries & GPU pattern:** CUDA thread blocks for 3D finite-difference red-black iteration; shared memory for stencil computation; cuSPARSE for sparse Laplacian matrix; GPU texture memory for dielectric boundary representation.

---

### 2.10 Protein Design / Inverse Folding Inference 🟢 · Established

- **Deep dive:** Inverse folding (sequence design) asks: given a backbone structure, what amino acid sequences will fold into it? ProteinMPNN uses a graph neural network that processes backbone geometry (Cα coordinates, virtual Cβ, backbone dihedrals) to autoregressively decode sequences. GPU inference generates diverse sequences at ~1–2 seconds per protein per 100 residues. Wet-lab validation shows 50–55% native sequence recovery. Integration with structure prediction (RFdiffusion for backbone generation → ProteinMPNN for sequence → AlphaFold2 for validation) creates a fully GPU-accelerated computational protein design pipeline.
- **Key algorithms:** Autoregressive sequence decoding on protein graph, message-passing GNN over backbone geometry, tied decoding for symmetric oligomers, temperature-controlled sampling, order-agnostic decoding, LigandMPNN for small-molecule-aware design.
- **Datasets:** CATH protein structure database — 500k+ domain structures (https://www.cathdb.info); PDB training set for ProteinMPNN (https://www.rcsb.org); ProteinGym benchmark — mutational fitness (https://github.com/OATML-Markslab/ProteinGym); CAMEO validation (https://www.cameo3d.org).
- **Starter repos/tools:** ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — official GPU inverse folding model (Baker Lab); LigandMPNN (https://github.com/dauparas/LigandMPNN) — inverse folding with ligand context; ESM-IF1 (https://github.com/facebookresearch/esm) — ESM inverse folding model; RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — backbone diffusion for de novo design coupled with ProteinMPNN.
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA GNN layers for protein graph; GPU autoregressive decoding with kv-cache; FP16 mixed precision; batched sequence generation across multiple backbone inputs; cuDNN for backbone encoder.

---

### 2.11 Cryo-EM CTF Estimation & Particle Picking 🟡 · Active R&D

- **Deep dive:** Before reconstruction, cryo-EM processing requires estimating the contrast transfer function (CTF) from each micrograph and then detecting protein particle positions (particle picking). CTF estimation fits a parametric model to power spectra computed via GPU FFT. Particle picking using template matching requires cross-correlation of the micrograph with reference projections — an O(N·M) operation over image patches and reference orientations, naturally GPU-parallelized. Deep learning pickers (TOPAZ, crYOLO) run GPU CNN inference on tiled micrographs. Both stages process thousands of micrographs in real time.
- **Key algorithms:** CTF power spectrum estimation (Thon rings fitting), 2D cross-correlation template matching, GPU-batched FFT for power spectra, CNN-based particle detection (YOLO/TOPAZ), active learning picker improvement.
- **Datasets:** EMPIAR micrograph archives (https://www.ebi.ac.uk/empiar/); EMPIAR-10025 (β-galactosidase), EMPIAR-10064 (80S ribosome); curated picking benchmarks from CryoBench (verify URL); RELION tutorial datasets (https://relion.readthedocs.io).
- **Starter repos/tools:** RELION CtfFind/MotionCor2 interface (https://github.com/3dem/relion) — GPU CTF + motion correction; TOPAZ (https://github.com/tbepler/topaz) — deep learning particle picker with GPU; crYOLO (https://cryolo.readthedocs.io) — YOLO-based GPU particle detector; CTFFIND4 (https://grigoriefflab.umassmed.edu/ctffind4) — GPU-accelerated CTF estimation.
- **CUDA libraries & GPU pattern:** cuFFT for micrograph power spectrum; CUDA 2D cross-correlation for template matching; custom CUDA FFT-based NCC; PyTorch GPU CNN for TOPAZ/crYOLO inference; multi-stream processing for parallel micrograph batches.

---

### 2.12 Flexible Fitting / MDFF 🟡 · Active R&D

- **Deep dive:** Molecular Dynamics Flexible Fitting (MDFF) fits an atomic model into a cryo-EM density map by adding density-derived forces to GPU MD, deforming the model to match the experimental map. This hybrid approach uses the GPU MD engine (NAMD or OpenMM) to handle sterics and covalent geometry while the density map acts as an external potential. GPU acceleration enables rapid convergence of the fitting for large complexes (ribosomes, viral capsids). Applications include fitting into sub-5 Å cryo-EM maps and interpreting conformational states.
- **Key algorithms:** MDFF density-derived forces (cross-correlation gradient), EMFIT potential, real-space refinement, phenix.real_space_refine, morphing between states, flexible backbone MDFF.
- **Datasets:** EMDB reference maps for MDFF (https://www.ebi.ac.uk/emdb/); EMPIAR raw particle data (https://www.ebi.ac.uk/empiar/); ribosome MDFF benchmarks (PDB 3J7Y, 4V6X); viral capsid fitting datasets.
- **Starter repos/tools:** NAMD MDFF (https://www.ks.uiuc.edu/Research/namd/) — production flexible fitting with CUDA; VMD MDFF plugin (https://www.ks.uiuc.edu/Research/vmd/) — MDFF setup and visualization; phenix.real_space_refine (https://phenix-online.org) — GPU-accelerated real-space refinement; Coot (https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/) — interactive model building into density.
- **CUDA libraries & GPU pattern:** Full GPU MD (NAMD CUDA) with additional density-gradient force kernel; cuFFT for cross-correlation computation in reciprocal space; GPU-parallel evaluation of density at atom positions via trilinear interpolation CUDA kernel.

---

### 2.13 MSA Generation Acceleration 🟡 · Active R&D

- **Deep dive:** Multiple sequence alignment (MSA) construction for AlphaFold2 is a major bottleneck: HHblits and Jackhmmer search the UniRef90 database (210GB) requiring hours of CPU time. GPU acceleration of profile hidden Markov model (HMM) search is an active area: GPU-HMMER uses CUDA to parallelize the Viterbi/Forward-Backward dynamic programming recursion over thousands of sequence targets simultaneously. Accelerating MSA generation could remove one of the last CPU-bound steps in the AF2 prediction pipeline, enabling rapid large-scale proteome annotation.
- **Key algorithms:** Profile HMM Viterbi algorithm, Forward-Backward DP, Smith-Waterman alignment, position-specific scoring matrix (PSSM) search, k-mer seed hashing, HHblits iterated profile-profile alignment.
- **Datasets:** UniRef90 — 210GB protein sequence database (https://www.uniprot.org/help/uniref); UniClust30 (https://uniclust.mmseqs.com); MGnify metagenomics sequences (https://www.ebi.ac.uk/metagenomics/); BFD — Big Fantastic Database (https://bfd.mmseqs.com).
- **Starter repos/tools:** MMseqs2 (https://github.com/soedinglab/MMseqs2) — ultra-fast protein search and clustering (GPU-capable via SIMD/GPU versions); ColabFold MSA server (https://github.com/sokrypton/ColabFold) — GPU-accelerated MSA for AlphaFold2; GPU-HMMER (verify URL) — CUDA Viterbi HMM search; Linclust (https://github.com/soedinglab/MMseqs2) — GPU-accelerated sequence clustering.
- **CUDA libraries & GPU pattern:** CUDA DP recursion for HMM Viterbi (row-parallel); GPU parallel Smith-Waterman via CUDASW++; warp-parallel query-vs-target scoring; GPU hash tables for k-mer seed lookup.

---

### 2.14 Protein-Ligand Co-Folding 🟡 · Active R&D

- **Deep dive:** Co-folding models simultaneously predict protein structure and ligand binding pose in a single forward pass, bypassing separate docking steps. Boltz-1 and AlphaFold3 accept ligand SMILES and protein sequence as joint inputs to a diffusion model conditioned on molecular features. GPU inference generates protein-ligand complex structures at near-FEP accuracy for pose prediction in minutes per complex. The GPU bottleneck is the diffusion sampling loop (50–200 denoising steps), each requiring a full attention forward pass over the joint protein-ligand token sequence.
- **Key algorithms:** Joint protein-ligand diffusion (DDPM on 3D positions), conditional atom-type and geometry generation, atom-level self-attention with periodic boundary handling, confidence (pLDDT/iPAE) scoring, cross-attention between protein and ligand tokens.
- **Datasets:** PoseBusters benchmark — 428 recently released PDB complexes (https://github.com/maabuu/posebusters); PDB-bind v2020 (http://www.pdbbind.org.cn); Astex Diverse Set — 85 drug-like ligand complex structures (verify URL); CASF cross-docking benchmarks (http://www.pdbbind.org.cn/casf.php).
- **Starter repos/tools:** Boltz-1 (https://github.com/jwohlwend/boltz) — GPU co-folding of protein-ligand-nucleic acid complexes; NeuralPLexer3 (https://github.com/zrqiao/NeuralPLexer) — state-specific co-folding with CUDA; AlphaFold3 (https://github.com/google-deepmind/alphafold3) — official AF3 with ligand support; DiffDock (https://github.com/gcorso/DiffDock) — diffusion docking without co-folding (complementary).
- **CUDA libraries & GPU pattern:** Flash attention (FlashAttention2) for long joint sequences; cuDNN transformer blocks; GPU diffusion denoising loop with CUDA noise schedules; FP16/BF16 precision; multi-GPU model parallelism for large complexes.

---

### 2.15 Antibody Structure Prediction 🟡 · Active R&D

- **Deep dive:** Antibody structure prediction is specialized because the CDR-H3 loop is hypervariable and controls antigen specificity. Tools like IgFold, ABodyBuilder3, and IMGT-optimized AlphaFold2 models predict full antibody Fv region structures including flexible CDR loops. GPU inference enables high-throughput prediction for antibody library screening — thousands of sequences per GPU-hour. ABodyBuilder3 uses language model embeddings (ESM-2) and optimized GPU vectorization from OpenFold. Applications include antibody humanization, affinity maturation design, and developability assessment.
- **Key algorithms:** Attention-based CDR loop prediction, language model (ESM-2/IgLM) embeddings for antibody sequences, IMGT-numbered structure prediction, CDR-H3 loop sampling via diffusion, disulfide bond geometry constraints.
- **Datasets:** SAbDab — Structural Antibody Database (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); OAS (Observed Antibody Space) — 2B antibody sequences (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); CASP-Ab benchmarks; Thera-SAbDab — therapeutic antibody database (https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/).
- **Starter repos/tools:** IgFold (https://github.com/Graylab/IgFold) — fast antibody structure prediction on GPU; ABodyBuilder3 (verify GitHub URL) — GPU-optimized AF2 antibody model; AbNatiV (verify URL) — antibody naturalness scoring; AbDiffuser (verify URL) — antibody sequence+structure diffusion.
- **CUDA libraries & GPU pattern:** cuDNN multi-head attention for ESM-2 backbone; custom CDR attention CUDA kernels; FP16 inference with Flash attention; GPU-batched prediction for antibody library screening; PyTorch distributed for multi-GPU fine-tuning.

---

### 2.16 ΔΔG Stability Prediction 🟡 · Active R&D

- **Deep dive:** Predicting the thermodynamic stability change upon single amino acid mutation (ΔΔG) is critical for protein engineering, antibody optimization, and understanding disease variants. ML approaches train on experimental ΔΔG datasets (Protherm, Megascale) using structural features (ProteinMPNN ddG, ThermoMPNN), sequence language models (ESM-1v, EVmutation), or structure-sequence joint models. GPU training on millions of mutation datapoints and GPU inference for saturation mutagenesis scanning (all 20 AA × every position) makes library-scale ΔΔG feasible.
- **Key algorithms:** ProteinMPNN fixed-backbone energy decomposition, ESM-1v zero-shot log-likelihood scoring, Rosetta ddG monomer protocol (FoldX, Cartesian ddG), GNN per-residue embedding, saturation mutagenesis scanning.
- **Datasets:** Protherm database — >25k experimental ΔΔG values (https://www.abren.net/protherm/); Megascale dataset — 2.5M thermodynamic stability measurements (https://github.com/Rocklin-Lab/cdna-display-proteolysis-datasets); ProteinGym substitutions benchmark (https://github.com/OATML-Markslab/ProteinGym); S669 curated stability benchmark (verify URL).
- **Starter repos/tools:** ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN) — GPU ΔΔG prediction from ProteinMPNN; ProteinMPNN-ddG (https://github.com/PeptoneLtd/proteinmpnn_ddg) — saturation mutagenesis ΔΔG; ESM-1v (https://github.com/facebookresearch/esm) — zero-shot stability from language model; FoldX (https://foldxsuite.crg.eu) — fast empirical ΔΔG.
- **CUDA libraries & GPU pattern:** GPU GNN inference for per-residue stability; batched language model forward passes (cuDNN attention); GPU saturation mutagenesis via batched masked prediction; PyTorch Distributed for large-scale training.

---

### 2.17 Allosteric Network Analysis 🟡 · Active R&D

- **Deep dive:** Allostery — ligand binding at one site affecting activity at a distant site — is a major drug target mechanism. Computational allostery detection uses MD trajectory correlation analysis, perturbation response scanning, community detection on protein contact graphs, and causal DCC (dynamical cross-correlation). GPU-accelerated MD generates the long trajectories needed; GPU matrix operations parallelize the N×N residue-residue cross-correlation calculation. Community detection on large protein network graphs benefits from GPU graph algorithms. Applications include identifying cryptic allosteric pockets for undruggable targets.
- **Key algorithms:** Dynamic cross-correlation matrix (DCC), mutual information between residue fluctuations, Linear Response Theory (LRT) perturbation scanning, graph community detection (Girvan-Newman, Louvain), protein contact network analysis, Shortest-path allostery communication (WORDOM).
- **Datasets:** GPCRmd allosteric trajectory archive (https://gpcrmd.org); MDAnalysis trajectory benchmarks (https://github.com/MDAnalysis/mdanalysis); ProDy benchmark datasets (https://github.com/prody/ProDy); allosteric dataset ASD (http://mdl.shsmu.edu.cn/ASD/).
- **Starter repos/tools:** ProDy (https://github.com/prody/ProDy) — allosteric analysis with ANM/DCC; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — trajectory analysis including correlation; PyInteraph2 (https://github.com/ELELAB/pyinteraph2) — protein interaction network analysis; Bio3D R package (https://thegrantlab.org/bio3d/) — cross-correlation and network analysis.
- **CUDA libraries & GPU pattern:** GPU cross-correlation matrix via cuBLAS outer product; custom CUDA kernel for pairwise distance monitoring; GPU Louvain community detection via RAPIDS cuGraph; GPU-parallel trajectory featurization.

---

### 2.18 NMR Structure Refinement 🟡 · Active R&D

- **Deep dive:** NMR structure determination requires satisfying distance restraints (NOE: <5 Å), dihedral angle restraints (J-couplings), and RDC (residual dipolar coupling) data via simulated annealing MD. GPU MD accelerates the restrained simulated annealing protocol, especially for large proteins where many restraint evaluations occur per timestep. GPU-accelerated CYANA/XPLOR-NIH can run hundreds of independent SA trajectories simultaneously — essential for ensemble NMR structure determination. Structure validation against chemical shift back-calculation is also GPU-acceleratable.
- **Key algorithms:** Simulated annealing MD with NOE/dihedral/RDC restraints, distance geometry embedding, torsion angle dynamics (CYANA), refinement against CSROSETTA chemical shifts, back-calculation of NMR observables.
- **Datasets:** BMRB — Biological Magnetic Resonance Bank (https://bmrb.io); PDB NMR-derived structures (https://www.rcsb.org); RECOORD — recalculated NMR structures (verify URL); CASD-NMR automated structure determination benchmarks (verify URL).
- **Starter repos/tools:** XPLOR-NIH (https://nmr.cit.nih.gov/xplor-nih/) — restrained MD for NMR with GPU support (via NAMD); CYANA (http://www.cyana.org) — torsion angle dynamics for NMR; AMBER NMR refinement (https://ambermd.org) — pmemd.cuda with NMR restraints; ARIA (http://aria.pasteur.fr) — automated NMR assignment and refinement.
- **CUDA libraries & GPU pattern:** Full GPU MD for restrained SA (pmemd.cuda); CUDA kernel for NOE energy and gradient computation; GPU-parallel independent SA replica array via MPI+CUDA; GPU chemical shift back-calculation via ShiftX2-GPU (verify URL).

---

### 2.19 Membrane Protein Simulation 🟢 · Established

- **Deep dive:** Membrane proteins (GPCRs, ion channels, transporters, integrins) are embedded in lipid bilayers and represent >50% of current drug targets. Explicit membrane MD requires building asymmetric bilayers with physiological lipid compositions and running microsecond simulations to sample conformational changes. CHARMM-GUI automates system building; GPU GROMACS/NAMD runs production simulations. Key challenges include equilibrating the membrane (~100 ns), maintaining bilayer asymmetry, and capturing slow conformational transitions. GPU-accelerated CG-MARTINI pre-equilibration (1–10 μs) followed by backmapping to all-atom provides a common pipeline.
- **Key algorithms:** CHARMM36 lipid force field, POPE/POPC/cholesterol bilayer assembly, semi-isotropic barostat (NPT-xy coupling), PME for charged bilayer system, CG-to-AA backmapping, k-means clustering of ion channel gate states.
- **Datasets:** MemProtMD — 3133 membrane proteins in lipid bilayers (https://memprotmd.bioch.ox.ac.uk); GPCRdb — GPCR structures and MD data (https://gpcrdb.org); CGMD Platform benchmark systems (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/); OPM — orientations of proteins in membranes (https://opm.phar.umich.edu).
- **Starter repos/tools:** CHARMM-GUI Membrane Builder (https://charmm-gui.org) — automated bilayer + protein setup; GROMACS (https://github.com/gromacs/gromacs) — GPU membrane protein MD; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated membrane protein pipeline; packmol-memgen (https://github.com/memembranes) — AMBER membrane system builder.
- **CUDA libraries & GPU pattern:** GPU semi-isotropic barostat coupling; cuFFT for PME with charged bilayer; custom CUDA PME corrections for 2D slab geometry; multi-GPU domain decomposition along z-axis; GPU neighbor list for heterogeneous lipid-protein system.

---

### 2.20 Heterogeneous Cryo-EM Reconstruction (3D Variability) 🟡 · Active R&D

- **Deep dive:** Real protein complexes adopt multiple conformational states simultaneously. Heterogeneous reconstruction methods disentangle these states from particle images. CryoDRGN uses a variational autoencoder (VAE) with an amortized encoder that maps each particle image to a latent code representing its conformation, and a decoder that generates the 3D density from the latent code via a coordinate MLP. GPU training is essential: a cryoDRGN run on 100k particles requires hours on A100. 3DVA (cryoSPARC) uses PCA-like linear subspace methods. Applications reveal continuous flexibility in ribosomes, GPCR complexes, and viral assembly intermediates.
- **Key algorithms:** Variational autoencoder (VAE) with image encoder and volume decoder, coordinate-based implicit neural representation (NeRF/MLP decoder), 3D variability analysis (PCA on volume subspace), pose estimation EM, Fourier-slice theorem in the network, manifold learning.
- **Datasets:** EMPIAR-10180 (spliceosome), EMPIAR-10076 (80S ribosome), EMPIAR-10028 (TRPV1) (all at https://www.ebi.ac.uk/empiar/); cryoDRGN benchmark datasets (https://github.com/ml-struct-bio/cryodrgn); simulated heterogeneous datasets from IgG/spike protein.
- **Starter repos/tools:** CryoDRGN (https://github.com/ml-struct-bio/cryodrgn) — GPU VAE for heterogeneous reconstruction; cryoSPARC 3DVA (https://cryosparc.com) — GPU linear 3D variability analysis; Recovar (verify URL) — GPU regularized covariance heterogeneous reconstruction; DrgnAI (verify URL) — neural 3D reconstruction with pose optimization.
- **CUDA libraries & GPU pattern:** PyTorch CUDA for VAE encoder/decoder; FlashAttention for particle image attention layers; GPU Fourier-slice theorem evaluation via differentiable nufft; cuFFT for power spectrum during training; FP16 mixed precision.

---

### 2.21 Protein-Nucleic Acid Docking & Co-Folding 🟡 · Active R&D

- **Deep dive:** Protein-RNA and protein-DNA interactions are central to gene regulation, CRISPR editing, and RNA therapeutics. Structure prediction for protein-nucleic acid complexes requires modeling both the nucleic acid secondary/tertiary structure and the protein-NA interface. Boltz-1/AlphaFold3 support protein+RNA/DNA inputs in a single diffusion model; RoseTTAFold2NA was trained on protein-nucleic acid complexes. GPU inference handles the expanded token vocabulary (amino acids + nucleotides) and geometric complexity. Applications include CRISPR-Cas structure prediction, RNA-binding protein design, and aptamer development.
- **Key algorithms:** Unified atom-level diffusion for protein+RNA/DNA, joint MSA+RNA template search, coevolutionary coupling for RNA structure, RNA secondary structure prediction (ViennaRNA), protein-NA interface scoring.
- **Datasets:** PDB protein-nucleic acid complexes (https://www.rcsb.org); RNA structure benchmarks from RNA-Puzzles (https://github.com/RNA-Puzzles); PDB-NA complex benchmark sets (verify URL); Rfam RNA family database (https://rfam.org).
- **Starter repos/tools:** RoseTTAFold2NA (https://github.com/uw-ipd/RoseTTAFold2NA) — protein-RNA/DNA structure prediction; Boltz-1 (https://github.com/jwohlwend/boltz) — unified biomolecular complex prediction; ViennaRNA (https://github.com/ViennaRNA/ViennaRNA) — RNA secondary structure (CPU, used as preprocessing); AlphaFold3 (https://github.com/google-deepmind/alphafold3) — nucleic acid + protein + ligand co-folding.
- **CUDA libraries & GPU pattern:** Flash attention for joint protein-NA token sequence; custom CUDA for nucleotide geometric features; GPU diffusion sampling over protein-NA complex; multi-GPU for large CRISPR-guide+target complexes.

---

### 2.22 Electron Density Map Analysis & Model Validation 🟢 · Established

- **Deep dive:** Crystallographic and cryo-EM electron density maps must be validated for model-to-map fit quality before deposition. GPU-accelerated real-space correlation coefficient (RSCC) and Fourier shell correlation (FSC) calculations over millions of voxels enable rapid quality assessment. Phenix, CCP4, and GEMMI provide GPU-accelerated map manipulation. Structure factor calculation (Fcalc vs. Fobs difference maps in crystallography) requires GPU FFT over large reciprocal-space datasets. For cryo-EM, local resolution estimation (MonoRes, ResMap) computes local FSC across the map in sliding windows — GPU-parallelized.
- **Key algorithms:** Real-space correlation coefficient (RSCC), Fourier shell correlation (FSC), difference map calculation (Fo-Fc, 2Fo-Fc), R-factor / R-free crystallographic validation, local resolution estimation, model-to-map fit scoring.
- **Datasets:** EMDB validation maps (https://www.ebi.ac.uk/emdb/); PDB structure factors (https://www.rcsb.org); IUCr validation standards datasets (verify URL); wwPDB OneDep validation pipeline (https://deposit.wwpdb.org).
- **Starter repos/tools:** Phenix (https://phenix-online.org) — crystallography and cryo-EM refinement with GPU acceleration; CCP4 (https://www.ccp4.ac.uk) — crystallographic computing suite; GEMMI (https://github.com/project-gemmi/gemmi) — GPU-friendly CIF/map library; EMAN2 (https://blake.bcm.edu/emanwiki/EMAN2) — GPU cryo-EM processing suite.
- **CUDA libraries & GPU pattern:** cuFFT for structure factor FFT; custom CUDA correlation coefficient computation over map voxels; GPU FSC computation via batched FFT ring averaging; GPU local resolution sliding window in parallel.

---

### 2.23 Protein-Ligand Interaction Energy Decomposition 🟡 · Active R&D

- **Deep dive:** Per-residue energy decomposition (MM-GBSA per-residue, FEP energy components) identifies which protein residues contribute most to ligand binding, guiding lead optimization and resistance mutation analysis. GPU MD trajectories provide snapshots; GPU-parallel per-residue energy evaluation attributes contributions from each residue. This reveals hot-spot residues for mutational scanning, identifies water-mediated interactions, and explains selectivity across protein family members. Kinase resistance mutation mapping in oncology is a prime application.
- **Key algorithms:** MM-GBSA per-residue energy decomposition, pairwise interaction energy, electrostatic + VDW component separation, water bridge detection, solvent contribution per residue, FEP component analysis.
- **Datasets:** PDB-bind (http://www.pdbbind.org.cn); resistance mutation datasets (ClinVar, https://www.ncbi.nlm.nih.gov/clinvar/); KLIFS kinase binding data (https://klifs.net); ChEMBL activity data for target families (https://www.ebi.ac.uk/chembl/).
- **Starter repos/tools:** AMBER MMPBSA.py decomp (https://ambermd.org/AmberTools.php) — per-residue energy decomposition; gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — GROMACS MM-GBSA decomposition; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — pairwise residue-ligand contact analysis; ProLIF (https://github.com/chemosim-lab/ProLIF) — IFP for binding mode decomposition.
- **CUDA libraries & GPU pattern:** GPU MD trajectory generation; CUDA parallel per-residue GB energy evaluation; GPU-batched snapshot processing (N frames × M residues); cuBLAS for energy matrix accumulation.

---

### 2.24 SAXS / SANS Data-Driven Structure Modeling 🟡 · Active R&D

- **Deep dive:** Small-angle X-ray/neutron scattering (SAXS/SANS) provides solution-phase structural information about proteins and complexes as a 1D intensity profile I(q). Fitting atomic or CG models to SAXS data requires rapid forward calculation of the scattering intensity from 3D coordinates via Debye formula or spherical harmonic expansion — a pairwise summation over all atoms that is GPU-parallelizable. GPU-MD + SAXS ensemble refinement (EROS, BioEn) samples thousands of conformers and reweights to match experimental SAXS. Applications include intrinsically disordered protein (IDP) ensemble characterization.
- **Key algorithms:** Debye scattering formula (O(N²) GPU-parallel), CRYSOL implicit solvent scattering model, spherical harmonic expansion for SAXS, SAXS-restrained MD ensemble refinement (EROS/BioEn), maximum entropy reweighting, atomistic vs CG SAXS prediction.
- **Datasets:** SASBDB — small-angle scattering biological data bank (https://www.sasbdb.org); PDB-SAXS depositions (https://www.rcsb.org); BIOISIS benchmark (verify URL); simulated SAXS from MD trajectories.
- **Starter repos/tools:** CRYSOL (https://www.embl-hamburg.de/biosaxs/crysol.html) — analytical SAXS computation; FOXS (https://modbase.compbio.ucsf.edu/foxs/) — fast SAXS fitting; WAXSiS (verify URL) — GPU-accelerated wide-angle scattering; MDAnalysis SAXS module (https://github.com/MDAnalysis/mdanalysis) — trajectory SAXS averaging.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for O(N²) Debye summation over atom pairs; GPU partial sum reduction for form factors; cuBLAS for spherical harmonic coefficients; GPU-parallel ensemble member scoring.

---

### 2.25 Coevolutionary Contact Prediction & MSA Transformer 🟡 · Active R&D

- **Deep dive:** Coevolutionary analysis of MSAs (correlated mutations between residue positions) reveals protein contact maps that drive structure prediction. EVcouplings uses PLMC (pseudolikelihood-maximized direct coupling analysis) — an L×L matrix inversion and optimization problem where L is sequence length. GPU acceleration via direct CUDA implementation or PyTorch autograd parallelizes the DCA learning over position pairs. MSA Transformer (ESM-MSA-1b) processes MSA rows and columns via tied axial attention on GPU, producing contact predictions and rich evolutionary embeddings for downstream tasks.
- **Key algorithms:** Mutual information (MI), Direct Coupling Analysis (DCA), pseudolikelihood-maximized DCA (PLMC), message-passing DCA (mpDCA), MSA Transformer (axial row/column attention), coevolutionary coupling score to contact map.
- **Datasets:** UniRef50/UniRef90 for MSA construction (https://www.uniprot.org); Pfam MSA database (https://pfam.xfam.org); EVcouplings benchmark contact sets (https://github.com/debbiemarkslab/EVcouplings); CASP14 contact prediction benchmarks (https://predictioncenter.org).
- **Starter repos/tools:** EVcouplings (https://github.com/debbiemarkslab/EVcouplings) — DCA-based coevolutionary analysis; ESM-MSA-1b (https://github.com/facebookresearch/esm) — GPU MSA Transformer; CCMpred (https://github.com/soedinglab/CCMpred) — GPU-accelerated DCA with CUDA implementation; HHpred (https://toolkit.tuebingen.mpg.de/tools/hhpred) — profile-profile alignment for MSA.
- **CUDA libraries & GPU pattern:** CCMpred custom CUDA kernels for DCA gradient computation; cuBLAS for L×L coupling matrix products; PyTorch CUDA axial attention for MSA Transformer; GPU-parallel MSA column featurization.

---

### 2.26 Hydrogen Bond Network & Water Placement Analysis 🟡 · Active R&D

- **Deep dive:** Water molecules mediate protein-ligand interactions at binding sites; their correct placement is critical for accurate docking and scoring. GPU-accelerated MD generates explicit water trajectories from which statistical water occupancy maps (WaterMap, GIST) are computed. The Grid Inhomogeneous Solvation Theory (GIST) requires computing per-voxel thermodynamic quantities (energy, entropy) across millions of trajectory frames — a GPU-parallelizable grid accumulation problem. High-occupancy waters indicate entropically costly displacement sites; displacing them with ligand atoms typically yields affinity gains.
- **Key algorithms:** Grid Inhomogeneous Solvation Theory (GIST), Inhomogeneous Fluid Solvation Theory (IFST), 3D water occupancy map from MD, nearest-neighbor entropy estimation, water bridge H-bond network graph, explicit water clustering.
- **Datasets:** SAMPL water placement challenges (https://github.com/samplchallenges/SAMPL); explicit-solvent PDB structures (https://www.rcsb.org); benchmark sets for WaterMap validation (Schrodinger, verify URL); GIST reference calculations for T4 lysozyme and FKBP12.
- **Starter repos/tools:** GISTPP (https://github.com/liedlgroup/gist-pp) — GIST water thermodynamics analysis; cpptraj GIST (https://github.com/Amber-MD/cpptraj) — AMBER trajectory analysis with GIST; MDAnalysis water analysis (https://github.com/MDAnalysis/mdanalysis) — H-bond and water bridge analysis; WaterMD (verify URL) — GPU-accelerated solvation free energy.
- **CUDA libraries & GPU pattern:** GPU grid accumulation kernels for GIST voxel energy/entropy (atomic updates); custom CUDA nearest-neighbor entropy estimation; MDAnalysis GPU trajectory streaming; GPU-parallel water oxygen occupancy histogramming.

---

### 2.27 Polarizable Water Model GPU Dynamics 🟡 · Active R&D

- **Deep dive:** Accurate water models are foundational to all biomolecular simulation. The MB-pol many-body water potential, TIP4P-D, and OPC3/OPC water models improve upon TIP3P for protein hydration dynamics, but many-body polarizable water (MB-pol) is orders of magnitude more expensive due to 2-body and 3-body interaction terms. GPU acceleration of MB-pol via GPU-accelerated MB-nrg and the MBX library enables multi-nanosecond production simulations. Applications include accurate protein solvation thermodynamics and dielectric constant convergence for force field validation.
- **Key algorithms:** MB-pol many-body potential, 2-body dispersion and 3-body induction, inducible dipole iteration for polarizable water (SWM4-NDP, BK3), TIP4P-D/OPC4 fixed-charge models, water density anomalies benchmarking.
- **Datasets:** NIST water thermophysical properties (https://webbook.nist.gov); HBond dynamics NMR benchmark datasets; MD2PDB water trajectory archives; SPC/E, TIP4P-2005 reference simulation datasets.
- **Starter repos/tools:** MBX library (https://github.com/paesanilab/MBX) — GPU-accelerated many-body water potential; OpenMM polarizable models (https://github.com/openmm/openmm) — GPU inducible dipole water; Tinker-HP AMOEBA water (https://github.com/TinkerTools/tinker-hp) — polarizable water GPU MD; i-PI (https://github.com/i-pi/i-pi) — path integral water dynamics driver.
- **CUDA libraries & GPU pattern:** CUDA kernels for 2-body/3-body interaction evaluation over molecular clusters; custom GPU conjugate-gradient inducible dipole solver; cuBLAS for many-body expansion tensor contractions; GPU batched cluster neighbor lists.

---

### 2.28 Replica Exchange Solute Tempering (REST2) on GPU 🟡 · Active R&D

- **Deep dive:** REST2 (Replica Exchange with Solute Tempering version 2) selectively heats only the solute (protein/ligand) degrees of freedom rather than the whole system, making replica exchange practical for large solvated systems where heating all water would be prohibitively expensive. Effective temperature scaling is applied only to protein internal and protein-water interactions, while water-water interactions remain at 300K. GPU MD runs each replica independently; NCCL/MPI handles exchange coordinate communication between replicas at swap intervals. Applications include enhanced sampling of protein-ligand binding, loop conformational changes, and protein folding.
- **Key algorithms:** Scaled Hamiltonian construction for solute interactions, Metropolis exchange criterion across replicas, potential energy re-scaling (protein-protein + protein-solvent), REST2 vs HREX comparison, virtual replica exchange (vRE-REST2).
- **Datasets:** Shaw millisecond folding trajectories for validation; SAMPL challenges (https://github.com/samplchallenges/SAMPL); GPCRmd REST2 enhanced sampling data (https://gpcrmd.org); chignolin/Trp-cage fast-folder benchmarks.
- **Starter repos/tools:** GROMACS + PLUMED REST2 (https://github.com/gromacs/gromacs) — Hamiltonian REMD on GPU; NAMD REST2 (https://www.ks.uiuc.edu/Research/namd/) — GPU replica exchange; OpenMM REST2 via openmmtools (https://github.com/choderalab/openmmtools) — Python REST2 on GPU; DESMOND REST2 (Schrodinger, commercial) — GPU REST2 for FEP.
- **CUDA libraries & GPU pattern:** Independent GPU MD per replica; NCCL for energy exchange between GPUs; CUDA Hamiltonian scaling kernel (scale protein-water pair forces); MPI inter-node replica exchange; GPU-parallel Metropolis criterion evaluation.

---

### 2.29 Ion Channel Gating & Permeation Simulation 🟡 · Active R&D

- **Deep dive:** Ion channels (Nav, Kv, CFTR, VGCC) are major drug targets whose gating mechanisms operate on microsecond-to-millisecond timescales. GPU MD enables simulation of ion permeation events (channel open → ion transit → channel close) that occur on the ~100 ns timescale at high ion concentration. Umbrella sampling + PMF calculation along the channel axis gives free energy barriers to permeation. GPU-accelerated HOLE algorithm measures pore radius along trajectories. Voltage-clamp conductance simulations require non-equilibrium MD with applied electric field — a simple CUDA modification to the integrator.
- **Key algorithms:** Applied field MD (non-equilibrium ion permeation), umbrella sampling along pore axis (PMF), HOLE pore radius algorithm, mean-first-passage-time for conductance, Brownian dynamics for multi-ion permeation, POISSON-BOLTZMANN pore electrostatics.
- **Datasets:** MemProtMD — membrane protein MD trajectories (https://memprotmd.bioch.ox.ac.uk); PDB ion channel structures (https://www.rcsb.org); patch-clamp electrophysiology data from Channelpedia (https://channelpedia.epfl.ch); GPCRdb for GPCR ion channels (https://gpcrdb.org).
- **Starter repos/tools:** GROMACS + HOLE2 (https://github.com/gromacs/gromacs) — GPU membrane MD with pore analysis; NAMD + VMD ion channel tools (https://www.ks.uiuc.edu/Research/vmd/) — GPU ion permeation analysis; ChannelAnnotation (verify URL) — automated channel analysis from MD; MDAnalysis ion permeation analysis (https://github.com/MDAnalysis/mdanalysis) — trajectory-based conductance estimation.
- **CUDA libraries & GPU pattern:** GPU MD with CUDA electric field integrator; CUDA pore radius kernel (HOLE-inspired sphere packing); multi-GPU for multiple independent permeation runs; GPU-parallel ion position histogram accumulation.

---

### 2.30 Protein Solubility & Phase Separation Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Liquid-liquid phase separation (LLPS) of intrinsically disordered proteins (IDPs) and RNA-binding proteins underlies formation of biomolecular condensates (stress granules, P-bodies, nucleolus). Simulating LLPS requires system sizes of millions of CG atoms over millisecond timescales — only accessible with GPU CG-MD. FUS, TDP-43, and hnRNPA1 condensate-forming domains have been simulated with MARTINI or HPS (hydrophobicity scale) CG models on GPU. Phase diagrams are computed by running multiple concentration conditions simultaneously. Applications include predicting condensate-forming mutations and designing condensate-disrupting drugs.
- **Key algorithms:** Coarse-grained HPS/Kim-Hummer IDP model, MARTINI IDR parameters, Gibbs ensemble MC for phase coexistence, density functional theory for phase diagram, metadynamics order parameter for condensate formation, finite-size scaling for phase boundary.
- **Datasets:** FuzDB — fuzzy protein complex database (https://fuzdb.org); PhaSePro — proteins undergoing LLPS (https://phasepro.elte.hu); DisProt — intrinsically disordered proteins (https://disprot.org); human proteome LLPS predictor datasets (catGRANULE, PScore).
- **Starter repos/tools:** LAMMPS + HPS model (https://github.com/lammps/lammps) — GPU IDP LLPS simulation; OpenMM HPS (https://github.com/openmm/openmm) — Python IDP CG MD; CALVADOS 2 (https://github.com/KULL-Centre/CALVADOS) — residue-level IDP model for LLPS; GROMACS MARTINI IDR (https://github.com/gromacs/gromacs) — GPU CG LLPS.
- **CUDA libraries & GPU pattern:** GPU CG-MD for multi-million-bead IDP system; CUDA kernel for simplified HPS non-bonded interactions; GPU-parallel concentration ensemble (multiple boxes); GPU-accelerated order parameter clustering for phase detection.

---

### 2.31 Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction 🟡 · Active R&D

- **Deep dive:** Cryo-ET tilt-series reconstruction requires (1) frame alignment (beam-induced motion), (2) tilt-series alignment (fiducial or fiducial-free), and (3) tomogram reconstruction (weighted back-projection or iterative SART/ASTRA). All three steps are GPU-parallelizable: GPU-accelerated SART iterates over projection angles simultaneously; WBP uses GPU FFT and filter application. IMOD, AreTomo, and etomo handle tilt-series alignment; the ASTRA Toolbox provides GPU iterative reconstruction via CUDA. Cryo-ET remains limited by the missing wedge artifact, which deep learning (IsoNet) corrects post hoc on GPU.
- **Key algorithms:** Weighted back-projection (WBP), SART (simultaneous algebraic reconstruction), AreTomo beam-induced motion correction, fiducial marker alignment, beam-induced motion correction (MotionCor2-TomoTilt), iterative reconstruction convergence.
- **Datasets:** EMPIAR tilt series archives (https://www.ebi.ac.uk/empiar/); EMDB subtomogram averages (https://www.ebi.ac.uk/emdb/); SHREC cryo-ET benchmark (verify URL); in situ ribosome tilt series (EMPIAR-10045).
- **Starter repos/tools:** IMOD (https://bio3d.colorado.edu/imod/) — standard tomographic reconstruction suite; ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU CUDA reconstruction algorithms; AreTomo2 (https://github.com/czimaginginstitute/AreTomo2) — GPU tilt-series alignment; IsoNet (https://github.com/IsoNet-cryoET/IsoNet) — GPU deep learning missing wedge correction.
- **CUDA libraries & GPU pattern:** Custom CUDA WBP kernel over tilt projection angles; cuFFT for filter application in filtered back-projection; GPU SART iteration with CUDA atomic updates; PyTorch CNN for IsoNet missing-wedge correction; multi-GPU for large tomogram reconstruction.

---

### 2.32 Protein Folding Pathway Extraction (Transition Path Sampling) 🔴 · Frontier/Theoretical

- **Deep dive:** Transition Path Sampling (TPS) harvests rare folding/unfolding events by shooting from configurations near the transition state and accepting/rejecting trajectories that connect folded and unfolded basins. GPU MD makes it practical to run many short (~1–100 ns) shooting moves in parallel. AIMMD (AI-augmented MD) uses GPU-trained neural networks to identify committor isosurfaces, accelerating TPS convergence. Applications include protein folding mechanism elucidation, cryptic pocket opening pathways, and drug unbinding kinetics (τRAMD, WExplore).
- **Key algorithms:** Transition path sampling (TPS) shooting move, aimless shooting, committor analysis, path collective variables (PathCV), weighted ensemble sampling (WExplore/WE-H), τRAMD unbinding kinetics, AIMMD neural committor.
- **Datasets:** Anton/Shaw millisecond trajectories as TPS starting configurations; GPCRmd pathway datasets (https://gpcrmd.org); folding benchmarks: Trp-cage, chignolin, WW domain; SAMPL host-guest kinetics challenges (verify URL).
- **Starter repos/tools:** OpenPathSampling (https://github.com/openpathsampling/openpathsampling) — TPS on GPU via OpenMM; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated adaptive sampling; WESTPA (https://westpa.github.io/westpa/) — weighted ensemble sampling on GPU MD; AIMMD (https://github.com/bioRxiv AIMMD, verify URL) — AI-augmented TPS with GPU neural committor.
- **CUDA libraries & GPU pattern:** GPU MD for fast shooting trajectories; GPU neural network committor inference in AIMMD; NCCL for WE parent-child trajectory coordination; embarrassingly parallel independent shooter array on multi-GPU.

---

### 2.33 Structure-Based Pharmacophore Modeling from MD Ensembles 🟡 · Active R&D

- **Deep dive:** Static pharmacophore models miss receptor flexibility; ensemble pharmacophore modeling derives features from MD trajectory frames, capturing induced-fit and cryptic-pocket binding geometries. GPU-accelerated MD generates the conformational ensemble; GPU-parallel feature extraction (H-bond donor/acceptor, hydrophobic contact maps) across millions of frames clusters into a consensus pharmacophore. The resulting ensemble pharmacophore is used for 3D similarity screening with GPU ROCS/FastROCS against billion-compound libraries, bridging MD insights with ultra-large-scale screening.
- **Key algorithms:** Dynamic pharmacophore feature extraction from MD, ensemble pharmacophore clustering (DBSCAN on feature vectors), 3D Gaussian overlap scoring (ROCS), pharmacophore SMARTS matching, common hits approach (CHA), water-displacement pharmacophore.
- **Datasets:** GPCRmd trajectory archive (https://gpcrmd.org); DUD-E actives/decoys for validation (https://dude.docking.org); PDB structures of target classes (https://www.rcsb.org); ZINC drug-like library for screening (https://zinc20.docking.org).
- **Starter repos/tools:** Pharmer (https://github.com/dkoes/pharmer) — pharmacophore screening tool; MDpocket (https://github.com/Discngine/fpocket) — pocket detection across MD trajectories; HTMD pharmacophore (https://github.com/Acellera/htmd) — ensemble pharmacophore from GPU MD; OpenEye ROCS (https://www.eyesopen.com/rocs) — GPU 3D shape+pharmacophore screening.
- **CUDA libraries & GPU pattern:** GPU Gaussian overlap for ROCS pharmacophore scoring; CUDA H-bond/hydrophobic feature extraction over MD frames; cuML DBSCAN for pharmacophore cluster detection; GPU batch pharmacophore matching over compound library.

---

### 2.34 Biophysical Simulation of Biomolecular Condensates (Active Learning Loop) 🔴 · Frontier/Theoretical

- **Deep dive:** Understanding the sequence determinants of biomolecular condensate properties (surface tension, viscosity, partition coefficients of client molecules) requires an active learning loop: GPU CG-MD generates condensate properties, a surrogate model (GNN on sequence) learns the property landscape, and Bayesian optimization proposes new sequences. This closes the loop between sequence, structure, and function for disordered proteins. GPU acceleration enables the necessary throughput (hundreds of condensate simulations per iteration). Applications include designing condensate-targeting therapeutics and understanding IDR evolution.
- **Key algorithms:** Bayesian active learning on sequence space, GNN surrogate for condensate properties, GPU CG-MD with IDP force fields, coexistence concentration estimation, diffusion coefficient estimation from MSD, transfer matrix for condensate-client partition.
- **Datasets:** PhaSePro (https://phasepro.elte.hu); DisProt (https://disprot.org); experimental LLPS partition coefficient datasets (verify URL); published condensate MD trajectory datasets (FUS, TDP-43, hnRNPA1).
- **Starter repos/tools:** CALVADOS 2 (https://github.com/KULL-Centre/CALVADOS) — GPU-compatible residue-level IDP model; OpenMM + GNN surrogate (https://github.com/openmm/openmm) — active learning condensate loop; LAMMPS GPU (https://github.com/lammps/lammps) — large-scale CG condensate simulation; BoTorch (https://github.com/pytorch/botorch) — GPU Bayesian optimization for sequence design.
- **CUDA libraries & GPU pattern:** GPU CG-MD for condensate equilibration; PyTorch GNN surrogate on sequence features; BoTorch GPU Bayesian optimization; multi-GPU ensemble of condensate simulation replicas; GPU MSD calculation for diffusion coefficient.

---

### 2.35 Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling 🔴 · Frontier/Theoretical

- **Deep dive:** DEER (Double Electron-Electron Resonance) distance measurements between spin labels constrain the conformational ensemble of flexible proteins and membrane proteins in their native membrane environment. GPU-accelerated MD restrained by DEER distance distributions enables ensemble refinement of proteins that cannot be crystallized. The GPU compute pattern parallelize over hundreds of independent MD replicas, each evaluated against DEER restraints (population-weighted distance distribution comparison). Applications include ABC transporter gating, GPCR dynamics, and IDR backbone sampling.
- **Key algorithms:** DEER distance distribution back-calculation from MD ensemble, maximum entropy ensemble reweighting (EROS/BioEn), rotamer library convolution for spin-label placement (MTSSL), GPU MD with soft DEER restraints, population re-weighting.
- **Datasets:** SASBDB EPR-constrained structures (verify URL); published DEER datasets for membrane transporters; EPR.cxls community datasets (verify URL); PDB structures refined with EPR data.
- **Starter repos/tools:** MMM (Multiscale Modeling of Macromolecules, https://www.epr.ethz.ch/software/mmm.html) — EPR-driven ensemble modeling; DEER-PREdict (verify URL) — DEER distance prediction from MD; EnsembleFit/BioEn (https://github.com/bio-phys/BioEN) — GPU Bayesian ensemble reweighting; OpenMM DEER restraints (https://github.com/openmm/openmm) — soft distance restraints from DEER.
- **CUDA libraries & GPU pattern:** GPU MD array for ensemble members; CUDA DEER back-calculation kernel (rotamer convolution over N spin-label positions); GPU population reweighting via maximum entropy; multi-GPU replica ensemble with shared experimental target.

---

---

## 3. Genomics, Sequencing & Bioinformatics

### 3.1 Smith-Waterman / Needleman-Wunsch Alignment 🟢 · Established
- **Deep dive:** Smith-Waterman (SW) computes the optimal local alignment between two sequences via a dynamic-programming (DP) score matrix filled cell-by-cell; at protein-database scale this means quadratic work per query against millions of targets. GPUs collapse this into anti-diagonal wavefront parallelism: all cells on the same anti-diagonal are independent and can be computed simultaneously across thousands of CUDA threads, eliminating the serial dependency that cripples CPUs. CUDASW++4.0 (2024) achieves up to 5.71 TCUPS on an H100 by exploiting Hopper's DPX integer-DP instructions, hardware-native to the architecture, alongside tile-based matrix partitioning and sequence-database chunking for maximal occupancy. The specific bottleneck parallelised is the per-cell recurrence max(H[i-1,j-1]+s, H[i,j-1]-g, H[i-1,j]-g) across the anti-diagonal frontier.
- **Key algorithms:** Smith-Waterman anti-diagonal DP wavefront; Needleman-Wunsch global DP; striped SIMD inter-sequence parallelism; affine gap scoring; DPX hardware DP instructions (Hopper); sequence-database tiling and batched kernel launch.
- **Datasets:** UniProtKB/Swiss-Prot — curated protein sequence database, ~570 k entries (https://www.uniprot.org/downloads); NCBI nr (non-redundant protein) — comprehensive protein database, 100 M+ sequences (https://ftp.ncbi.nlm.nih.gov/blast/db/); PDB sequences — structural protein sequences for benchmarking alignments (https://www.rcsb.org/downloads); NCBI RefSeq — reference nucleotide and protein sequences (https://ftp.ncbi.nlm.nih.gov/refseq/).
- **Starter repos/tools:** CUDASW4 (https://github.com/asbschmidt/CUDASW4) — CUDASW++4.0, H100/A100/L40S optimised, DPX, up to 5.71 TCUPS; GenomeWorks / ClaraGenomics SDK (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — NVIDIA CUDA pairwise alignment primitives for both protein and nucleotide; WFA-GPU (verify URL: github.com/quim0/WFA-GPU) — wavefront alignment algorithm on GPU, gap-affine, ultra-fast for long DNA; Parasail (https://github.com/jeffdaily/parasail) — SIMD/CUDA pairwise alignment library used as reference.
- **CUDA libraries & GPU pattern:** cuBLAS (score accumulation); thrust (sort, scan); CUB (warp-level reduction); custom anti-diagonal kernels with shared memory tiling; inter-sequence batching (one CUDA block per query–target pair or striped across warps); DPX integer instructions on Hopper SM90.

---

### 3.2 Short-Read Mapping / Alignment 🟢 · Established
- **Deep dive:** Short-read mapping (50–300 bp Illumina reads) first seeds candidate positions in a reference genome index (FM-index or hash table), then extends seeds with banded SW. At whole-genome scale (30× coverage ≈ 900 M reads for human), the seed-extension and CIGAR-string generation steps dominate runtime. GPU acceleration batches thousands of read-to-reference extensions simultaneously, each assigned a CUDA thread block with shared-memory score matrix, while FM-index backward search runs as a parallel BFS across thread groups. NVIDIA Parabricks (v4.7, 2025) completes a 30× WGS in under 10 minutes on an H100, vs. >30 hours CPU BWA-MEM, by reimplementing BWA-MEM's seed-chain-extend pipeline in CUDA.
- **Key algorithms:** FM-index / BWT backward search; seed chaining (sparse DP); banded Smith-Waterman extension; CIGAR encoding; markduplicates hashing; Burrows-Wheeler transform; seeding by minimisers.
- **Datasets:** 1000 Genomes Project — 2504 human WGS samples, short reads (https://www.internationalgenome.org/data); Genome in a Bottle (GiaB) NA12878 / HG002 — benchmark short-read WGS datasets (https://www.nist.gov/programs-projects/genome-bottle); SRA FASTQ archives — petabyte-scale short reads (https://www.ncbi.nlm.nih.gov/sra); ENCODE ChIP/RNA-seq FASTQs — curated short-read functional data (https://www.encodeproject.org/).
- **Starter repos/tools:** NVIDIA Parabricks (https://docs.nvidia.com/clara/parabricks/latest/) — GPU-accelerated BWA-MEM + GATK pipeline, 50× faster than CPU; CUSHAW2-GPU (https://github.com/asbschmidt/CUSHAW3) — banded SW seed extension on GPU; Scrooge (https://github.com/CMU-SAFARI/Scrooge) — GPU/CPU co-designed aligner; GenomeWorks (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — pairwise overlap kernels underpinning mapping.
- **CUDA libraries & GPU pattern:** cuSPARSE (index look-ups); thrust (sorting seeds); custom banded-SW kernels with shared-memory tiling; persistent warp-per-read extension; multi-GPU data parallelism via NCCL.

---

### 3.3 Variant Calling Acceleration 🟢 · Established
- **Deep dive:** Germline variant calling applies the Haplotype Caller algorithm: local de novo assembly of active regions, PairHMM forward-algorithm computation of read-haplotype likelihoods, and genotype likelihood calculation. PairHMM is by far the dominant runtime cost—each read must be compared against every candidate haplotype via an O(R×H) DP table. GPU parallelism fills an entire PairHMM table per thread block, running thousands of read-haplotype pairs simultaneously. Parabricks GPU HaplotypeCaller reduces 30× WGS germline calling from ~9 hours CPU to under 10 minutes on an H100 using GATK-identical math. DeepVariant's CNN pileup scoring is a further candidate for batched GPU inference.
- **Key algorithms:** PairHMM forward algorithm; local de novo assembly (De Bruijn graph over active regions); Viterbi realignment; genotype likelihood calculation (GL/PL); base quality score recalibration (BQSR); DeepVariant convolutional inference.
- **Datasets:** GiaB truth sets HG001–HG007 — gold-standard variant calls for benchmarking (https://www.nist.gov/programs-projects/genome-bottle); ClinVar — clinically interpreted variants (https://www.ncbi.nlm.nih.gov/clinvar/); gnomAD v4 — population allele frequencies (https://gnomad.broadinstitute.org/); 1000 Genomes high-coverage WGS (https://www.internationalgenome.org/data).
- **Starter repos/tools:** NVIDIA Parabricks HaplotypeCaller / DeepVariant module (https://docs.nvidia.com/clara/parabricks/latest/) — GATK-identical GPU variant calling; DeepVariant (https://github.com/google/deepvariant) — CNN-based caller deployable on GPU; GATK (https://github.com/broadinstitute/gatk) — CPU reference for parity testing; Clairvoyante / Clair3 (https://github.com/HKU-BAL/Clair3) — deep learning variant caller with GPU inference.
- **CUDA libraries & GPU pattern:** cuDNN (DeepVariant CNN inference); custom PairHMM CUDA kernels with one block per read-haplotype pair; shared-memory DP tables; multi-GPU pipeline parallelism (BQSR → alignment → calling); CUDA streams for pipelining I/O and compute.

---

### 3.4 Nanopore Basecalling 🟢 · Established
- **Deep dive:** Nanopore basecalling translates raw ionic-current signal samples (electrical squiggles) from the sequencer into DNA/RNA base sequences. Oxford Nanopore's Dorado uses a recurrent neural network (transformer + CTC decoder in current "SUP" models) trained to map signal windows to base probabilities. The bottleneck is the RNN/transformer inference over millions of signal events per run hour, a perfect GPU workload: batched matrix multiplications across reads mapped to thousands of CUDA cores. Dorado achieves up to 30% speed improvement for HAC models on Ampere/Ada/Blackwell GPUs over previous versions and scales linearly across multiple GPUs. The GPU also powers modified base (methylation) calling simultaneously during basecalling.
- **Key algorithms:** Bidirectional LSTM / Transformer encoder; Connectionist Temporal Classification (CTC) decoding; beam search decoding; adaptive banded event alignment (f5c); Modified base (5mC, 6mA) classification heads.
- **Datasets:** ONT Open Dataset (PromethION human WGS) — available via SRA / ENA (https://www.ncbi.nlm.nih.gov/sra); R9.4.1 and R10.4.1 benchmark datasets released by ONT (https://github.com/GoekeLab/awesome-nanopore); GIAB ONT ultra-long reads — NA12878/HG002 nanopore truth sets (https://www.nist.gov/programs-projects/genome-bottle); ENA Project PRJNA594038 — public multi-species ONT data (https://www.ebi.ac.uk/ena).
- **Starter repos/tools:** Dorado (https://github.com/nanoporetech/dorado) — ONT's official GPU basecaller, multi-GPU, CUDA-optimised, supports MOD calling; f5c (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation calling and event alignment; awesome-nanopore (https://github.com/GoekeLab/awesome-nanopore) — curated tool index including GPU-enabled callers; Guppy — legacy ONT CUDA basecaller, GPU-only, superseded by Dorado.
- **CUDA libraries & GPU pattern:** cuDNN (RNN/transformer), TensorRT (inference optimisation), cuBLAS (GEMM), CUDA streams (pipelining signal batches); multi-GPU with NVLink/NCCL; persistent thread blocks for stateful RNN across signal chunks.

---

### 3.5 De Novo Genome Assembly 🟡 · Active R&D
- **Deep dive:** De novo assembly reconstructs a genome from raw reads without a reference. The three GPU-amenable bottlenecks are: (1) all-vs-all read overlap detection (O(n²) pairwise alignment), (2) string-graph / De Bruijn graph construction from k-mers, and (3) consensus polishing of draft contigs. NVIDIA's GenomeWorks / racon-GPU accelerates the polishing stage (partial-order alignment MSA) by 70× vs. CPU. The Darwin accelerator paper showed 109× GPU speedup for read overlap on PacBio data. Modern HiFi assembly (hifiasm) is CPU-centric for the string-graph phase, but GPU kernels for pairwise overlap computation are an active insertion point; NVIDIA's Clara de novo pipeline on NGC wraps these components.
- **Key algorithms:** All-vs-all minimiser-based overlap (minimap2 kernel); De Bruijn graph construction and traversal; string-graph simplification (unitig / contig threading); partial-order alignment (POA) for polishing consensus; repeat resolution by Hi-C scaffolding.
- **Datasets:** CHM13 telomere-to-telomere human genome — the T2T gold standard for assembly benchmarking (https://github.com/marbl/CHM13); GenomeArk — vertebrate genome assembly data (https://genomeark.github.io/); Human Pangenome Reference Consortium data (https://humanpangenome.org/); SRA PacBio HiFi and ONT datasets — species-specific de novo projects (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** GenomeWorks / racon-GPU (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — GPU-accelerated overlap and polishing; Clara De Novo Assembly (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/clara_denovo_assembly_pipeline) — NVIDIA NGC end-to-end pipeline; hifiasm (https://github.com/chhylp123/hifiasm) — state-of-the-art HiFi assembler (CPU, GPU overlap insertion point); Racon CPU reference (https://github.com/lbcb-sci/racon) — CPU polishing baseline.
- **CUDA libraries & GPU pattern:** Custom POA kernels in GenomeWorks (shared-memory DP); CUDA thrust for k-mer sorting; minimiser hash tables in GPU global memory; multi-GPU for embarrassingly parallel read-pair overlaps.

---

### 3.6 k-mer Counting & Minimiser Sketching 🟢 · Established
- **Deep dive:** k-mer counting determines the frequency of every length-k substring in a read set, foundational to genome-size estimation, error detection, assembly, and metagenomics. For a 30× human genome (~270 Gb of sequence, k=21), the table has ~4 billion distinct k-mers; efficient parallel hashing and atomic counting saturate GPU memory bandwidth. Gerbil uses GPU-resident hash tables and achieves >10× speed over Jellyfish. Minimiser sketching (selecting a canonical subset of k-mers per window) reduces data by ~5× and enables the MinHash / HyperMinHash distance computations used in species typing; all operations parallelise across reads with one GPU thread per minimiser.
- **Key algorithms:** Radix-sort-based k-mer canonicalisation; GPU hash table with cuckoo / Robin Hood probing; count-min sketch for approximate counting; minimiser extraction (window function); MinHash / Jaccard distance estimation; HyperLogLog cardinality estimation.
- **Datasets:** Illumina WGS of NA12878 — human reference dataset (https://www.ncbi.nlm.nih.gov/sra/SRR622457); GAGE benchmark — multi-species short reads for assembly tools (http://gage.cbcb.umd.edu/); GenomeTrakr pathogen WGS — bacterial surveillance reads (https://www.ncbi.nlm.nih.gov/bioproject/PRJNA183844); Sequence Read Archive (SRA) — global repository (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** Gerbil (https://github.com/uni-halle/gerbil) — k-mer counter with GPU support; KMC3 (https://github.com/refresh-bio/KMC) — disk-I/O efficient CPU k-mer counter (GPU comparison baseline); Jellyfish (https://github.com/gmarcais/Jellyfish) — lock-free hash k-mer counter; GenomeScope2 (https://github.com/tbenavi1/genomescope2.0) — genome profiling from k-mer histograms.
- **CUDA libraries & GPU pattern:** CUDA atomic operations (atomicAdd for count tables); thrust::sort_by_key for radix sort; warp-level ballot and shuffle for minimiser window reduction; cuRAND for sketch initialisation.

---

### 3.7 BLAST-Style Homology Search 🟢 · Established
- **Deep dive:** Homology search finds sequences in a database that are evolutionarily related to a query, using seed-filter-extend logic (BLAST) or k-mer prefiltering + ungapped alignment (MMseqs2 / DIAMOND). At the scale of AlphaFold2 structure prediction (MSA search dominates 70–90% of total inference time), GPU acceleration is transformative. MMseqs2-GPU (2025, Nature Methods) replaces the CPU k-mer prefilter with a GPU-parallel gapless scoring pass across all database sequences simultaneously, achieving 20× speedup and 71× cost reduction vs. 128-core CPU. The bottleneck parallelised is the embarrassingly parallel pairwise k-mer match scanning across millions of database sequences per query batch.
- **Key algorithms:** K-mer prefilter seeding; gapless diagonal scoring; Smith-Waterman extension (affine gaps); profile-profile scoring (PSI-BLAST); iterative profile construction; DIAMOND's double-indexed seed matching.
- **Datasets:** UniRef50/90 — clustered UniProt sequences for homology (https://www.uniprot.org/help/uniref); NCBI nr protein database (https://ftp.ncbi.nlm.nih.gov/blast/db/); PDB70 — representative PDB sequences (https://www.rcsb.org/downloads); Pfam — protein family HMM database (https://www.ebi.ac.uk/interpro/download/).
- **Starter repos/tools:** MMseqs2 + GPU branch (https://github.com/soedinglab/MMseqs2) — official repo with GPU support in 2025 release; DIAMOND (https://github.com/bbuchfink/diamond) — ultra-fast protein aligner (CPU baseline); CUDASW4 (https://github.com/asbschmidt/CUDASW4) — full SW on GPU for deep alignments; NVIDIA NIM MMseqs2 microservice (https://developer.nvidia.com/blog/accelerated-sequence-alignment-for-protein-design-with-mmseqs2-and-nvidia-nim/) — cloud-API GPU search.
- **CUDA libraries & GPU pattern:** Custom CUDA gapless scoring kernels (one warp per query-target pair); batched SW extension with shared memory; GPU hash table for seed look-ups; multi-GPU data parallelism across database shards; CUDA streams for overlapping I/O and compute.

---

### 3.8 Multiple Sequence Alignment (MSA) 🟡 · Active R&D
- **Deep dive:** MSA aligns N sequences simultaneously, core to phylogenetics, variant analysis, and as input to protein structure prediction. Progressive MSA (ClustalW, MAFFT PartTree) first computes an N×N pairwise distance matrix (O(N²) SW comparisons), then builds a guide tree and folds sequences in. On GPU, the distance matrix computation is embarrassingly parallel—each thread block computes one pair—yielding reported 6× speedup for the MAFFT-PartTree distance phase on GPU. CUK-Band (2024) implements center-star MSA on GPU using banded DP. For protein MSA in AlphaFold2 pipelines, MMseqs2-GPU now accelerates the iterative search that builds deep MSAs, the most time-consuming preprocessing step.
- **Key algorithms:** Progressive alignment via guide tree (Neighbor-Joining); center-star alignment reduction; banded Smith-Waterman pairwise DP; profile-profile alignment; Sum-of-Pairs scoring; MAFFT Parttree distance matrix; iterative MSA refinement.
- **Datasets:** BAliBASE — benchmark MSA reference set (https://www.lbgi.fr/balibase/); HomFam — large homologous family MSA benchmark (verify URL); OXFam benchmark (verify URL); Pfam seed alignments (https://www.ebi.ac.uk/interpro/download/).
- **Starter repos/tools:** MAFFT (https://mafft.cbrc.jp/alignment/software/) — fastest large-scale CPU MSA with GPU-accelerated distance phase prototype; CUDA-ClustalW — parallel GPU progressive MSA (https://github.com/topics/multiple-sequence-alignment); CUK-Band (https://link.springer.com/chapter/10.1007/978-981-97-5692-6_8) — 2024 CUDA center-star MSA; MMseqs2 GPU (https://github.com/soedinglab/MMseqs2) — GPU-accelerated MSA search for structure prediction pipelines.
- **CUDA libraries & GPU pattern:** One CUDA thread block per pairwise alignment (distance matrix phase); shared-memory banded DP; thrust for distance matrix sort; cuBLAS GEMM for profile-profile scoring; CUDA streams for guide-tree-ordered batch alignments.

---

### 3.9 Phylogenetic Likelihood / Tree Inference 🟡 · Active R&D
- **Deep dive:** Maximum-likelihood phylogenetic inference evaluates the Felsenstein pruning recursion—computing site likelihood at each internal node by multiplying branch transition probability matrices (4×4 or 20×20 per site, per node) up the tree—for millions of alignment columns and hundreds of tree search moves (NNI, SPR). For large trees (thousands of taxa, genome-scale alignments), the log-likelihood computation is the bottleneck and is embarrassingly parallel across alignment sites. Bayesian phylogenetics (MrBayes) runs thousands of MCMC steps each requiring full-tree likelihood evaluation; GPU acceleration reported 63× speedup vs. serial CPU by assigning each site to a thread. RAxML-NG and IQ-TREE GPU are active development targets.
- **Key algorithms:** Felsenstein pruning / Felsinstein's pruning recursion; substitution model matrix exponentiation (GTR, WAG, LG); nearest-neighbor interchange (NNI) and subtree pruning/regrafting (SPR) tree search; Metropolis-Hastings MCMC (Bayesian); bootstrap resampling.
- **Datasets:** TreeBASE — curated phylogenetic alignments and trees (https://www.treebase.org/); SILVA rRNA database — large rRNA alignment for phylogenetics (https://www.arb-silva.de/); NCBI CDD — conserved domain alignments (https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml); OpenTreeOfLife — aggregated phylogenetic data (https://opentreeoflife.github.io/).
- **Starter repos/tools:** IQ-TREE 2 (https://iqtree.github.io/) — state-of-the-art ML tree inference (GPU extension in development); RAxML-NG (https://github.com/amkozlov/raxml-ng) — fast ML inference with GPU acceleration hooks; MrBayes (https://github.com/NBISweden/MrBayes) — Bayesian inference with CUDA-accelerated site likelihood; BeagleLib (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library used by MrBayes/BEAST.
- **CUDA libraries & GPU pattern:** BeagleLib uses custom CUDA kernels for 4×4/20×20 matrix-vector products per site per node; one CUDA thread per alignment site within a likelihood pass; cuBLAS for transition matrix exponentiation; multi-GPU over tree partitions.

---

### 3.10 RNA Secondary-Structure Prediction 🟡 · Active R&D
- **Deep dive:** RNA folds into hairpins and stems governed by free-energy minimisation via the Zuker algorithm (O(n³) time, O(n²) space). For sequences >10 kb (rRNA, lncRNA), the cubic cost is prohibitive on CPU. GPU parallelism exploits the diagonal wavefront of the DP table: all cells (i,j) on the same diagonal d=j-i are independent and can be updated simultaneously by CUDA threads, similar to SW alignment. CUDA RNAfold achieves 14× speedup for sequences up to 30 kb. LinearFold reduces the complexity to O(n) using a beam-search approximation and lends itself to GPU batch processing of thousands of short RNAs in parallel.
- **Key algorithms:** Zuker free-energy minimisation (partition function DP); McCaskill partition function (base-pair probabilities); anti-diagonal wavefront parallelism; LinearFold beam-search O(n); Vienna RNA thermodynamic model; stochastic context-free grammar (SCFG) parsing.
- **Datasets:** Rfam — RNA family alignments and secondary structures (https://rfam.org/); RNAcentral — comprehensive RNA sequence database (https://rnacentral.org/); PDB RNA structures — known 3D-validated secondary structures (https://www.rcsb.org/); ArchiveII benchmark — curated RNA secondary structure data (verify URL).
- **Starter repos/tools:** CUDA RNAfold (https://www.biorxiv.org/content/10.1101/298885v1.full) — GPU-parallelised Vienna RNAfold, 14× speedup; LinearFold (https://github.com/LinearFold/LinearFold) — O(n) RNA folding with GPU batch variant; LinearAlifold (https://github.com/LinearFold/LinearAlifold) — consensus structure prediction; EternaFold (https://github.com/eternagame/EternaFold) — ML-trained folding model for GPU inference.
- **CUDA libraries & GPU pattern:** Anti-diagonal wavefront kernel (custom CUDA, shared-memory tiling of DP triangle); one warp per diagonal cell group; thrust for energy table initialization; cuFFT (not standard here, but used in some spectral RNA analyses); batch RNA folding with one CTA per sequence.

---

### 3.11 GWAS at Scale 🟡 · Active R&D
- **Deep dive:** Genome-wide association studies test millions of genetic variants (SNPs) for association with phenotypes, requiring mixed linear model (LMM) corrections to control population stratification. The computational bottleneck is constructing the genetic relatedness matrix (GRM), an N×N matrix of pairwise genomic similarity across N individuals (N ~ 500 k in UK Biobank), and then fitting LMM scores per variant. On GPU, the GRM is a large dense matrix multiply of the genotype matrix (N × M) with itself, directly accelerated by cuBLAS GEMM. GWAS-Flow (GPU) exploits this for a fast LMM approximation, and RAPIDS GPU-GWAS uses GPU-native logistic regression across all variants simultaneously.
- **Key algorithms:** Linear mixed model (LMM) with LOCO correction; genetic relatedness matrix (GRM) construction via GEMM; ridge regression / BOLT-LMM variance component estimation; logistic regression per variant; principal component analysis (PCA) for stratification correction.
- **Datasets:** UK Biobank — 500 k individuals, 800 k variants (https://www.ukbiobank.ac.uk/); GWAS Catalog — curated published associations (https://www.ebi.ac.uk/gwas/); dbGaP — controlled-access GWAS datasets (https://www.ncbi.nlm.nih.gov/gap/); gnomAD LD reference panels (https://gnomad.broadinstitute.org/).
- **Starter repos/tools:** GWAS-Flow (https://www.biorxiv.org/content/10.1101/783100) — GPU LMM-based GWAS framework; GPU-GWAS / G2WAS (https://github.com/STRIDES-Codes/GPU-GWAS) — RAPIDS-based GPU GWAS pipeline; REGENIE (https://github.com/rgcgithub/regenie) — whole-genome regression GWAS (CPU, GPU integration target); PLINK2 (https://www.cog-genomics.org/plink/2.0/) — CPU reference with GPU matrix paths via OpenBLAS/MKL.
- **CUDA libraries & GPU pattern:** cuBLAS DGEMM for GRM construction (N×M times M×N); cuSolver for matrix decomposition; RAPIDS cuDF for genotype data loading; cuML logistic regression per-SNP in batches; multi-GPU via NCCL for large N.

---

### 3.12 Single-Cell RNA-seq Analysis 🟡 · Active R&D
- **Deep dive:** Single-cell RNA-seq (scRNA-seq) produces count matrices for tens of millions of cells × 30 k genes; downstream analysis involves normalisation, highly variable gene selection, PCA, k-nearest-neighbour graph construction (O(n²) naive, accelerated by approximate nearest neighbours), UMAP / t-SNE embedding, Leiden/Louvain clustering, and differential expression. rapids-singlecell (scverse, 2024) replaces Scanpy's NumPy/SciPy backend with cuPy, cuML, and cuGraph equivalents, achieving >20× speedup for datasets up to 20 M cells. The KNN graph construction and UMAP optimisation are the most GPU-impactful steps, turning hours into minutes.
- **Key algorithms:** Normalised count transformation (scran/Seurat); PCA on sparse count matrix; approximate KNN (Faiss, HNSWLIB GPU); UMAP force-directed layout; Leiden graph clustering; negative binomial GLM for differential expression; doublet detection.
- **Datasets:** Human Cell Atlas — multi-organ scRNA-seq compendium (https://www.humancellatlas.org/); 10× Genomics public datasets (https://www.10xgenomics.com/resources/datasets); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); NCBI GEO — thousands of scRNA-seq studies (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** rapids-singlecell (https://github.com/scverse/rapids_singlecell) — drop-in GPU Scanpy replacement, cuPy/cuML/cuGraph; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmark notebooks up to 1 M+ cells; ScaleSC (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/) — GPU scRNA pipeline, 20× speed, 20 M cells on A100; Scanpy (https://github.com/scverse/scanpy) — CPU reference with GPU-aware backends.
- **CUDA libraries & GPU pattern:** cuPy sparse GEMM (count matrix ops); cuML PCA, UMAP, KNN; cuGraph Leiden/Louvain; Faiss-GPU HNSW index for ANN; cuDF for dataframe operations; multi-GPU Dask for datasets exceeding GPU RAM.

---

### 3.13 Pangenome Graph Alignment 🟡 · Active R&D
- **Deep dive:** Pangenome graphs encode the genomic variation of an entire population as a sequence graph (GFA format) rather than a single linear reference; aligning reads to this graph involves generalised DP over a DAG of paths rather than a 1D reference. The vg toolkit's graph alignment applies a generalised Smith-Waterman on the graph DAG, which is harder to parallelise than linear alignment due to irregular memory access. A 2024 SC paper demonstrated GPU-accelerated pangenome layout achieving 57.3× speedup over multi-core CPU for the ODGI layout algorithm by mapping node-force computations to GPU threads. Graph seeding via GBWT/r-index also benefits from parallelised BWT operations.
- **Key algorithms:** Generalised DAG DP alignment; GBWTgraph / r-index graph BWT; pangenome graph layout (force-directed, GPU particles); ODGI path sorting and sorting optimisation; seqwish overlap-to-graph induction; wfmash wavefront alignment for all-to-all seeding.
- **Datasets:** Human Pangenome Reference Consortium (HPRC) — 94 haplotype-resolved assemblies (https://humanpangenome.org/); 1000 Genomes Project GVCFs — variant calls for graph construction (https://www.internationalgenome.org/data); Ensembl Pangenome — multi-species graphs (https://www.ensembl.org/); PGGB tutorial data (https://github.com/pangenome/pggb).
- **Starter repos/tools:** vg (https://github.com/vgteam/vg) — comprehensive variation graph toolkit; PGGB (https://github.com/pangenome/pggb) — Pangenome Graph Builder pipeline; ODGI (https://github.com/pangenome/odgi) — GPU layout algorithms; Rapid GPU-based pangenome layout paper (https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf) — 57× speedup reference.
- **CUDA libraries & GPU pattern:** Custom CUDA force-directed layout kernels (Barnes-Hut approximation on GPU); parallel graph BFS for BWT construction; thrust for node-position sort; cuSPARSE for sparse adjacency matrix traversal; one CUDA thread per node-force computation.

---

### 3.14 Metagenomic Taxonomic Classification 🟡 · Active R&D
- **Deep dive:** Metagenomic classification assigns every sequencing read to a taxon by matching k-mers against a database of reference genomes (Kraken2 uses an exact k-mer LCA hash map; Centrifuge uses FM-index). At clinical sequencing throughput (millions of reads/minute), the hash look-up and LCA traversal become the bottleneck. MetaCache-GPU parallelises the k-mer-to-taxon hash look-up on the GPU, batching thousands of reads simultaneously, each read's k-mers queried via parallel hash table probes. Real-time GPU classification is critical for point-of-care diagnostics and pandemic surveillance.
- **Key algorithms:** K-mer exact hash matching (Kraken2 minimiser-LCA); lowest common ancestor (LCA) traversal; FM-index backward search (Centrifuge); Jaccard / MinHash distance (Mash Screen); Clark discriminative k-mer selection; GPU cuckoo hash table probing.
- **Datasets:** NCBI RefSeq complete microbial genomes — standard Kraken2 database (https://ftp.ncbi.nlm.nih.gov/refseq/); CAMI challenge benchmark datasets — simulated metagenomes (https://data.cami-challenge.org/); HMP (Human Microbiome Project) reads (https://www.hmpdacc.org/); SRA metagenomics projects (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU k-mer classification, ultra-fast; Kraken2 (https://github.com/DerrickWood/kraken2) — CPU reference, GPU hash port target; Centrifuge (https://github.com/DaehwanKimLab/centrifuge) — FM-index based, GPU extension possible; Bracken (https://github.com/jenniferlu717/Bracken) — Bayesian abundance re-estimation downstream of Kraken2.
- **CUDA libraries & GPU pattern:** Custom GPU cuckoo / robin-hood hash tables for k-mer look-up; thrust for k-mer sort and dedup; atomic CAS for concurrent hash insertions; one CUDA thread block per read, threads per k-mer; persistent kernel pattern for streaming reads.

---

### 3.15 Hi-C / 3D Genome Contact Analysis 🟡 · Active R&D
- **Deep dive:** Hi-C maps chromatin contacts genome-wide, producing sparse contact matrices of size (genome_bins × genome_bins) at 1–10 kb resolution. Downstream analysis—matrix normalisation (ICE/KR balancing), TAD boundary calling, compartment A/B classification, and loop detection—involves iterative matrix operations on matrices with 3×10⁶ bins (3 Gb of data at 1 kb). GPU acceleration of the ICE iterative correction algorithm (repeated sparse matrix-vector products) and the 2D convolution-based loop caller (HiCCUPS) is particularly impactful. ChromaFold (2024) trains a lightweight CNN on a GPU to predict 3D contact maps from 1D accessibility signals.
- **Key algorithms:** ICE / KR iterative matrix balancing (sparse MVM); eigendecomposition for A/B compartments; 1D insulation score for TAD boundary detection; HiCCUPS 2D Gaussian peak calling; Donut kernel convolution for loop enrichment; 3D polymer simulation constrained by Hi-C.
- **Datasets:** 4DN (4D Nucleome) Data Portal — Hi-C across cell types and time (https://data.4dnucleome.org/); ENCODE Hi-C datasets — cell-line 3D contacts (https://www.encodeproject.org/); GEO Hi-C studies (GSE63525 Rao 2014 etc.) (https://www.ncbi.nlm.nih.gov/geo/); OpenChromatin Consortium ATAC/Hi-C (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** Higashi (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU-accelerated hypergraph model; HiCCUPS (part of Juicer, https://github.com/aidenlab/juicer) — GPU-accelerated loop caller; ChromaFold (https://www.nature.com/articles/s41467-024-53628-0) — GPU CNN for contact prediction; cooler (https://github.com/open2c/cooler) — cool format Hi-C I/O (CPU, GPU matrix ops as next step).
- **CUDA libraries & GPU pattern:** cuSPARSE for sparse ICE/KR matrix balancing; cuBLAS for dense compartment eigendecomposition; custom 2D convolution kernels (HiCCUPS); cuDNN for CNN-based contact prediction; GPU-resident contact matrix as CSR/CSC sparse format.

---

### 3.16 Sequence Error Correction 🟡 · Active R&D
- **Deep dive:** Error correction removes sequencing artefacts before assembly. For short reads, the dominant method is k-mer spectrum analysis: k-mers below a coverage threshold are likely errors; correcting a base changes the read k-mer into a trusted one. For long reads (ONT, PacBio CLR), self-correction aligns multiple raw reads against each other and computes a consensus. CARE (https://github.com/fkallen/CARE) is a CUDA-accelerated short-read error corrector that keeps the k-mer hash table in GPU memory and processes millions of reads per second. GPU-accelerated partial-order alignment (POA) for long-read correction is implemented in GenomeWorks racon-GPU.
- **Key algorithms:** K-mer spectrum analysis (trusted-k-mer correction); Bloom filter for inexact k-mer membership; multiple sequence alignment (POA / MSA) for long-read consensus; BFC (BWT-based correction); de Bruijn graph compaction for error pruning; expectation-maximisation for error model learning.
- **Datasets:** GAGE short-read datasets — benchmark reads with known errors (http://gage.cbcb.umd.edu/); GiaB HG001-HG007 — truth-set comparison for corrected reads (https://www.nist.gov/programs-projects/genome-bottle); ONT long-read SRA archives (https://www.ncbi.nlm.nih.gov/sra); PacBio CLR SRA datasets — high-error long reads (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** CARE (https://github.com/fkallen/CARE) — CUDA short-read error corrector, GPU hash tables, Pascal+ required; racon-GPU (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — GPU POA polishing/correction; CONSENT (https://github.com/morispi/CONSENT) — long-read self-correction via local De Bruijn graphs (CPU, GPU POA target); Medaka (https://github.com/nanoporetech/medaka) — RNN-based long-read correction with GPU inference.
- **CUDA libraries & GPU pattern:** GPU hash tables with atomic CAS for k-mer counting; warp-level vote for consensus base determination; cuBLAS / custom GEMM for MSA scoring; one CUDA block per read during POA alignment; batched kernel launches across millions of reads.

---

### 3.17 CRISPR Guide Design & Off-Target Scoring 🟡 · Active R&D
- **Deep dive:** Designing effective CRISPR guide RNAs requires genome-wide off-target assessment: every 20-mer protospacer must be compared against all near-matches in the genome (allowing mismatches and bulges). For a 3 Gb human genome, this is ~300 M potential off-target sites per guide; Cas-OFFinder uses GPU to enumerate all combinations of mismatches in parallel. Scoring each off-target for actual cutting probability requires a learned model (CFD score, CNN, transformer), which GPU inference accelerates in batch over all candidate sites. FlashFry precomputes a compressed binary index enabling fast GPU-scalable off-target database look-ups.
- **Key algorithms:** Exact/approximate string matching with bounded mismatches (BFS over mismatch graph); CFD (cutting frequency determination) scoring; CNN/RNN on-target efficiency prediction; protein language model (PLM) for Cas9 variant activity (PLM-CRISPR); off-target enumeration via FM-index or hash-based inexact search.
- **Datasets:** CRISPOR benchmark — validated guide efficiencies and off-targets (https://crispor.gi.ucsc.edu/); GeCKO v2 library — genome-scale CRISPR knockout screen guides (https://www.addgene.org/pooled-library/leczkowski-gecko-v2/); Azimuth / Rule Set 2 training data — published guide efficiency datasets (verify URL); hg38/mm10 reference genomes — for off-target genome scanning (https://genome.ucsc.edu/).
- **Starter repos/tools:** Cas-OFFinder (https://github.com/snugel/cas-offinder) — GPU-accelerated off-target search, mismatch + RNA bulge enumeration; FlashFry (https://github.com/aaronmck/FlashFry) — scalable CRISPR target design with binary index; CRISPOR (https://github.com/maximilianh/crisporPaper) — comprehensive on/off-target scoring pipeline; PLM-CRISPR (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12254127/) — protein LM for Cas9 variant activity prediction with GPU inference.
- **CUDA libraries & GPU pattern:** Custom CUDA mismatch enumeration kernels (parallel BFS across mismatch positions); GPU-resident genome index in constant/global memory; cuDNN for CNN on-target scoring; batched transformer inference (ESM / PLM) on GPU; one CUDA thread per genome position.

---

### 3.18 Protein Language Model Inference 🟡 · Active R&D
- **Deep dive:** Protein language models (PLMs) such as Meta's ESM-2 (650 M–15 B parameters) learn evolutionary constraints from hundreds of millions of protein sequences; their residue embeddings encode structure, function, and mutational effects. ESMFold uses ESM-2 as a trunk to predict 3D structure without MSA, making it dramatically faster than AlphaFold2 for single-sequence predictions. GPU acceleration of the multi-head self-attention layers (O(L²) per layer for sequence length L) is essential—H100 Tensor Cores achieve >3× MFU for these GEMM workloads. Inference of 10 M UniProt proteins via ESMFold required a dedicated GPU cluster; GPU batching of mixed-length proteins with padding optimisation is the key engineering challenge.
- **Key algorithms:** Transformer multi-head self-attention (Q×K^T scaling, softmax, V aggregation); rotary positional embeddings; evoformer-style structure module; invariant point attention (IPA); masked language model (MLM) training; FlashAttention memory-efficient attention.
- **Datasets:** UniRef50/90 — training corpus for PLMs (https://www.uniprot.org/help/uniref); ESM Metagenomic Atlas — 700 M metagenomic protein structures (https://esmatlas.com/); PDB structures — validation set for ESMFold (https://www.rcsb.org/); CATH / SCOP — structural classification databases (https://www.cathdb.info/).
- **Starter repos/tools:** fair-esm (https://github.com/facebookresearch/esm) — Meta's ESM-2 and ESMFold, official CUDA inference code; EvolutionaryScale ESM3 (https://github.com/evolutionaryscale/esm) — latest multimodal protein model; ColabFold (https://github.com/sokrypton/ColabFold) — fast MSA + AlphaFold2 on GPU; xTrimoPGLM (https://huggingface.co/BonjwrAI/xTrimoPGLM-100B) — 100 B protein LM (verify URL).
- **CUDA libraries & GPU pattern:** cuDNN / Apex / FlashAttention-2 for attention; cuBLAS GEMM for feed-forward layers; Tensor Core FP16/BF16 mixed precision; multi-GPU tensor + pipeline parallelism (Megatron-LM / DeepSpeed); dynamic batching by sequence length bucket.

---

### 3.19 Variant Effect / Pathogenicity Prediction 🟡 · Active R&D
- **Deep dive:** Predicting whether a DNA variant is pathogenic combines evolutionary conservation scores (SIFT, PolyPhen), deep mutational scanning models, and increasingly large genomic foundation models (Nucleotide Transformer, Enformer, AlphaMissense). The GPU bottleneck is batched inference of deep networks over millions of variants: each SNP generates a pair of (reference, alternate) sequence context windows; the difference in model output is the predicted effect. AlphaMissense scored all 71 M possible human missense variants using GPU clusters. Enformer's convolutional-attention model for regulatory predictions runs on GPU in batch over 200 kb sequence windows.
- **Key algorithms:** Deep CNN / transformer variant effect scoring; in silico saturation mutagenesis; log-odds ratio (LOR) pathogenicity score from PLM; Enformer dilated convolutional attention; CADD-style ensemble scoring; AlphaMissense cross-chain distance potential.
- **Datasets:** ClinVar pathogenic/benign variants (https://www.ncbi.nlm.nih.gov/clinvar/); gnomAD constraint scores (https://gnomad.broadinstitute.org/); DMS deep mutational scanning atlas (https://www.mavedb.org/); HGMD (Human Gene Mutation Database) (http://www.hgmd.cf.ac.uk/).
- **Starter repos/tools:** AlphaMissense (https://github.com/google-deepmind/alphamissense) — DeepMind GPU-inferred pathogenicity for all human missense variants; Enformer (https://github.com/google-deepmind/deepmind-research/tree/master/enformer) — regulatory variant effect model; EVE / ESM-1v (https://github.com/facebookresearch/esm) — evolutionary PLM for variant scoring; Nucleotide Transformer (https://github.com/instadeepai/nucleotide-transformer) — genomic foundation model for variant effect.
- **CUDA libraries & GPU pattern:** cuDNN for CNN/transformer inference; TensorRT for deployment-time optimisation; batched pair (ref/alt) inference on GPU; Tensor Core BF16 for large models; CUDA Graphs for low-latency repeated inference over many variants.

---

### 3.20 Long-Read HiFi Assembly Overlap & Polishing 🟡 · Active R&D
- **Deep dive:** PacBio HiFi reads (10–25 kb, >99.5% accuracy) enable near-perfect de novo assemblies but the all-vs-all read overlap step—finding which reads share sequence—is computationally prohibitive: N=20 M reads requires O(N²) comparisons naively. GPU parallelism accelerates the minimiser-based seed look-up and seed chain extension across read pairs. The Darwin read overlapper GPU implementation achieved 109× speedup by storing the minimiser hash table in GPU global memory and resolving seed chains in parallel CUDA blocks. Post-overlap polishing (racon, medaka) is similarly accelerated by GPU POA and RNN inference kernels.
- **Key algorithms:** Minimiser hashing for all-vs-all overlap seeding; sparse chain DP for seed-chain scoring; partial-order alignment (POA) for consensus polishing; string graph simplification and unitig generation; haplotype phasing via heterozygous marker threading.
- **Datasets:** PacBio SMRT Human WGS (HG002/HG003/HG004 trio) (https://www.ncbi.nlm.nih.gov/sra); Vertebrate Genomes Project PacBio HiFi assemblies (https://vertebrategenomesproject.org/); GenomeArk HiFi datasets (https://genomeark.github.io/); CHM13 T2T HiFi reads (https://github.com/marbl/CHM13).
- **Starter repos/tools:** Darwin GPU overlapper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7495891/) — 109× GPU speedup for PacBio overlap; hifiasm (https://github.com/chhylp123/hifiasm) — state-of-the-art HiFi assembler; racon-GPU (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — GPU consensus polishing; Medaka (https://github.com/nanoporetech/medaka) — RNN-based polishing with GPU inference.
- **CUDA libraries & GPU pattern:** GPU-resident minimiser hash map; custom seed-chain CUDA kernels; POA DP in shared memory per thread block; cuDNN RNN for Medaka polishing; CUDA streams pipelining I/O and compute.

---

### 3.21 Structural Variant (SV) Calling 🟡 · Active R&D
- **Deep dive:** Structural variants (deletions, insertions, inversions, translocations ≥50 bp) are detected by read-support signatures: split reads, discordant pairs, and assembly-based breakpoint realignment. GPU acceleration applies at two points: (1) rapid re-alignment of split-read candidates using banded SW to pinpoint breakpoints precisely, and (2) batched deep learning inference (convolutional models on pileup images) to genotype and filter SVs. Sniffles2 uses a fast clustering algorithm for ONT/HiFi; pbsv uses local realignment. GPU-accelerated genotyping (similar to DeepVariant's image-based approach) is an emerging direction for SV filtering at population scale.
- **Key algorithms:** Split-read alignment and breakpoint clustering; discordant pair signature scoring; local assembly with miniasm/hifiasm at breakpoints; convolutional image-based genotyping (DeepSV style); SV merging across samples (SURVIVOR); genotype likelihood calculation.
- **Datasets:** GiaB SV benchmark (HG002) — gold-standard deletion/insertion/inversion calls (https://www.nist.gov/programs-projects/genome-bottle); PacBio SV benchmark (https://github.com/PacificBiosciences/sv-benchmark); 1000 Genomes SV catalog (https://www.internationalgenome.org/data); ENCODE long-read SV studies (https://www.encodeproject.org/).
- **Starter repos/tools:** Sniffles2 (https://github.com/fritzsedlazeck/Sniffles) — fast ONT/HiFi SV caller; PBSV (https://github.com/PacificBiosciences/pbsv) — PacBio SV caller; cuteSV (https://github.com/tjiangHIT/cuteSV) — clustering-based SV caller; NGSEP (https://github.com/NGSEP/NGSEPcore) — variant calling suite with GPU-amenable CNN scoring.
- **CUDA libraries & GPU pattern:** Banded SW CUDA kernels for breakpoint realignment; cuDNN CNN for SV image genotyping; batched pileup image inference; thrust for read cluster sorting; multi-GPU for population-scale SV genotyping.

---

### 3.22 RNA-seq Quantification / Pseudo-alignment 🟢 · Established
- **Deep dive:** Pseudo-alignment (kallisto, Salmon) bypasses full read alignment by mapping k-mers directly to equivalence classes of transcripts, then running the EM algorithm to estimate transcript abundances. GPU acceleration of kallisto redesigns the k-mer compatibility look-up and EM optimisation for GPU throughput: k-mer hash table queries map naturally to parallel GPU hash probes, and the EM update over millions of reads is a dense GEMV. A 2026 study ("RNA-seq analysis in seconds using GPUs," Melsted et al.) demonstrates GPU kallisto completing quantification in seconds vs. minutes on CPU. Salmon's variational Bayes EM is similarly GPU-amenable.
- **Key algorithms:** K-mer de Bruijn graph construction for transcriptome index; pseudoalignment compatibility class assignment; expectation-maximisation (EM) for abundance estimation; variational Bayes EM (Salmon); bootstrap resampling for uncertainty; quasi-mapping hash-based alignment.
- **Datasets:** GENCODE human transcriptome — reference transcript index (https://www.gencodegenes.org/); ENCODE RNA-seq FASTQs — diverse cell-type transcriptomes (https://www.encodeproject.org/); GTEx v9 — tissue RNA-seq compendium (https://gtexportal.org/); SRA RNA-seq studies (https://www.ncbi.nlm.nih.gov/sra).
- **Starter repos/tools:** kallisto GPU branch (https://github.com/pachterlab/kallisto) — GPU branch for pseudo-alignment; Salmon (https://github.com/COMBINE-lab/salmon) — quasi-mapping quantification (GPU EM target); bustools (https://github.com/BUStools/bustools) — BUS file manipulation for scRNA-seq downstream; alevin-fry (https://github.com/COMBINE-lab/alevin-fry) — fast single-cell quantification, GPU-amenable.
- **CUDA libraries & GPU pattern:** GPU hash table for k-mer to equivalence class look-up; custom EM kernel (sparse GEMV per read per EM iteration); warp-level reduction for abundance accumulation; cuSPARSE for sparse equivalence class matrices; CUDA streams for I/O and compute overlap.

---

### 3.23 Splice-Aware RNA Alignment 🟡 · Active R&D
- **Deep dive:** Splice-aware aligners (STAR, HISAT2) map RNA-seq reads across exon-exon junctions, requiring the aligner to simultaneously find the best gapped alignment across multi-exon gene models. STAR uses a suffix array for ultra-fast seeding, then extends seeds across splice junctions; HISAT2 uses a graph FM-index encoding known splice sites. GPU acceleration targets the seed-extension step (banded SW across exon pairs) and the loading/querying of the large (28 Gb for STAR human genome) suffix arrays from a GPU-resident or page-locked memory index. For long-read transcriptomics (minimap2 -ax splice), GPU wavefront alignment handles much longer reads across complex splicing.
- **Key algorithms:** Suffix array seeding (STAR); graph HISAT index with splice-site encoded BWT; banded SW for exon extension; maximum-entropy splice-site scoring; CIGAR encoding with N (intron) operations; Hamming-distance seed extension; minimap2 chaining across introns.
- **Datasets:** ENCODE RNA-seq FASTQs (https://www.encodeproject.org/); GENCODE annotation (https://www.gencodegenes.org/); SRA RNA-seq benchmarks (SEQC/MAQC) (https://www.ncbi.nlm.nih.gov/sra); GTEx tissue RNA-seq (https://gtexportal.org/).
- **Starter repos/tools:** STAR (https://github.com/alexdobin/STAR) — fastest spliced RNA aligner (GPU suffix-array querying target); HISAT2 (https://github.com/DaehwanKimLab/hisat2) — graph-index RNA aligner; minimap2 (https://github.com/lh3/minimap2) — long-read splice-aware (GPU wavefront extension target); AGAThA — GPU-accelerated guided sequence alignment for long-read mapping (verify URL).
- **CUDA libraries & GPU pattern:** Page-locked host memory for suffix array loaded by GPU; custom banded-SW CUDA kernels for exon-exon extension; GPU hash tables for splice-junction index; thrust sort for seed clustering; CUDA streams for multi-sample pipelining.

---

### 3.24 Methylation / Modified-Base Calling 🟡 · Active R&D
- **Deep dive:** Detection of DNA methylation (5mC, 5hmC) and other modifications (6mA, BrdU) from nanopore raw signal requires classifying the ionic current waveform at each potentially modified site. f5c's GPU-accelerated adaptive banded event alignment assigns signal events to reference positions using GPU-parallelised DP, then scores modification probability. ONT Remora trains small CNN/LSTM models to classify modifications directly from raw signals, with GPU inference integrated into Dorado basecalling. Galaxy-methyl achieves 3–5× GPU speedup over f5c via parallelised methylation score kernels. Accurate genome-wide 5mCG calling at 30× ONT coverage processes billions of signal samples.
- **Key algorithms:** Adaptive banded event-alignment DP (f5c); CTC basecalling with modification-aware output alphabet (Dorado/Remora); CNN/LSTM classification of signal windows per site; log-likelihood ratio modification scoring; binomial model for allele-specific methylation; bisulfite-seq Viterbi (for BS-seq comparison).
- **Datasets:** ENCODE WGBS — genome-wide bisulfite methylation reference (https://www.encodeproject.org/); Oxford Nanopore open datasets — R10.4.1 with 5mC/6mA labels (https://github.com/GoekeLab/awesome-nanopore); NCBI GEO methylation studies (https://www.ncbi.nlm.nih.gov/geo/); ENCODE long-read methylation data (https://www.encodeproject.org/).
- **Starter repos/tools:** f5c (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation calling and event alignment; Remora (https://github.com/nanoporetech/remora) — ONT modified base model training and calling; Dorado (https://github.com/nanoporetech/dorado) — integrates modification calling during basecalling on GPU; Modkit (https://github.com/nanoporetech/modkit) — modified base analysis downstream of Dorado.
- **CUDA libraries & GPU pattern:** Adaptive banded DP in CUDA shared memory; cuDNN for RNN/CNN modification classifier; persistent threads for streaming signal batches; CUDA streams for multi-read GPU pipeline; warp-level primitives for log-likelihood reduction.

---

### 3.25 Base Quality Score Recalibration (BQSR) 🟢 · Established
- **Deep dive:** BQSR models and corrects systematic machine errors in Illumina base quality scores by regressing quality on covariates: read group, cycle position, sequence context (dinucleotide), and current reported quality. It requires scanning every base of every read (~1 trillion bases for a population study) against a known-variants database, computing covariate tables, then recalibrating scores. NVIDIA Parabricks GPU BQSR reimplements GATK's BaseRecalibrator in CUDA, processing a 30× WGS BQSR step in ~6 minutes on a DGX system vs. 4–9 hours on CPU, by parallelising covariate collection across reads in GPU thread blocks.
- **Key algorithms:** Log-linear regression over quality covariates; covariate table accumulation (parallel prefix sums); known-variant masking via hash look-up; empirical quality recalibration via quantised count table; dbSNP interval tree querying.
- **Datasets:** dbSNP build 155 — known variant positions for masking (https://www.ncbi.nlm.nih.gov/snp/); GiaB known-variant VCFs (https://www.nist.gov/programs-projects/genome-bottle); Mills and 1000G indels — GATK bundle known indels (https://storage.googleapis.com/genomics-public-data/); 1000 Genomes high-coverage WGS (https://www.internationalgenome.org/data).
- **Starter repos/tools:** NVIDIA Parabricks BQSR (https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_bqsr.html) — GPU BQSR, GATK-identical output; GATK4 BaseRecalibrator (https://github.com/broadinstitute/gatk) — CPU reference implementation; DeepVariant (https://github.com/google/deepvariant) — alternative CNN caller that bypasses BQSR need; Parabricks fq2bam (https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_fq2bam.html) — integrated BWA+BQSR+dedup pipeline.
- **CUDA libraries & GPU pattern:** Parallel covariate table reduction via atomicAdd; GPU hash table for known-variant look-up; shared-memory read buffers; cuBLAS for regression solve; one CUDA thread block per read batch; CUDA streams for pipelined I/O.

---

### 3.26 GPU BAM Sorting & Deduplication 🟡 · Active R&D
- **Deep dive:** Post-alignment BAM sorting (by genomic coordinate) and duplicate read marking are canonical bottlenecks in sequencing pipelines processing terabyte-scale BAM files. Coordinate sort is a radix sort on (chromosome, position, strand) keys; GPU radix sort via thrust achieves far higher throughput than samtools CPU sort. Duplicate marking requires grouping reads by (start, end, orientation) and keeping only the highest-base-quality copy; this is a parallel hash-aggregation problem ideal for GPU hash maps. Parabricks integrates GPU sort and markdup in its fq2bam tool, running in the same 6-minute wall time as the alignment step by overlapping GPU sort with alignment I/O.
- **Key algorithms:** Radix sort by (chromosome, position) key; hash-based read grouping for duplicate detection; Picard MarkDuplicates scoring (sum base quality); UMI-aware duplicate collapsing; coordinate index (BAI/CSI) construction via parallel prefix.
- **Datasets:** 1000 Genomes WGS BAM archives (https://www.internationalgenome.org/data); TCGA cancer WGS BAM files (https://portal.gdc.cancer.gov/); ENCODE ChIP-seq BAM (https://www.encodeproject.org/); ICGC PCAWG BAMs (https://dcc.icgc.org/).
- **Starter repos/tools:** Parabricks fq2bam / bamsort (https://docs.nvidia.com/clara/parabricks/latest/) — integrated GPU BAM sort + dedup; biobambam2 (https://github.com/gt1/biobambam2) — CPU sort/dedup reference with parallel threads; Samtools (https://github.com/samtools/samtools) — CPU BAM toolkit; FastDup (https://arxiv.org/pdf/2505.06127) — speculation-and-test GPU duplicate marking.
- **CUDA libraries & GPU pattern:** thrust::sort_by_key for radix coordinate sort; GPU robin-hood hash map for duplicate grouping; thrust::reduce_by_key for per-group best-quality selection; CUDA managed memory for BAM record streaming; multi-GPU shard-and-merge pattern.

---

### 3.27 Suffix Array / BWT / FM-Index Construction 🟡 · Active R&D
- **Deep dive:** The BWT (Burrows-Wheeler Transform) and its associated FM-index enable sub-linear text search and are the backbone of short-read aligners (BWA, Bowtie2), assemblers (string graphs), and text compression. Constructing the BWT of a 3 Gb genome involves building the suffix array (SA) then applying the BWT permutation. GPU suffix array construction via parallel prefix-doubling achieves 7.9× speedup over prior GPU skew algorithms, with all n suffixes sorted simultaneously using (log n) radix-sort rounds. At metagenomics or pangenome scale (terabases), GPU construction of a BWT over millions of reads (Big-BWT / ropebwt2) is a research frontier, with CUDA CUDPP's parallel BWT used as a primitives baseline.
- **Key algorithms:** Prefix-doubling suffix array construction (DC3/skew algorithm adapted for GPU); radix sort by 2k-character rank pairs; Burrows-Wheeler permutation; FM-index backward step (LF mapping); wavelet tree construction for rank/select; Big-BWT external-memory algorithm.
- **Datasets:** GRCh38 human reference genome — 3 Gb target for BWT construction (https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/); 1000 Genomes read collections for pan-read BWT (https://www.internationalgenome.org/data); NCBI RefSeq complete microbial genomes (https://ftp.ncbi.nlm.nih.gov/refseq/); Human Pangenome sequences for pan-BWT (https://humanpangenome.org/).
- **Starter repos/tools:** GPU suffix array prefix-doubling (https://www.researchgate.net/publication/303594470) — fast parallel SA construction on GPU; ropebwt2 (https://github.com/lh3/ropebwt2) — incremental BWT construction (CPU, GPU K40 tested); CUDPP BWT (https://devblogs.nvidia.com/cutting-edge-parallel-algorithms-research-cuda/) — CUDA Data Parallel Primitives BWT; Big-BWT (https://github.com/alshai/Big-BWT) — external-memory BWT for terabase strings.
- **CUDA libraries & GPU pattern:** thrust::sort_by_key for radix-sort based SA construction; parallel prefix sums (CUB) for rank array update; GPU-resident suffix-rank arrays; custom LF-mapping kernel; persistent warp pattern for backward search.

---

### 3.28 Profile HMM (Viterbi / Forward) 🟡 · Active R&D
- **Deep dive:** Profile HMMs (pHMMs) model protein families as position-specific probability distributions; HMMER3 searches databases by applying a cascade: MSV/SSV (Multi-Segment Viterbi) filter, P7Viterbi, and Forward-Backward scoring. MSV/SSV alone consumes 72% of runtime. CUDAMPF parallelises the MSV/Viterbi recurrence across database sequences: each CUDA thread block processes one query-profile versus one database sequence, computing the N×M score matrix in shared memory. For very deep database scans (>10⁹ sequences in metagenomics), GPU pHMM search reduces days to hours.
- **Key algorithms:** MSV/SSV Multi-Segment Viterbi; P7Viterbi DP over profile-sequence grid; Forward-Backward algorithm (sum-product); Viterbi traceback; plan-7 profile HMM architecture; hit reporting with E-value calculation.
- **Datasets:** Pfam-A — 20 k protein family profiles (https://www.ebi.ac.uk/interpro/download/); UniRef50 — protein sequences for database search (https://www.uniprot.org/help/uniref); Rfam — RNA family profiles (https://rfam.org/); JGI metagenome proteins — environmental pHMM targets (https://genome.jgi.doe.gov/).
- **Starter repos/tools:** CUDAMPF (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4) — multi-tiered CUDA HMMER acceleration; HMMER3 (https://github.com/EddyLab/hmmer) — CPU reference, CUDA port target; MMseqs2 profile search (https://github.com/soedinglab/MMseqs2) — faster alternative using k-mer prefilter; GPU-HMMER speculative search (verify URL) — speculative HMMER implementation on GPU.
- **CUDA libraries & GPU pattern:** Custom shared-memory MSV/Viterbi kernel (one block per sequence); vectorised score matrix with CUDA float4; CUB warp-level max for Viterbi path; multi-GPU sequence database partitioning; CUDA streams for I/O and compute overlap.

---

### 3.29 Motif Finding in Genomic Sequences 🟡 · Active R&D
- **Deep dive:** Transcription factor motif discovery from ChIP-seq peaks searches for over-represented sequence patterns (IUPAC or position weight matrices) against a background model. Expectation-Maximisation over all N×W sequence windows (N peaks × W-k+1 positions per peak) is O(N×W×4^k) for exhaustive search; GPU parallelism assigns one thread to each window position, computing the PWM score via a parallel dot product. mCUDA-MEME achieves orders-of-magnitude speedup by distributing MEME's EM steps across GPU cores and GPU clusters. For genome-scale ChIP-seq (millions of peaks), this turns multi-day CPU runs into hours.
- **Key algorithms:** MEME expectation-maximisation over sequence windows; position weight matrix (PWM) scoring; ZOOPS/OOPS/TCM motif occurrence models; FIMO discrete log-sum-over-PWM scoring; Gibbs sampling for motif discovery; JASPAR database PWM matching.
- **Datasets:** ENCODE ChIP-seq peak BED files — thousands of TF experiments (https://www.encodeproject.org/); JASPAR 2024 — curated PWM database (https://jaspar.elixir.no/); ReMap 2022 — regulatory elements from 5 k ChIP-seq experiments (https://remap.univ-amu.fr/); GEO ChIP-seq datasets (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** CUDA-MEME / mCUDA-MEME (https://cuda-meme.sourceforge.io/homepage.htm) — GPU cluster MEME, ultrafast motif discovery; Argo_CUDA (https://pubmed.ncbi.nlm.nih.gov/29281953/) — exhaustive GPU motif discovery for large datasets; MEME Suite (https://meme-suite.org/) — reference CPU motif toolkit; HOMER (https://github.com/samtools/homer — verify URL, originally http://homer.ucsd.edu/) — CPU ChIP-seq motif enrichment tool.
- **CUDA libraries & GPU pattern:** One CUDA thread per sequence window for PWM scoring; shared-memory PWM matrix loaded once per kernel; warp-level sum for log-probability accumulation; thrust for top-k motif score extraction; batched EM outer loops with inter-GPU synchronisation.

---

### 3.30 Pangenome Graph Construction 🟡 · Active R&D
- **Deep dive:** Building a pangenome variation graph from dozens to thousands of genome assemblies requires all-to-all pairwise alignment (seqwish induces the graph from alignment PAF) and progressive normalisation (smoothxg). At the scale of the HPRC 94-haplotype human pangenome, wfmash all-to-all alignment is the dominant cost. GPU wavefront alignment (WFA) directly accelerates this: the wavefront DP's diagonal-based expansion is anti-diagonal parallel, mapping naturally to GPU threads. ODGI's GPU layout (57.3× speedup over multi-core CPU) demonstrates that force-directed node positioning—a core graph visualisation and ordering step—is highly GPU-amenable via particle-based physics simulation.
- **Key algorithms:** Wavefront alignment (WFA) for all-to-all pairwise seeding; seqwish overlap-to-graph induction; smoothxg POA-based block normalisation; ODGI force-directed layout (SGD / simulated annealing); GFA graph compaction; r-index for graph BWT.
- **Datasets:** HPRC year-1 assemblies — 94 haplotypes, human pangenome (https://humanpangenome.org/); Ensembl non-human pangenome data (https://www.ensembl.org/); Vertebrate Genomes Project assemblies (https://vertebrategenomesproject.org/); NCBI RefSeq complete genomes for bacterial pangenomes (https://ftp.ncbi.nlm.nih.gov/refseq/).
- **Starter repos/tools:** PGGB (https://github.com/pangenome/pggb) — pangenome graph builder pipeline; ODGI with GPU layout (https://github.com/pangenome/odgi) — GPU-accelerated graph layout and operations; wfmash (https://github.com/waveygang/wfmash) — WFA-based all-to-all aligner; vg (https://github.com/vgteam/vg) — comprehensive graph alignment toolkit.
- **CUDA libraries & GPU pattern:** Custom WFA CUDA kernels (anti-diagonal wavefront expansion); GPU-resident pairwise alignment matrix per genome pair; CUDA force-directed layout with node-force parallelism; thrust for wavefront front management; multi-GPU for large-scale all-to-all (N² / num_GPUs pairs per GPU).

---

---

## 4. Medical Imaging & Image Reconstruction

### 4.1 CT Reconstruction — Filtered Backprojection 🟢 · Established
- **Deep dive:** Computes a 3D volume from a set of 2D X-ray projections by applying a ramp (Ram-Lak) filter in the frequency domain to each sinogram row, then smearing each filtered projection back across the reconstructed volume. The Feldkamp-Davis-Kress (FDK) algorithm extends this to cone-beam geometry used in modern scanners and linac on-board imagers. GPU acceleration is decisive: for a 512³ volume and 1,000 projections, each backprojection step touches ~10⁹ voxel-projection pairs, making serial CPU execution intractable for real-time or high-resolution use. CUDA texture memory provides hardware-interpolated trilinear sampling of projection data at near-zero extra cost, and the entire backprojection kernel saturates GPU memory bandwidth. Achieving sub-second reconstruction at clinical resolutions requires tens of TFLOPS, available only on GPU.
- **Key algorithms:** Feldkamp-Davis-Kress FBP, Ram-Lak / Shepp-Logan ramp filter, Parker short-scan weighting, GPU ray-driven and voxel-driven backprojection, helical cone-beam FDK with Katsevich exact reconstruction.
- **Datasets:** LUNA16/LIDC-IDRI — 888 annotated thoracic CTs from TCIA (https://luna16.grand-challenge.org/); TCIA (The Cancer Imaging Archive) — large multi-collection public CT/MRI archive (https://www.cancerimagingarchive.net/); LoDoPaB-CT — low-dose CT sinogram/reconstruction pairs for benchmarking (https://zenodo.org/record/3384092); 2016 AAPM Low-Dose CT Grand Challenge — paired full-/quarter-dose CT scans (https://www.aapm.org/grandchallenge/lowdosect/).
- **Starter repos/tools:** RTK (RTKConsortium/RTK, https://github.com/RTKConsortium/RTK) — ITK-based, GPU FDK and iterative, multi-GPU, clinical DICOM-RT support; ASTRA Toolbox (https://astra-toolbox.com/, https://github.com/astra-toolbox/astra-toolbox) — MATLAB/Python/C++ GPU forward/back-projection primitives for 2D/3D, supports fan/cone/parallel; TIGRE (https://github.com/CERN/TIGRE) — MATLAB/Python CUDA toolbox with FDK plus 10+ iterative algorithms, real-dataset focus; Plastimatch (https://plastimatch.org/) — GPU FDK, deformable registration, DRR; open-source, clinical-grade C++.
- **CUDA libraries & GPU pattern:** cuFFT (ramp filter in k-space), CUDA texture memory (hardware trilinear backprojection interpolation), cuBLAS; kernel pattern: one CUDA thread per output voxel, loops over projections; multi-GPU split over projection subsets.

---

### 4.2 Iterative / Model-Based CT Reconstruction 🟡 · Active R&D
- **Deep dive:** Instead of a single analytical inversion, iterative methods repeatedly forward-project a current volume estimate, compare to measured sinogram data, then backproject the residual with statistical weighting. Penalized weighted least squares (PWLS) with total-variation (TV) or dictionary priors reduces noise by 30–50% at matched dose compared with FBP. Each outer iteration performs one full forward-projection and one backprojection — exactly the same GPU kernel bottleneck as FBP but repeated 20–200 times, making GPU mandatory for clinical throughput. ADMM decouples the data-fidelity and regularization sub-problems, enabling efficient GPU-friendly matrix-vector operations. Statistical models (Poisson likelihood for photon counts) can be incorporated for dose-optimal reconstruction.
- **Key algorithms:** SIRT, SART, OS-EM for CT, PWLS-TV, PWLS with dictionary/wavelet priors, ADMM, primal-dual splitting (Chambolle-Pock), model-based iterative reconstruction (MBIR), plug-and-play ADMM with DnCNN denoiser.
- **Datasets:** 2016 AAPM Low-Dose CT Grand Challenge (https://www.aapm.org/grandchallenge/lowdosect/); Mayo Clinic Low-Dose CT dataset (available via TCIA); LIDC-IDRI via TCIA (https://www.cancerimagingarchive.net/).
- **Starter repos/tools:** ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU primitives, build iterative loops in Python/MATLAB; TIGRE (https://github.com/CERN/TIGRE) — includes OS-TV, SART, CGLS with GPU acceleration; ODL (Operator Discretization Library, https://github.com/odlgroup/odl) — Python framework wrapping ASTRA for variational reconstruction; LEAP (https://github.com/LLNL/LEAP) — LLNL GPU-accelerated CT reconstruction library with penalized-likelihood support.
- **CUDA libraries & GPU pattern:** cuSPARSE (sparse system matrix), cuFFT, custom CUDA kernels for voxel-driven projection; outer loop on CPU, inner GPU kernel per OS subset; shared-memory tile reuse for cone-beam geometry.

---

### 4.3 MRI Reconstruction with Compressed Sensing 🟡 · Active R&D
- **Deep dive:** MRI acquires k-space (Fourier-domain) samples; compressed sensing (CS) reconstructs images from highly under-sampled k-space using sparsity priors (wavelet, total variation), enabling 4–8× scan acceleration. The core computation is a sequence of non-uniform FFTs (NUFFT/NFFT) for arbitrary k-space trajectories, followed by iterative soft-thresholding or proximal operators. NUFFT on a 3D grid at clinical resolution (~256³) involves ~10⁹ operations per iteration; GPU parallelism reduces each NUFFT to milliseconds vs. seconds on CPU, enabling real-time feedback. Multi-channel parallel imaging (SENSE, GRAPPA, PICS) adds per-coil FFTs (~32 channels), multiplying the compute by the coil count and making GPU essential.
- **Key algorithms:** SENSE, GRAPPA, non-uniform FFT (NUFFT/NFFT3), PICS (Parallel Imaging + CS), Split-Bregman / ADMM, FISTA, total variation, wavelet sparsity, k-t SENSE for dynamic MRI.
- **Datasets:** fastMRI (NYU/Facebook, https://fastmri.med.nyu.edu/ and https://github.com/facebookresearch/fastMRI) — 1,500+ knee and 6,970+ brain raw k-space MRI scans; Calgary-Campinas-359 — multi-channel brain MRI k-space (https://sites.google.com/view/calgary-campinas-dataset/); SKM-TEA (Stanford knee MRI, https://github.com/StanfordMIMI/skm-tea).
- **Starter repos/tools:** BART (Berkeley Advanced Reconstruction Toolbox, https://github.com/mrirecon/bart) — production CS-MRI tool, GPU-accelerated PICS, SENSE, NUFFT; SigPy (https://github.com/mikgroup/sigpy) — Python GPU (CuPy) MRI signal-processing and NUFFT; MIRT (Michigan Image Reconstruction Toolbox, https://github.com/JeffFessler/MIRT.jl) — Julia/MATLAB iterative reconstruction with NUFFT; PyNUFFT (https://github.com/jyhmiinlin/pynufft) — Python NUFFT with CUDA/OpenCL backends.
- **CUDA libraries & GPU pattern:** cuFFT for gridded FFT; custom CUDA NUFFT gridding kernels; cuBLAS for coil combination; per-coil FFT parallelized across CUDA streams; shared memory for gridding accumulation.

---

### 4.4 Deep-Learning MRI/CT Reconstruction 🟡 · Active R&D
- **Deep dive:** Learned reconstruction networks replace hand-crafted priors with data-driven mappings from under-sampled/degraded k-space or sinogram to fully-sampled images. End-to-end variational networks (E2E-VarNet) unroll gradient descent iterations as network layers, each with trainable sensitivity maps and refinement modules; these run entirely on GPU during both training (batch gradient descent) and inference. Training on large multi-coil raw k-space datasets (fastMRI) requires TB-scale data loading with GPU-pinned memory and mixed-precision FP16/BF16 tensor cores. Inference at 256² × 32 coils can achieve sub-100 ms per volume on a single A100, enabling real-time clinical deployment.
- **Key algorithms:** E2E-VarNet (variational network with learned sensitivity maps), unrolled ADMM-Net, deep cascade of CNN, U-Net in image domain, score-based diffusion models for MRI (DiffusionMBIR), plug-and-play denoising priors, recurrent unrolled networks.
- **Datasets:** fastMRI (https://fastmri.med.nyu.edu/) — raw multi-coil k-space, knee/brain, gold-standard reference; fastMRI+ with radiologist annotations (https://github.com/StanfordMIMI/fastMRI_plus); 2016 AAPM Low-Dose CT Challenge for CT reconstruction learning.
- **Starter repos/tools:** fastMRI baseline code (https://github.com/facebookresearch/fastMRI) — PyTorch E2E-VarNet, U-Net, evaluation scripts; BART (https://github.com/mrirecon/bart) — Deep MRI reconstruction via BART-learn module; Direct (https://github.com/directgroup/direct) — modular PyTorch framework for DL MRI reconstruction (multiple unrolled architectures); Hugging Face Medical Imaging (https://huggingface.co/datasets?search=mri) — model hub with pretrained MRI reconstruction checkpoints.
- **CUDA libraries & GPU pattern:** cuDNN (conv layers), Tensor Cores (FP16 mixed precision), PyTorch CUDA autograd; pipeline: data → pinned host memory → GPU → network forward pass → loss → backward; multi-GPU DDP training via NCCL.

---

### 4.5 PET Image Reconstruction 🟡 · Active R&D
- **Deep dive:** Positron Emission Tomography (PET) detects coincident 511 keV gamma pairs and inverts the resulting sinogram to recover tracer-distribution volumes. Maximum-likelihood expectation-maximization (MLEM) and its ordered-subsets accelerator OS-EM dominate clinically; each EM iteration requires a full system-matrix forward-projection and backprojection, accounting for detector geometry, attenuation, scatter, and randoms. Modern scanners produce list-mode data at ~10⁸ events and sinograms with ~10⁹ elements; a single MLEM iteration on a clinical dataset takes seconds on CPU, motivating GPU parallelization of the projection step across LORs (lines of response). Dynamic PET adds a time dimension, multiplying reconstruction cost by the number of frames.
- **Key algorithms:** MLEM, OS-EM (Hudson-Larkin), RAMLA, MAP-EM with Gibbs priors, PSF (point spread function) modelling, TOF-PET reconstruction (time-of-flight), list-mode ML-EM, PET/MRI joint reconstruction, penalized likelihood with MR-guided priors.
- **Datasets:** OpenNEURO PET datasets (https://openneuro.org/); TCIA PET collections (https://www.cancerimagingarchive.net/); PETRIC challenge datasets (https://github.com/SyneRBI/PETRIC); Siemens mMR phantom datasets (publicly available through STIR/SIRF).
- **Starter repos/tools:** STIR (Software for Tomographic Image Reconstruction, https://github.com/SyneRBI/STIR) — C++, OS-EM, TOF, scatter, CUDA via parallelproj; SIRF (Synergistic Image Reconstruction Framework, https://github.com/SyneRBI/SIRF) — Python/MATLAB wrapper around STIR + Gadgetron for joint PET/MR; parallelproj (https://github.com/gschramm/parallelproj) — CUDA/OpenCL GPU projectors for PET; CASToR (https://castor-project.org/) — multi-threaded/GPU-capable PET/SPECT reconstruction (verify URL).
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for LOR-parallel projection; cuBLAS for correction factors; warp-level reduction for scatter estimation; one thread per LOR in forward/back projection; CUDA streams for overlapping compute and host-device transfer.

---

### 4.6 Ultrasound Beamforming 🟢 · Established
- **Deep dive:** Delay-and-sum (DAS) beamforming reconstructs B-mode images by computing time-delayed sums of per-element receive signals for every pixel in the image grid. For a 128-element linear array, a 512×512 image, and 4,000 scan lines per second, DAS requires ~3.4 × 10¹⁰ multiply-accumulate operations per second — far beyond real-time CPU capability. GPU parallelism maps each output pixel to a CUDA thread, computes focal delays from element geometry, interpolates raw RF data, and sums across elements; a single RTX-class GPU achieves interactive frame rates for 3D volumetric beamforming. Coherence-based techniques (DMAS, CF) add per-pixel statistics but remain embarrassingly parallel.
- **Key algorithms:** Delay-and-sum (DAS), f-k migration, synthetic aperture focusing (SAFT), coherence factor (CF), DMAS (delay-multiply-and-sum), compressed sensing beamforming, Fourier domain reconstruction, adaptive minimum variance beamforming.
- **Datasets:** Plane-Wave Imaging Challenge in Medical Ultrasound (PICMUS, https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/) — RF data for beamforming evaluation; UltraSound SegLab dataset; IQ ultrasound datasets from open research groups (verify URL at creatis.insa-lyon.fr).
- **Starter repos/tools:** GPU-accelerated US beamforming repos on GitHub (search "CUDA ultrasound beamforming"); MUST (MATLAB Ultrasound Toolbox, https://www.biomecardio.com/MUST/) — reference DAS + GPU wrappers; Field II (https://field-ii.dk/) — simulation toolbox (CPU, but generates RF data for GPU DAS); k-Wave CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA) — CUDA time-domain acoustic propagation for full-wave ultrasound.
- **CUDA libraries & GPU pattern:** cuBLAS for element-weighted summation; custom CUDA kernel: one thread per image pixel, loads element positions into shared memory, vectorized delay computation via `__fmaf_rn`; texture fetch for interpolated RF data; coalesced global memory access across scan-line dimension.

---

### 4.7 Medical Image Segmentation (Deep Learning) 🟢 · Established
- **Deep dive:** Volumetric segmentation of anatomical structures (organs, tumors, vessels) in CT/MRI using encoder-decoder CNNs operates on 3D patches or whole volumes; a 512×512×200 CT volume processed in a 3D U-Net with standard batch size requires ~16 GB GPU memory and ~200 GFLOPS per forward pass. nnU-Net automatically configures patch size, batch size, network topology, and augmentation to dataset fingerprints, making it a strong universal baseline. Inference of whole-body CT (TotalSegmentator, 117 structures) completes in 20–50 s on a GPU vs. 40–50 min on CPU. Transformer architectures (Swin-UNETR) add self-attention with quadratic memory cost in sequence length, further motivating large-VRAM GPUs.
- **Key algorithms:** 3D U-Net, nnU-Net, Swin-UNETR, TransUNet, DeepMedic, V-Net, residual encoder-decoder, cascaded networks, multi-scale feature pyramid, conditional random fields (CRF) post-processing, semi-supervised pseudo-labeling.
- **Datasets:** Medical Segmentation Decathlon (http://medicaldecathlon.com/) — 10 tasks, ~2,500 volumes total; TotalSegmentator training set (Zenodo, ~1,200 CT with 117 structure labels; https://zenodo.org/record/6802614); KiTS23 kidney tumor challenge (https://kits-challenge.org/kits23/); BraTS brain tumor dataset (https://www.synapse.org/#!Synapse:syn27046444).
- **Starter repos/tools:** nnU-Net (https://github.com/MIC-DKFZ/nnUNet) — self-configuring, handles 2D/3D, GPU training and inference; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — 117-class whole-body CT segmentation, GPU inference in <1 min; MONAI (https://github.com/Project-MONAI/MONAI) — PyTorch medical AI framework with GPU-optimized transforms and network zoo; Swin-UNETR reference (https://github.com/Project-MONAI/research-contributions) — transformer-based 3D segmentation.
- **CUDA libraries & GPU pattern:** cuDNN (3D convolutions), Tensor Cores (FP16/BF16), CUDA Unified Memory for large volumes; mixed-precision training; patch-based inference with sliding window; multi-GPU via PyTorch DDP + NCCL; GPU-resident data augmentation (random flips, elastic deformations) via MONAI or NVIDIA DALI.

---

### 4.8 Deformable Image Registration 🟡 · Active R&D
- **Deep dive:** Deformable image registration (DIR) estimates a dense displacement vector field (DVF) that maps a moving image to a fixed image, minimizing an image dissimilarity metric (NCC, NMI, SSD) subject to a regularization penalty (bending energy, diffusion). Classical optimization (Demons, B-spline free-form deformation) requires hundreds of gradient descent iterations on each voxel of a dense DVF — ~10⁹ parameters for a 256³ volume — making per-iteration GPU parallelism essential. Learning-based methods (VoxelMorph) infer the DVF in a single forward pass (<1 s GPU vs. 2+ hrs ANTs CPU), but training requires large GPU memory for 3D batch processing. LDDMM (Large Deformation Diffeomorphic Metric Mapping) adds geodesic shooting on the diffeomorphism group, computable via GPU-accelerated Fourier-domain operators.
- **Key algorithms:** Demons, diffeomorphic Demons, B-spline FFD (free-form deformation), LDDMM geodesic shooting, VoxelMorph (CNN-based), TransMorph (transformer-based), symmetric diffeomorphic normalization (SyN/ANTs), normalized cross-correlation (NCC) in GPU sliding-window.
- **Datasets:** OASIS brain MRI (https://www.oasis-brains.org/) — used in Learn2Reg challenge; Learn2Reg 2022 challenge (https://learn2reg.grand-challenge.org/) — lung, brain, abdominal; DIR-Lab lung CT deformation dataset (https://dir-lab.com/); 4D-CT lung datasets for respiratory motion.
- **Starter repos/tools:** VoxelMorph (https://github.com/voxelmorph/voxelmorph) — TF/PyTorch unsupervised GPU registration; Plastimatch (https://plastimatch.org/) — GPU B-spline and Demons, DICOM-RT support; ANTs (https://github.com/ANTsX/ANTs) — gold-standard SyN (CPU-only but widely used for ground truth); TransMorph (https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) — Swin-transformer DIR, GPU-accelerated.
- **CUDA libraries & GPU pattern:** cuFFT for LDDMM geodesic shooting; custom CUDA trilinear interpolation kernel for warp; cuBLAS for regularization Hessian; memory pattern: DVF and image volumes in GPU global memory; gradient computation via cuDNN autograd.

---

### 4.9 Image Denoising & Restoration 🟡 · Active R&D
- **Deep dive:** Medical images suffer from quantum noise (CT, PET, X-ray), thermal noise (MRI), and speckle (ultrasound). Deep denoising networks (DnCNN, RED-CNN for low-dose CT, Noise2Void for unsupervised fluorescence) process 2D or 3D patches through many conv layers, requiring substantial FLOPS and large GPU memory for 3D volumetric batches. Diffusion-model denoisers now achieve state-of-the-art perceptual quality but require iterative reverse-diffusion steps (50–1,000 denoising steps), each a full forward pass through a large UNet, making GPU mandatory. Non-learning methods (NLM, BM4D) have O(N²) complexity in voxel count, acceleratable via CUDA block-matching and nearest-neighbor search.
- **Key algorithms:** DnCNN, RED-CNN (residual encoder-decoder CNN for low-dose CT), Noise2Void (N2V), Noise2Self, score-based diffusion denoising (DDPM, DDIM), BM3D/BM4D, non-local means (NLM), wavelet shrinkage, total variation denoising.
- **Datasets:** 2016 AAPM Low-Dose CT Challenge (https://www.aapm.org/grandchallenge/lowdosect/) — quarter-dose / full-dose pairs; NLST (National Lung Screening Trial) via TCIA; Fluorescence Microscopy Noise Dataset (https://github.com/juglab/n2v) — for Noise2Void; SIDD smartphone noise dataset (image domain).
- **Starter repos/tools:** N2V / Noise2Void (https://github.com/juglab/n2v) — self-supervised GPU denoising for microscopy and MRI; MONAI model zoo — RED-CNN and DnCNN for CT; DnCNN PyTorch (https://github.com/cszn/DnCNN) — GPU-accelerated Gaussian denoiser; DiffusionMBIR (https://github.com/HJ-harry/DiffusionMBIR) — score-based diffusion for CT reconstruction/denoising.
- **CUDA libraries & GPU pattern:** cuDNN (2D/3D dilated convolutions in DnCNN); custom CUDA for NLM block matching (each thread computes patch distance vs. all neighbors); cuBLAS for fully connected layers; FP16 inference via TensorRT for clinical deployment.

---

### 4.10 Super-Resolution Microscopy Reconstruction 🟡 · Active R&D
- **Deep dive:** STORM/PALM single-molecule localization microscopy (SMLM) acquires thousands of diffraction-limited frames; each fluorophore's sub-pixel position is estimated by fitting a 2D Gaussian PSF to sparse activated emitters. Processing 10⁴–10⁵ raw frames at 512×512 per acquisition demands massively parallel PSF fitting — each detected spot is independent, creating an embarrassingly parallel workload ideal for GPU. SRRF (Super-Resolution Radial Fluctuations) and SOFI (Second-Order Fluctuation Imaging) compute cross-correlations or cumulants over time stacks, with O(N·T) operations per pixel. Structured Illumination Microscopy (SIM) reconstruction requires per-orientation/phase FFT and OTF inversion, naturally parallelizable across k-space.
- **Key algorithms:** Gaussian/PSF maximum-likelihood fitting (SMLM), SRRF (radial fluctuation analysis), SOFI (cumulant imaging), SIM reconstruction (OTF inversion, Wiener filter), deconvolution (Richardson-Lucy + GPU), DECODE (deep probabilistic SMLM localization).
- **Datasets:** EPFL SMLM Challenge dataset (https://srm.epfl.ch/srm/dataset/challenge-2016/) — synthetic and real STORM/PALM frames; BioImage Archive SMLM collections (https://www.ebi.ac.uk/biostudies/bioimages); OpenMicroscopy Environment (OME-TIFF standard).
- **Starter repos/tools:** DECODE (https://github.com/TuragaLab/DECODE) — deep learning GPU SMLM localizer, orders of magnitude faster than MLE; ThunderSTORM (FIJI plugin, GPU-optional); NanoJ-SRRF (https://github.com/HenriquesLab/NanoJ-SRRF) — GPU-accelerated SRRF in ImageJ; fairSIM (https://github.com/fairSIM/fairSIM) — GPU-enabled SIM reconstruction.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for per-emitter Gaussian fitting (one warp per candidate emitter); cuFFT for SIM phase/OTF; shared memory for 7×7 PSF patch fitting; atomic additions for localization histogram accumulation.

---

### 4.11 Digital Pathology / Whole-Slide Image Analysis 🟡 · Active R&D
- **Deep dive:** Whole-slide images (WSIs) scanned at 40× magnification produce multi-gigapixel TIFF pyramids (0.5–5 GB per slide). Analysis requires GPU-accelerated tile extraction, feature extraction via pretrained CNNs (ResNet, ViT), and weakly supervised classification with attention-based multiple-instance learning (MIL). The tiling step alone for 10,000 slides produces ~500 million 224×224 patches; GPU DataLoaders must pipeline tile decompression, normalization, and augmentation to prevent GPU starvation. Spatial transcriptomics integration adds genomic annotations per spatial position, requiring co-registration of histology and sequencing data — a second-order GPU workload.
- **Key algorithms:** Attention-based MIL (CLAM, ABMIL), patch-level feature extraction (ResNet-50, ViT, UNI foundation model), stain normalization (Macenko, Vahadane), Otsu thresholding for tissue detection, tumor microenvironment clustering (DINO, MAE pretraining), survival prediction.
- **Datasets:** TCGA (The Cancer Genome Atlas) slides — access via GDC Data Portal (https://portal.gdc.cancer.gov/); CAMELYON16/17 lymph node metastasis detection (https://camelyon17.grand-challenge.org/); PanCancer Atlas WSIs via TCGA; TUPAC16 tumor proliferation.
- **Starter repos/tools:** CLAM (https://github.com/mahmoodlab/CLAM) — GPU-accelerated attention MIL for WSI classification, standard baseline; OpenSlide Python (https://openslide.org/) — library for reading WSI file formats; HistomicsTK (https://github.com/DigitalSlideArchive/HistomicsTK) — GPU-accelerated WSI analysis toolkit; UNI pathology foundation model (https://github.com/mahmoodlab/UNI) — pretrained ViT on 100k WSIs.
- **CUDA libraries & GPU pattern:** cuDNN (ResNet/ViT feature extraction per tile); DALI (GPU tile decode/augment pipeline); GPU-resident attention matrix for MIL (cuBLAS); batched tile inference with pinned memory transfer; multi-GPU feature extraction with `torch.multiprocessing`.

---

### 4.12 Optical Coherence Tomography Processing 🟡 · Active R&D
- **Deep dive:** Spectral-domain OCT acquires spectra per A-scan (axial line); reconstruction requires dispersion compensation, interpolation from wavelength to wavenumber space, and 1D FFT per A-scan. A single B-scan of 2,048 A-scans × 2,048 spectral pixels requires 2,048 FFTs of length 2,048, easily parallelizable in GPU batches. Real-time 3D OCT volumes for surgical guidance require processing ~100 B-scans/second (~4 × 10⁸ FFT points/s), achievable only with GPU. Downstream retinal layer segmentation (8 boundaries, 3D graph search) and fluid detection (intra/subretinal, FLIO) add CNN inference workload; TensorRT-optimized U-Net achieves 3.5 ms/B-scan inference.
- **Key algorithms:** Spectral-domain FFT reconstruction, dispersion compensation, k-space resampling (NUFFT), GPU-batched FFT (cuFFT), 3D graph-cut layer segmentation, deep learning retinal layer segmentation (U-Net, U-NetRT), fluid detection CNN, Doppler OCT velocity mapping.
- **Datasets:** OCTDL (https://www.nature.com/articles/s41597-024-03182-7) — 2,064 labeled OCT B-scans; Duke DME OCT dataset (https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm) — 110 annotated volumes; OCTA-500 (https://arxiv.org/abs/2012.07261) — OCT angiography volumes with labels.
- **Starter repos/tools:** OCT-Marker (https://github.com/neurodial/OCT-Marker) — annotation tool for OCT B-scans; Iowa Reference Algorithms (https://www.iibi.uiowa.edu/content/shared-software-Iowa-reference-algorithms) — graph-based segmentation (verify URL); k-Wave CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA) — relevant for photoacoustic OCT extensions; real-time OCT reconstruction demos available in NVIDIA cuFFT samples.
- **CUDA libraries & GPU pattern:** cuFFT batched 1D FFT (one FFT per A-scan, entire B-scan in one cuFFT call); custom CUDA kernel for dispersion phase correction; cuDNN for U-Net inference; CUDA streams for pipelining A-scan acquisition and reconstruction.

---

### 4.13 Photoacoustic Image Reconstruction 🟡 · Active R&D
- **Deep dive:** Photoacoustic imaging (PAI) generates ultrasound waves by pulsed laser absorption in tissue; images are reconstructed from time-series pressure data on a sensor surface. Delay-and-sum backprojection is analogous to ultrasound but in 3D; for 1,024 sensors and a 256³ volume, ~68 billion delay-and-sum operations are required per image — tractable only on GPU. Model-based iterative reconstruction solves the wave equation numerically (k-space pseudospectral method via cuFFT), enabling quantitative PAI with accurate acoustic attenuation and heterogeneous speed-of-sound modelling. Real-time 3D PA imaging for interventional guidance requires GPU throughput of multiple frames/second.
- **Key algorithms:** Delay-and-sum backprojection, time-reversal reconstruction, universal back-projection, k-space pseudo-spectral wave propagation (k-Wave), iterative model-based PA reconstruction, compressed sensing PAI, deep learning end-to-end PA reconstruction.
- **Datasets:** k-Wave simulation datasets (generated locally); USCT (Ultrasound Computed Tomography) benchmark data (verify URL); in vivo photoacoustic datasets from Nature Communications publications (open access); PASCAA challenge datasets (verify URL at photoacoustics.org).
- **Starter repos/tools:** k-Wave (http://www.k-wave.org/, CUDA C++ version at https://github.com/klepo/k-Wave-Fluid-CUDA) — industry-standard PA/US simulation and reconstruction toolbox; OpenMSOT (open multi-spectral optoacoustic tomography framework, verify URL); k-Wave MATLAB + CUDA backend for fast GPU wave simulation; PyTomography (https://github.com/lukepolson/pytomography) — Python GPU tomographic reconstruction including photoacoustic.
- **CUDA libraries & GPU pattern:** cuFFT for k-space wave propagation; custom CUDA kernel for DAS (one thread per voxel, loop over sensors); CUDA texture for time-series data interpolation; shared memory for sensor geometry LUT; multi-GPU decomposition over k-space planes.

---

### 4.14 Digital Breast Tomosynthesis 🟡 · Active R&D
- **Deep dive:** Digital breast tomosynthesis (DBT) acquires 9–25 low-dose projections over a limited angular range (~15–50°), then reconstructs thin slabs through compressed breast tissue. The limited-angle geometry makes analytical FBP unstable, so iterative methods (OS-EM, SART, ASD-POCS) with total-variation regularization dominate for artifact reduction. The breast is a low-contrast, soft-tissue object where noise and blur from the limited angle severely reduce lesion conspicuity, making statistical reconstruction critical. A single DBT volume (~800 × 700 × 60 slices at 85 µm) represents ~30 GB of raw projection data; GPU acceleration reduces OS-EM reconstruction from hours to under a minute. Deep learning methods (U-Net denoising on FBP outputs) additionally require GPU for inference.
- **Key algorithms:** FBP with limited-angle filter, OS-EM (ordered-subsets EM), SART, ASD-POCS with total variation, model-based iterative reconstruction (MBIR), DBT-specific PSF/MTF modelling, deep learning denoising and artifact reduction, mass detection CNNs.
- **Datasets:** OPTIMAM Mammography Image Database (OMI-DB, access via ICR UK); CBIS-DDSM (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — 2,620 mammograms via TCIA; VinDr-Mammo (https://physionet.org/content/vindr-mammo/1.0.0/); BCS-DBT (Duke DBT challenge dataset, https://bcs-dbt.grand-challenge.org/).
- **Starter repos/tools:** ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU forward/back-projection for arbitrary cone-beam geometry; RTK (https://github.com/RTKConsortium/RTK) — FDK and iterative DBT-capable; TIGRE (https://github.com/CERN/TIGRE) — DBT-compatible geometry; OpenDBT (verify URL) — research-focused DBT reconstruction framework.
- **CUDA libraries & GPU pattern:** cuFFT for ramp filter; CUDA voxel-driven backprojection with compressed breast geometry; texture memory for projection interpolation; limited-angle geometry stored in constant memory; ADMM inner loop GPU-resident.

---

### 4.15 Diffusion MRI & Tractography 🟡 · Active R&D
- **Deep dive:** Diffusion MRI models water diffusion anisotropy in tissue to map white-matter fiber orientations. Fitting diffusion models (DTI, DKI, NODDI) per voxel is trivially parallel — each voxel is independent — and for a 2 mm isotropic brain (~10⁵ voxels × 100 diffusion directions), batch GPU fitting is 50–100× faster than serial CPU. Constrained spherical deconvolution (CSD) solves a per-voxel fiber orientation distribution function (fODF), requiring spherical harmonic decomposition (cuBLAS) at each voxel. Probabilistic tractography (particle filtering, iFOD2) samples millions of streamlines simultaneously, with each streamline step requiring trilinear interpolation of the fODF field — massively parallel across streamlines on GPU. BEDPOSTX GPU accelerates Markov chain Monte Carlo fiber model fitting by 200× vs. CPU.
- **Key algorithms:** DTI (diffusion tensor imaging), NODDI (neurite orientation dispersion), constrained spherical deconvolution (CSD), iFOD2 probabilistic tractography, SIFT/SIFT2 streamline filtering, multi-tissue CSD, particle filtering tractography, deep learning tractography (TractSeg).
- **Datasets:** Human Connectome Project (HCP) — 1,200 subjects, 3T/7T multi-shell dMRI (https://db.humanconnectome.org/); ABCD Study dMRI (https://abcdstudy.org/); UK Biobank dMRI (https://www.ukbiobank.ac.uk/); TMS-EEG Tractography Contest (verify URL).
- **Starter repos/tools:** MRtrix3 (https://github.com/MRtrix3/mrtrix3) — gold-standard CSD, iFOD2, SIFT2, GPU-accelerated deconvolution; FSL BEDPOSTX GPU (https://fsl.fmrib.ox.ac.uk/) — GPU Bayesian fiber orientation estimation (200× speedup); TractSeg (https://github.com/MIC-DKFZ/TractSeg) — direct CNN white-matter tract segmentation; DIPY (https://github.com/dipy/dipy) — Python dMRI analysis with GPU-compatible operations.
- **CUDA libraries & GPU pattern:** cuBLAS for spherical harmonic matrix products (CSD); custom CUDA kernel for per-voxel DTI tensor fitting (SVD); CUDA random number generation (cuRAND) for probabilistic streamline sampling; texture memory for fODF field interpolation during tractography.

---

### 4.16 Functional MRI Analysis 🟡 · Active R&D
- **Deep dive:** fMRI BOLD signal analysis involves preprocessing pipelines (motion correction, slice-timing, smoothing, registration) and statistical modeling (general linear model, GLM) across hundreds of thousands of voxels and thousands of time points. ICA (independent component analysis) via MELODIC decomposes a T × V spatiotemporal matrix; for 1,200 TRs and 150,000 gray-matter voxels, the matrix-SVD and subsequent unmixing are natural cuBLAS workloads. Resting-state functional connectivity computes a V × V correlation matrix — for 100,000 voxels this is a 10¹⁰-element matrix — computed efficiently on GPU via batched inner products. Dynamic functional connectivity via sliding-window or HMM approaches further multiply this cost, requiring GPU for tractable runtimes.
- **Key algorithms:** GLM (HRF convolution and OLS/WLS per voxel), ICA (MELODIC), seed-based connectivity, graph-theoretic brain network analysis, HMM dynamic connectivity, diffusion embedding, CNN/transformer resting-state biomarker extraction, k-means parcellation on GPU.
- **Datasets:** HCP fMRI (https://db.humanconnectome.org/) — resting-state and task fMRI, 7T/3T; OpenFMRI / OpenNeuro (https://openneuro.org/) — thousands of fMRI datasets in BIDS; ABIDE autism fMRI (http://fcon_1000.projects.nitrc.org/indi/abide/); UK Biobank fMRI (https://www.ukbiobank.ac.uk/).
- **Starter repos/tools:** FSL (https://fsl.fmrib.ox.ac.uk/) — MELODIC GPU ICA, FEAT GLM, BEDPOSTX; Nilearn (https://nilearn.github.io/) — Python fMRI statistical learning with scikit-learn; BrainSpace (https://github.com/MICA-MNI/BrainSpace) — gradient analysis on GPU; fMRIPrep (https://github.com/nipreps/fmriprep) — standardized preprocessing pipeline (CUDA-accelerated ANTs registration within).
- **CUDA libraries & GPU pattern:** cuBLAS for GLM design-matrix product (V × T × T × T^-1 × T × V batched); cuSOLVER for ICA SVD; cuRAND for permutation testing; GPU histogram for parcellation; multi-GPU via PyTorch for DL resting-state classifiers.

---

### 4.17 Real-Time Intraoperative / Image-Guided Surgery 🟡 · Active R&D
- **Deep dive:** Image-guided surgery (IGS) fuses preoperative MRI/CT with intraoperative imaging (ultrasound, CBCT, fluorescence) to track surgical instruments and tumor margins in real time. The latency budget is <100 ms for tool tracking and <1 s for image update. GPU acceleration is required at every stage: intraoperative CBCT reconstruction (FDK in <1 s), deformable registration of pre/intra-operative volumes (<5 s), instrument segmentation from camera or US feed (<50 ms/frame), and DRR generation for X-ray/CT registration (<20 ms). Brain shift correction requires deformable surface registration incorporating intraoperative US and biomechanical models, solvable via GPU finite-element methods.
- **Key algorithms:** GPU FDK (CBCT intraoperative), Iterated closest point (ICP) for surface registration, GPU Demons for deformable brain-shift correction, CNN-based instrument segmentation (U-Net, YOLOv8), neural radiance fields (NeRF) for surgical scene reconstruction, Kalman filtering for tool tracking.
- **Datasets:** Cholec80 laparoscopic video dataset (https://camma.u-strasbg.fr/datasets); ReMIND2Reg 2025 brain resection multimodal dataset (https://arxiv.org/abs/2508.09649); EndoVis MICCAI challenge datasets (https://endovis.grand-challenge.org/); SurgT benchmark for surgical tool tracking.
- **Starter repos/tools:** PLUS (Public Software Library for Ultrasound Imaging Research, https://github.com/PlusToolkit/PlusLib) — real-time US acquisition/reconstruction; 3D Slicer (https://github.com/Slicer/Slicer) — OpenIGTLink for intraoperative GPU-accelerated 3D rendering; NVIDIA Clara Holoscan (https://github.com/nvidia-holoscan/holoscan-sdk) — real-time medical imaging SDK with GPU pipeline; RTK (https://github.com/RTKConsortium/RTK) — intraoperative CBCT reconstruction.
- **CUDA libraries & GPU pattern:** cuFFT + custom CUDA FDK for sub-second CBCT; cuBLAS for ICP normal-equation solve; cuDNN for instrument seg CNN inference; CUDA OpenGL interop for real-time 3D visualization overlay; NVIDIA Holoscan pipeline for <10 ms latency.

---

### 4.18 Image-Based 3D Printing / Model Generation for Surgery 🟢 · Established
- **Deep dive:** Patient-specific anatomical models for surgical rehearsal require segmenting CT/MRI volumes (GPU CNN inference), smoothing and decimating meshes (GPU geometry processing), and generating printable STL/OBJ files. For a full torso CT at 0.5 mm isotropic resolution the input volume is ~10⁹ voxels; running marching cubes on GPU (NVIDIA CUB-accelerated or CUDA-native) reduces the surface extraction step from minutes to seconds. Multi-material prints (bone, soft tissue, vessels) require multi-label segmentation and per-label mesh Boolean operations — all benefiting from GPU parallelism. Finite-element simulation for patient-specific implant design (titanium plates, aortic stents) additionally uses GPU FEM solvers.
- **Key algorithms:** GPU marching cubes (isosurface extraction), mesh smoothing (Laplacian, Taubin), Boolean mesh operations, multi-material voxel-to-mesh, TotalSegmentator CNN segmentation, GPU FEM (finite element method) for biomechanics, support structure generation for FDM printing.
- **Datasets:** TCIA body CT collections; OsteoArthritis Initiative (OAI) for knee models (https://nda.nih.gov/oai/); VerSe vertebral segmentation dataset (https://github.com/anjany/verse); TotalSegmentator dataset (https://zenodo.org/record/6802614).
- **Starter repos/tools:** 3D Slicer (https://github.com/Slicer/Slicer) — GPU-accelerated volume rendering, segmentation, STL export via SlicerRT; VTK (https://vtk.org/) — GPU-accelerated marching cubes and mesh operations; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast GPU segmentation for print-ready model prep; OpenVDB (https://www.openvdb.org/) — GPU sparse volume processing for complex anatomies.
- **CUDA libraries & GPU pattern:** CUDA marching cubes (thrust scan for compact output); cuBLAS for FEM stiffness matrix assembly; GPU ray-casting for volume rendering overlay; custom CUDA for Laplacian smoothing (per-vertex neighbor average); cuSPARSE for FEM linear system.

---

### 4.19 Motion-Compensated 4D-CT Reconstruction 🟡 · Active R&D
- **Deep dive:** 4D-CT captures respiratory motion by sorting ~4,000 projections into 10 breathing phases, then reconstructing each phase — effectively 10 independent 3D reconstruction problems with very few (~400) projections each (severe under-sampling). Simultaneous motion-compensated reconstruction (MCR) jointly estimates the reference volume and DVF by alternating between image reconstruction and non-rigid registration steps, each of which is a GPU-intensive computation. 4D-CBCT for adaptive radiotherapy is even more challenging (sparser projections, imaging dose constraints) and requires GPU-accelerated iterative reconstruction with motion-model regularization. Deep learning methods (4D Gaussian splatting, score-based priors) now push 4D-CBCT quality toward 4D-CT standards using GPU-trained priors.
- **Key algorithms:** Phase-binning and amplitude-binning 4D sorting, McKinnon-Bates 4D FDK, simultaneous MCR (PICCS, ROOSTER), GPU SART with deformable motion model, respiratory motion model (PCA-based surrogate), 4D neural radiance fields, 4D Gaussian splatting reconstruction.
- **Datasets:** DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/) — 10 cases with expert landmark pairs; TCIA 4D-CT lung radiotherapy collections; POPI model (https://www.creatis.insa-lyon.fr/rio/popi-model); CIRS dynamic lung phantom data.
- **Starter repos/tools:** RTK (https://github.com/RTKConsortium/RTK) — 4D ROOSTER motion-compensated reconstruction; ASTRA (https://github.com/astra-toolbox/astra-toolbox) — GPU projection kernels for 4D iterative; TIGRE (https://github.com/CERN/TIGRE) — 4D-capable iterative; Plastimatch (https://plastimatch.org/) — DIR integration with 4D dose.
- **CUDA libraries & GPU pattern:** GPU SART kernel for each phase subset; CUDA Demons for inter-phase registration; cuFFT for motion model PCA basis; texture memory for 4D DVF interpolation; alternating GPU compute between reconstruction and registration steps.

---

### 4.20 Dual-Energy / Spectral CT Reconstruction 🟡 · Active R&D
- **Deep dive:** Dual-energy CT (DECT) acquires sinograms at two X-ray spectra (e.g., 80 kV and 140 kV) to enable material decomposition (separating water vs. iodine basis materials, or bone vs. soft tissue). Material decomposition in projection space requires solving a 2×2 nonlinear system per sinogram bin (~10⁸ bins), each requiring Newton iteration — trivially parallel across bins on GPU. Photon-counting CT (PCCT) extends this to 4–8 energy bins, increasing the system size to 8×8 and multiplying GPU compute by 4× but enabling K-edge imaging of contrast agents. Image-domain decomposition avoids projection-space issues but requires iterative reconstruction at each energy.
- **Key algorithms:** Projection-domain material decomposition (Newton iteration per sinogram bin), image-domain material decomposition, basis-material iterative CT (ADMM), virtual monoenergetic imaging, K-edge subtraction, photon-counting spectral reconstruction, GPU splitting-based DECT ADMM.
- **Datasets:** AAPM Spectral CT challenge datasets (verify URL at aapm.org); MARS photon-counting CT datasets (https://www.marsbioimaging.com/); TCIA DECT collections; simulated DECT from published XCAT phantom.
- **Starter repos/tools:** ASTRA (https://github.com/astra-toolbox/astra-toolbox) — multi-energy projection/backprojection primitives; TIGRE (https://github.com/CERN/TIGRE) — spectral CT reconstruction; ODL (https://github.com/odlgroup/odl) — material decomposition operators; splitting-based GPU DECT paper code (https://arxiv.org/abs/1905.00934 — verify repo link in paper).
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for per-bin Newton iteration (one thread per sinogram bin, 2×2 system solve in registers); cuFFT for spectral filter; shared memory for energy-bin grouped bins; cuBLAS for joint iterative reconstruction across energy channels.

---

### 4.21 MR Fingerprinting Reconstruction 🟡 · Active R&D
- **Deep dive:** MR Fingerprinting (MRF) acquires a sequence of images with pseudorandom flip angles and TRs; each voxel's signal time course is matched to a dictionary of simulated Bloch-equation evolutions to simultaneously estimate T1, T2, and other parameters. The dictionary (10⁵–10⁶ entries × 1,000 time points) must be searched for each of ~10⁵ voxels, resulting in ~10¹¹ inner products — efficiently computed as a single large matrix-matrix product on GPU (cuBLAS GEMM). Compressed MRF combines partial k-space acquisition with low-rank tensor reconstruction, reducing the GPU workload to manageable batches. Non-Cartesian MRF trajectories require NUFFT-based reconstruction, adding a cuFFT step.
- **Key algorithms:** Bloch-simulation dictionary generation, dot-product template matching (inner product per voxel per dictionary entry as GEMM), low-rank subspace reconstruction, ADMM+MRF, physics-driven neural network MRF (DeepMRF), sequence optimization via Cramér-Rao bound.
- **Datasets:** fastMRI MRF (verify URL at fastmri.org); Cleveland Clinic MRF dataset (via IEEE DataPort, verify URL); synthetic MRF datasets generated from XCAT/BrainWeb phantoms; public multi-parametric MRI from qMRI.org (verify URL).
- **Starter repos/tools:** BART (https://github.com/mrirecon/bart) — low-rank subspace MRF reconstruction; MRzero (https://github.com/MRsimulator/MRzero) — differentiable MR sequence simulation for MRF design; PyTorch MRF dictionary matching (search GitHub for "MRF dictionary matching PyTorch"); SigPy (https://github.com/mikgroup/sigpy) — NUFFT-based MRF reconstruction.
- **CUDA libraries & GPU pattern:** cuBLAS SGEMM for dictionary matching (entire voxel×time matrix vs. dictionary×time matrix); cuFFT for NUFFT in non-Cartesian MRF; GPU-pinned memory for dictionary transfer; batched GEMM across slices via cuBLAS-Xt.

---

### 4.22 Quantitative Susceptibility Mapping (QSM) 🟡 · Active R&D
- **Deep dive:** QSM reconstructs tissue magnetic susceptibility (χ) from gradient-echo phase data in a 3D volume. The pipeline involves phase unwrapping (PUROR, ROMEO), background field removal (PDF, SHARP, VSHARP), and dipole inversion (MEDI, TKD, iLSQR, deep learning). The dipole inversion is the computational bottleneck: the forward model in k-space is a multiplication by a dipole kernel (analytically known), but inversion is ill-posed at the magic angle (cone of zero crossing). Iterative MEDI minimization requires O(100) iterations of 3D FFT + gradient updates on a 256³ volume, each costing ~30 ms GPU vs. seconds CPU. Deep learning QSM (QSMnet, xQSM) replaces MEDI with a single GPU network forward pass (<1 s).
- **Key algorithms:** Phase unwrapping (PUROR, ROMEO, BEST path), SHARP/V-SHARP background removal, MEDI (morphology-enabled dipole inversion), TKD (threshold-based k-space division), iterative least-squares (iLSQR), deep learning dipole inversion (QSMnet, xQSM), total-variation regularized inversion.
- **Datasets:** QSM Reconstruction Challenge 2.0 (https://doi.org/10.1101/2020.11.25.397695 — data on Zenodo); HCP 7T multiecho GRE data (https://db.humanconnectome.org/); AHEAD dataset (Amsterdam Ultra-high field Adult lifespan Database); BioBank UKB (https://www.ukbiobank.ac.uk/).
- **Starter repos/tools:** QSMnet (https://github.com/SNU-LIST/QSMnet) — deep learning QSM on GPU; MEDI toolbox (http://pre.weill.cornell.edu/mri/pages/qsm.html — verify URL) — MATLAB MEDI + GPU options; ROMEO (https://github.com/korbinian90/ROMEO) — fast phase unwrapping; STISuite (verify URL) — STI + QSM MATLAB toolbox.
- **CUDA libraries & GPU pattern:** cuFFT for dipole kernel multiplication in k-space per MEDI iteration; custom CUDA gradient/divergence operators for TV regularization; cuBLAS for conjugate gradient solver; memory layout: complex float32 arrays, FFT in-place.

---

### 4.23 Arterial Spin Labeling & Perfusion Imaging 🟡 · Active R&D
- **Deep dive:** Arterial spin labeling (ASL) magnetically labels water protons in arterial blood upstream and images the resulting perfusion-weighted signal difference (labeled minus control). The signal change is only 0.5–1% of background signal, requiring averaging many pairs to achieve adequate SNR; acquisition of dynamic (time-resolved) ASL with 100+ pairs at 2 mm resolution produces datasets where kinetic model fitting (single/multi-delay Buxton model) per voxel is a Bayesian inverse problem amenable to GPU parallelization. Oxford_asl/BASIL uses variational Bayes inference, parallelized across voxels on GPU. 3D multi-delay ASL combined with compressed sensing requires per-timepoint NUFFT reconstruction — same GPU bottleneck as standard CS-MRI.
- **Key algorithms:** Buxton kinetic model (single/multi-delay), pulsed ASL (PASL), pseudo-continuous ASL (pCASL), Bayesian kinetic model fitting (BASIL), variational Bayes per voxel, compressed sensing 3D dynamic ASL, T1 partial-volume correction.
- **Datasets:** HCP ASL data (https://db.humanconnectome.org/); OpenNeuro ASL datasets (https://openneuro.org/ — search "ASL"); ISMRM 2015 ASL challenge data; UK Biobank ASL pilot data.
- **Starter repos/tools:** FSL BASIL (https://fsl.fmrib.ox.ac.uk/fsl/docs/physiological/basil.html) — Bayesian ASL analysis, GPU-parallelizable voxel fits; BART (https://github.com/mrirecon/bart) — dynamic ASL CS reconstruction; ExploreASL (https://github.com/ExploreASL/ExploreASL) — multi-center ASL pipeline; SigPy (https://github.com/mikgroup/sigpy) — dynamic CS-ASL reconstruction.
- **CUDA libraries & GPU pattern:** Per-voxel independent Bayesian fit (one CUDA thread per voxel, Newton-Raphson or variational updates); cuBLAS for kinetic model matrix products; shared memory for model time-course templates; cuFFT for dynamic CS-ASL k-space reconstruction.

---

### 4.24 CT/MRI Super-Resolution 🟡 · Active R&D
- **Deep dive:** Clinical CT/MRI is acquired at sub-optimal resolution due to dose constraints, scan time, or scanner capability; super-resolution (SR) enhances images 2–4× isotropically using deep neural networks. For MRI, anisotropic SR (thick slice → isotropic) upsamples a 5 mm axial slice to 1 mm coronal/sagittal using networks trained on pairs of high/low-resolution volumes. GANs (ESRGAN-Med, CycleGAN) generate perceptually sharp images; diffusion SR models produce hallucination-free probabilistic outputs. Processing a 512×512×100 CT volume through a 3D ESRGAN requires ~500 GFLOPS per forward pass; clinical deployment requires TensorRT-optimized inference at <5 s/volume on a single GPU.
- **Key algorithms:** ESRGAN (enhanced SRGAN), 3D U-Net SR, CycleGAN for unpaired SR, diffusion model SR (SR3, DDPM), subpixel convolution (ICNR), attention U-Net SR, learned upsampling (LIIF), perceptual and adversarial losses, self-supervised SR.
- **Datasets:** HCP 7T/3T paired brain MRI (https://db.humanconnectome.org/); fastMRI (https://fastmri.med.nyu.edu/) — implicitly used for SR evaluation; IXI brain MRI dataset (https://brain-development.org/ixi-dataset/); MSD CT tasks for resolution enhancement.
- **Starter repos/tools:** MONAI SR examples (https://github.com/Project-MONAI/MONAI) — 3D medical SR reference implementations; BasicSR (https://github.com/XPixelGroup/BasicSR) — general GPU SR framework adaptable to medical; SynthSR (https://github.com/BBillot/SynthSR) — multi-contrast MRI SR/synthesis; MedSRGAN (search GitHub for "medical image super resolution GAN").
- **CUDA libraries & GPU pattern:** cuDNN (3D transposed convolutions, pixel shuffle); Tensor Cores for FP16 SR training; gradient penalty in discriminator (cuBLAS); CuPy for efficient patch extraction; TensorRT for INT8/FP16 inference deployment.

---

### 4.25 Image Harmonization Across Scanners/Sites 🟡 · Active R&D
- **Deep dive:** Multi-site MRI and CT studies suffer from scanner-induced intensity variability (field strength, vendor, protocol) that confounds downstream analysis. Statistical harmonization (ComBat) removes batch effects from extracted features; image-level harmonization (CycleGAN, CALAMITI, DeepHarmony) transforms image appearance between scanner protocols using unpaired deep learning. Training a CycleGAN on multi-site MRI (256³ volumes, ~8 GB/volume) requires large GPU VRAM and long training cycles (~100 GPU-hours); inference adds only a single forward pass. Federated learning across sites to train harmonization models without sharing raw data adds communication overhead managed by GPU-resident model weights.
- **Key algorithms:** ComBat (batch effect correction), CycleGAN (unpaired image translation), CALAMITI (multi-contrast harmonization), StarGAN-v2, DeepHarmony, contrastive harmonization (CUTS), diffusion-based scanner simulation.
- **Datasets:** ABIDE multi-site autism fMRI (http://fcon_1000.projects.nitrc.org/indi/abide/); UK Biobank imaging (https://www.ukbiobank.ac.uk/); ADNI (Alzheimer's disease, multi-scanner, https://adni.loni.usc.edu/); IXI multi-site brain MRI (https://brain-development.org/ixi-dataset/).
- **Starter repos/tools:** CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — adaptable to MRI harmonization; NiftyMIC (https://github.com/gift-surg/NiftyMIC) — multi-contrast MRI reconstruction/harmonization; NeuroComBat (https://github.com/Jfortin1/ComBatHarmonization) — statistical ComBat for neuroimaging; CALAMITI (search GitHub for "CALAMITI harmonization") — multi-contrast GPU harmonization.
- **CUDA libraries & GPU pattern:** cuDNN (CycleGAN generator/discriminator); NCCL for multi-site federated weight averaging; Tensor Cores (FP16 CycleGAN training); cuBLAS for ComBat linear algebra; GPU-accelerated data augmentation (DALI) for multi-site batch sampling.

---

### 4.26 Vessel Segmentation & Centerline Extraction 🟡 · Active R&D
- **Deep dive:** Vascular tree segmentation in CT angiography (CTA) detects tubular structures as small as 1–2 mm diameter in noisy 3D volumes; GPU-accelerated Hessian-based vesselness filters (Frangi) compute the full 3×3 Hessian eigenvalue decomposition per voxel — ~10⁶ symmetric 3×3 Eigen-decompositions for a clinical CTA. U-Net-based vessel segmentation processes the full 3D volume in overlapping patches, requiring GPU for interactive-speed inference. Centerline extraction via fast-marching or geodesic path algorithms is inherently sequential but GPU implementations exist via parallel priority queues. Clinical applications include coronary CTA FFRCT computation and aortic endograft planning.
- **Key algorithms:** Hessian-based vesselness filter (Frangi, Sato), multi-scale vesselness (scale-space), 3D U-Net vessel segmentation, V-Net, nnDetection for tubular object detection, fast-marching centerline (FMM on GPU), minimum-path centerline (Dijkstra-like), vascular topology graph extraction.
- **Datasets:** ASOCA (Automated Segmentation of Coronary Arteries, https://asoca.grand-challenge.org/); VesselMAP (cerebral vessels, verify URL); IRCAD 3D-IRCADb-01 abdominal (https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/); ImageCAS coronary artery dataset (https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).
- **Starter repos/tools:** VMTK (Vascular Modeling Toolkit, https://github.com/vmtk/vmtk) — centerline extraction, meshing, CFD integration; SlicerVMTK (https://github.com/vmtk/SlicerExtension-VMTK) — 3D Slicer integration; MONAI (https://github.com/Project-MONAI/MONAI) — 3D vessel segmentation networks; nnDetection (https://github.com/MIC-DKFZ/nnDetection) — GPU object detection for tubular structures.
- **CUDA libraries & GPU pattern:** Custom CUDA Hessian kernel (per-voxel 3×3 eigendecomposition using Jacobi iteration); cuDNN (3D U-Net inference); GPU priority queue for parallel fast-marching (thrust); shared memory for neighborhood gradient computation.

---

### 4.27 Radiomics Feature Extraction 🟡 · Active R&D
- **Deep dive:** Radiomics extracts hundreds of quantitative features (shape, first-order statistics, texture: GLCM, GLRLM, GLSZM, NGTDM) from 3D segmented ROIs in CT/PET/MRI. For a cohort of 10,000 patients with large ROIs (~10⁶ voxels each), CPU-based PyRadiomics takes 10–30 min per patient; GPU-accelerated cuRadiomics and PyRadiomics-CUDA achieve 143× speedup by parallelizing all histogram and co-occurrence matrix computations across voxels on GPU. Texture features require computing co-occurrence matrices from 26 3D neighbor directions simultaneously — each direction's computation is independent, enabling massive GPU parallelism. Radiomics biomarker discovery pipelines must process thousands of scans for statistical power.
- **Key algorithms:** GLCM (gray-level co-occurrence matrix), GLRLM (run-length matrix), GLSZM (size-zone matrix), NGTDM (neighborhood gray-tone difference matrix), first-order statistics, 3D shape descriptors, wavelet-decomposition features, multi-scale radiomics, IBSI (Image Biomarker Standardization Initiative) compliant features.
- **Datasets:** TCIA NSCLC-Radiomics (https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) — 422 lung CTs with survival; RIDER Breast MRI (via TCIA); QIN-HEADNECK (via TCIA) — head and neck RT; TCGA collections (https://portal.gdc.cancer.gov/).
- **Starter repos/tools:** PyRadiomics-CUDA (https://arxiv.org/abs/2510.02894 — code on https://github.com/mis-wut/pyradiomics-CUDA) — GPU radiomics, 143× speedup; cuRadiomics (verify URL — published in AAPM proceedings) — CUDA texture/GLCM GPU extraction; PyRadiomics CPU baseline (https://github.com/AIM-Harvard/pyradiomics) — IBSI-compliant reference; MONAI (https://github.com/Project-MONAI/MONAI) — integrated GPU radiomics pipeline.
- **CUDA libraries & GPU pattern:** Custom CUDA for co-occurrence matrix (atomic add into per-direction GLCM per thread block); shared memory for voxel neighborhood; parallel histogram across all voxels (CUB block histogram); warp-level reductions for matrix statistics.

---

### 4.28 GPU-Accelerated DRR Generation for 2D/3D Registration 🟢 · Established
- **Deep dive:** Digitally reconstructed radiographs (DRRs) simulate X-ray images from 3D CT volumes for 2D/3D registration (aligning daily X-ray portal images to planning CT). Each DRR pixel integrates CT Hounsfield units along a ray path through the volume (Siddon's ray-tracing or tri-linear ray-marching); for a 400×400 DRR from a 512³ CT, ~6.4 × 10⁸ tri-linear interpolations are needed per DRR. Intensity-based 2D/3D registration requires 50–200 DRRs per optimization iteration (~10¹¹ operations total on CPU). GPU texture memory's built-in tri-linear hardware interpolation and embarrassing parallelism (one CUDA thread per DRR pixel) make this an ideal GPU workload, achieving 100×+ speedup.
- **Key algorithms:** Siddon ray-tracing, tri-linear ray-marching (GPU texture), Splatting vs. ray-casting DRR, mutual information / NCC / gradient-magnitude similarity, stochastic gradient descent 2D/3D registration, differentiable DRR (DiffDRR), neural DRR for fast iteration.
- **Datasets:** Gold Atlas prostate CT (https://www.goldenatlasproject.com/ — verify URL); TCIA prostate/lung CTs; AAPM TG-132 test cases; clinical CBCT + kV images (institutional IRB).
- **Starter repos/tools:** Plastimatch (https://plastimatch.org/) — GPU DRR generation tool; CUDA_DigitallyReconstructedRadiographs (https://github.com/fabio86d/CUDA_DigitallyReconstructedRadiographs) — GPU DRR Python library; DiffDRR (https://github.com/eigenvivek/DiffDRR) — differentiable DRR for gradient-based 2D/3D registration; RTK (https://github.com/RTKConsortium/RTK) — GPU ray-casting for DRR.
- **CUDA libraries & GPU pattern:** CUDA 3D texture with hardware tri-linear interpolation (zero-cost); one CUDA thread per output DRR pixel; ray-step loop in kernel; constant memory for projection geometry; multiple CUDA streams for simultaneous multi-view DRR generation.

---

### 4.29 Light-Sheet Microscopy Reconstruction 🟡 · Active R&D
- **Deep dive:** Light-sheet fluorescence microscopy (LSFM / selective plane illumination, SPIM) acquires terabyte-scale datasets of developing embryos or cleared organs by illuminating a thin optical plane; the resulting multi-view 3D stacks must be: (1) registered across views/illuminations, (2) fused via multi-view deconvolution, and (3) stitched from tiled acquisitions. Multi-view deconvolution (Richardson-Lucy per view, Gaussian PSF model) on a 10³ × 10³ × 10³ sub-volume requires ~10¹² multiply-accumulates per outer iteration — GPU essential. BigStitcher (Fiji/ImageJ) uses GPU-accelerated image correlation for tile alignment and multi-GPU deconvolution for simultaneous multi-view fusion.
- **Key algorithms:** Multi-view Richardson-Lucy deconvolution (GPU), entropy-based content-weighted fusion, phase correlation tile stitching, BigStitcher alignment, iterative PSF estimation (blind deconvolution), SPIM dual-illumination fusion, 4D cell tracking (convolutional tracker).
- **Datasets:** OpenOrganelle (https://openorganelle.janelia.org/) — FIB-SEM and light-sheet neuroscience; EMBL LSFM public datasets (https://www.embl.org/); Zebrafish SPIM atlas data from Nature Methods papers; BioImage Archive LSFM collections (https://www.ebi.ac.uk/biostudies/bioimages).
- **Starter repos/tools:** BigStitcher (https://github.com/PreibischLab/BigStitcher) — GPU-accelerated LSFM stitching/fusion; CSBDeep/CARE (https://github.com/CSBDeep/CSBDeep) — deep learning LSFM denoising/restoration; N2V (https://github.com/juglab/n2v) — self-supervised GPU denoising for LSFM; DeconvolutionLab2 (https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) — multi-algorithm deconvolution with GPU.
- **CUDA libraries & GPU pattern:** cuFFT for Fourier-domain deconvolution (Richardson-Lucy in k-space); cuBLAS for view-weight matrix products; custom CUDA for phase-correlation peak detection; multi-GPU domain decomposition across z-planes for large volumes; pinned host memory for streaming TB-scale data.

---

### 4.30 Deconvolution Microscopy 🟢 · Established
- **Deep dive:** Wide-field and confocal fluorescence microscopes suffer from out-of-focus blur described by the point spread function (PSF); iterative deconvolution (Richardson-Lucy, Landweber) sharpens images by deblurring via the known or estimated PSF. Each R-L iteration requires two 3D FFT-based convolutions (forward: image×PSF; backward: ratio×PSF_flipped) on a volume as large as 2,048³; GPU cuFFT reduces each convolution from minutes to seconds. Blind deconvolution jointly estimates the PSF, requiring a second optimization variable and more iterations. Commercial instruments (Zeiss, Leica) offer GPU-accelerated deconvolution; open-source tools (CSBDeep, DeconvolutionLab2) provide GPU implementations for research.
- **Key algorithms:** Richardson-Lucy (RL) deconvolution, accelerated RL with TV regularization, Wiener deconvolution, total-variation deconvolution, blind PSF estimation (EM-algorithm), 3D FFT-based convolution via cuFFT, PSF measurement (bead-based calibration), CARE (content-aware image restoration).
- **Datasets:** BioImage Archive fluorescence microscopy datasets (https://www.ebi.ac.uk/biostudies/bioimages); EPFL Biomedical Imaging Group benchmark datasets (https://bigwww.epfl.ch/deconvolution/); ImageJ/Fiji sample datasets (https://imagej.net/); COBA microscopy benchmark.
- **Starter repos/tools:** CSBDeep/CARE (https://github.com/CSBDeep/CSBDeep) — GPU-accelerated content-aware restoration network; DeconvolutionLab2 (https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) — multi-algorithm with GPU; FlowDec (https://github.com/hammerlab/flowdec) — TF-based GPU deconvolution; N2V (https://github.com/juglab/n2v) — self-supervised denoising prior to deconvolution.
- **CUDA libraries & GPU pattern:** cuFFT 3D in-place FFT for PSF convolution; custom CUDA kernel for R-L multiplicative update (element-wise ratio); texture memory for PSF; batched cuFFT for simultaneous channel deconvolution; pinned memory for streaming large microscopy volumes.

---

### 4.31 Virtual Colonoscopy & CT Colonography 🟢 · Established
- **Deep dive:** CT colonography (CTC) acquires a supine/prone CT of the air-distended colon, then generates a virtual endoscopic fly-through rendered from inside the colonic lumen. The rendering pipeline involves: (1) colon segmentation from 512³ CT (GPU CNN inference), (2) electronic colon cleansing, (3) colonic centerline extraction (GPU fast-marching), (4) real-time volume rendering of the lumen interior (GPU ray-casting), and (5) computer-aided polyp detection (CNN classifier on rendered views or 3D patches). Real-time fly-through at 60 FPS requires GPU-accelerated volume rendering; polyp detection on an annotated virtual endoscopy dataset requires GPU training and inference.
- **Key algorithms:** GPU volume ray-casting (gradient-magnitude + Phong shading), electronic colon cleansing (thin-plate spline tagged material subtraction), centerline fast-marching, nnU-Net colon segmentation, 3D CNN polyp detection, CTC U-Net for lumen segmentation, shape index / curvedness for polyp candidates.
- **Datasets:** TCIA CT Colonography dataset (https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography); MICCAI 2018 colon challenge; ACR Radiology Lung-RADS CT dataset; NLST CT colonography subsets.
- **Starter repos/tools:** 3D Slicer (https://github.com/Slicer/Slicer) — GPU volume rendering and colon seg module; VTK (https://vtk.org/) — GPU volume ray-casting engine; MONAI (https://github.com/Project-MONAI/MONAI) — nnU-Net colon segmentation; VisIt (https://visit-dav.github.io/visit-website/) — GPU visualization for large CT volumes.
- **CUDA libraries & GPU pattern:** CUDA OptiX / OpenGL ray-casting with volume texture; custom CUDA for gradient magnitude (Sobel, per-voxel 26-neighbor); cuDNN for polyp detection CNN; CUDA 3D texture for fast trilinear lookup during fly-through; GPU-resident colon mesh for real-time rendering.

---

### 4.32 GPU-Accelerated Landmark Detection 🟡 · Active R&D
- **Deep dive:** Anatomical landmark detection localizes clinically relevant points (vertebral endplates, femoral head centers, dental cusps) in 3D medical images for registration initialization, measurement, and surgical planning. Deep learning heatmap regression (stacked hourglass, U-Net with Gaussian target maps) predicts a 3D heatmap per landmark; for 100 landmarks in a 512³ CT, the output tensor is 100 × 512³ ~ 13 GB requiring GPU. Reinforcement learning landmark detection (DQN, MARL — multi-agent RL) has each agent navigate the volume independently, with GPU parallelizing all agents simultaneously. GPU is essential for training on large 3D datasets and for achieving clinical inference speeds.
- **Key algorithms:** Stacked hourglass heatmap regression, 3D U-Net landmark heatmap, coordinate regression CNN, multi-agent RL landmark detection (MARL-DQN), iterative PatchMatch landmark search, cascade coarse-to-fine detection, anatomy-guided priors.
- **Datasets:** VerSe vertebral challenge (https://github.com/anjany/verse) — 374 CT scans with 26 vertebral landmarks; RSNA Vertebral Fracture Detection (https://rsna-vertebral-labeling-level-detection.grand-challenge.org/); CephaloNet cephalometric landmark dataset; MICCAI 2015 prostate challenge landmark dataset.
- **Starter repos/tools:** MONAI (https://github.com/Project-MONAI/MONAI) — landmark detection transforms; nnDetection (https://github.com/MIC-DKFZ/nnDetection) — GPU object/landmark detection framework; VertXNet vertebral landmark (search GitHub — verify URL); MARL landmark detection (https://github.com/amiralansari/marl-landmark — verify URL).
- **CUDA libraries & GPU pattern:** cuDNN (3D hourglass/U-Net); Tensor Cores for heatmap regression training; cuBLAS for regression head; GPU-resident augmentation (elastic deformation, CUDA Gaussian blur); batched 3D convolution for multi-landmark parallel heatmap prediction.

---

### 4.33 Real-Time MRI Reconstruction 🟡 · Active R&D
- **Deep dive:** Interventional and cardiac MRI require image reconstruction latency <100 ms to enable real-time guidance (catheter navigation, cardiac function monitoring). Online adaptive compressed sensing with sliding window or XD-GRASP (extra-dimensional GRASP) processes continuously acquired non-Cartesian k-space (radial, spiral) with GPU NUFFT and compressed sensing reconstruction running in a locked pipeline with acquisition. Gadgetron, an open-source streaming MR reconstruction framework, pipelines coil compression, NUFFT, GRAPPA, and deep learning inference on GPU with acquisition-synchronous operation. The cardiac cycle adds a gating dimension, requiring 4D (3D + cardiac phase) reconstruction at interactive speeds only feasible on GPU.
- **Key algorithms:** XD-GRASP (multi-dimensional golden-angle radial), sliding-window NUFFT, online GRAPPA, low-rank + sparse reconstruction, compressed sensing NUFFT with TV, cardiac-gated CS (XTREAM, L+S), neural network real-time reconstruction (MoDL-S), real-time MRI with physiological monitoring.
- **Datasets:** Cardiac MRI datasets from ACDC challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); CMRxRecon 2023 challenge (https://cmrxrecon.github.io/); dynamic cardiac MRI from MRXCAT simulation (verify URL); real-time fetal MRI from research groups.
- **Starter repos/tools:** Gadgetron (https://github.com/gadgetron/gadgetron) — GPU streaming MRI reconstruction server, GRAPPA/NUFFT/DL plugins; BART (https://github.com/mrirecon/bart) — GPU GRASP/CS-MRI for batch; MRzero (https://github.com/MRsimulator/MRzero) — differentiable real-time MR simulation; SigPy (https://github.com/mikgroup/sigpy) — Python NUFFT/CUDA for real-time prototyping.
- **CUDA libraries & GPU pattern:** cuFFT for NUFFT gridding; CUDA streams for acquisition-synchronous pipeline (double-buffering: acquire on CPU/scanner, reconstruct on GPU simultaneously); cuDNN for online DL inference; CUDA thrust for dynamic radial k-space sorting; multi-GPU for parallel cardiac phase reconstruction.

---

---

## 5. Radiation Therapy & Medical Physics

### 5.1 Monte Carlo Dose Calculation 🟡 · Active R&D
- **Deep dive:** Monte Carlo (MC) simulation tracks individual particle histories through patient CT geometry, sampling physics interactions (Compton scatter, pair production, photoelectric effect) stochastically. Clinical accuracy requires ~10⁸–10⁹ particle histories; on CPU (e.g., EGSnrc, MCNP), a single prostate plan takes hours. GPU MC exploits the independence of particle histories: each CUDA thread tracks one particle, with warp-level divergence managed by sorting particles by material. GPU codes (DPM-GPU, gDPM, Acuros, FRED) achieve 100× speedups over single-CPU. The primary GPU challenge is divergent execution paths when different threads take different interaction branches and managing the CT voxel geometry lookup efficiently in constant/texture memory.
- **Key algorithms:** Condensed-history electron transport, class-II MC (Berger/ICRU), photon interaction sampling (Klein-Nishina, photoelectric), bremsstrahlung production, Russian roulette / splitting variance reduction, GPU divergence management (particle sorting by material), macro-MC for ultra-fast TPS dose.
- **Datasets:** IAEA benchmark photon beam data (https://www.iaea.org/resources/databases/iaea-photon-electron-interaction-data-library); AAPM TG-119 IMRT QA phantom dataset; clinical patient CT + plan DICOM from departmental archives (IRB-required); CIRS anthropomorphic phantom CT datasets.
- **Starter repos/tools:** EGSnrc (https://github.com/nrc-cnrc/EGSnrc) — reference CPU MC for photon/electron, GPU extensions in literature; GATE 10 (https://github.com/OpenGATE/opengate) — Python-based Geant4 wrapper, GPU-capable via Geant4 MT; gDPM / DPM-GPU (verify URL, published by Ma et al.) — GPU photon/electron MC dose; FRED (https://www.fredonline.eu/) — GPU MC for proton/ion therapy (verify URL); MC-GPU (https://github.com/adler-j/GPUMC) — CUDA GPU photon MC, open source.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for particle transport loop (one thread per particle history); physics tables in constant/texture memory; warp-divergence reduction via material sorting before interaction step; atomic adds to dose voxel array; batch history generation via cuRAND.

---

### 5.2 Radiotherapy Treatment-Plan Optimization 🟡 · Active R&D
- **Deep dive:** IMRT/VMAT plan optimization solves a large-scale constrained optimization: minimize dose to OARs subject to PTV coverage constraints, with variables being beam aperture shapes or fluence maps. The dose-influence matrix D (N_voxels × N_beamlets, typically 10⁶ × 10⁴) must be computed and stored on GPU; the iterative optimizer (gradient descent, IPOPT, L-BFGS) performs repeated sparse matrix-vector products (D·x) per iteration. GPU SpMV reduces each DMAT-vector product from seconds to milliseconds, enabling real-time adaptive re-optimization. Biological-effect optimization (TCP/NTCP) and robust optimization over uncertainty scenarios further multiply the compute by the number of scenarios (~50–100 for robust plans).
- **Key algorithms:** Fluence-map optimization (quadratic programming, L-BFGS), direct aperture optimization (DAO), volumetric modulated arc therapy (VMAT) optimization, robust optimization (minimax), biological TCP/NTCP optimization, multi-criteria optimization (Pareto front navigation), deep learning dose prediction (U-Net).
- **Datasets:** OpenKBP (knowledge-based planning) dataset (https://github.com/ababier/open-kbp) — 340 head-and-neck IMRT plans; TCIA RT datasets; PlanIQ (verify URL); AAPM TG-263 structure naming dataset; OpenTPS test datasets.
- **Starter repos/tools:** matRad (https://github.com/e0404/matRad) — open-source MATLAB treatment planning, photon/proton/carbon; pyRadPlan (https://github.com/e0404/pyRadPlan) — Python interoperable extension of matRad; CERR (https://github.com/cerr/CERR) — MATLAB comprehensive RT research platform with DICOM-RT; OpenTPS (https://opentps.org/) — open-source Python/GPU treatment planning system (verify URL).
- **CUDA libraries & GPU pattern:** cuSPARSE (SpMV for D·fluence products); cuBLAS (OAR/PTV dose-volume histogram computation); CUDA warp-level reductions for DVH statistics; GPU-resident D-matrix in CSR format; multi-GPU for scenario-parallel robust optimization.

---

### 5.3 Proton & Heavy-Ion Therapy Dose 🟡 · Active R&D
- **Deep dive:** Proton and carbon-ion beams deposit dose with a sharp Bragg peak distal to the target, enabling sparing of surrounding normal tissue. Analytical dose engines (pencil-beam algorithm, PBA) convolve pencil-beam kernels with CT stopping-power maps; GPU parallelizes the per-spot convolution across the ~10⁴ spots in a plan, reducing a full plan from minutes to seconds. Full Monte Carlo (FRED, TOPAS, GATE) simulates hadronic physics including nuclear fragmentation (dominant for carbon ions), requiring GPU for clinical throughput. Range uncertainty (due to CT Hounsfield-unit–to–stopping-power conversion) is managed by robust optimization over 3 mm / 3.5% scenarios, multiplying GPU compute requirements.
- **Key algorithms:** Pencil-beam algorithm (PBA), analytical Bragg-peak model, GPU MC (FRED, MOQUI, gPMC), nuclear fragmentation transport (Geant4-TOPAS), LET (linear energy transfer) calculation, RBE (relative biological effectiveness) weighting, multi-field optimization, robust proton optimization.
- **Datasets:** TOPAS/GATE benchmark proton beam data; clinical proton CT datasets (develop via institution); TCIA proton treatment response datasets; POPI model for proton treatment planning (https://www.creatis.insa-lyon.fr/rio/popi-model).
- **Starter repos/tools:** FRED (https://www.fredonline.eu/) — GPU fast MC for ions, clinical-grade, DICOM-RT input; MOQUI (https://github.com/mghro/moquimc) — GPU proton MC for quick dose recalculation (MGH, open source); OpenTOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — open fork of TOPAS, Geant4-based proton MC; matRad (https://github.com/e0404/matRad) — analytic proton dose engine with GPU-parallel spot convolution.
- **CUDA libraries & GPU pattern:** Custom CUDA for per-spot pencil-beam convolution (one thread per spot × voxel pair); cuFFT for convolution in k-space; texture memory for CT stopping-power map; cuRAND for MC sampling; CUDA atomic adds for dose histogram accumulation.

---

### 5.4 Collapsed-Cone / Superposition-Convolution Dose 🟢 · Established
- **Deep dive:** Superposition-convolution (SC) dose computation convolves Monte Carlo-derived photon energy-deposition kernels (polyenergetic dose-spread arrays, DSAs) with the TERMA (total energy released per unit mass) computed from CT. Collapsed-cone convolution (CCC) discretizes the kernel into angular cones and propagates dose along ray paths at each angle. For a 512³ CT volume and ~400 cone directions, each cone sweep is a 1D scan along the CT in that direction — embarrassingly parallel across cones and voxels. GPU parallelization across cone directions and voxel planes reduces a CCC plan from ~10 min to <10 s. This algorithm underlies most commercial photon dose engines (Eclipse AXB, RayStation).
- **Key algorithms:** Superposition/convolution with poly-energetic DSA kernels, collapsed-cone convolution (CCC), anisotropic analytical algorithm (AAA), Acuros XB (linear Boltzmann transport), TERMA ray-tracing (Siddon/ray-voxel), heterogeneity correction via density scaling.
- **Datasets:** AAPM TG-105 test cases (heterogeneous media dose benchmarks); IROC lung phantom CT + dosimetry data; TCIA clinical photon planning datasets; CIRS IMRT verification phantom data.
- **Starter repos/tools:** matRad (https://github.com/e0404/matRad) — photon pencil-beam + CC dose engine; Plastimatch (https://plastimatch.org/) — GPU-accelerated dose engine components; CERR (https://github.com/cerr/CERR) — dose calculation framework; open AAPM TG-105 reference datasets with comparison code (verify URL at aapm.org).
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for TERMA ray-trace (Siddon's algorithm, one thread per ray); cone-direction parallel sweep in CCC (one CUDA block per cone direction); shared memory for density strip along current cone ray; reduction for energy normalization.

---

### 5.5 Deformable Dose Accumulation & Adaptive Radiotherapy 🟡 · Active R&D
- **Deep dive:** Adaptive radiotherapy (ART) adjusts the treatment plan during a course of fractions based on daily imaging (CBCT), requiring: (1) daily GPU CBCT reconstruction, (2) deformable image registration (DIR) between planning CT and daily image, (3) deformable warping of the dose distribution via the DVF to accumulate physically meaningful total dose. DIR and dose warping on a 512³ volume require iterative GPU Demons/B-spline followed by trilinear interpolation of the 3D DVF — each voxel's dose is mapped to its deformed position. Online ART workflows (MR-Linac) must complete all steps in <5 min, achievable only with GPU. Uncertainty in DIR propagates to dose uncertainty, motivating ensemble DIR and probabilistic dose accumulation on GPU.
- **Key algorithms:** Diffeomorphic Demons DIR, B-spline FFD, VoxelMorph for daily DIR, trilinear DVF warp for dose accumulation, summation-of-deformed-doses vs. energy-mass-transfer method, DIR uncertainty quantification, plan re-optimization on adapted anatomy.
- **Datasets:** TCIA CT-on-rails / CBCT datasets; DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/); AAPM TG-132 DIR test cases; CREATIS deformable lung phantom (https://www.creatis.insa-lyon.fr/).
- **Starter repos/tools:** Plastimatch (https://plastimatch.org/) — GPU B-spline DIR + dose warping, DICOM-RT; VoxelMorph (https://github.com/voxelmorph/voxelmorph) — DL DIR for daily CBCT to CT; CERR (https://github.com/cerr/CERR) — deformable dose accumulation pipeline; pyRadPlan (https://github.com/e0404/pyRadPlan) — adaptive plan re-optimization.
- **CUDA libraries & GPU pattern:** GPU Demons iterative kernel (per-voxel force computation + Gaussian smoothing via cuFFT); custom CUDA trilinear warp for dose mapping; cuBLAS for B-spline coefficient computation; CUDA atomic adds for accumulated dose histogram.

---

### 5.6 GPU Boltzmann Transport (Deterministic Dose) 🟡 · Active R&D
- **Deep dive:** The linear Boltzmann transport equation (LBTE) describes radiation transport deterministically: it tracks the fluence distribution of particles as a function of position, direction, and energy without stochastic noise. Solving it on a clinical 6-DoF phase-space grid (x, y, z, θ, φ, E) discretized at clinical resolution yields a system with ~10⁹–10¹⁰ unknowns; iterative solvers (source iteration, diffusion synthetic acceleration) require GPU to be tractable. Acuros XB (Varian Eclipse) implements a GPU-accelerated LBTE solver that outperforms superposition-convolution in heterogeneous tissue. The 3D_RZ geometry and electron transport coupling make Boltzmann dose accurate in lung, bone/tissue interfaces where MC is preferred but slow.
- **Key algorithms:** Discrete ordinates (Sₙ) method, source iteration (SI), diffusion synthetic acceleration (DSA), multi-group energy discretization, linear discontinuous spatial FEM, Legendre polynomial scattering expansion, Acuros XB algorithm, coupled photon-electron LBTE.
- **Datasets:** AAPM TG-105 lung benchmark; IROC heterogeneity phantom datasets; IAEA photon cross-section library; Acuros XB validation datasets from Varian white papers (publicly documented).
- **Starter repos/tools:** OpenMC (https://github.com/openmc-dev/openmc) — open MC but with deterministic capabilities; Attila (commercial) and Denovo (https://github.com/ORNL-CEES/Exnihilo — verify URL) — deterministic transport; AHOTN (analytical and hybrid ordinates) codes (verify URL); GPU-accelerated Sₙ solvers in nuclear engineering literature (search "GPU Sn transport CUDA").
- **CUDA libraries & GPU pattern:** cuSPARSE for angular flux sweep (upwind differencing); cuFFT not applicable; custom CUDA kernel for inner transport sweep (spatial + angular decomposition); GPU memory: angular flux tensor in global memory, scattering source in shared memory; wavefront parallelism across spatial cells.

---

### 5.7 Brachytherapy Dose & Source Modeling 🟢 · Established
- **Deep dive:** Brachytherapy (BT) delivers dose from radioactive sources (Ir-192 HDR, Pd-103, I-125) implanted inside or adjacent to the tumor. TG-43 formalism computes dose analytically from tabulated radial and anisotropy functions per source dwell position; for an HDR plan with 50 dwell positions in a prostate implant, GPU parallelization across (source, voxel) pairs reduces plan calculation from seconds to milliseconds. Beyond TG-43, model-based dose algorithms (MBDCA) — Acuros BT, Monte Carlo — account for tissue heterogeneity and inter-source shielding, requiring the same GPU particle-transport infrastructure as external-beam MC. Real-time BT dose visualization on TRUS/fluoroscopy feed requires GPU latency <100 ms.
- **Key algorithms:** TG-43 dose formalism (radial dose function, anisotropy function), superposition of point-source kernels, MBDCA (model-based dose calculation algorithm), MC for BT (Geant4-TOPAS, EGSnrc BrachyDose), shielding correction for multi-source, real-time dose overlay on TRUS imaging.
- **Datasets:** AAPM TG-43 consensus datasets (radial/anisotropy tables — https://www.aapm.org/pubs/reports/); TCIA prostate BT CT datasets; ESTRO ACROP BT guideline test cases; BrachyView QA data (verify URL).
- **Starter repos/tools:** BrachyDose (via EGSnrc, https://github.com/nrc-cnrc/EGSnrc) — EGSnrc BT MC user code; TOPAS-BrachyDose (https://github.com/topasmc) — Geant4-based BT MC; PyTG43 (https://github.com/GregSal/PyTG43 — verify URL) — Python TG-43 dose calculator; matRad BT module (https://github.com/e0404/matRad) — MATLAB BT dose and optimization.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for TG-43 dose (grid of threads covering output voxels; inner loop over source dwell positions; tables in constant memory); cuRAND for MC BT photon sampling; texture memory for 2D anisotropy function tables; warp-level reduction for summing source contributions.

---

### 5.8 Linac QA & Machine Performance Assessment 🟢 · Established
- **Deep dive:** Linear accelerator (linac) quality assurance measures beam output, flatness, symmetry, and MLC leaf positions from portal dosimetry images or log files. GPU acceleration is applied in three areas: (1) rapid gamma-index computation comparing measured vs. planned dose distributions (3D gamma on a 200³ dose grid requires ~10⁹ distance searches), (2) EPID (electronic portal imaging device) image-based dose reconstruction converting 2D portal images to 3D dose via a GPU MC kernel, and (3) machine learning prediction of machine failures from large log-file datasets (training on GPU). Automated daily QA with immediate GPU-based analysis enables real-time feedback before the treatment session.
- **Key algorithms:** Gamma-index dose comparison (3D, distance-to-agreement + dose-difference), EPID portal dose reconstruction (MC kernel convolution on GPU), MLC leaf-gap analysis, Winston-Lutz test automation, trajectory log analysis, ML anomaly detection on linac logs.
- **Datasets:** AAPM TG-119 IMRT QA test cases; AAPM TG-218 tolerance criteria datasets; TCIA linac log datasets (verify URL); Varian/Elekta log file datasets from published QA studies; OpenMedPhys (https://github.com/jrkerns/awesome-medphys) reference datasets.
- **Starter repos/tools:** Pylinac (https://github.com/jrkerns/pylinac) — Python linac QA automation (image analysis, log files); PRIMO MC linac simulator (https://www.primoproject.net/ — verify URL); Plastimatch (https://plastimatch.org/) — GPU-accelerated gamma index; matRad (https://github.com/e0404/matRad) — plan-vs-measurement comparison.
- **CUDA libraries & GPU pattern:** Custom CUDA for 3D gamma index (each thread manages one reference-dose point, searches neighbor distance sphere in delivered dose volume); texture memory for delivered dose field; cuBLAS for log-file ML feature matrix; warp-level min-reduction for closest distance search.

---

### 5.9 Gamma-Index Dose Comparison 🟢 · Established
- **Deep dive:** The gamma index (γ) at each reference point searches for the minimum normalized Euclidean distance in combined dose-difference / distance-to-agreement (DTA) space over all evaluated points: γ(r_ref) = min_r √[(Δd/Δd_crit)² + (Δr/DTA_crit)²]. For 3D clinical distributions at 2 mm DTA and 2% dose criterion, the exhaustive search over a 200³ evaluation grid from each of 200³ reference points is O(N⁶) naively, reduced to O(N³ × K) by limiting the search radius. GPU parallelizes this: one thread per reference point, searches a kernel of neighbor evaluated points; with shared-memory tiling this achieves 100–1,000× speedup over CPU, enabling sub-second 3D gamma on clinical GPUs. This is critical for patient-specific IMRT/VMAT pre-treatment verification.
- **Key algorithms:** 3D gamma index exhaustive search (distance-limited), fast gamma approximations (1D cross-plane), GPU kernel tiling for shared-memory neighbour caching, global gamma pass-rate statistics, normalized agreement testing (NAT), χ (chi) factor dose comparison.
- **Datasets:** AAPM TG-218 patient-specific IMRT QA reference data; plan+measurement DICOM pairs from departmental QA systems; IROC-Houston phantom dose datasets; linac EPID measurement datasets.
- **Starter repos/tools:** Pymedphys (https://github.com/pymedphys/pymedphys) — Python gamma index, DICOM dose tools; Plastimatch (https://plastimatch.org/) — GPU gamma-index C++ library; gamma-index GPU (https://pubmed.ncbi.nlm.nih.gov/21317484/ — verify GitHub from paper) — UCSD GPU gamma; OpenGATE (https://github.com/OpenGATE/opengate) — includes dose comparison utilities.
- **CUDA libraries & GPU pattern:** One CUDA thread per reference point; shared-memory tile of evaluated dose grid (tiled by distance radius); minimum reduction in registers; atomic min for tie-breaking; cuBLAS for vectorized pass/fail statistics across patient cohort.

---

### 5.10 Secondary Cancer Risk & Stray-Dose Monte Carlo 🔴 · Frontier/Theoretical
- **Deep dive:** Radiotherapy delivers dose not only to the target but also to distant organs via stray radiation (leakage, scatter, neutrons from proton therapy nuclear interactions), creating secondary cancer risk. Stray-dose is ~3–4 orders of magnitude lower than target dose, requiring 10¹¹–10¹²+ particle histories per calculation for statistical precision — intractable even on GPU without variance reduction (splitting, forced detection, geometry importance). GPU-based stray-dose MC requires importance sampling and photon-electron transport over the full body habitus beyond the treated field, rarely implemented in commercial systems. Secondary neutron fluence from proton therapy high-Z nozzle elements requires hadronic physics in Geant4/TOPAS, adding GPU parallelization complexity.
- **Key algorithms:** Forced detection variance reduction, splitting/Russian roulette, photonuclear interaction cross-sections, hadronic interaction model (INCL, BERT) for secondary neutrons, whole-body geometric phantom integration (ICRP110 voxel phantoms), Lifetime Risk Model (BEIR VII) convolution with dose distribution.
- **Datasets:** ICRP 110 voxel phantoms (adult male/female, https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110); NIST photon cross-section databases (https://www.nist.gov/pml/xcom-photon-cross-sections); secondary dose measurements from literature; TCIA proton therapy planning CTs.
- **Starter repos/tools:** TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — full hadronic transport, stray-dose extensions; GATE 10 (https://github.com/OpenGATE/opengate) — neutron transport, out-of-field dose scoring; EGSnrc (https://github.com/nrc-cnrc/EGSnrc) — photon/electron with advanced variance reduction; PHITS (https://phits.jaea.go.jp/ — verify URL) — hadronic + neutron transport for radiation protection.
- **CUDA libraries & GPU pattern:** Custom CUDA hadronic transport kernel (one thread per particle, nested interaction sampling loop); constant memory for cross-section tables; variance reduction handled per-thread (splitting → thread forking via particle stack on GPU); cuRAND for correlated sampling sequences.

---

### 5.11 Microdosimetry & Track-Structure Simulation 🔴 · Frontier/Theoretical
- **Deep dive:** Microdosimetry and nanodosimetry characterize the stochastic distribution of energy deposition events in microscopic volumes (µm–nm scale), relevant for predicting DNA damage and biological effectiveness. Track-structure codes (Geant4-DNA, MPEXS-DNA) simulate every electron interaction step-by-step, requiring liquid water cross-sections down to sub-eV energies; a single proton track produces ~10⁵ secondary interactions. GPU parallelization across simultaneous primary particle tracks (one thread per track) achieves 50–70× speedup. Applications include carbon-ion RBE calculation, targeted radionuclide dosimetry (alpha emitters), and predicting clustered DNA damage yields from mixed radiation fields.
- **Key algorithms:** Event-by-event track structure (Geant4-DNA cross-sections), step-by-step condensed random walk, DNA damage scoring (DSB, SSB, base damage), diffusion-reaction chemistry simulation (radiolysis), nanodosimeter simulation, LET spectrum calculation, biological effectiveness prediction.
- **Datasets:** Geant4-DNA physics validation data (https://geant4-dna.in2p3.fr/); NIST electron stopping powers (https://www.nist.gov/pml/estar); AAPM/NCRP microdosimetry benchmark datasets; published DNA damage yield datasets from radiobiology experiments.
- **Starter repos/tools:** Geant4-DNA (https://geant4-dna.in2p3.fr/ — part of Geant4, https://github.com/Geant4/geant4) — standard track-structure code; MPEXS-DNA (CUDA GPU version, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6850505/ — verify GitHub) — GPU microdosimetry and radiolysis; TOPAS-nBio (https://github.com/topas-nbio/TOPAS-nBio) — nano-biological extension of TOPAS; PARTRAC (verify URL) — track structure specialized for DNA damage.
- **CUDA libraries & GPU pattern:** Custom CUDA per-track simulation (one warp per track, reaction lookup in constant memory); divergence minimized by sorting tracks by interaction type before step; cuRAND Philox generator for per-track random sequences; atomic adds to DNA damage histogram; shared memory for cross-section table of current material step.

---

### 5.12 FLASH Radiotherapy GPU Modeling 🔴 · Frontier/Theoretical
- **Deep dive:** FLASH-RT delivers doses at ultra-high dose rates (>40 Gy/s, typically >10⁴ Gy/s for electrons, >100 Gy/s for protons) in millisecond pulses, sparing normal tissue while maintaining tumor control. Modeling the FLASH effect requires coupled radiation-chemistry simulation: (1) GPU MC particle transport to compute local dose deposition patterns, (2) GPU track-structure to generate initial radical (OH•, H₂O₂, e⁻ₐq) distributions, and (3) GPU diffusion-reaction kinetics to simulate oxygen depletion and radical recombination in tissue. The MPEXS2.1-DNA code implements GPU water radiolysis under UHDR. Biological effect modeling requires stochastic ODE integration over microscopic reaction networks — a GPU-parallel task across millions of spatial positions.
- **Key algorithms:** GPU MC particle transport at UHDR pulse structure, water radiolysis reaction-diffusion (Gillespie SSA on GPU), oxygen depletion kinetics, stochastic diffusion-reaction (MPEXS2.1-DNA), LET-dependent radical yield models, oxygen enhancement ratio (OER) map computation, pulse-by-pulse dose accumulation.
- **Datasets:** FLASH-RT experimental dosimetry from CERN/CLEAR, UCLouvain, Stanford FLASH programs (verify access); AAPM FLASH-RT working group benchmark datasets (verify URL); published oxygen tension measurements in tumors; GEANT4-DNA radiolysis validation datasets.
- **Starter repos/tools:** MPEXS2.1-DNA (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12551771/ — verify GitHub URL from paper) — GPU water radiolysis for UHDR; GATE 10 (https://github.com/OpenGATE/opengate) — FLASH macro-dose MC; TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — FLASH dosimetry extensions; Geant4-DNA (https://github.com/Geant4/geant4) — micro-kinetics for FLASH effect modeling.
- **CUDA libraries & GPU pattern:** Custom CUDA diffusion-reaction kernel (per-spatial-voxel Gillespie SSA, one thread block per µm³ tissue voxel); cuRAND for stochastic reaction channel selection; shared memory for local species concentration array; CUDA streams for pipelining pulse-by-pulse dose transport and chemistry; atomic ops for species count updates.

---

### 5.13 BNCT Dose Calculation & Optimization 🔴 · Frontier/Theoretical
- **Deep dive:** Boron Neutron Capture Therapy (BNCT) delivers therapeutic dose by targeting tumor cells loaded with ¹⁰B, which captures thermal neutrons to release high-LET alpha particles and lithium recoils. Dose calculation involves: (1) neutron transport (diffusion or discrete ordinates / Monte Carlo) to compute thermal neutron flux maps, (2) boron dose from ¹⁰B(n,α)⁷Li reaction rates, (3) high-LET photon dose, and (4) fast neutron dose — each requiring separate cross-section libraries and requiring GPU-parallel transport. The compound biological effectiveness (CBE) factor and boron uptake heterogeneity add biological modeling complexity. Treatment planning must jointly optimize beam direction and boron carrier dosing.
- **Key algorithms:** Monte Carlo neutron transport (OpenMC, MCNP, GATE), discrete ordinates neutron transport (Sₙ), multi-group cross-section library (ENDF/B-VIII), boron dose kernel convolution, CBE-weighted biological dose, neutron activation analysis on GPU, joint boron+neutron beam optimization.
- **Datasets:** IAEA BNCT benchmark cases (verify URL at iaea.org); BNCT clinical trial CT data from Finnish accelerator BNCT program; OpenMC validation datasets (https://github.com/openmc-dev/openmc/tree/develop/tests); NIST neutron cross-section data (verify URL).
- **Starter repos/tools:** OpenMC (https://github.com/openmc-dev/openmc) — open-source GPU-capable neutron MC (OpenMP/GPU via OpenMP target offload); GATE 10 (https://github.com/OpenGATE/opengate) — neutron transport for BNCT; COMPASS BNCT MC (verified in Nature Scientific Reports, https://pmc.ncbi.nlm.nih.gov/articles/PMC10366114/); OpenMC MeVisLab BNCT pipeline (https://www.hplpb.com.cn/en/article/doi/10.11884/HPLPB202537.250246).
- **CUDA libraries & GPU pattern:** GPU neutron transport via OpenMP offload or custom CUDA kernel (one thread per neutron history); material cross-section tables in texture memory; boron concentration map in 3D GPU array; cuBLAS for multi-group matrix-vector flux equations; warp-divergence mitigation by material-sorted particle batches.

---

### 5.14 GPU-Accelerated Adaptive MR-Linac Workflow 🟡 · Active R&D
- **Deep dive:** MR-Linac (MRL) systems (Elekta Unity, ViewRay MRIdian) combine MRI with simultaneous radiation delivery, enabling online adaptive radiotherapy (oART) where each fraction's plan is re-optimized based on daily anatomy. The oART workflow must complete all steps within a 30–90 minute treatment slot: (1) real-time MRI reconstruction (GPU NUFFT, <1 s), (2) deformable MR-to-MR registration (GPU Demons/VoxelMorph, <30 s), (3) synthetic CT generation (deep learning CT from MRI, GPU CNN, <10 s), (4) GPU dose recalculation on adapted anatomy (<30 s via collapsed-cone or MC), and (5) re-optimization (<2 min). Every step requires GPU; the entire chain is a GPU pipeline.
- **Key algorithms:** Real-time MRI reconstruction (radial GRASP GPU), MR-to-MR deformable registration (Demons, SyN), synthetic CT generation (CNN: MR→sCT), GPU collapsed-cone dose on sCT, GPU proton or photon dose recalculation, warm-start IMRT fluence re-optimization, plan approval metric computation.
- **Datasets:** MR-Linac Consortium shared datasets (verify URL at mrlinac.org); TCIA MR-guided RT datasets; AAPM MR-Linac WG test cases; MRI-only radiotherapy datasets from published cohorts.
- **Starter repos/tools:** Gadgetron (https://github.com/gadgetron/gadgetron) — real-time GPU MRI reconstruction for MRL; Plastimatch (https://plastimatch.org/) — GPU DIR + sCT generation; matRad (https://github.com/e0404/matRad) — dose re-optimization kernel; MONAI (https://github.com/Project-MONAI/MONAI) — CNN for MR→sCT translation.
- **CUDA libraries & GPU pattern:** CUDA streams pipeline: acquisition → cuFFT NUFFT → cuDNN sCT CNN → GPU dose kernel → cuSPARSE optimizer → display; each stage double-buffered to overlap computation with data transfer; multi-GPU across the 5-stage pipeline.

---

### 5.15 Proton CT & Ion Imaging Reconstruction 🔴 · Frontier/Theoretical
- **Deep dive:** Proton CT (pCT) measures the residual range of individual protons after traversing a patient, converting to relative stopping power (RSP) maps directly for treatment planning — eliminating the Hounsfield-unit–to–RSP conversion uncertainty in X-ray CT. Each proton's path through tissue is a curved most-likely path (MLP) rather than a straight line; for 10⁸ protons per scan, computing all MLPs and binning them into a sinogram for reconstruction is a massively parallel GPU task. Iterative pCT reconstruction (POCS with RSP constraints, MLSD) requires forward/backprojection along curved proton paths, fundamentally different from X-ray cone-beam and requiring custom GPU geometry kernels. Clinical pCT scanners (IBA, PRaVDA) generate data at 10⁸ proton events/second — GPU is mandatory for any real-time capability.
- **Key algorithms:** Most-likely path (MLP) estimation (Highland formula, Gaussian scattering), list-mode proton CT reconstruction (CSPACS, MLSD), POCS with RSP box constraints, proton trajectory binning for FBP, iterative proton CT with scattering regularization, proton radiography for range verification.
- **Datasets:** PRaVDA proton CT datasets (verify URL); PRIMA proton CT consortium data (verify URL); TOPAS-generated pCT simulation data; ACE collaboration proton CT phantom datasets.
- **Starter repos/tools:** pCT reconstruction code from UCI/Santa Cruz collaboration (verify URL); TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — proton CT simulation; FRED (https://www.fredonline.eu/) — proton transport/range imaging; custom CUDA MLP projection repos (search GitHub "proton CT GPU most likely path").
- **CUDA libraries & GPU pattern:** One CUDA thread per detected proton (massively parallel MLP computation); cuBLAS for scattering covariance matrix updates; thrust sort for proton trajectory binning by projection angle; custom CUDA backprojection along curved MLP geometry; cuRAND for proton beam Monte Carlo sampling.

---

## 6. Computational Physiology & Systems Biology

### 6.1 Cardiac Electrophysiology Simulation 🟡 · Active R&D
- **Deep dive:** Simulates transmembrane voltage propagation across cardiac tissue by solving the monodomain or bidomain reaction-diffusion PDE coupled to stiff ODEs representing ionic channel kinetics (e.g., ten Tusscher-Panfilov, O'Hara-Rudy). Each voxel integrates 50–200 state variables per time step at sub-millisecond temporal resolution; a whole-heart simulation at 0.1 mm spatial resolution yields ~10⁸ nodes, making the per-node ODE update embarrassingly parallel. The GPU eliminates the otherwise serial per-cell Rush-Larsen / RL2 exponential gating integration. Operator splitting decouples the reaction (GPU-parallel ODE) from diffusion (sparse linear solve), and CUDA kernels saturate memory bandwidth on the former while cuSPARSE handles the latter.
- **Key algorithms:** Monodomain/bidomain reaction-diffusion, operator splitting (Strang/Godunov), Rush-Larsen explicit gating, Crank-Nicolson implicit diffusion, conjugate gradient with ILU(0) preconditioning, finite volume/finite element spatial discretization.
- **Datasets:** PhysioNet MIT-BIH & MIMIC-III Waveform — 40 000+ ICU ECG/hemodynamic waveforms (https://physionet.org); CellML Physiome Repository — curated ionic cell models in CellML/SBML format importable by openCARP (https://models.physiomeproject.org); UK Biobank Cardiac MRI — 100 000+ cine CMR studies, access via application (https://www.ukbiobank.ac.uk); ACDC MICCAI Cardiac Challenge — 100-patient CMR with LV/RV/myocardium ground truth (https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- **Starter repos/tools:** openCARP (https://git.opencarp.org/openCARP/openCARP) — MPI+CUDA cardiac EP solver, CARPutils Python scripting, v19.0 April 2026; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — finite-volume GPU monodomain solver with Purkinje coupling and MPI batch dispatch; Cardioid/LLNL (https://github.com/llnl/cardioid) — multiscale cardiac suite (EP + mechanics + ECG), CUDA optional, Gordon Bell finalist; Chaste (https://github.com/Chaste/Chaste) — Oxford bidomain solver with cardiac mechanics module.
- **CUDA libraries & GPU pattern:** cuSPARSE (diffusion SpMV), cuSOLVER (linear system), CUDA custom kernels (per-cell ODE Rush-Larsen); pattern: fine-grained thread-per-cell ODE + coarse SpMV for diffusion; streams for overlapping compute and halo exchange.

---

### 6.2 Whole-Heart Digital Twin 🟡 · Active R&D
- **Deep dive:** Integrates patient-specific cardiac geometry (from CMR segmentation), fiber orientation (rule-based or DTI), EP simulation, and mechanical contraction into a unified virtual organ calibrated to clinical measurements. Building the twin requires iterative parameter estimation loops—thousands of forward simulations of the EP+mechanics PDE system—making GPU acceleration critical not just for each simulation but for the ensemble inference step. Differentiable simulators (e.g., TorchCor) allow gradient-based parameter fitting through the forward model. Hemodynamic boundary conditions couple the twin to a lumped Windkessel circulation model.
- **Key algorithms:** Bidomain/monodomain EP, active-strain / active-stress cardiac mechanics (nonlinear elasticity), Windkessel 3-element lumped circulation, rule-based fiber assignment (Bayer-Blake-Plank), adjoint-based or ensemble Kalman filter parameter estimation, finite element method (FEM) with tetrahedral meshes.
- **Datasets:** UK Biobank Cardiac MRI — 100 000+ cine CMR (https://www.ukbiobank.ac.uk); Zenodo Synthetic Biventricular Heart Meshes — 1 000 virtual cohort meshes (https://zenodo.org/records/4506930); Visible Human Project — full-body cryosection + CT + MRI (https://www.nlm.nih.gov/research/visible/visible_human.html); ACDC MICCAI — 100-patient CMR segmentations (https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- **Starter repos/tools:** openCARP (https://git.opencarp.org/openCARP/openCARP) — EP component of twins; simcardems (https://github.com/ComputationalPhysiology/simcardems) — FEniCS-based cardiac electromechanics coupling; TorchCor (https://github.com/sagebei/torchcor) — PyTorch GPU cardiac EP FEM for differentiable twin fitting; Awesome-Cardiac-Digital-Twins list (https://github.com/lileitech/Awesome-Cardiac-Digital-Twins) — curated resource index.
- **CUDA libraries & GPU pattern:** cuSPARSE + cuSOLVER (FEM assembly/solve), cuBLAS (adjoint vector ops), custom CUDA kernels (ionic ODE batch); pattern: batched forward solves across ensemble members for parameter inference; mixed precision (FP16 forward, FP32 gradient accumulation).

---

### 6.3 Hemodynamics / Blood-Flow CFD 🟡 · Active R&D
- **Deep dive:** Solves the incompressible Navier-Stokes equations on patient-specific vascular geometries (aorta, coronary arteries, cerebral vasculature) reconstructed from CT/MRI angiography. Non-Newtonian blood rheology (Carreau-Yasuda model) and fluid-structure interaction (FSI) with compliant vessel walls add computational stiffness. Wall shear stress (WSS) and oscillatory shear index (OSI) fields—risk factors for atherosclerosis—require temporally resolved solutions across the cardiac cycle (~1000 time steps). GPU parallelism maps naturally onto the unstructured mesh cell updates.
- **Key algorithms:** Incompressible Navier-Stokes (fractional-step / SIMPLE / PISO), ALE formulation for FSI, non-Newtonian viscosity (Carreau-Yasuda), arbitrary Lagrangian-Eulerian mesh motion, finite volume method on unstructured polyhedral meshes, multigrid pressure solver, RBF mesh morphing.
- **Datasets:** PhysioNet MIMIC-III waveforms — invasive pressure/flow recordings (https://physionet.org/content/mimiciii/1.4/); Vascular Model Repository — patient-specific vascular geometries (http://www.vascularmodel.com); Zenodo Cardiac Mechanics Emulation dataset (https://zenodo.org/records/7075055); UK Biobank aortic flow (4D flow MRI subset) (https://www.ukbiobank.ac.uk).
- **Starter repos/tools:** SimVascular/svFSI (https://github.com/SimVascular/svFSI) — open-source image-to-simulation pipeline with GPU-capable parallel solver; OpenFOAM-dev (https://github.com/OpenFOAM/OpenFOAM-dev) — general CFD with biomedical application via custom boundary conditions; Chaste (https://github.com/Chaste/Chaste) — includes vascular network flow module; HemeLB (https://github.com/hemelb-codes/hemelb) — sparse vascular lattice-Boltzmann alternative.
- **CUDA libraries & GPU pattern:** AmgX (GPU multigrid pressure solver), cuSPARSE (SpMV for flux assembly), NVIDIA RAPIDS for mesh preprocessing; pattern: domain decomposition with MPI+CUDA, halo-exchange via NCCL, time-stepping loop with async memory transfers.

---

### 6.4 Lattice-Boltzmann Blood/Airflow Solver 🟡 · Active R&D
- **Deep dive:** The lattice-Boltzmann method (LBM) replaces continuum Navier-Stokes with a mesoscale kinetic equation for particle distribution functions on a regular grid—ideal for GPUs because each lattice site updates independently using only nearest-neighbor communication (the BGK collision step). Blood in complex vascular trees, red blood cell suspension rheology, and pulmonary airflow through bronchial trees all benefit from this approach. HemeLB achieves ~29.5 billion lattice site updates per second on thousands of cores; GPU versions (e.g., HemeLB GPU branch, PALABOS GPU) push throughput further with shared-memory streaming.
- **Key algorithms:** BGK (Bhatnagar-Gross-Krook) collision operator, multi-relaxation time (MRT) LBM, D3Q19/D3Q27 velocity stencils, bounce-back boundary conditions for no-slip walls, Shan-Chen multiphase LBM, immersed boundary method for red blood cell membranes, Palabos fluid-particle coupling.
- **Datasets:** PhysioNet coronary/aortic waveforms (https://physionet.org); Vascular Model Repository geometries (http://www.vascularmodel.com); open-access bronchial tree CT data from LIDC-IDRI (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); UK Biobank aortic flow MRI (https://www.ukbiobank.ac.uk).
- **Starter repos/tools:** HemeLB (https://github.com/hemelb-codes/hemelb) — sparse-geometry vascular LBM, MPI+GPU, scales to 32 000+ cores; HemePure GPU variant (https://github.com/hemelb-codes/HemePure) — cleaned GPU-first branch; PALABOS (https://gitlab.com/unigespc/palabos) — full-featured C++ LBM framework including multiphase and thermal extensions; USERMESO-2.0 (https://github.com/AnselGitAccount/USERMESO-2.0) — GPU red blood cell hemodynamics with deformable membrane.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for BGK streaming+collision in a single fused pass; shared memory for D3Q19 population arrays; texture memory for geometry masks; NCCL for GPU-direct halo exchange; pattern: one-thread-per-lattice-site with coalesced memory access on SOA layout.

---

### 6.5 Respiratory / Lung Airflow & Particle Deposition 🟡 · Active R&D
- **Deep dive:** Simulates inspiratory/expiratory flow through the conducting airways (generations 0–16, reconstructed from CT) and tracks inhaled aerosol/drug particle trajectories via Lagrangian particle tracking. The lung's tree topology means ~10⁶–10⁷ computational cells in the airway geometry and millions of particle trajectories evaluated each breath cycle—both trivially parallelizable on GPU. Alveolar gas exchange adds a reaction-diffusion layer for O₂/CO₂ that couples to a 1D ventilation model for the respiratory tree periphery.
- **Key algorithms:** Incompressible Navier-Stokes (finite volume), Lagrangian discrete-phase particle tracking (drag + Brownian + Saffman lift forces), Stokes drag law, k-ω SST RANS turbulence, LBM for alveolar-scale flow, convection-diffusion for gas species, quasi-1D ventilation model (Horsfield tree).
- **Datasets:** LIDC-IDRI lung CT — 1 010 cases with nodule annotations, TCIA (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); COPDGene lung CT dataset — 10 000 subjects (https://www.copdgene.org); SPIROMICS bronchial CT (https://www.spiromics.org); PhysioNet respiratory waveform databases (https://physionet.org).
- **Starter repos/tools:** OpenFOAM-dev (https://github.com/OpenFOAM/OpenFOAM-dev) — Lagrangian particle tracking (DPMFoam) with GPU-capable solver via GPU-accelerated AmgX pressure solve; SimVascular (https://github.com/SimVascular) — vascular flow basis adaptable to airways; PALABOS (https://gitlab.com/unigespc/palabos) — LBM for alveolar flow; 3D Slicer + SlicerMorph (https://github.com/SlicerMorph/SlicerMorph) — airway segmentation from CT.
- **CUDA libraries & GPU pattern:** CUDA Thrust for particle sort/bin operations; custom CUDA kernels for Lagrangian force integration (one thread per particle); cuSPARSE for airflow linear solve; pattern: dual-stream approach—Eulerian fluid on one SM partition, Lagrangian particles on another with atomic-add deposition counters.

---

### 6.6 Neuronal Network Simulation (Biophysical) 🟡 · Active R&D
- **Deep dive:** Simulates networks of morphologically detailed (multi-compartment) neurons using Hodgkin-Huxley-style conductance-based kinetics in each dendritic/axonal segment. A single layer-5 pyramidal cell may have 1 000+ compartments each with 10–30 gating variables, and a cortical column model contains thousands of such cells—resulting in millions of coupled ODEs. The Hines solver (tridiagonal Thomas algorithm along each dendritic tree branch) enables efficient per-cell compartmental integration, but parallelizing across cells and synapses is where GPUs excel. Spike delivery (synaptic event processing) introduces irregular memory access that benefits from GPU-side event queues.
- **Key algorithms:** Hodgkin-Huxley conductance-based kinetics, Hines tridiagonal solver (branching cable equation), Rush-Larsen exponential integration for gates, event-driven spike delivery, exponential synapse models (AMPA/NMDA/GABA), adaptive time-stepping (CVODE).
- **Datasets:** NeuroMorpho.Org — 200 000+ 3D neuronal reconstructions across 900+ species (https://neuromorpho.org); ModelDB / modeldb.science — curated computational neuron models with NEURON/GENESIS files (https://modeldb.science); Allen Brain Cell Atlas — single-cell transcriptomics + patch-seq morpho-electric data (https://portal.brain-map.org); DANDI Archive — neurophysiology datasets in NWB format (https://dandiarchive.org).
- **Starter repos/tools:** NEURON + CoreNEURON GPU (https://github.com/neuronsimulator/nrn) — canonical compartmental simulator with CUDA backend via CoreNEURON; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — multiscale network builder on top of NEURON with HPC support; MOOSE (https://github.com/BhallaLab/moose-core) — multiscale OO simulator for neuronal + biochemical networks; Blue Brain / Open Brain Institute (https://github.com/BlueBrain) — production-grade cortical column models.
- **CUDA libraries & GPU pattern:** CoreNEURON uses cuSPARSE for Hines matrix batches; custom CUDA kernels for gate ODEs; cuRAND for stochastic synaptic release; pattern: one CUDA thread-block per cell, warp-level branching for dendritic trees; SOA memory layout for coalesced gating variable access.

---

### 6.7 Spiking Neural Network (Point-Neuron) Simulation 🟡 · Active R&D
- **Deep dive:** Point-neuron SNN models (leaky integrate-and-fire, Izhikevich, adaptive exponential IF) sacrifice morphological detail in exchange for simulating networks of millions to billions of neurons in real time. Each neuron updates a handful of state variables per time step; spikes generate synaptic current injections to thousands of target neurons via a connectivity matrix that is typically sparse (~10 000 synapses/neuron). GeNN generates custom CUDA kernels from user model descriptions, achieving real-time simulation of 10⁶-neuron Izhikevich networks on a single GPU. NEST GPU and Brian2CUDA follow similar kernel-generation approaches.
- **Key algorithms:** Leaky integrate-and-fire (LIF), Izhikevich neuron model, adaptive exponential integrate-and-fire (AdEx), spike-timing-dependent plasticity (STDP), exponential/alpha synapse kernels, delay-line spike queues, random balanced-network (Brunel) connectivity.
- **Datasets:** Allen Brain Observatory — visual cortex spiking data from Neuropixels (https://portal.brain-map.org); DANDI Archive — electrophysiology datasets NWB format (https://dandiarchive.org); OpenNeuro — EEG/MEG recordings for network model validation (https://openneuro.org); Human Connectome Project structural connectivity matrices (https://db.humanconnectome.org).
- **Starter repos/tools:** GeNN (https://github.com/genn-team/genn) — GPU-enhanced SNN code generator (CUDA + HIP), includes Brian2GeNN and ml_genn deep SNN; SpikingJelly (https://github.com/fangwei123456/spikingjelly) — PyTorch-based SNN framework with CUDA extensions; Brian2CUDA (https://github.com/brian-team/brian2cuda) — CUDA code generation backend for Brian2; NEST GPU (https://github.com/nest/nest-simulator) — multi-GPU NEST backend scaling to 10⁹ neurons.
- **CUDA libraries & GPU pattern:** Custom generated CUDA kernels (GeNN/Brian2CUDA), cuSPARSE for synaptic current summation via sparse matrix-vector product, cuRAND for Poisson spike generation; pattern: one thread per neuron for state update, warp-shuffle for local spike detection, atomic-add for synaptic current accumulation.

---

### 6.8 Tumor Growth & Treatment-Response Modeling 🟡 · Active R&D
- **Deep dive:** Continuum-PDE models (reaction-diffusion for nutrient/oxygen, tumor cell density, and treatment drug concentration) combined with discrete cell-based models capture avascular-to-vascular tumor growth, hypoxia-driven necrosis, and response to radiation or chemotherapy. GPU acceleration is essential for solving coupled PDE systems on 3D grids (512³ voxels = 1.3×10⁸ cells) at each time step of a multi-day simulation. Parameter sweeps for virtual clinical trials (thousands of parameter sets) are embarrassingly parallel across the GPU grid.
- **Key algorithms:** Fisher-KPP reaction-diffusion (tumor cell density), oxygen/nutrient diffusion-consumption (Green's function or FD), phenomenological radiobiological model (linear-quadratic), drug PK/PD compartment coupling, phase-field tumor morphology, level-set interface tracking for tumor boundary.
- **Datasets:** TCGA (The Cancer Genome Atlas) — multi-omics + imaging for model calibration (https://portal.gdc.cancer.gov); TCIA (The Cancer Imaging Archive) — multi-institutional tumor imaging (https://www.cancerimagingarchive.net); PhysioNet oncology waveforms (https://physionet.org); Zenodo tumor growth simulation datasets (search zenodo.org for "tumor growth simulation").
- **Starter repos/tools:** PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D agent-based multicellular simulator with diffusing substrates, scales linearly in cell count; PhysiBoSS (https://github.com/PhysiBoSS/PhysiBoSS) — extends PhysiCell with Boolean network intracellular signaling (MaBoSS); Chaste (https://github.com/Chaste/Chaste) — includes tumor spheroid and crypt models; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — used for drug delivery flow simulations.
- **CUDA libraries & GPU pattern:** Custom CUDA FD stencil kernels (3D 7-point Laplacian on oxygen/drug grids), CUDA Thrust for per-cell agent sorting and binning, cuRAND for stochastic division/death events; pattern: 3D CUDA thread grid for PDE, separate kernel for agent-based cell loop with shared-memory neighborhood queries.

---

### 6.9 Agent-Based Tissue / Immune Simulation 🟡 · Active R&D
- **Deep dive:** Tissue is modeled as a population of autonomous agents (cells) each tracking position, velocity, cycle state, secretion rates, and mechanistic signaling. Cell-cell mechanical interactions (overlap repulsion, adhesion) require pairwise neighbor search that scales as O(N²) naively but drops to O(N) with spatial binning on GPU. Immune cell migration, cytokine diffusion, and tumor-immune coevolution are natural applications. PhysiCell supports 10⁵–10⁶ cells in 3D with GPU-accelerated substrate diffusion.
- **Key algorithms:** Center-based mechanics (soft-sphere repulsion + adhesion), cell cycle models (Ki67 basic/advanced, flow cytometry), substrate diffusion (Thomas ADI or explicit FD on Cartesian grid), chemotaxis gradient following, receptor-ligand binding kinetics, Boolean intracellular signaling (MaBoSS), spatial hashing for neighbor search.
- **Datasets:** CancerSEA single-cell functional states (http://biocc.hrbmu.edu.cn/CancerSEA/); TCGA pan-cancer immune landscape (https://portal.gdc.cancer.gov); MIBI/IMC imaging mass cytometry datasets (various Zenodo deposits); TCIA immunotherapy imaging (https://www.cancerimagingarchive.net).
- **Starter repos/tools:** PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D multicellular simulator with physics + biotransport; PhysiBoSS (https://github.com/PhysiBoSS/PhysiBoSS) — Boolean network–PhysiCell coupling for signaling; Chaste (https://github.com/Chaste/Chaste) — off-lattice cell-based models with vertex/Voronoi mechanics; MOOSE (https://github.com/BhallaLab/moose-core) — chemical signaling within cells.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for substrate PDE (explicit or ADI); CUDA Thrust for cell sort by spatial bin; atomic-add for cytokine source terms from agent loop; pattern: hybrid CPU (agent logic) + GPU (PDE + neighbor search) with pinned memory for cell state transfer.

---

### 6.10 Systems-Biology ODE/SDE Network Solver 🟡 · Active R&D
- **Deep dive:** Gene regulatory networks, signaling cascades, and metabolic models are encoded as systems of potentially thousands of nonlinear ODEs/SDEs (e.g., SBML models from BioModels). Integrating a single model is fast, but parameter sweeps, uncertainty quantification, and multi-cell applications require solving thousands of independent instances simultaneously—a perfectly GPU-parallel batch problem. SUNDIALS/CVODE-GPU and libRoadRunner's LLVM JIT backend both target this batch-ODE pattern.
- **Key algorithms:** CVODE adaptive BDF/Adams multistep integrator, explicit Euler / Runge-Kutta (RK4, RK45 Dormand-Prince) for stiff-moderate systems, implicit trapezoidal, chemical Langevin equation (CLE) for SDE, sensitivity equations (CVODES/IDAS), SBML parsing and JIT compilation.
- **Datasets:** BioModels Database (EMBL-EBI) — 1000+ curated SBML models (https://www.ebi.ac.uk/biomodels); Reactome pathways — curated molecular interaction data (https://reactome.org); BioGRID interaction network (https://thebiogrid.org); VCell curated models (https://vcell.org).
- **Starter repos/tools:** SUNDIALS/CVODE GPU (https://github.com/LLNL/sundials) — LLNL ODE/DAE solver with CUDA NVector and GPU-accelerated batch CVODE; libRoadRunner (https://github.com/sys-bio/roadrunner) — high-performance SBML ODE integrator with LLVM JIT, GPU batch mode in development; Tellurium (https://github.com/sys-bio/tellurium) — Python systems biology platform built on roadrunner; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — SSA + tau-leaping + CLE stochastic solver.
- **CUDA libraries & GPU pattern:** CUDA batched ODE: one CUDA thread-block per ODE system; shared memory for Jacobian; cuSPARSE for large sparse Jacobians; SUNDIALS CUDA NVector; pattern: batch-CVODE with user-supplied CUDA right-hand-side (RHS) kernel.

---

### 6.11 Stochastic (Gillespie) Biochemical Simulation 🟢 · Established
- **Deep dive:** The Gillespie Stochastic Simulation Algorithm (SSA) exactly samples the master equation for discrete molecular counts in a well-mixed chemical reaction network—critical when molecule numbers are small (transcription factors, signaling molecules). Each stochastic trajectory is independent, so GPU parallelism maps one trajectory per thread. With 1 000–10 000 trajectories needed for statistics, GPU batch SSA achieves orders-of-magnitude speedup. Tau-leaping approximations (binomial/Poisson) trade exactness for speed at higher copy numbers.
- **Key algorithms:** Gillespie SSA (direct method), Gibson-Bruck next-reaction method, tau-leaping (explicit/implicit/binomial), R-leaping, chemical Langevin equation (CLE), reaction-diffusion master equation (RDME) for spatial stochastic simulation.
- **Datasets:** BioModels Database — curated stochastic SBML models (https://www.ebi.ac.uk/biomodels); NIST Chemical Kinetics Database (https://kinetics.nist.gov); single-molecule tracking datasets on DANDI (https://dandiarchive.org); smFISH gene expression data (various GEO deposits at https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** GillesPy2 (https://github.com/GillesPy2/GillesPy2) — Python SSA + tau-leaping + CLE, GPU backend in progress; StochPy (https://github.com/SystemsBioinformatics/stochpy) — Python stochastic simulation with SSA and tau-leaping; cuTauLeaping (verify URL — CUDA tau-leaping reference implementations in CUDA samples literature); MOOSE (https://github.com/BhallaLab/moose-core) — compartmental stochastic kinetic simulations.
- **CUDA libraries & GPU pattern:** cuRAND for per-trajectory random exponential/uniform variates (one cuRAND stream per thread); CUDA Thrust for propensity prefix-sum (direct-method reaction selection); pattern: one CUDA thread per trajectory, independent RNG state in registers; atomic operations avoided by design (each thread is fully independent).

---

### 6.12 Metabolic Flux / Constraint-Based Modeling 🟢 · Established
- **Deep dive:** Flux balance analysis (FBA) finds optimal metabolic fluxes by solving a linear program (LP) constrained by stoichiometry, thermodynamics, and enzyme capacity on genome-scale metabolic models (GEMs) with 3 000–8 000 reactions. GPU parallelism enters through solving thousands of LP instances in parallel (e.g., for all conditions in a drug screen, or all single-gene knockouts in an essentiality screen). Mixed-integer programming (MILP) variants for gap-filling and thermodynamic FBA benefit from GPU-accelerated interior-point methods.
- **Key algorithms:** Flux balance analysis (FBA), flux variability analysis (FVA), parsimonious FBA (pFBA), thermodynamic FBA (tFBA), MILP gap-filling, minimal cut sets, COBRA toolbox algorithms, interior-point LP (revised simplex), shadow price / sensitivity analysis.
- **Datasets:** Recon3D — human genome-scale metabolic model (https://github.com/SBRG/Recon3D); HMDB — Human Metabolome Database (https://hmdb.ca); Reactome (https://reactome.org); BiGG Models Database — curated GEMs (http://bigg.ucsd.edu).
- **Starter repos/tools:** COBRApy (https://github.com/opencobra/cobrapy) — Python FBA/FVA with multiple LP/MILP solver backends; Recon3D model files (https://github.com/SBRG/Recon3D); Virtual Metabolic Human (https://vmh.life) — interactive Recon3D portal; SUNDIALS (https://github.com/LLNL/sundials) — for dynamic FBA ODE integration.
- **CUDA libraries & GPU pattern:** cuSOLVER dense LP factor (batch small LP); custom CUDA interior-point primal-dual kernel for LP batches; ArrayFire (https://github.com/arrayfire/arrayfire) for dense matrix batches; pattern: one LP per CUDA block, shared memory for constraint matrix, warp-level reduction for objective gradient.

---

### 6.13 Gene Regulatory Network Inference 🟡 · Active R&D
- **Deep dive:** Infers the directed causal graph of transcription factor-gene interactions from single-cell RNA-seq (scRNA-seq) time-series or perturbation data. State-of-the-art methods use mutual information, GENIE3 random forests, or neural ODE formulations. Computing pairwise mutual information across 20 000 genes requires O(N²) comparisons—a 200-million-pair problem on a GPU. Bayesian network structure learning and variational inference for large graph posteriors are also GPU-amenable.
- **Key algorithms:** GENIE3 random forest (feature importance), ARACNE mutual information + data processing inequality, PANDA message-passing network inference, neural ODE (torchdiffeq) for dynamics, variational autoencoder (scVI) for expression latent space, LASSO/elastic-net for linear GRN, Granger causality.
- **Datasets:** Gene Expression Omnibus (GEO) — tens of thousands of scRNA-seq datasets (https://www.ncbi.nlm.nih.gov/geo/); ENCODE TF binding ChIP-seq (https://www.encodeproject.org); BEELINE benchmark GRN datasets (https://github.com/Murali-group/BEELINE); Human Cell Atlas scRNA-seq (https://www.humancellatlas.org).
- **Starter repos/tools:** BEELINE GRN benchmark (https://github.com/Murali-group/BEELINE) — benchmarking framework for GRN inference methods; scVI (https://github.com/scverse/scvi-tools) — deep generative models for scRNA-seq on GPU via PyTorch; torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU neural ODE for dynamics inference; Scanpy (https://github.com/scverse/scanpy) — scRNA-seq analysis with GPU-accelerated backends (rapids-singlecell).
- **CUDA libraries & GPU pattern:** cuBLAS for pairwise correlation matrix (N×N outer product); CUDA Thrust for per-gene sort/rank (MI estimation); GPU neural ODE via PyTorch autograd + custom CUDA adjoint; pattern: tiled matrix multiply for pairwise MI, one-tile-per-gene-pair block.

---

### 6.14 Multi-Scale Physiological Modeling 🟡 · Active R&D
- **Deep dive:** Couples models operating at different spatial/temporal scales: molecular (ion channel kinetics, μs–ms), cellular (action potential, ms), tissue (wave propagation, ms–s), organ (cardiac output, heartbeat), and system (circulation, minutes). The computational challenge is that fine-scale models (cell ODE) must be solved at each quadrature point of a coarse FEM mesh simultaneously—yielding millions of ODE instances per time step. GPU batch-ODE solving (CVODE GPU) fills this role. The Virtual Physiological Human (VPH) framework coordinates inter-scale coupling.
- **Key algorithms:** Heterogeneous multiscale method (HMM), operator splitting for scale coupling, homogenization, batch CVODE for cell-level ODEs at FEM quadrature points, Windkessel/1D vessel network for circulation, FEM for organ-level mechanics/EP, co-simulation coupling (FMI standard).
- **Datasets:** Physiome Model Repository — VPH-standard CellML models (https://models.physiomeproject.org); BioModels Database (https://www.ebi.ac.uk/biomodels); UK Biobank multi-modal phenotyping (https://www.ukbiobank.ac.uk); OpenCMISS examples (https://github.com/OpenCMISS/examples).
- **Starter repos/tools:** OpenCMISS/cm (https://github.com/OpenCMISS/cm) — multi-physics multi-scale FEM framework; SUNDIALS batch CVODE GPU (https://github.com/LLNL/sundials) — batch ODE for sub-grid cell models; simcardems (https://github.com/ComputationalPhysiology/simcardems) — cardiac electromechanics multi-scale coupling; Chaste (https://github.com/Chaste/Chaste) — multi-scale cardiac + lung + tumor modeling.
- **CUDA libraries & GPU pattern:** SUNDIALS CUDA NVector + batch CVODE (cell ODE at quadrature points); cuSPARSE for coarse-mesh FEM assembly; CUDA streams for asynchronous scale coupling; pattern: two-level parallelism—CUDA grid over FEM elements, threads over per-element ODE RHS evaluation.

---

### 6.15 PK/PD & PBPK Modeling 🟢 · Established
- **Deep dive:** Pharmacokinetic/pharmacodynamic (PK/PD) and physiologically-based PK (PBPK) models are compartmental ODE systems describing drug absorption, distribution, metabolism, and excretion across tissues. Population PK analysis (NLME) requires solving the ODE model for each individual in a cohort (hundreds to thousands) with Monte Carlo sampling of parameter distributions—perfectly GPU-parallel. GPU speedup reaches 10–100× for population-level stochastic simulations and Bayesian posterior sampling (HMC/NUTS).
- **Key algorithms:** Compartmental ODE integration (1-cpt, 2-cpt, PBPK), nonlinear mixed-effects (NLME) estimation, empirical Bayes estimation (EBE), Monte Carlo simulation, Bayesian MCMC (Hamiltonian Monte Carlo, NUTS), sensitivity analysis (Morris screening, Sobol indices), indirect-response PD models, transit compartment absorption.
- **Datasets:** PhysioNet MIMIC clinical PK data (https://physionet.org); FDA Adverse Event Reporting System (FAERS) (https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers); PBPK model library — OSP Suite (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); DDMoRe model repository (https://ddmore.eu/models-tools).
- **Starter repos/tools:** Open Systems Pharmacology Suite (https://github.com/Open-Systems-Pharmacology) — PK-Sim + MoBi PBPK platform; mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — R-based ODE PK/PD simulation; Pumas-AI (https://pumas.ai) — Julia pharmacometrics platform with GPU-accelerated population PK; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — stochastic PK variant simulation.
- **CUDA libraries & GPU pattern:** SUNDIALS batch CVODE on GPU (population member = one GPU batch element); cuRAND for Monte Carlo parameter sampling; custom CUDA kernel for NLME gradient (sum over individuals); pattern: one CUDA thread per subject for ODE integration, warp-level reduction for population log-likelihood.

---

### 6.16 Cardiac Mechanics & Electromechanical Coupling 🟡 · Active R&D
- **Deep dive:** Extends electrophysiology simulation by coupling electrical activation to active mechanical contraction through calcium-troponin cross-bridge kinetics (e.g., Rice-Wang-Bers model). The resulting system couples a stiff ODE (ionic + cross-bridge) at each integration point to a nonlinear FEM problem (hyperelastic myocardium with active stress/strain). GPU accelerates both the per-Gauss-point ODE batch and the global Newton-Raphson iterations for the mechanical equilibrium solve. Ventricular pressure-volume loops, ejection fraction, and wall stress distributions are clinical outputs.
- **Key algorithms:** Active-stress / active-strain formulations, Holzapfel-Ogden hyperelastic constitutive law, Rice-Wang-Bers cross-bridge kinetics, monodomain EP coupling, Newton-Raphson nonlinear FEM, Guccione passive strain energy, incompressibility via penalty/mixed formulation, Windkessel boundary conditions.
- **Datasets:** UK Biobank CMR + strain imaging (https://www.ukbiobank.ac.uk); Zenodo cardiac mechanics emulation dataset (https://zenodo.org/records/7075055); ACDC segmentation challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); MICCAI STACOM cardiac mechanics challenge data (verify URL on grand-challenge.org).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — nonlinear FEM cardiac/soft-tissue mechanics solver; simcardems (https://github.com/ComputationalPhysiology/simcardems) — FEniCS-based EP+mechanics coupling; OpenCMISS/cm (https://github.com/OpenCMISS/cm) — multi-physics FEM framework; Chaste (https://github.com/Chaste/Chaste) — cardiac electromechanics tutorial.
- **CUDA libraries & GPU pattern:** Batch CVODE GPU for per-Gauss-point ODE; cuSOLVER for Newton linear solve; cuSPARSE SpMV for stiffness matrix assembly; pattern: two-level CUDA grid—elements outer, Gauss points inner—with shared memory for per-element stiffness matrix accumulation.

---

### 6.17 Purkinje System & Conduction System Modeling 🟡 · Active R&D
- **Deep dive:** The cardiac conduction system (sinoatrial node, AV node, His bundle, bundle branches, Purkinje fiber network) initiates and coordinates ventricular activation. Simulating the Purkinje tree requires a 1D cable equation solver on a fractal branching network of ~10⁵ segments, coupled at Purkinje-muscle junctions (PMJs) to the 3D ventricular myocardium. GPU parallelism across the large number of independent 1D cable segments accelerates conduction pathway simulations for pacemaker dysfunction and re-entry arrhythmia studies.
- **Key algorithms:** 1D cable equation (monodomain) on Purkinje tree, PMJ coupling via gap-junction conductance, Stewart-Zhang Purkinje ionic model, His-Purkinje conduction velocity calibration, tree generation algorithms (L-system or rule-based branching), graph-based conduction delay computation.
- **Datasets:** openCARP community Purkinje experiments (https://opencarp.org/community/community-experiments); MonoAlg3D_C Purkinje examples (https://github.com/rsachetto/MonoAlg3D_C); NeuroMorpho (morphological analogy for tree datasets) (https://neuromorpho.org); PhysioNet His-bundle electrogram databases (https://physionet.org).
- **Starter repos/tools:** MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU monodomain solver with integrated Purkinje network and PMJ calibration; openCARP (https://git.opencarp.org/openCARP/openCARP) — supports Purkinje cable coupling; Cardioid/LLNL (https://github.com/llnl/cardioid) — includes Purkinje conduction modeling; Chaste (https://github.com/Chaste/Chaste) — 1D cable equation infrastructure.
- **CUDA libraries & GPU pattern:** Batch tridiagonal solvers (cuSPARSE batched Thomas) for 1D cable segments; custom CUDA kernels for ionic ODEs at each Purkinje node; CUDA graph for recurring per-beat computation pattern; pattern: one thread per Purkinje node, shared memory for tridiagonal coefficients within a segment.

---

### 6.18 ECG Forward Problem & Body-Surface Potential Mapping 🟢 · Established
- **Deep dive:** The ECG forward problem maps cardiac electrical sources (transmembrane currents from EP simulation) to body-surface potentials via the quasi-static Poisson equation on a torso volume conductor model. The transfer matrix (lead-field matrix) is computed once by solving many FEM boundary value problems (one per electrode), then applied repeatedly as a dense matrix-vector product at each time step of the EP simulation. GPU acceleration is ideal for both the batched FEM assembly and the dense matrix-vector multiply.
- **Key algorithms:** Quasi-static Poisson equation (torso conductivity model), finite element method on torso mesh, lead-field/transfer matrix computation, multipole source representation, method of fundamental solutions, ECG inverse problem (regularized Tikhonov, total variation), boundary element method (BEM).
- **Datasets:** PhysioNet ECG databases (https://physionet.org); EDGAR body-surface potential database (https://edgar.sci.utah.edu — verify URL); Cardioid ECG module examples (https://github.com/llnl/cardioid); Visible Human torso geometry (https://www.nlm.nih.gov/research/visible/visible_human.html).
- **Starter repos/tools:** Cardioid/LLNL (https://github.com/llnl/cardioid) — includes ECG forward solver module; openCARP (https://git.opencarp.org/openCARP/openCARP) — ECG lead calculation post-processing; SCIRun (https://github.com/SCIInstitute/SCIRun) — Utah scientific computing platform for ECG forward/inverse; APBS (https://github.com/Electrostatics/apbs) — electrostatics PDE solver adaptable to torso geometry.
- **CUDA libraries & GPU pattern:** cuBLAS DGEMV for transfer-matrix application at each time step; cuSOLVER for FEM system solve during transfer-matrix construction; batched cuSOLVER for simultaneous electrode-source BVPs; pattern: parallel BVP solves (one per electrode) with shared torso mesh.

---

### 6.19 Defibrillation & High-Voltage Shock Simulation 🟡 · Active R&D
- **Deep dive:** Defibrillation delivers a high-voltage electric field across the myocardium to terminate ventricular fibrillation. Simulating shock efficacy requires solving the bidomain equations driven by extracellular electrode currents, capturing virtual electrode polarization (VEP)—regions of depolarization and hyperpolarization induced at tissue boundaries—and subsequent re-entry termination. The nonlinear ionic response during shock (10 V/cm field, sub-ms timescale) and the fine spatial resolution needed (~0.1 mm) make GPU acceleration mandatory for whole-heart shock simulations.
- **Key algorithms:** Bidomain equations with extracellular stimulus, virtual electrode polarization theory, finite volume/element discretization, operator splitting with Rush-Larsen ionic integration, conjugate gradient linear solver, shock-protocol optimization (monophasic vs. biphasic), defibrillation threshold (DFT) estimation.
- **Datasets:** PhysioNet fibrillation/defibrillation recordings (https://physionet.org); openCARP defibrillation tutorial cases (https://opencarp.org); Cardioid (https://github.com/llnl/cardioid) — bidomain shock examples; patient-specific ICD placement datasets (verify institutional access).
- **Starter repos/tools:** openCARP (https://git.opencarp.org/openCARP/openCARP) — bidomain solver with extracellular stimulus for defibrillation studies; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU bidomain-capable extension; Cardioid/LLNL (https://github.com/llnl/cardioid) — cardiac EP + shock; Chaste (https://github.com/Chaste/Chaste) — bidomain with electrode boundary conditions.
- **CUDA libraries & GPU pattern:** cuSPARSE conjugate gradient for bidomain elliptic solve; custom CUDA kernels for per-cell ionic ODE during shock timescale (0.01 ms dt); CUDA Unified Memory for large torso+heart mesh; pattern: dual-grid approach—fine heart mesh on GPU, coarse torso on CPU, coupled via interface boundary.

---

### 6.20 Coronary Autoregulation & Microvascular Perfusion 🟡 · Active R&D
- **Deep dive:** Coronary blood flow is regulated by metabolic (adenosine), myogenic, and neural mechanisms operating across scales from capillaries (5 µm) to epicardial arteries (4 mm). GPU simulation of a microvascular network with 10⁴–10⁶ vessel segments requires solving a large sparse linear system (network Poiseuille flow) coupled to oxygen transport (convection-diffusion along each segment) and auto-regulatory feedback ODEs. Real-time coronary perfusion models support fractional flow reserve (FFR) virtual assessment for stenosis evaluation.
- **Key algorithms:** Network Poiseuille flow (sparse linear system), convection-diffusion oxygen transport along segments, Green's function oxygen transport in tissue, myogenic/metabolic regulation ODE, 1D structured-tree Windkessel for coronary outlet, FFR virtual computation, Fåhræus-Lindqvist effect (hematocrit-dependent viscosity).
- **Datasets:** UK Biobank coronary CTA (subset) (https://www.ukbiobank.ac.uk); PhysioNet coronary pressure/flow waveforms (https://physionet.org); Vascular Model Repository coronary geometries (http://www.vascularmodel.com); MICCAI coronary artery tracking challenge datasets (grand-challenge.org).
- **Starter repos/tools:** SimVascular (https://github.com/SimVascular/svFSI) — coronary flow boundary conditions (structured tree); HemeLB (https://github.com/hemelb-codes/hemelb) — sparse vascular LBM for microvascular beds; APBS (https://github.com/Electrostatics/apbs) — electrostatics analogy for oxygen transport; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — coronary CFD with custom UDF.
- **CUDA libraries & GPU pattern:** cuSPARSE for network flow linear system (sparse symmetric positive definite); cuSPARSE SpMV for iterative CG; CUDA Thrust for per-segment oxygen PDE; pattern: one thread per vessel segment for transport update, shared memory for branching connectivity.

---

### 6.21 Microcirculation & Oxygen Transport 🟡 · Active R&D
- **Deep dive:** Oxygen delivery from red blood cells to tissue parenchyma involves convection in capillaries, diffusion through capillary walls and interstitium (Krogh cylinder / Green's function models), and intracellular O₂ reaction/consumption (Michaelis-Menten kinetics). A realistic tissue volume (~1 mm³) contains thousands of capillaries forming a 3D network; GPU parallelism is applied to the per-segment convection-diffusion solves and the volumetric Green's function integrals (which are an O(N²) operation accelerated to O(N log N) via multipole or GPU-NUFFT).
- **Key algorithms:** Krogh cylinder O₂ transport, Green's function method (Secomb Hsu), 1D convection-diffusion along capillary segments, Michaelis-Menten O₂ consumption, fast multipole method (FMM) for Green's function sums, hemoglobin saturation curve (Hill equation), hematocrit-dependent RBC flux partitioning.
- **Datasets:** Vascular Model Repository (http://www.vascularmodel.com); two-photon microscopy microvascular datasets from Allen Institute (https://portal.brain-map.org); PhysioNet oxygen saturation waveforms (https://physionet.org); published microvascular network datasets (Secomb group, verify at secomb.org).
- **Starter repos/tools:** HemeLB (https://github.com/hemelb-codes/hemelb) — sparse LBM for capillary flow; USERMESO-2.0 (https://github.com/AnselGitAccount/USERMESO-2.0) — GPU red blood cell hemodynamics with deformable membranes; APBS (https://github.com/Electrostatics/apbs) — electrostatics solver repurposable for O₂ diffusion; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — volume-average tissue oxygenation.
- **CUDA libraries & GPU pattern:** CUDA NUFFT or FMM (cuFMM) for Green's function O₂ sums; custom CUDA kernels for per-segment RBC oxygen release; cuSPARSE for network flow solve; pattern: segment-parallel threads for convection update + shared-memory reduction for junction mass balance.

---

### 6.22 Bone Remodeling Simulation 🟡 · Active R&D
- **Deep dive:** Bone continually remodels in response to mechanical loading: osteoclasts resorb bone and osteoblasts form new bone in a coupled feedback loop mediated by RANKL/OPG signaling. GPU simulation enables voxel-level finite element analysis of trabecular bone microstructure (µCT at 10–50 µm resolution yields 10⁸ voxels) and tracking remodeling over years of simulated time. Topology optimization algorithms (SIMP) on GPU-FEM underlie both bone remodeling models and prosthesis design.
- **Key algorithms:** Mechano-regulation theory (Prendergast/Huiskes), local strain energy density (SED) remodeling rule, cellular automata bone remodeling, RANKL/OPG ODE signaling network, nonlinear FEM for bone microstructure, SIMP topology optimization, homogenization for apparent stiffness.
- **Datasets:** PhysioNet bone-related datasets (https://physionet.org); OsteoArthritis Initiative (OAI) µCT and radiograph dataset (https://nda.nih.gov/oai/); BoneJ plugin morphometric datasets (https://bonej.org); MICCAI bone segmentation challenge datasets (grand-challenge.org).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — nonlinear FEM for bone and cartilage; FreeFEM++ GPU extensions (verify URL) — PDE solver adaptable to remodeling; VoxFEM (verify URL — GPU voxel FEM from ETH Zurich research group); OpenFOAM for fluid-structure poroelastic bone (https://github.com/OpenFOAM/OpenFOAM-dev).
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for per-voxel SED computation; cuSPARSE for voxel FEM assembly (structured sparse); cuSOLVER PCG for linear system; pattern: 3D CUDA thread grid matching voxel layout, shared memory for element stiffness assembly.

---

### 6.23 Glucose-Insulin Dynamics & Artificial Pancreas 🟡 · Active R&D
- **Deep dive:** Type 1 diabetes management via a closed-loop artificial pancreas requires real-time simulation of glucose-insulin dynamics (Bergman minimal model, UVA/Padova T1D simulator) for controller design, in-silico trial, and reinforcement learning (RL) training. GPU acceleration enables parallel virtual patient cohort simulation for RL policy optimization and Monte Carlo variability analysis. The UVA/Padova simulator has been FDA-accepted for in-silico clinical trials.
- **Key algorithms:** Bergman minimal model (3-compartment ODE), UVA/Padova T1D simulator (13-compartment ODE), PID and model-predictive control (MPC), deep RL (PPO, SAC) for insulin dosing policy, glucose meal appearance (Gastric Emptying model), Kalman filter for CGM noise filtering.
- **Datasets:** OhioT1DM dataset — 12-week CGM + insulin data for 12 T1D subjects (https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html); JAEB CGMS datasets (https://public.jaeb.org); simglucose simulator virtual patient population (https://github.com/jxx123/simglucose); DirecNet CGM datasets (https://public.jaeb.org/direcnet).
- **Starter repos/tools:** simglucose (https://github.com/jxx123/simglucose) — Python UVA/Padova T1D simulator, gym environment for RL; GluCoEnv (https://github.com/chirathyh/GluCoEnv) — GPU-accelerated glucose control RL environment (PyTorch); G2P2C (https://github.com/RL4H/G2P2C) — RL artificial pancreas; OpenAPS oref0 (https://github.com/openaps/oref0) — open-source reference algorithm.
- **CUDA libraries & GPU pattern:** Batched ODE integration (cusolve / custom RK4 kernel) for ensemble of virtual patients; cuRAND for meal disturbance sampling; PyTorch GPU for RL policy network training; pattern: embarrassingly parallel virtual patient simulation—one CUDA thread per patient per time step.

---

### 6.24 Reaction-Diffusion Morphogenesis (Turing Patterns) 🟢 · Established
- **Deep dive:** Turing's 1952 reaction-diffusion system produces spatial patterns (spots, stripes, labyrinthine) from uniform initial conditions through short-range activation and long-range inhibition. Biological applications include skin pigmentation, hair follicle spacing, digit patterning, and cortical folding. GPU simulation on large 2D/3D domains enables the parameter sweep needed to map pattern-forming regions of parameter space and to study stochastic effects on pattern selection.
- **Key algorithms:** Turing activator-inhibitor ODE (Gierer-Meinhardt, Schnakenberg, Gray-Scott), explicit or semi-implicit Euler FD, 5-point/7-point Laplacian stencil, Turing instability linear stability analysis (dispersion relation), stochastic Turing patterns (reaction-diffusion master equation), level-set for 3D surface reaction-diffusion.
- **Datasets:** Synthetic datasets generated by simulation (no dedicated repository); pigmentation pattern image datasets (leopard, zebrafish from public image sources); cortical folding atlases from HCP (https://db.humanconnectome.org); DANDI morphogenesis imaging (https://dandiarchive.org).
- **Starter repos/tools:** Custom CUDA stencil kernel (textbook starting point — NVIDIA cuda-samples: https://github.com/NVIDIA/cuda-samples); VCell (https://vcell.org) — GUI reaction-diffusion PDE simulator with spatial stochastic mode; MOOSE (https://github.com/BhallaLab/moose-core) — compartmental spatial simulation; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — stochastic Turing pattern simulation.
- **CUDA libraries & GPU pattern:** Custom 2D/3D CUDA stencil kernels with halo-exchange; texture memory for read-only species arrays; shared memory for 7-point stencil tile computation; CUDA Thrust for reduction (global mass conservation check); pattern: 3D thread-block tiling, one thread per grid cell.

---

### 6.25 Liver & Kidney Perfusion Modeling 🟡 · Active R&D
- **Deep dive:** Liver lobules and kidney nephrons are structurally repetitive functional units that process blood to clear metabolites, drugs, and toxins. GPU simulation of drug/toxin clearance across millions of sinusoidal segments in a liver or tubular segments in a nephron network enables virtual pharmacotoxicology and organ-on-chip digital twins. Oxygen-zone-specific metabolism (periportal vs. centrilobular) and countercurrent exchange in the renal medullary vasa recta add physiological complexity.
- **Key algorithms:** Zonal liver sinusoid transport model, countercurrent exchange (renal medullary), convection-diffusion-reaction along network segments, Michaelis-Menten hepatic clearance, filtration-reabsorption-secretion nephron model, 3D lobular vascular network flow, PBPK liver/kidney sub-model.
- **Datasets:** Human Protein Atlas liver expression (https://www.proteinatlas.org); HMDB liver metabolomics (https://hmdb.ca); Open Systems Pharmacology PBPK model library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); PhysioNet renal function datasets (https://physionet.org).
- **Starter repos/tools:** Open Systems Pharmacology Suite (https://github.com/Open-Systems-Pharmacology) — organ-level PBPK with liver/kidney compartments; mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — ODE-based organ pharmacokinetics; SimVascular (https://github.com/SimVascular/svFSI) — vascular tree flow for portal vein; HemeLB (https://github.com/hemelb-codes/hemelb) — microvessel LBM for sinusoidal flow.
- **CUDA libraries & GPU pattern:** Batch ODE (one thread per lobule unit or nephron segment); cuSPARSE for lobular network linear system; custom CUDA kernels for Michaelis-Menten reaction in each zone; pattern: hierarchical parallelism—CUDA blocks per lobule, threads per sinusoidal segment.

---

### 6.26 Virtual Population Generation & Sensitivity Analysis 🟡 · Active R&D
- **Deep dive:** Virtual patient populations are created by sampling physiological parameter distributions (body weight, organ volumes, enzyme expression, sex, age) from measured databases (NHANES, WHO) and propagating them through PBPK/PD models to generate simulated trial cohorts. Sobol sensitivity analysis requires O(N×(2k+2)) model evaluations for k parameters—typically millions of forward ODE integrations. GPU batch simulation reduces this from days to hours.
- **Key algorithms:** Latin hypercube sampling (LHS), Sobol quasi-random sequences, Morris one-at-a-time elementary effects, Sobol variance-based sensitivity indices, polynomial chaos expansion (PCE), Gaussian process surrogate (emulator), MCMC parameter estimation (Metropolis-Hastings, NUTS), bootstrap confidence intervals.
- **Datasets:** NHANES anthropometric/physiological data (https://www.cdc.gov/nchs/nhanes/); WHO growth reference datasets (https://www.who.int/tools/growth-reference-data-for-5to19-years); OSP PBPK model library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); FDA drug label PK data (https://www.fda.gov/drugs).
- **Starter repos/tools:** SALib sensitivity analysis library (https://github.com/SALib/SALib) — Morris, Sobol, FAST methods for Python; Open Systems Pharmacology (https://github.com/Open-Systems-Pharmacology) — virtual population creation module (PK-Sim); mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — fast ODE PK batch simulation; SUNDIALS batch CVODE (https://github.com/LLNL/sundials) — GPU ODE ensemble.
- **CUDA libraries & GPU pattern:** cuRAND for Sobol/Halton quasi-random sequences; batch CVODE GPU for ensemble ODE; cuBLAS for PCE coefficient matrix operations; pattern: one CUDA thread per virtual patient, Sobol sensitivity via GPU-parallel model evaluations; thrust::transform for per-sample output extraction.

---

### 6.27 Parameter Estimation & Data Assimilation for Physiological Models 🟡 · Active R&D
- **Deep dive:** Fitting ODE/PDE physiological models to patient-specific clinical data (ECG, pressure waveforms, biomarker time series) requires repeated forward simulation within an optimization or Bayesian inference loop. Ensemble Kalman filters (EnKF) update an ensemble of 50–500 model states in parallel with incoming observations; unscented Kalman filters (UKF) propagate 2N+1 sigma points. GPU acceleration of the forward model ensemble is the bottleneck.
- **Key algorithms:** Ensemble Kalman filter (EnKF), unscented Kalman filter (UKF), particle filter (sequential Monte Carlo), adjoint-based gradient optimization (L-BFGS), variational data assimilation (4D-Var), Gaussian process emulator surrogate, Bayesian optimization, trust-region methods.
- **Datasets:** PhysioNet MIMIC clinical waveforms (https://physionet.org); UK Biobank cardiac functional parameters (https://www.ukbiobank.ac.uk); Zenodo cardiac mechanics emulation dataset (https://zenodo.org/records/7075055); openCARP community experiments (https://opencarp.org/community/community-experiments).
- **Starter repos/tools:** SUNDIALS/CVODES (https://github.com/LLNL/sundials) — sensitivity-aware ODE integrator for adjoint gradient; simcardems (https://github.com/ComputationalPhysiology/simcardems) — cardiac twin with parameter fitting; SALib (https://github.com/SALib/SALib) — sensitivity analysis for parameter prioritization; PyMC (https://github.com/pymc-devs/pymc) — probabilistic programming with GPU via JAX/Aesara backend.
- **CUDA libraries & GPU pattern:** Batch forward ODE on GPU (ensemble members); cuBLAS for EnKF covariance update (N×N matrix operations); cuSOLVER for Kalman gain; CUDA Thrust for particle resampling; pattern: ensemble-parallel forward solves + host-side EnKF analysis step.

---

---

## 7. Medical AI & Clinical Deep Learning

### 7.1 Diagnostic Imaging Classifier 🟢 · Established

- **Deep dive:** Trains convolutional and transformer-based networks to classify pathologies (malignancy, disease grade, anatomical anomaly) from 2D/3D medical images — CT, MRI, X-ray, ultrasound. GPUs provide the tensor-parallel matrix multiply needed to process high-resolution volumetric input in minibatches; a single 512×512 CT slice stack can reach tens of millions of pixels. Backbone convolutions (3D U-Net, ResNet-50, EfficientNet, ViT-B) are the compute-dominant operation, mapping directly onto CUDA tensor cores. Mixed-precision FP16/BF16 training via cuDNN doubles effective throughput versus FP32 while preserving classification accuracy. Inference on edge devices is further accelerated with TensorRT INT8 quantisation.
- **Key algorithms:** 3D convolutional neural networks (ResNet-3D, DenseNet), Vision Transformers (ViT, Swin-T), EfficientNet, data augmentation with random affine/elastic transforms, AUC-optimised losses, Grad-CAM explainability, TTA (test-time augmentation) ensembling.
- **Datasets:**
  - MIMIC-CXR — 227,827 labelled chest X-ray studies with radiology reports from Beth Israel Deaconess (https://physionet.org/content/mimic-cxr/)
  - CheXpert — 224,316 chest X-rays from Stanford, 14 pathology labels (https://stanfordmlgroup.github.io/competitions/chexpert/)
  - LIDC-IDRI — 1,018 CT lung nodule cases with radiologist consensus annotations (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
  - The Cancer Imaging Archive (TCIA) — multi-modal oncology imaging across dozens of curated collections (https://www.cancerimagingarchive.net/)
- **Starter repos/tools:**
  - MONAI (https://github.com/Project-MONAI/MONAI) — PyTorch-native medical imaging framework with C++/CUDA extensions for resampling and transforms
  - TorchXRayVision (https://github.com/mlmed/torchxrayvision) — pre-trained chest X-ray models, loaders for CheXpert/MIMIC-CXR
  - nnU-Net (https://github.com/MIC-DKFZ/nnUNet) — auto-configuring segmentation/classification baseline that wins most medical imaging benchmarks
  - TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — 104-structure CT segmentation built on nnU-Net
- **CUDA libraries & GPU pattern:** cuDNN for convolution kernels, NCCL for multi-GPU data-parallel training, TensorRT for deployment; pattern: minibatch data parallelism with NCCL all-reduce, optional model parallelism for 3D volumes that exceed single-GPU VRAM.

---

### 7.2 Drug-Target Interaction Prediction (GNN) 🟡 · Active R&D

- **Deep dive:** Predicts whether a small molecule (drug) will bind to a protein target and estimates binding affinity (Kd/Ki) or binary interaction labels. Molecular graphs have irregular topology, so graph neural message-passing aggregates neighbour features in parallel across thousands of candidate pairs simultaneously on GPU. Protein sequences can be encoded via transformer attention (ESM-2, ProtTrans) whose quadratic attention is accelerated by Flash Attention on CUDA. The bottleneck is the cross-attention between drug graph embeddings and protein sequence embeddings over large virtual screening libraries (millions of compounds), which maps to batched sparse matrix operations. GPU throughput determines how many candidates can be scored per day in drug discovery pipelines.
- **Key algorithms:** Message Passing Neural Networks (MPNN), Graph Attention Networks (GAT), Directed Message Passing (DMPNN), transformer cross-attention, contrastive DTI objectives, Graph Isomorphism Networks (GIN), graph-level pooling, Bayesian hyperparameter optimisation.
- **Datasets:**
  - BindingDB — ~2.9 million measured binding affinities for drug-target pairs (https://www.bindingdb.org/)
  - ChEMBL — curated bioactivity database with >20M activity records (https://www.ebi.ac.uk/chembl/)
  - Davis Kinase Dataset — kinase inhibitor affinities for 442 kinases × 68 drugs (verify URL)
  - KIBA — integrated kinase inhibitor bioactivity benchmark (verify URL)
- **Starter repos/tools:**
  - DeepPurpose (https://github.com/kexinhuang12345/DeepPurpose) — 15 drug/protein encoders, 50+ architectures for DTI
  - TorchDrug (https://github.com/DeepGraphLearning/torchdrug) — GPU-accelerated graph learning library for drug discovery
  - DGL-LifeSci (https://github.com/awslabs/dgl-lifesci) — DGL-based molecular GNN toolkit with CUDA-backed sparse ops
  - DTA-GNN (https://github.com/lennylv/DTA-GNN) — toolkit for target-specific DTA dataset construction and GNN training
- **CUDA libraries & GPU pattern:** DGL/PyG sparse adjacency ops on GPU, Flash Attention 2 for protein encoders, cuDNN for MLP heads; pattern: heterogeneous data parallelism (drug batch × protein batch), optional multi-GPU model parallelism for large protein encoders.

---

### 7.3 Clinical NLP over Notes & Records 🟢 · Established

- **Deep dive:** Applies transformer language models to de-identified electronic health record (EHR) free-text — discharge summaries, radiology reports, nursing notes — for named entity recognition, relation extraction, ICD coding, phenotyping, and clinical event prediction. BERT-style pretraining on billions of clinical tokens (MIMIC-IV notes) is highly GPU-bound: multi-head self-attention scales O(n²) in sequence length, making long-document clinical notes particularly expensive. Flash Attention reduces this cost from O(n²) to near-linear in memory, enabling 8192-token context windows. The parallel bottleneck is the batched matrix multiplications in each transformer layer, exploiting GPU tensor cores. Fine-tuning on task-specific clinical benchmarks (NER, RE) requires additional GPU compute for gradient accumulation across long sequences.
- **Key algorithms:** BERT masked language modelling, next-sentence prediction, Flash Attention, Rotary Positional Embeddings (RoPE), BIO-tagging for NER, CRF output layers, relation extraction with span pairs, multi-label ICD classification, instruction-tuning with clinical instruction sets.
- **Datasets:**
  - MIMIC-IV Clinical Notes — 331,794 de-identified patient notes from Beth Israel Deaconess (https://physionet.org/content/mimic-iv-note/)
  - i2b2/n2c2 NLP Challenge Datasets — named entity, coreference, and relation tasks in clinical text (https://n2c2.dbmi.hms.harvard.edu/)
  - MTSamples — 4,999 transcribed medical reports across 40 specialties (https://mtsamples.com/)
  - MedQA / MedMCQA — medical question answering benchmarks for evaluating clinical LLMs (verify URL)
- **Starter repos/tools:**
  - BioClinicalBERT (https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — BERT pretrained on MIMIC-III notes
  - Clinical ModernBERT (https://github.com/Simonlee711/Clinical_ModernBERT) — ModernBERT fine-tuned on 13B tokens of PubMed + MIMIC-IV with 8192-token context
  - medSpaCy (https://github.com/medspacy/medspacy) — spaCy-based clinical NLP pipeline with GPU inference support
  - GatorTron (https://huggingface.co/UFNLP/gatortron-base) — large clinical LLM pretrained on 82B tokens of clinical text (verify URL)
- **CUDA libraries & GPU pattern:** Flash Attention 2, cuBLAS for GEMM-dominated transformer layers, NCCL for data-parallel pretraining; pattern: data parallelism across multiple A100/H100 GPUs, gradient checkpointing to fit long-context batches in VRAM.

---

### 7.4 Medical Image Synthesis & Augmentation (Generative) 🟡 · Active R&D

- **Deep dive:** Generates synthetic medical images to augment scarce annotated datasets or enable domain transfer — e.g., synthesising MRI from CT, generating rare pathology variants, or creating paired segmentation masks. Generative models (GANs, diffusion models, VAEs) are training-compute-intensive: a diffusion UNet iterates 1000 denoising steps at full 3D resolution, with each forward pass bottlenecked by 3D convolutions. GANs require simultaneous forward/backward passes through discriminator and generator on the same GPU batch. GPU parallelism over the spatial dimensions of 3D volumes provides the necessary throughput. Diffusion models in particular benefit from mixed-precision training and gradient checkpointing to fit large 3D UNets in GPU memory.
- **Key algorithms:** Denoising Diffusion Probabilistic Models (DDPM), Score-Based Generative Models (SGBM), CycleGAN, Pix2Pix, VQVAE, Latent Diffusion Models (LDM), FID/FRD evaluation metrics, style-transfer augmentation.
- **Datasets:**
  - BraTS (Brain Tumor Segmentation) — multi-institutional MRI with ground-truth tumour masks (https://www.synapse.org/Synapse:syn51156910/wiki/)
  - ADNI (Alzheimer's Disease Neuroimaging Initiative) — longitudinal MRI/PET with clinical data (https://adni.loni.usc.edu/)
  - TCIA — public CT/MRI collections enabling synthesis training (https://www.cancerimagingarchive.net/)
  - GaNDLF-Synth benchmark (https://arxiv.org/abs/2410.00173) — multi-site synthetic pathology image benchmark (verify URL)
- **Starter repos/tools:**
  - MONAI Generative (https://github.com/Project-MONAI/GenerativeModels) — diffusion, VQVAE, GAN modules on GPU
  - SynthSeg (https://github.com/BBillot/SynthSeg) — label-conditioned MRI synthesis for segmentation
  - MedSynAnalyser / StableDiffusion-Medical (verify URL) — medical image fine-tuning pipelines for latent diffusion
  - HealthyGAN / CycleGAN-3D (verify URL) — unpaired MRI-CT translation
- **CUDA libraries & GPU pattern:** cuDNN 3D grouped convolutions, FlashAttention for diffusion attention blocks, NCCL for multi-GPU; pattern: data-parallel training with gradient checkpointing, NVLink for A100/H100 inter-GPU communication during large 3D batch training.

---

### 7.5 Federated Learning for Healthcare 🟡 · Active R&D

- **Deep dive:** Trains a single global model across multiple hospitals without sharing raw patient data: each site trains on local data and sends only model gradients or weights to a central aggregator. The GPU bottleneck on each client is identical to standard local training; additional communication cost arises from the aggregation step. NVIDIA FLARE orchestrates GPU-based local training with differential privacy noise injection and secure aggregation. Heterogeneous GPU fleets across hospitals (V100 at one site, A100 at another) require adaptive batch sizing and mixed-precision logic. The primary research challenge is handling statistical data heterogeneity (non-IID distributions) while maintaining convergence.
- **Key algorithms:** FedAvg, FedProx, SCAFFOLD, FedNova, personalised federated learning, differential privacy (Gaussian mechanism, moments accountant), secure aggregation with homomorphic encryption, communication compression (gradient sparsification, quantisation).
- **Datasets:**
  - TCGA (The Cancer Genome Atlas) — multi-institutional genomics + histopathology (https://www.cancer.gov/tcga)
  - MIMIC-IV — EHR data used in federated simulation across synthetic partitions (https://physionet.org/content/mimiciv/)
  - NIH Chest X-ray Dataset — 112,120 chest X-rays for FL benchmarks (https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - Medical Segmentation Decathlon — multi-task dataset used in FL challenges (http://medicaldecathlon.com/)
- **Starter repos/tools:**
  - NVIDIA FLARE (https://github.com/NVIDIA/NVFlare) — production-grade federated learning SDK with GPU-native training loops
  - OpenFL (https://github.com/securefederatedai/openfl) — Intel/Linux Foundation FL framework supporting PyTorch/TF on GPU
  - Flower (https://github.com/adap/flower) — lightweight, framework-agnostic FL with GPU support
  - PySyft (https://github.com/OpenMined/PySyft) — privacy-preserving FL with differential privacy on GPU
- **CUDA libraries & GPU pattern:** cuDNN for local model training, NCCL for efficient intra-site multi-GPU; pattern: data parallelism within site, synchronous or asynchronous gradient aggregation between sites via secure channels.

---

### 7.6 Survival Analysis & Risk Prediction 🟡 · Active R&D

- **Deep dive:** Estimates time-to-event outcomes (death, hospital readmission, disease progression) from longitudinal EHR data, imaging, or omics using neural extensions of Cox proportional hazards (DeepSurv), discrete-time survival models (DeepHit), and dynamic variational approaches (DySurv). The GPU bottleneck is batched forward passes through deep neural networks that process long time-series of irregular clinical observations. Computing the partial likelihood loss (Cox) requires sorting survival times and summing risk sets, which can be parallelised as a GPU prefix-sum operation. Large cohort training (>100k patients with hundreds of clinical features) sustains high GPU utilisation throughout.
- **Key algorithms:** Cox Proportional Hazards (DeepSurv), Discrete Survival (DeepHit), Dynamic Bayesian survival (DySurv), random survival forests on GPU, competing risks (Fine-Gray), Concordance Index (C-index) optimisation, deep conditional transformation models, inverse probability of censoring weighting (IPCW).
- **Datasets:**
  - UK Biobank — 500k participant longitudinal cohort with genetic, imaging, and EHR data (https://www.ukbiobank.ac.uk/)
  - SEER Database — cancer incidence and survival from US National Cancer Institute (https://seer.cancer.gov/)
  - eICU Collaborative Research Database — 200k+ critical care admissions across 200 hospitals (https://eicu-crd.mit.edu/)
  - TCGA clinical outcomes — survival labels linked to molecular profiling (https://www.cancer.gov/tcga)
- **Starter repos/tools:**
  - PyCox (https://github.com/havakv/pycox) — GPU-accelerated DeepSurv, DeepHit, MTLR implementations in PyTorch
  - Lifelines (https://github.com/CamDavidsonPilon/lifelines) — classical survival library (CPU); pairs with GPU model backends
  - DySurv (verify URL) — CVAE-based dynamic survival from EHR time series
  - scikit-survival (https://github.com/sebp/scikit-survival) — ensemble survival methods; GPU ensemble via XGBoost integration
- **CUDA libraries & GPU pattern:** cuDNN for network forward/backward, custom CUDA prefix-sum kernels for Cox risk set computation, Thrust for efficient sorting of survival times; pattern: data-parallel minibatch with custom loss kernel.

---

### 7.7 Multi-Omics Integration 🟡 · Active R&D

- **Deep dive:** Combines heterogeneous molecular data layers — genomics (SNP/CNV), transcriptomics (RNA-seq), proteomics, metabolomics, and epigenomics — to predict disease subtype, drug response, or patient outcome. Integrating these layers requires jointly embedding high-dimensional sparse matrices (gene expression: 20k genes × 10k patients) with dense low-dimensional clinical vectors. GPUs accelerate the large embedding layers and transformer attention that learn cross-modal correspondences; a single multi-omics autoencoder can have hundreds of millions of parameters when modelling all layers simultaneously. scGPT-style tokenisation of omics measurements treats genes as tokens and uses CUDA-accelerated attention. Sparse input matrices benefit from cuSPARSE SpMM operations.
- **Key algorithms:** Multi-modal autoencoders (VAE, VQVAE), Graph Neural Networks over molecular interaction networks, transformer tokenisation (scGPT, mosGraphGPT), MOFA+ factor analysis, multi-task learning across omics, contrastive multi-omics pre-training, pathway-guided sparse attention.
- **Datasets:**
  - TCGA Pan-Cancer Atlas — genomic, transcriptomic, proteomic data for 33 cancer types (https://www.cancer.gov/tcga)
  - GEO (Gene Expression Omnibus) — 5M+ omics samples across species/conditions (https://www.ncbi.nlm.nih.gov/geo/)
  - CPTAC (Clinical Proteomic Tumor Analysis Consortium) — proteogenomics across tumour types (https://proteomics.cancer.gov/programs/cptac)
  - ENCODE — chromatin, transcription factor, and RNA datasets (https://www.encodeproject.org/)
- **Starter repos/tools:**
  - scGPT (https://github.com/bowang-lab/scGPT) — GPT-style multi-omics foundation model with GPU pretraining
  - MOFA+ (https://github.com/bioFAM/MOFA2) — factor analysis for multi-omics (CPU; GPU via JAX backend)
  - TF-DWGNet (https://arxiv.org/abs/2509.16301) — directed weighted GNN for multi-omics cancer subtype classification (verify URL)
  - MOLI / Concrete Autoencoder (https://github.com/mims-harvard/Madrigal) — multi-omics latent integration (verify URL)
- **CUDA libraries & GPU pattern:** cuSPARSE for sparse omics matrices, Flash Attention for gene-token sequences, NCCL multi-GPU; pattern: column-parallel embedding for gene dimension, row-parallel for sample dimension.

---

### 7.8 Reinforcement Learning for Treatment Policies 🟡 · Active R&D

- **Deep dive:** Learns optimal dynamic treatment regimens — sepsis fluid and vasopressor dosing, mechanical ventilation settings, chemotherapy scheduling — from retrospective EHR trajectories using offline reinforcement learning. The GPU bottleneck is the batch Q-network or policy gradient updates across thousands of patient trajectories with hundreds of time steps each. Offline RL (Conservative Q-Learning, BEAR, TD3+BC) requires sampling large replay buffers and computing bootstrapped targets in parallel. Digital twin environments for safe exploration run population-level ODE simulations accelerated on GPU. Each policy evaluation step scores all actions for all patients simultaneously on GPU.
- **Key algorithms:** Conservative Q-Learning (CQL), Behaviour Constrained Policy Optimisation (BCPO), TD3+BC, Proximal Policy Optimisation (PPO) in simulation, Dueling DQN, Soft Actor-Critic (SAC), inverse RL, doubly-robust off-policy evaluation, OGSRL (Offline Guarded Safe RL).
- **Datasets:**
  - MIMIC-IV — ICU trajectories for sepsis, ventilation, and medication studies (https://physionet.org/content/mimiciv/)
  - eICU-CRD — multi-site ICU cohort for cross-hospital policy generalisation (https://eicu-crd.mit.edu/)
  - MIMIC-Sepsis benchmark (https://arxiv.org/abs/2510.24500) — curated sepsis trajectory benchmark from MIMIC
  - AmsterdamUMCdb — 23k ICU patients, open-access (https://amsterdammedicaldatascience.nl/)
- **Starter repos/tools:**
  - d3rlpy (https://github.com/takuseno/d3rlpy) — offline RL library with CUDA-accelerated Q-learning (PyTorch)
  - MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) — standardised MIMIC-III/IV feature extraction for RL
  - AI Clinician (https://github.com/matthieukomorowski/AI_Clinician) — seminal offline RL sepsis treatment repo
  - HealthGym (https://github.com/healthylaife/healthgym) — clinical offline RL environments built on MIMIC data
- **CUDA libraries & GPU pattern:** cuDNN for policy/Q-networks, cuBLAS for experience replay batch matmuls, custom CUDA kernels for parallelised Bellman backup over large replay buffers; pattern: GPU replay buffer sampling with pinned memory for fast CPU→GPU transfer.

---

### 7.9 Real-Time Edge Inference for Medical Devices 🟡 · Active R&D

- **Deep dive:** Deploys neural networks on embedded GPUs (NVIDIA Jetson, AMD Versal) or medical device SoCs for real-time inference — ECG arrhythmia detection, pulse oximetry anomaly, surgical robot vision, ultrasound B-mode AI. The challenge is matching model latency to physiological sampling rates (e.g., 500 Hz ECG requires <2 ms inference). TensorRT INT8 quantisation reduces model size 4× with minimal accuracy loss. Layer fusion fuses sequential convolutions, activations, and normalisation into single CUDA kernels, eliminating memory bandwidth bottlenecks. NVIDIA Jetson Orin delivers 275 TOPS at 60 W, enabling full clinical-grade CNN inference locally without cloud round-trips.
- **Key algorithms:** Post-training quantisation (INT8, INT4), knowledge distillation, neural architecture search for edge (MobileNetV3, EfficientDet-Lite), layer fusion, structured pruning, TensorRT engine optimisation, latency-aware NAS.
- **Datasets:**
  - PhysioNet Challenge datasets — ECG, SpO2, EEG for device validation (https://physionet.org/)
  - CAMUS cardiac ultrasound segmentation dataset (https://www.creatis.insa-lyon.fr/Challenge/camus/)
  - EyePACS retinal fundus — used for on-device DR screening validation (verify URL)
  - MIMIC-III Waveform Database — high-freq bedside monitor signals (https://physionet.org/content/mimicdb/)
- **Starter repos/tools:**
  - TensorRT (https://github.com/NVIDIA/TensorRT) — inference optimisation with INT8 calibration and layer fusion
  - NVIDIA Jetson SDK (https://developer.nvidia.com/embedded/jetpack) — Jetson-optimised libraries for edge GPU inference
  - MONAI Deploy (https://github.com/Project-MONAI/monai-deploy) — clinical AI deployment framework with TensorRT backend
  - OpenVINO (https://github.com/openvinotoolkit/openvino) — Intel edge inference toolkit for x86+iGPU devices
- **CUDA libraries & GPU pattern:** TensorRT INT8 for quantised inference, cuDNN for on-device convolutions, Triton Inference Server for multi-model serving; pattern: streaming inference pipeline with zero-copy pinned memory between sensor DMA and GPU.

---

### 7.10 Physiological Signal & Waveform Analysis 🟡 · Active R&D

- **Deep dive:** Processes continuous high-frequency physiological waveforms — ECG (500–2000 Hz), EEG (256–2048 Hz), arterial blood pressure, photoplethysmography — for automated diagnosis, anomaly detection, and prognostication. Long waveform segments (minutes to hours) require 1D temporal convolutions or transformer attention over thousands of time steps; both operations are GPU-bound. Processing multi-lead ECG simultaneously (12 leads × 5000 samples) as a 2D image enables CNN classification with no waveform-specific code. Batch processing of thousands of 24-hour Holter monitors in parallel on GPU is the primary throughput bottleneck in clinical annotation pipelines.
- **Key algorithms:** 1D ResNet / Inception, temporal convolutional networks (TCN), WaveNet, Bidirectional LSTM, self-supervised waveform pretraining (wav2vec 2.0 for ECG), Short-Time Fourier Transform (STFT) + CNN, multi-scale attention, event detection with anchor-free detection heads.
- **Datasets:**
  - PhysioNet Computing in Cardiology Challenge 2021 — 12-lead ECG from multiple cohorts (https://physionet.org/content/challenge-2021/)
  - MIMIC-IV-ECG — 800k+ ECGs from MIMIC patients (https://physionet.org/content/mimic-iv-ecg/)
  - PTB-XL — 21,837 12-lead ECGs with cardiologist labels (https://physionet.org/content/ptb-xl/)
  - Temple University EEG Corpus (TUEG) — 20k+ hours of clinical EEG (https://isip.piconepress.com/projects/tuh_eeg/)
- **Starter repos/tools:**
  - ECG-FM (https://github.com/bowang-lab/ecg-fm) — wav2vec-based ECG foundation model, 90M params, GPU-pretrained
  - ESI (https://github.com/comp-well-org/ESI) — multimodal ECG + text contrastive pretraining foundation model
  - CLEF ECG (https://github.com/Nokia-Bell-Labs/ecg-foundation-model) — single-lead ECG foundation model pretrained on 161k MIMIC patients
  - MNE-Python (https://github.com/mne-tools/mne-python) — EEG/MEG processing; GPU via deep learning backends
- **CUDA libraries & GPU pattern:** cuFFT for Fourier-domain convolutions on waveforms, cuDNN for 1D temporal convolutions, Flash Attention for long-sequence transformers; pattern: data-parallel batch processing across thousands of waveform windows, streaming input pipeline from waveform database.

---

### 7.11 Medical Foundation-Model Pretraining & Inference 🟡 · Active R&D

- **Deep dive:** Pretrains large-scale (1B–70B parameter) language, vision, or multimodal models on domain-specific medical corpora — PubMed, MIMIC clinical notes, radiology report databases, pathology image collections — to produce general-purpose medical representations. Pretraining is massively GPU-bound: the matrix multiplications in transformer attention and feed-forward layers constitute >95% of FLOPs. Tensor-parallel and pipeline-parallel model partitioning across hundreds of A100/H100 GPUs (via Megatron-LM or DeepSpeed) is necessary for 70B-parameter models. Inference serving uses Flash Attention, continuous batching (vLLM), and INT8/GPTQ quantisation to handle concurrent clinical queries.
- **Key algorithms:** Autoregressive pretraining (GPT), masked language modelling (BERT), instruction tuning (SFT + RLHF), Vision-Language Contrastive pretraining (CLIP, FLAVA), Mixture-of-Experts (MoE), FlashAttention-2, LoRA/QLoRA fine-tuning, GPTQ quantisation.
- **Datasets:**
  - PubMed Central Open Access — 4M+ full biomedical articles (https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)
  - MIMIC-IV Notes — 331,794 clinical notes (https://physionet.org/content/mimic-iv-note/)
  - The Pile: Pile-MedMent / S2ORC — broad scientific pretraining corpora (https://pile.eleuther.ai/)
  - OpenPath / PathCap — pathology image-caption pairs for vision-language pretraining (verify URL)
- **Starter repos/tools:**
  - MEDITRON (https://github.com/epfLLM/meditron) — Llama-2 70B adapted for medicine with GPU pretraining scripts
  - Awesome Healthcare Foundation Models (https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models) — curated model list
  - Awesome Foundation Models in Medical Imaging (https://github.com/xmindflow/Awesome-Foundation-Models-in-Medical-Imaging) — curated vision-language models
  - vLLM (https://github.com/vllm-project/vllm) — continuous batching inference engine for serving medical LLMs on GPU
- **CUDA libraries & GPU pattern:** Megatron-LM tensor parallelism, DeepSpeed ZeRO, Flash Attention 2, NCCL all-reduce; pattern: 3D parallelism (tensor × pipeline × data), NVLink high-bandwidth GPU fabric required.

---

### 7.12 Sepsis Early Warning System 🟡 · Active R&D

- **Deep dive:** Predicts the onset of sepsis 3–6 hours before clinical recognition from streaming ICU vitals, lab values, and medication records using recurrent or transformer architectures. The GPU bottleneck is batched forward passes through temporal models (LSTM, GRU, Transformer-XL) over thousands of patient time series simultaneously. Real-time deployment requires sub-second latency over continuously appended EHR streams. Processing irregular time-series (lab values arrive at non-uniform intervals) requires attention mechanisms that weigh observations by recency and relevance — these attention operations are CUDA-accelerated. Large training cohorts (>100k ICU admissions) sustain continuous GPU utilisation throughout training.
- **Key algorithms:** LSTM/GRU temporal classifiers, Transformer-XL for long EHR sequences, Temporal Fusion Transformers (TFT), missing-value imputation via learned decay, AUROC-calibrated threshold selection, early stopping with Clinical Early Warning Scores (qSOFA, SOFA) as baselines, conformal prediction for uncertainty.
- **Datasets:**
  - MIMIC-Sepsis benchmark (https://arxiv.org/abs/2510.24500) — curated sepsis trajectory subset of MIMIC-IV
  - eICU-CRD — 200k+ admissions, multi-site for generalisation testing (https://eicu-crd.mit.edu/)
  - PhysioNet/Computing in Cardiology Challenge 2019 — sepsis prediction from ICU time series (https://physionet.org/content/challenge-2019/)
  - HiRID — high-resolution ICU dataset from Bern University Hospital (https://physionet.org/content/hirid/)
- **Starter repos/tools:**
  - MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) — standardised MIMIC ICU feature tables
  - PyHealth (https://github.com/sunlabuiuc/PyHealth) — healthcare AI library with ICU prediction tasks on GPU
  - ETHOS (verify URL) — transformer-based sepsis prediction on EHR tokens
  - Temporal Fusion Transformer (https://github.com/jdb78/pytorch-forecasting) — multi-horizon temporal model with GPU support
- **CUDA libraries & GPU pattern:** cuDNN for LSTM/GRU cells, Flash Attention for transformer EHR models, Thrust for sorting irregular timestamps; pattern: padded minibatch of patient time series with masking, GPU-resident rolling window inference for real-time alerting.

---

### 7.13 Radiology Report Generation (Vision-Language) 🟡 · Active R&D

- **Deep dive:** Generates free-text radiology reports from chest X-rays, CT, or MRI scans by jointly encoding the image and decoding text autoregressively. This is a vision-language task requiring large cross-modal attention blocks: ViT image encoder + GPT-style decoder with cross-attention, both bottlenecked by CUDA tensor-core matrix multiplies. Generating a 200-word radiology report at inference requires hundreds of autoregressive decoder steps, each a full forward pass through a multi-layer transformer; batch decoding on GPU with KV-caching provides the necessary throughput for clinical deployment. Training requires paired image-report datasets and auxiliary pathology-label supervision, running on multi-GPU clusters.
- **Key algorithms:** Cross-modal attention, Vision Transformer (ViT) encoder, GPT decoder, contrastive image-text pretraining (CLIP), CheXpert labeller for evaluation, RadGraph F1 metric, layer-wise anatomical attention, chain-of-thought report generation, LEAD / LLaVA-TA architectures.
- **Datasets:**
  - MIMIC-CXR — 227,827 chest X-ray + report pairs (https://physionet.org/content/mimic-cxr/)
  - CheXpert — 224k X-rays with pathology labels (https://stanfordmlgroup.github.io/competitions/chexpert/)
  - OpenI — Indiana University chest X-ray + report dataset (https://openi.nlm.nih.gov/)
  - PadChest — 160k chest X-rays with 174-label taxonomy (https://bimcv.cipf.es/bimcv-projects/padchest/)
- **Starter repos/tools:**
  - Awesome-Radiology-Report-Generation (https://github.com/mk-runner/Awesome-Radiology-Report-Generation) — curated paper/dataset/code list
  - R2Gen / R2GenCMN (https://github.com/cuhksz-nlp/R2Gen) — seminal cross-modal radiology generation models
  - MIMIC-CXR multimodal repo (https://github.com/yuanditang/MIMIC-CXR) — ResNet + LLaMA-3.2 vision-instruction pipeline
  - CheXagent (verify URL) — instruction-tuned radiology report generation model
- **CUDA libraries & GPU pattern:** Flash Attention 2 for cross-modal attention, TensorRT for decoder inference acceleration, KV-cache with CUDA persistent memory; pattern: encoder-decoder parallelism with batched beam search on GPU.

---

### 7.14 Self-Supervised Image Pretraining for Medical Imaging 🟡 · Active R&D

- **Deep dive:** Pre-trains visual encoders on large unlabelled medical image collections using contrastive or masked image modelling objectives, so downstream tasks (classification, segmentation) require only small labelled datasets. SimCLR, MoCo-v3, DINO, and MAE all reduce to large batched matrix multiplications during projection head training and attention; MoCo avoids the large-batch requirement of SimCLR by using a momentum encoder and GPU-resident queue of negatives. Medical images differ from natural images (greyscale, 3D, complex domain shifts), requiring domain-specific augmentation policies. GPU memory is the primary constraint: SimCLR needs large batch (4096+) to fill the negative pool, demanding multi-GPU with NCCL all-reduce.
- **Key algorithms:** SimCLR, MoCo-v2/v3, BYOL, SimSiam, DINO, MAE (Masked Autoencoder), MoCo-CXR (chest X-ray adaptation), momentum encoder, projection head, cosine-similarity loss, online-vs-target network paradigm.
- **Datasets:**
  - ChestX-ray14 / NIH — 112k chest X-rays with 14 disease labels (https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - MIMIC-CXR — 227k X-rays for self-supervised pretraining (https://physionet.org/content/mimic-cxr/)
  - RadImageNet — 1.35M radiology images across CT/MRI/US for SSL pretraining (verify URL)
  - BraTS + TCIA — unlabelled MRI/CT volumes for 3D MAE pretraining (https://www.cancerimagingarchive.net/)
- **Starter repos/tools:**
  - MoCo-CXR (https://arxiv.org/abs/2010.05352) — MoCo applied to chest X-ray, code available (verify URL)
  - MONAI self-supervised (https://github.com/Project-MONAI/research-contributions) — MAE and contrastive pretraining for 3D medical images
  - DINO (https://github.com/facebookresearch/dino) — self-supervised ViT; adaptable to medical imaging
  - lightly (https://github.com/lightly-ai/lightly) — SSL framework supporting SimCLR/DINO/MoCo on GPU
- **CUDA libraries & GPU pattern:** NCCL all-reduce for multi-GPU negative queue synchronisation, cuDNN for backbone convolutions, cuBLAS for projection head; pattern: large-batch contrastive with NCCL gradient sync, or momentum queue stored in GPU SRAM.

---

### 7.15 Uncertainty & Out-of-Distribution Detection in Medical AI 🔴 · Frontier/Theoretical

- **Deep dive:** Quantifies model uncertainty for medical predictions to flag distribution shift (e.g., a new imaging protocol, rare pathology) and prevent silent failures in deployed AI. Bayesian approximations (MC Dropout, Deep Ensembles, SWAG) require multiple stochastic forward passes — naturally parallelised across ensemble members on GPU. Conformal prediction calibrates coverage guarantees without distribution assumptions, requiring GPU-parallelised score computation over large calibration sets. Energy-based OOD scoring on medical images computes a scalar energy per sample in a parallelised batch. The research challenge is calibrating uncertainty to actual clinical risk without access to labelled OOD data.
- **Key algorithms:** MC Dropout (Gal & Ghahramani), Deep Ensembles, SWAG (SWA-Gaussian), Normalising Flows for density estimation, Energy-Based Models, Mahalanobis OOD detection, Conformal Prediction (split, cross-conformal), Temperature Scaling, Label Smoothing.
- **Datasets:**
  - Camelyon17 (WILDS) — histopathology with explicit hospital distribution shift (https://wilds.stanford.edu/datasets/#camelyon17)
  - MIMIC-CXR — train/test splits across demographic strata for OOD evaluation (https://physionet.org/content/mimic-cxr/)
  - RSNA Pneumonia Detection — Kaggle competition dataset for OOD robustness benchmarks (verify URL)
  - MedMNIST — 18 standardised 2D/3D medical classification tasks for OOD benchmarking (https://medmnist.com/)
- **Starter repos/tools:**
  - WILDS (https://github.com/p-lambda/wilds) — distribution shift benchmark with CausalML support
  - PyTorch-Uncertainty (https://github.com/ENSTA-U2IS-AI/torch-uncertainty) — uncertainty methods on GPU
  - ConformalCI (https://github.com/aangelopoulos/conformal-prediction) — conformal prediction calibration
  - Laplace-Redux (https://github.com/AlexImmer/Laplace) — post-hoc Laplace approximation for pretrained NNs
- **CUDA libraries & GPU pattern:** cuDNN for parallel ensemble member forward passes, cuBLAS for Mahalanobis computation over feature covariance matrix; pattern: batch-parallel ensemble evaluation with stacked model weights.

---

### 7.16 Continual Learning & Concept Drift Adaptation 🔴 · Frontier/Theoretical

- **Deep dive:** Enables deployed medical AI models to continuously incorporate new clinical data (new patient cohorts, updated imaging protocols, population shifts) without catastrophic forgetting of previously learned tasks. Experience replay stores old data in a GPU-resident memory buffer; elastic weight consolidation (EWC) computes Fisher information diagonals — a batched gradient-squared operation on GPU. Gradient Episodic Memory (GEM) requires projecting gradients to the feasible cone defined by old-task gradients, a GPU-parallelised quadratic program. Healthcare settings impose strict constraints: models cannot forget rare disease patterns seen only in early training.
- **Key algorithms:** Elastic Weight Consolidation (EWC), Progressive Neural Networks, PackNet, Gradient Episodic Memory (GEM), Experience Replay (ER), Dark Experience Replay (DER++), Learning Without Forgetting (LwF), Online EWC.
- **Datasets:**
  - MIMIC-IV — temporal partitioning by year to simulate concept drift (https://physionet.org/content/mimiciv/)
  - CheXpert / MIMIC-CXR — multi-cohort splits for sequential task training (https://stanfordmlgroup.github.io/competitions/chexpert/)
  - MedMNIST — 18-task sequential benchmark (https://medmnist.com/)
  - Skin Lesion datasets (ISIC archive) — year-stratified splits for drift simulation (https://www.isic-archive.com/)
- **Starter repos/tools:**
  - Avalanche (https://github.com/ContinualAI/avalanche) — continual learning library with GPU support and medical imaging plugins
  - Mammoth (https://github.com/aimagelab/mammoth) — GPU continual learning framework with DER++, GEM, EWC
  - FACIL (https://github.com/mmasana/FACIL) — class-incremental learning on GPU for image classifiers
  - CLMNIST / MedicalCL (verify URL) — medical imaging continual learning benchmarks
- **CUDA libraries & GPU pattern:** cuBLAS for Fisher diagonal computation, CUDA replay buffer sampling with pinned memory; pattern: gradient projection via CUDA-parallelised QP over constraint matrices.

---

### 7.17 Model Distillation & Compression for Edge Deployment 🟡 · Active R&D

- **Deep dive:** Compresses large clinical AI models (ViT-L, GPT-style) into small student networks that fit on embedded devices by matching teacher logits, intermediate features, or attention maps. Knowledge distillation training is GPU-bound: both teacher and student run forward passes in every iteration, doubling compute vs. standard training. Structured pruning removes entire channels or attention heads; the resulting sparse model still benefits from GPU execution through efficient sparse tensor routines. INT8 quantisation-aware training (QAT) uses fake-quantisation operators that are CUDA-kernel-friendly and allow recovery of accuracy on medical benchmarks.
- **Key algorithms:** Soft label distillation (Hinton et al.), feature matching (FitNets), attention transfer, data-free distillation, structured/unstructured pruning, magnitude-based weight pruning, quantisation-aware training (QAT), GPTQ, AWQ, LoRA for student warm-start.
- **Datasets:**
  - ImageNet (pre-training teachers) + CheXpert (domain fine-tuning) — dual-dataset compression pipeline
  - MedMNIST — small-scale medical benchmark for student evaluation (https://medmnist.com/)
  - PTB-XL — ECG dataset for waveform model compression evaluation (https://physionet.org/content/ptb-xl/)
  - LIDC-IDRI — CT nodule dataset for compressed segmentation model evaluation (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- **Starter repos/tools:**
  - TensorRT (https://github.com/NVIDIA/TensorRT) — quantisation and layer fusion for medical inference
  - Intel Neural Compressor (https://github.com/intel/neural-compressor) — INT8 QAT and pruning on GPU/CPU
  - Pytorch-Distiller (verify URL) — knowledge distillation toolkit
  - Once-for-All (https://github.com/mit-han-lab/once-for-all) — NAS + distillation for efficient medical model families
- **CUDA libraries & GPU pattern:** cuDNN for both teacher/student simultaneous forward passes, TensorRT PTQ/QAT calibration; pattern: data-parallel joint teacher-student training with teacher frozen on GPU.

---

### 7.18 Retinal Fundus AI Screening 🟢 · Established

- **Deep dive:** Classifies diabetic retinopathy, glaucoma, and age-related macular degeneration from colour fundus photographs or OCT scans. High-resolution fundus images (typically 2048×2048) require significant GPU memory for batch processing; ResNet, EfficientNet, and Swin-Transformer backbones fine-tuned on annotated fundus datasets are the standard approach. GPU tensor cores accelerate the backbone convolutions in batch; simultaneous inference across both eyes and multiple pathologies (multi-task heads) doubles effective throughput. Real-world screening pipelines process millions of images annually, making GPU throughput a primary operational concern.
- **Key algorithms:** EfficientNet-B4/B5 (winner of EyePACS 2019), Swin Transformer, Grad-CAM for lesion localisation, multi-task classification (DR grade + glaucoma + AMD), self-supervised pretraining on unlabelled fundus images, uncertainty calibration for referral decisions.
- **Datasets:**
  - EyePACS — 88,000 labelled fundus images, 5-grade DR severity (Kaggle, verify URL)
  - APTOS 2019 — 3,662 fundus images, DR grading competition (Kaggle, verify URL)
  - DRIVE / STARE — retinal vessel segmentation datasets (verify URL)
  - UK Biobank Retinal Imaging — 68k retinal fundus images with linked health records (https://www.ukbiobank.ac.uk/)
- **Starter repos/tools:**
  - EfficientDet / EfficientNet (https://github.com/google/automl/tree/master/efficientnet) — strong fundus baselines
  - MONAI (https://github.com/Project-MONAI/MONAI) — fundus classification pipelines
  - RETFound (https://github.com/rmaphoh/RETFound_MAE) — MAE-pretrained retinal foundation model on 1.6M fundus images
  - DeepDR Plus (verify URL) — end-to-end diabetic retinopathy screening system
- **CUDA libraries & GPU pattern:** cuDNN for EfficientNet/Swin convolutions, TensorRT for clinic deployment; pattern: data-parallel fine-tuning on high-resolution fundus batches with gradient accumulation.

---

### 7.19 Polygenic Risk Score Computation at Scale 🟡 · Active R&D

- **Deep dive:** Computes polygenic risk scores (PRS) for millions of individuals by summing effect sizes from thousands to millions of GWAS-identified SNPs across the genome. The core operation is a large sparse matrix-vector multiply: individual genotype matrix (N_samples × M_SNPs, typically stored as INT2 or INT8 allele dosages) times a weight vector of SNP effect sizes. For UK Biobank scale (500k individuals × 6M SNPs), this is a 3 TB sparse matrix multiply best suited to GPU execution via cuSPARSE. SAIGE-GPU accelerates mixed-model GWAS (which underlies PRS weight estimation) using GPU-optimised linear algebra, enabling phenome-wide PRS across hundreds of traits simultaneously.
- **Key algorithms:** Clumping and Thresholding (C+T), LDpred2, PRS-CS, lassosum, SAIGE mixed-model GWAS on GPU, LD pruning, population stratification correction (PCA), multi-ancestry PRS meta-analysis.
- **Datasets:**
  - UK Biobank — 500k WGS individuals, 7000 phenotypes (https://www.ukbiobank.ac.uk/)
  - All of Us Research Program — >680k diverse participants, whole genome sequencing (https://allofus.nih.gov/)
  - FinnGen — 500k Finnish participants with national registry linkage (https://www.finngen.fi/en)
  - dbGaP GWAS Summary Stats — thousands of published GWAS across traits (https://www.ncbi.nlm.nih.gov/gap/)
- **Starter repos/tools:**
  - SAIGE-GPU (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12960912/) — GPU-accelerated mixed-model GWAS (verify GitHub URL)
  - PRSice-2 (https://github.com/choishingwan/PRSice) — PRS computation tool (CPU; GPU via matrix backend)
  - LDpred2 (https://github.com/privefl/bigsnpr) — Bayesian PRS in R with parallelism via bigstatsr
  - PLINK 2.0 (https://www.cog-genomics.org/plink/2.0/) — genome-wide association toolkit with GPU-accelerated linear algebra
- **CUDA libraries & GPU pattern:** cuSPARSE SpMV for genotype × effect-size matrix, cuBLAS for LD matrix operations, Thrust for sorting variant effect sizes; pattern: chunked sparse matrix multiply with INT2 genotype encoding to maximise VRAM utilisation.

---

---

## 8. Neuroscience & Brain-Computer Interfaces

### 8.1 Real-Time Neural Decoding for BCIs 🟡 · Active R&D
- **Deep dive:** Brain-computer interfaces decode motor intent, speech, or cognitive state from simultaneous recordings of 100–1 000+ neural channels at 30 kHz sampling. The decoding pipeline—bandpass filtering, spike detection, feature extraction, Kalman/population vector decode, output command generation—must complete within 5–50 ms to feel natural to the user. GPU acceleration allows running deep neural decoder networks (1D-CNN, transformer, WaveNet) directly in the decode loop without sacrificing latency through CUDA stream pipelining.
- **Key algorithms:** Kalman filter (linear decoder), population vector algorithm (PVA), Wiener filter, linear discriminant analysis (LDA), point-process filter, recurrent neural networks (GRU/LSTM), convolutional temporal decoder, optimal linear estimator (OLE), variational autoencoder latent space decoding.
- **Datasets:** BrainGate clinical trial data (https://www.braingate.org — access via collaboration); DANDI Archive intracortical array datasets (https://dandiarchive.org); Allen Brain Observatory Neuropixels data (https://portal.brain-map.org); NLB (Neural Latents Benchmark) — standardized BCI decode benchmarks (https://neurallatents.github.io).
- **Starter repos/tools:** BrainFlow (https://github.com/brainflow-dev/brainflow) — unified BCI SDK with real-time GPU-compatible data streaming; OpenBCI GUI (https://github.com/OpenBCI/OpenBCI_GUI) — open-source BCI hardware + software; NLB challenge tools (https://github.com/neurallatents/nlb_tools) — neural latents benchmark evaluation; NDT2/FALCON BCI decode benchmark (verify URL on neurallatents.github.io).
- **CUDA libraries & GPU pattern:** cuBLAS for real-time matrix multiply in Kalman predict/update; TensorRT for inference-optimized deep decoder; CUDA streams for pipelined acquire→decode→output with <5 ms latency; pattern: producer-consumer stream pipeline with pinned host memory for zero-copy data ingestion from acquisition hardware.

---

### 8.2 Spike Sorting 🟢 · Established
- **Deep dive:** Spike sorting identifies the firing times and cellular identities of individual neurons from raw extracellular voltage traces recorded on multi-electrode arrays (MEAs) or Neuropixels probes (384 channels × 30 kHz). The GPU bottleneck is template-matching: cross-correlating detected waveforms against hundreds of neuron templates across all channels simultaneously. Kilosort4 achieves this via GPU-accelerated template convolution, reducing hours of CPU sorting to minutes and enabling automated curation for large-scale Neuropixels datasets.
- **Key algorithms:** Whitening and common-average reference (CAR) preprocessing, threshold-based spike detection, PCA dimensionality reduction, template-matching (cross-correlation), expectation-maximization (EM) clustering, drift correction via continuous template registration, Gaussian mixture model (GMM) classification.
- **Datasets:** DANDI Archive Neuropixels datasets (https://dandiarchive.org); Allen Brain Observatory Neuropixels visual coding dataset (https://portal.brain-map.org); SpikeInterface benchmark datasets (https://spikeinterface.readthedocs.io); MountainSort benchmark datasets on Zenodo (search zenodo.org "spike sorting benchmark").
- **Starter repos/tools:** Kilosort4 (https://github.com/MouseLand/Kilosort) — GPU template-matching spike sorter, Python, CUDA; MountainSort5 (https://github.com/flatironinstitute/mountainsort5) — Flatiron Institute sorter with GPU preprocessing; SpikeInterface (https://github.com/SpikeInterface/spikeinterface) — unified Python framework wrapping 10+ sorters including GPU ones; Phy (https://github.com/cortex-lab/phy) — manual curation GUI for Kilosort output.
- **CUDA libraries & GPU pattern:** cuFFT for template convolution (FFT-based cross-correlation); cuBLAS for waveform-template matrix multiply; cuSPARSE for sparse cluster assignment; CUDA Thrust for peak-finding in filtered traces; pattern: sliding-window batch convolution with cuFFT, one FFT per channel-template pair in a batched call.

---

### 8.3 EEG/MEG Source Localization & Processing 🟢 · Established
- **Deep dive:** EEG/MEG source localization solves the ill-posed inverse problem of estimating the distribution of neural current sources inside the brain from measurements at 64–306 scalp/sensor locations. Forward model computation (leadfield matrix) via BEM/FEM over a realistic head model is a one-time GPU-amenable precomputation. Inverse methods range from beamforming (spatial filtering) to sparse Bayesian learning (Champagne, SESAME) with large-scale matrix factorizations that benefit from GPU. Real-time EEG filtering for BCI or epilepsy monitoring requires FIR/IIR at 1 000–10 000 Hz on 256 channels.
- **Key algorithms:** Boundary element method (BEM) for leadfield computation, minimum norm estimate (MNE), LORETA / eLORETA, beamforming (LCMV, DICS), sparse Bayesian learning, MUSIC dipole scan, dynamical statistical parametric mapping (dSPM), time-frequency analysis (Morlet wavelet, multitaper).
- **Datasets:** OpenNeuro EEG/MEG datasets in BIDS (https://openneuro.org); DANDI neurophysiology archive (https://dandiarchive.org); Human Connectome Project MEG (https://db.humanconnectome.org); TUAB / TUEG Temple University Hospital EEG corpus (https://isip.piconepress.com/projects/tuh_eeg/).
- **Starter repos/tools:** MNE-Python (https://github.com/mne-tools/mne-python) — comprehensive EEG/MEG analysis with GPU-accelerated backends; FieldTrip (https://github.com/fieldtrip/fieldtrip) — MATLAB MEG/EEG toolbox with parallel toolbox support; Brainstorm (https://github.com/brainstorm-users/brainstorm) — GUI EEG/MEG analysis; EEGLAB (https://github.com/sccn/eeglab) — MATLAB plugin ecosystem for EEG.
- **CUDA libraries & GPU pattern:** cuBLAS DGEMM for leadfield matrix multiply and beamformer weight computation; cuSOLVER for minimum-norm pseudoinverse; cuFFT for spectral analysis (all channels simultaneously); pattern: channel × time matrix operations on GPU, batch FFT across all channel pairs for coherence analysis.

---

### 8.4 Connectomics / EM Image Reconstruction 🟡 · Active R&D
- **Deep dive:** Volume electron microscopy (serial-section TEM, FIB-SEM) generates terabyte-to-petabyte image volumes (4 nm/voxel for nanometer-resolution synaptic ultrastructure). GPU-accelerated convolutional neural networks (3D U-Net, flood-filling networks) perform dense semantic segmentation of neurons, mitochondria, and synapses. Watershed-based instance segmentation and agglomeration follow, then automated synapse detection and connectivity graph extraction. The H01 human cortical connectome dataset is 1.4 PB; the FlyEM hemibrain is 26 TB.
- **Key algorithms:** 3D U-Net for voxel affinity prediction, flood-filling networks (recurrent CNN), watershed agglomeration (Kruskal/Prim on affinity graph), multicut graph partitioning, synapse detection (3D detection network), stitching and alignment (SIFT + RANSAC), mean shift/DBSCAN for spine detection.
- **Datasets:** Google H01 Human Cortex Connectome — 1.4 PB, 1 mm³ human cortex (https://h01-release.storage.googleapis.com/landing.html); FlyEM Janelia Hemibrain — Drosophila full connectome (https://neuprint.janelia.org); CREMI challenge — Drosophila larval neuromuscular junction EM (https://cremi.org); SNEMI3D — mouse cortex EM (https://snemi3d.grand-challenge.org/).
- **Starter repos/tools:** PyTorch Connectomics (https://github.com/zudi-lin/pytorch_connectomics) — modular GPU connectomics segmentation framework; DVID (https://github.com/janelia-flyem/dvid) — Janelia distributed EM data management; NeuTu (https://github.com/janelia-flyem/NeuTu) — proofreading and reconstruction visualization; VAST (verify URL — Harvard Lichtman lab large volume annotation tool).
- **CUDA libraries & GPU pattern:** cuDNN for 3D convolution in U-Net (dominant cost); NCCL for multi-GPU tensor-parallel training on large 3D crops; cuSPARSE for agglomeration graph operations; pattern: 3D sub-volume data parallelism across GPUs; sliding-window inference with overlap-tile strategy; mixed FP16/FP32 training.

---

### 8.5 Neural Mass / Whole-Brain Dynamics Models 🟡 · Active R&D
- **Deep dive:** Neural mass models (Wilson-Cowan, Jansen-Rit, Kuramoto oscillators, Stuart-Landau) approximate the mean firing rate of cortical regions, coupled by structural connectivity matrices from diffusion tractography. The Virtual Brain (TVB) simulates 84–360 cortical + subcortical regions, each with an ODE system of 2–8 state variables, coupled via a time-delayed connectivity matrix (50–100 ms conduction delays). GPU parallelism is exploited both for region-level ODE integration and for ensemble simulations fitting personalized connectomes.
- **Key algorithms:** Wilson-Cowan / Jansen-Rit neural mass ODEs, Kuramoto phase oscillator network, Stuart-Landau Hopf normal form, delay differential equations (DDE) with ring buffer, structural connectivity eigenspectrum analysis, Bayesian parameter inference for connectome fitting, graph-theoretic network analysis.
- **Datasets:** Human Connectome Project structural connectivity (https://db.humanconnectome.org); TVB compatible connectome datasets (https://www.thevirtualbrain.org/tvb/zwei/client-area); OpenNeuro fMRI for BOLD signal comparison (https://openneuro.org); ADNI structural MRI for patient-specific connectomes (https://adni.loni.usc.edu).
- **Starter repos/tools:** The Virtual Brain (https://github.com/the-virtual-brain/tvb-root) — whole-brain neural mass simulator with GPU via Numba/CUDA backends; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — multiscale NEURON network with structural connectivity import; Brian2 (https://github.com/brian-team/brian2) — network ODE with Brian2CUDA; MOOSE (https://github.com/BhallaLab/moose-core) — compartmental neural mass implementation.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for per-region ODE with ring-buffer delay lookup; cuBLAS for connectivity matrix-vector multiply (coupling term); CUDA Thrust for eigenvalue analysis; pattern: one CUDA thread per region, delay ring-buffer in shared memory, connectivity matrix in texture memory for caching.

---

### 8.6 Deep Brain Stimulation / Neurostimulation Modeling 🟡 · Active R&D
- **Deep dive:** Deep brain stimulation (DBS) for Parkinson's disease delivers high-frequency (~130 Hz) electrical pulses from implanted electrodes in the subthalamic nucleus (STN). Predicting stimulation volume and network effects requires solving the quasi-static Poisson equation in a patient-specific brain volume conductor (DT-MRI-derived anisotropic conductivity), coupled to cable equation models of axons in the stimulation field. GPU parallelizes both the FEM Poisson solve and the hundreds of independent axon cable simulations needed to map activation thresholds.
- **Key algorithms:** Quasi-static Poisson equation (anisotropic conductivity from DTI), FEM on tetrahedral brain mesh, cable equation for myelinated axons (McNeal model, MRG model), chronaxie-rheobase threshold estimation, volume of tissue activated (VTA) mapping, network oscillation modeling (basal ganglia-thalamo-cortical loop ODEs).
- **Datasets:** ADNI DT-MRI datasets (https://adni.loni.usc.edu); Human Connectome Project DT-MRI (https://db.humanconnectome.org); OpenNeuro DBS patient imaging (https://openneuro.org); OSS-DBS example cases (verify URL on github.com/OSS-DBSv2 or similar).
- **Starter repos/tools:** OSS-DBS v2 (https://github.com/SFB-ELAINE/OSS-DBS — verify URL) — open-source DBS simulation platform (FEM + axon models); SCIRun (https://github.com/SCIInstitute/SCIRun) — Utah electrodes + FEM neurostimulation; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — basal ganglia network models for DBS effect simulation; NEURON (https://github.com/neuronsimulator/nrn) — canonical axon cable equation solver.
- **CUDA libraries & GPU pattern:** cuSPARSE CG for anisotropic FEM Poisson solve; batch cable ODE across hundreds of axon trajectories (cuSOLVER batched tridiagonal or custom Thomas algorithm CUDA kernel); cuBLAS for DBS-induced voltage interpolation; pattern: parallel FEM solve then embarrassingly parallel axon threshold sweeps.

---

### 8.7 EEG Seizure Detection & Prediction 🟡 · Active R&D
- **Deep dive:** Epileptic seizure prediction from scalp EEG requires continuous multi-channel spectral feature extraction and classification over rolling windows with latencies <1 s. The preictal period (minutes to hours before seizure onset) exhibits subtle changes in high-frequency oscillations (HFOs), phase-amplitude coupling, and cross-channel coherence. GPU allows real-time feature extraction from 256 channels × 2 500 Hz using cuFFT spectrograms, simultaneous CNN/LSTM classification, and sliding-window cross-correlation for connectivity graphs.
- **Key algorithms:** Short-time Fourier transform (STFT), Morlet wavelet, phase-amplitude coupling (PAC), graph-theoretic seizure propagation, 1D-CNN and BiLSTM classifiers, attention transformer for long-range EEG context, support vector machine (SVM) on spectral features, SEEG source imaging.
- **Datasets:** Temple University Hospital EEG Corpus (TUAB/TUEG) — 30 000+ EEG recordings (https://isip.piconepress.com/projects/tuh_eeg/); CHB-MIT Scalp EEG Database (PhysioNet) (https://physionet.org/content/chbmit/1.0.0/); IEEG Portal — intracranial EEG for epilepsy (https://www.ieeg.org); OpenNeuro epilepsy datasets (https://openneuro.org).
- **Starter repos/tools:** MNE-Python (https://github.com/mne-tools/mne-python) — EEG processing with parallel backend; PyTorch EEG (https://github.com/torcheeg/torcheeg) — GPU deep learning for EEG; EEGLAB (https://github.com/sccn/eeglab) — MATLAB seizure analysis plugins; BrainFlow (https://github.com/brainflow-dev/brainflow) — real-time streaming for wearable seizure monitors.
- **CUDA libraries & GPU pattern:** cuFFT batched STFT across all channels simultaneously; cuDNN for CNN classifier inference; custom CUDA kernel for phase-amplitude coupling across channel pairs; pattern: rolling window with circular GPU buffer, cuFFT on each frame, classifier inference on extracted features via TensorRT.

---

### 8.8 Real-Time Tractography for Neurosurgical Navigation 🟡 · Active R&D
- **Deep dive:** Diffusion tensor imaging (DTI) tractography traces white matter fiber bundles from seed ROIs by integrating principal diffusion directions through the 3D DTI field (streamline tracking). Intraoperative real-time tractography updates the fiber map as brain shift occurs during surgery, requiring sub-second computation. GPU parallelizes thousands of independent streamline integrations (CUDA: one thread per seed). Probabilistic tractography (FSL BEDPOSTX) samples from diffusion parameter posteriors—thousands of Monte Carlo streamlines per seed—is also GPU-amenable.
- **Key algorithms:** Deterministic streamline tractography (FACT, Runge-Kutta 4th order), probabilistic tractography (FSL BEDPOSTX ball-and-stick model), fiber orientation distribution (FOD) from HARDI (spherical deconvolution), constrained spherical deconvolution (CSD), DSI/Q-ball imaging, anatomical tract atlas registration (MNI-space), curvature-limited streamline termination.
- **Datasets:** Human Connectome Project DT-MRI (https://db.humanconnectome.org); ADNI diffusion MRI (https://adni.loni.usc.edu); ISMRM 2015 Tractography Challenge dataset (verify URL — tractometer.org or ismrm.org); OpenNeuro diffusion MRI datasets (https://openneuro.org).
- **Starter repos/tools:** DIPY (https://github.com/dipy/dipy) — Python DTI/HARDI tractography with GPU acceleration via CuPy; FSL GPU tractography (GPU BEDPOSTX) (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT) — CUDA-accelerated probabilistic tractography; MRtrix3 (https://github.com/MRtrix3/mrtrix3) — constrained spherical deconvolution + tractography; TrackVis/DiffusionTool (verify URL) — surgical navigation-oriented fiber display.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for parallel streamline integration (one thread per seed, RK4 over DTI field in texture memory); cuBLAS for tensor field operations; cuFFT for spherical harmonic convolution in CSD; pattern: texture-memory DTI field for fast interpolation, warp-level thread divergence handled by fixed-step integration.

---

### 8.9 Calcium Imaging Analysis & Neural Population Dynamics 🟢 · Established
- **Deep dive:** Two-photon calcium imaging records fluorescence from GCaMP-expressing neurons (~1 000–100 000 cells per session at 30 Hz) but requires computationally intensive post-processing: rigid/non-rigid motion correction (GPU-accelerated phase-correlation), ROI detection (NMF or CNN-based source separation), neuropil subtraction, and deconvolution of calcium transients to infer spike timing. Suite2p's GPU pipeline reduces processing of a 60-minute session from hours to minutes. Simultaneous GPU inference of population state dynamics (LFADS, CEBRA, Pi-VAE) ties calcium activity to behavior.
- **Key algorithms:** Phase-correlation motion correction (cuFFT), constrained NMF (CNMF) for source separation, graph-based ROI detection (Suite2p), OASIS/FOOPSI spike deconvolution (LASSO), LFADS latent factor analysis via dynamical systems (LSTM encoder-decoder), t-SNE/UMAP for population visualization.
- **Datasets:** Allen Brain Observatory calcium imaging (https://portal.brain-map.org); DANDI calcium imaging datasets (https://dandiarchive.org); OpenNeuro two-photon datasets (https://openneuro.org); CaImAn demo datasets (https://github.com/flatironinstitute/CaImAn).
- **Starter repos/tools:** Suite2p (https://github.com/MouseLand/suite2p) — fast GPU calcium imaging pipeline (registration + detection + deconvolution); CaImAn (https://github.com/flatironinstitute/CaImAn) — Flatiron CNMF with GPU motion correction; CellProfiler (https://github.com/CellProfiler/CellProfiler) — general imaging analysis applicable to calcium phenotyping; LFADS (https://github.com/google-research/google-research/tree/master/lfads) — GPU latent factor analysis for population dynamics.
- **CUDA libraries & GPU pattern:** cuFFT for phase-correlation motion correction (all frames batch-FFT); cuDNN for CNN ROI detection; custom CUDA NMF solver (multiplicative update with shared-memory A^T A computation); pattern: frame-parallel GPU processing pipeline with pinned memory ring buffer for continuous acquisition.

---

### 8.10 Neural ODE / Dynamical Systems Models of Brain 🔴 · Frontier/Theoretical
- **Deep dive:** Neural ODEs parameterize the time derivative of hidden neural state as a neural network, enabling continuous-time models of brain dynamics that can be fit to irregular-interval neural recordings and extrapolated to unseen time points. Applied to whole-brain fMRI or calcium imaging, they learn latent dynamical manifolds underlying cognition. Adjoint sensitivity (checkpointed backpropagation through the ODE solver) is memory-intensive and GPU-critical; the adjoint method requires storing only a constant number of activations regardless of integration depth.
- **Key algorithms:** Neural ODE (Runge-Kutta adjoint), augmented neural ODE (ANODE), latent ODE / SDE (VAE + neural ODE), flow matching, score-based generative modeling for neural trajectories, continuous normalizing flow (CNF), Gaussian process ODE, reservoir computing (echo state networks).
- **Datasets:** Human Connectome Project resting-state fMRI (https://db.humanconnectome.org); DANDI electrophysiology (https://dandiarchive.org); Allen Brain Observatory calcium imaging (https://portal.brain-map.org); NLB Neural Latents Benchmark (https://neurallatents.github.io).
- **Starter repos/tools:** torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU neural ODE with adjoint backpropagation; torchsde (https://github.com/google-research/torchsde) — stochastic differential equation neural models on GPU; LFADS (https://github.com/google-research/google-research/tree/master/lfads) — RNN-based latent factor analysis; Diffrax (https://github.com/patrick-kidger/diffrax) — JAX-based GPU ODE/SDE solver suite.
- **CUDA libraries & GPU pattern:** cuDNN for neural network RHS evaluation; checkpointed adjoint via custom CUDA memory management; cuRAND for SDE noise sampling; pattern: time-reversed adjoint integration with activations recomputed on-the-fly, CUDA graph for repeated-pattern ODE step.

---

### 8.11 Retinal Circuit Modeling 🟡 · Active R&D
- **Deep dive:** The retina performs significant pre-processing of visual information through its 5-layer circuit (photoreceptors → bipolar cells → retinal ganglion cells, with amacrine/horizontal lateral inhibition). GPU simulation of a patch of retina (~10⁴ RGCs in a 1 mm² patch) with biophysical conductance-based models allows in-silico testing of prosthetic stimulation strategies (retinal implants) and understanding of center-surround RF structure, direction selectivity, and disease models (AMD, retinitis pigmentosa).
- **Key algorithms:** Phototransduction cascade ODE (Lamb-Pugh model), conductance-based synaptic kinetics (AMPA/GABA), integrate-and-fire RGC models, difference-of-Gaussian RF (center-surround), motion energy model, population-code visual stimulus reconstruction, NEST retina model, spiking response to prosthetic pulse trains.
- **Datasets:** DANDI retinal electrophysiology datasets (https://dandiarchive.org); Natural Scenes Dataset for RGC response (verify URL at naturalscenesdataset.org); Allen Brain Observatory visual stimuli + RGC responses (https://portal.brain-map.org); UK Biobank retinal OCT (https://www.ukbiobank.ac.uk).
- **Starter repos/tools:** COREM retina simulator (https://github.com/pablomc88/COREM) — C++ NEST-compatible GPU-amenable retinal simulator; Virtual Retina (https://github.com/jahuth/virtualretina) — Python retinal model; NEST simulator with retinal models (https://github.com/nest/nest-simulator) — RGC population simulation; GeNN (https://github.com/genn-team/genn) — GPU SNN for RGC population simulation.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for photoreceptor ODE lattice (spatially organized pixel-by-pixel); cuFFT for center-surround RF convolution in frequency domain; cuDNN for neural response estimation in data-driven retinal models; pattern: 2D CUDA grid over photoreceptor array, warps handle horizontal cell lateral diffusion via shared-memory stencil.

---

### 8.12 Cochlear Mechanics & Auditory Processing 🟡 · Active R&D
- **Deep dive:** The cochlea performs mechanical frequency decomposition via basilar membrane (BM) traveling waves, transforming sound to a tonotopic neural code via inner hair cells (IHCs) and auditory nerve fibers (ANFs). GPU simulation of a 3D BM model (finite element) or active cochlear model (outer hair cell electromotility — prestin) with coupled fluid mechanics and IHC/ANF spike generation supports hearing prosthesis design, audiogram prediction, and noise-induced hearing loss modeling.
- **Key algorithms:** 1D/2D/3D basilar membrane wave equation (FEM/FD), fluid-structure interaction for perilymph-BM coupling, outer hair cell electromotility (Prestin ODE), inner hair cell transducer (MET channel), auditory nerve fiber spike model (Zilany-Bruce), gammatone filterbank (frequency-domain equivalent), cochlear implant electrode models.
- **Datasets:** NH Hearing database (verify URL at nhlibrary.org); Auditory Model Toolbox benchmark datasets (https://amtoolbox.org); PhysioNet auditory brainstem response datasets (https://physionet.org); cochlear implant stimulation datasets from Cochlear Ltd (proprietary; verify institutional access).
- **Starter repos/tools:** CoNNear cochlea (https://github.com/HearingTechnology/CoNNear_cochlea) — PyTorch DNN cochlear mechanics model for real-time inference; mrkrd/cochlea (https://github.com/mrkrd/cochlea) — Python inner ear models interfacing NEURON/Brian; Auditory Model Toolbox (https://amtoolbox.org) — MATLAB/Octave/Python cochlear models; NEST simulator (https://github.com/nest/nest-simulator) — ANF population spiking.
- **CUDA libraries & GPU pattern:** cuFFT for gammatone filterbank (bank of FIR filters via FFT convolution); custom CUDA FEM kernel for 1D BM wave equation (Thomas tridiagonal along BM length); batched ODE for ANF spike generation (one thread per fiber); pattern: frequency-band-parallel GPU computation, each warp handles one characteristic frequency band.

---

### 8.13 Vestibular System & Sensorimotor Integration 🔴 · Frontier/Theoretical
- **Deep dive:** The vestibular system detects head motion via semicircular canals (angular velocity → cupula deflection → hair cell activation) and otolith organs (linear acceleration). GPU simulation of the full cupula-endolymph fluid-structure interaction (FSI) in all three canals plus otolith membrane mechanics, coupled to downstream neural coding (irregular vs. regular afferents) and central vestibulo-ocular reflex (VOR) circuitry, is computationally demanding but tractable with GPU. Applications include space medicine, motion sickness modeling, and vestibular implant design.
- **Key algorithms:** Cupula-endolymph FSI (Stokes flow + elastic membrane), hair bundle adaptation ODE, afferent spike coding (van Hemmen model), torsion pendulum model, Kalman-filter Bayesian internal model, VOR motor command ODE, cerebellar Purkinje cell learning (Marr-Albus-Ito).
- **Datasets:** Vestibular electrophysiology data from DANDI (https://dandiarchive.org); Human Connectome Project functional connectivity (vestibular cortex) (https://db.humanconnectome.org); PhysioNet balance/posturography datasets (https://physionet.org); published cupula FSI experimental datasets (verify via institutional access).
- **Starter repos/tools:** NEST simulator (https://github.com/nest/nest-simulator) — vestibular afferent and VOR circuit models; GeNN (https://github.com/genn-team/genn) — GPU SNN for VOR + cerebellar learning; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — semicircular canal endolymph FSI; FEBio (https://github.com/febiosoftware/FEBio) — otolith membrane FEM.
- **CUDA libraries & GPU pattern:** Custom CUDA Stokes flow solver for endolymph; batch ODE for hair bundle + afferent dynamics (one thread per hair cell); cuBLAS for cerebellar parallel fiber weight matrix updates; pattern: fluid-structure coupling via immersed boundary method on GPU with split-step FSI.

---

### 8.14 Whole-Brain Simulation at Cellular Resolution 🔴 · Frontier/Theoretical
- **Deep dive:** Simulating the entire mouse brain (~70 million neurons, ~1 trillion synapses) or human brain (~86 billion neurons) at point-neuron resolution requires exascale computing. Current GPU-capable implementations target mouse brain at simplified LIF models and are a grand-challenge benchmark for neuromorphic hardware. Even 1% of the human brain (~860 million neurons) needs ~10 GB of synaptic state alone. GPU cluster approaches (NEST GPU across many nodes, or NVIDIA H100 NVLink cluster) target this regime; the key bottleneck is sparse synaptic event communication.
- **Key algorithms:** Leaky integrate-and-fire / Izhikevich / AdEx at scale, distributed spike event routing (MPI + NCCL), synaptic delay management (distributed ring buffers), STDP online learning at scale, heterogeneous connectivity (random, small-world, structural), balanced E/I network dynamics (Brunel network).
- **Datasets:** Allen Mouse Brain Connectivity Atlas (https://portal.brain-map.org); HCP structural connectivity (https://db.humanconnectome.org); FlyEM Janelia Drosophila connectome for validation (https://neuprint.janelia.org); Blue Brain Cell Atlas (https://portal.brain-map.org).
- **Starter repos/tools:** NEST GPU (https://github.com/nest/nest-simulator) — multi-GPU NEST with CUDA kernel for large network simulation; GeNN (https://github.com/genn-team/genn) — GPU SNN code generation targeting large networks; The Virtual Brain (https://github.com/the-virtual-brain/tvb-root) — whole-brain mean-field at lower resolution; SpikingJelly (https://github.com/fangwei123456/spikingjelly) — PyTorch SNN framework scalable to large populations.
- **CUDA libraries & GPU pattern:** NCCL for multi-GPU spike event all-to-all communication; custom CUDA kernels for per-neuron state update with register-resident state; cuSPARSE for connectivity matrix-vector product; pattern: GPU-direct MPI for spike routing, neuron state in global memory with warp-coalesced access, NVLink for intra-node GPU communication.

---

### 8.15 Optogenetics Stimulation Modeling 🟡 · Active R&D
- **Deep dive:** Optogenetics uses light-gated ion channels (channelrhodopsin-2, halorhodopsin) to activate or silence neurons with light. GPU simulation of an optogenetic stimulation experiment requires: (1) Monte Carlo photon transport in scattering brain tissue (thousands of photons per simulation, one independent random walk per photon → embarrassingly GPU-parallel); (2) ChR2 4-state photocycle ODE at each illuminated neuron; (3) network-level spiking response. Predicting light spread and activation volumes guides implant design.
- **Key algorithms:** Monte Carlo photon transport (random walk with scattering/absorption, Henyey-Greenstein phase function), ChR2 4-state kinetic model (Hegemann), 3-state simplified model, Beer-Lambert for superficial tissue, network SNN with light-activated conductance, activation map computation.
- **Datasets:** Allen Brain Atlas gene expression for ChR2 targeting (https://portal.brain-map.org); DANDI optogenetics experimental datasets (https://dandiarchive.org); openMC photon transport validation cases (verify at openmc.org); OpenNeuro optogenetics fMRI datasets (https://openneuro.org).
- **Starter repos/tools:** MCX (Monte Carlo eXtreme) (https://github.com/fangq/mcx) — GPU-accelerated photon transport in biological tissue (CUDA, 1000× CPU speedup); GeNN (https://github.com/genn-team/genn) — SNN with ChR2 conductance kinetics; NEST simulator (https://github.com/nest/nest-simulator) — optogenetics module; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — network simulation with optogenetic inputs.
- **CUDA libraries & GPU pattern:** Custom CUDA Monte Carlo kernel (one thread per photon packet, cuRAND for scattering events, atomic-add for fluence accumulation in voxel grid); cuSPARSE for network spike propagation; pattern: seed-parallel photon launch with cuRAND Sobol sequences, shared-memory partial fluence accumulation per thread-block.

---

### 8.16 Neural Signal Compression & Wireless BCI Transmission 🟡 · Active R&D
- **Deep dive:** Fully implanted high-channel-count BCIs (1 024–65 000 electrodes in emerging platforms) cannot transmit raw 30 kHz × N-channel data wirelessly due to power/bandwidth limits. GPU-accelerated on-device compression (threshold crossing, wavelet compression, PCA projection, spike detection) must reduce data 100–1 000× before wireless transmission. Implantable ASICs perform this in hardware, but GPU simulation of compression algorithms enables algorithm design and fidelity evaluation before silicon tape-out.
- **Key algorithms:** Threshold-based spike detection, wavelet packet decomposition (WPD), compressed sensing (L1 minimization / OMP), PCA projection for dimensionality reduction, delta-encoding, Huffman/arithmetic coding, matched filter spike detection, signal reconstruction via iterative thresholding (ISTA/FISTA).
- **Datasets:** DANDI Neuropixels recordings (https://dandiarchive.org); BrainGate implanted array datasets (https://www.braingate.org); SpikeInterface benchmark recordings (https://spikeinterface.readthedocs.io); PhysioNet neural datasets (https://physionet.org).
- **Starter repos/tools:** BrainFlow (https://github.com/brainflow-dev/brainflow) — real-time neural signal SDK; SpikeInterface (https://github.com/SpikeInterface/spikeinterface) — spike detection and feature extraction pipeline; PyWavelets (https://github.com/PyWavelets/pywt) — wavelet decomposition with CuPy GPU backend; FISTA implementations in PyTorch (verify URL — numerous public repos).
- **CUDA libraries & GPU pattern:** cuFFT for wavelet and frequency-domain feature extraction; cuBLAS for PCA projection (matrix-vector multiply); CUDA Thrust for threshold scan across all channels; pattern: streaming pipeline—raw samples in via DMA, CUDA kernels for detection and projection, compressed output via pinned host memory ring buffer.

---

*All GitHub URLs have been verified against search results as of June 2026. URLs marked (verify URL) could not be confirmed from available search results and should be independently checked. Key caveats: the Blue Brain Project GitHub org (https://github.com/BlueBrain) remains accessible but active development has migrated to https://github.com/openbraininstitute following the project's conclusion in December 2024. ModelDB is migrating from https://senselab.med.yale.edu/ModelDB to https://modeldb.science. NVIDIA Thrust is archived in favour of the unified CCCL repo at https://github.com/NVIDIA/cccl.*


## Sections 7, 9, and 13 — Exhaustive Deep-Dive Reference

---

---

## 9. Epidemiology & Public Health

### 9.1 Agent-Based Epidemic Simulation 🟡 · Active R&D

- **Deep dive:** Simulates individual-level epidemic spread across millions of synthetic agents, each with behavioural rules governing contact, infection, and recovery. GPU parallelism maps each agent to a thread or thread group: state updates (susceptible → exposed → infectious → recovered) are embarrassingly parallel across the population. The bottleneck is computing pairwise contacts within spatial proximity grids or synthetic social networks; cuGraph adjacency traversal accelerates this. Non-Markovian (renewal) dynamics require tracking each agent's infectious age distribution, a memory-intensive operation that fits within GPU SRAM when using compressed state representations. FlashSpread achieves end-to-end GPU execution with kernel-fused dense stepping.
- **Key algorithms:** SIR/SEIR/SEIRD state machines per agent, contact kernel simulation (household, workplace, school stratification), non-Markovian renewal spreading, GPU-parallel BFS over contact graphs, Monte Carlo ensemble averaging, importance sampling for rare events, spatial hashing for local contact discovery.
- **Datasets:**
  - GLEAM / GLEaMviz global mobility + population data (https://www.gleamviz.org/)
  - US Census TIGER/Line shapefiles + ACS commuting data (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
  - Mossong et al. POLYMOD contact matrices — age-structured contact rates across 8 European countries (verify URL)
  - SafeGraph / Dewey mobility data — retail foot traffic and mobility patterns (verify URL)
- **Starter repos/tools:**
  - FRED (Framework for Reconstructing Epidemic Dynamics) (https://github.com/PublicHealthDynamicsLab/FRED) — individual-level US epidemic simulator
  - FlashSpread (https://arxiv.org/abs/2604.22092) — end-to-end GPU framework for non-Markovian network spreading (verify GitHub URL)
  - MEmilio (https://github.com/SciCompMod/memilio) — high-performance modular epidemic simulation software with GPU support
  - Epiabm (https://github.com/RESIDE-ICL/epiabm) — GPU-parallelised ABM framework for epidemic simulation
- **CUDA libraries & GPU pattern:** cuGraph for contact network BFS/DFS, cuRAND for stochastic transition sampling, custom CUDA kernels for per-agent state update; pattern: one CUDA thread per agent with shared-memory contact lookup tables, warp-level primitives for neighbour enumeration.

---

### 9.2 Large-Scale Compartmental & Metapopulation Models 🟡 · Active R&D

- **Deep dive:** Solves large systems of ODEs or stochastic differential equations (SDEs) describing disease dynamics across thousands of geographic patches interconnected by mobility flows (SIR at metapopulation scale, seasonal forcing, age structure). ODE integration over thousands of patches with coupling matrices is equivalent to a batched sparse matrix-vector multiply at each time step — a cuSPARSE-accelerated operation. Monte Carlo uncertainty quantification requires thousands of independent ODE solves in parallel on GPU, each with different parameter samples. GPU-based adaptive stepsize RK4/5 solvers (Torchdiffeq's `dopri5` on GPU) handle stiff biological dynamics efficiently.
- **Key algorithms:** Runge-Kutta 4/5 ODE integration on GPU, tau-leaping for stochastic compartmental models, MCMC parameter inference (ensemble MCMC), Approximate Bayesian Computation (ABC), metapopulation coupling via mobility matrix, seasonal forcing with Fourier series, age-structured SEIR with contact matrices.
- **Datasets:**
  - GLEAM — global airline + commuting network for metapopulation coupling (https://www.gleamviz.org/)
  - WHO Weekly Epidemiological Reports — case counts for parameter calibration (https://www.who.int/emergencies/situations)
  - CDC FluView — US influenza surveillance by week and region (https://www.cdc.gov/flu/weekly/)
  - COVID-19 Data Repository by CSSE at Johns Hopkins (archived) — global case/death time series (https://github.com/CSSEGISandData/COVID-19)
- **Starter repos/tools:**
  - Epiflows / EpiModel (https://github.com/EpiModel/EpiModel) — network-based compartmental modelling in R
  - Torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU-accelerated neural ODE and standard ODE solvers
  - MEmilio (https://github.com/SciCompMod/memilio) — high-performance C++/CUDA epidemic simulation
  - PyGOM (https://github.com/ukhsa-collaboration/pygom) — Python compartmental ODE modelling framework
- **CUDA libraries & GPU pattern:** cuSPARSE for mobility matrix coupling, cuRAND for stochastic tau-leaping, custom RK4 CUDA kernel for parallel ODE batch; pattern: each CUDA thread block integrates one metapopulation patch ODE system, with shared memory holding coupling matrices.

---

### 9.3 Contact-Network & Graph Epidemic Dynamics 🟡 · Active R&D

- **Deep dive:** Simulates epidemic spread on empirical or synthetic contact networks where nodes are individuals and weighted edges encode contact intensity. GPU graph traversal (BFS/DFS) across networks with millions of nodes enables exploration of counterfactual intervention scenarios (edge removal, node vaccination) in seconds vs. hours on CPU. The Replay tool transforms empirical timestamped contact data into duration-weighted adjacency matrices and uses GPU sparse matrix operations for realistic epidemic simulation. cuGraph's PageRank and community detection accelerate identification of superspreader hubs for targeted interventions.
- **Key algorithms:** SIR/SEIR stochastic simulation on contact graphs, Gillespie algorithm for continuous-time Markov chains, non-Markovian renewal kernels (FlashSpread), Belief Propagation for marginal inference on sparse graphs, community detection (Louvain, Leiden), targeted vaccination on high-degree nodes, R0 spectral radius estimation.
- **Datasets:**
  - SocioPatterns proximity contact data — face-to-face contacts in hospitals, schools, conferences (http://www.sociopatterns.org/)
  - Copenhagen Networks Study — Bluetooth proximity + mobile data for 800 students (verify URL)
  - GLEAM global mobility network (https://www.gleamviz.org/)
  - NiemaGraphGen synthetic contact networks (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10038133/) — memory-efficient global-scale simulation toolkit
- **Starter repos/tools:**
  - FlashSpread (https://arxiv.org/abs/2604.22092) — GPU framework for network epidemic dynamics (verify GitHub URL)
  - Replay (https://link.springer.com/article/10.1186/s12911-025-03310-2) — GPU-accelerated temporal contact network epidemiology tool
  - cuGraph (https://github.com/rapidsai/cugraph) — GPU graph analytics (PageRank, BFS, community detection) via RAPIDS
  - EoN (Epidemics on Networks) (https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks) — Python network epidemic simulation
- **CUDA libraries & GPU pattern:** cuGraph BFS/SSSP for infection spread on GPU-resident adjacency, cuSPARSE SpMV for transition probability matrices, cuRAND for stochastic edge activation; pattern: BFS-based wavefront parallelism with atomic state update per node.

---

### 9.4 Phylodynamics & Pathogen Genomic Surveillance 🟡 · Active R&D

- **Deep dive:** Infers the evolutionary and epidemiological history of pathogens from genomic sequences using Bayesian phylodynamic models (BEAST2, TreeTime). The computational bottleneck is evaluating the phylogenetic likelihood across millions of trees sampled by MCMC — each likelihood evaluation requires computing evolutionary substitution probabilities across thousands of sequence sites and tree branches. BEAGLE (Broad-platform Evolutionary Analysis General Likelihood Evaluator) provides a GPU-accelerated library for this core computation, delivering 20–50× speedup over CPU BEAST. GPU-accelerated variant calling pipelines (DNAnexus, NVIDIA Parabricks) feed surveillance outputs into phylodynamic pipelines.
- **Key algorithms:** Bayesian phylogenetic MCMC (Metropolis-Hastings), HKY/GTR nucleotide substitution models, Kingman's coalescent, birth-death diversification models, skyline population size estimation, ancestral state reconstruction, phylogeographic diffusion, TreeTime maximum-likelihood dating.
- **Datasets:**
  - GISAID — 15M+ SARS-CoV-2 and influenza sequences with metadata (https://www.gisaid.org/)
  - NCBI Pathogen Detection Database — real-time foodborne pathogen genomics (https://www.ncbi.nlm.nih.gov/pathogens/)
  - GenBank — nucleotide sequence archive for all pathogens (https://www.ncbi.nlm.nih.gov/genbank/)
  - Nextstrain data pipelines — curated SARS-CoV-2, influenza, mpox builds (https://nextstrain.org/)
- **Starter repos/tools:**
  - BEAST 2 (https://www.beast2.org/) — Bayesian phylogenetic inference; GPU via BEAGLE library
  - BEAGLE (https://github.com/beagle-dev/beagle-lib) — GPU-accelerated phylogenetic likelihood library (CUDA/OpenCL)
  - Nextstrain (https://github.com/nextstrain/augur) — real-time pathogen genomic surveillance pipeline
  - NVIDIA Parabricks (https://github.com/clara-parabricks) — GPU-accelerated variant calling and genome analysis
- **CUDA libraries & GPU pattern:** BEAGLE CUDA kernels for transition probability matrix exponentiation across tree branches, cuBLAS for substitution rate matrix multiplies; pattern: embarrassingly parallel site-likelihood computation across sequence columns, aggregated with parallel prefix products across tree branches.

---

### 9.5 Spatial Disease Mapping & Forecasting 🟡 · Active R&D

- **Deep dive:** Estimates disease incidence surfaces and spatiotemporal risk across geographic grids using Bayesian geostatistical models (BYM, INLA, Gaussian Process regression). The Gaussian process kernel matrix computation scales as O(N²) in the number of spatial locations — for a 10k-pixel grid this is a 10⁸-element covariance matrix, whose Cholesky decomposition is dominated by GPU-accelerated dense linear algebra (cuBLAS). GPU-based MCMC samplers (BlackJAX on CUDA, Greta with GPU backend) achieve 380× speedup for epidemic forecasting models. Interpolating national case-counts to sub-district resolution using kriging is entirely parallelisable across prediction locations on GPU.
- **Key algorithms:** Besag-York-Mollié (BYM) spatial smoothing, Integrated Nested Laplace Approximation (INLA), Gaussian Process regression, kriging interpolation, spatiotemporal Kalman filtering, Bayesian hierarchical Poisson regression, neural ODE spatial models, ensemble Kalman filters.
- **Datasets:**
  - WHO Mortality Database — ICD-coded deaths by country and cause (https://www.who.int/data/data-collection-tools/who-mortality-database)
  - IHME Global Burden of Disease — country-level disease incidence estimates (https://www.healthdata.org/gbd)
  - CDC Wonder — US county-level disease surveillance data (https://wonder.cdc.gov/)
  - NASA SEDAC Global Population Data — gridded population for exposure modelling (https://sedac.ciesin.columbia.edu/)
- **Starter repos/tools:**
  - INLA / R-INLA (https://www.r-inla.org/) — fast Bayesian spatial modelling; GPU via PARDISO sparse solver
  - BlackJAX (https://github.com/blackjax-devs/blackjax) — GPU-accelerated Bayesian sampling (HMC, NUTS) via JAX
  - Greta (https://github.com/greta-dev/greta) — probabilistic programming with TensorFlow GPU backend for spatial models
  - CARBayes (https://github.com/duncanplee/CARBayes) — R package for spatial Bayesian modelling (CPU; parallelisable)
- **CUDA libraries & GPU pattern:** cuBLAS for GP covariance matrix Cholesky, cuSPARSE for ICAR precision matrix operations, JAX XLA for GPU-accelerated MCMC; pattern: batch kriging over prediction grid points with fully GPU-resident covariance kernel.

---

### 9.6 Branching-Process Outbreak Detection & Estimation 🔴 · Frontier/Theoretical

- **Deep dive:** Models early epidemic growth as a Galton-Watson branching process or Hawkes point process to estimate the effective reproduction number Rt in near-real-time from case count time series. GPU parallelism enables simultaneous estimation of Rt across thousands of geographic units (counties, countries) simultaneously using batched Bayesian updates. The Hawkes process likelihood requires summing exponential kernels over all past events — a GPU-parallelised prefix sum operation. Branching process simulation (for outbreak probability calculations) is embarrassingly parallel: simulate 10⁵ independent outbreak realisations simultaneously on GPU to estimate extinction probabilities.
- **Key algorithms:** Galton-Watson branching process simulation, Hawkes self-exciting point process MLE, EpiEstim sliding-window Rt estimation, renewal equation Rt inference (Cori method), sequential Monte Carlo (particle filters) for real-time estimation, negative-binomial offspring distribution fitting, overdispersion estimation.
- **Datasets:**
  - CDC FluView — weekly US influenza-like illness surveillance (https://www.cdc.gov/flu/weekly/)
  - WHO Disease Outbreak News — global outbreak event data (https://www.who.int/emergencies/disease-outbreak-news)
  - COVID-19 Data Repository (CSSE Johns Hopkins) — archived case/death time series (https://github.com/CSSEGISandData/COVID-19)
  - ECDC Surveillance Atlas — European communicable disease surveillance (https://atlas.ecdc.europa.eu/)
- **Starter repos/tools:**
  - EpiEstim (https://github.com/mrc-ide/EpiEstim) — R package for Rt estimation (CPU; GPU via batched extension)
  - EpiNow2 (https://github.com/epiforecasts/EpiNow2) — Bayesian nowcasting and Rt estimation with Stan GPU backend
  - tick (https://github.com/X-DataInitiative/tick) — GPU-accelerated Hawkes process learning library
  - PyEpidemics (verify URL) — Python branching process simulation framework
- **CUDA libraries & GPU pattern:** cuRAND for Monte Carlo branching process simulation, Thrust parallel prefix sum for Hawkes likelihood, JAX/BlackJAX for GPU-based posterior inference; pattern: embarrassingly parallel ensemble simulation — each CUDA thread simulates one outbreak trajectory.

---

### 9.7 Vaccine Allocation & Intervention Optimisation 🟡 · Active R&D

- **Deep dive:** Determines optimal allocation of limited vaccines, treatments, or non-pharmaceutical interventions across age groups, geographic regions, or risk strata to minimise deaths or infections under resource constraints. GPU-accelerated simulation (agent-based or compartmental) enables rapid evaluation of thousands of candidate allocation policies within an optimisation loop. Reinforcement learning approaches (PPO, SAC) train on GPU-simulated environments where the epidemic simulator is the transition function. Multi-objective Pareto optimisation across equity and efficiency criteria requires GPU-parallelised NSGA-II or similar evolutionary algorithms.
- **Key algorithms:** Multi-objective optimisation (NSGA-II, NSGA-III), Proximal Policy Optimisation (PPO), Deep Q-Networks on simulation environments, Thompson sampling for adaptive allocation, network-based vaccinating-hub strategies (targeted vs. random), stochastic programming under epidemiological uncertainty, integer linear programming for logistics.
- **Datasets:**
  - GLEAM global mobility network for spatial allocation (https://www.gleamviz.org/)
  - WHO Immunisation Data — vaccination coverage by country and vaccine (https://immunizationdata.who.int/)
  - US Census commuting flows — for workplace transmission modelling (https://www.census.gov/)
  - COVID-19 vaccination time series (Our World in Data) — historical rollout data for calibration (https://ourworldindata.org/covid-vaccinations)
- **Starter repos/tools:**
  - Covasim (https://github.com/InstituteforDiseaseModeling/covasim) — GPU-friendly Python COVID-19 agent-based model
  - EMOD (https://github.com/InstituteforDiseaseModeling/EMOD) — high-performance individual-based disease model
  - Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3) — GPU RL library for policy training on epidemic environments
  - Pymoo (https://github.com/anyoptimization/pymoo) — multi-objective optimisation with GPU evaluation support
- **CUDA libraries & GPU pattern:** cuRAND for stochastic epidemic simulation, custom CUDA ODE kernels for compartmental model evaluation, CUDA graph for repeated fixed-topology GPU execution; pattern: population of candidate policies evaluated simultaneously across GPU thread blocks.

---

### 9.8 Wastewater-Based Epidemiology & Signal Detection 🟡 · Active R&D

- **Deep dive:** Infers community-level pathogen prevalence from viral RNA concentrations in wastewater, combining RT-qPCR signal time series with meteorological, demographic, and mobility covariates to nowcast and forecast disease incidence. GPU-accelerated deep learning (LSTM, Temporal Fusion Transformers) processes multivariate time series from thousands of sampling sites simultaneously; the data dimensionality is high (dozens of wastewater markers × weather variables × mobility indices per site). Bayesian hierarchical models fitted on GPU (via Stan with GPU backend or JAX) account for spatial correlation across sewage catchments. Deconvolution of wastewater signal to estimate case counts involves non-negative least-squares problems solved in parallel across sites.
- **Key algorithms:** Non-negative least-squares deconvolution, LSTM/GRU time series prediction, Temporal Fusion Transformers (TFT), Bayesian hierarchical regression, anomaly detection (isolation forests, CUSUM control charts), Poisson regression for count outcomes, spatial kriging for site interpolation.
- **Datasets:**
  - NWSS (National Wastewater Surveillance System) — US wastewater SARS-CoV-2 and flu data (https://www.cdc.gov/nwss/)
  - EU Sewage Sentinel System for SARS-CoV-2 (verify URL) — European wastewater surveillance
  - WastewaterSCAN — Stanford-led multi-pathogen wastewater monitoring (https://www.wastewaterscan.org/)
  - OpenWastewaterData (verify URL) — aggregated global wastewater surveillance
- **Starter repos/tools:**
  - PyTorch-Forecasting (https://github.com/jdb78/pytorch-forecasting) — TFT and LSTM for multivariate time series on GPU
  - Pyro (https://github.com/pyro-ppl/pyro) — GPU probabilistic programming for Bayesian wastewater signal deconvolution
  - Darts (https://github.com/unit8co/darts) — time series forecasting library with GPU support
  - NWSS Data Dashboard tools (https://www.cdc.gov/nwss/wastewater-surveillance-data-reporting.html) — CDC reference implementation
- **CUDA libraries & GPU pattern:** cuDNN for temporal model training, Pyro ELBO optimisation on GPU, cuBLAS for deconvolution least-squares; pattern: data-parallel forecasting across thousands of wastewater sites on GPU.

---

### 9.9 Hospital Capacity & Surge Demand Forecasting 🟡 · Active R&D

- **Deep dive:** Predicts short-term hospital admission volumes, ICU occupancy, and ventilator demand to enable proactive resource allocation during epidemic surges or seasonal peaks. GPU-accelerated LSTM, Transformer, and ensemble models trained on EHR admission records, regional case counts, wastewater signals, and mobility data produce rolling 14-day forecasts. The volume of hospital time series (thousands of hospitals × dozens of admission types × 365 days/year) is processed in parallel on GPU; each hospital's time series is a separate batch element. Real-time retraining on streaming data requires frequent mini-batch SGD on GPU to adapt to evolving epidemic waves.
- **Key algorithms:** LSTM/GRU multi-step forecasting, Temporal Fusion Transformers, N-BEATS, Prophet (Bayesian decomposition), Gaussian process regression for uncertainty, hierarchical reconciliation (MinT), ensemble averaging, ARIMA + neural hybrids, conformal prediction intervals.
- **Datasets:**
  - HHS Protect Hospital Capacity Data — US hospital capacity and admissions (https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/6xf2-c3ie)
  - ECDC Hospital Data — European hospital admissions and ICU occupancy (https://www.ecdc.europa.eu/en/covid-19/data)
  - NHS England Situation Reports — UK hospital admissions and bed occupancy (https://www.england.nhs.uk/statistics/)
  - COVID-19 Forecast Hub submissions — ensemble of >50 models (https://covid19forecasthub.org/)
- **Starter repos/tools:**
  - PyTorch-Forecasting (https://github.com/jdb78/pytorch-forecasting) — TFT, LSTM, N-BEATS on GPU
  - Darts (https://github.com/unit8co/darts) — multi-model time series forecasting with GPU backend
  - COVID-19 Forecast Hub (https://github.com/reichlab/covid19-forecast-hub) — ensemble model aggregation infrastructure
  - GluonTS (https://github.com/awslabs/gluonts) — probabilistic time series on GPU via MXNet/PyTorch
- **CUDA libraries & GPU pattern:** cuDNN for temporal model training, JAX XLA for parallelised Gaussian process forecasting, NCCL for multi-GPU ensemble training; pattern: panel data parallel — each hospital's time series as a batch element.

---

### 9.10 Mobility-Based Epidemic Nowcasting 🟡 · Active R&D

- **Deep dive:** Infers current epidemic state and short-term trajectory from human mobility data (mobile phone GPS, retail foot traffic, transit ridership) using data assimilation methods that combine mobility signals with epidemiological models. GPU enables rapid sequential Monte Carlo (particle filter) updates as new mobility observations arrive hourly, running thousands of particles simultaneously. Graph neural networks learn spatial transmission patterns from mobility flow matrices — a GPU-parallelised sparse graph convolution. The bottleneck is the batched epidemic ODE integration for all particles in the ensemble simultaneously.
- **Key algorithms:** Sequential Monte Carlo (particle filtering), ensemble Kalman filter (EnKF), graph convolutional networks on mobility graphs, LSTM encoder-decoder for mobility sequence learning, MAP estimation for transmission rate, community mobility indices as predictors (Google CMR).
- **Datasets:**
  - Google Community Mobility Reports — country/region mobility indices during COVID-19 (https://www.google.com/covid19/mobility/)
  - SafeGraph/Dewey POI visit data — US retail foot traffic (verify access terms)
  - Apple Mobility Trends — routing request data by transit type (verify URL)
  - Citymapper Mobility Index — urban mobility across 40 cities (verify URL)
- **Starter repos/tools:**
  - GLEAM mobility pipeline (https://www.gleamviz.org/) — global airline + commuting mobility for epidemic modelling
  - CuPy (https://github.com/cupy/cupy) — GPU NumPy for particle filter implementation
  - Epiforecast (verify URL) — real-time epidemic nowcasting framework
  - PYMC (https://github.com/pymc-devs/pymc) — probabilistic programming with GPU JAX/Numba backend for data assimilation
- **CUDA libraries & GPU pattern:** cuRAND for particle resampling, cuBLAS for ensemble matrix operations, cuGraph for mobility graph convolutions; pattern: particle filter with GPU-parallel ODE integration and resampling.

---

---

## 10. Biomechanics, Biomedical Devices & Surgery

### 10.1 FEA of Bone & Tissue 🟢 · Established

- **Deep dive:** Finite-element analysis of bone and soft tissue solves systems of millions of coupled equations relating stress, strain, and material nonlinearity under physiological loading. GPU parallelism targets the sparse-matrix assembly and iterative linear-solver (conjugate-gradient or multigrid) phases, which dominate wall time in large 3D meshes. Co-rotational and total-Lagrangian explicit dynamics (TLED) formulations map naturally to SIMT execution because each element's stiffness update is independent. Bone-remodeling simulations (Wolff's law) couple mechanical fields with density update rules, requiring repeated solve-update-resolve cycles that each benefit from CUDA acceleration. Real-world targets include vertebral fracture prediction, hip-implant stress-shielding, and micro-CT-derived trabecular models with >10 M elements.
- **Key algorithms:** Total Lagrangian Explicit Dynamics (TLED), co-rotational FEM, neo-Hookean / Mooney-Rivlin hyperelasticity, preconditioned conjugate gradient (PCG) with Jacobi or incomplete-Cholesky preconditioners, bone-remodeling (Beaupré–Carter) adaptation loops.
- **Datasets:** FEBio Benchmark Suite — verified test problems for nonlinear biomechanical FEA (https://febio.org/knowledgebase/); Open Knee(s) — subject-specific knee joint FE models with segmented cartilage/bone (https://simtk.org/projects/openknee); Visible Human Project — full CT/MRI cadaver data for mesh generation (https://www.nlm.nih.gov/research/visible/visible_human.html); Bone-Load Database (Bergmann et al.) — in vivo implant load telemetry for hip and knee (https://orthoload.com/).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — open-source nonlinear FE solver for biomechanics, C++, with GPU-solver hooks; NiftySim (https://github.com/eloygarcia/niftysim) — CUDA TLED soft-tissue FE toolkit from UCL; NVIDIA CUDALibrarySamples (https://github.com/NVIDIA/CUDALibrarySamples) — cuSPARSE/cuSolver conjugate-gradient templates; Awesome-Biomechanics (https://github.com/modenaxe/awesome-biomechanics) — curated dataset/tool index.
- **CUDA libraries & GPU pattern:** cuSPARSE (SpMV in PCG inner loop), cuSolver (direct sparse factorization), cuBLAS (dense BLAS), Thrust (parallel reductions); pattern: one CUDA thread per element for stiffness assembly → global atomic scatter into CSR matrix → iterative solver in cuSPARSE.

---

### 10.2 Real-Time Soft-Tissue Deformation for Surgical Simulation 🟡 · Active R&D

- **Deep dive:** Surgical simulators require sub-10 ms deformation updates on organ meshes of tens to hundreds of thousands of elements so that haptic devices can deliver force feedback without perceived lag. Position-Based Dynamics (PBD) and its extended variant XPBD run all constraint projections in parallel, with each particle or constraint mapped to a CUDA thread. The 2024 dissection simulator demonstrated real-time performance on >100 K particles, including topological cuts, using parallelized graph-based shape matching on GPU. Material Point Method (MPM) on GPU further handles cutting and tearing by decoupling Eulerian background grids from Lagrangian particles. Hybrid organ models combining rigid bones with deformable soft tissue use adaptive octree refinement on GPU to concentrate resolution near contact zones.
- **Key algorithms:** Position-Based Dynamics (PBD/XPBD), Total Lagrangian Explicit Dynamics (TLED), graph-based shape matching, Material Point Method (MPM), corotational linear FEM, multigrid preconditioned conjugate gradient, near-second-order Jacobi/Gauss-Seidel elastodynamics (JGS2).
- **Datasets:** SOFA Framework benchmark scenes — laparoscopic and open-surgery deformable organ models (https://www.sofa-framework.org/); Kaggle Liver CT Segmentation — 3D liver meshes for deformation benchmarking (https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation); MRI Breast Tissue Segmentation (nnU-Net preprocessed) for biomechanical modeling (https://arxiv.org/abs/2411.18784); iMSTK Test Suite — pre-built surgical scenario meshes (https://www.imstk.org/).
- **Starter repos/tools:** SOFA Framework (https://github.com/sofa-framework/sofa) — open-source physics engine with GPU PBD plugins and haptic coupling; iMSTK (https://github.com/Kitware/iMSTK) — interactive medical simulation toolkit with CUDA deformation; NVIDIA FleX (https://github.com/NVIDIAGameWorks/FleX) — GPU PBD particle solver adapted for surgical contexts; CRESSim-MPM (verify URL, search "CRESSim MPM surgical simulation GPU") — GPU MPM library for cutting/suturing simulation.
- **CUDA libraries & GPU pattern:** CUDA kernels for per-constraint projection (one thread per constraint in parallel Gauss-Seidel with graph coloring), Thrust for particle neighbor search, cuSPARSE for global stiffness assembly; pattern: coloring-based Gauss-Seidel to avoid write conflicts → warp-shuffle reductions for constraint residuals → atomic updates on shared boundary nodes.

---

### 10.3 Implant & Prosthetic Design Optimization 🟡 · Active R&D

- **Deep dive:** Patient-specific implants (hip, knee, spinal, dental) require iterative structural optimization over high-resolution 3D voxel grids (>1 M elements), where density or level-set fields evolve based on sensitivity analysis from repeated FEA solves. GPU acceleration makes three-dimensional SIMP (Solid Isotropic Material with Penalization) topology optimization tractable: a single density update pass over a 256³ grid requires ~16 M stiffness evaluations that execute in parallel. Lattice-structure implants for osseointegration require multiscale homogenization, computing effective elastic tensors for thousands of unit-cell configurations in parallel on GPU. Bone-remodeling feedback loops then validate implant geometry by simulating load transfer over years of use.
- **Key algorithms:** SIMP topology optimization, density-based level-set method, homogenization of periodic lattices, finite-element sensitivity analysis, optimality criteria (OC) update, bone-remodeling (Weinans/Beaupré) adaptation, multi-objective Pareto optimization.
- **Datasets:** OrthoLoad Implant Loading Database — in vivo hip/knee/spine implant force telemetry (https://orthoload.com/); MICCAI 2023 VerSe Challenge — vertebral shape dataset for spinal implant design (https://verse-challenge.github.io/); Hip Implant Topology Dataset — validated micro-FE lattice endoprostheses (see https://www.nature.com/articles/s41598-024-56327-4); FDA Orthopaedic Simulator Database — standardized fatigue loading profiles (verify URL via FDA.gov).
- **Starter repos/tools:** GPU-Accelerated Topology Optimization (Paulino group, Princeton) (https://paulino.princeton.edu/journal_papers/2013/SMO_13_TowardGPUAccelerated.pdf) — multigrid GPU SIMP reference implementation; Simple and Efficient GPU TO (https://www.sciencedirect.com/science/article/pii/S0045782523001676) — open-source GPU TO code from 2023 CMAME paper (verify repo link in supplementary); FEBio (https://github.com/febiosoftware/FEBio) — sensitivity analysis infrastructure; ToPy (https://github.com/williamhunter/topy) — Python 2D/3D TO (CPU, GPU-extensible reference).
- **CUDA libraries & GPU pattern:** cuSPARSE for repeated sparse FE solves, cuDNN for CNN-based TO surrogate acceleration, Thrust for parallel density-filter convolutions; pattern: element-parallel stiffness + sensitivity computation → parallel density update → GPU multigrid V-cycle for equilibrium solve.

---

### 10.4 Haptic Rendering for Medical Training 🟡 · Active R&D

- **Deep dive:** Haptic devices require force updates at 1 kHz or faster; the GPU must solve deformation and contact in under 1 ms per cycle. Energy-based haptic rendering computes virtual coupling forces from the difference between haptic device position and simulated tissue surface, requiring rapid contact detection and signed-distance-field (SDF) queries. GPU-accelerated SDFs pre-computed on volumetric grids enable sub-millisecond closest-point queries. Arterial catheter simulators, endoscopy trainers, and bone-drilling trainers all demand layered material models (mucosa, submucosa, muscle) with distinct stiffness, requiring per-layer GPU FE subsolvers. The bottleneck is contact resolution at the tool-tissue interface, parallelized over candidate contact pairs.
- **Key algorithms:** Energy-based haptic rendering with virtual coupling, signed-distance-field (SDF) contact detection, XPBD constraint projection, layered viscoelastic material models (Kelvin-Voigt), penumbra-based friction, god-object method for haptic proxy.
- **Datasets:** SOFA haptic benchmark scenes (liver puncture, needle insertion) (https://www.sofa-framework.org/); CholecT50 — laparoscopic cholecystectomy video for ground-truth tissue interaction reference (https://github.com/CAMMA-public/cholect50); Hamlyn Centre Laparoscopic / Robotic Video Dataset (http://hamlyn.doc.ic.ac.uk/vision/); Human Tissue Mechanical Properties Database (Picinbono et al., verify via SpringerLink).
- **Starter repos/tools:** SOFA Framework (https://github.com/sofa-framework/sofa) — modular GPU haptic-enabled simulator with OpenHaptics integration; Haptics-Medical-Simulation (https://github.com/HarrisKomn/Haptics-Medical-Simulation) — SOFA-based lung/bronchus haptic trainer with Geomagic Touch; Open-Source Visuo-Haptic Simulator (https://github.com/ChiaraSapo/Open-Source-Visuo-Haptic-Simulator-for-Surgical-Training) — SOFA-based multi-task haptic trainer; CHAI3D (https://www.chai3d.org) — haptic rendering framework with GPU geometry kernel support.
- **CUDA libraries & GPU pattern:** CUDA kernels for SDF ray marching and contact pair query, cuSPARSE for tissue stiffness subsolve, Thrust for collision broadphase; pattern: GPU-resident SDF updated each deformation step → parallel contact pair generation → energy-gradient force computation → CPU haptic device readout at 1 kHz via shared ring buffer.

---

### 10.5 Gait & Motion-Capture Biomechanics 🟢 · Established

- **Deep dive:** Musculoskeletal gait analysis solves inverse kinematics (IK) and inverse dynamics (ID) to compute joint torques, followed by static optimization or forward-dynamics muscle recruitment minimizing metabolic cost. With 80+ muscles per limb and 200+ time frames per trial, the problem scales linearly with subjects in a cohort, making GPU batch-parallelism over trials the key acceleration strategy. Forward-dynamics predictive simulation using direct collocation (Moco) parallelizes across the collocation mesh nodes. GPU acceleration of Jacobian evaluation in trajectory optimization can achieve 7.7× speedup. Real-time IMU-based gait analysis on edge GPUs allows clinic-floor biomechanics without motion-capture labs.
- **Key algorithms:** Inverse kinematics (damped least-squares), inverse dynamics (Newton-Euler recursive), static optimization (bounded quadratic programming), direct collocation optimal control (Hermite-Simpson), musculotendon Hill-type models, contact detection in foot–ground models, Kalman-filter IMU fusion.
- **Datasets:** GaitRec — 2,084 patient bilateral ground reaction force (GRF) walking trials + 211 healthy controls (https://www.nature.com/articles/s41597-020-0481-z); CMU Motion Capture Database — 2500+ mocap sequences across diverse activities (http://mocap.cs.cmu.edu/); PhysioNet Gait/Posture Database — multi-camera + 17-IMU multimodal gait (https://physionet.org/content/multi-gait-posture/1.0.0/); Gait120 — comprehensive EMG + kinematic dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12177048/).
- **Starter repos/tools:** OpenSim (https://github.com/opensim-org/opensim-core) — gold-standard musculoskeletal simulation; OpenSim Moco (https://github.com/opensim-org/opensim-moco) — direct collocation optimal control with multicore parallelism; Awesome-Biomechanics (https://github.com/modenaxe/awesome-biomechanics) — curated dataset/software index; PyBiomech (https://github.com/felixlb/pybiomech) — Python IMU processing pipeline (GPU-extensible).
- **CUDA libraries & GPU pattern:** cuBLAS for batch matrix inversion in IK Jacobians, Thrust for parallel over-trial static optimization QP, CUDA kernels for Hill-model force-velocity lookup tables; pattern: batch subject/trial parallelism → per-frame Jacobian assembly on GPU → CPU-side IPOPT/CasADi optimal-control solve with GPU Jacobian callbacks.

---

### 10.6 Wearable & Continuous-Sensor Signal Processing 🟢 · Established

- **Deep dive:** Wearable ECG, EMG, EEG, PPG, and IMU streams generate high-throughput multi-channel time-series that require real-time filtering, feature extraction, and classification. GPU acceleration enables sliding-window FFT across 64+ channels simultaneously, convolution-based band-pass filtering in the frequency domain, and inference of deep neural networks (CNN-LSTM, Transformer) on segmented epochs. Continuous-glucose-monitor (CGM) and wearable ECG (Holter) datasets reach billions of samples per patient-day, requiring GPU-accelerated dynamic time warping, anomaly detection, and arrhythmia classification pipelines. Edge GPU (Jetson Orin) deployments compress and quantize models for on-device inference.
- **Key algorithms:** Short-time Fourier transform (STFT), wavelet packet decomposition, matched-filter arrhythmia detection, CNN-LSTM for HAR (human activity recognition), dynamic time warping (DTW), Kalman/Madgwick filter for IMU fusion, federated learning over distributed wearables.
- **Datasets:** PhysioNet/CinC Challenge — ECG arrhythmia (https://physionet.org/); PAMAP2 Physical Activity Monitoring — IMU + heart rate across 18 activities (https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring); MIT-BIH Arrhythmia Database — annotated 2-channel ECG (https://physionet.org/content/mitdb/1.0.0/); CHB-MIT Scalp EEG — epileptic seizure monitoring (https://physionet.org/content/chbmit/1.0.0/).
- **Starter repos/tools:** cuSignal (https://github.com/rapidsai/cusignal) — RAPIDS GPU signal processing library (drop-in scipy.signal on GPU); NeuroKit2 (https://github.com/neuropsychology/NeuroKit) — biosignal processing (CPU; GPU backend extensible); TorchEEG (https://github.com/torcheeg/torcheeg) — GPU-accelerated EEG deep-learning benchmark framework; PhysioNet WFDB Python (https://github.com/MIT-LCP/wfdb-python) — waveform database I/O.
- **CUDA libraries & GPU pattern:** cuFFT (batch FFT over channels), cuDNN (CNN/LSTM inference), CUDA kernels for sliding-window feature extraction; pattern: ring-buffer ingest → batch cuFFT per window → 1D convolution via cuDNN → softmax classification → alert emission with sub-10 ms latency.

---

### 10.7 Smart Prosthetics & Exoskeleton Control 🟡 · Active R&D

- **Deep dive:** Myoelectric prosthetics and powered exoskeletons decode surface EMG or EEG in real time to predict user intent, then execute low-latency torque commands. GPU acceleration runs deep CNNs and recurrent networks for intent classification in under 5 ms, meeting the ~50 ms end-to-end control loop budget. Reinforcement-learning-trained controllers for exoskeleton gait assistance require millions of simulated steps during training (parallelized in GPU physics engines like IsaacGym), then deploy on edge GPUs. Impedance control and admittance control loops for compliant interaction simulate full body-device co-dynamics, with contact forces between limb and exoskeleton computed via GPU FSI.
- **Key algorithms:** CNN / LSTM / Transformer EMG intent classification, model-predictive control (MPC), impedance/admittance control, reinforcement learning (PPO/SAC in IsaacGym), Kalman/extended-Kalman observer for joint state estimation, proportional myoelectric control, adaptive gain scheduling.
- **Datasets:** NinaPro DB5 — 10-DOF hand gestures, surface EMG + IMU from 53 subjects (http://ninapro.hevs.ch/); PhysioNet Lower Limb Prosthetics — transtibial amputee locomotion (https://physionet.org/); BCI Competition IV — motor imagery EEG for upper-limb control (https://www.bbci.de/competition/iv/); exo-H3 IMU dataset — powered exoskeleton kinematics (verify URL via IEEE DataPort).
- **Starter repos/tools:** NVIDIA IsaacGym (https://developer.nvidia.com/isaac-gym) — GPU-parallel RL training for robotic/exoskeleton control; legged_gym (https://github.com/leggedrobotics/legged_gym) — GPU-parallel locomotion RL on IsaacGym; Biopatrec (https://github.com/g-guo/biopatrec) — EMG pattern recognition benchmark platform; BioSig (https://biosig.sourceforge.net/) — open biosignal toolbox with EMG classifiers (CPU, GPU-extensible).
- **CUDA libraries & GPU pattern:** cuDNN (CNN inference for EMG/EEG classification), CUDA kernels for batch EMG windowing and feature extraction, IsaacGym GPU physics for RL training; pattern: 4096 parallel simulated human-exoskeleton environments in IsaacGym → policy gradient update on GPU → policy distillation → edge GPU (Jetson) deployment.

---

### 10.8 Computational Fluid-Structure Interaction for Devices 🟡 · Active R&D

- **Deep dive:** Heart valves, stents, LVADs, and arterial stents involve tightly coupled incompressible fluid (blood) and elastic/rigid solid (leaflets, walls) dynamics that must be co-solved. Immersed boundary methods (IBM) embed flexible structures in Eulerian fluid grids, requiring interpolation and spreading operations that are GPU-parallelized across boundary points. SPH (smoothed particle hydrodynamics) replaces grids with Lagrangian particles, enabling free-surface and high-deformation flows suitable for LVAD impeller modeling. The FSEI-GPU code solves fluid-structure-electrophysiology interaction of the full left heart on a few GPU cards, completing one heartbeat in hours instead of days. Multi-GPU domain decomposition via NCCL enables scaling to whole-cardiovascular-system models.
- **Key algorithms:** Immersed Boundary Method (IBM), Lattice-Boltzmann Method (LBM), ISPH/TLSPH Smoothed Particle Hydrodynamics, arbitrary Lagrangian-Eulerian (ALE) formulation, Navier-Stokes fractional-step solver, hemolysis (GKM model) and thrombosis (biochemical agonist) submodels.
- **Datasets:** 4D Flow MRI Benchmark (HEArt) — time-resolved 3D velocity fields in cardiac chambers (https://arxiv.org/abs/2111.00720); HeartFlow FFRCT coronary dataset (commercial, academic access); Aortic Flow Simulation Database from SimVascular (https://simvascular.github.io/); OpenHeart MRI cohort — segmented cardiac geometries (verify URL via Zenodo).
- **Starter repos/tools:** FSEI-GPU (https://arxiv.org/abs/2103.15187) — CUDA Fortran FSI+electrophysiology heart solver (see ScienceDirect for code link); SimVascular (https://github.com/SimVascular/SimVascular) — patient-specific cardiovascular FSI pipeline; GPU-accelerated IB solver (Bhalla group, https://arxiv.org/html/2605.04335) — OpenACC + CUDA + NCCL extreme-scale IBM; PyFR (https://github.com/PyFR/PyFR) — GPU-native high-order Navier-Stokes solver adaptable to biofluid domains.
- **CUDA libraries & GPU pattern:** CUDA kernels for IBM force-spreading/interpolation, cuFFT for Poisson pressure solve, NCCL for multi-GPU halo exchange, cuSPARSE for FSI coupling matrix; pattern: Eulerian fluid grid partitioned across GPUs → IBM Lagrangian marker forces spread to fluid grid via CUDA kernel → pressure solve via FFT → structure positions updated → halo exchange via NCCL.

---

### 10.9 Dental & Orthodontic Biomechanics Simulation 🟡 · Active R&D

- **Deep dive:** Orthodontic tooth movement depends on PDL (periodontal ligament) stress distribution, alveolar bone remodeling, and contact forces between brackets, wires, and clear aligners — all requiring nonlinear FEA on individually segmented CBCT geometries. GPU acceleration allows the dense contact constraint systems (dozens of tooth-aligner contact pairs per timestep) to be assembled and solved in parallel, enabling treatment planning that runs in minutes rather than hours. Dental implant osseointegration modeling couples elastic bone FEM with a poroelastic fluid-in-pore submodel at the implant interface. Population-scale virtual clinical trials — thousands of patient-specific models run simultaneously — become feasible on GPU clusters.
- **Key algorithms:** Hyperelastic PDL material models (Mooney-Rivlin, Ogden), bone-remodeling (Frost mechanostat), penalty-based contact, mortar contact formulation, thermo-mechanical coupling for composite restorations, coupled poroelastic FEM.
- **Datasets:** CBCT Tooth Segmentation Challenge (ToothFairy, MICCAI 2023) — annotated dental CBCT (https://toothfairy.grand-challenge.org/); 3D Dental Mesh Dataset (Teeth3DS) — 1800 intraoral scans (https://github.com/abenhamadou/3DTeethSeg22_challenge); NIH NIDCR FaceBase craniofacial CT atlas (https://www.facebase.org/); Open Dental Science datasets — clinical records + x-rays (verify URL via opendentalsoftware.com).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — handles PDL and bone-remodeling constitutive models; CGAL (https://github.com/CGAL/cgal) — mesh generation from CBCT segmentations; ITK-SNAP (https://www.itksnap.org/) — CBCT segmentation to mesh pipeline; 3DTeethSeg (https://github.com/abenhamadou/3DTeethSeg22_challenge) — tooth segmentation model for mesh generation.
- **CUDA libraries & GPU pattern:** cuSPARSE/cuSolver for contact-augmented stiffness matrix, CUDA kernels for per-element PDL stress update, Thrust for mortar contact pair enumeration; pattern: element-parallel stiffness assembly → penalty contact augmentation → PCG solve on GPU → bone-density update → geometry export for aligner CAD.

---

### 10.10 Spinal Biomechanics & Intervertebral Disc Modeling 🟡 · Active R&D

- **Deep dive:** The lumbar spine involves poroelastic disc mechanics, facet-joint contact, and large deformation under combined flexion-compression loads, requiring multi-physics FEA with >500 K DOF per motion segment. GPU parallelism compresses the 97.9% time-reduction already demonstrated in automated MRI-to-FEM pipelines (Frontiers 2024) by further accelerating the PCG solver for the full lumbar assembly. Population virtual trials — evaluating thousands of patient-specific spinal constructs after fusion surgery — run overnight on GPU clusters, replacing months of cadaveric testing. GPU-resident bone-density maps updated with DXA-calibrated HU values enable patient-specific fracture risk prediction on clinical timescales.
- **Key algorithms:** Biphasic/poroelastic FEM (Mow-Holmes disc model), hyperelastic anulus fibrosus (fiber-reinforced), penalty facet-joint contact, bone-remodeling, automated mesh generation (Laplacian smoothing + decimation), shape correspondence via non-rigid ICP.
- **Datasets:** VerSe Challenge — 374 CT scans with vertebral shape annotation (https://verse-challenge.github.io/); MICCAI SpineSeg — lumbar MRI segmentation (verify URL via Grand Challenge); CT Spine Dataset (verse2020, Zenodo) — 355 CTs with vertebral instance masks (https://doi.org/10.5281/zenodo.3755323); OrthoLoad Lumbar — in vivo spinal implant forces (https://orthoload.com/).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — built-in biphasic and fiber-reinforced disc models; SpineWeb toolkit — vertebral mesh atlas (http://spineweb.digitalimaginggroup.ca/); TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast CT organ+vertebra segmentation for mesh input; MRI-to-FEM pipeline (Frontiers 2024, verify Zenodo for code) — automated lumbar FE model generation.
- **CUDA libraries & GPU pattern:** cuSPARSE PCG for multi-physics coupled system, CUDA kernels for fiber-reinforced anisotropic stress update, cuDNN for DXA HU calibration regression; pattern: GPU-resident CT density map → automatic mesh generation → fiber orientation interpolation on GPU → coupled solid-fluid PCG solve → fracture risk post-processing.

---

### 10.11 Cell-Membrane & Microstructural Mechanics 🟡 · Active R&D

- **Deep dive:** Red blood cell (RBC) deformability, cancer-cell invasion, and vesicle dynamics are governed by membrane bending elasticity (Helfrich model), cytoskeletal spectrin-network stretching, and viscous fluid-membrane coupling. GPU-accelerated dissipative particle dynamics (DPD) or multi-GPU LBM-IBM simulates thousands of RBCs simultaneously in microchannel flows, capturing population-level distributions of deformability index that are diagnostically relevant. Molecular dynamics of membrane lipid bilayers (GROMACS on GPU) resolves pore formation during electroporation or drug insertion. The bottleneck is the N-body neighbor-list update at each timestep, parallelized via CUDA cell-lists.
- **Key algorithms:** Dissipative particle dynamics (DPD), spectrin-link spring network for RBC cytoskeleton, Helfrich bending elasticity, LBM-IBM coupling, coarse-grained MD (MARTINI force field), Monte Carlo moves for lipid flip-flop.
- **Datasets:** RBC deformability measurements (ektacytometry), DIADEM microfluidic datasets (verify URL); RCSB PDB lipid bilayer structures for MD initialization; Red Cell Project DPD parameter database (verify URL via pubs.acs.org); OpenRBC benchmark — large-scale DPD RBC simulations (verify URL).
- **Starter repos/tools:** GROMACS (https://github.com/gromacs/gromacs) — GPU MD with CUDA/HIP backend, supports CG membrane models; OpenRBC (https://github.com/pnnl/OpenRBC) — massively parallel DPD red blood cell simulator; LAMMPS (https://github.com/lammps/lammps) — GPU-accelerated MD/DPD with many membrane force fields; HemeLB (https://github.com/UCL/hemelb) — LBM blood flow with deformable cell coupling.
- **CUDA libraries & GPU pattern:** CUDA cell-list neighbor search, cuBLAS for force accumulation, NCCL for multi-GPU domain decomposition; pattern: spatial cell-list on GPU → O(N) pair-force evaluation in CUDA kernels → Verlet or velocity-Störmer integrator → periodic boundary halo exchange via NCCL.

---

### 10.12 Microfluidic Device & Organ-on-Chip Simulation 🟡 · Active R&D

- **Deep dive:** Lab-on-a-chip and organ-on-chip devices feature micrometer-scale channels where Re < 1 and Péclet numbers span orders of magnitude, demanding accurate Navier-Stokes + advection-diffusion solutions on geometrically complex domains. Lattice-Boltzmann Method (LBM) maps perfectly to GPU: each lattice node streams and collides independently, achieving memory-bandwidth-bound performance near GPU peak. GPU LBM-DEM (discrete element method) co-simulates cell transport, adhesion, and deformation through microchannels. Design optimization of pillar geometry, channel bifurcations, and gradient generators runs via adjoint sensitivity on GPU, drastically accelerating the design-of-experiment cycle for organ-chip platforms.
- **Key algorithms:** D3Q19/D3Q27 LBM with BGK or MRT collision, immersed boundary coupling for deformable cells, lattice-DEM for rigid particle transport, advection-diffusion for chemical gradient generation, adjoint sensitivity analysis for geometry optimization.
- **Datasets:** Microfluidic Gradient Generator Benchmark (LBM validation, Zenodo); PhysioMimetics organ-chip flow data (verify URL); OpenFOAM microfluidic validation cases (https://www.openfoam.com/); Glioblastoma-on-chip CFD dataset (Frontiers Bioeng 2025) (verify Zenodo).
- **Starter repos/tools:** Palabos (https://gitlab.com/unigespc/palabos) — GPU-capable LBM library for complex fluid dynamics; LEDDS (https://arxiv.org/abs/2512.04997) — portable LBM-DEM GPU simulations; waLBerla (https://www.walberla.net/) — massively parallel LBM framework with GPU support; OpenFOAM (https://github.com/OpenFOAM) — with GPU-accelerated linear solvers via PETSc-CUDA backend.
- **CUDA libraries & GPU pattern:** CUDA kernels for per-node stream-and-collide (one thread per lattice node), cuFFT for spectral pressure solve, Thrust for particle tracking; pattern: GPU-resident 3D lattice → CUDA stream-and-collide kernel → IBM force spreading for deformable cells → chemical concentration advection-diffusion update → device geometry optimization via adjoint.

---

### 10.13 3D Bioprinting Toolpath & Bioink Process Simulation 🟡 · Active R&D

- **Deep dive:** Extrusion-based bioprinting deposits cell-laden hydrogels through a nozzle, where shear stress during extrusion determines post-print cell viability. GPU-accelerated CFD of the nozzle + deposition region (non-Newtonian Carreau fluid) predicts wall shear stress as a function of nozzle geometry, ink rheology, and print speed, enabling parameter optimization in silico before costly biological experiments. Lattice-structure scaffold design — maximizing permeability for nutrient transport while maintaining mechanical stiffness — uses GPU topology optimization with fluid-flow homogenization. Thermal modeling of photopolymerization in DLP/SLA bioprinting on GPU resolves crosslink-front propagation in real time.
- **Key algorithms:** Non-Newtonian Navier-Stokes (Carreau-Yasuda viscosity model), topology optimization with permeability (Darcy-Stokes coupling), heat-transfer / photo-crosslinking kinetics, support-structure generation via GPU ray casting, ML surrogate (XGBoost/MLP) for viability prediction.
- **Datasets:** In silico Bioink Viability Dataset (Zenodo) — extrusion viability vs. shear-stress features (https://zenodo.org/records/11545357); BioInk Rheology Database (verify URL via Biofabrication journal); 3D Bioprinting Benchmarks (verify URL via Zenodo); Scaffold Permeability Benchmark (https://arxiv.org/abs/1104.1028).
- **Starter repos/tools:** in-silico-bioink-viability-prediction (https://github.com/KORINZ/in-silico-bioink-viability-prediction) — ML viability prediction from shear stress; OpenFOAM (https://github.com/OpenFOAM) — non-Newtonian flow solver for nozzle CFD; FEBio (https://github.com/febiosoftware/FEBio) — scaffold mechanical FEA; TPMS Scaffold Generator (verify URL via GitHub) — GPU-accelerated triply-periodic-minimal-surface lattice generation.
- **CUDA libraries & GPU pattern:** CUDA kernels for non-Newtonian viscosity update per cell, cuFFT for spectral pressure solve, cuDNN for surrogate viability model inference; pattern: parametric nozzle geometry → GPU Navier-Stokes solve for shear-stress field → shear-stress statistics fed to GPU ML surrogate → output: print parameters vs. predicted viability Pareto front.

---

### 10.14 LVAD / Rotary Blood Pump CFD & Hemolysis Prediction 🟡 · Active R&D

- **Deep dive:** Left ventricular assist devices (LVADs) expose blood to high shear stress at impeller blades, triggering hemolysis and thrombus formation. Patient-specific CFD with a rotating reference frame and moving mesh requires GPU-accelerated Navier-Stokes solutions on unstructured grids with ~5 M cells. The hemolysis index (power-law Giersiepen-Wurzinger model) is integrated along particle pathlines, computed by GPU-resident Lagrangian particle tracking. The 2024 simulation study demonstrated that hyperadhesion of activated platelets plays a dominant role in LVAD thrombosis at high rotor speeds. Design variants (impeller blade count, tip clearance) are evaluated in batches on GPU to build surrogate response surfaces for optimization.
- **Key algorithms:** Rotating reference frame Navier-Stokes (MRF/sliding mesh), Lagrangian particle tracking for hemolysis integration, platelet activation and thrombosis model (7-agonist biochemical cascade), power-law hemolysis (GKM), Euler-Euler two-phase (plasma + RBC) formulation, immersed boundary for rotor blades.
- **Datasets:** FDA Benchmark Pump Dataset — PIV-measured flow in centrifugal/axial blood pumps (https://www.fda.gov/science-research/about-science-research-fda/computational-modeling-biomedical-devices); Multi-GPU IB Hemodynamics Benchmark (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7402620/); LVAD Thrombosis Simulation Archive (see https://arxiv.org/abs/2312.04761); HeartMate 3 geometry (anonymized, verify via Frontiers Cardiovasc Med).
- **Starter repos/tools:** OpenFOAM (https://github.com/OpenFOAM) — rotating machinery solvers (MRFSimpleFoam) with GPU linear-algebra backends; HemeLB (https://github.com/UCL/hemelb) — GPU LBM for cardiovascular flows; IBM at Extreme Scale (https://arxiv.org/html/2605.04335) — OpenACC+CUDA+NCCL IBM solver; CUDA particle tracking kernel templates (https://github.com/NVIDIA/CUDALibrarySamples).
- **CUDA libraries & GPU pattern:** CUDA rotating-frame velocity interpolation kernels, cuSPARSE for pressure-velocity coupling, Thrust for particle trajectory integration; pattern: GPU unstructured CFD mesh → MRF velocity correction per cell → Lagrangian particle release → CUDA pathline integration → per-particle hemolysis accumulation → thrombosis probability map.

---

### 10.15 Cochlear Implant Computational Modeling 🟡 · Active R&D

- **Deep dive:** Cochlear implant (CI) electrodes stimulate spiral ganglion neurons via current fields that spread through complex fluid-filled scala tympani geometries. GPU-accelerated FEM on micro-CT-derived cochlear geometries computes the full 3D voltage distribution across the spiral ganglion fiber population in under a second, enabling real-time comparison of electrode array designs. Multi-compartment auditory nerve fiber (ANF) cable models are integrated in parallel on GPU — one thread per fiber per timestep — to predict neural firing patterns from arbitrary stimulation waveforms. Population-model simulations over thousands of virtual patients with varying cochlear anatomy quantify inter-subject variability in electrode coupling.
- **Key algorithms:** Volume-conductor FEM (bidomain), multi-compartment Hodgkin-Huxley cable models for ANF, psychoacoustic loudness growth modeling, Green's function electrode-impedance computation, Monte Carlo sampling over cochlear geometry populations.
- **Datasets:** Cochlear Micro-CT Atlas (25 ANF traced geometries, see https://www.frontiersin.org/articles/10.3389/fnins.2025.1639092); Electrical Stimulation Human Cochlea Dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6915103/); SIMBIOsys Cochlear Models (https://www.upf.edu/web/simbiosys/cochlear-implants); PhysioNet auditory nerve response databases (verify via physionet.org).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — bidomain volume conductor FEM; NEURON simulator GPU branch (https://github.com/neuronsimulator/nrn) — parallel ANF cable integration; SimNIBS (https://github.com/simnibs/simnibs) — FEM for electrostimulation (adaptable to cochlear geometry); Cochlear FEM pipeline (SIMBIOsys UPF, verify URL at UPF site) — CI-specific meshing and solving workflow.
- **CUDA libraries & GPU pattern:** cuSPARSE/cuSolver for bidomain FEM voltage solve, CUDA kernels for per-fiber HH cable ODE integration (embarrassingly parallel over ANFs), cuRAND for stochastic threshold variability; pattern: GPU FEM voltage field → per-fiber interpolation of extracellular potential → parallel ODE integration of HH equations → spike-time extraction → population audiogram prediction.

---

### 10.16 Surgical Robot Path Planning & Collision Detection 🟡 · Active R&D

- **Deep dive:** Robotic-assisted surgery (e.g., da Vinci, Mako) requires real-time collision-free trajectories for multiple articulated arms moving near deformable anatomy. GPU parallel motion planning (RRT*, PRM) checks thousands of configuration-space samples for collision against a GPU-resident signed-distance-field (SDF) of the patient anatomy simultaneously, achieving path generation in under 100 ms — 50–100× faster than CPU planners. Deep-learning collision detectors trained in simulation (Learning-from-Simulation, 2025) replace explicit geometric checks with GPU neural networks, handling soft-tissue deformation that classical rigid-body checkers cannot. The GPU also runs online model-predictive controllers that re-plan at 50 Hz as tissue moves during respiration.
- **Key algorithms:** GPU-parallel RRT*/PRM with SDF collision query, signed-distance-field generation via GPU ray marching, neural collision detector (implicit neural representation), MPC for force-controlled insertion, generalized momentum observer for external-force estimation.
- **Datasets:** SurgRobotics Dataset — da Vinci tool tracking + anatomy meshes (verify URL via MICCAI); SCARED Dataset — stereo depth reconstruction in laparoscopy (https://endovissub2019-scared.grand-challenge.org/); MICCAI 2024 Surgical Scene Segmentation Challenge (verify URL via Grand Challenge); CholecT50 (https://github.com/CAMMA-public/cholect50) — tool-tissue interaction labels.
- **Starter repos/tools:** GPU-based Parallel Collision Detection (UNC Gamma group, http://gamma.cs.unc.edu/gplanner/) — GPU PRM reference; cuRobo (https://github.com/NVlabs/curobo) — NVIDIA CUDA-accelerated robot motion generation; SOFA Framework (https://github.com/sofa-framework/sofa) — deformable anatomy + robot coupling; IsaacGym (https://developer.nvidia.com/isaac-gym) — GPU parallel surgical-robot RL training.
- **CUDA libraries & GPU pattern:** CUDA kernels for SDF generation (parallel ray marching), cuDNN for neural collision network inference, Thrust for parallel RRT sample feasibility checks; pattern: GPU SDF updated from tissue deformation → 4096 configuration samples checked in parallel → feasible path selected → MPC re-plan at 50 Hz → torque commands dispatched.

---

### 10.17 AR/VR Surgical Visualization & Real-Time Volume Rendering 🟡 · Active R&D

- **Deep dive:** Augmented-reality surgical guidance requires sub-20 ms end-to-end latency from imaging sensor to rendered overlay, encompassing depth estimation, organ segmentation, tissue deformation tracking, and volume rendering on a single GPU. Ray-cast volume rendering of intraoperative ultrasound or cone-beam CT benefits from GPU empty-space skipping (sparse voxel octrees) and gradient-based shading. Neural rendering (NeRF / Gaussian splatting) trained on intraoperative images can reconstruct deforming organ surfaces in real time on an RTX GPU. The GPU parallelizes pixel-independent ray traversal, making volume rendering a textbook GPU workload with one thread per pixel.
- **Key algorithms:** Ray-cast volume rendering, gradient-magnitude transfer functions, sparse voxel octree traversal, NeRF / 3D Gaussian splatting for scene reconstruction, SLAM-based tracking, depth-from-stereo (disparity networks), mesh rasterization for AR overlay.
- **Datasets:** SciVis Contest Medical Volumes — benchmark CT/MR volumes for rendering (https://scivis.github.io/); SCARED stereo laparoscopy depth dataset (https://endovissub2019-scared.grand-challenge.org/); Hamlyn Robotic Vision Dataset (http://hamlyn.doc.ic.ac.uk/vision/); MICCAI 2023 Endoscopic Vision Challenge (verify URL via Grand Challenge).
- **Starter repos/tools:** NVIDIA CUDA-GL rendering samples (https://github.com/NVIDIA/cuda-samples) — volumerender sample; 3D Gaussian Splatting (https://github.com/graphdeco-inria/gaussian-splatting) — real-time neural rendering; VTK/vtkVolume (https://github.com/Kitware/VTK) — volume rendering with GPU acceleration; MONAI Label (https://github.com/Project-MONAI/MONAILabel) — real-time intraoperative segmentation.
- **CUDA libraries & GPU pattern:** CUDA texture objects (hardware-interpolated volume sampling), cuDNN for segmentation inference, OpenGL-CUDA interop for zero-copy display; pattern: intraoperative CT/US volume uploaded as 3D CUDA texture → one thread per display pixel ray-marches texture → alpha-compositing accumulation → OpenGL framebuffer blit → AR overlay.

---

---

## 11. Biotechnology, Bioprocess & Synthetic Biology

### 11.1 Protein Engineering / Directed Evolution In Silico 🟡 · Active R&D

- **Deep dive:** Machine-learning-guided directed evolution replaces physical screening with GPU-accelerated fitness prediction, scoring millions of sequence variants per second using protein language models (ESM-2) or structure-based Rosetta energy functions. EVOLVEpro (Science 2025) demonstrated rapid in silico directed evolution by proposing and filtering variants with GPU-deployed LLM embeddings. Batched GPU inference over combinatorial mutation libraries (10⁸–10¹² sequences) identifies beneficial mutations orders of magnitude faster than laboratory selection. The key parallelism is embarrassingly parallel: each sequence variant scores independently.
- **Key algorithms:** Protein language model (ESM-2) embeddings + fitness regression, directed evolution with Bayesian optimization (GP or Bayesian neural network), structure-based ΔΔG prediction (Rosetta fast-relax, FoldX), zero-shot fitness scoring via masked-language-model log-odds, gradient-based sequence optimization via differentiable fitness surrogate.
- **Datasets:** ProteinGym Substitution Benchmarks — 250+ deep mutational scanning (DMS) datasets across protein families (https://proteingym.org/); Envision (PABP, UBE4B) fitness landscapes; Fluorescent Protein Dataset (GFP) — 56 K variants with fluorescence labels (https://github.com/fhalab/FLIP); FLIP Benchmarks — standardized fitness landscape benchmarks (https://github.com/J-SNACKKB/FLIP).
- **Starter repos/tools:** ESM (https://github.com/facebookresearch/esm) — Meta FAIR ESM-2 + ESMFold GPU protein language model; EVOLVEpro (verify URL at bakerlab.org or GitHub) — in silico directed evolution pipeline; ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU sequence design from backbone; Fitness-Prediction-Benchmark (https://github.com/J-SNACKKB/FLIP) — DMS benchmark datasets and baseline models.
- **CUDA libraries & GPU pattern:** cuDNN for Transformer forward pass over batched sequences, Flash Attention for memory-efficient long-sequence attention, mixed-precision (BF16) for throughput; pattern: encode 10⁶ variants as token batch → GPU LLM forward pass → fitness score vector → Bayesian acquisition function selects next round → iterate.

---

### 11.2 Enzyme Design & Catalysis Modeling 🟡 · Active R&D

- **Deep dive:** Computational enzyme design requires evaluating active-site geometry, transition-state stabilization, and substrate binding simultaneously. GPU-accelerated QM/MM (quantum mechanics / molecular mechanics) couples a DFT or semi-empirical QM region around the catalytic residues with a classical MM region of the full enzyme, enabling thousands of candidate enzyme structures to be ranked. Rosetta enzyme design generates theozyme scaffolds and then repacks surrounding residues on GPU. AlphaFold-2 structure prediction + ProteinMPNN sequence design creates novel enzyme candidates at scale. De novo enzyme design for non-natural reactions (Diels-Alder, retro-aldol) has been demonstrated computationally; GPU acceleration is the bottleneck to scaling to large combinatorial searches.
- **Key algorithms:** Rosetta enzyme design (RIF docking, match/scaffold search), QM/MM (ONIOM, pDynamo), transition-state theory rate prediction, directed evolution fitness landscape modeling, SE(3)-equivariant active-site design (BindCraft/RFdiffusion), Monte Carlo backrub for enzyme refinement.
- **Datasets:** BRENDA Enzyme Database — kinetics, substrates, organisms (https://www.brenda-enzymes.org/); SABIO-RK — enzyme kinetic parameters (https://sabiork.h-its.org/); UniProt/SwissProt enzyme entries (https://www.uniprot.org/); M-CSA Mechanism and Catalytic Site Atlas (https://www.ebi.ac.uk/thornton-srv/m-csa/).
- **Starter repos/tools:** RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — diffusion-based active-site design; PyRosetta (https://github.com/RosettaCommons/pyrosetta) — GPU-compatible Rosetta Python bindings; GROMACS (https://github.com/gromacs/gromacs) — GPU QM/MM enzyme MD via ORCA/CP2K coupling; DeepMind AlphaFold2 (https://github.com/google-deepmind/alphafold) — structure prediction for enzyme scaffold validation.
- **CUDA libraries & GPU pattern:** cuDNN for Rosetta energy term neural-network surrogate, CUDA ONIOM QM/MM kernels via GROMACS GPU engine, cuFFT for periodic electrostatics (PME); pattern: RFdiffusion generates active-site scaffold on GPU → ProteinMPNN designs sequence → GPU MD relaxation → GPU QM/MM ΔG‡ evaluation → rank and select.

---

### 11.3 Antibody Design & Affinity Maturation 🟡 · Active R&D

- **Deep dive:** Antibody engineering spans CDR-loop design, affinity maturation, and developability optimization — each requiring GPU inference over large sequence/structure spaces. RFdiffusion-Antibody (Baker Lab, 2025) generates novel CDR-H3 loops conditioned on antigen epitopes via SE(3)-equivariant diffusion on GPU. Affinity maturation via flow matching (AffinityFlow, 2025) guides sequence trajectories toward high-affinity regions on GPU. Structure-aware inverse folding (AbMPNN) redesigns CDR sequences while preserving Fv geometry. The AbBiBench benchmark (2025) standardizes evaluation across 10+ affinity maturation methods. The FDA approved 13 new monoclonal antibodies in 2024, underlining the industrial importance of accelerated in silico design.
- **Key algorithms:** SE(3)-equivariant diffusion (RFdiffusion), flow matching for affinity maturation (AffinityFlow), inverse folding (AbMPNN/ProteinMPNN), language-model-guided combinatorial optimization (LLM + genetic algorithm + simulated annealing), ΔΔG binding affinity prediction (Rosetta flex_ddg, FoldX), multi-objective developability scoring.
- **Datasets:** SAbDab — Structural Antibody Database, 10000+ Fv structures (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); AbBiBench Benchmark — standardized affinity maturation evaluation (https://arxiv.org/abs/2506.04235); OAS — Observed Antibody Space, 2B+ sequences (https://opig.stats.ox.ac.uk/webapps/oas/oas); CoV-AbDab — SARS-CoV-2 antibody database (https://opig.stats.ox.ac.uk/webapps/covabdab/).
- **Starter repos/tools:** RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — CDR design via SE(3) diffusion (RFdiffusion2 available 2025); ABodyBuilder3 (https://github.com/oxpig/ABDB) — GPU antibody structure prediction; ImmuneBuilder (https://github.com/oxpig/ImmuneBuilder) — GPU-fast Fv structure modeling; AbMPNN/ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU CDR sequence design.
- **CUDA libraries & GPU pattern:** Flash Attention for long CDR+antigen context, cuDNN Transformer inference for LLM-based sequence scoring, CUDA kernels for parallel ΔΔG evaluation; pattern: antigen epitope input → RFdiffusion GPU generates CDR scaffold ensemble → AbMPNN scores/redesigns sequences in batch → GPU ΔΔG filter → developability scoring → top candidates to wet lab.

---

### 11.4 Synthetic-Biology Genetic-Circuit Design & Simulation 🟡 · Active R&D

- **Deep dive:** Genetic-circuit design requires stochastic simulation (Gillespie SSA) of regulatory networks with hundreds of species and reactions, then optimization of promoter strengths, RBS sequences, and protein copy numbers to achieve target transfer-function shapes. GPU parallelism runs thousands of independent SSA trajectories simultaneously on a single card — each trajectory is a separate CUDA stream — reducing Monte Carlo ensemble variance estimation from hours to seconds. Deterministic ODE simulation (Hill kinetics) of large gene regulatory networks (GRNs) further benefits from GPU batch-ODE solvers (cuSolver + custom RK4). Bayesian optimization over the genetic parameter space closes the design loop.
- **Key algorithms:** Gillespie Stochastic Simulation Algorithm (SSA), tau-leaping (accelerated SSA), deterministic ODE integration with Hill-function kinetics, Bayesian optimization (GP-UCB) for parameter tuning, coarse-grained thermodynamic models for promoter strength, Boolean logic gate composition.
- **Datasets:** iGEM Registry of Standard Biological Parts — promoter/RBS/gene part catalog (https://parts.igem.org/); SBOL Designer parts library (https://sboldesigner.github.io/); BioBrick Characterization Database (verify URL via SynBioHub); Promoter Strength Library (Anderson promoter series) (verify URL via parts.igem.org).
- **Starter repos/tools:** Tellurium (https://github.com/sys-bio/tellurium) — Python ODE/SSA simulator for SBML models with CUDA-extensible solvers; GillesPy2 (https://github.com/StochSS/GillesPy2) — Python SSA with GPU acceleration roadmap; COPASI (https://github.com/copasi/COPASI) — biochemical network simulator with parallel parameter scanning; iBioSim (https://github.com/MyersResearchGroup/iBioSim) — genetic circuit design + simulation framework.
- **CUDA libraries & GPU pattern:** CUDA kernels for parallel SSA trajectories (one trajectory per thread block), cuRAND for per-trajectory random number streams, cuSolver for stiff ODE Jacobian factorization; pattern: genetic circuit model → 10⁴ GPU SSA trajectories in parallel → histogram-based transfer-function estimation → Bayesian optimizer proposes new promoter parameters → iterate.

---

### 11.5 Bioreactor & Fermentation CFD 🟡 · Active R&D

- **Deep dive:** Industrial bioreactors exhibit complex turbulent flow, gas-liquid mass transfer (O₂/CO₂), and biological reactions that mutually couple over timescales from milliseconds (bubble coalescence) to hours (cell growth). GPU-accelerated LBM or finite-volume CFD resolves the multi-phase (broth + bubbles) hydrodynamics on meshes with millions of cells, enabling scale-up prediction from bench to 10,000-L fermenters. CFD-metabolic hybrid models link local glucose/O₂ concentrations (from CFD) to spatially-resolved metabolic rates (from flux-balance analysis), identifying gradients that stress industrial cultures. Real-time digital twins combining online sensor data with GPU CFD surrogates enable closed-loop bioreactor control.
- **Key algorithms:** Turbulent Navier-Stokes (k-ε / k-ω SST), volume-of-fluid (VOF) gas-liquid interface, population balance model for bubble size distribution, Euler-Euler two-phase flow, lattice-Boltzmann for pore-scale mass transfer, physics-informed neural network surrogate, computational morphology (impeller blade design).
- **Datasets:** DECHEMA Bioreactor Flow Dataset — PIV measurements in stirred tanks (verify URL via dechema.de); OpenFOAM BioReactor Tutorial Cases (https://www.openfoam.com/); CHO Fed-Batch Time Course Data (BioNumbers DB, https://bionumbers.hms.harvard.edu/); Zenodo fermentation monitoring datasets (search Zenodo "fed-batch bioreactor").
- **Starter repos/tools:** OpenFOAM (https://github.com/OpenFOAM) — gas-liquid bioreactor multiphase solvers (multiphaseEulerFoam) with GPU linear algebra; Palabos (https://gitlab.com/unigespc/palabos) — GPU LBM for porous-media and bubble-column flows; NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — physics-informed surrogate training for CFD; COBRApy (https://github.com/opencobra/cobrapy) — flux-balance metabolic modeling for CFD coupling.
- **CUDA libraries & GPU pattern:** cuSPARSE for pressure-velocity coupling in SIMPLE algorithm, CUDA kernels for VOF interface reconstruction, cuDNN for PINN surrogate inference; pattern: full CFD on GPU with AMG preconditioner → extract local O₂/glucose fields → pass to GPU flux-balance metabolic model → update volumetric reaction terms → iterate time step.

---

### 11.6 Metabolic Engineering & Strain Design 🟡 · Active R&D

- **Deep dive:** Metabolic engineering seeks genetic modifications (gene knockouts, overexpression, heterologous pathway insertion) that maximize desired metabolite production. GPU acceleration enables genome-scale flux-balance analysis (FBA) to be solved for millions of genetic perturbation combinations in parallel — each FBA is an independent LP problem — dramatically outpacing CPU batch FBA. Constraint-based strain design algorithms (OptKnock, MOMA) search exponentially large combinatorial spaces, tractable only with GPU parallelism. Kinetic whole-pathway models (ODEs with hundreds of reactions) can be fitted to multi-omics data using GPU-accelerated Bayesian MCMC (NUTS/HMC).
- **Key algorithms:** Flux Balance Analysis (LP, GPU batch), Dynamic FBA, OptKnock / RobustKnock strain design, ensemble kinetic modeling (EKM), Bayesian MCMC parameter estimation (NUTS/HMC), genome-scale metabolic network reduction (data-driven, 2025).
- **Datasets:** BiGG Models — 108 genome-scale metabolic models (https://bigg.ucsd.edu/); KEGG Metabolic Pathways (https://www.kegg.jp/kegg/pathway.html); MetaboLights — metabolomics raw data (https://www.ebi.ac.uk/metabolights/); CHO-GEM Genome-Scale Model — CHO cell metabolic network (verify URL via Zenodo/BioModels).
- **Starter repos/tools:** COBRApy (https://github.com/opencobra/cobrapy) — FBA/FVA in Python; cameo (https://github.com/biosustain/cameo) — strain design algorithms including OptKnock; MICOM (https://github.com/micom-dev/micom) — microbiome community FBA; GPU-FBA (verify URL, search "GPU flux balance analysis CUDA") — CUDA batch LP solver for parallel strain enumeration.
- **CUDA libraries & GPU pattern:** CUDA LP solver (per-combination parallel simplex/interior-point), cuBLAS for stoichiometric matrix operations, Thrust for parallel combinatorial enumeration; pattern: stoichiometric matrix resident on GPU → one thread block per genetic perturbation combination → parallel FBA solve → objective value reduction → top strains ranked.

---

### 11.7 mRNA / Vaccine Sequence Design 🟡 · Active R&D

- **Deep dive:** mRNA vaccine efficacy depends on optimal codon usage (for high ribosome translation), minimum free-energy (MFE) secondary structure (for stability), and 5'-UTR/3'-UTR element design. LinearDesign finds near-optimal MFE + CAI jointly in 11 minutes via dynamic programming on a lattice (analogous to CYK parsing), and GPU parallelization of the lattice can further accelerate multi-target vaccine design. VaxPress (2024) runs iterative codon optimization with customizable scoring functions including codon adaptation index, GC content, repeat minimization, and vaccine-specific immune stimulation features. Deep generative models (Nature Communications 2025) optimize codon sequences via GPU-trained VAEs, improving translation efficiency measurably in cell-free expression.
- **Key algorithms:** Minimum-free-energy (MFE) RNA folding (Zuker dynamic programming), codon adaptation index (CAI) optimization, LinearDesign lattice-DP algorithm, epitope prediction (MHC-I/II binding), RNA-structure gradient optimization, deep generative codon design (VAE/flow matching).
- **Datasets:** NCBI RefSeq CDS — validated coding sequences for codon usage tables (https://www.ncbi.nlm.nih.gov/refseq/); RNAcentral — non-coding RNA + UTR sequences (https://rnacentral.org/); VaxPress Test Suite — 100 vaccine antigens for benchmarking (https://github.com/ChangLabSNU/VaxPress); IEDB — immune epitope database for T/B cell responses (https://www.iedb.org/).
- **Starter repos/tools:** LinearDesign (https://github.com/LinearDesignSoftware/LinearDesign) — fast MFE+CAI co-optimization; VaxPress (https://github.com/ChangLabSNU/VaxPress) — codon optimizer with LinearDesign integration; VaxLab (https://github.com/ChangLabSNU/VaxLab) — integrated design platform; CodonBERT (verify URL, search "CodonBERT GitHub") — BERT-based codon optimization model (GPU inference).
- **CUDA libraries & GPU pattern:** cuDNN for Transformer-based codon sequence scoring (CodonBERT), CUDA dynamic-programming kernels for parallel MFE computation across sequence windows, Flash Attention for long mRNA sequence context; pattern: target antigen CDS → GPU LinearDesign DP → VaxPress iterative refinement on GPU → GPU epitope scoring → ranked candidates.

---

### 11.8 CRISPR System Design & Modeling 🟡 · Active R&D

- **Deep dive:** CRISPR guide RNA (gRNA) design requires genome-wide off-target site enumeration (all 20-mer matches with ≤4 mismatches in 3 billion base pairs), scoring each off-target's likelihood based on mismatch position and type. GPU-accelerated exact string matching (GPU BWT/FM-index) reduces the genome scanning from hours to minutes. Deep learning off-target predictors (CNN, BiGRU, BERT-based LLMs) run on GPU over millions of candidate gRNAs in parallel. The CRISOT tool suite derives RNA-DNA molecular interaction fingerprints from GPU-accelerated MD simulations of Cas9-gRNA-DNA ternary complexes to compute structural off-target scores.
- **Key algorithms:** FM-index / BWT genome search on GPU, CNN/BiGRU/Transformer off-target classifiers, molecular dynamics of Cas9 R-loop formation, energy minimization for gRNA thermodynamic stability, seqmap-style GPU hash table for rapid k-mer matching, CRISOT molecular fingerprinting.
- **Datasets:** CRISPOR Guide RNA Dataset — experimentally validated on/off-target activities (https://crispor.tefor.net/); CIRCLE-seq Off-Target Dataset (Tsai et al., Nature Methods) — unbiased off-target identification; Genome-wide CRISPR off-target benchmark (https://www.nature.com/articles/s41467-023-42695-4); ClinVar — disease-relevant on-target loci for therapeutic gRNA selection (https://www.ncbi.nlm.nih.gov/clinvar/).
- **Starter repos/tools:** CRISPOR (https://github.com/maximilianh/crisporWebsite) — GPU-accelerated guide design pipeline; CRISPRscan (https://www.crisprscan.org/) — on/off-target prediction (verify GitHub URL); DeepCRISPR (https://github.com/jieccccc/DeepCRISPR) — CNN off-target prediction with GPU inference; GROMACS (https://github.com/gromacs/gromacs) — GPU MD of Cas9 R-loop for CRISOT-style fingerprinting.
- **CUDA libraries & GPU pattern:** CUDA BWT string index for GPU genome scanning, cuDNN for CNN/Transformer off-target scoring over batches of gRNAs, cuRAND for MD trajectory generation; pattern: 20-mer gRNA → GPU BWT scan of genome → candidate off-target list → batch GPU DL scoring → filter by specificity score → MD fingerprint for top candidates.

---

### 11.9 Flow Cytometry & High-Content Screening Analysis 🟡 · Active R&D

- **Deep dive:** Modern cell sorters generate 10⁶ cells/second at 20–50 parameters per event; high-content screening (HCS) platforms image millions of cells per plate with 10+ channels. GPU-accelerated dimensionality reduction (GPU-UMAP, GPU-TSNE via RAPIDS cuML) and clustering (GPU-HDBSCAN, GPU-PhenoGraph) turn 30-minute analyses into seconds, enabling real-time sort gates. GPU-accelerated CellProfiler-style morphological feature extraction processes 96-well plate images in minutes instead of hours. Deep-learning classifiers (ResNet, ViT) deployed on GPU identify rare phenotypes (1-in-10⁵ events) with high sensitivity.
- **Key algorithms:** GPU-UMAP (approximate nearest-neighbor with NN-descent), GPU-HDBSCAN, GPU FlowSOM self-organizing map, GPU PhenoGraph graph-based clustering, GPU CellPose segmentation for HCS, Wasserstein distance for batch-effect correction, GPU deep learning rare-event classifier.
- **Datasets:** FlowRepository — public flow cytometry FCS files (https://flowrepository.org/); JUMP-CP — 116 K compound HCS morphological profiles, RxRx cell-painting images (https://jump-cellpainting.broadinstitute.org/); Cell Painting Gallery (Broad Institute) — 140 TB cell images (https://registry.opendata.aws/cellpainting-gallery/); Human Protein Atlas imaging (https://www.proteinatlas.org/).
- **Starter repos/tools:** RAPIDS cuML (https://github.com/rapidsai/cuml) — GPU UMAP/TSNE/HDBSCAN for cytometry analysis; CellProfiler (https://github.com/CellProfiler/CellProfiler) — HCS morphological profiling (with GPU CellPose segmentation); CellPose (https://github.com/mouseland/cellpose) — GPU-accelerated cell segmentation; FlowKit (https://github.com/whitews/FlowKit) — FCS file processing (CPU; upstream of GPU analysis).
- **CUDA libraries & GPU pattern:** cuML GPU-UMAP, cuDNN for ResNet cell image classifier, CUDA 2D convolution kernels for morphological feature extraction; pattern: FCS/image batch ingest → GPU feature extraction → GPU-UMAP embedding → GPU-HDBSCAN clustering → rare-event gating → real-time sort decisions.

---

### 11.10 Antibody Developability Prediction & Optimization 🟡 · Active R&D

- **Deep dive:** Even potent antibodies fail if they aggregate, have high viscosity, polyreact with off-targets, or are immunogenic — properties collectively called developability. Predicting all six key developability flags (pI, hydrophobicity, aggregation propensity, poly-specificity, expression level, immunogenicity) from sequence alone via GPU-trained BERT-style models enables early-stage winnowing of design libraries with millions of variants. Multi-property Pareto optimization across affinity and developability runs on GPU via multi-objective Bayesian optimization over learned surrogate surfaces.
- **Key algorithms:** Protein LLM fine-tuning for developability regression, multi-objective Bayesian optimization (qParEGO), aggregation prediction (camsol/spatial aggregation propensity), immunogenicity prediction (T-cell epitope presentation MHC-II), expression-level prediction from sequence.
- **Datasets:** SAFit dataset — self-association from AstraZeneca (verify URL via Bioinformatics journal); TAP dataset — Therapeutic Antibody Profiler developability (https://opig.stats.ox.ac.uk/webapps/oas/tap); OAS (https://opig.stats.ox.ac.uk/webapps/oas/oas) — natural antibody sequence space for pre-training; CoV-AbDab (https://opig.stats.ox.ac.uk/webapps/covabdab/) — experimental affinity + neutralization data.
- **Starter repos/tools:** Therapeutic Antibody Profiler (TAP) (https://opig.stats.ox.ac.uk/webapps/oas/tap) — web server + scoring functions; AbLang (https://github.com/oxpig/AbLang) — antibody language model pre-training; AntiFold (https://github.com/oxpig/AntiFold) — GPU antibody inverse folding for sequence redesign; ANARCI (https://github.com/oxpig/ANARCI) — antibody numbering for feature alignment.
- **CUDA libraries & GPU pattern:** cuDNN for Transformer LLM inference over antibody sequence batches, Flash Attention for variable-length CDR context, CUDA kernels for parallel developability feature computation; pattern: million-variant library → batch GPU LLM embedding → multi-property regression → GPU Pareto front computation → top candidates advance to wet-lab synthesis.

---

### 11.11 CHO Cell & Mammalian Bioprocess Digital Twin 🟡 · Active R&D

- **Deep dive:** Chinese Hamster Ovary (CHO) cell fed-batch cultures for monoclonal antibody production exhibit complex interplay of metabolism, glycosylation, dissolved oxygen, and pH dynamics that are expensive to characterize experimentally. GPU-accelerated hybrid digital twins (Nature npj 2026) couple ODE kinetic models with genome-scale FBA on GPU, with LSTM networks trained on GPU correcting model-plant mismatch online. Bayesian parameter estimation with HMC (GPU-accelerated via NumPyro/JAX) fits hundreds of kinetic parameters to multi-omics fed-batch data in hours. Real-time digital twins receive PAT (process analytical technology) sensor streams and predict glycoform distributions ahead of time for automated feeding control.
- **Key algorithms:** Hybrid mechanistic-ML (ODE + LSTM), genome-scale metabolic modeling (FBA, GEM reduction), Bayesian HMC parameter estimation, Gaussian process regression for process uncertainty, PLS/PCA for spectroscopic soft sensing, dynamic FBA (dFBA).
- **Datasets:** CHO Fed-Batch Time-Course Metabolomics (BioRxiv 2025, Zenodo) — 12 cultures with 80+ metabolite time profiles; BioNumbers Database — CHO-specific growth/uptake rates (https://bionumbers.hms.harvard.edu/); BioModels Database — published CHO kinetic models (https://www.ebi.ac.uk/biomodels/); JGI/DBTBS gene expression compendium for CHO pathway analysis (verify URL).
- **Starter repos/tools:** COBRApy (https://github.com/opencobra/cobrapy) — GEM FBA for CHO; NumPyro (https://github.com/pyro-ppl/numpyro) — GPU Bayesian HMC for kinetic parameter estimation; PyTorch LSTM (https://pytorch.org/) — hybrid ODE-LSTM digital twin training; Pyomo (https://github.com/Pyomo/pyomo) — algebraic modeling for dynamic FBA optimization.
- **CUDA libraries & GPU pattern:** cuDNN for LSTM training/inference, JAX GPU backend for HMC MCMC, CUDA batch LP for parallel FBA across time points; pattern: online PAT sensor feed → GPU LSTM state update → GPU GEM FBA at current metabolite concentrations → kinetic ODE integration → feeding strategy MPC → compare to lab measurements → Bayesian posterior update.

---

### 11.12 Downstream Processing & Chromatography Simulation 🟡 · Active R&D

- **Deep dive:** Protein A affinity, ion-exchange, and size-exclusion chromatography columns for antibody purification are governed by advection-dispersion-reaction (ADR) PDEs coupled with adsorption isotherm equations (steric mass action, SMA). GPU-accelerated PDE solvers (finite-volume or spectral methods) simulate full column dynamics in seconds per run, enabling in silico process characterization (DoE) across 100s of loading, wash, and elution conditions in parallel. Inverse problem fitting of SMA parameters from batch isotherm experiments uses GPU-accelerated Bayesian optimization. The bottleneck is the large stiff ODE system for multi-component competitive adsorption.
- **Key algorithms:** Advection-dispersion-reaction PDE (Godunov scheme / WENO), steric mass action (SMA) isotherm model, general rate model (GRM), shrinking core diffusion model, Bayesian optimization for process development, GPU-parallel Latin hypercube DoE.
- **Datasets:** CADET Benchmark Cases — chromatography simulation validation (https://github.com/modsim/CADET); USP Bioprocess Data Repository — chromatography process development records (verify URL via NIST/USP); PDB-based antibody charge maps for adsorption prediction; OpenChrom mass-spectrometry chromatography datasets (https://www.openchrom.net/).
- **Starter repos/tools:** CADET (https://github.com/modsim/CADET) — Chromatography Analysis and Design Toolkit, CPU reference; CADET-Process (https://github.com/modsim/CADET-Process) — Python optimization wrapper for CADET; GPU-ADR solvers via CUDA finite-volume (custom implementation, verify via GitHub search "GPU chromatography simulation"); PyTorch surrogate for chromatography (verify URL via Biotechnology Journal 2024).
- **CUDA libraries & GPU pattern:** CUDA finite-volume kernels for 1D PDE time-stepping (one thread per spatial grid point), cuSPARSE for implicit diffusion system, Thrust for parallel DoE condition enumeration; pattern: 200 chromatography conditions enumerated → GPU PDE solve per condition in parallel → elution profile extraction → Bayesian optimizer selects next DoE → iterate until convergence.

---

---

## 12. Analytical & Omics Data Processing

### 12.1 Mass-Spectrometry Proteomics Search 🟢 · Established
- **Deep dive:** Database peptide search correlates each observed MS/MS spectrum against thousands of theoretical peptide spectra from a protein sequence database, the most time-consuming step in proteomics. For a dataset of 100 k spectra against a human tryptic database of 1 M peptides (× 100 modifications), the search space is 10¹¹ comparisons; GPU parallelises scoring of thousands of theoretical spectra simultaneously per observed spectrum. GiCOPS (GPU-accelerated HiCOPS) achieves 1.2–5× speedup over CPU HiCOPS and >10× over older GPU tools like Tempest, using fragment-ion indexing on GPU. MSFragger uses hash-based fragment indexing on CPU but its inner scoring loop is a GPU acceleration target.
- **Key algorithms:** Fragment-ion indexing (hash/sorted lists of b/y-ions); Xcorr / HyperScore spectral dot product; fragment index mass offset search (open search); XCorr normalised cross-correlation; peptide-spectrum match (PSM) q-value estimation (Percolator); precursor mass matching and charge state deconvolution.
- **Datasets:** PRIDE / ProteomeXchange — proteomics data repository (https://www.ebi.ac.uk/pride/); PeptideAtlas — validated human peptide spectral library (https://www.peptideatlas.org/); CPTAC cancer proteomics datasets (https://proteomics.cancer.gov/); MassIVE — mass spectrometry data repository (https://massive.ucsd.edu/).
- **Starter repos/tools:** GiCOPS (https://github.com/pcdslab/gicops) — GPU HPC framework for database peptide search; MSFragger (https://github.com/Nesvilab/MSFragger) — ultra-fast hash-index search (CPU, GPU inner loop target); Tempest — CUDA spectral scoring (verify URL; legacy); OpenMS (https://github.com/OpenMS/OpenMS) — proteomics framework with GPU integration potential.
- **CUDA libraries & GPU pattern:** GPU hash tables for fragment ion indexing; batched dot-product CUDA kernels (one thread per theoretical peptide per observed spectrum); shared-memory spectral vector loading; cuFFT-based cross-correlation; multi-GPU database sharding.

---

### 12.2 Metabolomics Spectral Processing 🟡 · Active R&D
- **Deep dive:** Metabolomics LC-MS/MS produces thousands of spectra per sample that must be denoised, deconvoluted, and matched against spectral libraries (e.g., MassBank, HMDB). Key GPU-amenable steps: (1) denoising via 2D Gaussian filtering on the (m/z, retention-time) ion map, (2) spectral library matching via batched dot-product between observed and reference spectra (identical to proteomics search but with small molecule fragmentation patterns), and (3) isotope deconvolution using the Averagine model for charge-state assignment. GPU batch cross-correlation across tens of thousands of library entries per observed spectrum replaces sequential CPU loops.
- **Key algorithms:** Gaussian kernel smoothing on MS1 ion maps; isotope deconvolution via Averagine model; dot-product spectral library matching; modified cosine similarity for spectral networking (GNPS); mass-defect filtering; retention time alignment via dynamic time warping (DTW).
- **Datasets:** GNPS / MassIVE metabolomics datasets (https://gnps.ucsd.edu/); HMDB — Human Metabolome Database spectral library (https://hmdb.ca/); MetaboLights — metabolomics studies repository (https://www.ebi.ac.uk/metabolights/); MassBank of North America — MS/MS spectral library (https://mona.fiehnlab.ucdavis.edu/).
- **Starter repos/tools:** GNPS (https://gnps.ucsd.edu/) — spectral networking platform (GPU matching target); MZmine3 (https://github.com/mzmine/mzmine3) — open-source LC-MS processing (GPU acceleration integration target); SIRIUS (https://github.com/boecker-lab/sirius) — molecular formula / structure prediction; OpenMS (https://github.com/OpenMS/OpenMS) — LC-MS processing suite.
- **CUDA libraries & GPU pattern:** cuFFT for cross-correlation in spectral library matching; custom 2D Gaussian smoothing CUDA kernels on ion maps; thrust for m/z sorted spectral vector operations; batched cosine similarity via cuBLAS GEMM (spectra as rows of a matrix); GPU-resident library matrix for parallel dot-product.

---

### 12.3 Spatial Transcriptomics Analysis 🟡 · Active R&D
- **Deep dive:** Spatial transcriptomics (10x Visium, MERFISH, Xenium) measures gene expression at spatially defined locations (thousands of spots or millions of FISH-resolved single cells), producing large dense expression × spatial matrices. GPU acceleration applies to: (1) image-based spot detection and signal decoding for MERFISH (GPU-accelerated FISH barcode decoding), (2) dimension reduction and clustering (GPU UMAP / Leiden), and (3) spatial autocorrelation statistics (Moran's I computed as a sparse matrix-vector product over spatial neighbours). A 2025 biorxiv preprint describes GPU-accelerated 3D multiplexed iterative RNA-FISH decoding, and rctd-py delivers 9–41× GPU speedup for cell-type deconvolution of Visium HD (~400 k spots).
- **Key algorithms:** FISH barcode decoding (minimum-Hamming-distance matching, GPU parallel); spatial KNN graph construction; Moran's I spatial autocorrelation (sparse MVM); NMF/NNLS for deconvolution; SpatialDE spatially variable gene regression; GPU UMAP for spatial embedding.
- **Datasets:** 10x Genomics public spatial datasets — Visium/VisiumHD human tissue (https://www.10xgenomics.com/resources/datasets); Allen Brain Cell Atlas — spatial transcriptomics of whole mouse brain (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); 4DN spatial data portal (https://data.4dnucleome.org/); MERSCOPE (Vizgen) public datasets (https://vizgen.com/data-release-program/).
- **Starter repos/tools:** rctd-py (https://github.com/p-gueguen/rctd-py) — GPU-accelerated RCTD deconvolution, 9–41× speedup; rapids-singlecell + Squidpy integration (https://github.com/scverse/rapids_singlecell) — GPU spatial analysis; GPU-accelerated RNA-FISH decoding (https://www.biorxiv.org/content/10.1101/2025.10.10.681751.full.pdf) — 3D FISH GPU processing; Squidpy (https://github.com/scverse/squidpy) — spatial omics analysis toolkit (GPU extension via rapids-singlecell).
- **CUDA libraries & GPU pattern:** cuML UMAP / KNN for spatial graphs; cuSPARSE for spatial autocorrelation (Moran's I sparse MVM); cuDNN for FISH image decoding CNN; batched minimum-Hamming-distance kernels for MERFISH barcode matching; GPU tensor for dense spot × gene expression matrix.

---

### 12.4 Cryo-EM / ET Image Preprocessing 🟢 · Established
- **Deep dive:** Cryo-electron microscopy produces thousands of noisy micrographs (4k×4k pixels) that must be motion-corrected, CTF-estimated, particle-picked, and 2D/3D classified before structure determination. CryoSPARC and RELION both natively use CUDA for all major processing steps: motion correction via cross-correlation in Fourier space (cuFFT), CTF estimation via Thon ring fitting on GPU, particle picking via neural network (Topaz, crYOLO), and 3D refinement via GPU-accelerated back-projection and real-space expectation-maximisation. A single H100 processes hundreds of micrographs per minute end-to-end, enabling real-time feedback during cryo-EM sessions.
- **Key algorithms:** Fourier-space cross-correlation for frame alignment (MotionCor2); CTF fitting via Thon ring power spectrum (CTFFIND); 2D class averaging (RELION E-M); 3D gold-standard FSC refinement; CNN particle picking (Topaz); back-projection 3D reconstruction; Wiener filter CTF correction.
- **Datasets:** EMDB — Electron Microscopy Data Bank, raw micrographs and maps (https://www.ebi.ac.uk/emdb/); EMPIAR — raw cryo-EM micrograph repository (https://www.ebi.ac.uk/empiar/); wwPDB cryo-EM entries (https://www.rcsb.org/); CryoSPARC demo datasets (https://cryosparc.com/download).
- **Starter repos/tools:** CryoSPARC (https://cryosparc.com/) — fully GPU-native cryo-EM pipeline, particle picking through 3D refinement; RELION4 (https://github.com/3dem/relion) — GPU-accelerated 3D classification and refinement; Topaz (https://github.com/tbepler/topaz) — GPU CNN particle picker; MotionCor2 (verify URL — Zheng lab UCSF) — GPU frame alignment.
- **CUDA libraries & GPU pattern:** cuFFT for Fourier-domain frame alignment and CTF power spectrum; cuDNN for CNN particle picking; custom back-projection CUDA kernels; atomic operations for back-projection accumulation; multi-GPU 3D refinement with gradient averaging.

---

### 12.5 Real-Time Sequencing Analysis / Adaptive Sampling 🟡 · Active R&D
- **Deep dive:** Oxford Nanopore adaptive sampling (ReadUntil API) allows the sequencer to reject reads in real time (within 200 ms per read) based on a computational decision—requiring GPU basecalling and alignment to complete in under ~100 ms per read chunk. The pipeline: raw signal → GPU basecalling (Dorado, HAC model) → GPU seed-extension to reference → accept/reject decision → signal to sequencer. GPU processing is not optional; CPU pipelines are too slow for the 200 ms window. This enables on-target enrichment without library preparation: unwanted chromosomal regions are skipped by reversing the voltage to eject the DNA strand.
- **Key algorithms:** GPU CTC basecalling (Dorado transformer); approximate hash seed alignment (minimap2 GPU); streaming input buffer management; read-until decision tree; pore blocking prediction; real-time sequence classification (pathogen typing).
- **Datasets:** ONT open datasets with ReadUntil metadata (https://github.com/GoekeLab/awesome-nanopore); NCBI SRA real-time sequencing runs (https://www.ncbi.nlm.nih.gov/sra); ENA clinical nanopore studies (https://www.ebi.ac.uk/ena); Oxford Nanopore public data portal (https://labs.epi2me.io/dataindex/).
- **Starter repos/tools:** Dorado (https://github.com/nanoporetech/dorado) — GPU basecaller with low-latency streaming mode; ReadFish (https://github.com/looselab/readfish) — ReadUntil adaptive sampling controller; Icarust (https://github.com/LooseLab/Icarust) — real-time nanopore simulator for pipeline testing; MinKNOW (ONT proprietary) — sequencer control with GPU basecalling integration.
- **CUDA libraries & GPU pattern:** TensorRT for ultra-low-latency RNN inference; CUDA streams for overlapping signal decode and alignment; persistent GPU kernel for continuous signal ingestion; GPU ring buffer for streaming POD5 signal; multi-GPU for PromethION multi-flow-cell setups.

---

### 12.6 Microbiome & Antimicrobial-Resistance Analytics 🟡 · Active R&D
- **Deep dive:** Microbiome profiling from shotgun metagenomics combines taxonomic classification (GPU Kraken2 / MetaCache) with functional annotation (GPU DIAMOND / MMseqs2 vs. CARD/ResFinder for AMR genes) and community ecology statistics. The AMR gene identification step—aligning millions of reads against thousands of resistance gene models (RGI uses DIAMOND + CARD)—is the most GPU-amenable component. Deep learning models (MSDeepAMR, DeepARG) trained on genomic features or mass spectrometry (MALDI-TOF) patterns predict resistance phenotypes and are accelerated by GPU inference. Metagenome-assembled genome (MAG) binning via deep learning (DAS_Tool) is also GPU-amenable.
- **Key algorithms:** K-mer-based taxonomic classification (Kraken2/MetaCache); protein homology search vs. AMR databases (DIAMOND/CARD); profile HMM search for resistance gene families; MALDI-TOF spectral CNN for phenotypic AMR prediction; random forest / gradient boosting for AMR genotype-to-phenotype; deep learning MAG binning.
- **Datasets:** CARD — Comprehensive Antibiotic Resistance Database (https://card.mcmaster.ca/); PATRIC / BV-BRC — bacterial pathogen genomes (https://www.bv-brc.org/); CAMDA AMR challenge datasets (http://www.camda.info/); HMP2 (Human Microbiome Project Phase 2) (https://www.hmpdacc.org/).
- **Starter repos/tools:** MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU metagenomic classifier; DIAMOND (https://github.com/bbuchfink/diamond) — GPU-targetable protein aligner for AMR annotation; DeepARG (https://github.com/gaarangoa/deeparg) — deep learning AMR gene predictor (GPU inference); RGI (https://github.com/arpcard/rgi) — Resistance Gene Identifier using CARD database.
- **CUDA libraries & GPU pattern:** GPU hash tables for k-mer AMR classification; batched cuDNN CNN inference for MALDI spectral AMR prediction; cuBLAS for alignment scoring matrix; thrust for read partition by taxon; RAPIDS cuDF for large microbiome count matrix operations.

---

### 12.7 DIA Proteomics Spectral Deconvolution 🟡 · Active R&D
- **Deep dive:** Data-Independent Acquisition (DIA) proteomics (Spectronaut, DIA-NN, FragPipe-DIA) co-isolates and co-fragments all precursors in wide isolation windows, requiring deconvolution of chimeric MS2 spectra containing overlapping fragment ion series. The GPU bottleneck is the inner-loop scoring: for each DIA window, thousands of peptide fragment ion templates must be correlated with the observed chromatographic fragment traces (XIC), a batched sliding-window cross-correlation problem. DIA-BERT (2025) is a GPU-enabled transformer approach treating DIA spectrum sequences analogously to language tokens, enabling improved feature extraction with GPU inference.
- **Key algorithms:** Extracted ion chromatogram (XIC) correlation scoring; deconvolution of chimeric spectra via library matching; Gaussian smoothing of chromatographic peaks; semi-empirical spectral library generation; transformer-based DIA spectrum encoding (DIA-BERT); target-decoy FDR estimation.
- **Datasets:** PRIDE ProteomeXchange DIA datasets (https://www.ebi.ac.uk/pride/); CPTAC DIA cancer proteomics (https://proteomics.cancer.gov/); Proteome profiler benchmark DIA datasets (verify URL); DIA-NN benchmark datasets (https://github.com/vdemichev/DiaNN).
- **Starter repos/tools:** DIA-NN (https://github.com/vdemichev/DiaNN) — fast DIA software (GPU inner-loop target); FragPipe (https://github.com/Nesvilab/FragPipe) — MSFragger-based DIA pipeline; DIA-BERT (https://proteomicsnews.blogspot.com/2025/05/dia-bert-gpu-enabled-dia-analysis.html) — GPU transformer for DIA; Spectronaut (commercial, Biognosys) — industry DIA software.
- **CUDA libraries & GPU pattern:** cuFFT cross-correlation for XIC fragment trace matching; cuDNN transformer for DIA-BERT; batched sliding-window scoring kernels; GPU tensor for precursor×fragment scoring matrix; thrust for peak apex detection; multi-GPU for large clinical DIA cohorts.

---

### 12.8 Isotope Pattern Matching & Charge Deconvolution 🟡 · Active R&D
- **Deep dive:** High-resolution mass spectrometry resolves isotope envelopes (the pattern of ¹²C, ¹³C, ²H, ¹⁸O peaks) that report the charge state and monoisotopic mass of each peptide or metabolite. Matching observed isotope patterns against theoretical Averagine distributions (or exact elemental isotope calculations via IsoSpec) across millions of features per LC-MS run is a quadratic search problem. GPU parallelism assigns one thread per candidate mass window, computing the dot product between observed and theoretical isotope patterns simultaneously across thousands of charge states and masses, replacing the sequential CPU sweep.
- **Key algorithms:** Averagine model for average elemental composition; Mercury / IsoSpec exact isotope pattern calculation via Poisson convolution; dot-product / cosine-similarity matching of isotope envelopes; Maximum Likelihood charge state assignment; THRASH deconvolution algorithm; Wavelet transform for isotope detection (IsotopeWavelet).
- **Datasets:** PRIDE ProteomeXchange high-resolution datasets (https://www.ebi.ac.uk/pride/); HMDB high-resolution metabolomics spectra (https://hmdb.ca/); MassBank (https://massbank.eu/); CPTAC iTRAQ/TMT quantitative proteomics (https://proteomics.cancer.gov/).
- **Starter repos/tools:** OpenMS (https://github.com/OpenMS/OpenMS) — comprehensive LC-MS toolkit with GPU integration hooks; IsoSpec (https://github.com/MatteoLacki/IsoSpec) — exact isotope pattern computation; Xtract (Thermo Fisher, proprietary) — charge deconvolution; pyOpenMS (https://github.com/OpenMS/OpenMS) — Python bindings for proteomics.
- **CUDA libraries & GPU pattern:** Batched dot-product CUDA kernels (one warp per candidate m/z window); cuFFT for wavelet-based isotope detection; shared-memory Averagine lookup tables; thrust for peak list sorting and deduplication; cuBLAS GEMM for charge-state × m/z scoring matrix.

---

### 12.9 GPU UMAP / t-SNE for Single-Cell Omics 🟢 · Established
- **Deep dive:** UMAP and t-SNE dimensionality reduction are the universal visualisation steps in single-cell omics (scRNA-seq, scATAC-seq, CyTOF, CITE-seq). For a million-cell dataset, standard CPU UMAP takes hours; GPU UMAP (cuML) and GPU t-SNE (RAPIDS) reduce this to minutes by parallelising the KNN graph construction (Faiss-GPU approximate nearest neighbours) and the repulsive/attractive force optimisation (each cell's gradient update is independent given the current embedding). The NVIDIA blog demonstrates GPU UMAP on 1.3 M cells processing in ~1 minute vs. ~40 minutes CPU.
- **Key algorithms:** Exact/approximate KNN (Faiss IVF-PQ, HNSWLIB-GPU); fuzzy simplicial set construction (UMAP); stochastic gradient descent with negative sampling (UMAP layout); t-SNE Barnes-Hut or FIt-SNE approximation; PCA for pre-reduction; Leiden/Louvain graph clustering.
- **Datasets:** Human Cell Atlas 10x datasets (https://www.humancellatlas.org/); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); 10x Genomics 1.3 M mouse brain dataset (https://www.10xgenomics.com/resources/datasets); NCBI GEO scRNA-seq compendium (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU UMAP/Leiden/PCA for scRNA-seq; cuML (https://github.com/rapidsai/cuml) — GPU UMAP and t-SNE via RAPIDS; Faiss (https://github.com/facebookresearch/faiss) — GPU KNN for UMAP graph construction; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmarked notebooks.
- **CUDA libraries & GPU pattern:** cuML UMAP (GPU KNN + SGD layout); Faiss-GPU IVF-PQ approximate nearest neighbours; cuGraph for Leiden clustering; CUB warp-level reduction for gradient accumulation; atomic updates for asynchronous UMAP layout; multi-GPU via Dask for >10 M cells.

---

### 12.10 Cell-Type Annotation in Single-Cell Studies 🟡 · Active R&D
- **Deep dive:** Cell-type annotation assigns biological identity to each sequenced cell by comparing its gene expression profile against reference atlases or marker gene signatures. GPU acceleration applies to: (1) nearest-centroid or KNN classification in high-dimensional gene space (GPU KNN via Faiss or cuML), (2) label transfer via GPU matrix multiplication (Seurat/Harmony), and (3) foundation model inference (scGPT, Geneformer, CellMaster) that takes tokenised gene expression as input to a transformer, with GPU inference on batches of cells. scGPT (2024) fine-tuned on 33 M cells demonstrates that GPU-accelerated transformer inference at cell-type annotation is now at least as accurate as classical marker-based methods.
- **Key algorithms:** KNN label transfer in PCA-reduced gene space; Seurat anchor-based integration (CCA); marker-gene enrichment scoring (GSEA); transformer token attention over expressed genes (scGPT, Geneformer); logistic regression classifiers; hierarchical label propagation.
- **Datasets:** Human Cell Atlas (https://www.humancellatlas.org/); CellxGene Census (https://cellxgene.cziscience.com/); Azimuth reference atlases — curated cell-type references (https://azimuth.hubmapconsortium.org/); PanglaoDB — marker gene database (https://panglaodb.se/).
- **Starter repos/tools:** scGPT (https://github.com/bowang-lab/scGPT) — single-cell foundation model, GPU transformer inference; Geneformer (https://huggingface.co/ctheodoris/Geneformer) — transformer pre-trained on 30 M cells; rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU KNN label transfer; CellMaster (https://arxiv.org/pdf/2602.13346) — collaborative annotation with LLM reasoning.
- **CUDA libraries & GPU pattern:** cuML KNN for label transfer; cuDNN transformer inference for scGPT / Geneformer; batched tokenised cell embedding via GPU; Faiss-GPU for reference atlas similarity search; multi-GPU gradient accumulation for foundation model fine-tuning.

---

### 12.11 Trajectory Inference & Pseudotime Analysis 🟡 · Active R&D
- **Deep dive:** Trajectory inference reconstructs continuous developmental processes from snapshot scRNA-seq data by ordering cells along a pseudotime axis representing biological progression (differentiation, cell cycle, immune activation). Algorithms range from principal curve fitting (Monocle3) to diffusion-map graph-based approaches (Scanpy PAGA) and optimal transport (Waddington-OT). GPU acceleration targets the KNN graph construction (the shared first step), the diffusion map eigensolver (cuSolver), and the optimal transport (Sinkhorn algorithm) computation. For atlas-scale data (>1 M cells), GPU trajectory inference with RAPIDS reduces hours to minutes.
- **Key algorithms:** Principal curve / elastic principal graph (DDRTree); diffusion pseudotime (DPT) via diffusion map eigenvectors; PAGA graph abstraction; RNA velocity (scVelo) splicing dynamics EM; Sinkhorn optimal transport for fate probability; graph-based geodesic distances for branch assignment.
- **Datasets:** Human Cell Atlas developmental atlases (https://www.humancellatlas.org/); GEO scRNA-seq differentiation time-course datasets (https://www.ncbi.nlm.nih.gov/geo/); Allen Brain Cell Atlas (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); ENCODE iPSC differentiation scRNA-seq (https://www.encodeproject.org/).
- **Starter repos/tools:** rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU diffusion pseudotime and UMAP; Scanpy with GPU backend (https://github.com/scverse/scanpy) — PAGA trajectory analysis; scVelo (https://github.com/theislab/scvelo) — RNA velocity (GPU EM target); Monocle3 (https://github.com/cole-trapnell-lab/monocle3) — principal graph trajectory inference.
- **CUDA libraries & GPU pattern:** cuSolver eigenvalue solver for diffusion map; cuSPARSE for KNN graph Laplacian operations; cuML for PCA pre-reduction; custom Sinkhorn CUDA kernels (iterative row/column normalisation); GPU optimal transport via POT library with CUDA backend.

---

### 12.12 Spatial Deconvolution of Cell Types 🟡 · Active R&D
- **Deep dive:** Spatial transcriptomics spots (Visium: 55 µm, ~10 cells/spot; Visium HD: 8 µm, ~1 cell) contain mixed gene expression signals from multiple cell types; deconvolution estimates cell-type proportions per spot using a scRNA-seq reference. RCTD (Robust Cell-Type Decomposition) fits a Poisson regression per spot independently—embarrassingly parallel—enabling GPU acceleration. rctd-py achieves 9–14× GPU speedup in doublet mode and 41× in multi-cell mode on VisiumHD (~400 k spots processed in ~1 minute on a Blackwell GPU). Cell2Location uses a hierarchical Bayesian model (pyro/PyTorch) with GPU MCMC; Tangram uses optimal transport GPU acceleration.
- **Key algorithms:** Poisson regression per spot (RCTD); negative binomial regression (Cell2Location); optimal transport spot-to-reference matching (Tangram); NMF for reference-free deconvolution; stereoscope GLM; spot clustering with GPU Leiden.
- **Datasets:** 10x Genomics Visium Human Tissue datasets (https://www.10xgenomics.com/resources/datasets); Allen Brain Cell Atlas spatial data (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); MERSCOPE public data (https://vizgen.com/data-release-program/); 4DN spatial data (https://data.4dnucleome.org/).
- **Starter repos/tools:** rctd-py (https://github.com/p-gueguen/rctd-py) — GPU-accelerated RCTD, PyTorch backend; Cell2Location (https://github.com/BayraktarLab/cell2location) — hierarchical Bayesian GPU deconvolution; Tangram (https://github.com/broadinstitute/Tangram) — OT-based GPU spatial mapping; Squidpy (https://github.com/scverse/squidpy) — spatial analysis toolkit with rapids-singlecell integration.
- **CUDA libraries & GPU pattern:** Batched Poisson regression CUDA kernels (one CUDA block per spot); PyTorch CUDA for Bayesian MCMC; GPU optimal transport (POT/GeomLoss); cuML for reference PCA; multi-GPU Dask for VisiumHD-scale spot counts.

---

### 12.13 Real-Time Pathogen Identification (Clinical) 🟡 · Active R&D
- **Deep dive:** Clinical metagenomic next-generation sequencing (mNGS) for pathogen identification requires processing millions of reads within 1–2 hours of sample collection to guide antibiotic therapy. The critical path is: GPU basecalling (Dorado) → GPU k-mer classification (MetaCache/GPU Kraken2) → GPU AMR gene annotation (DIAMOND vs. CARD) → statistical confidence scoring. A 2024 MDPI paper describes a GPU-integrated nanopore workstation running CUDA-accelerated basecalling and classification in real time, enabling same-day bloodstream infection pathogen identification. GPU parallelism is the enabling technology for clinical mNGS turnaround within therapeutic decision windows.
- **Key algorithms:** GPU CTC basecalling; GPU k-mer LCA classification; minimap2 GPU alignment to pathogen reference panel; GPU AMR gene DIAMOND search; Bayesian abundance estimation (Bracken); clinical decision threshold scoring; antimicrobial susceptibility genotype prediction.
- **Datasets:** NCBI RefSeq pathogen reference sequences (https://ftp.ncbi.nlm.nih.gov/refseq/); CARD AMR database (https://card.mcmaster.ca/); IDseq / Chan Zuckerberg clinical mNGS data (https://czid.org/); NCBI Pathogen Detection (https://www.ncbi.nlm.nih.gov/pathogens/).
- **Starter repos/tools:** Dorado (https://github.com/nanoporetech/dorado) — GPU basecaller; MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU real-time classification; DIAMOND (https://github.com/bbuchfink/diamond) — fast AMR gene annotation; CZID/IDseq (https://github.com/chanzuckerberg/czid-workflows) — cloud mNGS pipeline.
- **CUDA libraries & GPU pattern:** TensorRT for low-latency basecalling; GPU hash tables for k-mer classification; CUDA streams for read-by-read pipeline; cuBLAS for alignment score matrices; real-time CUDA ring buffer for streaming POD5 signal; multi-GPU pipelining of basecall → classify → annotate.

---

### 12.14 Peptide De Novo Sequencing 🟡 · Active R&D
- **Deep dive:** De novo peptide sequencing infers amino acid sequences directly from MS/MS spectra without a protein database, critical for non-model organisms, immunopeptidomics, and modified peptides. Algorithms generate candidate sequences by traversing a spectrum graph (nodes = fragment ions, edges = amino acid mass differences) via beam search or dynamic programming. GPU acceleration applies to: (1) GPU-parallel beam search over thousands of candidate sequences simultaneously, (2) batched transformer/LSTM scoring of candidate sequences, and (3) the CUDA-accelerated knapsack DP ensuring precursor mass consistency. NovoBench (NeurIPS 2024) benchmarks GPU-accelerated deep learning de novo sequencers.
- **Key algorithms:** Spectrum graph construction (b/y-ion nodes); beam-search decoding with GPU-parallel branches; CUDA knapsack DP for precursor mass constraint; seq2seq transformer (Casanovo, PointNovo); bidirectional LSTM encoder; attention over fragment ion sequence; PTM-tolerant open search.
- **Datasets:** PRIDE ProteomeXchange benchmark de novo datasets (https://www.ebi.ac.uk/pride/); NovoBench benchmark (https://github.com/jingbo02/NovoBench) — standardised deep learning de novo benchmark; MassIVE (https://massive.ucsd.edu/); PeptideAtlas synthetic peptide datasets (https://www.peptideatlas.org/).
- **Starter repos/tools:** Casanovo (https://github.com/Noble-Lab/casanovo) — transformer-based GPU de novo sequencer; NovoBench (https://github.com/jingbo02/NovoBench) — NeurIPS 2024 benchmark suite; PointNovo (verify URL, from Ma et al.) — deep learning de novo with GPU inference; DeepNovo (https://github.com/nh2tran/DeepNovo) — original LSTM-based GPU de novo sequencer.
- **CUDA libraries & GPU pattern:** cuDNN transformer/LSTM inference; CUDA knapsack DP (shared-memory DP table per spectrum); batched beam search with GPU-parallel candidate scoring; Tensor Core BF16 for transformer scoring; one CUDA stream per spectrum batch.

---

### 12.15 Codon Usage & Synonymous Evolution Analysis 🟢 · Established
- **Deep dive:** Codon usage analysis computes codon adaptation index (CAI), relative synonymous codon usage (RSCU), and dN/dS (non-synonymous to synonymous substitution ratio) across thousands of gene alignments, often in population genomics or viral evolution studies. dN/dS computation requires pairwise codon alignment followed by branch-model likelihood evaluation per codon triplet—a compute-intensive phylogenetic likelihood calculation. For genome-scale dN/dS scans (10⁶ gene pairs), GPU parallelism assigns one CUDA thread per gene pair, with codon frequency tables in shared memory. Combined with phylogenetic likelihood (Section 3.9 BeagleLib), GPU codon models enable population-scale selection scans.
- **Key algorithms:** Codon substitution model (Goldman-Yang GY94, MG94); dN/dS branch-site model likelihood; RSCU and CAI calculation; synonymous site rate estimation; Fisher's exact test for codon usage bias; maximum likelihood codon tree.
- **Datasets:** Ensembl CDS sequences — comparative codon data across species (https://www.ensembl.org/); NCBI RefSeq CDS archives (https://ftp.ncbi.nlm.nih.gov/refseq/); GISAID SARS-CoV-2 genomes — viral codon evolution dataset (https://www.gisaid.org/); PopHuman dN/dS datasets (verify URL).
- **Starter repos/tools:** BeagleLib (https://github.com/beagle-dev/beagle-lib) — GPU codon model likelihood evaluation; HyPhy (https://github.com/veg/hyphy) — GPU-capable dN/dS and selection analysis framework; PAML (http://abacus.gene.ucl.ac.uk/software/paml.html) — CPU dN/dS reference; BEAST2 (https://github.com/CompEvol/beast2) — Bayesian molecular evolution using BeagleLib GPU.
- **CUDA libraries & GPU pattern:** BeagleLib CUDA kernels for 60×60 codon matrix-vector products; one CUDA thread per alignment column per codon model; cuBLAS for codon substitution matrix exponentiation; GPU-resident codon frequency tables; multi-GPU tree partitioning.

---

### 12.16 GPU-Accelerated Hi-C Contact Loop Calling 🟡 · Active R&D
- **Deep dive:** Hi-C loop calling (HiCCUPS) identifies chromatin loops as enriched point interactions above a 2D background, estimated by a sliding Donut kernel convolution over the contact map. At 5 kb resolution, a human contact map is ~600 k × 600 k (sparse, stored as pairs); the Donut convolution at each potential loop pixel is a GPU embarrassingly parallel 2D operation. NVIDIA's original HiCCUPS paper used a GPU implementation as the default, making this one of the earliest established GPU genomics tools. Recent deep-learning loop callers (Peakachu) apply CNNs to local contact map patches, each patch independently inferrable on GPU.
- **Key algorithms:** Donut kernel background estimation (2D convolution on sparse contact map); Poisson enrichment scoring per pixel; multi-resolution peak merging; FDR control for loop calls; Peakachu CNN local patch classification; Fit-Hi-C probability model; anchor pair exhaustive scoring.
- **Datasets:** 4DN Hi-C datasets (https://data.4dnucleome.org/); GEO GSE63525 (Rao 2014) — original HiCCUPS benchmark (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525); ENCODE Hi-C (https://www.encodeproject.org/); 3D Genome Browser datasets (http://3dgenome.fsm.northwestern.edu/).
- **Starter repos/tools:** Juicer / HiCCUPS (https://github.com/aidenlab/juicer) — GPU loop caller, original CUDA implementation; Peakachu (https://github.com/tariks/peakachu) — CNN-based loop caller (GPU inference); Higashi (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU model; MUSTACHE (https://github.com/ay-lab/mustache) — multi-scale Hi-C loop caller.
- **CUDA libraries & GPU pattern:** Custom 2D convolution kernels for Donut background; cuSPARSE for sparse contact matrix operations; cuDNN for CNN local-patch loop classification; thrust for sparse pixel sorting; GPU-resident contact map tiles in texture memory.

---

### 12.17 Metagenome-Assembled Genome (MAG) Binning 🟡 · Active R&D
- **Deep dive:** MAG binning clusters assembled contigs into genome bins representing distinct microbial species, using tetranucleotide frequency (TNF, a 256-dimensional feature vector per contig) and coverage across samples. The binning problem is a clustering problem in 256+N_sample dimensional space; GPU UMAP + GPU clustering (Leiden) of millions of contigs from complex soil or gut metagenomes reduces hours-long CPU pipelines to minutes. Deep learning binners (CONCOCT, SemiBin2) use variational autoencoders or self-supervised contrastive learning whose training and inference are GPU-native.
- **Key algorithms:** Tetranucleotide frequency (TNF) 256-dim feature extraction; GPU UMAP dimensionality reduction of contig TNF+coverage; Leiden clustering of contig UMAP graph; variational autoencoder (CONCOCT style) for contiguous-binning; contrastive learning (SemiBin2); checkM completeness/contamination scoring.
- **Datasets:** CAMI metagenome benchmarks (https://data.cami-challenge.org/); HMP2 gut metagenomes (https://www.hmpdacc.org/); JGI IMG/M — environmental metagenomes (https://img.jgi.doe.gov/); MGnify metagenome assemblies (https://www.ebi.ac.uk/metagenomics/).
- **Starter repos/tools:** SemiBin2 (https://github.com/BigDataBiology/SemiBin) — self-supervised contrastive learning binner (GPU-trainable); CONCOCT (https://github.com/BinPro/CONCOCT) — Gaussian mixture model binner; Vamb (https://github.com/RasmussenLab/vamb) — variational autoencoder MAG binner with GPU training; rapids-singlecell UMAP (https://github.com/scverse/rapids_singlecell) — GPU UMAP for contig embedding.
- **CUDA libraries & GPU pattern:** cuML UMAP for TNF+coverage contig embedding; cuGraph Leiden for contig clustering; cuDNN for VAE encoder/decoder training; cuDF for contig feature matrix; one CUDA thread per contig coverage computation; multi-GPU gradient reduction for VAE training.

---

**Sources used for verification:**
- [CUDASW++4.0 — BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-024-05965-6)
- [CUDASW4 GitHub](https://github.com/asbschmidt/CUDASW4)
- [MMseqs2-GPU — Nature Methods](https://www.nature.com/articles/s41592-025-02819-8)
- [MMseqs2 GitHub](https://github.com/soedinglab/MMseqs2)
- [NVIDIA Parabricks Documentation](https://docs.nvidia.com/clara/parabricks/latest/)
- [Dorado GitHub](https://github.com/nanoporetech/dorado)
- [f5c GitHub](https://github.com/hasindu2008/f5c)
- [Remora GitHub](https://github.com/nanoporetech/remora)
- [GenomeWorks GitHub](https://github.com/NVIDIA-Genomics-Research/GenomeWorks)
- [racon-GPU GitHub](https://github.com/NVIDIA-Genomics-Research/racon-gpu)
- [rapids-singlecell GitHub](https://github.com/scverse/rapids_singlecell)
- [RAPIDS single-cell examples GitHub](https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples)
- [ScaleSC — Bioinformatics Advances](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/)
- [rctd-py GitHub](https://github.com/p-gueguen/rctd-py)
- [Rapid GPU pangenome layout — SC 2024](https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf)
- [PGGB GitHub](https://github.com/pangenome/pggb)
- [CARE GitHub](https://github.com/fkallen/CARE)
- [MetaCache-GPU preprint](https://arxiv.org/pdf/2106.08150)
- [GiCOPS GPU proteomics](https://www.nature.com/articles/s41598-023-43033-w)
- [NovoBench GitHub](https://github.com/jingbo02/NovoBench)
- [Cas-OFFinder GitHub](https://github.com/snugel/cas-offinder)
- [fair-esm GitHub](https://github.com/facebookresearch/esm)
- [Juicer/HiCCUPS GitHub](https://github.com/aidenlab/juicer)
- [DIA-BERT GPU DIA analysis](https://proteomicsnews.blogspot.com/2025/05/dia-bert-gpu-enabled-dia-analysis.html)
- [Darwin GPU overlap paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7495891/)
- [CUDAMPF HMMER GPU — BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4)
- [CUDA-MEME / mCUDA-MEME](https://cuda-meme.sourceforge.io/homepage.htm)
- [GPU GWAS-Flow preprint](https://www.biorxiv.org/content/10.1101/783100)
- [GPU-GWAS GitHub](https://github.com/STRIDES-Codes/GPU-GWAS)
- [GPU-accelerated methylation — Bioinformatics Advances](https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac088/6855011)
- [GPU RNA-FISH decoding preprint](https://www.biorxiv.org/content/10.1101/2025.10.10.681751.full.pdf)

---

## 13. Pharmacology & Clinical Quantitative Modeling

### 13.1 Population PK/PD (Nonlinear Mixed-Effects) Fitting 🟢 · Established

- **Deep dive:** Fits nonlinear mixed-effects (NLME) models to sparse individual PK/PD data from clinical trials to characterise population mean parameters (fixed effects) and between-subject variability (random effects). The computational bottleneck is the Monte Carlo EM inner loop: at each iteration, thousands of individual ODE trajectories must be integrated (one per subject per Monte Carlo sample) to compute the expected log-likelihood. GPU parallelism across subjects × MC samples provides the key speedup. A hybrid GPU-CPU implementation of parallelised MCEM for PK models (ResearchGate, 2013) demonstrated early feasibility; modern implementations using CUDA-batched RK4 ODE solvers are now standard in Pumas and experimental NONMEM backends. Each ODE evaluation for a two-compartment PK model takes ~microseconds, but millions of evaluations per EM iteration require GPU throughput.
- **Key algorithms:** FOCE (First-Order Conditional Estimation), SAEM (Stochastic Approximation EM), Laplacian approximation, Quasi-Newton BFGS optimisation, Importance Sampling for individual Bayesian estimation, Extended Kalman Filter (EKF) for continuous-time models, inter-individual variability (IIV) via log-normal random effects, correlation structures via OMEGA matrices.
- **Datasets:**
  - NONMEM Example Dataset archive — shipped with NONMEM for benchmark (verify URL via ICON plc)
  - PharmPK listserv dataset collections — published population PK datasets (verify URL)
  - CDISC SDTM/ADaM clinical trial datasets — standardised PK trial formats (https://www.cdisc.org/)
  - Warfarin PK/PD open dataset — widely used for mixed-effects benchmark (verify URL)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — Julia-based population PK/PD with GPU acceleration via CUDA.jl
  - Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan extensions for PK/PD ODE solving with GPU potential
  - Monolix (https://lixoft.com/products/monolix/) — commercial SAEM-based NLME (verify GPU backend availability)
  - nlmixr2 (https://github.com/nlmixr2/nlmixr2) — open-source R NLME fitting with SAEM and FOCE
- **CUDA libraries & GPU pattern:** Custom CUDA RK4/RK45 batched ODE kernels, cuBLAS for OMEGA matrix operations, cuRAND for SAEM stochastic approximation draws; pattern: one CUDA thread block per subject, with inner MC samples parallelised within the block.

---

### 13.2 PBPK at Scale 🟡 · Active R&D

- **Deep dive:** Physiologically based pharmacokinetic (PBPK) models describe drug disposition through ~15 interconnected physiological compartments (blood, liver, kidney, lung, fat, muscle, etc.), each defined by ODEs parameterised by tissue volumes, blood flows, and metabolic rate constants. High-throughput virtual screening of thousands of compounds requires solving the full PBPK ODE system (30–60 ODEs) for each compound simultaneously — a batch of 10,000 compounds is 600,000 simultaneous ODEs, well-suited to GPU-parallel Runge-Kutta integration. NVIDIA's nvQSP implements a GPU-accelerated RODAS4 stiff ODE solver specifically for QSP/PBPK population studies. Monte Carlo virtual population simulations (500–5000 virtual subjects per compound) further multiply the parallelism requirement.
- **Key algorithms:** RODAS4 stiff ODE solver (GPU implementation), Runge-Kutta 4/5, adaptive stepsize control, PBPK parameter estimation via Bayesian MCMC, machine-learning-predicted ADME inputs (logP, Vd, CLint), tissue-plasma partition coefficient estimation (Rodgers-Rowland, Berezhkovskiy).
- **Datasets:**
  - Open Systems Pharmacology PBPK model repository (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library) — 100+ validated human PBPK models
  - DrugBank ADME data — 14k+ drugs with physicochemical and metabolic parameters (https://www.drugbank.com/)
  - FDA/EMA drug approval submission PK data — publicly available pharmacokinetic data from drug labels (verify URL)
  - ChEMBL ADMET data — assay-based ADME measurements (https://www.ebi.ac.uk/chembl/)
- **Starter repos/tools:**
  - PK-Sim (https://github.com/Open-Systems-Pharmacology/PK-Sim) — open-source whole-body PBPK software (C#; GPU via OSP Suite)
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — NVIDIA GPU-accelerated QSP/PBPK ODE solvers (CUDA)
  - SimBiology (MATLAB) — PBPK modelling with parallel computing toolbox for GPU (verify URL)
  - PBPKsim (verify URL) — Python PBPK simulation framework
- **CUDA libraries & GPU pattern:** Custom CUDA RODAS4/RK45 stiff ODE solver kernels, cuBLAS for Jacobian evaluation, Thrust for adaptive stepsize selection; pattern: one CUDA thread block per virtual subject, with ODE compartments mapped to shared memory.

---

### 13.3 Receptor Binding Kinetics & Occupancy 🟡 · Active R&D

- **Deep dive:** Simulates drug-receptor association, dissociation, and signalling downstream of receptor occupancy using differential equation models (two-state, ternary complex, operational models of agonism). In receptor occupancy (RO) imaging data analysis, GPU parallelism enables simultaneous fitting of PET tracer binding across thousands of brain voxels. For in silico virtual screening, GPU batch evaluation of binding kinetics models for thousands of drug candidates (each with different kon/koff) is the bottleneck — solved with CUDA-batched ODE integration. Extended kinetic models (induced-fit docking, conformational selection) couple binding kinetics to structural biology force fields for GPU-accelerated MD-enhanced occupancy predictions.
- **Key algorithms:** Two-state receptor model ODE, Ternary Complex Model (TCM), Operational Model of Agonism, kinetic rate equation fitting (kon, koff, Kd), PET Logan reference method, Receptor Occupancy ED50 estimation, cAMP/calcium signalling cascade ODEs, mean-field receptor population models.
- **Datasets:**
  - ChEMBL binding kinetics data — kon/koff/Kd for thousands of drug-receptor pairs (https://www.ebi.ac.uk/chembl/)
  - BindingDB kinetics subset (https://www.bindingdb.org/)
  - OpenNeuro PET datasets — receptor occupancy imaging data (https://openneuro.org/)
  - Guide to Pharmacology (GtoPdb) — curated receptor/ligand database (https://www.guidetopharmacology.org/)
- **Starter repos/tools:**
  - PyDyNo (verify URL) — dynamic receptor simulation in Python
  - RTKI (Receptor-Target Kinetics Interface) (verify URL) — kinetics fitting framework
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU ODE batching applicable to receptor kinetics models
  - PySB (https://github.com/pysb/pysb) — Python rule-based biochemical network modelling
- **CUDA libraries & GPU pattern:** Custom CUDA RK4 batched ODE kernels for receptor kinetics, cuRAND for parameter uncertainty propagation, cuBLAS for Jacobian computation; pattern: one CUDA thread per drug candidate, each solving receptor binding ODEs in parallel.

---

### 13.4 Drug-Drug Interaction Prediction 🟡 · Active R&D

- **Deep dive:** Predicts pharmacokinetic drug-drug interactions (PK-DDI) caused by CYP enzyme inhibition/induction, transporter competition, and protein binding displacement; also predicts pharmacodynamic DDI from synergistic/antagonistic receptor effects. Graph neural networks encode drug molecular structure; bipartite interaction graphs model shared enzyme substrates. GPU parallelism across large drug-pair combination spaces is essential — the DrugBank DDI graph has ~250k interaction edges from 2.4k drugs, but virtual screening explores millions of hypothetical pairs. Static mechanistic models (R-value, AUC ratio prediction) are solved in batched parallel ODEs on GPU for all pairs simultaneously.
- **Key algorithms:** GNN on drug molecular graphs with edge-level DDI prediction, DeepDDI (sequence-based DDI), TransE/RotatE knowledge graph embedding for DDI, R-value static mechanistic model, AUC ratio DDI prediction, CYP inhibition ODE models, PBPK-embedded DDI simulation.
- **Datasets:**
  - DrugBank DDI — 250k+ drug interaction records with mechanism (https://www.drugbank.com/)
  - TWOSIDES — 3.7M adverse event pairs from spontaneous reports (verify URL; originally published by Tatonetti lab)
  - OFFSIDES — off-label adverse effects dataset (verify URL; Tatonetti lab)
  - FDA Adverse Event Reporting System (FAERS) (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- **Starter repos/tools:**
  - DeepDDI (https://github.com/NCIBI/DeepDDI) — deep learning DDI prediction from drug SMILES
  - SkipGNN (verify URL) — graph neural network for DDI on drug interaction graphs
  - TorchDrug (https://github.com/DeepGraphLearning/torchdrug) — GPU molecular GNN framework applicable to DDI
  - STITCH — chemical-protein interactions database (http://stitch.embl.de/) with downloadable interaction files
- **CUDA libraries & GPU pattern:** DGL/PyG sparse message passing on drug interaction graphs, cuBLAS for PBPK ODE Jacobians, custom CUDA DDI scoring kernels; pattern: batch-parallel DDI pair scoring over millions of drug combinations on GPU.

---

### 13.5 In Silico Virtual Clinical Trials 🟡 · Active R&D

- **Deep dive:** Generates virtual patient populations and runs complete simulated clinical trials in silico to optimise dose, schedule, and eligibility criteria before committing to expensive Phase II/III studies. Each virtual patient is characterised by a parameter set sampled from a PBPK/PD population distribution; simulating 5000 virtual patients through 24-week dose schedules requires 5000 independent ODE trajectories, each with ~50 compartments and hundreds of time steps. GPU-parallel batched ODE integration reduces trial simulation time from hours to seconds. Optimal virtual trial design uses GPU-resident Bayesian optimisation over dose/schedule space.
- **Key algorithms:** Monte Carlo virtual population generation, population PBPK/PD ODE integration, Latin hypercube sampling of parameter space, Bayesian optimisation of trial design parameters (dose, schedule, N), survival analysis on simulated endpoints, regulatory-grade power calculation, sensitivity analysis (Morris screening, Sobol indices).
- **Datasets:**
  - Open Systems Pharmacology virtual patient databases (https://github.com/Open-Systems-Pharmacology/)
  - ClinicalTrials.gov schema — trial design parameters for calibration (https://clinicaltrials.gov/)
  - FDA CDER pharmacometric review datasets (verify URL via FDA)
  - Published dose-finding trial datasets in CDISC format (https://www.cdisc.org/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU-accelerated virtual clinical trials in Julia
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU PBPK ODE solver for virtual patient simulation
  - SimBiology (MATLAB Parallel Computing Toolbox) — virtual trial simulation with cluster/GPU backend (verify URL)
  - PKPD Simulator (verify URL) — open Python framework for virtual trial simulation
- **CUDA libraries & GPU pattern:** CUDA batched RK45 for thousands of simultaneous patient ODE trajectories, cuRAND for virtual population parameter sampling, Thrust for summary statistic aggregation; pattern: SIMD-parallel ODE integration with each virtual patient in a CUDA warp.

---

### 13.6 Quantitative Systems Pharmacology 🟡 · Active R&D

- **Deep dive:** QSP models integrate pharmacokinetics with mechanistic biology (immune signalling, tumour growth, disease pathway models) through large ODE systems (100–10,000 equations). Stiff ODE integration dominates compute: a QSP model with 1,000 equations × 1,000 virtual patients requires solving 10⁶ coupled ODEs simultaneously. NVIDIA's nvQSP implements GPU-accelerated RODAS4 (an L-stable solver for stiff systems) specifically for this purpose, achieving orders-of-magnitude speedup. Virtual twin patient simulations for oncology trials run thousands of patient ODEs simultaneously, with GPU thread blocks each solving one patient's equation system. Physics-Informed Neural Networks (PINNs) are emerging as GPU-native surrogates that learn QSP system dynamics from data.
- **Key algorithms:** RODAS4/LSODA stiff ODE integration, sensitivity analysis (forward/adjoint), global parameter search (population Monte Carlo, ABC), PBPK-QSP coupling, immune checkpoint model ODEs (anti-PD1, CAR-T dynamics), tumour growth inhibition models, Physics-Informed Neural Networks (PINNs), QSP model reduction (MBAM).
- **Datasets:**
  - QSP model repository (DDMoRe consortium) — interoperable QSP models (https://www.ddmore.eu/)
  - BioModels Database — 2000+ curated mathematical models of biological processes (https://www.ebi.ac.uk/biomodels/)
  - NIH Systems Biology Data (verify URL) — mechanistic pathway data
  - Open Systems Pharmacology QSP library (https://github.com/Open-Systems-Pharmacology/QSP-PK-Model-Library)
- **Starter repos/tools:**
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — NVIDIA GPU-accelerated QSP ODE solver (CUDA RODAS4)
  - SBML/Tellurium (https://github.com/sys-bio/tellurium) — systems biology model simulator; GPU backend emerging
  - SBMLtoODEjl (verify URL) — Julia ODE generator from SBML for GPU integration via CUDA.jl
  - Copasi (https://copasi.org/) — biochemical network simulator; parallel via COPASI MPI interface
- **CUDA libraries & GPU pattern:** Custom CUDA RODAS4 stiff ODE kernels, cuBLAS for Jacobian LU factorisation, cuSPARSE for sparse ODE right-hand-side; pattern: one CUDA thread block per virtual patient, each thread within block updates one ODE compartment per step.

---

### 13.7 Adverse-Event & Pharmacovigilance Signal Detection 🟡 · Active R&D

- **Deep dive:** Detects unexpected drug safety signals from spontaneous reporting systems (FAERS, EudraVigilance) by applying disproportionality analysis and machine learning over millions of case reports. Reporting Odds Ratio (ROR) and Information Component (IC) calculations across all drug-AE pairs are parallelisable on GPU as batched sparse contingency table computations. Deep learning NLP models (BioBERT, ClinicalBERT) applied to FAERS narrative free-text are GPU-bound transformer inference. Longitudinal signal monitoring with Bayesian information component (multi-item gamma Poisson shrinker, MGPS) across a drug×AE matrix of 10⁶+ pairs requires GPU-resident sparse tensor operations.
- **Key algorithms:** Reporting Odds Ratio (ROR), Proportional Reporting Ratio (PRR), Multi-item Gamma Poisson Shrinker (MGPS), Bayesian Confidence Propagation Neural Network (BCPNN), NLP-based signal extraction (BERT NER on adverse event text), longitudinal CUSUM signal monitoring, graph-based drug-AE network analysis.
- **Datasets:**
  - FDA FAERS (Adverse Event Reporting System) — 25M+ individual case safety reports (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
  - EudraVigilance — EMA adverse event reporting database (https://www.adrreports.eu/)
  - WHO VigiAccess — global drug adverse reaction database (https://www.vigiaccess.org/)
  - SIDER — side-effect data from drug package inserts (http://sideeffects.embl.de/)
- **Starter repos/tools:**
  - PhViD (https://cran.r-project.org/web/packages/PhViD/) — R pharmacovigilance disproportionality package
  - pyVigilance (verify URL) — Python FDA FAERS signal detection package
  - BioBERT (https://github.com/dmis-lab/biobert) — GPU-pretrained biomedical BERT for FAERS NLP
  - OpenVigil 2.1 (http://openvigil.pharmacology.uni-kiel.de/) — web-based pharmacovigilance signal detection tool
- **CUDA libraries & GPU pattern:** cuSPARSE for sparse drug-AE contingency matrix, cuBLAS for MGPS matrix operations, cuDNN for BERT-based NLP inference; pattern: batch-parallel disproportionality computation across all drug-AE pairs on GPU.

---

### 13.8 Therapeutic Dose Individualisation / Model-Informed Dosing 🟡 · Active R&D

- **Deep dive:** Adapts drug dosing for individual patients using Bayesian updating of a population PK/PD prior with the patient's own concentration measurements (therapeutic drug monitoring, TDM). GPU acceleration is relevant in three ways: (1) population model fitting on GPU (as in 13.1); (2) real-time posterior ODE integration for thousands of candidate dose levels simultaneously to find the optimal dose; (3) simulation-based model averaging across uncertainty in individual parameters. The AUC-target dosing problem reduces to: for each candidate dose schedule, integrate the PK ODE forward for 30 days and check whether AUC hits target — parallelised across doses on GPU. Pumas and Bayesian NONMEM implement this on GPU.
- **Key algorithms:** Bayesian individual parameter estimation (MAP, full Bayes), AUC-target optimisation via GPU-parallel ODE forward simulation, MAP-adaptive dosing, Model Predictive Control (MPC) for infusion rate optimisation, optimal sampling time selection (D-optimal), individual dose prediction with uncertainty propagation, neural ODE for personalised PK.
- **Datasets:**
  - Published TDM datasets (vancomycin, aminoglycosides, tacrolimus) — available through PharmPK listserv (verify URL)
  - NONMEM example datasets — shipped with NONMEM installation (verify URL)
  - Latent Neural-ODE paper dataset (https://arxiv.org/abs/2602.03215) — personalised dosing with neural ODE
  - MIMIC-IV medication and lab data — vancomycin AUC retrospective cohorts (https://physionet.org/content/mimiciv/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU-accelerated Bayesian dose individualisation in Julia
  - Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan extension for PK ODE solving; Bayesian TDM
  - InsightRx (verify URL) — commercial Bayesian dosing platform
  - BayesPK (verify URL) — open-source Bayesian PK software for TDM
- **CUDA libraries & GPU pattern:** Custom CUDA ODE kernels for forward simulation across dose grid, cuRAND for uncertainty sampling, Thrust for AUC computation; pattern: dose-grid-parallel ODE integration — each CUDA thread evaluates one dose schedule forward simulation.

---

### 13.9 Target-Mediated Drug Disposition (TMDD) 🟡 · Active R&D

- **Deep dive:** TMDD models describe biologics (monoclonal antibodies, bispecifics) whose elimination is dominated by saturable binding to their pharmacological target, producing nonlinear, dose-dependent PK. The full TMDD ODE system (Mager-Jusko, 2001) is stiff due to fast receptor association/dissociation kinetics, requiring implicit stiff solvers. GPU parallelism is critical for virtual patient population simulations: fitting 1000 virtual patients × 100 dose schedules × stiff ODE = 10⁵ independent stiff integrations run simultaneously on GPU. Approximations (quasi-steady-state, Michaelis-Menten) reduce stiffness but must be validated against full TMDD for each compound — GPU enables this validation across large parameter grids cheaply.
- **Key algorithms:** Full TMDD ODE system (4 equations: free drug, free receptor, drug-receptor complex, total drug), Quasi-Equilibrium (QE) approximation, Quasi-Steady-State (QSS) / Michaelis-Menten approximation, stiff LSODA/CVODE integration, bivalent TMDD extensions (2025 Straube model), population NLME fitting of TMDD, slow-binding approximation.
- **Datasets:**
  - Published mAb PK datasets from Phase I trials (verify via PharmPK or ClinicalPharmacology.nih.gov)
  - Open Systems Pharmacology TMDD model examples (https://github.com/Open-Systems-Pharmacology/)
  - NONMEM TMDD example scripts (verify URL)
  - BioModels Database TMDD models (https://www.ebi.ac.uk/biomodels/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU population TMDD fitting in Julia
  - NONMEM (https://www.iconplc.com/solutions/technologies/nonmem/) — industry standard NLME for TMDD (verify GPU support status)
  - Monolix TMDD library (https://lixoft.com/model-libraries/pkpd-library/) — pre-built TMDD models (verify URL)
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver applicable to TMDD virtual patient simulations
- **CUDA libraries & GPU pattern:** Custom CUDA CVODE/RODAS4 stiff solver, cuBLAS for Jacobian LU factorisation in implicit integration; pattern: batch-parallel stiff ODE integration — one virtual patient per CUDA thread block, receptor binding equations in shared memory.

---

### 13.10 Allometric Scaling & Cross-Species PK Translation 🟢 · Established

- **Deep dive:** Translates preclinical animal PK parameters to human predictions using allometric power laws, species-specific physiological scaling, and mechanistic PBPK bridging. When applied at scale — scoring thousands of drug candidates from an in vivo animal study to prioritise compounds for human trials — the PBPK-based cross-species translation requires solving complete animal and human PBPK ODE systems for each candidate. GPU batch ODE integration across thousands of candidates simultaneously is the core acceleration; each candidate requires solving ~15-compartment human and rat/mouse PBPK systems in parallel. Machine learning models (trained on ChEMBL animal-to-human PK datasets) that predict human CL, Vd, and t½ from molecular features are GPU-accelerated via neural forward passes.
- **Key algorithms:** Simple allometry (body weight power law), Maximum Lifespan Potential (MLP) correction, Rule of Exponents, PBPK-based cross-species translation, in vitro-in vivo extrapolation (IVIVE), machine-learning regression from molecular descriptors to PK parameters, QSAR-PK modelling.
- **Datasets:**
  - ChEMBL PK dataset — 18k+ compounds with preclinical and human PK data (https://www.ebi.ac.uk/chembl/)
  - Lombardo et al. drug PK dataset — 1352 drugs with CL, Vd, t½ in humans and animals (verify URL)
  - Obach et al. clearance dataset — metabolic clearance measurements (verify URL)
  - Open Systems Pharmacology species parameter databases (https://github.com/Open-Systems-Pharmacology/PK-Sim)
- **Starter repos/tools:**
  - PK-Sim (https://github.com/Open-Systems-Pharmacology/PK-Sim) — PBPK allometric scaling built-in
  - pkNCA (https://github.com/billdenney/pknca) — non-compartmental PK analysis in R
  - DeepPK (verify URL) — deep learning PK prediction for allometric scaling
  - ADMET-AI (https://github.com/swansonk14/admet_ai) — ML-based ADME/PK prediction pipeline
- **CUDA libraries & GPU pattern:** CUDA batched RK4 for species ODE systems, cuBLAS for regression model forward pass, cuDNN for molecular graph encoder; pattern: batch-parallel PBPK translation — one compound per CUDA thread group.

---

### 13.11 Bayesian PK/PD Inference on GPU (Torsten/Stan) 🟡 · Active R&D

- **Deep dive:** Full Bayesian inference for PK/PD models using Hamiltonian Monte Carlo (HMC/NUTS) within Stan + Torsten, where the log-posterior gradient requires integrating population ODE trajectories and evaluating the likelihood. Each HMC leapfrog step requires one full ODE solve per patient in the dataset — for 1000 patients × 2000 HMC iterations × 10 leapfrog steps = 20M ODE solves per chain. GPU acceleration of these batched ODE solves provides the critical speedup. The `reduce_sum` function in Stan enables within-chain parallelism across patients on multi-core CPU; true GPU acceleration requires the CUDA ODE integration backends available through Pumas or experimental Stan GPU interfaces.
- **Key algorithms:** Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS), automatic differentiation through ODE solvers (adjoint sensitivity), Runge-Kutta ODE integration, adaptive dual-averaging stepsize, Bayesian predictive check (PPC), R-hat convergence diagnostics, Bayesian cross-validation (LOO-CV).
- **Datasets:**
  - Torsten example models (https://github.com/metrumresearchgroup/Torsten) — 2-compartment, PKPD, TMDD Stan models
  - Somatrogon population PK dataset (ResearchGate, 2024) — Bayesian NLME application with Torsten
  - Warfarin PK/PD dataset — standard Bayesian NLME benchmark (verify URL)
  - MIMIC-IV medication + lab values — vancomycin TDM for Bayesian dosing (https://physionet.org/content/mimiciv/)
- **Starter repos/tools:**
  - Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan ODE extensions for PK/PD; SAEM and HMC
  - CmdStanR / CmdStanPy (https://mc-stan.org/cmdstanr/) — Stan interface for running GPU-parallel chains
  - Pumas (https://pumas.ai/) — Julia Bayesian PK/PD with GPU-accelerated HMC via CUDA.jl
  - MCMCChains (https://github.com/TuringLang/MCMCChains.jl) — MCMC diagnostics for population PK posteriors
- **CUDA libraries & GPU pattern:** GPU-parallelised ODE solvers called from Stan adjoint sensitivity method, cuBLAS for Hessian approximation, NCCL for multi-chain parallelism; pattern: multi-GPU chains run in parallel with NCCL synchronisation for diagnostics.

---

### 13.12 Exposure-Response & Dose-Response Modelling 🟡 · Active R&D

- **Deep dive:** Quantifies the relationship between drug exposure metrics (AUC, Cmax, trough) and clinical or safety endpoints (tumour response, biomarker change, toxicity probability) using GPU-accelerated nonlinear regression and machine learning. In dose-finding trials (Phase I/II), Bayesian model-based dose-escalation designs (EWOC, mTPI-2, BLRM) require rapid posterior sampling after each dose cohort — GPU-accelerated MCMC provides the turnaround speed needed for within-day decision support. Sigmoidal Emax models, logistic regression, and exposure-toxicity models are fitted to cumulative clinical datasets with GPU-parallel gradient computation. The key bottleneck is running thousands of simulated future trial realisations in parallel for adaptive design decision criteria.
- **Key algorithms:** Sigmoidal Emax / Hill equation fitting, Bayesian Logistic Regression Model (BLRM), Escalation With Overdose Control (EWOC), modified Toxicity Probability Interval (mTPI), Emax-time models, power models, direct vs. indirect response PD models, mixture models for responder/non-responder subpopulations, concordance dose-response index.
- **Datasets:**
  - FDA Pharmacometrics Reviews — dose-response data from NDA/BLA submissions (https://www.fda.gov/drugs/drug-approvals-and-databases/pharmacometrics-reviews)
  - Published dose-escalation trial data in Oncology (verify individual publications)
  - DoseFinding R package example datasets (verify URL)
  - CDISC ADaM dose-response trial data formats (https://www.cdisc.org/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU Bayesian E-R modelling in Julia
  - DoseFinding R package (https://cran.r-project.org/web/packages/DoseFinding/) — classical dose-finding model fitting
  - BOIN (https://cran.r-project.org/web/packages/BOIN/) — Bayesian optimal interval design for dose-finding
  - trialDesign (verify URL) — simulation platform for adaptive dose-escalation
- **CUDA libraries & GPU pattern:** cuRAND for Monte Carlo posterior simulation, cuBLAS for sigmoidal Emax regression Hessians, custom CUDA kernels for parallel trial simulation over candidate dose levels; pattern: GPU-parallel simulation of thousands of adaptive trial scenarios for decision support.

---

### 13.13 QT-Prolongation & Cardiac Safety Risk Assessment 🟡 · Active R&D

- **Deep dive:** Predicts drug-induced QT interval prolongation — a surrogate for fatal arrhythmia (Torsade de Pointes) — from drug structure, hERG channel IC50 measurements, and clinical ECG data. The CardioGenAI framework uses GPU-accelerated molecular graph neural networks to predict hERG block and re-engineer drug structures for reduced liability. Clinical ECG-based deep learning (3DRECON-QT) reconstructs 3D spatial QTc from single-lead recordings using CNN on GPU. Mechanistic cardiac action potential models (O'Hara-Rudy, Paci human iPSC-CM) simulate drug effects on ion channels at thousands of drug concentrations simultaneously on GPU — each simulation is an ODE stiff system on the 40+ state Hodgkin-Huxley-type action potential model.
- **Key algorithms:** GNN-based hERG IC50 prediction from SMILES, 3DRECON-QT spatial reconstruction, O'Hara-Rudy action potential ODE, voltage-clamp state machine (Markov model for hERG), torsade de pointes risk classification (TdP risk categories), dynamic clamp simulation on GPU, QTc Fridericia/Bazett correction.
- **Datasets:**
  - CiPA (Comprehensive in vitro Pro-arrhythmia Assay) ion channel datasets — multi-channel IC50 for 28 reference drugs (verify URL via FDA)
  - hERGCentral database — hERG patch-clamp measurements (verify URL)
  - MIMIC-IV-ECG — clinical QTc measurements linked to medication data (https://physionet.org/content/mimic-iv-ecg/)
  - CardioNet ECG database (verify URL) — large annotated ECG dataset for QT analysis
- **Starter repos/tools:**
  - CardioGenAI (https://github.com/mgreenig/CardioGenAI) — ML framework for re-engineering drugs for reduced hERG liability
  - myokit (https://github.com/myokit/myokit) — cardiac action potential ODE modelling; GPU via CUDA backend
  - OpenCARP (https://opencarp.org/) — cardiac electrophysiology simulator with GPU support
  - DeepHERG (verify URL) — deep learning hERG inhibition prediction
- **CUDA libraries & GPU pattern:** DGL for hERG GNN, custom CUDA Hodgkin-Huxley ODE kernels for action potential batch simulation, cuRAND for Monte Carlo drug concentration sweeps; pattern: one CUDA thread per drug concentration × cell simulation, with shared memory for ion channel state variables.

---

### 13.14 Optimal Experimental Design for Clinical PK Studies 🔴 · Frontier/Theoretical

- **Deep dive:** Identifies optimal blood sampling times and dose levels for population PK studies to maximise Fisher Information (D-optimality) or minimise cost given constraints on sample number and patient burden. The Fisher Information Matrix (FIM) for a nonlinear mixed-effects model requires integrating ODE trajectories at all candidate sampling times and evaluating the sensitivity of outputs to parameters — an O(N_times × N_params²) computation. GPU parallelism across candidate design grids (millions of combinations of sampling times × doses) enables global search in hours vs. days. Bayesian optimisation of design using surrogate models trained on GPU-simulated FIM evaluations represents the frontier approach.
- **Key algorithms:** D-optimal, A-optimal, E-optimal design criteria on Fisher Information Matrix, MFIM (Matrix FIM) computation for NLME, Bayesian D-optimality (BOED), Sequential Bayesian Experimental Design, derivative-informed neural operators for FIM surrogate, population FIM via importance sampling, D-optimal dose selection for Phase I.
- **Datasets:**
  - NONMEM example datasets for FIM validation (verify URL)
  - PopED example models — software-integrated benchmark designs (https://github.com/andrewhooker/PopED)
  - PAGE (Population Approach Group in Europe) OED workshop data (verify URL)
  - Optimal experiment design PK examples (https://pmc.ncbi.nlm.nih.gov/articles/PMC11996619/)
- **Starter repos/tools:**
  - PopED (https://github.com/andrewhooker/PopED) — R/MATLAB optimal experimental design for population PK
  - PFIM (verify URL) — R package for Fisher Information Matrix-based design
  - Pumas OptimalDesign extension (https://pumas.ai/) — GPU-accelerated OED in Julia
  - Pyomo.DoE (https://github.com/IDAES/idaes-pse) — Python optimal experimental design (verify URL)
- **CUDA libraries & GPU pattern:** Custom CUDA sensitivity ODE kernels for FIM computation, cuBLAS for FIM matrix determinant (log-det D-criterion), cuRAND for Bayesian design Monte Carlo; pattern: GPU grid search over sampling time combinations with parallel FIM evaluation per design point.

---

### 13.15 Drug-Induced Liver Injury (DILI) & Quantitative Systems Toxicology 🟡 · Active R&D

- **Deep dive:** Predicts and mechanistically explains drug-induced liver injury using multi-scale QST models (DILIsym) that integrate intracellular mitochondrial function, bile acid synthesis/transport, oxidative stress, and innate immune response with drug concentration-dependent perturbations. The stiff ODE system (300+ equations for intracellular biochemistry × hepatocyte populations × liver zonation) requires GPU-parallel stiff integration for virtual patient simulations. Graph convolutional networks on drug molecular graphs (BioGL-GCN) trained on hepatotoxicity labels enable rapid screening of new compounds. Combining GCN screening with mechanistic QST validation on GPU covers both speed and interpretability.
- **Key algorithms:** QST ODE integration (CVODE, RODAS4), mitochondrial membrane potential dynamics ODEs, bile acid transport ODE system, NF-κB signalling cascade, GCN/GNN on molecular graphs for hepatotoxicity classification, random forest + physicochemical feature DILI prediction, multiscale coupling of PBPK with intracellular QST.
- **Datasets:**
  - DILIst — curated DILI positive/negative drug list (verify URL; NCATS)
  - LiverTox — NIH database of drug-induced liver disease (https://www.ncbi.nlm.nih.gov/books/NBK547852/)
  - Tox21 — 12,000+ compounds with hepatotoxicity assay data (https://tox21.gov/)
  - DILIsym virtual patient database (Simulations Plus) — calibrated virtual liver population (verify URL)
- **Starter repos/tools:**
  - DILIsym (https://www.simulations-plus.com/software/dilisym/) — commercial QST DILI platform (Simulations Plus)
  - BioGL-GCN (verify URL) — graph convolutional network for DILI prediction from drug structures
  - DeepTox (https://github.com/bioinf-jku/tox21_networks) — deep learning Tox21 prediction baseline
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver for QST models
- **CUDA libraries & GPU pattern:** Custom CUDA CVODE/RODAS4 stiff ODE kernels for QST integration, DGL for hepatotoxicity GCN, cuBLAS for bile acid flux Jacobians; pattern: virtual patient batch — one CUDA block per patient, intracellular biochemistry compartments in shared memory.

---

### 13.16 Receptor Occupancy Imaging & PET Pharmacokinetics 🟡 · Active R&D

- **Deep dive:** Analyses PET neuroimaging data to quantify receptor occupancy by drug candidates across thousands of brain voxels simultaneously. The Logan reference tissue method and two-tissue compartmental models must be fitted to the time-activity curve (TAC) at each voxel — a problem with 100k+ independent nonlinear regressions that map directly to GPU parallelism. GPU-parallel voxel-wise model fitting achieves near-real-time analysis of 3D PET volumes (128×128×63 voxels). Virtual receptor occupancy simulations (coupling PBPK with brain RO submodel) for dose selection require batched ODE integration on GPU across candidate dose levels.
- **Key algorithms:** Logan reference tissue method, two-tissue compartmental model, simplified reference tissue model (SRTM), voxel-wise ODE fitting with Levenberg-Marquardt on GPU, Patlak graphical analysis, partial volume correction, kinetic parameter estimation (K1, k2, BP_ND).
- **Datasets:**
  - OpenNeuro PET datasets — open-access brain PET with kinetic data (https://openneuro.org/)
  - NeuroVault PET studies — aggregated neuroimaging PET data (https://neurovault.org/)
  - BrainPET benchmark datasets (verify URL — NIMH)
  - ADNI PET-amyloid data — longitudinal PET for Alzheimer imaging (https://adni.loni.usc.edu/)
- **Starter repos/tools:**
  - NiftyPAD (verify URL) — GPU-parallelised PET kinetic modelling toolkit
  - TPCCLIB (verify URL) — C library for PET kinetic analysis (CPU; GPU extension possible)
  - Pumas (https://pumas.ai/) — GPU-accelerated brain RO-PBPK coupling in Julia
  - SimplePET (https://github.com/UCL/simplicity) — Python PET simulation and analysis (verify URL)
- **CUDA libraries & GPU pattern:** Custom CUDA Levenberg-Marquardt kernels for per-voxel TAC fitting, cuBLAS for covariance matrix inversion, cuFFT for PET sinogram reconstruction; pattern: one CUDA thread per voxel, embarrassingly parallel kinetic fitting.

---

### 13.17 Neural-ODE & Physics-Informed Neural Networks for PK/PD 🔴 · Frontier/Theoretical

- **Deep dive:** Replaces explicit pharmacokinetic ODEs with neural networks embedded within differential equations (Neural ODEs) or constrains neural architectures to satisfy ODE physics (Physics-Informed Neural Networks, PINNs). This allows learning latent pharmacokinetic dynamics from sparse clinical observations without specifying a mechanistic compartmental model. The GPU bottleneck is differentiating through the ODE solver (adjoint sensitivity method) for backpropagation, implemented in torchdiffeq. For PINNs, the collocation loss (residual of the ODE at sample points) is evaluated in batches on GPU. Recent Latent Neural-ODE approaches (arXiv:2602.03215) model-informed precision dosing with 15% fewer AEs than standard dosing.
- **Key algorithms:** Neural ODE (Chen et al. 2018), adjoint sensitivity for backprop through ODE, Physics-Informed Neural Networks (PINNs), Universal Differential Equations (UDEs), Latent ODE with VAE encoder, Gaussian process ODE priors, Fourier Neural Operators for PDE-based dosing, symbolic regression to recover interpretable ODE from data.
- **Datasets:**
  - Latent Neural-ODE precision dosing dataset (https://arxiv.org/abs/2602.03215) — model-informed dosing with neural ODE
  - MIMIC-IV ICU PK data — vancomycin/aminoglycoside time series (https://physionet.org/content/mimiciv/)
  - Published population PK datasets (vancomycin, busulfan) from PharmPK listserv (verify URL)
  - Synthetic NLME benchmark datasets from Monolix/NONMEM validation suites (verify URL)
- **Starter repos/tools:**
  - torchdiffeq (https://github.com/rtqichen/torchdiffeq) — Neural ODE with GPU-accelerated adjoint sensitivity
  - DiffEqFlux.jl (https://github.com/SciML/DiffEqFlux.jl) — Universal Differential Equations in Julia with GPU
  - DeepXDE (https://github.com/lululxvi/deepxde) — GPU PINN framework for PDE/ODE-constrained learning
  - SciMLBenchmarks (https://github.com/SciML/SciMLBenchmarks.jl) — benchmarks for neural ODE solvers
- **CUDA libraries & GPU pattern:** torchdiffeq adjoint ODE solver on GPU via PyTorch CUDA, cuBLAS for neural ODE network forward pass, JAX XLA for JIT-compiled PINN training; pattern: batched neural ODE integration with GPU-resident adjoint sensitivity gradients.

---

### 13.18 Pharmacogenomics-Guided Precision Dosing 🔴 · Frontier/Theoretical

- **Deep dive:** Integrates individual genetic variants (CYP2D6, CYP2C19, VKORC1, SLCO1B1, UGT1A1) with demographic covariates and drug-specific models to predict optimal starting doses for precision medicine. GPU parallelism enables simulation of the full population variant space: with 50 common pharmacogenomic variants each having 2–3 allele combinations, the state space has ~10¹⁰ genotype combinations, requiring Monte Carlo sampling across thousands of virtual genotype profiles simultaneously. GWAS-based PK prediction models (GNN + genomic embedding) trained on biobank-scale data (UK Biobank + All of Us) are GPU-training-bound. Deep learning ensemble models that integrate genotype × drug × phenotype interactions require massive batched forward passes on GPU.
- **Key algorithms:** CPIC (Clinical Pharmacogenomics Implementation Consortium) rules, population PK covariate models incorporating genotype, GWAS-PK association studies, GNN on drug metabolic pathway graphs, random forest / gradient boosting with genomic features, Bayesian network for genotype-phenotype-PK integration, polygenic PK scores.
- **Datasets:**
  - PharmGKB — curated gene-drug relationships with evidence levels (https://www.pharmgkb.org/)
  - CPIC guidelines data — actionable pharmacogenomic recommendations (https://cpicpgx.org/)
  - UK Biobank genotype + prescription data — 500k individuals (https://www.ukbiobank.ac.uk/)
  - All of Us pharmacogenomics cohort (https://allofus.nih.gov/)
- **Starter repos/tools:**
  - PyPGx (https://github.com/sbslee/pypgx) — pharmacogenomics genotyping from next-generation sequencing
  - PharmCAT (https://github.com/PharmGKB/PharmCAT) — pharmacogenomics clinical annotation tool
  - Pumas PGx module (https://pumas.ai/) — pharmacogenomics-integrated PK/PD in Julia (verify URL)
  - SAIGE-GPU (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12960912/) — GPU GWAS for PK covariate discovery (verify GitHub URL)
- **CUDA libraries & GPU pattern:** cuSPARSE for genotype matrix operations, cuDNN for neural PGx-PK models, cuRAND for Monte Carlo genotype space exploration; pattern: GPU-parallel simulation across thousands of genotype profiles × dose schedules.

---

### 13.19 Antibody Pharmacokinetics & FcRn-Mediated Recycling 🔴 · Frontier/Theoretical

- **Deep dive:** Models the complex PK of monoclonal antibodies and bispecifics, which are dominated by FcRn-mediated endosomal recycling, target-mediated drug disposition, and antigen sink effects. The multi-compartment antibody PK model (plasma, interstitial, endosome, target tissue) coupled with FcRn binding dynamics is a stiff ODE system with widely separated time scales (hours vs. weeks). Population simulation of thousands of virtual patients with variable FcRn expression, antigen expression, and body composition requires GPU-parallel stiff ODE integration. Antibody engineering to optimise FcRn affinity and pH-dependent binding can be virtually screened at scale on GPU.
- **Key algorithms:** Two-compartment antibody model with FcRn recycling submodel (Dhanarajan-Meibohm), TMDD with high-affinity target binding, pH-dependent FcRn binding kinetics (endosomal pH 6.0 vs. plasma pH 7.4), neonatal clearance model, multi-target bispecific PK (dual TMDD), stiff LSODA/CVODE integration, population NLME for biologics.
- **Datasets:**
  - Published mAb PK datasets — Phase I first-in-human PK from IND/NDA submissions (verify via FDA label search)
  - BioModels antibody models (https://www.ebi.ac.uk/biomodels/)
  - Open Systems Pharmacology mAb PBPK library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library)
  - DrugBank biologic PK data (https://www.drugbank.com/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU biologics PK modelling in Julia
  - PK-Sim mAb models (https://github.com/Open-Systems-Pharmacology/PK-Sim) — PBPK for antibodies with FcRn
  - mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — R-based simulation of PKPD ODEs, parallelisable with OpenMP
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver for antibody PK virtual populations
- **CUDA libraries & GPU pattern:** Custom CUDA CVODE stiff ODE kernels with FcRn endosomal binding, cuBLAS for Jacobian LU factorisation, cuRAND for virtual patient FcRn expression sampling; pattern: one CUDA block per virtual patient, pH-dependent binding kinetics in thread-local registers.

---

## 14. Emerging, Theoretical & Grand-Challenge Frontiers

### 14.1 Whole-Cell Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Whole-cell simulation aspires to mechanistically model every gene, mRNA, protein, metabolite, and organelle in a single bacterium or yeast cell simultaneously. The scale challenge is staggering: E. coli has ~4,300 genes and ~1.5 M ribosomes; a complete stochastic reaction-diffusion simulation at molecular resolution would require centuries on a single CPU. GPU acceleration of spatial SSA (Gillespie/tau-leaping) over a discretized cell volume enables partial whole-cell models (gene expression + metabolism) to run in tractable time. The STEPS simulator (parallel, GPU-accelerated) handles reaction-diffusion on tetrahedral meshes representing subcellular geometry. Achieving true whole-cell simulation likely requires exascale GPU clusters.
- **Key algorithms:** Spatial Gillespie SSA (Next Subvolume Method / ISSA), tau-leaping with error control, next-reaction method (NRM), multiscale hybrid: ODE for deterministic fast species + SSA for rare events, GPU-parallel lattice microbes (LM) algorithm, whole-cell model composition (FBA + transcription/translation + signaling).
- **Datasets:** Mycoplasma genitalium whole-cell model (Karr et al. Cell 2012) parameters (https://simtk.org/projects/wc_models); E. coli K-12 transcriptomics (GEO GSE2198 and related); BioModels Database whole-cell models (https://www.ebi.ac.uk/biomodels/); JCVI Syn3A minimal genome datasets (https://www.jcvi.org/research/first-minimal-synthetic-bacterial-cell).
- **Starter repos/tools:** STEPS (https://github.com/CNS-OIST/STEPS) — GPU-accelerated stochastic spatial reaction-diffusion in tetrahedral meshes; Lattice Microbes (LM) (https://github.com/Luthey-Schulten-Lab/Lattice_Microbes) — GPU spatial stochastic simulator for E. coli; Smoldyn (https://github.com/ssandrews/Smoldyn) — off-lattice particle-based RD simulator (multi-GPU); WholeCellKB (https://github.com/CovertLab/WholeCell) — Karr whole-cell model framework.
- **CUDA libraries & GPU pattern:** CUDA kernels for parallel subvolume SSA reaction firing, cuRAND for per-subvolume random streams, NCCL for multi-GPU spatial domain decomposition; pattern: cell volume partitioned into tetrahedral subvolumes on GPU → parallel SSA firing per subvolume → diffusive transfer between subvolumes via CUDA inter-thread communication → global species count aggregation → repeat at nanosecond timescale.

---

### 14.2 Spatial / Whole-Cell Reaction-Diffusion at Molecular Resolution 🔴 · Frontier/Theoretical

- **Deep dive:** Particle-based reaction-diffusion (PBRD) simulators track each molecule as an individual particle, enabling sub-micron spatial resolution of signaling gradients, receptor clustering, and organelle targeting. GPU-accelerated PBRD (Smoldyn GPU, ReaDDy GPU) parallelizes over molecules: each particle diffuses and reacts independently, with nearest-neighbor checks via GPU cell-list algorithms. A full cytoplasm simulation at molecular resolution for even a minimal cell (~500 K unique molecules) at physiologically relevant timescales (milliseconds) requires O(10¹²) timestep-particle updates — tractable only on multi-GPU systems. eGFRD (enhanced Green's Function Reaction Dynamics) is theoretically the most accurate but computationally costly, a prime GPU target.
- **Key algorithms:** Brownian dynamics with reaction (Smoluchowski), eGFRD Green's function propagators, interaction-site model (ISSA), diffusion-limited reaction kernel sampling, GPU cell-list O(N) neighbor search, reactive molecular dynamics.
- **Datasets:** CellOrganizer — generative models of subcellular morphology for simulation domains (http://www.cellorganizer.org/); PDB molecular crowding configurations; SBML-spatial format models (BioModels); MCell neural synapse models (https://mcell.org/).
- **Starter repos/tools:** ReaDDy (https://github.com/readdy/readdy) — GPU-accelerated particle-based RD (CPU + GPU backends); Smoldyn (https://github.com/ssandrews/Smoldyn) — off-lattice GPU-capable PBRD; MCell (https://mcell.org/) — Monte Carlo 3D reaction-diffusion for neurons; STEPS (https://github.com/CNS-OIST/STEPS) — tetrahedral-mesh spatial SSA with GPU support.
- **CUDA libraries & GPU pattern:** CUDA cell-list neighbor search (one thread per particle for neighbor pair collection), cuRAND for per-particle Brownian displacement sampling, Thrust for reaction-event sorting; pattern: GPU cell-list built from particle positions → parallel Brownian displacement → reaction probability check for each particle pair → acceptance-rejection sampling → time step advance.

---

### 14.3 Patient / Organ Digital Twins 🟡 · Active R&D

- **Deep dive:** A patient digital twin integrates genomics, imaging, hemodynamics, metabolism, and pharmacokinetics into a continuously updated computational model that predicts disease progression and treatment response for a specific individual. GPU-accelerated cardiac digital twins (npj Systems Biology 2023) solve full multi-physics cardiac electromechanics in a few hours per heartbeat on 4 GPU cards — opening systematic cohort-scale simulation campaigns. Cancer digital twins ("cancer avatars") couple tumor growth PDE models with pharmacodynamic ODEs, parameterized from serial liquid biopsies, enabling adaptive treatment optimization. The GPU bottleneck is the repeated FEM/CFD solve at each patient update cycle.
- **Key algorithms:** Multi-physics cardiac electromechanics (bidomain + passive/active material), tumor growth (reaction-diffusion PDE, Go-or-Grow model), pharmacokinetic-pharmacodynamic (PKPD) ODE integration, Bayesian data assimilation (ensemble Kalman filter), physics-informed neural network surrogates, mesh morphing for anatomy personalization.
- **Datasets:** UK Biobank Imaging — cardiac MRI + genomics on 100 K subjects (https://www.ukbiobank.ac.uk/); TCIA — cancer imaging archive (https://www.cancerimagingarchive.net/); ClinicalTrials.gov synthetic patient cohorts; Digital Twin Cardiovascular Cohort (GPU-accelerated, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10203142/).
- **Starter repos/tools:** NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — physics-informed neural network surrogates for organ modeling; OpenCMISS-Iron (https://github.com/OpenCMISS/iron) — GPU-capable cardiac electromechanics finite-element solver; SimVascular (https://github.com/SimVascular/SimVascular) — patient-specific cardiovascular CFD; CHASTE (https://github.com/Chaste/Chaste) — cardiac and tumor multiscale simulation.
- **CUDA libraries & GPU pattern:** cuSPARSE for bidomain cardiac FEM, cuDNN for surrogate model inference, NCCL for multi-GPU organ assembly; pattern: patient imaging segmented → geometry personalized → GPU multi-physics solve → Bayesian assimilation of new biomarker measurements → treatment response prediction → update digital twin.

---

### 14.4 Quantum-Classical Hybrid Drug Design 🔴 · Frontier/Theoretical

- **Deep dive:** Quantum computers can solve electronic structure problems for drug-binding active sites more accurately than classical DFT, but current NISQ devices are noisy and limited to ~50–100 qubits. Hybrid quantum-classical algorithms (VQE for Hamiltonian ground-state energies, QAOA for docking optimization) run the quantum circuit on the QPU and the classical optimization loop on GPU clusters, with GPU accelerating the many-shot Pauli expectation-value estimation. AWS Quantum Computing Exploration for Drug Discovery (2024) demonstrates VQE-based protein folding in small fragments. GPU simultaneously handles the classical molecular mechanics components of QM/MM with GPU-accelerated DFT (CP2K, Psi4 on GPU). The practical near-term use case is 20–50 atom active-site electronic structure for tight binding-affinity ranking.
- **Key algorithms:** Variational Quantum Eigensolver (VQE), Quantum Approximate Optimization Algorithm (QAOA), GPU-accelerated density functional theory (DFT, B3LYP/PBE), QM/MM with quantum active site, orbital-free embedding, GPU-accelerated tensor network contraction for quantum state simulation.
- **Datasets:** PDBbind refined binding affinity dataset (http://www.pdbbind.org.cn/); ChEMBL (https://www.ebi.ac.uk/chembl/) for classical ML baseline; QM9 (GPU DFT benchmark, 134 K small molecules, https://paperswithcode.com/dataset/qm9); AWS Quantum Drug Discovery Benchmark (https://github.com/aws-solutions-library-samples/quantum-computing-exploration-for-drug-discovery-on-aws).
- **Starter repos/tools:** Qiskit (https://github.com/Qiskit/qiskit) — VQE/QAOA with GPU-accelerated statevector simulator (cuStateVec); PennyLane (https://github.com/PennyLaneAI/pennylane) — differentiable quantum ML with GPU backend; Psi4 (https://github.com/psi4/psi4) — GPU-accelerated QM (CUDA DFT integrals); CP2K (https://github.com/cp2k/cp2k) — GPU QM/MM with CUDA backend.
- **CUDA libraries & GPU pattern:** cuStateVec (NVIDIA cuQuantum) for GPU quantum circuit simulation, cuDNN for NN-guided ansatz optimization, CUDA DFT integral kernels (Psi4/CP2K); pattern: drug-protein complex → GPU DFT for electronic Hamiltonian → Pauli decomposition → VQE on GPU statevector sim (or QPU) → binding ΔG estimate → classical optimizer updates ansatz parameters.

---

### 14.5 Foundation Models for Biology & Chemistry 🟡 · Active R&D

- **Deep dive:** Large pre-trained models on biological sequences (ESM-2: 15B parameters on 250M protein sequences), genomic DNA (Nucleotide Transformer, Evo-1), and chemical SMILES (ChemBERTa, MolGPT) are rapidly becoming universal biological encoders. GPU training at scale (thousands of A100s) is the defining infrastructure requirement; GPU inference is the deployment bottleneck for drug discovery pipelines scoring millions of candidates. Fine-tuning foundation models on task-specific biomedical datasets (DMS, HTS, survival data) achieves state of the art across fitness prediction, structure prediction, and clinical outcome forecasting. AMix-1 (2025) demonstrates mixture-of-experts protein foundation models with test-time scaling.
- **Key algorithms:** Masked language modeling (MLM) pre-training, attention with rotary position encoding (RoPE), LoRA/QLoRA fine-tuning, retrieval-augmented generation for protein databases, multi-modal fusion (sequence + structure + expression), model distillation for edge deployment.
- **Datasets:** UniRef90/UniClust30 — protein sequence clusters for pre-training (https://www.uniprot.org/); PDB (https://www.rcsb.org/) — 230K+ structures for structure-aware pre-training; ChEMBL (https://www.ebi.ac.uk/chembl/) — 2.4M bioactive compounds; NCBI RefSeq — genomic DNA pre-training corpus (https://www.ncbi.nlm.nih.gov/refseq/).
- **Starter repos/tools:** ESM2/ESMFold (https://github.com/facebookresearch/esm) — FAIR protein LLM + structure prediction on GPU; Evo (https://github.com/evo-design/evo) — genomic DNA foundation model (Arc Institute); HuggingFace Transformers (https://github.com/huggingface/transformers) — training/fine-tuning infrastructure; NVIDIA BioNeMo (https://www.nvidia.com/en-us/clara/bionemo/) — GPU-optimized biology foundation model platform.
- **CUDA libraries & GPU pattern:** cuDNN FlashAttention-2 for memory-efficient attention, Tensor Core BF16/FP8 matmuls, NCCL for tensor/pipeline parallelism; pattern: sequence tokenization → distributed data-parallel GPU training → gradient checkpointing for memory → task-specific LoRA fine-tuning → GPU batch inference over candidate libraries.

---

### 14.6 Generative 3D Molecular & Structure Design (Diffusion) 🟡 · Active R&D

- **Deep dive:** SE(3)-equivariant diffusion models (RFdiffusion, DiffSBDD, DiffDock) generate novel protein backbones, ligands, and protein-ligand complexes by reversing a Gaussian noise process on atomic coordinates while respecting 3D rotational and translational symmetry. GPU acceleration is essential: RFdiffusion generates 1000 protein backbone samples in ~30 minutes on a single A100, versus days on CPU. RFdiffusion2 (Rosetta Commons, 2025) extends to more complex enzyme active-site design. Flow matching variants (SemlaFlow, 3D SE(3) flows) offer faster convergence with fewer NFE (number of function evaluations) on GPU. The core GPU computation is equivariant message-passing neural network (EGNN/SE3-Transformer) forward/backward passes.
- **Key algorithms:** SE(3)-equivariant score-based diffusion (DDPM on SO(3) × R³), flow matching on SE(3), EGNN/SE3-Transformer equivariant message passing, fragment-based ligand generation (DiffSBDD), all-atom diffusion (RFdiffusionAA), Clifford group equivariant diffusion.
- **Datasets:** PDB (https://www.rcsb.org/) — all experimentally determined protein structures; CrossDocked2020 — 22.5M protein-ligand poses for structure-based drug design (https://github.com/gnina/models); ProteinGym (https://proteingym.org/) — fitness evaluation of generated sequences; ZINC20 (https://zinc20.docking.org/) — 1.4B purchasable molecules for ligand generation benchmarks.
- **Starter repos/tools:** RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — protein backbone + binder design via SE(3) diffusion; RFdiffusionAA (https://github.com/baker-laboratory/rf_diffusion_all_atom) — all-atom enzyme design; DiffSBDD (https://github.com/arneschneuing/DiffSBDD) — 3D pocket-conditioned ligand generation; DiffDock (https://github.com/gcorso/DiffDock) — GPU diffusion-based molecular docking.
- **CUDA libraries & GPU pattern:** cuDNN for EGNN/SE3-Transformer message-passing, Flash Attention for multi-head attention over atom graphs, mixed-precision BF16 for diffusion score network; pattern: protein target coordinates → GPU diffusion noise schedule → iterative denoising via equivariant network → structure refinement with ProteinMPNN → GPU energy evaluation filter.

---

### 14.7 Closed-Loop Autonomous "Self-Driving" Labs 🔴 · Frontier/Theoretical

- **Deep dive:** Self-driving labs (SDLs) close the design-build-test-learn cycle by coupling GPU-accelerated Bayesian optimization (BO) or reinforcement learning to robotic liquid handlers, automated assays, and real-time data pipelines. The GPU role is the inner-loop inference: scoring thousands of candidate experiments via surrogate models (GP, neural network ensembles) in milliseconds, so the acquisition function evaluates faster than the robot can dispense. Active learning for drug discovery (e.g., Gaussian Process + batch BO with qEI) has been shown to find optima in 10–50× fewer experiments. Photonic lab automation systems integrate GPU-accelerated spectroscopic analysis (Raman, fluorescence) for real-time compound characterization.
- **Key algorithms:** Bayesian optimization with Gaussian process (GP-UCB, qEI), neural network ensemble surrogate, multi-fidelity BO, reinforcement learning (PPO for experiment selection), active learning, parallel batch BO (TurBO), uncertainty quantification via deep ensembles or MC dropout.
- **Datasets:** ChEMBL HTS screening data (https://www.ebi.ac.uk/chembl/); Open Reaction Database (ORD) — chemical reaction outcomes (https://open-reaction-database.org/); Therapeutic Data Commons (TDC) — multi-property drug benchmarks (https://tdcommons.ai/); Syngas Fermentation Simulator multi-fidelity dataset (https://arxiv.org/abs/2311.05776).
- **Starter repos/tools:** BoTorch (https://github.com/pytorch/botorch) — GPU Bayesian optimization with PyTorch; Ax (https://github.com/facebook/Ax) — adaptive experimentation platform using BoTorch; Summit (https://github.com/sustainable-processes/summit) — BO library for chemical process optimization; Olympus (https://github.com/aspuru-guzik-group/olympus) — benchmark framework for self-driving lab algorithms.
- **CUDA libraries & GPU pattern:** cuDNN for deep ensemble surrogate inference, Cholesky factorization via cuSolver for GP posterior, GPU-accelerated acquisition function optimization (batch gradient ascent); pattern: prior experiment observations → GPU GP/neural surrogate fit → parallel acquisition function maximization (256 candidates) → top-k experiments dispatched to robot → new measurements update surrogate.

---

### 14.8 Real-Time Genomic Pathogen Surveillance Networks 🟡 · Active R&D

- **Deep dive:** Epidemic genomic surveillance sequences thousands of viral/bacterial isolates per day, requiring near-real-time genome assembly, variant calling, phylogenetic placement, and transmission cluster detection. GPU-accelerated genome assembly (GPU-MEGAHIT) and variant calling (GPU Parabricks) reduce per-sample analysis from hours to minutes, enabling next-flight sequencing decisions during outbreak response. Phylogenetics on GPU (iqtree GPU, PhyML-CUDA) computes maximum likelihood trees on thousands of taxa. Real-time cluster detection via GPU-accelerated pairwise SNP distance matrices (all-vs-all on N×N matrix) parallelizes naturally over GPU threads.
- **Key algorithms:** GPU-accelerated de novo assembly (BWT-based, de Bruijn graph), GPU variant calling (Parabricks Haplotypecaller), maximum likelihood phylogenetics (GTR+Γ model), pairwise SNP distance matrix, Bayesian temporal phylogenetics (BEAST GPU backend), epidemic growth rate estimation (SEIR model on GPU).
- **Datasets:** GISAID EpiCoV — 17M+ SARS-CoV-2 genomes (https://gisaid.org/); NCBI SRA — all short-read sequencing submissions (https://www.ncbi.nlm.nih.gov/sra); Nextstrain builds — curated SARS-CoV-2 / influenza phylogenies (https://nextstrain.org/); PHA4GE pathogen genomics standards datasets (https://pha4ge.org/).
- **Starter repos/tools:** NVIDIA Clara Parabricks (https://www.nvidia.com/en-us/clara/parabricks/) — GPU genome assembly/variant calling (40× speedup over GATK); Nextstrain (https://github.com/nextstrain/ncov) — phylogenetic outbreak analysis pipeline; IQ-TREE (https://github.com/Cibiv/IQ-TREE) — ML phylogenetics (multi-GPU via CUDA); GPU-MEGAHIT (https://github.com/GPU-MEGAHIT/GPU-MEGAHIT) — GPU-accelerated metagenomics assembly.
- **CUDA libraries & GPU pattern:** CUDA BWT for GPU read alignment (BWA-MEM on CUDA), cuBLAS for SNP distance matrix computation, cuFFT for k-mer frequency analysis; pattern: raw reads → GPU assembly → GPU variant calling → pairwise SNP matrix on GPU → transmission cluster detection → phylogenetic placement → epidemiological alert.

---

### 14.9 Multi-Physics Tumor / Treatment Digital Twin 🔴 · Frontier/Theoretical

- **Deep dive:** A cancer digital twin couples tumor growth (reaction-diffusion PDE for cell density + nutrient + oxygen), mechanical deformation of surrounding tissue (FEM), vascular remodeling (angiogenesis ODE), drug pharmacokinetics (PKPD ODE), immunological response, and radiation damage (LQ model), all personalized from serial multimodal imaging. GPU parallelism tackles the stiff multi-physics coupling: the reaction-diffusion grid (512³ voxels), the FEM mesh (500K elements), and the vascular graph (10⁴ vessel segments) each run on separate GPU streams, synchronized at each time step. Multi-GPU inverse problem fitting of all biophysical parameters to longitudinal MRI + ctDNA data is the frontline computational challenge. The field saw publication of physics-informed ML digital twins for prostate cancer (PSA-driven) in Nature npj Digital Medicine 2025.
- **Key algorithms:** Anisotropic tumor-growth reaction-diffusion PDE (Fisher-Kolmogorov), vascular angiogenesis ODE (VEGF-driven), linear-quadratic (LQ) radiation damage model, pharmacokinetic two-compartment model, Bayesian ensemble Kalman filter for parameter assimilation, adjoint-based sensitivity for PDE inversion.
- **Datasets:** TCIA (The Cancer Imaging Archive) — multimodal tumor imaging (https://www.cancerimagingarchive.net/); TCGA (The Cancer Genome Atlas) — multi-omics tumor data (https://www.cancer.gov/tcga); ISPY2 — breast cancer treatment response imaging trial (https://www.ispy2.org/); NSCLC-Radiomics (Lung1) — CT + survival on 422 patients (https://www.cancerimagingarchive.net/).
- **Starter repos/tools:** CHASTE (https://github.com/Chaste/Chaste) — cancer multiscale + vascular simulation; OpenCMISS-Iron (https://github.com/OpenCMISS/iron) — GPU FEM for tumor-tissue mechanics; NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — PINN surrogates for tumor growth; TumorFEM (verify URL, search "tumor digital twin FEM GitHub") — patient-specific tumor mechanical FEM.
- **CUDA libraries & GPU pattern:** CUDA 3D stencil kernels for reaction-diffusion PDE, cuSPARSE for FEM tissue mechanics, cuSolver for vascular pressure-flow network, multi-GPU NCCL for coupled physics domains; pattern: patient MRI → tumor/tissue segmentation → multi-physics GPU simulation → synthetic MRI generation → Bayesian parameter assimilation → treatment prediction.

---

### 14.10 GPU-Accelerated Bayesian Inference Engine for Biomedicine 🟡 · Active R&D

- **Deep dive:** Bayesian inference over high-dimensional biomedical models (pharmacokinetic, genetic, epidemiological) requires Markov chain Monte Carlo (MCMC) or variational inference (VI) that is historically slow. GPU-accelerated Hamiltonian Monte Carlo (HMC/NUTS) in NumPyro or PyMC-JAX achieves 10–100× speedup over CPU Stan, enabling inference in population PKPD models with 10⁴ parameters and >10⁶ observations. GPU batch parallelism runs independent MCMC chains simultaneously, and GPU-accelerated gradients via JAX/autograd make HMC feasible for complex ODEs. Clinical trial simulation (tens of thousands of virtual patients) is a key use case.
- **Key algorithms:** Hamiltonian Monte Carlo (HMC) + No-U-Turn Sampler (NUTS), variational inference (ADVI, normalizing flows), sequential Monte Carlo (SMC), population PKPD (NONMEM-equivalent), Gaussian process inference, integrated nested Laplace approximation (INLA).
- **Datasets:** NONMEM Pharmacokinetic Reference Dataset (Holford NHG, verify URL); UK Biobank phenome-wide association studies (https://www.ukbiobank.ac.uk/); OpenFDA Drug Adverse Event database (https://open.fda.gov/apis/drug/event/); CDISC SDTM clinical trial datasets (verify URL via cdisc.org).
- **Starter repos/tools:** NumPyro (https://github.com/pyro-ppl/numpyro) — GPU HMC/NUTS via JAX; PyMC (https://github.com/pymc-devs/pymc) — probabilistic programming with JAX/GPU backend; BlackJAX (https://github.com/blackjax-devs/blackjax) — GPU MCMC kernels in JAX; Stan (https://github.com/stan-dev/stan) — reference Bayesian inference (CPU; GPU via GPU-compatible backend research).
- **CUDA libraries & GPU pattern:** JAX XLA GPU compilation for HMC gradient computation, cuBLAS for covariance matrix operations in GP inference, cuFFT for spectral MCMC methods; pattern: prior + likelihood specification in NumPyro → GPU JIT-compiled HMC kernel → parallel chains on GPU → posterior diagnostics (R-hat, ESS) → posterior predictive check.

---

### 14.11 Differentiable Simulation for Biomedicine 🟡 · Active R&D

- **Deep dive:** Differentiable physics simulators propagate gradients through the entire simulation (FEM, CFD, rigid-body dynamics, particle dynamics), enabling gradient-based optimization of simulator parameters, boundary conditions, or material properties against experimental observations. NVIDIA Warp achieves up to 669× CPU speedup for GPU-differentiable simulation with seamless PyTorch/JAX integration. In biomedicine, differentiable FEM tunes patient-specific tissue stiffness maps by fitting simulated deformation to intraoperative imaging; differentiable CFD optimizes catheter shape to minimize hemolysis; differentiable pharmacokinetic ODE systems fit drug absorption parameters from sparse clinical data. DiffXPBD extends differentiable position-based dynamics to compliant constraint systems.
- **Key algorithms:** Reverse-mode automatic differentiation through simulation (adjoint method), differentiable PBD (DiffXPBD), differentiable FEM (Warp/JAX), differentiable Lagrangian particle dynamics (MPM), physics-informed loss functions, gradient-based material parameter identification.
- **Datasets:** Patient-specific tissue deformation datasets from intraoperative US (Hamlyn); Cardiovascular 4D Flow MRI (HeartFlow); Warp Tutorial Benchmarks (https://github.com/NVIDIA/warp); DeepMind MuJoCo Warp benchmarks (https://github.com/google-deepmind/mujoco).
- **Starter repos/tools:** NVIDIA Warp (https://github.com/NVIDIA/warp) — Python GPU differentiable physics engine (JAX/PyTorch integration); DiffTaichi (https://github.com/taichi-dev/taichi) — differentiable GPU simulation via Taichi lang; JAX MD (https://github.com/google/jax-md) — differentiable molecular dynamics; FEniCSx + UFL (https://github.com/FEniCS/dolfinx) — differentiable FEM (adjoint via dolfin-adjoint).
- **CUDA libraries & GPU pattern:** CUDA with reverse-mode AD (Warp's gradient tape), cuDNN for neural-network coupling in hybrid sim-ML pipelines, Tensor Cores for mixed-precision Jacobian accumulation; pattern: simulation forward pass on GPU → gradient tape records operations → backward pass propagates gradients through PDE/ODE → gradient-based optimizer updates material parameters → iterate.

---

### 14.12 Cross-Modal "Virtual Staining" & Label-Free Imaging 🟡 · Active R&D

- **Deep dive:** Virtual staining uses deep learning to predict H&E, IHC, or other chemical stain images from label-free optical modalities (autofluorescence, quantitative phase, CARS, FTIR), eliminating destructive sample preparation. Pixel super-resolved virtual staining via diffusion models (Nature Communications 2025) achieves pathologist-grade tissue diagnostics from autofluorescence alone on GPU. GPU acceleration is essential: a single whole-slide image (100,000 × 100,000 pixels) requires tiled U-Net inference over ~10,000 patches per slide. Clinical-grade validation of autofluorescence virtual staining for prostate cancer (medRxiv 2024) demonstrates diagnostic equivalence to H&E. The GPU also enables real-time virtual staining during surgery for fresh frozen section replacement.
- **Key algorithms:** U-Net/ViT image translation (pix2pix, CycleGAN, diffusion model), pixel super-resolution (ESRGAN, diffusion), Fourier ptychographic reconstruction, stimulated Raman spectral unmixing on GPU, multi-modal image registration (DRIT++), diffusion-model inversion for unpaired translation.
- **Datasets:** Virtual Staining Dataset (Ozcan Lab, UCLA) — autofluorescence → H&E paired images (verify URL via nature.com supplementary); LCI-PARIS — unstained label-free vs. H&E pairs (verify URL); TCGA Digital Pathology Whole-Slide Images (https://portal.gdc.cancer.gov/); Human Protein Atlas — multimodal tissue images (https://www.proteinatlas.org/).
- **Starter repos/tools:** MONAI (https://github.com/Project-MONAI/MONAI) — GPU medical image segmentation + translation; pix2pix/CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — paired/unpaired GPU image translation; Stable Diffusion (huggingface) fine-tuned for pathology virtual staining; HistoStar (https://github.com/TissueImageAnalytics/tiatoolbox) — GPU whole-slide image analysis toolkit.
- **CUDA libraries & GPU pattern:** cuDNN for U-Net/ViT inference, Tensor Core FP16 for batch patch processing, cuFFT for Fourier ptychographic phase reconstruction; pattern: WSI tiled into 256×256 patches → GPU batch U-Net inference → tile stitching → GPU super-resolution → virtual H&E output for pathologist review.

---

### 14.13 In Silico Organoid Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Organoids — self-organizing 3D stem-cell-derived mini-organs — grow via coupled cell division, differentiation, migration, and mechanical deformation. GPU-accelerated vertex models, cellular Potts models (CPM), and off-lattice agent-based models (ABMs) simulate organoid morphogenesis across thousands to millions of cells. A key bottleneck is computing cell-cell contact forces and sorting energies for CPM (Metropolis Monte Carlo), which are embarrassingly parallel over lattice sites. Virtual tissue simulation from real image data (Frontiers 2024) uses GPU-segmented confocal images to initialize physics-based organoid models, enabling patient-specific drug response prediction for personalized oncology.
- **Key algorithms:** Cellular Potts Model (CPM) Metropolis Monte Carlo, vertex model for epithelial mechanics, off-lattice center-based model (CBM), reaction-diffusion morphogen fields (Turing), subcellular element model (SEM), mechanical feedback on gene regulatory network.
- **Datasets:** Kaggle Sartorius Cell Instance Segmentation (https://www.kaggle.com/c/sartorius-cell-instance-segmentation); OpenCell — protein localization in live cells (https://opencell.czbiohub.org/); CancerOrganoidDB — organoid drug response (verify URL via Hubrecht Institute); NeurIPS Cell Seg Challenge organoid images (verify URL via Grand Challenge).
- **Starter repos/tools:** CompuCell3D (https://compucell3d.org/) — GPU-capable CPM organoid simulation; Morpheus (https://morpheus.gitlab.io/) — GPU cellular Potts + reaction-diffusion; Chaste (https://github.com/Chaste/Chaste) — off-lattice ABM for organoid growth; PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D agent-based multicellular GPU-parallelized simulator.
- **CUDA libraries & GPU pattern:** CUDA checkerboard-parallel Metropolis updates for CPM (even/odd lattice coloring), CUDA reaction-diffusion 3D stencils, cuRAND for Monte Carlo move proposals; pattern: organoid image segmentation → GPU initialization of CPM lattice → parallel Metropolis sweeps (checkerboard coloring avoids conflicts) → reaction-diffusion morphogen update → cell-fate decision → geometry output for imaging comparison.

---

### 14.14 Molecular Machine & Motor Protein Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Molecular machines — kinesin walking on microtubules, ATP synthase rotating, ribosome translating — operate at nanoscale over microsecond-to-millisecond timescales that are far beyond conventional all-atom MD. GPU-accelerated enhanced sampling methods (metadynamics with PLUMED-CUDA, replica-exchange MD, HTMD adaptive sampling) extend the timescale window by orders of magnitude. Coarse-grained (MARTINI, CGMD) simulations on GPU model the full kinesin power stroke in minutes. The cryo-EM structural database provides high-resolution snapshots of machine conformations that seed GPU MD simulations of the mechanical cycle. Understanding motor protein dysfunction underpins treatments for neurodegeneration, cancer, and rare genetic diseases.
- **Key algorithms:** All-atom MD (GROMACS GPU, OpenMM), coarse-grained MD (MARTINI CGMD), metadynamics / funnel metadynamics with PLUMED-CUDA, replica-exchange MD (REMD), accelerated MD (aMD), elastic network model (ENM) for collective modes, Brownian ratchet mechanochemical models.
- **Datasets:** RCSB PDB motor protein structures — kinesin, dynein, myosin, ATP synthase (https://www.rcsb.org/); CHARMM-GUI membrane builder inputs (https://www.charmm-gui.org/); EMDB cryo-EM maps of conformational states (https://www.ebi.ac.uk/emdb/); GPCRdb for GPCR molecular machine models (https://gpcrdb.org/).
- **Starter repos/tools:** GROMACS (https://github.com/gromacs/gromacs) — GPU MD with CUDA/HIP, fastest production MD engine; OpenMM (https://github.com/openmm/openmm) — Python GPU MD with custom force plugins; PLUMED (https://github.com/plumed/plumed2) — GPU-compatible enhanced sampling (metadynamics) CV library; HTMD (https://github.com/Acellera/htmd) — GPU adaptive sampling for protein conformational exploration.
- **CUDA libraries & GPU pattern:** CUDA bonded/non-bonded force kernels (GROMACS native CUDA), cuFFT for PME long-range electrostatics, GPU neighbor-list Verlet scheme; pattern: cryo-EM structure → CHARMM-GUI parameterization → GPU REMD ensemble (N replicas × GPU) → PLUMED metadynamics bias application → free-energy surface reconstruction via WHAM.

---

### 14.15 GPU-Accelerated Neuromorphic Biology 🔴 · Frontier/Theoretical

- **Deep dive:** Biological neural networks (retina, hippocampus, cortex) integrate spiking dynamics across billions of neurons with trillions of synaptic connections, exhibiting emergent phenomena relevant to neurological disease models and brain-computer interfaces. GPU implementations of spiking neural network (SNN) simulators (GeNN, Brian2CUDA) parallelize over neurons and synaptic update rules, achieving ~1000× speedup over CPU NEST for large-scale cortical column models. GPU neuromorphic simulation of Parkinson's basal ganglia circuits tests deep-brain stimulation parameter spaces in silico. Connection with biology: NVIDIA's H100 NVLink GPU cluster serves as a short-term neuromorphic analog for connectome-scale (C. elegans: 302 neurons, Drosophila: 130K neurons) simulation.
- **Key algorithms:** Leaky integrate-and-fire (LIF), Hodgkin-Huxley conductance-based model, spike-timing-dependent plasticity (STDP), GPU event-driven simulation, surrogate gradient training for SNN backpropagation, structural plasticity, large-scale connectome simulation.
- **Datasets:** FlyWire Drosophila Connectome — 130K neuron wiring diagram (https://flywire.ai/); Allen Brain Connectivity Atlas (https://connectivity.brain-map.org/); Blue Brain Project neocortical data (https://bluebrain.epfl.ch/); OpenNeuromorphic benchmark datasets (verify URL via openneuromorphic.org).
- **Starter repos/tools:** GeNN (GPU-enhanced Neuronal Networks) (https://github.com/genn-team/genn) — GPU SNN simulator; Brian2CUDA (https://github.com/brian-team/brian2cuda) — GPU-compiled Brian2 spiking network simulator; PyNN (https://github.com/NeuralEnsemble/PyNN) — SNN abstraction layer; NEURON (GPU branch) (https://github.com/neuronsimulator/nrn) — biophysically detailed neuron simulation with GPU backend.
- **CUDA libraries & GPU pattern:** CUDA warp-level primitives for parallel synaptic weight updates, cuSPARSE for sparse connectivity matrix (connectome), cuRAND for Poisson spike generation; pattern: connectome adjacency matrix (sparse) → GPU spike-event driven propagation → per-neuron LIF/HH ODE integration → STDP weight update → population firing-rate statistics for disease-state comparison.

---

### 14.16 GPU Cellular Automata for Tissue Morphogenesis 🟡 · Active R&D

- **Deep dive:** Lattice-Gas Cellular Automata (LGCA) and Cellular Automata (CA) models simulate tumor invasion, wound healing, and developmental tissue patterning at the cell scale on million-element grids. Every lattice site updates in parallel based on local neighborhood rules — a perfectly SIMT workload. GPU CA for tumor growth integrates nutrient diffusion (CUDA stencil), cell-cycle progression, and proliferation/death rules, enabling parameter sweeps over invasion phenotypes that would be intractable on CPU. Hybrid CA-PDE models couple discrete cell lattice (CUDA) with continuous nutrient/oxygen fields (CUDA finite difference).
- **Key algorithms:** Lattice-Gas CA (LGCA) for cell migration, Cellular Automaton tumor model (Kansal-Torquato), Go-or-Grow phenotype switching, reaction-diffusion PDE for morphogens, Potts model for cell sorting, hybrid CA-FEM multiscale coupling.
- **Datasets:** CancerOrganoid Drug Response Images (verify URL via Hubrecht); TCGA pathology slides for CA calibration (https://portal.gdc.cancer.gov/); CellMorph — time-lapse cell migration datasets (verify URL); Wound-Healing Assay Image Repository (verify URL via protocols.io).
- **Starter repos/tools:** PhysiCell (https://github.com/MathCancer/PhysiCell) — GPU-parallelized 3D agent-based tissue simulator; CompuCell3D (https://compucell3d.org/) — multi-algorithm tissue simulator with GPU support; CancerSim (https://github.com/joancalvente/cancersim) — GPU CA tumor growth code; Morpheus (https://morpheus.gitlab.io/) — spatial cell model simulation with GPU backend.
- **CUDA libraries & GPU pattern:** CUDA 2D/3D stencil kernels for CA lattice update, cuRAND for stochastic cell-fate decisions, Thrust for parallel phenotype census; pattern: N×N×N GPU lattice → one CUDA thread per lattice site → local rule evaluation → stochastic update → reaction-diffusion field update → time-step advance → GPU-rendered morphology export.

---

## Cross-cutting CUDA patterns (how these map to GPU primitives)

- **N-body / pairwise forces** — MD, docking, electrostatics, coarse-grained, FSI particle methods → tiling + shared memory, neighbor/cell lists.
- **Dynamic-programming wavefronts** — sequence alignment, RNA folding, dose deformable matching → anti-diagonal parallelism, DPX instructions (Hopper).
- **Monte Carlo / independent trajectories** — radiation dose, stochastic biochemistry, Bayesian inference, virtual populations → cuRAND, massive parallel replicas, reductions.
- **FFT-based methods** — PME electrostatics, MRI/CT/PET reconstruction, ultrasound, cross-correlation docking → cuFFT, NUFFT.
- **Sparse linear algebra & PDE solves** — FEM, CFD, electrophysiology, Poisson-Boltzmann, networks → cuSPARSE, cuSOLVER, algebraic multigrid, conjugate gradient.
- **Dense linear algebra** — GWAS, normal modes, quantum chemistry, fMRI connectivity → cuBLAS, cuSOLVER, cuTENSOR.
- **Deep learning** — imaging, genomics, generative chemistry/structure, foundation models → cuDNN, cuTENSOR, TensorRT, NCCL, FlashAttention.
- **Graph algorithms** — assembly, connectomics, pangenomes, contact-network epidemics, polypharmacology → cuGraph, irregular-memory load balancing.
- **Image/voxel stencils** — reconstruction, segmentation, registration, dose superposition → texture memory, 3D tiling.
- **Stiff ODE ensembles** — systems biology, PK/PD, QSP, cardiac ionic models → parallel-across-replicates integrators (e.g., MPGOS, DiffEqGPU patterns).

## Core CUDA libraries & domain engines to lean on

**Math/primitives:** cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND, Thrust, CUB, cuTENSOR.
**Deep learning:** cuDNN, TensorRT, NCCL (multi-GPU), CUDA Graphs.
**Data:** RAPIDS (cuDF/cuML/cuGraph), CuPy.
**Graphics:** OptiX, CUDA-OpenGL/Vulkan interop.
**Quantum:** cuQuantum.
**Domain engines already exposing CUDA:** OpenMM, AMBER, GROMACS, NAMD, LAMMPS (MD); AutoDock-GPU, Uni-Dock, gnina (docking); RELION, cryoSPARC, WARP (cryo-EM); Parabricks (genomics); Dorado (basecalling); RAPIDS-singlecell (scRNA-seq); MONAI, RTK, ASTRA, SIRF (medical imaging); openCARP, MONAI Stream, SOFA/FEBio (physiology/FEA).

## How to pick a first project

Start where the GPU win is unambiguous and the reference material is thick: a 🟢 *Established* project with a named starter repo. Reproduce the repo's baseline, then profile it with Nsight Systems/Compute to find the dominant kernel, and optimize that one kernel. Once comfortable, climb to a 🟡 *Active R&D* project that solves a real problem you care about, and only then attempt a 🔴 *Frontier* item — where the contribution is often the GPU formulation itself.

*Difficulty and maturity tags are guides, not guarantees: many "established" domains still hold open research problems, and several "frontier" items already have early open-source starting points.*
