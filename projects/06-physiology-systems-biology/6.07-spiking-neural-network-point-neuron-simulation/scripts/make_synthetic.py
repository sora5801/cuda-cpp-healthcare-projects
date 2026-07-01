#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic SNN sample network
# ---------------------------------------------------------------------------
# Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
#
# WHY THIS EXISTS
#   The demo must run offline, so we ship a tiny, CLEARLY-SYNTHETIC network
#   config. There is no patient data here at all: an SNN "dataset" is just the
#   network's structural parameters (sizes, connection density, weights) plus the
#   neuron biophysics. Real structural connectomes (Human Connectome Project) or
#   in-vivo spike trains (Allen Brain Observatory, DANDI) are pointed to in
#   data/README.md and scripts/download_data.*, but they are NOT needed to learn
#   or verify the simulator -- this synthetic Brunel-style network exercises every
#   code path (leak, threshold, refractory, excitatory & inhibitory synapses,
#   the sparse atomic scatter) deterministically.
#
#   The output is a 4-line text file the C++ loader (reference_cpu.cpp::load_network)
#   parses. It is deterministic given the parameters, so demo/expected_output.txt
#   is stable.
#
# FILE FORMAT (whitespace-separated; see data/README.md for field meanings):
#   line 1:  n_exc n_inh out_degree
#   line 2:  w_exc w_inh ext_kick ext_every
#   line 3:  steps seed
#   line 4:  v_rest v_reset v_thresh tau_m tau_syn r_m refractory_ms dt
#
# USAGE
#   python scripts/make_synthetic.py                  # writes data/sample/network.txt
#   python scripts/make_synthetic.py --n-exc 800 --n-inh 200 --steps 500
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the project folder
OUT = ROOT / "data" / "sample" / "network.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic SNN network config.")
    # --- Network structure (Brunel-style excitatory/inhibitory balance) -------
    ap.add_argument("--n-exc", type=int, default=160, help="number of excitatory neurons")
    ap.add_argument("--n-inh", type=int, default=40, help="number of inhibitory neurons")
    ap.add_argument("--out-degree", type=int, default=16,
                    help="synapses per source neuron (sparse: << total neurons)")
    # --- Synaptic weights (drive units; inhibitory is negative) ---------------
    #   Defaults chosen so the demo is a BALANCED network: recurrent excitation
    #   noticeably recruits extra spikes over the external drive alone, and
    #   inhibition reins it back in (removing inhibition ~doubles activity). This
    #   is the classic excitation/inhibition balance the demo is meant to show.
    ap.add_argument("--w-exc", type=float, default=0.90,
                    help="excitatory synaptic weight (depolarizing)")
    ap.add_argument("--w-inh", type=float, default=-2.20,
                    help="inhibitory synaptic weight (hyperpolarizing; stronger than exc)")
    # --- Deterministic background drive (stand-in for Poisson input) ----------
    #   ext_kick is strong enough to reliably fire the driven subset (the seed
    #   spikes); ext_every=30 means ~1/30 of neurons are driven on any given step.
    ap.add_argument("--ext-kick", type=float, default=1.80,
                    help="fixed external drive delivered to a rotating subset each step")
    ap.add_argument("--ext-every", type=int, default=30,
                    help="on step t, neurons with id %% ext_every == t %% ext_every get the kick")
    # --- Integration ----------------------------------------------------------
    ap.add_argument("--steps", type=int, default=500, help="number of timesteps")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for wiring + init jitter")
    # --- LIF biophysics (mV / ms) --------------------------------------------
    ap.add_argument("--v-rest", type=float, default=-65.0)
    ap.add_argument("--v-reset", type=float, default=-65.0)
    ap.add_argument("--v-thresh", type=float, default=-50.0)
    ap.add_argument("--tau-m", type=float, default=20.0)
    ap.add_argument("--tau-syn", type=float, default=5.0)
    ap.add_argument("--r-m", type=float, default=10.0)
    ap.add_argument("--refractory-ms", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    lines = [
        f"{args.n_exc} {args.n_inh} {args.out_degree}",
        f"{args.w_exc:g} {args.w_inh:g} {args.ext_kick:g} {args.ext_every}",
        f"{args.steps} {args.seed}",
        (f"{args.v_rest:g} {args.v_reset:g} {args.v_thresh:g} {args.tau_m:g} "
         f"{args.tau_syn:g} {args.r_m:g} {args.refractory_ms:g} {args.dt:g}"),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = args.n_exc + args.n_inh
    print(f"[make_synthetic] wrote {args.out}  "
          f"({n} neurons, {n * args.out_degree} synapses, {args.steps} steps; SYNTHETIC)")


if __name__ == "__main__":
    main()
