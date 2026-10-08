#!/usr/bin/env bash
# Run all experiments. Prints to stdout; intermediate
# files go to ../outputs/ (about 20 MB for the 20 simulated replicates).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" demographics.py                 # dataset summary statistics, calibration targets
"$PY" triples.py                      # h, i10 and the four triples
"$PY" sweeps.py                       # beta sweeps
"$PY" decisiveness.py                 # tie rates
"$PY" g_index.py                      # g-index and g^2 / C
"$PY" calibrate_sim.py                # re-derives the simulation parameters in simulate.py
"$PY" simulate.py --replicates 20     # 20 replicates of 100 simulated careers
"$PY" simulation_cis.py               # longitudinal metrics with 95% CIs
