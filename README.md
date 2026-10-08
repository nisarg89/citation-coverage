# Why Leave Citations on the Table? — code and data

Code and data for *Why Leave Citations on the Table?* (Nisarg Shah and Vijay Vazirani).

The repository has three parts:

```
build/      code that sources the dataset (queries live APIs)
data/       the dataset and its format
scripts/    code that runs the experiments
```

## Requirements

Python 3.9+. The experiments need `numpy` and `scipy`; the dataset-sourcing code also needs `requests`.

```bash
pip install -r requirements.txt
```

## Sourcing the data

See [`build/README.md`](build/README.md). In short:

```bash
export OPENALEX_EMAIL=you@example.org
cd build
python 01_select_researchers.py
python 02_match_profiles.py
python 03_fetch_citation_arrays.py
python 04_export_dataset.py
```

The scripts query live services, so a re-run reflects their current contents.

## Dataset format

See [`data/README.md`](data/README.md). `data/citation_arrays.json` is a list of records, one per researcher:

```json
[{"author_id": "https://openalex.org/A5068811427", "citation_array": [3494, 531, 515, ...]}, ...]
```

## Running the experiments

```bash
scripts/run_all.sh
```

This runs every script below in order, prints results to stdout, and writes intermediate files to `outputs/`. It takes about 8 minutes; the calibration search and the 20 simulated replicates dominate. Use `PYTHON=/path/to/python scripts/run_all.sh` to select an interpreter.

| script | what it does |
|---|---|
| `demographics.py` | summary statistics and histograms of the dataset |
| `triples.py` | coverage, gains, seniority bins, Kendall τ and IoU for each index |
| `sweeps.py` | sweeps of the power and multiplier families over β grids |
| `decisiveness.py` | tie rates between researchers |
| `g_index.py` | g-index and g²/C |
| `calibrate_sim.py` | randomized search for the simulation parameters (`--trials N`, default 400) |
| `simulate.py` | simulates careers; `--replicates R` (default 20); replicate `r` uses seed `42 + r` |
| `simulation_cis.py` | longitudinal metrics over the simulated replicates, with 95% confidence intervals |

`simulation_cis.py` reads the replicates written by `simulate.py`. `indices.py` defines the indices and the coverage computation and is shared by all scripts; `data.py` loads the dataset.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`data/LICENSE.md`).
