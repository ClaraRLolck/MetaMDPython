# MetaMDPython

`MetaMDPython` runs molecular-dynamics-based metadynamics searches for reactions from a SMILES input. It relaxes an initial structure, runs biased dynamics, identifies and records structures/reactions, and can aggregate results from repeated runs. The workflow is written for the chemistry group's SLURM cluster and its installed computational chemistry software.

## Contents

- `metaMD.py` - single-run simulation driver and reaction/structure analysis.
- `control_metadyn_runs.py` - generates and submits repeated SLURM jobs from a CSV of SMILES.
- `combine_runs.py` - combines per-run reaction data and reports reaction counts and step statistics.
- `collect_products.py` - collects XYZ structures for products found in a reaction summary.
- `RMSD_opt.py` - RMSD optimization helper imported by `metaMD.py`.
- `xyz2mol_local.py` - local XYZ-to-molecule conversion implementation used by the workflow.

## Requirements and environment

The scripts do not provide a standalone dependency installer or environment file. The simulation imports Python packages including NumPy, pandas, RDKit, ASE, tblite, and FairChem (`fairchem.core`). The local `RMSD_opt.py` helper is included. `metaMD.py` imports a module named `xyz2mol`, while this directory contains `xyz2mol_local.py`; make that implementation importable as `xyz2mol` in the runtime environment.

The selected calculator also requires its external software and model resources. The default method is `g-xTB`, which expects the `xtb` executable; the `UMA` method uses FairChem's pretrained model. 

## Input format

The batch controller reads a CSV with an index column and a `smiles` column, for example:

```csv
,smiles
0,CCO
1,C=O
```

The index values become reaction/run directory names. SMILES are passed to the simulation as command-line arguments, so use values that survive shell quoting in the generated job script.


## Run one simulation

From a prepared run directory, the driver accepts ten positional arguments:

```text
python metaMD.py <smiles_index> <run_number> <smiles> <scale_factor> <time_ps> <hill_push> <alpha> <random_seed> <method> <with_products>
```

For example, the batch script invokes it in this argument order:

```text
python metaMD.py 0 0 'CCO' 0.8 5 0.05 0.3 12345 g-xTB False
```

`method` currently selects `g-xTB` or `UMA`. `with_products` must be the literal `True` or `False`. A single run creates `run<run_number>/` and writes its reaction dataframe as `dataframe.pkl` in the parent working directory. The run directory also contains a structure database, XYZ files, trajectory/analysis artifacts, and `timing.txt` when generated during execution.

## Submit repeated runs

`control_metadyn_runs.py` takes the input CSV as its only command-line argument:

```text
python control_metadyn_runs.py molecules.csv
```

The run count and simulation parameters are constants in its `__main__` block (`N_RUNS`, `S_FACTOR`, `TIME_PS`, `K_PUSH`, and `ALP`). It creates a parameter-named output directory, writes one SLURM script per run, submits jobs with `sbatch`, and limits submissions using `MAX_QUEUE`. Set `SCRIPT`, the SLURM partition/account/queue handling, memory/CPU settings, and scratch paths for your environment. The controller expects to be launched from the directory where outputs should be stored.

## Combine and collect results

After jobs finish and their `run<number>.pkl` files have been placed in a directory named for each input index, combine a reaction's runs with:

```text
python combine_runs.py molecules.csv
```

The script currently expects `N_RUNS = 100`. For each input index it reads `run0.pkl`, `run1.pkl`, and so on from `<index>/`, then writes `<index>_combined.csv` and `<index>_1step.csv` in that directory. The combined file groups identical reactions and counts occurrences; the one-step file includes forward reactions from the starting SMILES and reverse reactions recast as forward entries. The script also calculates mean and median first-reaction steps (including a statistic filtered to steps above 100), but does not currently save its final summary dataframe to disk.

To extract XYZ structures for products listed in an index's one-step CSV, run from the directory containing the index folder:

```text
python collect_products.py <index>
```

The collector reads `<index>/<index>_1step.csv` and each run's `<index>/run<number>_database.tar.gz`, then appends structures to `initial_structures.xyz` in the current directory. It defaults to checking 100 runs.

## Notes

- These scripts are research workflow code, not a packaged command-line application. Paths, cluster assumptions, run counts, and environment setup are currently configured in source files.
- Check that each job's expected output files are copied back successfully before combining runs; missing or failed run pickles are skipped by `combine_runs.py`.
