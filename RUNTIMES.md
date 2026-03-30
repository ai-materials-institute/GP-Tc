# Runtime Notes

This document records observed wall-clock runtimes for key `GP-Tc` setup and inference steps on a local development machine.

## Benchmark Input

- CIF file: `data/Nd0.8Sr0.2NiO2_synth_doped.cif`
- Prediction command:

```bash
/usr/bin/time -p .venv/bin/python src/predict_single_cif.py data/Nd0.8Sr0.2NiO2_synth_doped.cif
```

## Measured Runtimes

### Fresh Environment Install

A fresh local environment install from `pyproject.toml` and `uv.lock` completed in:

```text
real 9.29 s
user 3.44 s
sys 5.94 s
```

Command used:

```bash
/usr/bin/time -p env UV_CACHE_DIR=.uv-cache uv sync
```

### Dependency Sync

After adding the missing `xlrd` dependency required for reading `config/Space_group.xls`, the incremental environment sync completed in:

```text
real 3.44 s
user 1.20 s
sys 0.69 s
```

Command used:

```bash
/usr/bin/time -p env UV_CACHE_DIR=.uv-cache uv sync
```

### Single-CIF Prediction

Observed runtime for single-CIF inference:

```text
real 273.40 s
user 517.76 s
sys 140.46 s
```

Prediction output:

```text
Formula:               Sr0.2Nd0.8Ni1O2
Probability:           0.6203
Classification std:    0.4853
Predicted Tc:          15.82 K
Regression std:        4.78 K
```

## Notes

- These measurements reflect one local run and should be treated as approximate.
- First-run overhead can include Python import and model-loading costs.
- Runtime will vary with hardware, Python environment, filesystem speed, and whether caches are already warm.
