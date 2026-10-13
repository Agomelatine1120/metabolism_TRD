# Metabolism in treatment-resistant depression

This repository contains the deidentified inputs, analysis notebooks, result
tables, and figure components for regional SUVR, metabolic connectivity, and
metabolism-weighted centrality (MwC) analyses.

## Participants

| Group | n | Assessment |
|---|---:|---|
| Healthy controls | 23 | Baseline |
| non-TRD | 25 | Baseline |
| TRD1 | 35 | Baseline |
| TRD2 rTMS | 38 | Pre and post |
| TRD2 iTBS | 43 | Pre and post |
| TRD2 sham | 38 | Pre and post |

All participant IDs are deidentified. The repository includes only the data
used in the analyses.
Reported p values are uncorrected.

## Contents

| Path | Contents |
|---|---|
| `inputs/` | Deidentified analysis inputs |
| `analysis/` | Analysis and figure-generation code |
| `notebooks/` | Ten Main steps and six MwC steps with saved results |
| `outputs/tables/` | Statistical results |
| `outputs/figures/` | Figure components in PNG and PDF formats |

## Run

```bash
git lfs install
git clone https://github.com/Agomelatine1120/metabolism_TRD.git
cd metabolism_TRD
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/run_all.py
python analysis/figures.py
```
