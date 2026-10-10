# Metabolism in treatment-resistant depression

This repository contains pseudonymized inputs, executable notebooks, statistical
tables, and figure components for analyses of regional cerebral metabolism,
metabolic connectivity, and metabolism-weighted centrality (MwC) in depression.

## Analysis design

The formal circuit comprises 12 frontal and 14 limbic regions. Regional SUVR
and MwC therefore contain 26 features, while the frontal-to-limbic Cartesian
product contains 168 metabolic-connectivity edges.

MwC is calculated from three inputs: generalized canonical-correlation mutual
information (GCMI) estimated from corrected resting-state BOLD signals,
whole-brain-normalized FDG-PET SUVR, and a fixed top-10% normative structural
connectivity mask. The computation uses a 140-region AAL3 space. Four regions
with unavailable source measurements are excluded before calculation:
`Thal_VA_L`, `Thal_VA_R`, `Thal_LGN_L`, and `Thal_LGN_R`. Missing MwC values
are retained as missing; no imputation is performed.

## Samples and statistical models

| Cohort or arm | Role | n |
|---|---|---:|
| Healthy controls | Healthy reference | 23 |
| non-TRD | Baseline comparison | 25 |
| TRD1 | Baseline comparison | 35 |
| rTMS | TRD2 PET pairs | 38 |
| iTBS | TRD2 PET pairs | 43 |
| sham | TRD2 PET pairs | 38 |

The PET input uses GM > 0.5 and contains 119 paired TRD2 participants. GCMI
matrices were derived from corrected BOLD signals without individual
gray-matter masking. The MwC candidate sample contains 36 rTMS, 39 iTBS, and
35 sham participants. Featurewise complete-pair counts are 34–36, 37–39, and
32–35, respectively.

Healthy-reference deviations and between-group models adjust for age and sex.
Within-arm pre–post tests are paired t-tests without covariates. Active-versus-
sham change models and imaging-change/HDRS-17 partial Spearman correlations
adjust for age and sex. Baseline sensitivity models additionally adjust for
baseline HDRS-17. Sensitivity statistics are included as tables; no separate
sensitivity figure is generated.

## Repository contents

| Path | Contents |
|---|---|
| `inputs/clinical_data.csv` | One pseudonymized row per included participant |
| `inputs/session_pet_gcmi_AAL3_140.npz` | Session metadata, 140-region PET vectors, and GCMI matrices |
| `inputs/normative_SC_top10_mask.npz` | Fixed 140-region normative structural mask |
| `inputs/corticolimbic_12x14_ROI_definition.csv` | Formal 26-region circuit definition |
| `analysis/` | Reusable analysis and figure-generation code |
| `notebooks/` | Ten Main steps and six MwC steps, saved with inline results |
| `outputs/tables/` | Full-precision statistical results |
| `outputs/figures/` | Standalone analysis figures in PNG and PDF formats |

All participant identifiers are study-independent keys (`H###`, `N###`,
`T###`, and `L###`). The inputs contain only participants and variables used by
the analyses in this repository.

## Installation and execution

Python 3.11 or later and Git LFS are recommended.

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

The notebooks may also be opened and run in numerical order. Each committed
notebook contains its executed tables and figures directly below the relevant
cells.
