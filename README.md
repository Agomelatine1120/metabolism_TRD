# TRD metabolism analysis workflow

This repository contains the pseudonymized inputs, ordered notebooks,
statistical outputs, and figure components needed to rerun the study analyses
of regional cerebral metabolism, metabolic connectivity, and
metabolism-weighted centrality in non-treatment-resistant depression (non-TRD),
treatment-resistant depression (TRD), and longitudinal brain-stimulation data.

## Analysis framework

The workflow combines FDG-PET and resting-state fMRI at three levels:

1. **Regional metabolism:** whole-brain-normalized FDG uptake (SUVR).
2. **Inter-regional metabolic connectivity:** individualized deviation from a
   healthy reference network, estimated with a reference-and-perturbed-network
   approach.
3. **Metabolism-weighted centrality (MwC):** regional centrality derived from
   GCMI functional dependence, normative structural-connectivity constraints,
   and the metabolism of connected regions.

All measures are expressed relative to a healthy-control reference. The formal
analysis space contains **12 cortical and 14 limbic ROIs**, including bilateral
subgenual cingulate cortex (SGC). This produces **26 regional/MwC features and
168 cortical-to-limbic metabolic-connectivity edges**. GCMI matrices are used
only to calculate MwC.

## Analysis samples

The dataset contains 187 analyzed participants:

| Cohort | Analysis role | n |
|---|---|---:|
| Healthy controls | Healthy reference | 23 |
| non-TRD | Clinical comparison group | 25 |
| TRD1 | Primary TRD group analysis | 35 |
| TRD2 | Independent spatial comparison and stimulation cohort | 104 |

Baseline non-TRD-versus-TRD1 models adjust for age and sex. TRD1
and TRD2 deviation profiles are compared using spatial Pearson correlation,
directional agreement, and a magnitude-weighted signed Dice coefficient.

The longitudinal dataset contains **34 rTMS, 37 iTBS, and 33 sham** paired
participants. Within-arm changes are assessed with paired-sample t-tests. The
longitudinal plotting step programmatically displays all, and only, active-arm
metabolic-connectivity edges meeting **paired `p < .01`**. Active-versus-sham
group-by-time models adjust for age and sex. Clinical associations
use partial Spearman correlations between post-minus-pre imaging change and
pre-minus-post HDRS-17 improvement, adjusting for age and sex.

## Analyses included

The workflow performs the following operations:

1. Loads the minimized clinical, PET, GCMI, atlas, ROI-definition, and
   normative structural-connectivity inputs.
2. Constructs healthy-control references for regional SUVR, individualized
   metabolic connectivity, and MwC.
3. Calculates participant-level deviations for the 26 regional/MwC features
   and 168 cortical-to-limbic metabolic-connectivity edges.
4. Runs baseline non-TRD-versus-TRD1 models and TRD1–TRD2 spatial comparisons.
5. Calculates five participant-level global metabolic indices.
6. Runs paired pre-to-post analyses for rTMS, iTBS, and sham;
   active-versus-sham group-by-time models; and HDRS-17 partial Spearman
   analyses.
7. Repeats the baseline group comparison with baseline HDRS-17 included as an
   additional covariate.
8. Generates statistical workbooks, plotting data, heatmaps, circle diagrams,
   raincloud plots, correlation plots, and Nilearn glass-brain components.

Numerical results are stored in the generated workbooks under `outputs/`.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/` | Five ordered MwC notebooks, ten ordered Main notebooks, and one complete runner |
| `pipeline_steps/` | Canonical Python source corresponding to each notebook step |
| `scripts/` | Supporting calculations and figure generation |
| `inputs/` | Minimized pseudonymized inputs restricted to the locked analysis population |
| `outputs/mwc/` | MwC values, statistics, spatial comparisons, and figure components |
| `outputs/main_analyses/` | Regional, connectivity, longitudinal, global-index, sensitivity, and figure outputs |
| `docs/` | Workflow, data dictionary, analysis map, and analysis-output mapping |
| `manifests/` | File inventories, checksums, and machine-readable validation results |

For a detailed analysis-to-output mapping, see
[`docs/ANALYSIS_OUTPUTS.md`](docs/ANALYSIS_OUTPUTS.md). For the precise execution
order, see [`docs/NOTEBOOK_WORKFLOW.md`](docs/NOTEBOOK_WORKFLOW.md).

## Input package and population lock

Participant-level inputs are restricted before analysis to the prespecified
roster. The workflow does not load a larger participant pool and then remove
unused participants downstream.

- `inputs/core/analysis_population_roster.csv` defines every permitted public
  participant/session record.
- `inputs/core/clinical_data.xlsx` contains exactly 187 analyzed participants.
- `inputs/core/AAL3_PET_wbSUVR_Table.xlsx` contains 291 permitted PET sessions
  across 144 retained AAL3 regions: 187 baseline and 104 post-treatment.
- `inputs/mwc/normative_sc/session_gcmi_pet_AAL3_144.npz` contains the same
  291 permitted PET/GCMI sessions. GCMI matrices were derived from corrected
  BOLD signals without individual grey-matter masking.
- Longitudinal inputs contain 34 rTMS, 37 iTBS, and 33 sham participants with
  paired PET and fMRI data.

All cross-file identifiers use public keys (`P001`–`P187`). Direct identifiers,
unused participants, and unneeded clinical variables are not included. These
data are pseudonymized, not anonymous, and the repository should remain private
unless sharing has been separately authorized.

## Installation

Python 3.11 or later is recommended. Git LFS is required for the configured
binary inputs and outputs.

```bash
git lfs install
git clone https://github.com/Agomelatine1120/metabolism_TRD.git
cd metabolism_TRD
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the complete workflow

Start Jupyter from the repository root and run
`notebooks/00_run_all_steps.ipynb`. The same workflow can be executed from the
command line:

```bash
export METAFL_ROOT="$PWD"
jupyter nbconvert --to notebook --execute notebooks/00_run_all_steps.ipynb \
  --inplace \
  --ExecutePreprocessor.timeout=10800
```

The runner executes MwC 01–05 first and Main 01–10 second. The 15 chapter
notebooks are committed with execution counts, numerical outputs, and compact
table-and-figure previews at the bottom so their results are visible directly on GitHub. Full-resolution
PNG, PDF, and SVG components and their plotting workbooks are stored under
`outputs/`.

Final multi-panel sizing, panel lettering, and manually positioned anatomical
arrows are layout operations and are not part of the numerical workflow.

## Step guide

| Analysis task | Step |
|---|---|
| Cohort and treatment-arm characteristics | Main 02 |
| Regional SUVR, connectivity, and MwC group analyses | Main 03; MwC 02–05 |
| TRD1–TRD2 spatial comparisons | Main 06; MwC 03 |
| Regional/connectivity/MwC heatmaps and raincloud plots | Main 07; MwC 03–05 |
| Cross-sectional and longitudinal circle diagrams | Main 08 and Main 10 |
| Five global metabolic indices | Main 09 |
| Paired stimulation effects, group-by-time models, and HDRS-17 associations | Main 04–07 |
| Baseline-HDRS-17-adjusted sensitivity analyses | Main 03, 07, and 10 |
| Demographic and statistical summary tables | Main 02–05 and MwC 04–05 |

## Validation

`manifests/validation.json` records the input roster, model definitions,
significance counts, dynamic longitudinal selection, notebook execution state,
and deidentification checks. File inventories and SHA-256 checksums are stored
in the same directory.
