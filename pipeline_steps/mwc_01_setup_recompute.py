"""Auto-extracted pipeline step: MwC 01 — setup and AAL3-144 recomputation.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 2
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import Image, Markdown, display
from matplotlib.patches import Patch
from scipy import stats

SEED = 20260903
PET_SCALE = 100.0
N_ROIS = 26
N_ANALYSIS_NODES = 144

BASE = Path(os.environ.get("METAFL_ROOT", Path.cwd())).resolve()
INPUT_ROOT = BASE / "inputs"
MWC_INPUT_ROOT = INPUT_ROOT / "mwc"
NORMATIVE_ROOT = MWC_INPUT_ROOT / "normative_sc"
OUTPUT_DIR = BASE / "outputs" / "mwc"
FIGURE_DIR = OUTPUT_DIR / "figures"
MWC_RECOMPUTED_ROOT = OUTPUT_DIR / "recomputed_normative_sc"
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
FIGURE_DIR.mkdir(parents=True)
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
np.random.seed(SEED)

MWC_CALCULATOR = BASE / "scripts" / "03_compute_normative_sc_mwc.py"
MWC_CALCULATION_INPUT_FILE = NORMATIVE_ROOT / "session_gcmi_pet_AAL3_144.npz"
MWC_MASK_FILE = NORMATIVE_ROOT / "normative_SC_top10_mask.npz"
MWC_REFERENCE_WIDE_FILE = NORMATIVE_ROOT / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv"
MWC_REFERENCE_FULL_FILE = NORMATIVE_ROOT / "subject_level_AAL3_144_normativeSC10_MwC_wide.csv.gz"
MWC_WIDE_FILE = MWC_RECOMPUTED_ROOT / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv"
MWC_FULL_FILE = MWC_RECOMPUTED_ROOT / "subject_level_AAL3_144_normativeSC10_MwC_wide.csv.gz"
MWC_CALCULATION_VALIDATION_FILE = MWC_RECOMPUTED_ROOT / "calculation_validation_report.json"
AAL_VALIDATION_FILE = NORMATIVE_ROOT / "validation_report.json"
ROI_FILE = INPUT_ROOT / "core" / "corticolimbic_12x14_ROI_definition.csv"
CLINICAL_FILE = INPUT_ROOT / "core" / "clinical_data.xlsx"
MWC_ADJUSTED_26_FILE = (
    NORMATIVE_ROOT / "age_sex_adjusted" /
    "subject_level_corticolimbic26_age_sex_adjusted_normativeSC10_MwC_wide.csv"
)
MWC_DEVIATION_HEATMAP = FIGURE_DIR / "Figure_nTRD_TRD1_TRD2_normativeSC_MwC_26ROI_deviation_heatmap.png"
MWC_SPATIAL_FIGURE = FIGURE_DIR / "Figure_TRD1_TRD2_normativeSC_MwC_26ROI_spatial_correlation.png"
MWC_SPATIAL_SUMMARY_FILE = OUTPUT_DIR / "TRD1_TRD2_normativeSC_MwC_26ROI_spatial_summary.csv"

required_files = [
    MWC_CALCULATOR,
    MWC_CALCULATION_INPUT_FILE,
    MWC_MASK_FILE, MWC_REFERENCE_WIDE_FILE, MWC_REFERENCE_FULL_FILE,
    AAL_VALIDATION_FILE, ROI_FILE, CLINICAL_FILE, MWC_ADJUSTED_26_FILE,
]
missing = [str(input_path) for input_path in required_files if not input_path.exists()]
if missing:
    raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
display(pd.DataFrame({
    "input": [input_path.name for input_path in required_files],
    "size_kib": [round(input_path.stat().st_size / 1024, 1) for input_path in required_files],
}))


# Source cell 4
subprocess.run(
    [sys.executable, str(MWC_CALCULATOR), "--base", str(BASE)],
    check=True,
)
with MWC_CALCULATION_VALIDATION_FILE.open(encoding="utf-8") as stream:
    mwc_calculation_validation = json.load(stream)
assert mwc_calculation_validation["status"] == "PASS"
required_calculation_checks = [
    "sessions_255",
    "retained_AAL3_nodes_144",
    "fixed_dTOR_normative_SC_edges_1030",
    "gcmi_floor_bits_0p01",
    "row_normalization_error_below_1e_12",
    "all_144_nodes_computed_before_26_extraction",
    "isolated_nodes_remain_missing",
    "full_144_matches_validated_reference",
    "selected_26_matches_validated_reference",
]
assert all(mwc_calculation_validation["checks"][key] for key in required_calculation_checks)
display(pd.DataFrame([
    {"check": key, "passed": mwc_calculation_validation["checks"][key]}
    for key in required_calculation_checks
]))
