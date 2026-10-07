"""Auto-extracted pipeline step: Main 01 — setup, input loading, and validation.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 1
from pathlib import Path
import os
import json
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 180)

BASE = Path(os.environ.get("METAFL_ROOT", Path.cwd())).resolve()
INPUT_ROOT = BASE / "inputs"
PET_FILE = INPUT_ROOT / "core" / "AAL3_PET_wbSUVR_Table.xlsx"
CLINICAL_FILE = INPUT_ROOT / "core" / "clinical_data.xlsx"
ROI_FILE = INPUT_ROOT / "core" / "corticolimbic_12x14_ROI_definition.csv"
MWC_INPUT_FILE = (
    INPUT_ROOT / "mwc" / "normative_sc"
    / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv"
)
MWC_VALIDATION_FILE = INPUT_ROOT / "mwc" / "normative_sc" / "validation_report.json"
MWC_ADJUSTED_INPUT_ROOT = Path(os.environ.get(
    "METAFL_ADJUSTED_MWC_INPUT_ROOT",
    INPUT_ROOT / "mwc" / "normative_sc" / "age_sex_adjusted",
)).resolve()
MWC_ADJUSTED_WIDE_FILE = (
    MWC_ADJUSTED_INPUT_ROOT
    / "subject_level_corticolimbic26_age_sex_adjusted_normativeSC10_MwC_wide.csv"
)
MWC_BASELINE_HC3_FILE = MWC_ADJUSTED_INPUT_ROOT / "baseline_nTRD_vs_TRD1_age_sex_adjusted_MwC_HC3_all26.csv"
MWC_NORMATIVE_PARAMETERS_FILE = MWC_ADJUSTED_INPUT_ROOT / "HC_age_sex_normative_MwC_parameters.csv"
MWC_DEVIATION_INPUT_FILE = MWC_ADJUSTED_INPUT_ROOT / "subject_level_nTRD_TRD1_age_sex_adjusted_MwC_deviation.csv"
MWC_ADJUSTED_VALIDATION_FILE = MWC_ADJUSTED_INPUT_ROOT / "validation_report.json"
MWC_ANALYSIS_SCRIPT = BASE / "scripts" / "03_analyze_age_sex_adjusted_mwc.py"
OUTPUT_ROOT = BASE / "outputs" / "main_analyses"
RESET_OWNED_OUTPUTS = os.environ.get("METAFL_RESET_OUTPUTS", "1") == "1"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))

required_inputs = [
    PET_FILE, CLINICAL_FILE, ROI_FILE, MWC_INPUT_FILE, MWC_VALIDATION_FILE,
    MWC_ADJUSTED_WIDE_FILE, MWC_NORMATIVE_PARAMETERS_FILE,
    MWC_BASELINE_HC3_FILE, MWC_DEVIATION_INPUT_FILE,
    MWC_ADJUSTED_VALIDATION_FILE, MWC_ANALYSIS_SCRIPT,
]
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
if missing_inputs:
    raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing_inputs))

mwc_validation = json.loads(MWC_VALIDATION_FILE.read_text(encoding="utf-8"))
mwc_checks = mwc_validation.get("checks", {})
assert mwc_validation.get("status") == "PASS"
assert mwc_checks.get("retained_nodes_144") is True
assert mwc_checks.get("fixed_structural_edges_1030") is True

# Reset only directories owned by this notebook.
OWNED_OUTPUT_DIRS = [
    "01_TRD1_discovery_agesex",
    "02_TRD2_deviation_prepost_agesex",
    "03_TRD1_TRD2_spatial_correlation",
    "04_figures",
    "05_active_vs_sham_interaction",
    "08_TRD1_vs_nTRD_HDRS_sensitivity",
    "06_active_paired_complete_results",
    "07_age_sex_adjusted_MwC",
    "10_metabolic_signature_five_metrics",
]
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
if RESET_OWNED_OUTPUTS:
    for name in OWNED_OUTPUT_DIRS:
        path = OUTPUT_ROOT / name
        if path.exists():
            shutil.rmtree(path)
for name in OWNED_OUTPUT_DIRS:
    (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)

print("MetaFL package root:", BASE)
print("Main-analysis output root:", OUTPUT_ROOT)


# Source cell 3
GROUP_P_THRESHOLD = 0.01
PREPOST_P_THRESHOLD = 0.05
LONGITUDINAL_FIGURE_P_THRESHOLD = 0.01
CLINICAL_P_THRESHOLD = 0.05
FDR_THRESHOLD = 0.05
EXPECTED_COUNTS = {"HC": 23, "nTRD": 25, "TRD1": 35, "TRD2": 85}
EXPECTED_ARMS = {"iTBS": 31, "rTMS": 28, "sham": 26}
EXPECTED_PAIRED_ARMS = {"iTBS": 31, "rTMS": 28, "sham": 26}

roi_definition = pd.read_csv(ROI_FILE)
CORTICAL_IDS = roi_definition.loc[roi_definition["network"] == "cortical", "AAL3_ID"].astype(int).tolist()
LIMBIC_IDS = roi_definition.loc[roi_definition["network"] == "limbic", "AAL3_ID"].astype(int).tolist()
assert len(CORTICAL_IDS) == 12 and len(LIMBIC_IDS) == 14
assert len(set(CORTICAL_IDS) & set(LIMBIC_IDS)) == 0
print("Cortical IDs:", CORTICAL_IDS)
print("Limbic IDs:", LIMBIC_IDS)
print("Cross-edge family:", len(CORTICAL_IDS) * len(LIMBIC_IDS))


# Source cell 5
def norm_key(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"P(\d{3})", text)
    if not match:
        raise ValueError(f"Expected a deidentified P### subject key, got {text!r}")
    return f"P{int(match.group(1)):03d}"


def pet_column_to_key(column):
    match = re.search(r"swsub_(P\d{3})_", str(column))
    return norm_key(match.group(1)) if match else None


def pet_column_to_session(column):
    match = re.search(r"_ses_(\d+)_", str(column))
    return f"ses-{match.group(1)}" if match else None


def aal_id(roi_name):
    match = re.match(r"^AAL3_(\d+)_", str(roi_name))
    return int(match.group(1)) if match else None


def partial_corr_matrix(values, covariates):
    design = np.column_stack([np.ones(len(values)), covariates])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ values
    residuals = values - design @ beta
    return np.clip(np.corrcoef(residuals, rowvar=False), -1.0, 1.0)


def signed_partial_r(t_value, residual_df):
    return float(np.sign(t_value) * np.sqrt(t_value**2 / (t_value**2 + residual_df)))


def fit_group_glm(frame, outcome, covariates):
    columns = [outcome, "group"] + covariates
    data = frame[columns].dropna().astype(float)
    design = sm.add_constant(data[["group"] + covariates], has_constant="add")
    fit = sm.OLS(data[outcome], design).fit()
    t_value = float(fit.tvalues["group"])
    return {
        "t": t_value,
        "p": float(fit.pvalues["group"]),
        "partial_r": signed_partial_r(t_value, fit.df_resid),
        "n": int(len(data)),
    }


def partial_spearman_age_sex(x, y, age, sex):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(age) & np.isfinite(sex)
    x, y, age, sex = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(age)[mask], np.asarray(sex)[mask]
    n = len(x)
    if n < 6:
        return np.nan, np.nan, n
    xr, yr = stats.rankdata(x), stats.rankdata(y)
    covariates = np.column_stack([np.ones(n), stats.rankdata(age), stats.rankdata(sex)])
    projection = np.eye(n) - covariates @ np.linalg.pinv(covariates)
    x_residual, y_residual = projection @ xr, projection @ yr
    r_value = float(np.corrcoef(x_residual, y_residual)[0, 1])
    if not np.isfinite(r_value):
        return np.nan, np.nan, n
    r_value = float(np.clip(r_value, -0.999999999, 0.999999999))
    residual_df = n - 4
    t_value = r_value * np.sqrt(residual_df / (1 - r_value**2))
    p_value = float(2 * stats.t.sf(abs(t_value), residual_df))
    return r_value, p_value, n


def bh_fdr(p_values):
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        output[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return output


def save_group_family(frame, p_column, q_column, output_dir, stem):
    """Save all TRD1-vs-nTRD tests and the prespecified p<0.01 subset.

    BH-FDR q-values remain in the all-results table for auditability, but they are
    not used to define the reported TRD1 discovery set.
    """
    ordered = frame.sort_values(p_column).reset_index(drop=True)
    reported = ordered[ordered[p_column] < GROUP_P_THRESHOLD].copy()
    reported["selected_uncorrected_p_lt_0p01"] = True
    ordered.to_excel(output_dir / f"{stem}_all.xlsx", index=False)
    reported.to_excel(output_dir / f"{stem}_uncorrected_p_lt_0p01.xlsx", index=False)
    return ordered, reported


# Source cell 7
clinical = pd.read_excel(CLINICAL_FILE, sheet_name=0)
clinical["subject_key"] = clinical["subject_key"].map(norm_key)
clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")
clinical["sex"] = pd.to_numeric(clinical["sex"], errors="coerce")
clinical["education"] = pd.to_numeric(clinical["education_years"], errors="coerce")
clinical["tiv"] = pd.to_numeric(clinical["tiv"], errors="coerce")
clinical["msm_t"] = pd.to_numeric(clinical["msm_t"], errors="coerce")
clinical["msm_total"] = pd.to_numeric(clinical["msm_total"], errors="coerce")
clinical["hdrs_pre"] = pd.to_numeric(clinical["hdrs17_pre"], errors="coerce")
clinical["hdrs_post"] = pd.to_numeric(clinical["hdrs17_post"], errors="coerce")
clinical["hdrs_improvement"] = clinical["hdrs_pre"] - clinical["hdrs_post"]
clinical["analysis_cohort"] = clinical["analysis_cohort"].astype(str)
clinical["analysis_arm"] = clinical["analysis_arm"].replace({np.nan: ""}).astype(str)
assert not clinical["subject_key"].duplicated().any()
assert clinical["analysis_cohort"].value_counts().to_dict() == EXPECTED_COUNTS
assert clinical.loc[clinical["analysis_cohort"] == "TRD2", "analysis_arm"].value_counts().to_dict() == EXPECTED_ARMS

pet_table = pd.read_excel(PET_FILE)
all_roi_names = pet_table.iloc[:, 0].astype(str).tolist()
pet_columns = [column for column in pet_table.columns[1:] if "ses_01" in str(column) or "ses_02" in str(column)]
pet = pet_table.set_index(pet_table.columns[0])[pet_columns].T
pet.columns = all_roi_names
pet["subject_key"] = [pet_column_to_key(column) for column in pet.index]
pet["session"] = [pet_column_to_session(column) for column in pet.index]
pet = pet.dropna(subset=["subject_key", "session"]).drop_duplicates(["subject_key", "session"], keep="first")
for roi in all_roi_names:
    pet[roi] = pd.to_numeric(pet[roi], errors="coerce")

ROI_COLUMNS = [roi for roi in all_roi_names if aal_id(roi) in set(CORTICAL_IDS + LIMBIC_IDS)]
CORTICAL_ROIS = [roi for roi in ROI_COLUMNS if aal_id(roi) in CORTICAL_IDS]
LIMBIC_ROIS = [roi for roi in ROI_COLUMNS if aal_id(roi) in LIMBIC_IDS]
assert len(ROI_COLUMNS) == 26 and len(CORTICAL_ROIS) == 12 and len(LIMBIC_ROIS) == 14

pet_pre = pet[pet["session"] == "ses-01"].copy()
pet_post = pet[pet["session"] == "ses-02"].copy()

roi_position = {roi: index for index, roi in enumerate(ROI_COLUMNS)}
tri_full = np.triu_indices(len(ROI_COLUMNS), 1)
pair_to_full_edge = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(tri_full[0], tri_full[1]))}
cross_pairs = []
for cortical_roi in CORTICAL_ROIS:
    for limbic_roi in LIMBIC_ROIS:
        i, j = sorted((roi_position[cortical_roi], roi_position[limbic_roi]))
        cross_pairs.append((i, j, cortical_roi, limbic_roi, pair_to_full_edge[(i, j)]))
CROSS_EDGE_IDS = [row[4] for row in cross_pairs]
CROSS_EDGE_NAMES = [(row[2], row[3]) for row in cross_pairs]
assert len(CROSS_EDGE_IDS) == 168 and len(set(CROSS_EDGE_IDS)) == 168

def cohort_pet(cohort, session="ses-01"):
    imaging = pet_pre if session == "ses-01" else pet_post
    rows = clinical[clinical["analysis_cohort"] == cohort].merge(imaging, on="subject_key", how="inner")
    return rows.dropna(subset=ROI_COLUMNS).sort_values("subject_key").reset_index(drop=True)

cohort_summary = pd.DataFrame({
    "cohort": list(EXPECTED_COUNTS),
    "clinical_n": [int((clinical["analysis_cohort"] == name).sum()) for name in EXPECTED_COUNTS],
    "ses01_complete_n": [len(cohort_pet(name)) for name in EXPECTED_COUNTS],
})
assert (cohort_summary["clinical_n"] == cohort_summary["ses01_complete_n"]).all()
setup_output_dir = OUTPUT_ROOT / "00_setup"
setup_output_dir.mkdir(parents=True, exist_ok=True)
cohort_summary.to_csv(setup_output_dir / "cohort_summary.csv", index=False)
display(cohort_summary)


# Source cell 9
mwc_adjusted_validation = json.loads(MWC_ADJUSTED_VALIDATION_FILE.read_text(encoding="utf-8"))
mwc_adjusted_checks = mwc_adjusted_validation.get("checks", {})
assert mwc_adjusted_validation.get("status") == "PASS"
assert mwc_adjusted_checks.get("HC_age_sex_normative_model_26_ROIs") is True

mwc_adjusted = pd.read_csv(MWC_ADJUSTED_WIDE_FILE)
mwc_adjusted["subject_key"] = mwc_adjusted["subject_key"].map(norm_key)
mwc_roi_names = [str(roi).split("_", 2)[2] for roi in ROI_COLUMNS]
mwc_feature_columns = [f"MwC_{roi}" for roi in mwc_roi_names]
assert len(mwc_feature_columns) == 26
assert all(column in mwc_adjusted for column in mwc_feature_columns)
assert len(mwc_adjusted) == 253
assert not mwc_adjusted.duplicated(["subject_key", "session"]).any()
assert int((mwc_adjusted.analysis_cohort.eq("HC") & mwc_adjusted.session.eq("ses-01")).sum()) == 23

mwc_normative_parameters = pd.read_csv(MWC_NORMATIVE_PARAMETERS_FILE)
assert len(mwc_normative_parameters) == 26
assert np.nanmax(np.abs(mwc_normative_parameters["HC_residual_z_mean"])) < 1e-10
assert np.nanmax(np.abs(mwc_normative_parameters["HC_residual_z_SD_ddof1"] - 1.0)) < 1e-10

def mwc_baseline_matrix(cohort):
    rows = mwc_adjusted.loc[
        mwc_adjusted.analysis_cohort.eq(cohort) & mwc_adjusted.session.eq("ses-01"),
        mwc_feature_columns,
    ]
    return rows.to_numpy(float)

mwc_group_values = {
    "nTRD": mwc_baseline_matrix("nTRD"),
    "TRD1": mwc_baseline_matrix("TRD1"),
    "TRD2": mwc_baseline_matrix("TRD2"),
}
assert {key: len(value) for key, value in mwc_group_values.items()} == {
    "nTRD": 25, "TRD1": 35, "TRD2": 85,
}
display(pd.DataFrame({
    "cohort": list(mwc_group_values),
    "baseline_n": [len(value) for value in mwc_group_values.values()],
    "metric": "age/sex-adjusted normative-SC MwC deviation z",
}))
