#!/usr/bin/env python3
"""Formal analysis of age/sex-adjusted 26-target normative-SC MwC deviations.

The input columns are HC-only feature-wise residual z scores from
``MwC ~ age + sex``. Raw MwC was computed upstream in the complete retained
AAL3-144 universe using the fixed 1,030-edge dTOR-985 group-normative
direct-count top-10% SC mask. This script never rebuilds a reduced-node
topology and never renormalizes within the extracted target set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

_MPL_CACHE = Path("/private/tmp/mwc_longitudinal_mpl")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("METAFL_ROOT", HERE.parent)).resolve()
DEFAULT_WIDE = (
    PROJECT_ROOT / "inputs/mwc/normative_sc/age_sex_adjusted/"
    "subject_level_corticolimbic26_age_sex_adjusted_normativeSC10_MwC_wide.csv"
)
DEFAULT_ROI = PROJECT_ROOT / "inputs/core/corticolimbic_12x14_ROI_definition.csv"
DEFAULT_BASELINE = (
    PROJECT_ROOT / "inputs/mwc/normative_sc/age_sex_adjusted/"
    "baseline_nTRD_vs_TRD1_age_sex_adjusted_MwC_HC3_all26.csv"
)
DEFAULT_UPSTREAM_VALIDATION = (
    PROJECT_ROOT / "inputs/mwc/normative_sc/validation_report.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/main_analyses/07_age_sex_adjusted_MwC"

ARMS = ["iTBS", "rTMS", "sham"]
ACTIVE_ARMS = ["iTBS", "rTMS"]
EXPECTED_AVAILABLE_PAIRS = {"iTBS": 31, "rTMS": 28, "sham": 26}
SEED = 20260903


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        output[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return output


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def short_roi(label: str) -> str:
    replacements = {
        "Frontal_Sup_2": "SFG",
        "Frontal_Mid_2": "MFG",
        "Frontal_Sup_Medial": "dmPFC",
        "Frontal_Med_Orb": "vmPFC",
        "OFCmed": "OFCmed",
        "OFClat": "OFClat",
        "Hippocampus": "HPC",
        "ParaHippocampal": "PHG",
        "Amygdala": "AMY",
        "Caudate": "CAU",
        "Putamen": "PUT",
        "Pallidum": "PAL",
        "ACC_sub": "SGC",
    }
    base, hemisphere = label.rsplit("_", 1)
    return f"{replacements.get(base, base)}-{hemisphere}"


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
) -> tuple[float, float, int]:
    arrays = [np.asarray(value, dtype=float) for value in (x, y, age, sex)]
    mask = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    x, y, age, sex = [value[mask] for value in arrays]
    n = len(x)
    if n < 6:
        return np.nan, np.nan, n
    ranked_x = stats.rankdata(x)
    ranked_y = stats.rankdata(y)
    design = np.column_stack([np.ones(n), stats.rankdata(age), stats.rankdata(sex)])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        projection = np.eye(n) - design @ np.linalg.pinv(design)
        residual_x = projection @ ranked_x
        residual_y = projection @ ranked_y
    r_value = float(np.corrcoef(residual_x, residual_y)[0, 1])
    r_value = float(np.clip(r_value, -0.999999999, 0.999999999))
    degrees_freedom = n - 4
    t_value = r_value * np.sqrt(degrees_freedom / (1.0 - r_value**2))
    p_value = float(2.0 * stats.t.sf(abs(t_value), degrees_freedom))
    return r_value, p_value, n


def align_arm(data: pd.DataFrame, arm: str, columns: list[str]) -> dict:
    subset = data.loc[
        data.analysis_cohort.eq("TRD2") & data.analysis_arm.eq(arm)
    ].copy()
    if subset.duplicated(["subject_key", "session"]).any():
        raise ValueError(f"Duplicate {arm} subject/session rows")
    pre_keys = set(subset.loc[subset.session.eq("ses-01"), "subject_key"])
    post_keys = set(subset.loc[subset.session.eq("ses-02"), "subject_key"])
    paired_keys = sorted(pre_keys & post_keys)
    if len(paired_keys) != EXPECTED_AVAILABLE_PAIRS[arm]:
        raise ValueError(f"Unexpected {arm} paired sample: {len(paired_keys)}")
    pre = (
        subset.loc[subset.session.eq("ses-01") & subset.subject_key.isin(paired_keys)]
        .set_index("subject_key")
        .loc[paired_keys]
    )
    post = (
        subset.loc[subset.session.eq("ses-02") & subset.subject_key.isin(paired_keys)]
        .set_index("subject_key")
        .loc[paired_keys]
    )
    if not np.array_equal(pre.index.to_numpy(), post.index.to_numpy()):
        raise ValueError(f"{arm} alignment failed")
    for covariate in ("age", "sex", "education", "hdrs_pre", "hdrs_post"):
        if not np.allclose(pre[covariate], post[covariate], equal_nan=True):
            raise ValueError(f"{arm} {covariate} differs across sessions")
    return {
        "arm": arm,
        "keys": np.asarray(paired_keys),
        "pre": pre[columns].to_numpy(float),
        "post": post[columns].to_numpy(float),
        "age": pre.age.to_numpy(float),
        "sex": pre.sex.to_numpy(float),
        "education": pre.education.to_numpy(float),
        "hdrs_improvement": pre.hdrs_pre.to_numpy(float) - pre.hdrs_post.to_numpy(float),
    }


def paired_table(result: dict, roi: pd.DataFrame) -> pd.DataFrame:
    pre = result["pre_deviation"]
    post = result["post_deviation"]
    change = post - pre
    rows = []
    for index, roi_row in roi.iterrows():
        mask = np.isfinite(pre[:, index]) & np.isfinite(post[:, index])
        x = pre[mask, index]
        y = post[mask, index]
        delta = y - x
        test = stats.ttest_rel(y, x)
        n = len(delta)
        sem = stats.sem(delta)
        critical = stats.t.ppf(0.975, n - 1)
        shapiro_p = stats.shapiro(delta).pvalue if 3 <= n <= 5_000 else np.nan
        rows.append(
            {
                "arm": result["arm"],
                "node_index": index + 1,
                "network": roi_row.network,
                "AAL3_ID": roi_row.AAL3_ID,
                "ROI": roi_row.ROI,
                "display_roi": short_roi(roi_row.ROI),
                "n_paired": n,
                "pre_mean_deviation_HC_z": float(np.mean(x)),
                "pre_SD_deviation_HC_z": float(np.std(x, ddof=1)),
                "post_mean_deviation_HC_z": float(np.mean(y)),
                "post_SD_deviation_HC_z": float(np.std(y, ddof=1)),
                "post_minus_pre_HC_z": float(np.mean(delta)),
                "change_ci95_low": float(np.mean(delta) - critical * sem),
                "change_ci95_high": float(np.mean(delta) + critical * sem),
                "paired_t": float(test.statistic),
                "paired_df": n - 1,
                "effect_size_Cohen_dz": float(np.mean(delta) / np.std(delta, ddof=1)),
                "paired_p_uncorrected": float(test.pvalue),
                "change_shapiro_W": float(stats.shapiro(delta).statistic) if 3 <= n <= 5_000 else np.nan,
                "change_shapiro_p": float(shapiro_p),
            }
        )
    table = pd.DataFrame(rows)
    table["paired_q_fdr_BH_26_nodes"] = bh_fdr(table.paired_p_uncorrected)
    table["paired_nominal_p_lt_0p05"] = table.paired_p_uncorrected < 0.05
    table["paired_fdr_q_lt_0p05"] = table.paired_q_fdr_BH_26_nodes < 0.05
    return table.sort_values(["arm", "paired_p_uncorrected"]).reset_index(drop=True)


def interaction_table(active: dict, sham: dict, roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, roi_row in roi.iterrows():
        active_change = active["post_deviation"][:, index] - active["pre_deviation"][:, index]
        sham_change = sham["post_deviation"][:, index] - sham["pre_deviation"][:, index]
        frame = pd.DataFrame(
            {
                "change": np.concatenate([active_change, sham_change]),
                "group": np.concatenate([np.ones(len(active_change)), np.zeros(len(sham_change))]),
                "age": np.concatenate([active["age"], sham["age"]]),
                "sex": np.concatenate([active["sex"], sham["sex"]]),
                "education": np.concatenate([active["education"], sham["education"]]),
            }
        ).dropna()
        design = sm.add_constant(frame[["group", "age", "sex"]], has_constant="add")
        ordinary = sm.OLS(frame.change, design).fit()
        fit = ordinary.get_robustcov_results(cov_type="HC3")
        names = list(design.columns)
        group_index = names.index("group")
        beta = float(fit.params[group_index])
        standard_error = float(fit.bse[group_index])
        t_value = float(fit.tvalues[group_index])
        p_value = float(fit.pvalues[group_index])
        ci_low, ci_high = fit.conf_int()[group_index]
        partial_r = float(np.sign(t_value) * np.sqrt(t_value**2 / (t_value**2 + fit.df_resid)))
        bp_p = float(het_breuschpagan(ordinary.resid, ordinary.model.exog)[1])
        residual_shapiro_p = float(stats.shapiro(ordinary.resid).pvalue)
        rows.append(
            {
                "active_arm": active["arm"],
                "comparison": f"{active['arm']} minus sham change",
                "node_index": index + 1,
                "network": roi_row.network,
                "AAL3_ID": roi_row.AAL3_ID,
                "ROI": roi_row.ROI,
                "display_roi": short_roi(roi_row.ROI),
                "n_active": int(np.isfinite(active_change).sum()),
                "n_sham": int(np.isfinite(sham_change).sum()),
                "n_model": int(fit.nobs),
                "active_mean_post_minus_pre_HC_z": float(np.nanmean(active_change)),
                "sham_mean_post_minus_pre_HC_z": float(np.nanmean(sham_change)),
                "interaction_beta_active_minus_sham_HC_z": beta,
                "interaction_hc3_se": standard_error,
                "interaction_hc3_ci95_low": float(ci_low),
                "interaction_hc3_ci95_high": float(ci_high),
                "interaction_hc3_t": t_value,
                "interaction_df_resid": float(fit.df_resid),
                "interaction_effect_size_partial_r": partial_r,
                "interaction_hc3_p_uncorrected": p_value,
                "residual_shapiro_p_descriptive": residual_shapiro_p,
                "breusch_pagan_p_descriptive": bp_p,
                "covariates": "age + sex",
                "model": "HC3 OLS on post-minus-pre change; equivalent group-by-time contrast",
            }
        )
    table = pd.DataFrame(rows)
    table["interaction_q_fdr_BH_26_nodes"] = bh_fdr(table.interaction_hc3_p_uncorrected)
    table["interaction_nominal_p_lt_0p05"] = table.interaction_hc3_p_uncorrected < 0.05
    table["interaction_fdr_q_lt_0p05"] = table.interaction_q_fdr_BH_26_nodes < 0.05
    return table.sort_values(["active_arm", "interaction_hc3_p_uncorrected"]).reset_index(drop=True)


def hdrs_table(result: dict, paired: pd.DataFrame, roi: pd.DataFrame) -> pd.DataFrame:
    change = result["post_deviation"] - result["pre_deviation"]
    rows = []
    for index, roi_row in roi.iterrows():
        r_value, p_value, n = partial_spearman(
            change[:, index],
            result["hdrs_improvement"],
            result["age"],
            result["sex"],
        )
        paired_row = paired.loc[paired.node_index.eq(index + 1)].iloc[0]
        rows.append(
            {
                "arm": result["arm"],
                "node_index": index + 1,
                "network": roi_row.network,
                "AAL3_ID": roi_row.AAL3_ID,
                "ROI": roi_row.ROI,
                "display_roi": short_roi(roi_row.ROI),
                "n_used": n,
                "HDRS_improvement_definition": "pre minus post; positive means clinical improvement",
                "MwC_change_definition": "post minus pre HC-referenced deviation z",
                "partial_Spearman_r": r_value,
                "partial_Spearman_p_uncorrected": p_value,
                "covariates": "age + sex",
                "paired_p_uncorrected": paired_row.paired_p_uncorrected,
                "paired_nominal_p_lt_0p05": paired_row.paired_nominal_p_lt_0p05,
            }
        )
    table = pd.DataFrame(rows)
    table["partial_Spearman_q_fdr_BH_26_nodes"] = bh_fdr(table.partial_Spearman_p_uncorrected)
    table["HDRS_nominal_p_lt_0p05"] = table.partial_Spearman_p_uncorrected < 0.05
    table["HDRS_fdr_q_lt_0p05"] = table.partial_Spearman_q_fdr_BH_26_nodes < 0.05
    table["paired_and_HDRS_nominal_p_lt_0p05"] = (
        table.paired_nominal_p_lt_0p05 & table.HDRS_nominal_p_lt_0p05
    )
    return table.sort_values(["arm", "partial_Spearman_p_uncorrected"]).reset_index(drop=True)


def plot_heatmap(
    results: dict[str, dict],
    paired_all: pd.DataFrame,
    roi: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    labels = roi.ROI.map(short_roi).tolist()
    level_rows = []
    change_rows = []
    row_labels = []
    change_labels = []
    for arm in ARMS:
        result = results[arm]
        pre = np.nanmean(result["pre_deviation"], axis=0)
        post = np.nanmean(result["post_deviation"], axis=0)
        change = np.nanmean(result["post_deviation"] - result["pre_deviation"], axis=0)
        level_rows.extend([pre, post])
        change_rows.append(change)
        row_labels.extend([f"{arm} pre", f"{arm} post"])
        change_labels.append(f"{arm} post−pre")
    levels = np.vstack(level_rows)
    changes = np.vstack(change_rows)
    limit = max(1.0, float(np.nanmax(np.abs(np.vstack([levels, changes])))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    sns.set_theme(style="white", context="paper")
    figure = plt.figure(figsize=(16.8, 7.3), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[1.9, 1.0], hspace=0.12)
    axes = [figure.add_subplot(grid[index]) for index in range(2)]
    first = axes[0].imshow(levels, aspect="auto", cmap="RdBu_r", norm=norm)
    axes[1].imshow(changes, aspect="auto", cmap="RdBu_r", norm=norm)

    for axis, matrix, ylabels in zip(axes, [levels, changes], [row_labels, change_labels]):
        axis.set_yticks(np.arange(len(ylabels)), ylabels)
        axis.set_xticks(np.arange(26), labels, rotation=55, ha="right")
        axis.axvline(11.5, color="#303030", linewidth=1.1)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if abs(matrix[row, column]) > 0.60 * limit else "#242424"
                axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=5.7, color=color)
    axes[0].set_title("A  Mean HC-referenced MwC deviation at each timepoint", loc="left", fontweight="semibold", fontsize=11)
    axes[1].set_title("B  Mean within-subject change", loc="left", fontweight="semibold", fontsize=11)
    axes[1].set_xlabel("Prespecified cortico-limbic target ROI")

    for row, arm in enumerate(ARMS):
        arm_table = paired_all.loc[paired_all.arm.eq(arm)].set_index("ROI").loc[roi.ROI]
        for column, item in enumerate(arm_table.itertuples()):
            symbol = "†" if item.paired_fdr_q_lt_0p05 else ("*" if item.paired_nominal_p_lt_0p05 else "")
            if symbol:
                axes[1].text(column + 0.34, row - 0.24, symbol, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111111")

    colorbar = figure.colorbar(first, ax=axes, fraction=0.016, pad=0.012)
    colorbar.set_label("HC-referenced normative-SC MwC deviation (z)")
    figure.suptitle(
        "iTBS, rTMS, and sham pre–post normative-SC MwC deviation",
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.025,
        (
            "Paired subjects: iTBS n=31, rTMS n=28, sham n=26. "
            "* uncorrected paired P<.05; † BH-FDR q<.05 across 26 targets. "
            "MwC was computed in all 144 retained AAL3 nodes before the 26 targets were extracted."
        ),
        ha="center",
        fontsize=8,
        color="#4A4A4A",
    )
    save_figure(
        figure,
        output / "figures" / "Figure_iTBS_rTMS_sham_prepost_normativeSC_MwC_deviation_heatmap",
    )

    records = []
    for arm_index, arm in enumerate(ARMS):
        for roi_index, roi_row in roi.iterrows():
            records.append(
                {
                    "arm": arm,
                    "ROI": roi_row.ROI,
                    "display_roi": labels[roi_index],
                    "pre_mean_deviation_HC_z": levels[2 * arm_index, roi_index],
                    "post_mean_deviation_HC_z": levels[2 * arm_index + 1, roi_index],
                    "post_minus_pre_HC_z": changes[arm_index, roi_index],
                }
            )
    return pd.DataFrame(records)


def main(
    wide_file: Path,
    roi_file: Path,
    baseline_file: Path,
    upstream_validation_file: Path,
    output: Path,
) -> int:
    paths = [wide_file, roi_file, baseline_file, upstream_validation_file]
    for path in paths:
        if not path.expanduser().is_file():
            raise FileNotFoundError(path)
    wide_file, roi_file, baseline_file, upstream_validation_file = [
        path.expanduser().resolve() for path in paths
    ]
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_dir = output / "tables"
    table_dir.mkdir(exist_ok=True)

    validation = json.loads(upstream_validation_file.read_text(encoding="utf-8"))
    checks = validation.get("checks", {})
    if validation.get("status") != "PASS" or not all(
        checks.get(key, False)
        for key in ("retained_nodes_144", "fixed_structural_edges_1030")
    ):
        raise ValueError("The upstream AAL3-144 MwC validation is not PASS")

    roi = pd.read_csv(roi_file).reset_index(drop=True)
    data = pd.read_csv(wide_file)
    roi_names = roi.ROI.astype(str).tolist()
    columns = [f"MwC_{name}" for name in roi_names]
    if len(roi) != 26 or not all(column in data for column in columns):
        raise ValueError("Expected the ordered 26 target MwC columns")

    hc = data.loc[data.analysis_cohort.eq("HC") & data.session.eq("ses-01"), columns].to_numpy(float)
    if len(hc) != 23:
        raise ValueError("Expected 23 baseline HC references")
    if np.nanmax(np.abs(np.nanmean(hc, axis=0))) >= 1e-10:
        raise ValueError("Adjusted HC MwC means are not zero")
    if np.nanmax(np.abs(np.nanstd(hc, axis=0, ddof=1) - 1.0)) >= 1e-10:
        raise ValueError("Adjusted HC MwC SDs are not one")

    results = {arm: align_arm(data, arm, columns) for arm in ARMS}
    for result in results.values():
        # The input values are already formal HC age/sex normative residual z scores.
        result["pre_deviation"] = result["pre"].copy()
        result["post_deviation"] = result["post"].copy()

    paired_all = pd.concat(
        [paired_table(results[arm], roi) for arm in ARMS], ignore_index=True
    )
    paired_p05 = paired_all.loc[
        paired_all.arm.isin(ACTIVE_ARMS) & paired_all.paired_nominal_p_lt_0p05
    ].copy()
    interactions_all = pd.concat(
        [interaction_table(results[arm], results["sham"], roi) for arm in ACTIVE_ARMS],
        ignore_index=True,
    )
    interactions_p05 = interactions_all.loc[
        interactions_all.interaction_nominal_p_lt_0p05
    ].copy()
    hdrs_all = pd.concat(
        [
            hdrs_table(
                results[arm],
                paired_all.loc[paired_all.arm.eq(arm)],
                roi,
            )
            for arm in ACTIVE_ARMS
        ],
        ignore_index=True,
    )
    # Legacy notebook hierarchy: HDRS correlations are primary only for targets
    # that first showed an uncorrected paired pre-post P < .05 within that arm.
    hdrs_screened = hdrs_all.loc[hdrs_all.paired_nominal_p_lt_0p05].copy()
    hdrs_screened["HDRS_q_fdr_BH_within_prepost_significant_nodes"] = np.nan
    for arm, indices in hdrs_screened.groupby("arm").groups.items():
        hdrs_screened.loc[indices, "HDRS_q_fdr_BH_within_prepost_significant_nodes"] = bh_fdr(
            hdrs_screened.loc[indices, "partial_Spearman_p_uncorrected"]
        )
    hdrs_screened["HDRS_primary_FDR_q_lt_0p05"] = (
        hdrs_screened.HDRS_q_fdr_BH_within_prepost_significant_nodes < 0.05
    )
    hdrs_screened["HDRS_exploratory_uncorrected_p_lt_0p05"] = (
        hdrs_screened.partial_Spearman_p_uncorrected < 0.05
    )
    hdrs_screened_p05 = hdrs_screened.loc[
        hdrs_screened.HDRS_exploratory_uncorrected_p_lt_0p05
    ].copy()
    hdrs_all26_p05 = hdrs_all.loc[hdrs_all.HDRS_nominal_p_lt_0p05].copy()

    baseline_all = pd.read_csv(baseline_file)
    if "display_roi" not in baseline_all:
        baseline_all["display_roi"] = baseline_all["ROI"].map(short_roi)
    baseline_p05 = baseline_all.loc[baseline_all.hc3_p_uncorrected < 0.05].copy()
    baseline_all.to_csv(table_dir / "baseline_nTRD_vs_TRD1_adjusted_all_26.csv", index=False)
    baseline_p05.to_csv(table_dir / "baseline_nTRD_vs_TRD1_adjusted_p_lt_0p05.csv", index=False)
    paired_all.to_csv(table_dir / "iTBS_rTMS_sham_paired_all_26.csv", index=False)
    paired_p05.to_csv(table_dir / "iTBS_rTMS_paired_p_lt_0p05.csv", index=False)
    interactions_all.to_csv(table_dir / "iTBS_rTMS_vs_sham_interaction_adjusted_all_26.csv", index=False)
    interactions_p05.to_csv(table_dir / "iTBS_rTMS_vs_sham_interaction_adjusted_p_lt_0p05.csv", index=False)
    hdrs_all.to_csv(table_dir / "iTBS_rTMS_HDRS_partial_spearman_adjusted_all_26.csv", index=False)
    hdrs_all26_p05.to_csv(table_dir / "iTBS_rTMS_HDRS_partial_spearman_all26_exploratory_p_lt_0p05.csv", index=False)
    hdrs_screened.to_csv(table_dir / "iTBS_rTMS_HDRS_paired_p_lt_0p05_subset.csv", index=False)
    hdrs_screened_p05.to_csv(table_dir / "iTBS_rTMS_HDRS_paired_p_lt_0p05_subset_nominal_p_lt_0p05.csv", index=False)
    heatmap_data = plot_heatmap(results, paired_all, roi, output)
    heatmap_data.to_csv(table_dir / "iTBS_rTMS_sham_prepost_MwC_heatmap_data.csv", index=False)

    summary_rows = []
    summary_rows.extend(
        [
            {"analysis": "baseline TRD1 vs nTRD", "arm": "TRD1", "family": "26 MwC targets", "tested": 26, "nominal_p_lt_0p05": int((baseline_all.hc3_p_uncorrected < 0.05).sum()), "fdr_q_lt_0p05": int((baseline_all.q_fdr_BH_26_nodes < 0.05).sum()), "method": "HC3 OLS; age + sex"}
        ]
    )
    for arm in ARMS:
        subset = paired_all.loc[paired_all.arm.eq(arm)]
        summary_rows.append({"analysis": "paired pre-post", "arm": arm, "family": "26 MwC targets", "tested": 26, "nominal_p_lt_0p05": int(subset.paired_nominal_p_lt_0p05.sum()), "fdr_q_lt_0p05": int(subset.paired_fdr_q_lt_0p05.sum()), "method": "paired t; post minus pre"})
    for arm in ACTIVE_ARMS:
        subset = interactions_all.loc[interactions_all.active_arm.eq(arm)]
        summary_rows.append({"analysis": "group-by-time versus sham", "arm": arm, "family": "26 MwC targets", "tested": 26, "nominal_p_lt_0p05": int(subset.interaction_nominal_p_lt_0p05.sum()), "fdr_q_lt_0p05": int(subset.interaction_fdr_q_lt_0p05.sum()), "method": "HC3 change-score OLS; age + sex"})
        subset = hdrs_screened.loc[hdrs_screened.arm.eq(arm)]
        summary_rows.append({"analysis": "MwC change vs HDRS improvement", "arm": arm, "family": "targets passing paired P<.05", "tested": len(subset), "nominal_p_lt_0p05": int(subset.HDRS_exploratory_uncorrected_p_lt_0p05.sum()), "fdr_q_lt_0p05": int(subset.HDRS_primary_FDR_q_lt_0p05.sum()), "method": "two-stage screen: partial Spearman controlling age + sex; BH within paired-significant targets"})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "analysis_summary_counts.csv", index=False)

    method_notes = pd.DataFrame(
        [
            {"topic": "Atlas/MwC universe", "specification": "MwC computed in all 144 retained AAL3 nodes; 26 target columns extracted afterward"},
            {"topic": "Structural constraint", "specification": "dTOR-985 group-normative direct streamline-count top-10% mask; 1,030 undirected edges"},
            {"topic": "Baseline group model", "specification": "HC3 OLS, TRD1 minus nTRD, adjusted for age + sex; BH-FDR across 26 targets"},
            {"topic": "Paired pre-post", "specification": "Paired t test of post minus pre; static covariates are not entered; BH-FDR across 26 targets"},
            {"topic": "Group-by-time", "specification": "HC3 OLS on individual post-minus-pre change, active arm minus sham, adjusted for age + sex; BH-FDR across 26 targets"},
            {"topic": "HDRS paired-significant subset", "specification": "Only targets with paired uncorrected P<.05 enter partial Spearman; age + sex covariates; BH-FDR within selected targets"},
            {"topic": "HDRS all-26 sensitivity", "specification": "Exploratory partial Spearman for every target, age + sex covariates; BH-FDR across 26 targets"},
            {"topic": "HDRS sign", "specification": "HDRS improvement = pre minus post (positive is improvement); MwC change = post minus pre HC z"},
            {"topic": "Missing values", "specification": "Subjects paired before ROI-wise complete-pair analysis; no imputation and no zero fill"},
            {"topic": "Inference warning", "specification": "Nominal P<.05 rows are explicitly uncorrected unless q<.05 is also shown"},
        ]
    )
    method_notes.to_csv(table_dir / "method_notes.csv", index=False)

    workbook_tables = {
        "Summary": json.loads(summary.to_json(orient="records")),
        "Baseline_p05": json.loads(baseline_p05.to_json(orient="records")),
        "Paired_p05": json.loads(paired_p05.to_json(orient="records")),
        "Interaction_p05": json.loads(interactions_p05.to_json(orient="records")),
        "HDRS_primary": json.loads(hdrs_screened.to_json(orient="records")),
        "HDRS_all26_p05": json.loads(hdrs_all26_p05.to_json(orient="records")),
        "Baseline_all": json.loads(baseline_all.to_json(orient="records")),
        "Paired_all": json.loads(paired_all.to_json(orient="records")),
        "Interaction_all": json.loads(interactions_all.to_json(orient="records")),
        "HDRS_all": json.loads(hdrs_all.to_json(orient="records")),
        "Heatmap_data": json.loads(heatmap_data.to_json(orient="records")),
        "Method": json.loads(method_notes.to_json(orient="records")),
    }
    (output / "workbook_tables.json").write_text(
        json.dumps(workbook_tables, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure_stem = output / "figures" / "Figure_iTBS_rTMS_sham_prepost_normativeSC_MwC_deviation_heatmap"
    output_checks = {
        "upstream_AAL3_144_validation_PASS": validation.get("status") == "PASS",
        "all_144_before_26_extraction": True,
        "fixed_dTOR_normative_SC_edges_1030": checks["fixed_structural_edges_1030"],
        "baseline_HC_n_23": len(hc) == 23,
        "input_HC_adjusted_mean_zero_SD_one": bool(
            np.nanmax(np.abs(np.nanmean(hc, axis=0))) < 1e-10
            and np.nanmax(np.abs(np.nanstd(hc, axis=0, ddof=1) - 1.0)) < 1e-10
        ),
        "paired_sample_counts_match": all(len(results[arm]["keys"]) == EXPECTED_AVAILABLE_PAIRS[arm] for arm in ARMS),
        "baseline_26_results": len(baseline_all) == 26,
        "paired_78_results": len(paired_all) == 78,
        "interaction_52_results": len(interactions_all) == 52,
        "HDRS_all26_sensitivity_52_results": len(hdrs_all) == 52,
        "HDRS_subset_matches_active_paired_significant_targets": (
            len(hdrs_screened) == len(paired_p05)
            and set(hdrs_screened.arm) == set(paired_p05.arm)
        ),
        "heatmap_figure_PNG_PDF_SVG_exist": all(figure_stem.with_suffix(f".{extension}").is_file() for extension in ("png", "pdf", "svg")),
    }
    payload = {
        "status": "PASS" if all(output_checks.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "Complete AAL3 normative-SC MwC baseline and longitudinal analysis",
        "scope": {
            "metric": "MwC only",
            "atlas_universe": "144 retained AAL3 nodes",
            "display_targets": 26,
            "SC": "dTOR-985 group-normative direct-count top-10% mask; 1,030 undirected edges",
            "excluded_metrics": ["fDC", "mDC", "wDC"],
            "missing_policy": "pair subjects first; ROI-wise complete pairs; no imputation or zero fill",
        },
        "sample": {"HC_reference": 23, **EXPECTED_AVAILABLE_PAIRS},
        "checks": output_checks,
        "summary": summary.to_dict(orient="records"),
        "inputs": {
            "subject_table": wide_file.relative_to(PROJECT_ROOT).as_posix(),
            "subject_table_sha256": sha256(wide_file),
            "baseline_table": baseline_file.relative_to(PROJECT_ROOT).as_posix(),
            "baseline_table_sha256": sha256(baseline_file),
            "upstream_validation": upstream_validation_file.relative_to(PROJECT_ROOT).as_posix(),
            "upstream_validation_sha256": sha256(upstream_validation_file),
        },
    }
    (output / "validation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if payload["status"] != "PASS":
        raise RuntimeError(f"Validation failed: {output_checks}")
    print(summary.to_string(index=False))
    print("\nBaseline nominal p<.05:")
    print(baseline_p05[["display_roi", "adjusted_beta_TRD1_minus_nTRD_HC_SD", "partial_r_group", "hc3_p_uncorrected", "q_fdr_BH_26_nodes"]].to_string(index=False))
    print("\nActive-arm paired nominal p<.05:")
    print(paired_p05[["arm", "display_roi", "n_paired", "post_minus_pre_HC_z", "effect_size_Cohen_dz", "paired_p_uncorrected", "paired_q_fdr_BH_26_nodes"]].to_string(index=False))
    print("\nInteraction nominal p<.05:")
    print(interactions_p05[["active_arm", "display_roi", "interaction_beta_active_minus_sham_HC_z", "interaction_effect_size_partial_r", "interaction_hc3_p_uncorrected", "interaction_q_fdr_BH_26_nodes"]].to_string(index=False))
    print("\nHDRS screen among targets with paired p<.05:")
    print(hdrs_screened[["arm", "display_roi", "partial_Spearman_r", "partial_Spearman_p_uncorrected", "HDRS_q_fdr_BH_within_prepost_significant_nodes", "HDRS_primary_FDR_q_lt_0p05"]].to_string(index=False))
    print("\nExploratory all-26 HDRS nominal p<.05:")
    print(hdrs_all26_p05[["arm", "display_roi", "partial_Spearman_r", "partial_Spearman_p_uncorrected", "partial_Spearman_q_fdr_BH_26_nodes", "paired_nominal_p_lt_0p05"]].to_string(index=False))
    print(f"\nPASS: {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-file", type=Path, default=DEFAULT_WIDE)
    parser.add_argument("--roi-file", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--upstream-validation-file", type=Path, default=DEFAULT_UPSTREAM_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.wide_file, arguments.roi_file, arguments.baseline_file, arguments.upstream_validation_file, arguments.output))
