#!/usr/bin/env python3
"""Plot the 26-ROI TRD1-vs-nTRD circle adjusted for baseline severity.

The visual renderer and cross-sectional colour scales are imported directly
from the established bilateral-SGC circle-diagram script.  The only new
quantity is the adjusted group coefficient from one OLS model per node/edge:

    deviation ~ TRD1 group + age + sex + baseline HDRS-17

The group coefficient therefore represents adjusted TRD1 minus nTRD.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_MPL_CACHE = Path("/private/tmp/metafl_sgc26_baseline_severity_mpl")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D

import circle_renderer as circle


ORIGINAL_IDS = [
    3, 4, 5, 6, 19, 20, 21, 22, 25, 26, 31, 32,
    41, 42, 43, 44, 45, 46, 75, 76, 77, 78, 79, 80,
]
SGC_ROIS = ["AAL3_151_ACC_sub_L", "AAL3_152_ACC_sub_R"]
CURRENT_PANELS = [
    "nonTRD deviation",
    "TRD1 deviation",
    "TRD2 deviation",
    "TRD versus nonTRD",
]
ADJUSTED_PANEL = "TRD versus nonTRD\n(baseline severity adjusted)"


def full_roi_name(row: pd.Series) -> str:
    return f"AAL3_{int(row.AAL3_ID):03d}_{row.ROI}"


def adjusted_group_model(frame: pd.DataFrame, outcome: str) -> dict:
    columns = [
        outcome, "analysis_cohort", "age", "sex", "hdrs_pre"
    ]
    data = frame.loc[
        frame.analysis_cohort.isin(["nTRD", "TRD1"]), columns
    ].dropna().copy()
    data["group_TRD1"] = data.analysis_cohort.eq("TRD1").astype(float)
    design_columns = ["group_TRD1", "age", "sex", "hdrs_pre"]
    design = sm.add_constant(data[design_columns], has_constant="add")
    fit = sm.OLS(data[outcome].astype(float), design.astype(float)).fit()
    ci = fit.conf_int().loc["group_TRD1"]
    t_value = float(fit.tvalues["group_TRD1"])
    df_resid = float(fit.df_resid)
    partial_r = float(
        np.sign(t_value) * np.sqrt(t_value**2 / (t_value**2 + df_resid))
    )
    return {
        "feature": outcome,
        "comparison": "TRD1 - nTRD",
        "model": (
            "deviation ~ group_TRD1 + age + sex + baseline_HDRS17"
        ),
        "n_total": int(len(data)),
        "n_TRD1": int(data.group_TRD1.eq(1).sum()),
        "n_nTRD": int(data.group_TRD1.eq(0).sum()),
        "TRD1_unadjusted_mean": float(
            data.loc[data.group_TRD1.eq(1), outcome].mean()
        ),
        "nTRD_unadjusted_mean": float(
            data.loc[data.group_TRD1.eq(0), outcome].mean()
        ),
        "adjusted_group_beta_TRD1_minus_nTRD": float(fit.params["group_TRD1"]),
        "standard_error": float(fit.bse["group_TRD1"]),
        "ci95_low": float(ci.iloc[0]),
        "ci95_high": float(ci.iloc[1]),
        "t": t_value,
        "df_resid": df_resid,
        "partial_r": partial_r,
        "p_uncorrected": float(fit.pvalues["group_TRD1"]),
        "condition_number": float(fit.condition_number),
    }


def save_figure(fig, stem: Path) -> None:
    for suffix, kwargs in [(".png", {"dpi": 600}), (".pdf", {}), (".svg", {})]:
        fig.savefig(
            stem.with_suffix(suffix), facecolor="white", bbox_inches="tight", **kwargs
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(os.environ.get("METAFL_ROOT", Path(__file__).resolve().parents[1])),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    base = args.base.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else base / "outputs" / "main_analyses" / "11_SGC_26ROI_figures"
    )
    figure_dir = output_root / "figures"
    table_dir = output_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    roi = pd.read_csv(base / "inputs" / "core" / "corticolimbic_12x14_ROI_definition.csv")
    roi["full_roi"] = roi.apply(full_roi_name, axis=1)
    cortical_rois = roi.loc[roi.network.eq("cortical"), "full_roi"].tolist()
    limbic_rois = roi.loc[roi.network.eq("limbic"), "full_roi"].tolist()
    roi_columns = cortical_rois + limbic_rois
    cross_edges = [
        (cortical_roi, limbic_roi)
        for cortical_roi in cortical_rois
        for limbic_roi in limbic_rois
    ]
    if len(cortical_rois) != 12 or len(limbic_rois) != 14:
        raise ValueError("Expected 12 cortical and 14 limbic ROIs")
    if limbic_rois[-2:] != SGC_ROIS:
        raise ValueError("Expected bilateral SGC as final two limbic ROIs")

    clinical = pd.read_excel(base / "inputs" / "core" / "clinical_data.xlsx").rename(
        columns={
            "education_years": "education",
            "hdrs17_pre": "hdrs_pre",
            "hdrs17_post": "hdrs_post",
        }
    )
    covariates = clinical[[
        "subject_key", "analysis_cohort", "age", "sex", "hdrs_pre"
    ]].copy()

    discovery = base / "outputs" / "main_analyses" / "01_TRD1_discovery_agesex"
    regional_frames = []
    edge_frames = []
    edge_features = [
        f"connectivity::{roi_1}::{roi_2}" for roi_1, roi_2 in cross_edges
    ]
    for cohort in ("nTRD", "TRD1"):
        regional = pd.read_excel(discovery / f"regional_{cohort}_subjects.xlsx")
        if regional["subject_key"].duplicated().any():
            raise ValueError(f"Duplicate regional subject keys for {cohort}")
        regional_frames.append(regional.merge(covariates, on="subject_key", how="left", validate="one_to_one"))

        with np.load(discovery / f"individual_MCM_{cohort}_cross168.npz", allow_pickle=True) as archive:
            edge_names = [tuple(item) for item in archive["edge_names"].tolist()]
            if edge_names != cross_edges:
                raise ValueError(f"Unexpected edge order for {cohort}")
            edge = pd.DataFrame(archive["z_edges"], columns=edge_features)
            edge.insert(0, "subject_key", archive["subject_keys"].astype(str))
        if edge["subject_key"].duplicated().any():
            raise ValueError(f"Duplicate connectivity subject keys for {cohort}")
        edge_frames.append(edge.merge(covariates, on="subject_key", how="left", validate="one_to_one"))

    regional_baseline = pd.concat(regional_frames, ignore_index=True)
    edge_baseline = pd.concat(edge_frames, ignore_index=True)
    if regional_baseline["analysis_cohort"].value_counts().to_dict() != {"TRD1": 35, "nTRD": 25}:
        raise ValueError("Unexpected regional baseline cohort counts")
    if edge_baseline["analysis_cohort"].value_counts().to_dict() != {"TRD1": 35, "nTRD": 25}:
        raise ValueError("Unexpected connectivity baseline cohort counts")

    regional_results = pd.DataFrame([
        adjusted_group_model(regional_baseline, roi_name) for roi_name in roi_columns
    ])
    regional_results.insert(0, "ROI", roi_columns)
    regional_results.insert(
        1,
        "label",
        [circle.roi_short_label(roi_name) for roi_name in roi_columns],
    )
    regional_results.insert(
        2,
        "network",
        ["cortical"] * len(cortical_rois) + ["limbic"] * len(limbic_rois),
    )

    edge_results = pd.DataFrame([
        adjusted_group_model(edge_baseline, feature) for feature in edge_features
    ])
    edge_results.insert(0, "ROI_1", [edge[0] for edge in cross_edges])
    edge_results.insert(1, "ROI_2", [edge[1] for edge in cross_edges])
    edge_results.insert(
        2,
        "edge_label",
        [
            f"{circle.roi_short_label(edge[0])}-{circle.roi_short_label(edge[1])}"
            for edge in cross_edges
        ],
    )

    adjusted_nodes = regional_results[
        "adjusted_group_beta_TRD1_minus_nTRD"
    ].to_numpy(float)
    adjusted_edges = edge_results[
        "adjusted_group_beta_TRD1_minus_nTRD"
    ].to_numpy(float)

    current_data_file = (
        base / "outputs" / "main_analyses" / "04_figures" /
        "corticolimbic_circular_plot_data.xlsx"
    )
    current_nodes = pd.read_excel(current_data_file, sheet_name="regional_nodes")
    current_edges = pd.read_excel(current_data_file, sheet_name="connectivity_edges")

    panels: dict[str, dict[str, np.ndarray]] = {}
    for panel in CURRENT_PANELS:
        node_lookup = current_nodes.loc[current_nodes.panel.eq(panel)].set_index("ROI")["value"]
        edge_lookup = current_edges.loc[current_edges.panel.eq(panel)].set_index(
            ["ROI_1", "ROI_2"]
        )["value"]
        panels[panel] = {
            "regional": np.asarray([node_lookup.loc[name] for name in roi_columns], float),
            "edge": np.asarray([edge_lookup.loc[edge] for edge in cross_edges], float),
        }
    panels[ADJUSTED_PANEL] = {"regional": adjusted_nodes, "edge": adjusted_edges}

    settings = pd.read_excel(current_data_file, sheet_name="plot_settings").set_index("parameter")["value"]
    node_vmax = float(settings.loc["regional_shared_abs_scale_98pct"])
    edge_vmax = float(settings.loc["edge_shared_abs_scale_98pct"])

    individual_stem = figure_dir / (
        "corticolimbic_circular_26ROI_bilateral_SGC_"
        "TRD_vs_nonTRD_baseline_severity_adjusted"
    )
    fig, ax = plt.subplots(figsize=(4.1, 4.25), facecolor="white")
    circle.draw_corticolimbic_connectogram(
        ax,
        ADJUSTED_PANEL,
        adjusted_nodes,
        adjusted_edges,
        roi_columns,
        cortical_rois,
        limbic_rois,
        cross_edges,
        node_vmax,
        edge_vmax,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.03)
    save_figure(fig, individual_stem)
    plt.close(fig)

    five_panel_stem = figure_dir / (
        "corticolimbic_circular_26ROI_bilateral_SGC_"
        "cross_sectional_with_baseline_severity_adjusted"
    )
    panel_order = CURRENT_PANELS + [ADJUSTED_PANEL]
    fig, axes = plt.subplots(1, 5, figsize=(20.0, 4.6), facecolor="white")
    for ax, title in zip(axes, panel_order):
        circle.draw_corticolimbic_connectogram(
            ax,
            title,
            panels[title]["regional"],
            panels[title]["edge"],
            roi_columns,
            cortical_rois,
            limbic_rois,
            cross_edges,
            node_vmax,
            edge_vmax,
        )
    legend_handles = [
        Line2D(
            [0], [0], color=circle.CIRCULAR_POSITIVE, linewidth=2.0,
            marker="s", markerfacecolor=circle.CIRCULAR_POSITIVE,
            markeredgecolor="white", label="Positive deviation / higher in TRD1",
        ),
        Line2D(
            [0], [0], color=circle.CIRCULAR_NEGATIVE, linewidth=2.0,
            marker="s", markerfacecolor=circle.CIRCULAR_NEGATIVE,
            markeredgecolor="white", label="Negative deviation / lower in TRD1",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.005), fontsize=8,
    )
    fig.subplots_adjust(
        left=0.012, right=0.988, top=0.90, bottom=0.12, wspace=0.25
    )
    save_figure(fig, five_panel_stem)
    plt.close(fig)

    regional_file = table_dir / (
        "baseline_severity_adjusted_TRD1_vs_nTRD_regional26_SGC.csv"
    )
    edge_file = table_dir / (
        "baseline_severity_adjusted_TRD1_vs_nTRD_connectivity168_SGC.csv"
    )
    regional_results.to_csv(regional_file, index=False)
    edge_results.to_csv(edge_file, index=False)

    sample = clinical.loc[
        clinical.analysis_cohort.isin(["nTRD", "TRD1"]),
        ["subject_key", "analysis_cohort", "age", "sex", "hdrs_pre"],
    ].dropna()
    severity_by_group = sample.groupby("analysis_cohort").hdrs_pre.agg(
        ["count", "mean", "std", "min", "max"]
    )
    report = {
        "status": "PASS",
        "scope": "TRD1 versus nTRD 26-ROI SGC circle adjusted for baseline severity",
        "model": (
            "deviation ~ group_TRD1 + age + sex + baseline HDRS-17"
        ),
        "effect_displayed": "OLS group coefficient: adjusted TRD1 minus nTRD",
        "regional_nodes": len(adjusted_nodes),
        "cortical_limbic_edges": len(adjusted_edges),
        "n_TRD1": int(sample.analysis_cohort.eq("TRD1").sum()),
        "n_nTRD": int(sample.analysis_cohort.eq("nTRD").sum()),
        "baseline_HDRS17_by_group": severity_by_group.to_dict(orient="index"),
        "same_renderer_as_original": True,
        "same_node_scale_as_original_four_panels": node_vmax,
        "same_edge_scale_as_original_four_panels": edge_vmax,
        "checks": {
            "all_adjusted_node_values_finite": bool(np.isfinite(adjusted_nodes).all()),
            "all_adjusted_edge_values_finite": bool(np.isfinite(adjusted_edges).all()),
            "individual_png_exists": individual_stem.with_suffix(".png").is_file(),
            "five_panel_png_exists": five_panel_stem.with_suffix(".png").is_file(),
            "node_table_rows_26": len(regional_results) == 26,
            "edge_table_rows_168": len(edge_results) == 168,
        },
        "outputs": {
            "individual_PNG": str(individual_stem.with_suffix(".png")),
            "individual_PDF": str(individual_stem.with_suffix(".pdf")),
            "individual_SVG": str(individual_stem.with_suffix(".svg")),
            "five_panel_PNG": str(five_panel_stem.with_suffix(".png")),
            "five_panel_PDF": str(five_panel_stem.with_suffix(".pdf")),
            "five_panel_SVG": str(five_panel_stem.with_suffix(".svg")),
            "regional_model_table": str(regional_file),
            "connectivity_model_table": str(edge_file),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "FAIL"
    report_path = output_root / "baseline_severity_adjusted_circle_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
