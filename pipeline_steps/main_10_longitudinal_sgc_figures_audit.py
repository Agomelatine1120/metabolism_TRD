"""Auto-extracted pipeline step: Main 10 — longitudinal and SGC figures with final audit.

Do not edit generated copies independently of the source notebooks.
"""

import subprocess
import sys


# Source cell 69
# Four longitudinal circular panels using the exact renderer and visual language
# defined for the cross-sectional cortico-limbic connectogram.
LONGITUDINAL_CIRCULAR_OUT = FIGURE_OUT
LONGITUDINAL_CIRCULAR_OUT.mkdir(parents=True, exist_ok=True)


def _paired_effect_tables(result, panel_label):
    regional_delta = np.asarray(result["regional_delta"], dtype=float)
    edge_delta = np.asarray(result["edge_delta"], dtype=float)

    regional_t, regional_p = stats.ttest_rel(
        result["regional_post"], result["regional_pre"], axis=0, nan_policy="omit"
    )
    edge_t, edge_p = stats.ttest_rel(
        result["edge_post"], result["edge_pre"], axis=0, nan_policy="omit"
    )
    regional_n = np.sum(np.isfinite(result["regional_pre"]) & np.isfinite(result["regional_post"]), axis=0)
    edge_n = np.sum(np.isfinite(result["edge_pre"]) & np.isfinite(result["edge_post"]), axis=0)
    regional_sd = np.nanstd(regional_delta, axis=0, ddof=1)
    edge_sd = np.nanstd(edge_delta, axis=0, ddof=1)

    regional = pd.DataFrame({
        "panel": panel_label,
        "analysis": "paired post minus pre",
        "ROI": ROI_COLUMNS,
        "n": regional_n,
        "pre_mean_deviation": np.nanmean(result["regional_pre"], axis=0),
        "post_mean_deviation": np.nanmean(result["regional_post"], axis=0),
        "effect_value": np.nanmean(regional_delta, axis=0),
        "paired_t": regional_t,
        "effect_size_Cohen_dz": np.divide(
            np.nanmean(regional_delta, axis=0),
            regional_sd,
            out=np.full(len(ROI_COLUMNS), np.nan),
            where=regional_sd > 0,
        ),
        "p_uncorrected": regional_p,
        "q_fdr_BH_within_26_regions": bh_fdr(regional_p),
    })
    edge = pd.DataFrame({
        "panel": panel_label,
        "analysis": "paired post minus pre",
        "ROI_1": [pair[0] for pair in CROSS_EDGE_NAMES],
        "ROI_2": [pair[1] for pair in CROSS_EDGE_NAMES],
        "n": edge_n,
        "pre_mean_deviation": np.nanmean(result["edge_pre"], axis=0),
        "post_mean_deviation": np.nanmean(result["edge_post"], axis=0),
        "effect_value": np.nanmean(edge_delta, axis=0),
        "paired_t": edge_t,
        "effect_size_Cohen_dz": np.divide(
            np.nanmean(edge_delta, axis=0),
            edge_sd,
            out=np.full(len(CROSS_EDGE_NAMES), np.nan),
            where=edge_sd > 0,
        ),
        "p_uncorrected": edge_p,
        "q_fdr_BH_within_168_edges": bh_fdr(edge_p),
    })
    return regional, edge


def _pooled_active_vs_sham_glm(outcome_active, outcome_sham, feature_labels, modality):
    outcome_active = np.asarray(outcome_active, dtype=float)
    outcome_sham = np.asarray(outcome_sham, dtype=float)
    active_age = np.concatenate([itbs_result["age"], rtms_result["age"]])
    active_sex = np.concatenate([itbs_result["sex"], rtms_result["sex"]])
    active_education = np.concatenate([itbs_result["education"], rtms_result["education"]])
    sham_age = np.asarray(sham_result["age"], dtype=float)
    sham_sex = np.asarray(sham_result["sex"], dtype=float)
    sham_education = np.asarray(sham_result["education"], dtype=float)

    rows = []
    for feature_index, feature in enumerate(feature_labels):
        active_change = outcome_active[:, feature_index]
        sham_change = outcome_sham[:, feature_index]
        frame = pd.DataFrame({
            "change": np.concatenate([active_change, sham_change]),
            "group": np.concatenate([
                np.ones(len(active_change), dtype=float),
                np.zeros(len(sham_change), dtype=float),
            ]),
            "age": np.concatenate([active_age, sham_age]),
            "sex": np.concatenate([active_sex, sham_sex]),
            "education": np.concatenate([active_education, sham_education]),
        }).dropna()
        design = sm.add_constant(
            frame[["group", "age", "sex"]], has_constant="add"
        )
        fit = sm.OLS(frame["change"], design).fit()
        ci_low, ci_high = fit.conf_int().loc["group"]
        t_value = float(fit.tvalues["group"])
        row = {
            "panel": "Active versus sham",
            "analysis": "change GLM adjusted for age and sex",
            "n_active": int(np.isfinite(active_change).sum()),
            "n_sham": int(np.isfinite(sham_change).sum()),
            "active_mean_post_minus_pre": float(np.nanmean(active_change)),
            "sham_mean_post_minus_pre": float(np.nanmean(sham_change)),
            "effect_value": float(fit.params["group"]),
            "interaction_beta_active_minus_sham": float(fit.params["group"]),
            "interaction_CI95_low": float(ci_low),
            "interaction_CI95_high": float(ci_high),
            "interaction_t": t_value,
            "interaction_effect_size_partial_r": signed_partial_r(t_value, fit.df_resid),
            "interaction_df_resid": float(fit.df_resid),
            "p_uncorrected": float(fit.pvalues["group"]),
            "interaction_n": int(fit.nobs),
        }
        if modality == "regional":
            row["ROI"] = feature
        else:
            row["ROI_1"], row["ROI_2"] = feature
        rows.append(row)

    table = pd.DataFrame(rows)
    q_column = (
        "q_fdr_BH_within_26_regions"
        if modality == "regional"
        else "q_fdr_BH_within_168_edges"
    )
    table[q_column] = bh_fdr(table["p_uncorrected"])
    return table


longitudinal_regional_tables = []
longitudinal_edge_tables = []
for panel_label, result in [
    ("rTMS post-pre", rtms_result),
    ("iTBS post-pre", itbs_result),
    ("Sham post-pre", sham_result),
]:
    regional_table, edge_table = _paired_effect_tables(result, panel_label)
    longitudinal_regional_tables.append(regional_table)
    longitudinal_edge_tables.append(edge_table)

active_regional_delta = np.vstack([
    itbs_result["regional_delta"], rtms_result["regional_delta"]
])
active_edge_delta = np.vstack([
    itbs_result["edge_delta"], rtms_result["edge_delta"]
])
active_vs_sham_regional = _pooled_active_vs_sham_glm(
    active_regional_delta,
    sham_result["regional_delta"],
    ROI_COLUMNS,
    "regional",
)
active_vs_sham_edge = _pooled_active_vs_sham_glm(
    active_edge_delta,
    sham_result["edge_delta"],
    CROSS_EDGE_NAMES,
    "edge",
)
longitudinal_regional_tables.append(active_vs_sham_regional)
longitudinal_edge_tables.append(active_vs_sham_edge)

longitudinal_regional_all = pd.concat(
    longitudinal_regional_tables, ignore_index=True, sort=False
)
longitudinal_edge_all = pd.concat(
    longitudinal_edge_tables, ignore_index=True, sort=False
)

panel_order = [
    "rTMS post-pre",
    "iTBS post-pre",
    "Sham post-pre",
    "Active versus sham",
]
longitudinal_circular_panels = {}
for panel_label in panel_order:
    regional_table = longitudinal_regional_all[
        longitudinal_regional_all["panel"] == panel_label
    ]
    edge_table = longitudinal_edge_all[
        longitudinal_edge_all["panel"] == panel_label
    ]
    regional_lookup = regional_table.set_index("ROI")["effect_value"]
    edge_lookup = edge_table.set_index(["ROI_1", "ROI_2"])["effect_value"]
    longitudinal_circular_panels[panel_label] = {
        "regional": np.asarray(
            [regional_lookup.loc[roi] for roi in ROI_COLUMNS], dtype=float
        ),
        "edge": np.asarray(
            [edge_lookup.loc[pair] for pair in CROSS_EDGE_NAMES], dtype=float
        ),
        "regional_mask": None,
        "edge_mask": None,
        "stem": (
            panel_label.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("versus", "vs")
        ),
    }

# Shared robust scales allow direct visual comparison across the four panels.
longitudinal_node_vmax = float(np.nanquantile(np.abs(np.concatenate([
    item["regional"] for item in longitudinal_circular_panels.values()
])), 0.98))
longitudinal_edge_vmax = float(np.nanquantile(np.abs(np.concatenate([
    item["edge"] for item in longitudinal_circular_panels.values()
])), 0.98))
if not np.isfinite(longitudinal_node_vmax) or longitudinal_node_vmax <= 0:
    longitudinal_node_vmax = 1.0
if not np.isfinite(longitudinal_edge_vmax) or longitudinal_edge_vmax <= 0:
    longitudinal_edge_vmax = 1.0

longitudinal_data_file = (
    LONGITUDINAL_CIRCULAR_OUT
    / "corticolimbic_circular_longitudinal_effects_data.xlsx"
)
with pd.ExcelWriter(longitudinal_data_file, engine="openpyxl") as writer:
    longitudinal_regional_all.to_excel(
        writer, sheet_name="regional_effects", index=False
    )
    longitudinal_edge_all.to_excel(
        writer, sheet_name="connectivity_edges", index=False
    )
    pd.DataFrame([
        {
            "parameter": "rTMS_iTBS_sham_effect",
            "value": "mean subject-level post deviation minus pre deviation",
        },
        {
            "parameter": "active_vs_sham_effect",
            "value": (
                "OLS group beta on post-pre change; active=iTBS+rTMS; "
                "adjusted for age and sex"
            ),
        },
        {"parameter": "regional_display_rule", "value": "All 26 ROIs"},
        {"parameter": "edge_display_rule", "value": "All 168 cross-network edges"},
        {"parameter": "regional_shared_abs_scale_98pct", "value": longitudinal_node_vmax},
        {"parameter": "edge_shared_abs_scale_98pct", "value": longitudinal_edge_vmax},
        {"parameter": "positive_color", "value": CIRCULAR_POSITIVE},
        {"parameter": "negative_color", "value": CIRCULAR_NEGATIVE},
    ]).to_excel(writer, sheet_name="plot_settings", index=False)
style_excel_workbook(longitudinal_data_file)

# The established renderer reads its scales from these globals. Swap them only
# while drawing this figure, then restore the cross-sectional settings.
_previous_node_vmax = circular_node_vmax
_previous_edge_vmax = circular_edge_vmax
_previous_norm = circular_norm
circular_node_vmax = longitudinal_node_vmax
circular_edge_vmax = longitudinal_edge_vmax
circular_norm = TwoSlopeNorm(
    vmin=-longitudinal_node_vmax,
    vcenter=0,
    vmax=longitudinal_node_vmax,
)

fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6), facecolor="white")
for ax, (title, item) in zip(axes, longitudinal_circular_panels.items()):
    draw_corticolimbic_connectogram(
        ax,
        title,
        item["regional"],
        item["edge"],
        item["regional_mask"],
        item["edge_mask"],
    )
legend_handles = [
    Line2D(
        [0], [0], color=CIRCULAR_POSITIVE, linewidth=2.0,
        marker="s", markerfacecolor=CIRCULAR_POSITIVE, markeredgecolor="white",
        label="Positive post-pre change / active > sham",
    ),
    Line2D(
        [0], [0], color=CIRCULAR_NEGATIVE, linewidth=2.0,
        marker="s", markerfacecolor=CIRCULAR_NEGATIVE, markeredgecolor="white",
        label="Negative post-pre change / active < sham",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
    fontsize=8,
)
fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.12, wspace=0.28)

longitudinal_summary_stem = (
    LONGITUDINAL_CIRCULAR_OUT / "corticolimbic_circular_longitudinal_summary"
)
for extension, kwargs in [
    ("png", {"dpi": 600}),
    ("pdf", {}),
    ("svg", {}),
]:
    fig.savefig(
        longitudinal_summary_stem.with_suffix(f".{extension}"),
        facecolor="white",
        bbox_inches="tight",
        **kwargs,
    )
plt.close(fig)

longitudinal_individual_files = []
for title, item in longitudinal_circular_panels.items():
    fig, ax = plt.subplots(figsize=(4.1, 4.25), facecolor="white")
    draw_corticolimbic_connectogram(
        ax,
        title,
        item["regional"],
        item["edge"],
        item["regional_mask"],
        item["edge_mask"],
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.03)
    for extension, kwargs in [("png", {"dpi": 600}), ("pdf", {})]:
        path = LONGITUDINAL_CIRCULAR_OUT / (
            f"corticolimbic_circular_{item['stem']}.{extension}"
        )
        fig.savefig(path, facecolor="white", bbox_inches="tight", **kwargs)
        longitudinal_individual_files.append(path)
    plt.close(fig)

circular_node_vmax = _previous_node_vmax
circular_edge_vmax = _previous_edge_vmax
circular_norm = _previous_norm

print("Longitudinal circular summary:", longitudinal_summary_stem.with_suffix(".png"))
print("Longitudinal circular raw data:", longitudinal_data_file)
display(
    longitudinal_regional_all.groupby("panel")
    .agg(
        regional_mean_effect=("effect_value", "mean"),
        regional_min_effect=("effect_value", "min"),
        regional_max_effect=("effect_value", "max"),
    )
    .reindex(panel_order)
)


# Source cell 72
# Longitudinal circular-figure reproducibility checks.
assert list(longitudinal_circular_panels) == [
    "rTMS post-pre", "iTBS post-pre", "Sham post-pre", "Active versus sham"
]
assert all(len(item["regional"]) == 26 for item in longitudinal_circular_panels.values())
assert all(len(item["edge"]) == 168 for item in longitudinal_circular_panels.values())
assert len(active_regional_delta) == EXPECTED_PAIRED_ARMS["iTBS"] + EXPECTED_PAIRED_ARMS["rTMS"]
assert len(sham_result["regional_delta"]) == EXPECTED_PAIRED_ARMS["sham"]
assert (
    longitudinal_regional_all.groupby("panel").size().reindex(panel_order).tolist()
    == [26, 26, 26, 26]
)
assert (
    longitudinal_edge_all.groupby("panel").size().reindex(panel_order).tolist()
    == [168, 168, 168, 168]
)
for extension in ["png", "pdf", "svg"]:
    assert longitudinal_summary_stem.with_suffix(f".{extension}").exists()
assert longitudinal_data_file.exists()
print("Longitudinal circular connectogram checks passed.")


# Recreate the remaining analytical figure components from the bundled
# pseudonymized tables and bundled AAL3-144 atlas.
for script_name in [
    "13_plot_SGC_baseline_severity_adjusted_circle.py",
    "15_plot_left_MFG_SFG_MwC_glass_brain.py",
    "18_plot_sensitivity_adjusted_mPFC_striatopallidal_regional_SUVR_glass_brain.py",
    "19_plot_split_sensitivity_circuits_sampled_colours.py",
    "21_plot_nonTRD_TRD_significant_regional_glassbrain.py",
]:
    subprocess.run(
        [sys.executable, str(BASE / "scripts" / script_name)],
        cwd=BASE,
        env={**os.environ, "METAFL_ROOT": str(BASE)},
        check=True,
    )


# Source cell 74
created_files = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
required_final_files = [
    combined_file, sensitivity_file, spatial_png, spatial_pdf,
    boxplot_png, boxplot_pdf, boxplot_svg, boxplot_data_file,
    correlation_png, correlation_pdf, correlation_svg, correlation_data_file,
    circular_summary_png, circular_summary_pdf, circular_summary_svg,
    LONGITUDINAL_EDGE_FILE, mwc_heatmap_png, mwc_heatmap_pdf,
    mwc_heatmap_data_file, mwc_heatmap_input_csv,
    MWC_ANALYSIS_OUT / "validation_report.json",
    COMPLETE_ACTIVE_OUT / "Regional_SUVR_iTBS_rTMS_paired_p_lt_0p05_complete.csv",
    COMPLETE_ACTIVE_OUT / "MwC_iTBS_rTMS_paired_p_lt_0p05_complete.csv",
    COMPLETE_ACTIVE_OUT / "Regional_SUVR_and_MwC_iTBS_rTMS_paired_p_lt_0p05_complete.csv",
    COMPLETE_ACTIVE_OUT / "Regional_SUVR_selected_subject_level_values.csv",
    COMPLETE_ACTIVE_OUT / "MwC_selected_subject_level_values.csv",
    OUTPUT_ROOT / "11_SGC_26ROI_figures/figures/Figure_TRD_vs_nonTRD_significant_MwC_nilearn_glass_brain.png",
    OUTPUT_ROOT / "11_SGC_26ROI_figures/figures/corticolimbic_circular_26ROI_bilateral_SGC_TRD_vs_nonTRD_baseline_severity_adjusted.png",
    OUTPUT_ROOT / "13_sensitivity_mPFC_striatopallidal_glass_brain/figures/Figure_sensitivity_adjusted_bilateral_vmPFC_dmPFC_medialOFC_sampled_blue_nilearn_glass_brain.png",
    OUTPUT_ROOT / "13_sensitivity_mPFC_striatopallidal_glass_brain/figures/Figure_sensitivity_adjusted_bilateral_caudate_putamen_pallidum_sampled_pink_nilearn_glass_brain.png",
    OUTPUT_ROOT / "15_nonTRD_TRD_significant_regional_SUVR_glass_brain/figures/Figure_TRD_vs_nonTRD_significant_regional_SUVR_combined_nilearn_glass_brain.png",
]
assert all(path.exists() and path.stat().st_size > 0 for path in required_final_files)
assert all("msmtotal" not in column.lower() for table in [
    sensitivity_edge_all, sensitivity_regional_all, sensitivity_mwc_all,
] for column in table.columns)
print(f"Main-analysis scope audit passed: {len(created_files)} analytical outputs.")
