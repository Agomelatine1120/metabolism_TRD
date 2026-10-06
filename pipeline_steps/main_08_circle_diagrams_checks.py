"""Auto-extracted pipeline step: Main 08 — circle diagrams and reproducibility checks.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 36
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path as MplPath

# Mean-deviation vectors; all panels retain the same ROI and edge ordering.
circular_panels = {
    "nonTRD deviation": {
        "regional": trd1_result["ntrd_regional"].to_numpy(float).mean(axis=0),
        "edge": trd1_result["ntrd_edge"].mean(axis=0),
        "regional_mask": None,
        "edge_mask": None,
        "stem": "nonTRD_mean_deviation",
    },
    "TRD1 deviation": {
        "regional": trd1_result["case_regional"].to_numpy(float).mean(axis=0),
        "edge": trd1_result["case_edge"].mean(axis=0),
        "regional_mask": None,
        "edge_mask": None,
        "stem": "TRD1_mean_deviation",
    },
    "TRD2 deviation": {
        "regional": trd2_context["case_regional"].to_numpy(float).mean(axis=0),
        "edge": trd2_context["case_edge"].mean(axis=0),
        "regional_mask": None,
        "edge_mask": None,
        "stem": "TRD2_mean_deviation",
    },
    "TRD versus nonTRD": {
        "regional": (
            trd1_result["case_regional"].to_numpy(float).mean(axis=0)
            - trd1_result["ntrd_regional"].to_numpy(float).mean(axis=0)
        ),
        "edge": (
            trd1_result["case_edge"].mean(axis=0)
            - trd1_result["ntrd_edge"].mean(axis=0)
        ),
        "regional_mask": None,
        "edge_mask": None,
        "stem": "TRD1_vs_nonTRD_all",
    },
}

# Shared scales permit direct visual comparison across all four panels.
circular_node_vmax = float(np.nanquantile(np.abs(np.concatenate([
    item["regional"] for item in circular_panels.values()
])), 0.98))
circular_edge_vmax = float(np.nanquantile(np.abs(np.concatenate([
    item["edge"] for item in circular_panels.values()
])), 0.98))

# Nature-oriented, color-vision-conscious red/blue pairing.
CIRCULAR_NEGATIVE = "#2868A8"
CIRCULAR_POSITIVE = "#B64040"
CIRCULAR_NEUTRAL = "#F7F7F5"
CIRCULAR_GRAPHITE = "#30343B"
CIRCULAR_GUIDE = "#B8C0C8"
circular_cmap = LinearSegmentedColormap.from_list(
    "ibsc_nature_diverging",
    [CIRCULAR_NEGATIVE, "#C9DCEC", CIRCULAR_NEUTRAL, "#EBC9C3", CIRCULAR_POSITIVE],
    N=256,
)
circular_norm = TwoSlopeNorm(
    vmin=-circular_node_vmax, vcenter=0, vmax=circular_node_vmax
)

# Top semicircle: cortical; bottom semicircle: limbic.
cortical_angles = np.linspace(170, 10, len(CORTICAL_ROIS))
limbic_angles = np.linspace(190, 350, len(LIMBIC_ROIS))
roi_angles = {
    **dict(zip(CORTICAL_ROIS, cortical_angles)),
    **dict(zip(LIMBIC_ROIS, limbic_angles)),
}
roi_positions = {
    roi: np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    for roi, angle in roi_angles.items()
}
regional_index = {roi: index for index, roi in enumerate(ROI_COLUMNS)}
edge_index = {tuple(edge): index for index, edge in enumerate(CROSS_EDGE_NAMES)}


def draw_corticolimbic_connectogram(
    ax, title, regional_values, edge_values,
    regional_mask=None, edge_mask=None, panel_letter=None,
):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.52, 1.52)
    ax.set_ylim(-1.53, 1.63)
    ax.set_facecolor("white")

    # Quiet guide arcs separate node rings from the edge field.
    for start, end in [(5, 175), (185, 355)]:
        ax.add_patch(Wedge(
            (0, 0), 0.985, start, end, width=0.008,
            facecolor=CIRCULAR_GUIDE, edgecolor="none", zorder=0,
        ))

    # Draw weaker edges first so the strongest effects remain visible.
    visible_edges = []
    for edge, index in edge_index.items():
        value = float(edge_values[index])
        visible = edge_mask is None or edge in edge_mask
        if visible and np.isfinite(value) and abs(value) > 0:
            visible_edges.append((abs(value), edge[0], edge[1], value))
    visible_edges.sort()

    for magnitude, roi_1, roi_2, value in visible_edges:
        strength = min(magnitude / circular_edge_vmax, 1.0)
        color = CIRCULAR_POSITIVE if value > 0 else CIRCULAR_NEGATIVE
        start = roi_positions[roi_1] * 0.965
        end = roi_positions[roi_2] * 0.965
        path = MplPath(
            [start, start * 0.30, end * 0.30, end],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(PathPatch(
            path, facecolor="none", edgecolor=color,
            linewidth=0.28 + 1.00 * strength,
            alpha=0.09 + 0.48 * strength,
            zorder=1, capstyle="round",
        ))

    # Ring segments encode regional deviation using the same signed palette.
    for roi in CORTICAL_ROIS + LIMBIC_ROIS:
        angle = roi_angles[roi]
        value = float(regional_values[regional_index[roi]])
        visible = regional_mask is None or roi in regional_mask
        facecolor = (
            circular_cmap(circular_norm(np.clip(value, -circular_node_vmax, circular_node_vmax)))
            if visible else "#ECEFF1"
        )
        ax.add_patch(Wedge(
            (0, 0), 1.13, angle - 4.3, angle + 4.3, width=0.125,
            facecolor=facecolor, edgecolor="white", linewidth=0.8, zorder=4,
        ))

        # Signed anchor tick provides redundant encoding beyond ring color.
        tick_color = (
            CIRCULAR_POSITIVE if value > 0 else CIRCULAR_NEGATIVE
        ) if visible else "#B7BEC5"
        direction = np.array([
            np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        ])
        tick_start, tick_end = direction * 0.925, direction * 0.985
        ax.plot(
            [tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]],
            color=tick_color, linewidth=1.05, alpha=0.85, zorder=3,
        )

        # Radial labels avoid the central overlap produced by tangential labels.
        label_radius = 1.27
        x, y = label_radius * direction
        rotation = angle + (180 if 90 < angle < 270 else 0)
        horizontal_alignment = "right" if direction[0] < 0 else "left"
        ax.text(
            x, y, roi_short_label(roi),
            rotation=rotation, rotation_mode="anchor",
            ha=horizontal_alignment, va="center",
            fontsize=6.9, fontweight="semibold", color=CIRCULAR_GRAPHITE,
            zorder=5,
        )

    ax.set_title(title, fontsize=11, fontweight="bold", color="#22272E", pad=14)
    if panel_letter:
        ax.text(
            -1.47, 1.42, panel_letter,
            ha="left", va="top", fontsize=13, fontweight="bold", color="#111111",
        )


# Raw plotting data and explicit display masks.
node_rows, edge_rows = [], []
edge_stats_lookup = trd1_result["edge_all"].set_index(["ROI_1", "ROI_2"])
regional_stats_lookup = trd1_result["regional_all"].set_index("ROI")
for panel, item in circular_panels.items():
    for roi in ROI_COLUMNS:
        row = {
            "panel": panel,
            "network": "cortical" if roi in CORTICAL_ROIS else "limbic",
            "ROI": roi,
            "label": roi_short_label(roi),
            "value": float(item["regional"][regional_index[roi]]),
            "displayed": item["regional_mask"] is None or roi in item["regional_mask"],
        }
        if panel.startswith("TRD1 −"):
            row["p_group_uncorrected"] = float(regional_stats_lookup.loc[roi, "p_group_uncorrected"])
            row["q_fdr_BH_audit_only"] = float(regional_stats_lookup.loc[roi, "q_fdr_BH_26_regions_audit_only"])
        node_rows.append(row)
    for edge, index in edge_index.items():
        row = {
            "panel": panel,
            "ROI_1": edge[0],
            "ROI_2": edge[1],
            "value": float(item["edge"][index]),
            "displayed": item["edge_mask"] is None or edge in item["edge_mask"],
        }
        if panel.startswith("TRD1 −"):
            row["p_group_uncorrected"] = float(edge_stats_lookup.loc[edge, "p_group_uncorrected"])
            row["q_fdr_BH_audit_only"] = float(edge_stats_lookup.loc[edge, "q_fdr_BH_168_edges_audit_only"])
        edge_rows.append(row)

circular_data_file = FIGURE_OUT / "corticolimbic_circular_plot_data.xlsx"
with pd.ExcelWriter(circular_data_file, engine="openpyxl") as writer:
    pd.DataFrame(node_rows).to_excel(writer, sheet_name="regional_nodes", index=False)
    pd.DataFrame(edge_rows).to_excel(writer, sheet_name="connectivity_edges", index=False)
    pd.DataFrame([
        {"parameter": "regional_shared_abs_scale_98pct", "value": circular_node_vmax},
        {"parameter": "edge_shared_abs_scale_98pct", "value": circular_edge_vmax},
        {"parameter": "positive_color", "value": CIRCULAR_POSITIVE},
        {"parameter": "negative_color", "value": CIRCULAR_NEGATIVE},
        {"parameter": "contrast_edge_rule", "value": "All 168 cortico-limbic edges displayed"},
        {"parameter": "contrast_regional_rule", "value": "All 26 regional ROIs displayed"},
    ]).to_excel(writer, sheet_name="plot_settings", index=False)

# Combined wide figure: four panels in one row with generous separation.
fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6), facecolor="white")
for ax, (title, item) in zip(axes, circular_panels.items()):
    draw_corticolimbic_connectogram(
        ax, title, item["regional"], item["edge"],
        item["regional_mask"], item["edge_mask"],
    )
legend_handles = [
    Line2D(
        [0], [0], color=CIRCULAR_POSITIVE, linewidth=2.0,
        marker="s", markerfacecolor=CIRCULAR_POSITIVE, markeredgecolor="white",
        label="Positive deviation / higher in TRD1",
    ),
    Line2D(
        [0], [0], color=CIRCULAR_NEGATIVE, linewidth=2.0,
        marker="s", markerfacecolor=CIRCULAR_NEGATIVE, markeredgecolor="white",
        label="Negative deviation / lower in TRD1",
    ),
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=2,
    frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=8,
)
fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.12, wspace=0.28)

circular_summary_png = FIGURE_OUT / "corticolimbic_circular_summary.png"
circular_summary_pdf = FIGURE_OUT / "corticolimbic_circular_summary.pdf"
circular_summary_svg = FIGURE_OUT / "corticolimbic_circular_summary.svg"
fig.savefig(circular_summary_png, dpi=600, facecolor="white", bbox_inches="tight")
fig.savefig(circular_summary_pdf, facecolor="white", bbox_inches="tight")
fig.savefig(circular_summary_svg, facecolor="white", bbox_inches="tight")
plt.close(fig)

# Individual figures preserve label legibility for presentations and supplements.
circular_individual_files = []
for title, item in circular_panels.items():
    fig, ax = plt.subplots(figsize=(4.1, 4.25), facecolor="white")
    draw_corticolimbic_connectogram(
        ax, title, item["regional"], item["edge"],
        item["regional_mask"], item["edge_mask"],
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.03)
    png = FIGURE_OUT / f"corticolimbic_circular_{item['stem']}.png"
    pdf = FIGURE_OUT / f"corticolimbic_circular_{item['stem']}.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    circular_individual_files.extend([png, pdf])

print("Circular connectogram:", circular_summary_png)
print("Raw plotting data:", circular_data_file)


# Source cell 38
# Dynamic reproducibility checks for the high-quality PET-QC TRD2 analysis.
observed_active_results = {
    "TRD1_edge_p_lt_0p01": len(trd1_result["edge_reported"]),
    "TRD1_regional_p_lt_0p01": len(trd1_result["regional_reported"]),
    "iTBS_edge_prepost_p_lt_0p05": len(itbs_result["edge_selected"]),
    "iTBS_regional_prepost_p_lt_0p05": len(itbs_result["regional_selected"]),
    "rTMS_edge_prepost_p_lt_0p05": len(rtms_result["edge_selected"]),
    "rTMS_regional_prepost_p_lt_0p05": len(rtms_result["regional_selected"]),
    "sham_edge_prepost_p_lt_0p05": len(sham_result["edge_selected"]),
    "sham_regional_prepost_p_lt_0p05": len(sham_result["regional_selected"]),
    "active_vs_sham_interaction_p_lt_0p05": len(interaction_p05),
}
assert clinical["analysis_cohort"].value_counts().to_dict() == EXPECTED_COUNTS
assert clinical.loc[clinical["analysis_cohort"] == "TRD2", "analysis_arm"].value_counts().to_dict() == EXPECTED_ARMS
assert trd2_context["case_edge"].shape == (EXPECTED_COUNTS["TRD2"], 168)
assert trd2_context["case_regional"].shape == (EXPECTED_COUNTS["TRD2"], 26)
assert {result["arm"]: result["n"] for result in [itbs_result, rtms_result, sham_result]} == EXPECTED_PAIRED_ARMS
assert len(sensitivity_mwc_all) == 26
assert not (TRD2_OUT / "edge_all.xlsx").exists()
assert not (TRD2_OUT / "regional_all.xlsx").exists()
for output_file in [demographic_file, combined_file, sensitivity_file, interaction_file, spatial_png, spatial_pdf, regional_png, regional_pdf, edge_png, edge_pdf, boxplot_png, boxplot_pdf, correlation_png, correlation_pdf, circular_summary_png, circular_summary_pdf, circular_summary_svg, LONGITUDINAL_EDGE_FILE]:
    assert output_file.exists() and output_file.stat().st_size > 0, output_file
print("High-quality PET-QC TRD2 core analysis reproduced successfully.")
display(pd.DataFrame([observed_active_results]))
display(prepost_counts)
display(interaction_p05)
display(spatial_summary)
