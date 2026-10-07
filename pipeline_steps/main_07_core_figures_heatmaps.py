"""Auto-extracted pipeline step: Main 07 — core figures and SGC heatmaps.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 31
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

FIGURE_OUT = OUTPUT_ROOT / "04_figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)


def roi_short_label(roi):
    name = str(roi).split("_", 2)[2]
    hemisphere = "L" if name.endswith("_L") else ("R" if name.endswith("_R") else "")
    base = name[:-2] if hemisphere else name
    mapping = {
        "Frontal_Sup_2": "SFG",
        "Frontal_Mid_2": "MFG",
        "Frontal_Sup_Medial": "dmPFC",
        "Frontal_Med_Orb": "vmPFC",
        "OFCmed": "mOFC",
        "OFClat": "lOFC",
        "Hippocampus": "HPC",
        "ParaHippocampal": "PHG",
        "Amygdala": "AMY",
        "Caudate": "CAU",
        "Putamen": "PUT",
        "Pallidum": "PAL",
        "ACC_sub": "SGC",
    }
    label = mapping.get(base, base)
    if hemisphere:
        label = f"{'Lt' if hemisphere == 'L' else 'Rt'}.{label}"
    return label


def edge_mean_matrix(edge_values):
    # CROSS_EDGE_NAMES are ordered cortical-major, then limbic-minor.
    return np.nanmean(edge_values, axis=0).reshape(
        len(CORTICAL_ROIS), len(LIMBIC_ROIS)
    ).T


def regional_strip_matrix(regional_values):
    """Two rows on a 14-column canvas; the 12 cortical cells are right-aligned."""
    means = np.nanmean(np.asarray(regional_values, dtype=float), axis=0)
    positions = {roi: index for index, roi in enumerate(ROI_COLUMNS)}
    matrix = np.full((2, len(LIMBIC_ROIS)), np.nan, dtype=float)
    offset = len(LIMBIC_ROIS) - len(CORTICAL_ROIS)
    matrix[0, offset:] = [means[positions[roi]] for roi in CORTICAL_ROIS]
    matrix[1, :] = [means[positions[roi]] for roi in LIMBIC_ROIS]
    return matrix


cohort_edge_matrices = {
    "nTRD": edge_mean_matrix(trd1_result["ntrd_edge"]),
    "TRD1": edge_mean_matrix(trd1_result["case_edge"]),
    "TRD2": edge_mean_matrix(trd2_context["case_edge"]),
}
cohort_regional_matrices = {
    "nTRD": regional_strip_matrix(trd1_result["ntrd_regional"]),
    "TRD1": regional_strip_matrix(trd1_result["case_regional"]),
    "TRD2": regional_strip_matrix(trd2_context["case_regional"]),
}

shared_vmax = float(np.nanmax(np.abs(np.concatenate(
    [matrix.ravel() for matrix in cohort_edge_matrices.values()]
    + [matrix.ravel() for matrix in cohort_regional_matrices.values()]
))))
# Diverging palette sampled from the colorbar in Martins et al.,
# Nature Mental Health (2026), Fig. 4. The nine median RGB samples run
# from negative (blue) through zero (light gray) to positive (red).
HEATMAP_CMAP_NAME = "martins_2026_fig4_sampled"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    HEATMAP_CMAP_NAME,
    [
        "#2441B5", "#447AEC", "#6FA8FD", "#A4CAF8", "#D9D6D9",
        "#FEBCA5", "#FE8869", "#F6493F", "#BF0323",
    ],
    N=256,
)
cortical_labels = [roi_short_label(roi) for roi in CORTICAL_ROIS]
limbic_labels = [roi_short_label(roi) for roi in LIMBIC_ROIS]
cortical_offset = len(LIMBIC_ROIS) - len(CORTICAL_ROIS)

regional_data_file = FIGURE_OUT / "nTRD_TRD1_TRD2_regional26_strip_heatmap_data.xlsx"
with pd.ExcelWriter(regional_data_file, engine="openpyxl") as writer:
    for cohort in ["nTRD", "TRD1", "TRD2"]:
        matrix = cohort_regional_matrices[cohort]
        rows = []
        for network, rois, labels, values in [
            ("Cortical", CORTICAL_ROIS, cortical_labels, matrix[0, cortical_offset:]),
            ("Limbic", LIMBIC_ROIS, limbic_labels, matrix[1, :]),
        ]:
            rows.extend({
                "network": network, "position": position, "ROI": roi,
                "label": label, "mean_deviation": float(value),
            } for position, (roi, label, value) in enumerate(
                zip(rois, labels, values), start=1
            ))
        pd.DataFrame(rows).to_excel(writer, sheet_name=cohort, index=False)
    pd.DataFrame({
        "parameter": ["shared_vmin", "shared_vmax", "colormap"],
        "value": [-shared_vmax, shared_vmax, HEATMAP_CMAP_NAME],
    }).to_excel(writer, sheet_name="plot_settings", index=False)

fig = plt.figure(figsize=(10.8, 5.6), facecolor="white")
left, right = 0.18, 0.945
width = right - left
heatmap_height = 0.238
gap_ntrd_trd1 = 0.075
gap_trd1_trd2 = 0.032
trd2_y = 0.06
trd1_y = trd2_y + heatmap_height + gap_trd1_trd2
ntrd_y = trd1_y + heatmap_height + gap_ntrd_trd1
axes = [
    fig.add_axes([left, ntrd_y, width, heatmap_height]),
    fig.add_axes([left, trd1_y, width, heatmap_height]),
    fig.add_axes([left, trd2_y, width, heatmap_height]),
]

for ax, cohort in zip(axes, ["nTRD", "TRD1", "TRD2"]):
    show_names = cohort == "nTRD"
    sns.heatmap(
        cohort_regional_matrices[cohort],
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=-shared_vmax,
        vmax=shared_vmax,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        xticklabels=(limbic_labels if show_names else [""] * len(limbic_labels)),
        yticklabels=False,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Align cohort, cortical, and limbic labels to one common left edge.
    label_x = -0.105
    ax.text(
        label_x,
        1.025,
        cohort,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        label_x,
        0.75,
        "Cortical",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=14,
        clip_on=False,
    )
    ax.text(
        label_x,
        0.25,
        "Limbic",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=14,
        clip_on=False,
    )

    if show_names:
        for column, label in enumerate(cortical_labels, start=cortical_offset):
            ax.text(
                column + 0.5,
                -0.04,
                label,
                ha="left",
                va="bottom",
                fontsize=12,
                rotation=20,
                rotation_mode="anchor",
                clip_on=False,
            )
        ax.set_xticklabels(
            limbic_labels,
            rotation=20,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
        ax.tick_params(axis="x", pad=1)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", left=False)

# Use a longer shared colorbar while preserving identical heatmap widths.
colorbar_ax = fig.add_axes([0.962, 0.06, 0.012, 0.30])
fig.colorbar(axes[-1].collections[0], cax=colorbar_ax)

regional_png = FIGURE_OUT / "nTRD_TRD1_TRD2_regional26_strip_heatmap.png"
regional_pdf = FIGURE_OUT / "nTRD_TRD1_TRD2_regional26_strip_heatmap.pdf"
fig.savefig(regional_png, dpi=300, facecolor="white")
fig.savefig(regional_pdf, dpi=300, facecolor="white")
plt.close(fig)


# MwC heatmap: exact Regional SUVR strip-heatmap renderer and geometry; only input changes.
cohort_mwc_matrices = {
    cohort: regional_strip_matrix(values)
    for cohort, values in mwc_group_values.items()
}
mwc_shared_vmax = float(np.nanmax(np.abs(np.concatenate(
    [matrix.ravel() for matrix in cohort_mwc_matrices.values()]
))))
mwc_heatmap_data_file = FIGURE_OUT / "nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap_data.xlsx"
mwc_heatmap_input_csv = FIGURE_OUT / "nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap_input_values.csv"
mwc_heatmap_input_rows = []
with pd.ExcelWriter(mwc_heatmap_data_file, engine="openpyxl") as writer:
    for cohort in ["nTRD", "TRD1", "TRD2"]:
        matrix = cohort_mwc_matrices[cohort]
        cohort_rows = []
        for network, rois, labels, values in [
            ("Cortical", CORTICAL_ROIS, cortical_labels, matrix[0, cortical_offset:]),
            ("Limbic", LIMBIC_ROIS, limbic_labels, matrix[1, :]),
        ]:
            for position, (roi, label, value) in enumerate(
                zip(rois, labels, values), start=1
            ):
                cohort_rows.append({
                    "network": network, "position": position, "ROI": roi,
                    "label": label,
                    "mean_age_sex_adjusted_MwC_deviation_z": float(value),
                })
                mwc_heatmap_input_rows.append({
                    "cohort": cohort, "baseline_n": len(mwc_group_values[cohort]),
                    "network": network, "position": position, "ROI": roi,
                    "label": label, "mean_age_sex_adjusted_MwC_deviation_z": value,
                })
        pd.DataFrame(cohort_rows).to_excel(writer, sheet_name=cohort, index=False)
    pd.DataFrame({
        "parameter": ["shared_vmin", "shared_vmax", "colormap", "renderer"],
        "value": [-mwc_shared_vmax, mwc_shared_vmax, HEATMAP_CMAP_NAME,
                  "exact Regional SUVR strip-heatmap renderer; input-only substitution"],
    }).to_excel(writer, sheet_name="plot_settings", index=False)
pd.DataFrame(mwc_heatmap_input_rows).to_csv(mwc_heatmap_input_csv, index=False)

fig = plt.figure(figsize=(10.8, 5.6), facecolor="white")
left, right = 0.18, 0.945
width = right - left
heatmap_height = 0.238
gap_ntrd_trd1 = 0.075
gap_trd1_trd2 = 0.032
trd2_y = 0.06
trd1_y = trd2_y + heatmap_height + gap_trd1_trd2
ntrd_y = trd1_y + heatmap_height + gap_ntrd_trd1
mwc_axes = [
    fig.add_axes([left, ntrd_y, width, heatmap_height]),
    fig.add_axes([left, trd1_y, width, heatmap_height]),
    fig.add_axes([left, trd2_y, width, heatmap_height]),
]

for ax, cohort in zip(mwc_axes, ["nTRD", "TRD1", "TRD2"]):
    show_names = cohort == "nTRD"
    sns.heatmap(
        cohort_mwc_matrices[cohort], ax=ax, cmap=HEATMAP_CMAP,
        vmin=-mwc_shared_vmax, vmax=mwc_shared_vmax, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        xticklabels=(limbic_labels if show_names else [""] * len(limbic_labels)),
        yticklabels=False, linewidths=0.5, linecolor="white", cbar=False,
    )
    ax.set_title(""); ax.set_xlabel(""); ax.set_ylabel("")
    label_x = -0.105
    ax.text(label_x, 1.025, cohort, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, fontweight="bold", clip_on=False)
    ax.text(label_x, 0.75, "Cortical", transform=ax.transAxes, ha="left", va="center",
            fontsize=14, clip_on=False)
    ax.text(label_x, 0.25, "Limbic", transform=ax.transAxes, ha="left", va="center",
            fontsize=14, clip_on=False)
    if show_names:
        for column, label in enumerate(cortical_labels, start=cortical_offset):
            ax.text(column + 0.5, -0.04, label, ha="left", va="bottom", fontsize=12,
                    rotation=20, rotation_mode="anchor", clip_on=False)
        ax.set_xticklabels(limbic_labels, rotation=20, ha="right",
                           rotation_mode="anchor", fontsize=12)
        ax.tick_params(axis="x", pad=1)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", left=False)

mwc_colorbar_ax = fig.add_axes([0.962, 0.06, 0.012, 0.30])
fig.colorbar(mwc_axes[-1].collections[0], cax=mwc_colorbar_ax)
mwc_heatmap_png = FIGURE_OUT / "nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap.png"
mwc_heatmap_pdf = FIGURE_OUT / "nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap.pdf"
fig.savefig(mwc_heatmap_png, dpi=300, facecolor="white")
fig.savefig(mwc_heatmap_pdf, dpi=300, facecolor="white")
plt.close(fig)

edge_data_file = FIGURE_OUT / "nTRD_TRD1_TRD2_corticolimbic12x14_heatmap_data.xlsx"
with pd.ExcelWriter(edge_data_file, engine="openpyxl") as writer:
    for cohort in ["nTRD", "TRD1", "TRD2"]:
        pd.DataFrame(
            cohort_edge_matrices[cohort], index=limbic_labels, columns=cortical_labels
        ).to_excel(writer, sheet_name=cohort)
    pd.DataFrame({
        "parameter": ["shared_vmin", "shared_vmax", "colormap"],
        "value": [-shared_vmax, shared_vmax, HEATMAP_CMAP_NAME],
    }).to_excel(writer, sheet_name="plot_settings", index=False)

fig, axes = plt.subplots(1, 3, figsize=(26, 8.5), constrained_layout=False)
fig.patch.set_facecolor("white")
for ax, cohort in zip(axes, ["nTRD", "TRD1", "TRD2"]):
    # Present cortical ROIs as rows and limbic ROIs as the horizontal axis.
    matrix = cohort_edge_matrices[cohort].T
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=-shared_vmax,
        vmax=shared_vmax,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        xticklabels=limbic_labels,
        yticklabels=cortical_labels,
        linewidths=0.4,
        linecolor="white",
        cbar=False,
    )
    ax.set_title("")
    ax.text(
        0,
        1.015,
        cohort,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=19,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        limbic_labels,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=13,
    )
    ax.set_yticklabels(cortical_labels, rotation=0, fontsize=13)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)

fig.subplots_adjust(left=0.075, right=0.955, top=0.91, bottom=0.20, wspace=0.30)
colorbar_ax = fig.add_axes([0.968, 0.25, 0.012, 0.50])
fig.colorbar(axes[-1].collections[0], cax=colorbar_ax)
fig.suptitle(
    "Cortico-limbic 12×14 mean deviation (HC reference)",
    fontsize=20,
    fontweight="bold",
    y=0.985,
)

edge_png = FIGURE_OUT / "nTRD_TRD1_TRD2_corticolimbic12x14_heatmap.png"
edge_pdf = FIGURE_OUT / "nTRD_TRD1_TRD2_corticolimbic12x14_heatmap.pdf"
fig.savefig(edge_png, dpi=300, facecolor="white")
fig.savefig(edge_pdf, dpi=300, facecolor="white")
plt.close(fig)

print("Regional heatmap:", regional_png)
print("MwC heatmap (same renderer):", mwc_heatmap_png)
print("MwC heatmap input values:", mwc_heatmap_input_csv)
print("Connectivity heatmap:", edge_png)


# Source cell 34
from matplotlib.patches import Patch


RESULT_BY_ARM = {"rTMS": rtms_result, "iTBS": itbs_result}

# Figure panels are determined solely from the formal paired tests. No ROI or
# edge is preselected: every active-arm 12 x 14 edge with uncorrected paired
# p < .01 is plotted, and no edge failing that threshold is plotted.
SELECTED_LONGITUDINAL_EDGES = []
_cortical_order = {roi: index for index, roi in enumerate(CORTICAL_ROIS)}
_limbic_order = {roi: index for index, roi in enumerate(LIMBIC_ROIS)}
for arm in ["rTMS", "iTBS"]:
    selected = (
        RESULT_BY_ARM[arm]["edge_all"]
        .loc[lambda frame: frame["paired_p_uncorrected"] < LONGITUDINAL_FIGURE_P_THRESHOLD]
        .assign(
            _cortical_order=lambda frame: frame["ROI_1"].map(_cortical_order),
            _limbic_order=lambda frame: frame["ROI_2"].map(_limbic_order),
        )
        .sort_values(["_cortical_order", "_limbic_order"])
    )
    SELECTED_LONGITUDINAL_EDGES.extend(
        {
            "arm": arm,
            "feature": (row.ROI_1, row.ROI_2),
            "paired_p_uncorrected": float(row.paired_p_uncorrected),
        }
        for row in selected.itertuples()
    )
assert SELECTED_LONGITUDINAL_EDGES, "No active-arm edge met paired p < .01"
for specification in SELECTED_LONGITUDINAL_EDGES:
    feature = specification["feature"]
    if feature not in CROSS_EDGE_NAMES:
        raise ValueError(f"Unknown cortico-limbic edge: {specification}")


def feature_values(result, feature, timepoint):
    if feature not in CROSS_EDGE_NAMES:
        raise ValueError(f"Unknown Figure 5 feature: {feature}")
    return result[f"edge_{timepoint}"][:, CROSS_EDGE_NAMES.index(feature)]


FIGURE_ROI_LABELS = {
    "AAL3_151_ACC_sub_L": "Lt.SGC",
    "AAL3_152_ACC_sub_R": "Rt.SGC",
}


def feature_title(feature):
    left = FIGURE_ROI_LABELS.get(feature[0], roi_short_label(feature[0]))
    right = FIGURE_ROI_LABELS.get(feature[1], roi_short_label(feature[1]))
    return f"{left}-{right}"


panel_order = pd.DataFrame([
    {
        "row_source": specification["arm"],
        "panel_position": position,
        "feature_type": "edge",
        "ROI_1_or_region": specification["feature"][0],
        "ROI_2": specification["feature"][1],
        "feature_label": feature_title(specification["feature"]),
        "paired_p_uncorrected": specification["paired_p_uncorrected"],
        "selection_note": (
            "all and only active-arm edges with paired p < 0.01; "
            "ordered by arm, cortical ROI, and limbic ROI"
        ),
    }
    for position, specification in enumerate(SELECTED_LONGITUDINAL_EDGES, start=1)
])


# Exact geometry and blue/orange palette used by the active pre-post
# five-metric raincloud figure later in this notebook.
EDGE_RAINCLOUD_SEED = 20260830
EDGE_TIME_COLORS = {"Pre": "#73BCED", "Post": "#E8A85C"}
EDGE_TIME_FILL_COLORS = {"Pre": "#ACCFE2", "Post": "#E0C5A1"}
EDGE_POSITIONS = {"Pre": 0.0, "Post": 0.80}
EDGE_RAINCLOUD_WIDTH = 42.0  # preserve the original 12,600-pixel canvas at 300 dpi
EDGE_STAR_GROUP_OFFSET = 0.24  # visually separate yellow and blue star clusters


def edge_p_text(p_value):
    return r"$P < 0.001$" if p_value < 0.001 else rf"$P = {p_value:.3f}$"


def edge_density_y(values):
    values = np.asarray(values, dtype=float)
    spread = float(np.ptp(values))
    pad = max(0.12 * spread, np.std(values, ddof=1) * 0.25, 1e-6)
    y_values = np.linspace(values.min() - pad, values.max() + pad, 256)
    density = stats.gaussian_kde(values)(y_values)
    return y_values, density / density.max()


def edge_add_bracket(ax, x1, x2, y, height, label):
    ax.plot(
        [x1, x1, x2, x2], [y, y + height, y + height, y],
        color="#4D4D4D", linewidth=1.25, clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2, y + height * 1.35, label,
        ha="center", va="bottom", fontsize=16, color="#333333",
    )


def edge_star_count(p_value):
    if p_value < 0.005:
        return 3
    if p_value < 0.01:
        return 2
    if p_value < 0.05:
        return 1
    return 0


def draw_edge_prepost_raincloud(ax, feature, source_arm, seed):
    rng = np.random.default_rng(seed)
    results = {"iTBS": itbs_result, "rTMS": rtms_result, "sham": sham_result}
    arm_centers = {"iTBS": 0.0, "rTMS": 0.67, "sham": 1.34}
    time_offsets = {"Pre": -0.15, "Post": 0.15}
    arrays_by_arm = {}
    paired_by_arm = {}

    for arm, result in results.items():
        arrays = {
            "Pre": np.asarray(feature_values(result, feature, "pre"), dtype=float),
            "Post": np.asarray(feature_values(result, feature, "post"), dtype=float),
        }
        arrays = {key: values[np.isfinite(values)] for key, values in arrays.items()}
        arrays_by_arm[arm] = arrays
        paired_by_arm[arm] = paired_stats(arrays["Pre"], arrays["Post"])

        for key in ["Pre", "Post"]:
            position = arm_centers[arm] + time_offsets[key]
            values = arrays[key]
            color = EDGE_TIME_COLORS[key]
            cloud_x = position - 0.105 + rng.normal(0, 0.018, len(values))
            ax.scatter(
                cloud_x, values, s=14, color=color, alpha=0.80,
                edgecolors="none", zorder=3,
            )

            box = ax.boxplot(
                [values], positions=[position], widths=0.065, patch_artist=True,
                showfliers=False, manage_ticks=False,
                medianprops={"color": "#222222", "linewidth": 0.9},
                whiskerprops={"color": "#555555", "linewidth": 0.75},
                capprops={"color": "#555555", "linewidth": 0.75},
                boxprops={"edgecolor": "#555555", "linewidth": 0.75}, zorder=4,
            )
            box["boxes"][0].set_facecolor("white")
            box["boxes"][0].set_alpha(0.86)
            ax.scatter(
                [position], [np.mean(values)], marker="D", s=12,
                facecolor=color, edgecolor="white", linewidth=0.45, zorder=5,
            )

            y_values, density = edge_density_y(values)
            right = position + 0.035 + 0.15 * density
            ax.fill_betweenx(
                y_values, position + 0.035, right,
                facecolor=EDGE_TIME_FILL_COLORS[key], alpha=1.0,
                linewidth=0, edgecolor="none", zorder=1,
            )

    all_values = np.concatenate([
        arrays_by_arm[arm][timepoint]
        for arm in ["iTBS", "rTMS", "sham"]
        for timepoint in ["Pre", "Post"]
    ])
    data_min, data_max = float(all_values.min()), float(all_values.max())
    span = data_max - data_min
    if span == 0:
        span = max(abs(data_max), 1.0)

    bracket_y = data_max + 0.22 * span
    bracket_height = 0.018 * span
    star_y = data_max + 0.12 * span
    for arm in ["iTBS", "rTMS", "sham"]:
        pre_position = arm_centers[arm] + time_offsets["Pre"]
        post_position = arm_centers[arm] + time_offsets["Post"]
        p_value = paired_by_arm[arm]["paired_p_uncorrected"]
        ax.plot(
            [pre_position, pre_position, post_position, post_position],
            [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
            color="#4D4D4D", linewidth=0.8, clip_on=False,
        )
        ax.text(
            arm_centers[arm], bracket_y + bracket_height * 1.35,
            edge_p_text(p_value), ha="center", va="bottom",
            fontsize=22.8, color="#333333",
        )
        star_count = edge_star_count(p_value)
        if star_count:
            offsets = (np.arange(star_count) - (star_count - 1) / 2) * 0.105
            for key, opposite in (("Pre", "Post"), ("Post", "Pre")):
                star_color = EDGE_TIME_COLORS[opposite]
                star_center = arm_centers[arm] + (
                    -EDGE_STAR_GROUP_OFFSET if key == "Pre" else EDGE_STAR_GROUP_OFFSET
                )
                ax.scatter(
                    star_center + offsets,
                    np.full(star_count, star_y), marker="*", s=300,
                    facecolors=star_color, edgecolors=star_color,
                    linewidths=0.6, zorder=8,
                )

    ax.set_xlim(-0.42, 1.76)
    ax.set_ylim(data_min - 0.14 * span, data_max + 0.34 * span)
    ax.set_xticks([0.0, 0.67, 1.34], ["iTBS", "rTMS", "sham"])
    ax.set_title(
        f"{feature_title(feature)}",
        pad=10, fontsize=17, fontweight="semibold",
    )
    ax.grid(axis="y", color="#D5D8DC", linewidth=0.8, alpha=0.70)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="x", labelsize=26, length=0, pad=5)
    ax.tick_params(axis="y", labelsize=13, length=4)
    return arrays_by_arm, paired_by_arm

boxplot_raw_rows, paired_test_rows = [], []
fig, axes = plt.subplots(
    1, len(SELECTED_LONGITUDINAL_EDGES), figsize=(EDGE_RAINCLOUD_WIDTH, 8.5),
    gridspec_kw={"wspace": 0.16}, facecolor="white",
)
for index, (ax, specification) in enumerate(zip(axes, SELECTED_LONGITUDINAL_EDGES)):
    feature = specification["feature"]
    arrays_by_arm, paired_by_arm = draw_edge_prepost_raincloud(
        ax, feature, specification["arm"], EDGE_RAINCLOUD_SEED + index,
    )
    ax.set_ylabel(
        "Metabolic connectivity deviation (HC z)" if index == 0 else "",
        fontsize=15,
    )
    for arm, result in {"iTBS": itbs_result, "rTMS": rtms_result, "sham": sham_result}.items():
        paired = paired_by_arm[arm]
        paired_test_rows.append({
            "row_source": specification["arm"],
            "feature_type": "edge",
            "ROI_1_or_region": feature[0],
            "ROI_2": feature[1],
            "feature_label": feature_title(feature),
            "arm": arm,
            **paired,
            "star_p_lt_0p05": paired["paired_p_uncorrected"] < PREPOST_P_THRESHOLD,
        })
        for timepoint, values in arrays_by_arm[arm].items():
            for subject_key, value in zip(result["subject_keys"], values):
                boxplot_raw_rows.append({
                    "row_source": specification["arm"],
                    "feature_type": "edge",
                    "ROI_1_or_region": feature[0],
                    "ROI_2": feature[1],
                    "feature_label": feature_title(feature),
                    "arm": arm,
                    "subject_key": subject_key,
                    "session": timepoint.lower(),
                    "deviation": float(value),
                    "TRD2_definition": "joint_PET_fMRI_TRD2_n86_iTBS31_rTMS28_sham27",
                })

legend_handles = [
    Patch(facecolor=EDGE_TIME_COLORS["Pre"], edgecolor="none", label="Pre"),
    Patch(facecolor=EDGE_TIME_COLORS["Post"], edgecolor="none", label="Post"),
]
fig.legend(
    handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
    ncol=2, frameon=False, fontsize=16, handlelength=1.75, handleheight=1.75,
    columnspacing=2.4, handletextpad=0.55,
)
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.16, top=0.82)

boxplot_png = FIGURE_OUT / "TRD2_prepost_raincloud.png"
boxplot_pdf = FIGURE_OUT / "TRD2_prepost_raincloud.pdf"
boxplot_svg = FIGURE_OUT / "TRD2_prepost_raincloud.svg"
boxplot_data_file = FIGURE_OUT / "TRD2_prepost_raincloud_data.xlsx"
fig.savefig(boxplot_png, dpi=300, facecolor="white")
fig.savefig(boxplot_pdf, facecolor="white")
fig.savefig(boxplot_svg, facecolor="white")
plt.close(fig)
with pd.ExcelWriter(boxplot_data_file, engine="openpyxl") as writer:
    pd.DataFrame(boxplot_raw_rows).to_excel(writer, sheet_name="raw_long", index=False)
    panel_order.to_excel(writer, sheet_name="panel_order", index=False)
    pd.DataFrame(paired_test_rows).to_excel(writer, sheet_name="paired_tests", index=False)
    pd.DataFrame([{
        "TRD2_n": EXPECTED_COUNTS["TRD2"],
        "iTBS_n": EXPECTED_ARMS["iTBS"],
        "rTMS_n": EXPECTED_ARMS["rTMS"],
        "sham_n": EXPECTED_ARMS["sham"],
        "definition": "high-quality PET-QC cohort",
        "figure_rule": "all active-arm edges with paired p<0.01 at a fixed total width; iTBS/rTMS/sham each shown as paired pre-post mini-rainclouds",
        "edge_selection_note": (
            "all and only rTMS/iTBS edges with paired p<0.01; "
            "current data yield eight panels"
        ),
    }]).to_excel(writer, sheet_name="analysis_notes", index=False)



# Baseline-feature-correlation visual grammar: gray confidence shadow,
# colored regression line, and pale points with matching colored outlines.
# Canonical figure palette sampled from the supplied reference.
CORRELATION_PALETTES = {
    "green_reference": {
        # Pixel-sampled from the supplied figure reference.
        "dark": "#009B6E",
        "light": "#D0F4E6",
        "ci_color": "#DEDEDE",
        "ci_alpha": 1.00,
        "marker_linewidth": 4.0,
        "line_width": 6.0,
        "description": "green outline/line with pale-green points and opaque gray CI shadow",
    },
}


correlation_raw_rows = []
correlation_panel_data = []
for specification in SELECTED_LONGITUDINAL_EDGES:
    result = RESULT_BY_ARM[specification["arm"]]
    feature = specification["feature"]
    delta = feature_values(result, feature, "post") - feature_values(result, feature, "pre")
    hdrs = result["hdrs_improvement"]
    r_value, p_value, n_used = partial_spearman_age_sex(
        delta, hdrs, result["age"], result["sex"]
    )
    mask = np.isfinite(hdrs) & np.isfinite(delta)
    correlation_panel_data.append({
        "result": result,
        "feature": feature,
        "delta": delta[mask],
        "hdrs": hdrs[mask],
        "r_value": r_value,
        "p_value": p_value,
        "n_used": n_used,
    })
    for subject_key, x_value, y_value in zip(
        np.asarray(result["subject_keys"])[mask], delta[mask], hdrs[mask]
    ):
        correlation_raw_rows.append({
            "row_source": result["arm"],
            "feature_type": "edge",
            "ROI_1_or_region": feature[0],
            "ROI_2": feature[1],
            "feature_label": feature_title(feature),
            "subject_key": subject_key,
            "HDRS_improvement_pre_minus_post": float(y_value),
            "post_minus_pre_deviation": float(x_value),
            "partial_Spearman_r_age_sex": r_value,
            "partial_Spearman_p_age_sex": p_value,
            "n_used": n_used,
        })



def draw_hdrs_correlation(
    ax, panel_data, palette, panel_index, stats_fontsize=24
):
    result = panel_data["result"]
    feature = panel_data["feature"]
    delta = panel_data["delta"]
    hdrs = panel_data["hdrs"]
    r_value = panel_data["r_value"]
    p_value = panel_data["p_value"]

    # Seaborn estimates the regression and confidence interval once. The CI,
    # line, point fill, and marker outline are then styled from the palette.
    ci_color = palette["ci_color"]
    ci_alpha = palette["ci_alpha"]
    marker_linewidth = palette["marker_linewidth"]
    regression_line_width = palette["line_width"]
    ci_collection_start = len(ax.collections)
    sns.regplot(
        x=delta, y=hdrs, ax=ax, ci=95,
        scatter=False, color=ci_color,
        line_kws={"lw": regression_line_width}, seed=20260901 + panel_index,
    )
    if not ax.lines:
        raise RuntimeError("Regression line was not created.")
    ax.lines[-1].set_color(palette["dark"])
    for collection in ax.collections[ci_collection_start:]:
        collection.set_facecolor(ci_color)
        collection.set_edgecolor("none")
        collection.set_alpha(ci_alpha)
    ax.scatter(
        delta, hdrs, s=148,
        facecolors=palette["light"],
        edgecolors=palette["dark"],
        linewidths=marker_linewidth, alpha=1.0, zorder=4,
    )
    ax.set_title(
        f"{feature_title(feature)}",
        fontsize=31.5, fontweight="bold", pad=18,
    )
    star = "*" if p_value < 0.05 else ""
    ax.text(
        0.04, 0.96, f"r = {r_value:.2f}, p = {p_value:.3f}{star}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=stats_fontsize, fontweight="semibold", zorder=7,
    )
    ax.set_facecolor("white")
    ax.set_box_aspect(1)
    ax.grid(False)
    ax.set_xlabel("Edge change (post − pre, HC z)", fontsize=16)
    ax.set_ylabel(
        "HDRS improvement (pre − post)" if panel_index == 0 else "",
        fontsize=25.5,
    )
    ax.tick_params(axis="both", labelsize=14, length=4, color="#666666")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")


def save_hdrs_correlation_panel(
    palette_name, filename_stem, panel_data_list=None, stats_fontsize=24
):
    palette = CORRELATION_PALETTES[palette_name]
    if panel_data_list is None:
        panel_data_list = correlation_panel_data
    fig, axes = plt.subplots(
        1, len(panel_data_list), figsize=(42.0, 9.5), sharey=True,
        gridspec_kw={"wspace": 0.18}, facecolor="white",
    )
    for panel_index, (ax, panel_data) in enumerate(zip(axes, panel_data_list)):
        draw_hdrs_correlation(
            ax, panel_data, palette, panel_index,
            stats_fontsize=stats_fontsize,
        )
    # Reserve an empty band above the highest node for the enlarged r/p line.
    shared_y_min, shared_y_max = axes[0].get_ylim()
    shared_y_span = shared_y_max - shared_y_min
    for ax in axes:
        ax.set_ylim(shared_y_min, shared_y_max + 0.22 * shared_y_span)
    fig.subplots_adjust(
        left=0.045, right=0.995, top=0.80, bottom=0.13, wspace=0.18
    )
    output_paths = {
        suffix: FIGURE_OUT / f"{filename_stem}.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    fig.savefig(output_paths["png"], dpi=300, facecolor="white")
    fig.savefig(output_paths["pdf"], facecolor="white")
    fig.savefig(output_paths["svg"], facecolor="white")
    plt.close(fig)
    return output_paths



# Canonical partial-Spearman panel: age and sex are the only covariates.
green_paths = save_hdrs_correlation_panel(
    "green_reference",
    "TRD2_prepost_HDRS_correlation_panel_green_reference",
    panel_data_list=correlation_panel_data,
    stats_fontsize=31.2,
)
correlation_png = green_paths["png"]
correlation_pdf = green_paths["pdf"]
correlation_svg = green_paths["svg"]

age_sex_panel_statistics = pd.DataFrame([
    {
        "panel_position": position,
        "arm": panel["result"]["arm"],
        "feature_label": feature_title(panel["feature"]),
        "ROI_1": panel["feature"][0],
        "ROI_2": panel["feature"][1],
        "partial_Spearman_r_age_sex": panel["r_value"],
        "partial_Spearman_p_age_sex": panel["p_value"],
        "n_used": panel["n_used"],
        "covariates": "age; sex",
    }
    for position, panel in enumerate(correlation_panel_data, start=1)
])
age_sex_panel_statistics["BH_FDR_q_across_displayed_edges"] = bh_fdr(
    age_sex_panel_statistics["partial_Spearman_p_age_sex"]
)
age_sex_panel_statistics["uncorrected_p_lt_0p05"] = (
    age_sex_panel_statistics["partial_Spearman_p_age_sex"] < 0.05
)
age_sex_panel_statistics["BH_FDR_q_lt_0p05"] = (
    age_sex_panel_statistics["BH_FDR_q_across_displayed_edges"] < 0.05
)

correlation_data_file = (
    FIGURE_OUT / "TRD2_prepost_HDRS_correlation_panel_green_reference_data.xlsx"
)
with pd.ExcelWriter(correlation_data_file, engine="openpyxl") as writer:
    age_sex_panel_statistics.to_excel(writer, sheet_name="statistics", index=False)
    pd.DataFrame(correlation_raw_rows).to_excel(writer, sheet_name="raw_data", index=False)
    panel_order.to_excel(writer, sheet_name="panel_order", index=False)
    pd.DataFrame([{
        "method": "partial Spearman correlation",
        "covariates": "age; sex",
        "multiple_testing": "BH-FDR across all displayed paired-p<0.01 edges",
        "visualization_note": "points and fitted line show observed values; r and p are partial Spearman statistics adjusted for age and sex",
        "figure_palette": "green reference; canonical HDRS correlation version",
        "marker_linewidth": 4.0,
        "regression_line_width": 6.0,
        "statistics_fontsize": 31.2,
    }]).to_excel(writer, sheet_name="analysis_notes", index=False)

print("TRD2 pre-post raincloud:", boxplot_png)
print("Canonical green HDRS correlation panel (partial Spearman; age/sex):", correlation_png)
display(age_sex_panel_statistics)
