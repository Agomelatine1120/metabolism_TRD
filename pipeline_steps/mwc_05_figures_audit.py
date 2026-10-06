"""Auto-extracted pipeline step: MwC 05 — final raincloud figures and scope audit.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 14
COLORS = {"nTRD": "#44853E", "TRD1": "#83328D"}
GROUPS = ["nTRD", "TRD1"]
GROUP_LABELS = {"nTRD": "nTRD", "TRD1": "TRD"}
GROUP_SPACING = 0.80
GROUP_POSITIONS = {group: index * GROUP_SPACING for index, group in enumerate(GROUPS)}
P_VALUE_FONTSIZE = 10.5
YLABEL_FONTSIZE = 12.0
STAR_SIZE = 45

PLOT_METRICS = [
    {"region": "SFG_L", "title": "Lt. SFG MwC deviation"},
    {"region": "MFG_L", "title": "Lt. MFG MwC deviation"},
    {"region": "vmPFC_R", "title": "Rt. vmPFC MwC deviation"},
    {"region": "PAL_L", "title": "Lt. PAL MwC deviation"},
    {"region": "HPC_R", "title": "Rt. HPC MwC deviation"},
    {"region": "AMY_R", "title": "Rt. AMY MwC deviation"},
]

def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def p_text(p: float) -> str:
    return r"$P < 0.001$" if p < 0.001 else rf"$P = {p:.3f}$"


def density_y(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread = float(np.ptp(values))
    pad = max(0.12 * spread, np.std(values, ddof=1) * 0.25, 1e-6)
    y = np.linspace(values.min() - pad, values.max() + pad, 256)
    density = stats.gaussian_kde(values)(y)
    return y, density / density.max()


def add_comparison_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float, label: str) -> None:
    ax.plot(
        [x1, x1, x2, x2], [y, y + height, y + height, y],
        color="#4D4D4D", linewidth=0.75, clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2, y + height * 1.35, label,
        ha="center", va="bottom", fontsize=P_VALUE_FONTSIZE, color="#333333",
    )


def raincloud_panel(ax: plt.Axes, frame: pd.DataFrame, specification: dict[str, object], seed: int) -> None:
    rng = np.random.default_rng(seed)
    region = str(specification["region"])
    metric = SELECTED_ROIS[region]
    arrays = {
        group: frame.loc[frame.analysis_cohort.eq(group), metric].dropna().to_numpy(float)
        for group in GROUPS
    }

    for group in GROUPS:
        position = GROUP_POSITIONS[group]
        values = arrays[group]
        color = COLORS[group]
        cloud_x = position - 0.19 + rng.normal(0, 0.035, len(values))
        ax.scatter(cloud_x, values, s=11, color=color, alpha=0.78, edgecolors="none", zorder=3)

        box = ax.boxplot(
            [values], positions=[position], widths=0.10, patch_artist=True,
            showfliers=False, manage_ticks=False,
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
            boxprops={"edgecolor": "#555555", "linewidth": 0.8}, zorder=4,
        )
        box["boxes"][0].set_facecolor("white")
        box["boxes"][0].set_alpha(0.84)
        ax.scatter(
            [position], [np.mean(values)], marker="D", s=12,
            facecolor=color, edgecolor="white", linewidth=0.45, zorder=5,
        )

        y, density = density_y(values)
        right = position + 0.06 + 0.31 * density
        ax.fill_betweenx(
            y, position + 0.06, right, facecolor=color, alpha=0.48,
            linewidth=0, edgecolor="none", zorder=1,
        )

    all_values = np.concatenate(list(arrays.values()))
    data_min, data_max = float(all_values.min()), float(all_values.max())
    span = data_max - data_min
    if span == 0:
        span = max(abs(data_max), 1.0)

    row = adjusted_statistics.loc[adjusted_statistics.region.eq(region)].iloc[0]
    bracket_height = 0.025 * span
    bracket_y = data_max + 0.25 * span
    add_comparison_bracket(
        ax, GROUP_POSITIONS["nTRD"] - 0.05, GROUP_POSITIONS["TRD1"] + 0.05,
        bracket_y, bracket_height, p_text(row.p_adjusted),
    )

    # The symbol is determined from this execution's adjusted p value.
    if row.p_adjusted < 0.001:
        star_style, star_count = "solid", 3
    elif row.p_adjusted < 0.01:
        star_style, star_count = "solid", 2
    elif row.p_adjusted < 0.05:
        star_style, star_count = "solid", 1
    elif row.p_adjusted < 0.10:
        star_style, star_count = "outline", 1
    else:
        star_style, star_count = None, 0

    if star_style is not None:
        star_y = data_max + 0.15 * span
        star_offsets = (np.arange(star_count) - (star_count - 1) / 2) * 0.11
        for group, opposite in (("nTRD", "TRD1"), ("TRD1", "nTRD")):
            star_color = COLORS[opposite]
            ax.scatter(
                GROUP_POSITIONS[group] + star_offsets, np.full(star_count, star_y),
                marker="*", s=STAR_SIZE,
                facecolors=star_color if star_style == "solid" else "none",
                edgecolors=star_color, linewidths=0.9, zorder=8,
            )

    ax.set_xlim(-0.43, GROUP_POSITIONS["TRD1"] + 0.43)
    ax.set_ylim(data_min - 0.14 * span, data_max + 0.36 * span)
    ax.set_xticks([GROUP_POSITIONS[group] for group in GROUPS], ["", ""])
    ax.set_ylabel("MwC deviation (HC z)", fontsize=YLABEL_FONTSIZE)
    ax.set_title(str(specification["title"]), pad=7, fontweight="semibold")
    ax.grid(axis="y", color="#D5D8DC", linewidth=0.55, alpha=0.70)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="x", bottom=False, length=0)
    ax.tick_params(axis="y", length=3)


configure_style()
fig, axes = plt.subplots(
    1, 6, figsize=(12.3448, 3.40), gridspec_kw={"wspace": 0.32}
)
for index, (ax, specification) in enumerate(zip(axes, PLOT_METRICS)):
    raincloud_panel(ax, baseline, specification, SEED + index)

legend_handles = [
    Patch(facecolor=COLORS[group], edgecolor="none", label=GROUP_LABELS[group])
    for group in GROUPS
]
fig.legend(
    handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
    ncol=2, frameon=False, fontsize=8.5, handlelength=1.75, handleheight=1.75,
    columnspacing=2.4, handletextpad=0.55,
)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.86)

figure_stem = FIGURE_DIR / "Figure_selected_unilateral_MwC_deviation_raincloud"
fig.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(figure_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
plt.show()

print("Wrote:")
for suffix in (".png", ".pdf", ".svg"):
    print(" -", figure_stem.with_suffix(suffix))


# Source cell 16
expected = [
    MWC_FULL_FILE,
    MWC_WIDE_FILE,
    MWC_CALCULATION_VALIDATION_FILE,
    OUTPUT_DIR / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv",
    OUTPUT_DIR / "subject_level_corticolimbic26_normativeSC10_MwC_long.csv",
    OUTPUT_DIR / "subject_level_nTRD_TRD1_corticolimbic26_normativeSC10_MwC_HC_reference_deviation.csv",
    MWC_SPATIAL_SUMMARY_FILE,
    OUTPUT_DIR / "nTRD_TRD1_TRD2_normativeSC_MwC_26ROI_mean_deviations.csv",
    MWC_DEVIATION_HEATMAP,
    MWC_DEVIATION_HEATMAP.with_suffix(".pdf"),
    MWC_SPATIAL_FIGURE,
    MWC_SPATIAL_FIGURE.with_suffix(".pdf"),
    OUTPUT_DIR / "ntrd_vs_trd_selected_unilateral_mwc_deviation_age_sex.csv",
    FIGURE_DIR / "Figure_selected_unilateral_MwC_deviation_raincloud.png",
    FIGURE_DIR / "Figure_selected_unilateral_MwC_deviation_raincloud.pdf",
    FIGURE_DIR / "Figure_selected_unilateral_MwC_deviation_raincloud.svg",
]
expected = [Path(output_path).resolve() for output_path in expected]
assert all(output_path.exists() and output_path.stat().st_size > 0 for output_path in expected)
created = [
    output_path.resolve() for output_path in OUTPUT_DIR.rglob("*")
    if output_path.is_file() and output_path.name != ".DS_Store"
]
assert set(expected).issubset(set(created)), sorted(set(expected) - set(created))
allowed_additional = [path for path in created if path not in expected and path.parent == FIGURE_DIR.resolve()]
assert set(created) - set(expected) == set(allowed_additional)
print(f"MwC scope audit passed: {len(created)} normative-SC MwC files.")
