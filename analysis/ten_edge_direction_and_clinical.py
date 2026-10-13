#!/usr/bin/env python3
"""Render the ten paired-p<.01 metabolic edges in matched clinical/direction panels.

TRD values are age/sex-adjusted TRD1-minus-nTRD group coefficients. Arm values
are intercept coefficients from within-arm post-minus-pre changes, equivalent
to the mean paired change. All are expressed in HC-referenced deviation units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from core import FIGURES, TABLES, load_data


OUTPUT = FIGURES / "ten_edge_direction_clinical"

PANEL_AXIS_WIDTH = (42.0 - 42.0 * 0.025 - 42.0 * 0.005) / (8 + 7 * 0.18)
RIGHT_IN = 0.21
CORRELATION_AXIS_HEIGHT = 6.8 * (0.86 - 0.18)
CORRELATION_FIGURE_HEIGHT = 6.8
DIRECTION_HEIGHT = (6.8 / 2) * 1.5
CORRELATION_WSPACE = 0.18
DIRECTION_WSPACE = 0.46
DIRECTION_LEFT_IN = 2.10
ACTIVE_COLOR = "#D95F45"
SHAM_COLOR = "#56B4C9"
HATCH_BLUE = "#309AC8"
BAR_CORAL = "#E6846B"
ARM_ORDER = ("TRD", "rTMS", "iTBS", "sham")
DIRECTION_LABELS = (
    ("TRD", "−nTRD"),
    ("rTMS", "post−pre"),
    ("iTBS", "post−pre"),
    ("sham", "post−pre"),
)
DIRECTION_LABEL_FONTSIZE = 23.625
DIRECTION_NUMBER_FONTSIZE = 14.5
HATCH_LIMIT = 0.2


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.7,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_width(n_panels: int, *, direction: bool) -> float:
    left = DIRECTION_LEFT_IN if direction else 1.05
    wspace = DIRECTION_WSPACE if direction else CORRELATION_WSPACE
    return left + RIGHT_IN + PANEL_AXIS_WIDTH * (n_panels + (n_panels - 1) * wspace)


def make_axes(n_panels: int, *, direction: bool) -> tuple[plt.Figure, list[plt.Axes]]:
    width = figure_width(n_panels, direction=direction)
    left = DIRECTION_LEFT_IN if direction else 1.05
    wspace = DIRECTION_WSPACE if direction else CORRELATION_WSPACE
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(width, DIRECTION_HEIGHT if direction else CORRELATION_FIGURE_HEIGHT),
        gridspec_kw={"wspace": wspace},
        facecolor="white",
        squeeze=False,
    )
    fig.subplots_adjust(left=left / width, right=1 - RIGHT_IN / width,
                        bottom=0.18, top=0.86, wspace=wspace)
    return fig, list(axes.flat)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def partial_spearman_age_sex(
    x: np.ndarray, y: np.ndarray, age: np.ndarray, sex: np.ndarray
) -> tuple[float, float, int]:
    """Match the main analysis rank-residualization and n-4 t reference."""
    x, y, age, sex = map(lambda a: np.asarray(a, dtype=float), (x, y, age, sex))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(age) & np.isfinite(sex)
    x, y, age, sex = x[valid], y[valid], age[valid], sex[valid]
    n = len(x)
    if n < 6:
        return np.nan, np.nan, n
    covariates = np.column_stack([np.ones(n), stats.rankdata(age), stats.rankdata(sex)])
    projection = np.eye(n) - covariates @ np.linalg.pinv(covariates)
    rx = projection @ stats.rankdata(x)
    ry = projection @ stats.rankdata(y)
    rho = float(np.clip(np.corrcoef(rx, ry)[0, 1], -0.999999999, 0.999999999))
    t_value = rho * np.sqrt((n - 4) / (1 - rho**2))
    p_value = float(2 * stats.t.sf(abs(t_value), n - 4))
    return rho, p_value, n


def regression_se_band(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Unadjusted OLS visual guide with +/- one mean SE, as in the reference."""
    x_grid = np.linspace(float(x.min()), float(x.max()), 160)
    fit = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
    prediction = fit.get_prediction(sm.add_constant(x_grid, has_constant="add")).summary_frame(
        alpha=0.05
    )
    center = prediction["mean"].to_numpy(float)
    se = prediction["mean_se"].to_numpy(float)
    return x_grid, center, center - se, center + se


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_data()
    meta = data["meta"]
    paired_all = pd.read_csv(TABLES / "longitudinal_all_660_tests.csv")
    selected = paired_all.loc[
        paired_all.modality.eq("Metabolic connectivity")
        & paired_all.analysis_arm.isin(["rTMS", "iTBS"])
        & paired_all.paired_p_lt_0p01
    ].copy()
    assert len(selected) == 10 and selected.analysis_arm.value_counts().to_dict() == {
        "rTMS": 7,
        "iTBS": 3,
    }
    selected = selected.copy()
    selected["arm_sort"] = pd.Categorical(
        selected.analysis_arm, categories=["rTMS", "iTBS"], ordered=True
    )
    selected = selected.sort_values(["arm_sort", "paired_p_uncorrected"]).drop(
        columns="arm_sort"
    ).reset_index(drop=True)
    selected["edge_label"] = selected.feature
    selected["edge_key"] = selected.ROI_1 + "--" + selected.ROI_2
    assert selected.edge_key.is_unique
    primary = pd.read_csv(TABLES / "primary_connectivity_all.csv")
    metadata = meta.loc[
        meta.analysis_cohort.eq("TRD2") & meta.session.eq("ses-01")
    ].sort_values("participant_id").copy()
    assert len(metadata) == 119
    changes = metadata[
        ["participant_id", "analysis_arm", "age", "sex", "hdrs_improvement"]
    ].reset_index(drop=True)
    post = meta.loc[
        meta.analysis_cohort.eq("TRD2") & meta.session.eq("ses-02")
    ].sort_values("participant_id")
    assert metadata.participant_id.tolist() == post.participant_id.tolist()
    for row in selected.itertuples(index=False):
        changes[row.edge_key] = (
            data["edge"][post.index, row.feature_index]
            - data["edge"][metadata.index, row.feature_index]
        )
    assert changes[list(selected.edge_key)].notna().all().all()
    assert changes.analysis_arm.value_counts().to_dict() == {
        "rTMS": 38,
        "iTBS": 43,
        "sham": 38,
    }

    correlation_rows: list[dict] = []
    paired_index = paired_all.loc[
        paired_all.modality.eq("Metabolic connectivity")
    ].set_index(["analysis_arm", "ROI_1", "ROI_2"])
    for row in selected.itertuples(index=False):
        for arm in (row.analysis_arm, "sham"):
            sub = changes.loc[changes.analysis_arm.eq(arm)]
            rho, p_value, n = partial_spearman_age_sex(
                sub[row.edge_key], sub.hdrs_improvement, sub.age, sub.sex
            )
            if arm != "sham":
                stored = paired_index.loc[(arm, row.ROI_1, row.ROI_2)]
                np.testing.assert_allclose(
                    [rho, p_value, n],
                    [stored.HDRS_partial_Spearman_r_age_sex, stored.HDRS_p_uncorrected, stored.n_used],
                    rtol=0,
                    atol=1e-10,
                )
            correlation_rows.append(
                {"selected_arm": row.analysis_arm, "edge_label": row.edge_label,
                 "edge_key": row.edge_key, "plot_arm": arm, "n": n,
                 "partial_spearman_rho_age_sex": rho, "p_uncorrected": p_value}
            )
    correlations = pd.DataFrame(correlation_rows)

    primary_index = primary.set_index("feature")
    effects: list[dict] = []
    for row in selected.itertuples(index=False):
        baseline = primary_index.loc[row.edge_label]
        assert baseline.model == "age + sex" and baseline.n == 60
        effects.append(
            {"selected_arm": row.analysis_arm, "edge_label": row.edge_label,
             "display_arm": "TRD", "n": 60,
             "beta_HC_z": float(baseline.TRD1_minus_nTRD_beta),
             "ci95_low": float(baseline.CI95_low),
             "ci95_high": float(baseline.CI95_high),
             "p_uncorrected": float(baseline.p_uncorrected),
             "model": "TRD1 - nTRD | deviation ~ TRD1 + age + sex"}
        )
        for arm in ("rTMS", "iTBS", "sham"):
            paired = paired_index.loc[(arm, row.ROI_1, row.ROI_2)]
            sub = changes.loc[changes.analysis_arm.eq(arm), row.edge_key].to_numpy(float)
            np.testing.assert_allclose(sub.mean(), paired.post_minus_pre, rtol=0, atol=1e-10)
            effects.append(
                {"selected_arm": row.analysis_arm, "edge_label": row.edge_label,
                 "display_arm": arm, "n": len(sub),
                 "beta_HC_z": float(paired.post_minus_pre),
                 "ci95_low": float(paired.post_minus_pre_CI95_low),
                 "ci95_high": float(paired.post_minus_pre_CI95_high),
                 "p_uncorrected": float(paired.paired_p_uncorrected),
                 "model": "Intercept-only paired change (post - pre); beta = mean change"}
            )
    effects_frame = pd.DataFrame(effects)
    return selected, changes, correlations, effects_frame


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", colors="#555555", labelcolor="black", direction="out")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(False)


def number(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):.3f}".replace("0.", ".")


def probability(value: float) -> str:
    return f"{value:.4f}".replace("0.", ".")


def plot_correlation_panel(
    ax: plt.Axes,
    selected_row: pd.Series,
    changes: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    show_ylabel: bool,
) -> None:
    active_arm = str(selected_row.analysis_arm)
    edge_key = str(selected_row.edge_key)
    matched = correlations.loc[correlations.edge_key.eq(edge_key)]
    for arm, color, text_y in (
        (active_arm, ACTIVE_COLOR, 0.96),
        ("sham", SHAM_COLOR, 0.06),
    ):
        group = changes.loc[changes.analysis_arm.eq(arm)]
        x = group[edge_key].to_numpy(float)
        y = group.hdrs_improvement.to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        x_grid, center, lower, upper = regression_se_band(x, y)
        ax.fill_between(x_grid, lower, upper, color=color, alpha=0.20, linewidth=0, zorder=1)
        ax.plot(x_grid, center, color="black", linewidth=2.6, zorder=3)
        ax.scatter(x, y, s=92, color=color, edgecolor="none", alpha=0.82, zorder=4)
        stat = matched.loc[matched.plot_arm.eq(arm)].iloc[0]
        ax.text(
            0.985,
            text_y,
            f"ρ = {number(float(stat.partial_spearman_rho_age_sex))}, "
            f"p = {probability(float(stat.p_uncorrected))}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            color=color,
            fontsize=28.6,
            fontweight="bold" if float(stat.p_uncorrected) < 0.05 else "normal",
            fontstyle="italic",
            zorder=7,
        )
    pooled = changes.loc[changes.analysis_arm.isin([active_arm, "sham"]), edge_key]
    x_span = max(float(pooled.max() - pooled.min()), 0.5)
    ax.set_xlim(float(pooled.min() - 0.055 * x_span), float(pooled.max() + 0.055 * x_span))
    y_values = changes.loc[
        changes.analysis_arm.isin([active_arm, "sham"]), "hdrs_improvement"
    ].to_numpy(float)
    y_span = max(float(np.nanmax(y_values) - np.nanmin(y_values)), 5.0)
    ax.set_ylim(float(np.nanmin(y_values) - 0.20 * y_span),
                float(np.nanmax(y_values) + 0.19 * y_span))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("Edge change (post − pre, HC z)")
    ax.set_ylabel("HDRS-17 improvement" if show_ylabel else "")
    ax.set_title(str(selected_row.edge_label), fontsize=14.5,
                 fontweight="bold", pad=54)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=ACTIVE_COLOR, markerfacecolor=ACTIVE_COLOR,
                   linewidth=3.45, markersize=16.5, label=active_arm),
            Line2D([], [], marker="o", color=SHAM_COLOR, markerfacecolor=SHAM_COLOR,
                   linewidth=3.45, markersize=16.5, label="Sham"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=25.5,
        handlelength=1.25,
        columnspacing=0.6,
    )
    style_axis(ax)


def draw_signed_bar(ax: plt.Axes, estimate: float, y: float, height: float) -> None:
    split = np.sign(estimate) * min(abs(estimate), HATCH_LIMIT)
    hatched_left = min(0.0, split)
    ax.add_patch(
        Rectangle(
            (hatched_left, y - height / 2),
            abs(split),
            height,
            facecolor=HATCH_BLUE,
            edgecolor="white",
            hatch="///",
            linewidth=0.8,
            zorder=2,
        )
    )
    if abs(estimate) > HATCH_LIMIT:
        ax.add_patch(
            Rectangle(
                (min(split, estimate), y - height / 2),
                abs(estimate - split),
                height,
                facecolor=BAR_CORAL,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
        )


def plot_direction_panel(
    ax: plt.Axes, selected_row: pd.Series, effects: pd.DataFrame
) -> None:
    rows = effects.loc[effects.edge_label.eq(selected_row.edge_label)].set_index("display_arm")
    assert set(rows.index) == set(ARM_ORDER)
    betas = rows.beta_HC_z.to_numpy(float)
    span = max(float(np.nanmax(betas) - np.nanmin(betas)), 0.4)
    x_low = min(0.0, float(np.nanmin(betas))) - max(0.35 * span, 0.24)
    x_high = max(0.0, float(np.nanmax(betas))) + 0.33 * span
    ax.axvline(0, color="#A2A2A2", linewidth=1.55, zorder=0)
    for y, arm in zip((3, 2, 1, 0), ARM_ORDER):
        row = rows.loc[arm]
        estimate = float(row.beta_HC_z)
        draw_signed_bar(ax, estimate, y, 0.38)
        ax.scatter(
            [estimate], [y + 0.21], marker="D", s=69,
            facecolor="white", edgecolor="#222222", linewidth=1.9,
            zorder=5, clip_on=False,
        )
        gap = 0.045 * (x_high - x_low)
        ax.text(
            estimate + (gap if estimate >= 0 else -gap),
            y + 0.21,
            f"{estimate:+.3f}",
            ha="left" if estimate >= 0 else "right",
            va="center",
            fontsize=DIRECTION_NUMBER_FONTSIZE,
            fontweight="bold",
            zorder=6,
        )
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(-0.65, 3.7)
    ax.set_yticks([3, 2, 1, 0], ["", "", "", ""])
    ax.tick_params(axis="y", length=0)
    for y, (arm_label, contrast_label) in zip((3, 2, 1, 0), DIRECTION_LABELS):
        for text_y, label, weight in (
            (y + 0.20, arm_label, "bold"),
            (y - 0.20, contrast_label, "normal"),
        ):
            ax.text(
                -0.40,
                text_y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=DIRECTION_LABEL_FONTSIZE,
                fontweight=weight,
                clip_on=False,
            )
    ax.tick_params(axis="x", labelsize=DIRECTION_NUMBER_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("β (HC z units)", fontsize=DIRECTION_NUMBER_FONTSIZE)
    ax.set_title(str(selected_row.edge_label), fontsize=DIRECTION_NUMBER_FONTSIZE,
                 fontweight="bold", pad=10)
    style_axis(ax)


def plot_combined_overview(
    selected: pd.DataFrame,
    changes: pd.DataFrame,
    correlations: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    """Show all twenty panels in one 5-column, 4-row figure."""
    axis_width = PANEL_AXIS_WIDTH
    corr_height = CORRELATION_AXIS_HEIGHT
    direction_height = DIRECTION_HEIGHT * (0.86 - 0.18)
    left, gap_x, right = 2.10, 2.10, 0.28
    x_positions = [left + column * (axis_width + gap_x) for column in range(5)]
    width = x_positions[-1] + axis_width + right
    direction_lower = 0.76
    direction_upper = direction_lower + direction_height + 1.12
    correlation_lower = direction_upper + direction_height + 1.45
    correlation_upper = correlation_lower + corr_height + 1.66
    height = correlation_upper + corr_height + 0.95

    fig = plt.figure(figsize=(width, height), facecolor="white")
    for index, row in selected.iterrows():
        column = index % 5
        top = index < 5
        correlation_bottom = correlation_upper if top else correlation_lower
        direction_bottom = direction_upper if top else direction_lower
        corr_ax = fig.add_axes(
            [x_positions[column] / width, correlation_bottom / height,
             axis_width / width, corr_height / height]
        )
        plot_correlation_panel(
            corr_ax, row, changes, correlations, show_ylabel=column == 0
        )
        direction_ax = fig.add_axes(
            [x_positions[column] / width, direction_bottom / height,
             axis_width / width, direction_height / height]
        )
        plot_direction_panel(direction_ax, row, effects)

    stem = OUTPUT / "all_20_correlation_and_direction_panels"
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    configure()
    selected, changes, correlations, effects = load_sources()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(TABLES / "ten_edge_partial_spearman_active_and_sham.csv", index=False)
    effects.to_csv(TABLES / "ten_edge_TRD_and_arm_change_beta_with_ci.csv", index=False)

    for kind, plotter in (
        ("clinical_correlation", plot_correlation_panel),
        ("direction", plot_direction_panel),
    ):
        direction = kind == "direction"
        overview, axes = make_axes(len(selected), direction=direction)
        for index, row in selected.iterrows():
            if kind == "clinical_correlation":
                plotter(axes[index], row, changes, correlations, show_ylabel=index == 0)
            else:
                plotter(axes[index], row, effects)
        save(overview, OUTPUT / f"ten_edges_{kind}_overview")
    plot_combined_overview(selected, changes, correlations, effects)
    print(f"Ten selected edges; rTMS=7, iTBS=3. Output: {OUTPUT}")
    print(f"Correlation axes: {PANEL_AXIS_WIDTH:.6f} x {CORRELATION_AXIS_HEIGHT:.6f} inches")
    print(f"Direction axes: {PANEL_AXIS_WIDTH:.6f} x {DIRECTION_HEIGHT * (0.86 - 0.18):.6f} inches")


if __name__ == "__main__":
    main()
