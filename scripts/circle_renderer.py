"""Shared 26-node cortico-limbic circle-diagram renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Wedge


CIRCULAR_NEGATIVE = "#2868A8"
CIRCULAR_POSITIVE = "#B64040"
CIRCULAR_NEUTRAL = "#F7F7F5"
CIRCULAR_GRAPHITE = "#30343B"
CIRCULAR_GUIDE = "#B8C0C8"
CIRCULAR_CMAP = LinearSegmentedColormap.from_list(
    "metafl_circle",
    [CIRCULAR_NEGATIVE, CIRCULAR_NEUTRAL, CIRCULAR_POSITIVE],
)


def roi_short_label(roi: str) -> str:
    text = str(roi).removeprefix("AAL3_")
    pieces = text.split("_", 1)
    if pieces[0].isdigit() and len(pieces) == 2:
        text = pieces[1]
    hemisphere = "L" if text.endswith("_L") else ("R" if text.endswith("_R") else "")
    base = text[:-2] if hemisphere else text
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
    if not hemisphere:
        return label
    return f"{'Lt' if hemisphere == 'L' else 'Rt'}.{label}"


def draw_corticolimbic_connectogram(
    ax,
    title: str,
    regional_values: np.ndarray,
    edge_values: np.ndarray,
    roi_columns: list[str],
    cortical_rois: list[str],
    limbic_rois: list[str],
    cross_edges: list[tuple[str, str]],
    node_vmax: float,
    edge_vmax: float,
) -> None:
    """Draw the established 12-cortical/14-limbic circle diagram."""
    cortical_angles = np.linspace(170, 10, len(cortical_rois))
    limbic_angles = np.linspace(190, 350, len(limbic_rois))
    roi_angles = {
        **dict(zip(cortical_rois, cortical_angles)),
        **dict(zip(limbic_rois, limbic_angles)),
    }
    roi_positions = {
        roi: np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        for roi, angle in roi_angles.items()
    }
    regional_index = {roi: index for index, roi in enumerate(roi_columns)}
    edge_index = {tuple(edge): index for index, edge in enumerate(cross_edges)}
    circular_norm = TwoSlopeNorm(vmin=-node_vmax, vcenter=0, vmax=node_vmax)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.52, 1.52)
    ax.set_ylim(-1.53, 1.63)
    ax.set_facecolor("white")

    for start_angle, end_angle in [(5, 175), (185, 355)]:
        ax.add_patch(Wedge(
            (0, 0), 0.985, start_angle, end_angle, width=0.008,
            facecolor=CIRCULAR_GUIDE, edgecolor="none", zorder=0,
        ))

    visible_edges = []
    for edge, index in edge_index.items():
        value = float(edge_values[index])
        if np.isfinite(value) and abs(value) > 0:
            visible_edges.append((abs(value), edge[0], edge[1], value))
    visible_edges.sort()

    for magnitude, roi_1, roi_2, value in visible_edges:
        strength = min(magnitude / edge_vmax, 1.0)
        color = CIRCULAR_POSITIVE if value > 0 else CIRCULAR_NEGATIVE
        start = roi_positions[roi_1] * 0.965
        end = roi_positions[roi_2] * 0.965
        path = MplPath(
            [start, start * 0.30, end * 0.30, end],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=0.28 + 1.00 * strength,
            alpha=0.09 + 0.48 * strength,
            zorder=1,
            capstyle="round",
        ))

    for roi in cortical_rois + limbic_rois:
        angle = roi_angles[roi]
        value = float(regional_values[regional_index[roi]])
        facecolor = CIRCULAR_CMAP(
            circular_norm(np.clip(value, -node_vmax, node_vmax))
        )
        ax.add_patch(Wedge(
            (0, 0), 1.13, angle - 4.3, angle + 4.3, width=0.125,
            facecolor=facecolor, edgecolor="white", linewidth=0.8, zorder=4,
        ))

        tick_color = CIRCULAR_POSITIVE if value > 0 else CIRCULAR_NEGATIVE
        direction = np.array([
            np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        ])
        tick_start, tick_end = direction * 0.925, direction * 0.985
        ax.plot(
            [tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]],
            color=tick_color, linewidth=1.05, alpha=0.85, zorder=3,
        )

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
