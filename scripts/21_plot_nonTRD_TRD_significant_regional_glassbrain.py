#!/usr/bin/env python3
"""Combined Nilearn glass brain of primary TRD1 versus nTRD regional findings.

The selected ROIs come only from the formal age/sex-adjusted regional
SUVR table at the prespecified uncorrected p<.01 reporting threshold.  Display
colours are categorical and reuse the exact screenshot-sampled palette from
the preceding sensitivity glass-brain figures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("METAFL_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUT = Path(os.environ.get(
    "METAFL_PRIMARY_REGIONAL_GLASS_OUTPUT",
    PROJECT_ROOT / "outputs/main_analyses/15_nonTRD_TRD_significant_regional_SUVR_glass_brain",
)).resolve()
ATLAS_FILE = Path(os.environ.get(
    "METAFL_AAL3_DTOR_ATLAS",
    PROJECT_ROOT / "inputs/core/AAL3_144_space-PET_res-2_dseg.nii.gz",
)).resolve()
RESULT_FILE = PROJECT_ROOT / (
    "outputs/main_analyses/01_TRD1_discovery_agesex/"
    "regional_uncorrected_p_lt_0p01.xlsx"
)
SGC_AUDIT_FILE = PROJECT_ROOT / (
    "outputs/main_analyses/01_TRD1_discovery_agesex/regional_all.xlsx"
)
PALETTE_FILE = PROJECT_ROOT / (
    "outputs/main_analyses/13_sensitivity_mPFC_striatopallidal_glass_brain/"
    "tables/sampled_blue_pink_colormap_anchors.csv"
)

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/metafl_primary_regional_glass_mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd


DEEP_BLUE = "#A3C2F4"
DEEP_PINK = "#D2A1B1"
PALE_PINK = "#EAD8DE"
P_THRESHOLD = 0.01

ROI_ABBREVIATIONS = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solid_cmap(name: str, colour: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [colour, colour], N=2)


def format_p(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"={value:.3f}".replace("0.", ".")


def roi_display(roi: str) -> tuple[int, str, str]:
    match = re.match(r"^AAL3_(\d+)_(.+)_([LR])$", roi)
    if not match:
        raise ValueError(f"Unexpected AAL3 ROI label: {roi}")
    atlas_id = int(match.group(1))
    region = match.group(2)
    side = "Lt." if match.group(3) == "L" else "Rt."
    return atlas_id, f"{side} {ROI_ABBREVIATIONS.get(region, region)}", region


def prepare_values() -> pd.DataFrame:
    source = pd.read_excel(RESULT_FILE)
    if source.empty:
        raise ValueError("The formal p<.01 regional table is empty")
    if not source["selected_uncorrected_p_lt_0p01"].eq(True).all():
        raise ValueError("Formal selection flag is not true for every displayed ROI")
    if not source["p_group_uncorrected"].lt(P_THRESHOLD).all():
        raise ValueError("At least one displayed ROI does not meet p<.01")
    if source["ROI"].duplicated().any():
        raise ValueError("Duplicate ROI in formal regional table")

    records = []
    for row in source.to_dict("records"):
        atlas_id, label, region = roi_display(row["ROI"])
        difference = float(row["TRD1_deviation"] - row["nTRD_deviation"])
        direction = "higher in TRD" if difference > 0 else "lower in TRD"
        colour = (
            DEEP_BLUE if difference < 0
            else DEEP_PINK if region in {"Putamen", "Pallidum"}
            else PALE_PINK
        )
        records.append(
            {
                "AAL3_ID": atlas_id,
                "ROI": row["ROI"],
                "display_label": label,
                "display_direction": direction,
                "display_colour_hex": colour,
                "TRD1_mean_deviation": float(row["TRD1_deviation"]),
                "nTRD_mean_deviation": float(row["nTRD_deviation"]),
                "raw_mean_difference_TRD1_minus_nTRD": difference,
                "adjusted_group_t": float(row["t_group"]),
                "partial_r": float(row["effect_size_partial_r"]),
                "p_uncorrected": float(row["p_group_uncorrected"]),
                "q_BH26_audit_only": float(row["q_fdr_BH_26_regions_audit_only"]),
                "n_used": int(row["n_used"]),
                "covariates": "age + sex",
                "reporting_threshold": "uncorrected p<.01",
            }
        )
    result = pd.DataFrame(records)
    return result.sort_values(["display_direction", "AAL3_ID"]).reset_index(drop=True)


def confirm_sgc_not_selected() -> pd.DataFrame:
    audit = pd.read_excel(SGC_AUDIT_FILE)
    regional = audit.loc[
        audit["ROI"].isin({"AAL3_151_ACC_sub_L", "AAL3_152_ACC_sub_R"})
    ].copy()
    expected = {"AAL3_151_ACC_sub_L", "AAL3_152_ACC_sub_R"}
    if set(regional["ROI"]) != expected or len(regional) != 2:
        raise ValueError("Could not uniquely identify bilateral SGC regional results")
    if regional["p_group_uncorrected"].lt(P_THRESHOLD).any():
        raise ValueError("An SGC regional result meets p<.01 and must be added")
    regional["display_feature"] = regional["ROI"].map({
        "AAL3_151_ACC_sub_L": "Lt.SGC regional SUVR",
        "AAL3_152_ACC_sub_R": "Rt.SGC regional SUVR",
    })
    regional["beta_TRD1_minus_nTRD"] = regional["TRD1_deviation"] - regional["nTRD_deviation"]
    regional["covariates"] = "age + sex"
    return regional[
        [
            "ROI",
            "display_feature",
            "beta_TRD1_minus_nTRD",
            "t_group",
            "p_group_uncorrected",
            "covariates",
        ]
    ].rename(columns={
        "ROI": "sgc_roi",
        "t_group": "t",
        "p_group_uncorrected": "p_uncorrected",
    }).sort_values("sgc_roi")


def create_binary_map(
    atlas_img: nib.Nifti1Image,
    atlas: np.ndarray,
    ids: list[int],
    output_file: Path,
) -> dict[int, int]:
    data = np.zeros(atlas.shape, dtype=np.uint8)
    counts = {}
    for atlas_id in ids:
        region = atlas == atlas_id
        count = int(region.sum())
        if count == 0:
            raise ValueError(f"AAL3 ID {atlas_id} is absent from the atlas")
        data[region] = 1
        counts[atlas_id] = count
    header = atlas_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(data, atlas_img.affine, header), output_file)
    return counts


def render(values: pd.DataFrame, maps: dict[str, Path], output_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(11.8, 4.55), facecolor="white")
    display = plotting.plot_glass_brain(
        nib.load(maps["deep_blue"]),
        display_mode="lyrz",
        plot_abs=False,
        threshold=0.5,
        colorbar=False,
        cmap=solid_cmap("sampled_deep_blue", DEEP_BLUE),
        vmin=0.0,
        vmax=1.0,
        symmetric_cbar=False,
        black_bg=False,
        annotate=True,
        alpha=1.0,
        radiological=False,
        resampling_interpolation="nearest",
        figure=figure,
        axes=(0.018, 0.265, 0.965, 0.585),
    )
    display.add_overlay(
        nib.load(maps["pale_pink"]),
        threshold=0.5,
        colorbar=False,
        cmap=solid_cmap("sampled_pale_pink", PALE_PINK),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    display.add_overlay(
        nib.load(maps["deep_pink"]),
        threshold=0.5,
        colorbar=False,
        cmap=solid_cmap("sampled_deep_pink", DEEP_PINK),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    figure.suptitle(
        "TRD versus nonTRD: significant regional SUVR differences",
        x=0.5,
        y=0.973,
        fontsize=16.5,
        fontweight="bold",
        color="#40454D",
    )
    figure.text(
        0.5,
        0.910,
        f"Adjusted for age and sex · uncorrected p < .01 · {len(values)} AAL3 ROIs",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#5A5A5A",
    )

    legend_handles = [
        Patch(facecolor=DEEP_BLUE, edgecolor="none", label="Lower in TRD"),
        Patch(facecolor=PALE_PINK, edgecolor="none", label="Higher limbic SUVR in TRD"),
        Patch(facecolor=DEEP_PINK, edgecolor="none", label="Higher striatopallidal SUVR in TRD"),
    ]
    legend = figure.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.227),
        frameon=False,
        fontsize=8.8,
        ncol=3,
        columnspacing=1.8,
        handlelength=1.5,
        handleheight=1.1,
    )
    for text in legend.get_texts():
        text.set_color("#333333")

    lower_rows = values.loc[values["display_direction"].eq("lower in TRD")]
    higher_rows = values.loc[values["display_direction"].eq("higher in TRD")]
    lower_text = "   |   ".join(
        f"{row.display_label} p{format_p(row.p_uncorrected)}"
        for row in lower_rows.itertuples(index=False)
    )
    figure.text(
        0.5,
        0.147,
        "Lower in TRD: " + lower_text,
        ha="center",
        fontsize=8.9,
        color="#333333",
    )
    higher_text = "   |   ".join(
        f"{row.display_label} p{format_p(row.p_uncorrected)}"
        for row in higher_rows.itertuples(index=False)
    )
    figure.text(
        0.5,
        0.087,
        "Higher in TRD: " + higher_text,
        ha="center",
        fontsize=8.6,
        color="#333333",
    )
    figure.text(
        0.5,
        0.030,
        "Colours identify direction/structure; statistical magnitude is reported below rather than encoded by colour.",
        ha="center",
        fontsize=8.1,
        color="#747474",
    )

    for suffix, kwargs in [
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        display.savefig(output_stem.with_suffix(suffix), **kwargs)
    display.close()


def main() -> int:
    figure_dir = OUTPUT / "figures"
    mask_dir = OUTPUT / "masks"
    table_dir = OUTPUT / "tables"
    for directory in (figure_dir, mask_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    values = prepare_values()
    sgc_audit = confirm_sgc_not_selected()
    values_file = table_dir / "TRD1_vs_nTRD_regional_SUVR_p_lt_0p01_figure_values.csv"
    sgc_file = table_dir / "bilateral_SGC_regional_SUVR_p_threshold_audit.csv"
    values.to_csv(values_file, index=False)
    sgc_audit.to_csv(sgc_file, index=False)

    palette = pd.read_csv(PALETTE_FILE)
    required_colours = {DEEP_BLUE, DEEP_PINK, PALE_PINK}
    observed_colours = set(palette["actual_hex"].str.upper())
    if not required_colours.issubset(observed_colours):
        raise ValueError("The required configured colours are missing")
    palette.to_csv(table_dir / "reused_sampled_colour_anchors.csv", index=False)

    atlas_img = nib.load(ATLAS_FILE)
    atlas = np.asanyarray(atlas_img.dataobj)
    if atlas.ndim != 3 or not np.array_equal(atlas, np.rint(atlas)):
        raise ValueError("Atlas must be a finite 3D integer label image")

    groups = {
        "deep_blue": values.loc[values["display_colour_hex"].eq(DEEP_BLUE), "AAL3_ID"].tolist(),
        "pale_pink": values.loc[values["display_colour_hex"].eq(PALE_PINK), "AAL3_ID"].tolist(),
        "deep_pink": values.loc[values["display_colour_hex"].eq(DEEP_PINK), "AAL3_ID"].tolist(),
    }
    if any(len(ids) == 0 for ids in groups.values()):
        raise ValueError(f"A display colour group is unexpectedly empty: {groups}")

    maps = {}
    voxel_counts = {}
    for name, ids in groups.items():
        path = mask_dir / f"AAL3_TRD1_vs_nTRD_p_lt_0p01_{name}.nii.gz"
        counts = create_binary_map(atlas_img, atlas, ids, path)
        maps[name] = path
        voxel_counts.update(counts)
    values["voxel_count_1mm"] = values["AAL3_ID"].map(voxel_counts)
    values.to_csv(values_file, index=False)

    output_stem = figure_dir / "Figure_TRD_vs_nonTRD_significant_regional_SUVR_combined_nilearn_glass_brain"
    render(values, maps, output_stem)

    expected_outputs = [output_stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    checks = {
        "formal_p_lt_0p01_ROI_set_used_without_preselection": len(values) == len(pd.read_excel(RESULT_FILE)),
        "all_displayed_results_meet_uncorrected_p_lt_0p01": bool(values["p_uncorrected"].lt(P_THRESHOLD).all()),
        "direction_matches_TRD1_minus_nTRD_raw_mean_difference": bool(
            ((values["raw_mean_difference_TRD1_minus_nTRD"] < 0) == values["display_direction"].eq("lower in TRD")).all()
        ),
        "bilateral_SGC_does_not_meet_p_lt_0p01": bool(sgc_audit["p_uncorrected"].ge(P_THRESHOLD).all()),
        "palette_matches_configured_hex_values": required_colours.issubset(observed_colours),
        "all_selected_AAL3_labels_nonempty": bool(values["voxel_count_1mm"].gt(0).all()),
        "all_png_pdf_svg_outputs_exist": all(path.is_file() for path in expected_outputs),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "renderer": "Nilearn plot_glass_brain",
        "comparison": "TRD1 versus nTRD",
        "quantity": "regional SUVR deviation",
        "covariates": ["age", "sex"],
        "reporting_threshold": "uncorrected p<.01",
        "display_mode": "lyrz (four views)",
        "plot_abs": False,
        "display_colours_are_categorical": True,
        "statistics_recomputed": False,
        "palette": {
            "lower_regions": DEEP_BLUE,
            "higher_non_striatopallidal_regions": PALE_PINK,
            "higher_putamen_pallidum": DEEP_PINK,
        },
        "source_result_file": str(RESULT_FILE),
        "source_result_sha256": sha256(RESULT_FILE),
        "atlas": str(ATLAS_FILE),
        "atlas_sha256": sha256(ATLAS_FILE),
        "SGC_audit_source": str(SGC_AUDIT_FILE),
        "SGC_audit_source_sha256": sha256(SGC_AUDIT_FILE),
        "palette_source": str(PALETTE_FILE),
        "palette_source_sha256": sha256(PALETTE_FILE),
        "checks": checks,
        "outputs": {
            "PNG": str(output_stem.with_suffix(".png")),
            "PDF": str(output_stem.with_suffix(".pdf")),
            "SVG": str(output_stem.with_suffix(".svg")),
            "values": str(values_file),
            "SGC_threshold_audit": str(sgc_file),
        },
    }
    (OUTPUT / "glass_brain_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
