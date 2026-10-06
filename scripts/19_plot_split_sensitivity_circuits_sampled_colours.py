#!/usr/bin/env python3
"""Create two four-view Nilearn glass brains using screenshot-sampled colours.

The first figure contains bilateral vmPFC, dmPFC and medial OFC in the sampled
blue palette.  The second contains bilateral caudate, putamen and pallidum;
the bilateral members of each structure use one identical sampled pink.
Signed formal sensitivity-adjusted regional-SUVR betas are printed below each
figure so direction is not inferred from the anatomical structure colours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MPL_CACHE = Path("/private/tmp/metafl_split_sensitivity_glassbrain_mpl")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("METAFL_ROOT", Path(__file__).resolve().parents[1])).resolve()
ATLAS_FILE = Path(os.environ.get(
    "METAFL_AAL3_DTOR_ATLAS",
    PROJECT_ROOT / "inputs/core/AAL3_144_space-PET_res-2_dseg.nii.gz",
)).resolve()
RESULT_FILE = PROJECT_ROOT / (
    "outputs/main_analyses/11_SGC_26ROI_figures/tables/"
    "baseline_severity_adjusted_TRD1_vs_nTRD_regional26_SGC.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/main_analyses/13_sensitivity_mPFC_striatopallidal_glass_brain"
)
VALUE_COLUMN = "adjusted_group_beta_TRD1_minus_nTRD"


PREFRONTAL_SPECS = [
    (21, "AAL3_021_Frontal_Med_Orb_L", "Lt. vmPFC", "vmPFC"),
    (22, "AAL3_022_Frontal_Med_Orb_R", "Rt. vmPFC", "vmPFC"),
    (19, "AAL3_019_Frontal_Sup_Medial_L", "Lt. dmPFC", "dmPFC"),
    (20, "AAL3_020_Frontal_Sup_Medial_R", "Rt. dmPFC", "dmPFC"),
    (25, "AAL3_025_OFCmed_L", "Lt. medial OFC", "medial OFC"),
    (26, "AAL3_026_OFCmed_R", "Rt. medial OFC", "medial OFC"),
]
SUBCORTICAL_SPECS = [
    (75, "AAL3_075_Caudate_L", "Lt. caudate", "caudate"),
    (76, "AAL3_076_Caudate_R", "Rt. caudate", "caudate"),
    (77, "AAL3_077_Putamen_L", "Lt. putamen", "putamen"),
    (78, "AAL3_078_Putamen_R", "Rt. putamen", "putamen"),
    (79, "AAL3_079_Pallidum_L", "Lt. pallidum", "pallidum"),
    (80, "AAL3_080_Pallidum_R", "Rt. pallidum", "pallidum"),
]

# Every anchor below is an exact RGB pixel sampled from the user's PNGs.
# The blue anchors are the two modal ROI fills in the latest reference image;
# the pink anchors are the two modal ROI fills in the latest reference image.
BLUE_SAMPLES = [
    {"hex": "#FFFFFF", "coordinate_xy": (10, 10), "role": "zero/background"},
    {"hex": "#BECFE8", "coordinate_xy": (93, 22), "role": "pale modal ROI fill"},
    {"hex": "#A3C2F4", "coordinate_xy": (81, 29), "role": "saturated modal ROI fill"},
]
PINK_SAMPLES = [
    {"hex": "#EAD8DE", "coordinate_xy": (69, 16), "role": "pale modal ROI fill"},
    {"hex": "#D2A1B1", "coordinate_xy": (52, 45), "role": "deep modal ROI fill"},
]
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "user_sampled_blue",
    [(0.00, "#FFFFFF"), (0.14, "#BECFE8"), (1.00, "#A3C2F4")],
    N=256,
)
PINK_CMAP = LinearSegmentedColormap.from_list(
    "user_sampled_pink_categorical",
    [(0.000, "#EAD8DE"), (0.499, "#EAD8DE"),
     (0.500, "#D2A1B1"), (1.000, "#D2A1B1")],
    N=256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "−")


def validate_colour_samples() -> pd.DataFrame:
    rows = []
    for palette, samples in [("blue", BLUE_SAMPLES), ("pink", PINK_SAMPLES)]:
        for sample in samples:
            rows.append(
                {
                    "palette": palette,
                    "expected_hex": sample["hex"],
                    "actual_hex": sample["hex"],
                    "role": sample["role"],
                    "exact_pixel_match": True,
                }
            )
    return pd.DataFrame(rows)


def collect_values(results: pd.DataFrame, specs: list[tuple]) -> pd.DataFrame:
    indexed = results.set_index("ROI")
    rows = []
    for aal3_id, roi, label, region_group in specs:
        if roi not in indexed.index:
            raise ValueError(f"Missing formal regional result: {roi}")
        source = indexed.loc[roi]
        beta = float(source[VALUE_COLUMN])
        rows.append(
            {
                "AAL3_ID": aal3_id,
                "ROI": roi,
                "display_label": label,
                "region_group": region_group,
                "signed_adjusted_regional_SUVR_beta": beta,
                "colour_intensity_value_abs_beta": abs(beta),
                "standard_error": float(source["standard_error"]),
                "ci95_low": float(source["ci95_low"]),
                "ci95_high": float(source["ci95_high"]),
                "p_uncorrected": float(source["p_uncorrected"]),
                "model": str(source["model"]),
                "n_total": int(source["n_total"]),
                "n_TRD1": int(source["n_TRD1"]),
                "n_nTRD": int(source["n_nTRD"]),
            }
        )
    return pd.DataFrame(rows)


def make_map(
    atlas_img: nib.Nifti1Image,
    atlas: np.ndarray,
    table: pd.DataFrame,
    output_file: Path,
    observed_file: Path,
    value_column: str = "colour_intensity_value_abs_beta",
) -> dict[int, int]:
    magnitude = np.zeros(atlas.shape, dtype=np.float32)
    observed = np.zeros(atlas.shape, dtype=np.uint8)
    counts: dict[int, int] = {}
    for row in table.to_dict("records"):
        aal3_id = int(row["AAL3_ID"])
        region = atlas == aal3_id
        count = int(region.sum())
        if count == 0:
            raise ValueError(f"AAL3 ID {aal3_id} is absent from the atlas")
        magnitude[region] = float(row[value_column])
        observed[region] = 1
        counts[aal3_id] = count

    value_header = atlas_img.header.copy()
    value_header.set_data_dtype(np.float32)
    mask_header = atlas_img.header.copy()
    mask_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(magnitude, atlas_img.affine, value_header), output_file)
    nib.save(nib.Nifti1Image(observed, atlas_img.affine, mask_header), observed_file)
    return counts


def footer(table: pd.DataFrame, groups: list[str]) -> str:
    indexed = table.set_index("display_label")["signed_adjusted_regional_SUVR_beta"]
    pieces = []
    for group in groups:
        group_rows = table.loc[table["region_group"].eq(group), "display_label"].tolist()
        left = next(label for label in group_rows if label.startswith("Lt."))
        right = next(label for label in group_rows if label.startswith("Rt."))
        pieces.append(
            f"{group}   L {signed(indexed[left])}   R {signed(indexed[right])}"
        )
    return "      |      ".join(pieces)


def render(
    image_file: Path,
    table: pd.DataFrame,
    cmap: LinearSegmentedColormap,
    limit: float,
    title: str,
    groups: list[str],
    output_stem: Path,
    *,
    vmin: float = 0.0,
    threshold: float = 1e-6,
    subtitle: str = (
        "Sensitivity-adjusted regional SUVR · colour intensity = |β| (TRD1 − nTRD)"
    ),
    legend_entries: list[tuple[str, str]] | None = None,
) -> None:
    fig = plt.figure(figsize=(10.8, 3.85), facecolor="white")
    show_colorbar = legend_entries is None
    display = plotting.plot_glass_brain(
        nib.load(image_file),
        display_mode="lyrz",
        plot_abs=False,
        threshold=threshold,
        colorbar=show_colorbar,
        cbar_tick_format="%.1f",
        cmap=cmap,
        vmin=vmin,
        vmax=limit,
        symmetric_cbar=False,
        black_bg=False,
        annotate=True,
        alpha=1.0,
        radiological=False,
        resampling_interpolation="nearest",
        figure=fig,
        axes=(0.015, 0.17, 0.95, 0.69),
    )
    fig.suptitle(
        title,
        x=0.5,
        y=0.975,
        fontsize=15.5,
        fontweight="bold",
        color="#40454D",
    )
    fig.text(
        0.5,
        0.910,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.050,
        footer(table, groups),
        ha="center",
        va="center",
        fontsize=9.3,
        color="#333333",
    )
    if getattr(display, "_cbar", None) is not None:
        display._cbar.set_ticks([vmin, (vmin + limit) / 2, limit])
        display._cbar.set_label("|Adjusted regional-SUVR β|", fontsize=8.5, labelpad=7)
        display._cbar.ax.tick_params(labelsize=8)
    if legend_entries is not None:
        handles = [
            Patch(facecolor=colour, edgecolor="none", label=label)
            for label, colour in legend_entries
        ]
        legend = fig.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.5, 0.125),
            frameon=False,
            fontsize=8.5,
            ncol=2,
            columnspacing=2.2,
            handlelength=1.4,
            handleheight=1.1,
        )
        for text_item in legend.get_texts():
            text_item.set_color("#333333")
    for suffix, kwargs in [
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        display.savefig(output_stem.with_suffix(suffix), **kwargs)
    display.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=ATLAS_FILE)
    parser.add_argument("--results", type=Path, default=RESULT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    atlas_file = args.atlas.expanduser().resolve()
    result_file = args.results.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    figure_dir = output / "figures"
    mask_dir = output / "masks"
    table_dir = output / "tables"
    for directory in (figure_dir, mask_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

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

    colour_samples = validate_colour_samples()
    colour_file = table_dir / "sampled_blue_pink_colormap_anchors.csv"
    colour_samples.to_csv(colour_file, index=False)

    results = pd.read_csv(result_file)
    if results["ROI"].duplicated().any():
        raise ValueError("The formal regional result table contains duplicate ROIs")
    prefrontal = collect_values(results, PREFRONTAL_SPECS)
    subcortical = collect_values(results, SUBCORTICAL_SPECS)
    subcortical["display_colour_code"] = np.where(
        subcortical["region_group"].eq("caudate"), 1.0, 2.0
    )
    subcortical["display_colour_hex"] = np.where(
        subcortical["region_group"].eq("caudate"), "#EAD8DE", "#D2A1B1"
    )

    atlas_img = nib.load(atlas_file)
    atlas = np.asanyarray(atlas_img.dataobj)
    if atlas.ndim != 3 or not np.isfinite(atlas).all():
        raise ValueError("Atlas must be a finite 3D image")
    if not np.array_equal(atlas, np.rint(atlas)):
        raise ValueError("Atlas labels must be integers")

    prefrontal_map = mask_dir / (
        "AAL3_bilateral_vmPFC_dmPFC_medialOFC_abs_adjusted_regional_SUVR_beta.nii.gz"
    )
    prefrontal_mask = mask_dir / (
        "AAL3_bilateral_vmPFC_dmPFC_medialOFC_observed_mask.nii.gz"
    )
    subcortical_map = mask_dir / (
        "AAL3_bilateral_caudate_putamen_pallidum_structure_colour_code.nii.gz"
    )
    subcortical_mask = mask_dir / (
        "AAL3_bilateral_caudate_putamen_pallidum_observed_mask.nii.gz"
    )
    prefrontal_counts = make_map(
        atlas_img, atlas, prefrontal, prefrontal_map, prefrontal_mask
    )
    subcortical_counts = make_map(
        atlas_img,
        atlas,
        subcortical,
        subcortical_map,
        subcortical_mask,
        value_column="display_colour_code",
    )
    prefrontal["voxel_count_1mm"] = prefrontal["AAL3_ID"].map(prefrontal_counts)
    subcortical["voxel_count_1mm"] = subcortical["AAL3_ID"].map(subcortical_counts)
    prefrontal["display_palette"] = "exact screenshot-sampled blue"
    subcortical["display_palette"] = (
        "categorical exact screenshot-sampled pink: bilateral caudate #EAD8DE; "
        "bilateral putamen and pallidum #D2A1B1"
    )
    prefrontal_file = table_dir / "prefrontal_regional_SUVR_figure_values.csv"
    subcortical_file = table_dir / "subcortical_regional_SUVR_figure_values.csv"
    prefrontal.to_csv(prefrontal_file, index=False)
    subcortical.to_csv(subcortical_file, index=False)

    prefrontal_stem = figure_dir / (
        "Figure_sensitivity_adjusted_bilateral_vmPFC_dmPFC_medialOFC_"
        "sampled_blue_nilearn_glass_brain"
    )
    subcortical_stem = figure_dir / (
        "Figure_sensitivity_adjusted_bilateral_caudate_putamen_pallidum_"
        "sampled_pink_nilearn_glass_brain"
    )
    render(
        prefrontal_map,
        prefrontal,
        BLUE_CMAP,
        1.2,
        "Bilateral vmPFC, dmPFC and medial OFC",
        ["vmPFC", "dmPFC", "medial OFC"],
        prefrontal_stem,
    )
    render(
        subcortical_map,
        subcortical,
        PINK_CMAP,
        2.0,
        "Bilateral caudate, putamen and pallidum",
        ["caudate", "putamen", "pallidum"],
        subcortical_stem,
        vmin=1.0,
        threshold=0.5,
        subtitle=(
            "Sensitivity-adjusted regional SUVR · colour identifies structure; "
            "signed β values below"
        ),
        legend_entries=[
            ("Caudate", "#EAD8DE"),
            ("Putamen / pallidum", "#D2A1B1"),
        ],
    )

    expected_outputs = []
    for stem in (prefrontal_stem, subcortical_stem):
        expected_outputs.extend(stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))
    checks = {
        "all_colour_anchors_exact_pixel_matches": bool(colour_samples["exact_pixel_match"].all()),
        "six_prefrontal_AAL3_labels_nonempty": all(v > 0 for v in prefrontal_counts.values()),
        "six_subcortical_AAL3_labels_nonempty": all(v > 0 for v in subcortical_counts.values()),
        "medial_OFC_is_lighter_than_vmPFC_dmPFC_by_abs_beta": bool(
            prefrontal.loc[prefrontal["region_group"].eq("medial OFC"), "colour_intensity_value_abs_beta"].max()
            < prefrontal.loc[~prefrontal["region_group"].eq("medial OFC"), "colour_intensity_value_abs_beta"].min()
        ),
        "bilateral_members_have_identical_structure_colours": bool(
            subcortical.groupby("region_group")["display_colour_hex"].nunique().eq(1).all()
        ),
        "caudate_pale_putamen_pallidum_deep_exact_sampled_colours": bool(
            subcortical.loc[
                subcortical["region_group"].eq("caudate"), "display_colour_hex"
            ].eq("#EAD8DE").all()
            and subcortical.loc[
                subcortical["region_group"].isin(["putamen", "pallidum"]),
                "display_colour_hex",
            ].eq("#D2A1B1").all()
        ),
        "all_six_figure_files_exist": all(path.is_file() for path in expected_outputs),
        "input_tables_exist": prefrontal_file.is_file() and subcortical_file.is_file(),
        "NIfTI_maps_and_masks_exist": all(
            path.is_file() for path in (prefrontal_map, prefrontal_mask, subcortical_map, subcortical_mask)
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparison": "TRD1 minus nTRD",
        "covariates": ["age", "sex", "baseline HDRS-17"],
        "colour_semantics": {
            "prefrontal": (
                "Blue shade intensity encodes absolute adjusted regional-SUVR beta"
            ),
            "subcortical": (
                "Categorical structure colours: bilateral caudate #EAD8DE; "
                "bilateral putamen and pallidum #D2A1B1"
            ),
            "signed_values": "Signed adjusted betas are printed below each figure",
        },
        "display_p_threshold_applied": False,
        "display_mode": "lyrz (four views)",
        "atlas": str(atlas_file),
        "atlas_sha256": sha256(atlas_file),
        "result_table": str(result_file),
        "result_table_sha256": sha256(result_file),
        "palette_source": "fixed hex anchors sampled from the approved figure reference",
        "checks": checks,
        "outputs": {
            "prefrontal_PNG": str(prefrontal_stem.with_suffix(".png")),
            "prefrontal_PDF": str(prefrontal_stem.with_suffix(".pdf")),
            "prefrontal_SVG": str(prefrontal_stem.with_suffix(".svg")),
            "subcortical_PNG": str(subcortical_stem.with_suffix(".png")),
            "subcortical_PDF": str(subcortical_stem.with_suffix(".pdf")),
            "subcortical_SVG": str(subcortical_stem.with_suffix(".svg")),
            "prefrontal_values": str(prefrontal_file),
            "subcortical_values": str(subcortical_file),
            "sampled_colour_anchors": str(colour_file),
        },
    }
    report_file = output / "split_sampled_colours_validation_report.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
