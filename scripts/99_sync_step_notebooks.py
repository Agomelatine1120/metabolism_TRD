#!/usr/bin/env python3
"""Rebuild the 10 Main and 5 MwC notebooks from their canonical step scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
STEP_DIR = ROOT / "pipeline_steps"

BOOTSTRAP = """from pathlib import Path
import os

_repo = Path(os.environ.get("METAFL_ROOT", Path.cwd())).resolve()
if not (_repo / "pipeline_steps").is_dir() and (_repo.parent / "pipeline_steps").is_dir():
    _repo = _repo.parent
if not (_repo / "pipeline_steps").is_dir():
    raise FileNotFoundError("Set METAFL_ROOT to the repository root")
os.environ["METAFL_ROOT"] = str(_repo)
os.chdir(_repo)

def _run_step(relative_path):
    _path = _repo / "pipeline_steps" / relative_path
    _code = compile(_path.read_text(encoding="utf-8"), str(_path), "exec")
    exec(_code, get_ipython().user_ns)
"""

MAIN = [
    ("main_01_setup_load.py", "Main 01 — setup, input loading, and validation", "Load the formal 12-cortical + 14-limbic ROI definition, clinical/PET inputs, 26-node MwC data, and validate 168 cross-network edges."),
    ("main_02_demographics_hc_reference.py", "Main 02 — demographics and HC reference", "Build the HC references and subject-level regional SUVR and 12×14 metabolic-connectivity deviations."),
    ("main_03_baseline_discovery_sensitivity.py", "Main 03 — baseline discovery and sensitivity", "Run nTRD-versus-TRD baseline regional, connectivity, MwC, and baseline-severity sensitivity analyses."),
    ("main_04_longitudinal_paired_clinical.py", "Main 04 — longitudinal paired and clinical analyses", "Run iTBS, rTMS, and sham paired analyses plus prespecified clinical correlations."),
    ("main_05_interactions_mwc.py", "Main 05 — group-by-time and MwC analyses", "Run group-by-time models and the formal 26-ROI age/sex-adjusted MwC analysis."),
    ("main_06_spatial_replication_report.py", "Main 06 — spatial replication", "Quantify TRD1–TRD2 spatial replication for 26 regional, 168 connectivity, and 26 MwC features."),
    ("main_07_core_figures_heatmaps.py", "Main 07 — core figures and heatmaps", "Create formal 26-ROI and 12×14 heatmaps plus longitudinal feature figures directly from the unified inputs."),
    ("main_08_circle_diagrams_checks.py", "Main 08 — circle diagrams", "Create the four formal 26-node/168-edge cross-sectional circle diagrams and run reproducibility checks."),
    ("main_09_five_global_indexes.py", "Main 09 — five global indexes", "Compute the five global indexes from 26 regional, 168 connectivity, and 26 MwC features and create their figures."),
    ("main_10_longitudinal_sgc_figures_audit.py", "Main 10 — longitudinal circle diagrams and final audit", "Create the four formal longitudinal 26-node/168-edge circle diagrams and audit the complete Main pipeline."),
]

MWC = [
    ("mwc_01_setup_recompute.py", "MwC 01 — setup and AAL3-144 recomputation", "Recompute MwC across all 144 retained AAL3 nodes before extracting the formal 26 targets."),
    ("mwc_02_subject_level.py", "MwC 02 — subject-level 26-ROI extraction", "Extract and export the 12-cortical + 14-limbic subject-level MwC values."),
    ("mwc_03_deviation_spatial.py", "MwC 03 — deviation maps and spatial replication", "Create the formal 26-ROI age/sex-adjusted MwC deviation heatmap and TRD1–TRD2 replication result."),
    ("mwc_04_group_analysis.py", "MwC 04 — nTRD versus TRD analysis", "Test the selected MwC deviations using the formal 26-ROI age/sex-adjusted data."),
    ("mwc_05_figures_audit.py", "MwC 05 — figures and audit", "Create the selected MwC raincloud figure and audit the five-step MwC workflow."),
]


# Full-resolution results remain in outputs/. Each notebook ends with a compact
# table-and-figure preview so GitHub readers can inspect the corresponding
# numerical and visual results without downloading separate files first.
RESULT_TABLES = {
    "mwc_01_setup_recompute.py": [
        {"title": "MwC recomputation validation", "path": "outputs/mwc/recomputed_normative_sc/calculation_validation_report.json"},
    ],
    "mwc_02_subject_level.py": [
        {"title": "Subject-level 26-ROI MwC values", "path": "outputs/mwc/subject_level_corticolimbic26_normativeSC10_MwC_wide.csv", "max_rows": 10},
    ],
    "mwc_03_deviation_spatial.py": [
        {"title": "MwC spatial replication", "path": "outputs/mwc/TRD1_TRD2_normativeSC_MwC_26ROI_spatial_summary.csv"},
        {"title": "Group mean MwC deviations", "path": "outputs/mwc/nTRD_TRD1_TRD2_normativeSC_MwC_26ROI_mean_deviations.csv", "max_rows": 26},
    ],
    "mwc_04_group_analysis.py": [
        {"title": "Adjusted nTRD-versus-TRD MwC statistics", "path": "outputs/mwc/ntrd_vs_trd_selected_unilateral_mwc_deviation_age_sex.csv"},
    ],
    "mwc_05_figures_audit.py": [
        {"title": "MwC values and adjusted group statistics", "path": "outputs/mwc/ntrd_vs_trd_selected_unilateral_mwc_deviation_age_sex.csv"},
    ],
    "main_01_setup_load.py": [
        {"title": "Analysis cohort counts", "path": "outputs/main_analyses/00_setup/cohort_summary.csv"},
    ],
    "main_02_demographics_hc_reference.py": [
        {"title": "Cohort characteristics", "path": "outputs/main_analyses/demographic_and_clinical_summary.xlsx", "sheet": "demographic"},
        {"title": "Paired stimulation sample", "path": "outputs/main_analyses/demographic_and_clinical_summary.xlsx", "sheet": "stimulation_paired"},
    ],
    "main_03_baseline_discovery_sensitivity.py": [
        {"title": "Regional SUVR findings (p < .01)", "path": "outputs/main_analyses/01_TRD1_discovery_agesex/regional_uncorrected_p_lt_0p01.xlsx"},
        {"title": "Metabolic-connectivity findings (p < .01)", "path": "outputs/main_analyses/01_TRD1_discovery_agesex/edge_uncorrected_p_lt_0p01.xlsx", "max_rows": 30},
        {"title": "Baseline-severity sensitivity summary", "path": "outputs/main_analyses/08_TRD1_vs_nTRD_HDRS_sensitivity/TRD1_vs_nTRD_baseline_HDRS_sensitivity.xlsx", "sheet": "summary"},
    ],
    "main_04_longitudinal_paired_clinical.py": [
        {"title": "Longitudinal sample and result counts", "path": "outputs/main_analyses/02_TRD2_deviation_prepost_agesex/prepost_and_HDRS_summary_counts.xlsx"},
        {"title": "iTBS connectivity changes used for figures (paired p < .01)", "path": "outputs/main_analyses/02_TRD2_deviation_prepost_agesex/03_iTBS_prepost/edge_pairttest_all.xlsx", "query": "paired_p_uncorrected < 0.01"},
        {"title": "rTMS connectivity changes used for figures (paired p < .01)", "path": "outputs/main_analyses/02_TRD2_deviation_prepost_agesex/04_rTMS_prepost/edge_pairttest_all.xlsx", "query": "paired_p_uncorrected < 0.01"},
        {"title": "iTBS HDRS-17 associations", "path": "outputs/main_analyses/02_TRD2_deviation_prepost_agesex/03_iTBS_prepost/edge_HDRS_partial_spearman_all_prepost_p_lt_0p05.xlsx", "max_rows": 20},
        {"title": "rTMS HDRS-17 associations", "path": "outputs/main_analyses/02_TRD2_deviation_prepost_agesex/04_rTMS_prepost/edge_HDRS_partial_spearman_all_prepost_p_lt_0p05.xlsx", "max_rows": 20},
    ],
    "main_05_interactions_mwc.py": [
        {"title": "Group-by-time findings (p < .05)", "path": "outputs/main_analyses/05_active_vs_sham_interaction/active_treatment_vs_sham_group_by_time.xlsx", "sheet": "interaction_p_lt_0p05"},
        {"title": "Longitudinal MwC findings (paired p < .05)", "path": "outputs/main_analyses/07_age_sex_adjusted_MwC/tables/iTBS_rTMS_paired_p_lt_0p05.csv"},
    ],
    "main_06_spatial_replication_report.py": [
        {"title": "TRD1-TRD2 spatial replication", "path": "outputs/main_analyses/03_TRD1_TRD2_spatial_correlation/spatial_correlation_summary.csv"},
        {"title": "Integrated result summary", "path": "outputs/main_analyses/all_reported_results.xlsx", "sheet": "summary", "max_rows": 30},
    ],
    "main_07_core_figures_heatmaps.py": [
        {"title": "Longitudinal features selected dynamically at paired p < .01", "path": "outputs/main_analyses/04_figures/TRD2_prepost_raincloud_data.xlsx", "sheet": "paired_tests"},
        {"title": "HDRS-17 correlation statistics", "path": "outputs/main_analyses/04_figures/TRD2_prepost_HDRS_correlation_panel_green_reference_data.xlsx", "sheet": "statistics"},
    ],
    "main_08_circle_diagrams_checks.py": [
        {"title": "Cross-sectional regional node values", "path": "outputs/main_analyses/04_figures/corticolimbic_circular_plot_data.xlsx", "sheet": "regional_nodes", "max_rows": 30},
        {"title": "Cross-sectional connectivity edge values", "path": "outputs/main_analyses/04_figures/corticolimbic_circular_plot_data.xlsx", "sheet": "connectivity_edges", "max_rows": 30},
    ],
    "main_09_five_global_indexes.py": [
        {"title": "Five indexes: nTRD versus TRD", "path": "outputs/main_analyses/10_metabolic_signature_five_metrics/TRD1_vs_nonTRD_five_metrics.xlsx", "sheet": "all_five_metrics"},
        {"title": "Five indexes: active stimulation pre-post", "path": "outputs/main_analyses/10_metabolic_signature_five_metrics/active_postpre_five_metrics.xlsx", "sheet": "all_five_metrics"},
    ],
    "main_10_longitudinal_sgc_figures_audit.py": [
        {"title": "Longitudinal connectivity effects", "path": "outputs/main_analyses/04_figures/corticolimbic_circular_longitudinal_effects_data.xlsx", "sheet": "connectivity_edges", "max_rows": 30},
        {"title": "Baseline-severity-adjusted regional results", "path": "outputs/main_analyses/11_SGC_26ROI_figures/tables/baseline_severity_adjusted_TRD1_vs_nTRD_regional26_SGC.csv", "query": "p_uncorrected < 0.05", "max_rows": 30},
        {"title": "Baseline-severity-adjusted connectivity results", "path": "outputs/main_analyses/11_SGC_26ROI_figures/tables/baseline_severity_adjusted_TRD1_vs_nTRD_connectivity168_SGC.csv", "query": "p_uncorrected < 0.05", "max_rows": 30},
    ],
}

INLINE_FIGURES = {
    "mwc_03_deviation_spatial.py": [
        "outputs/mwc/figures/Figure_nTRD_TRD1_TRD2_normativeSC_MwC_26ROI_deviation_heatmap.png",
        "outputs/mwc/figures/Figure_TRD1_TRD2_normativeSC_MwC_26ROI_spatial_correlation.png",
    ],
    "mwc_05_figures_audit.py": [
        "outputs/mwc/figures/Figure_selected_unilateral_MwC_deviation_raincloud.png",
    ],
    "main_06_spatial_replication_report.py": [
        "outputs/main_analyses/04_figures/spatial_correlation_scatter.png",
    ],
    "main_07_core_figures_heatmaps.py": [
        "outputs/main_analyses/04_figures/nTRD_TRD1_TRD2_regional26_strip_heatmap.png",
        "outputs/main_analyses/04_figures/nTRD_TRD1_TRD2_corticolimbic12x14_heatmap.png",
        "outputs/main_analyses/04_figures/nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap.png",
        "outputs/main_analyses/04_figures/TRD2_prepost_raincloud.png",
        "outputs/main_analyses/04_figures/TRD2_prepost_HDRS_correlation_panel_green_reference.png",
    ],
    "main_08_circle_diagrams_checks.py": [
        "outputs/main_analyses/04_figures/corticolimbic_circular_summary.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_nonTRD_mean_deviation.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_TRD1_mean_deviation.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_TRD2_mean_deviation.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_TRD1_vs_nonTRD_all.png",
    ],
    "main_09_five_global_indexes.py": [
        "outputs/main_analyses/10_metabolic_signature_five_metrics/Figure_five_metric_nTRD_TRD_raincloud.png",
        "outputs/main_analyses/04_figures/nTRD_TRD_HDRS_sensitivity_top10_raincloud.png",
        "outputs/main_analyses/10_metabolic_signature_five_metrics/Figure_five_metric_active_pre_post_raincloud.png",
    ],
    "main_10_longitudinal_sgc_figures_audit.py": [
        "outputs/main_analyses/04_figures/corticolimbic_circular_longitudinal_summary.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_rtms_post_pre.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_itbs_post_pre.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_sham_post_pre.png",
        "outputs/main_analyses/04_figures/corticolimbic_circular_active_vs_sham.png",
        "outputs/main_analyses/11_SGC_26ROI_figures/figures/corticolimbic_circular_26ROI_bilateral_SGC_TRD_vs_nonTRD_baseline_severity_adjusted.png",
        "outputs/main_analyses/11_SGC_26ROI_figures/figures/Figure_TRD_vs_nonTRD_significant_MwC_nilearn_glass_brain.png",
        "outputs/main_analyses/13_sensitivity_mPFC_striatopallidal_glass_brain/figures/Figure_sensitivity_adjusted_bilateral_vmPFC_dmPFC_medialOFC_sampled_blue_nilearn_glass_brain.png",
        "outputs/main_analyses/13_sensitivity_mPFC_striatopallidal_glass_brain/figures/Figure_sensitivity_adjusted_bilateral_caudate_putamen_pallidum_sampled_pink_nilearn_glass_brain.png",
        "outputs/main_analyses/15_nonTRD_TRD_significant_regional_SUVR_glass_brain/figures/Figure_TRD_vs_nonTRD_significant_regional_SUVR_combined_nilearn_glass_brain.png",
    ],
}

RUN_ALL_TABLES = [
    {"title": "Integrated analysis summary", "path": "outputs/main_analyses/all_reported_results.xlsx", "sheet": "summary", "max_rows": 30},
    {"title": "TRD1-TRD2 spatial replication", "path": "outputs/main_analyses/03_TRD1_TRD2_spatial_correlation/spatial_correlation_summary.csv"},
    {"title": "Longitudinal features used for figures (paired p < .01)", "path": "outputs/main_analyses/04_figures/TRD2_prepost_raincloud_data.xlsx", "sheet": "paired_tests"},
    {"title": "Five indexes: nTRD versus TRD", "path": "outputs/main_analyses/10_metabolic_signature_five_metrics/TRD1_vs_nonTRD_five_metrics.xlsx", "sheet": "all_five_metrics"},
]

RUN_ALL_FIGURES = [
    "outputs/main_analyses/04_figures/nTRD_TRD1_TRD2_regional26_strip_heatmap.png",
    "outputs/main_analyses/04_figures/nTRD_TRD1_TRD2_corticolimbic12x14_heatmap.png",
    "outputs/main_analyses/04_figures/nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap.png",
    "outputs/main_analyses/04_figures/corticolimbic_circular_summary.png",
    "outputs/main_analyses/04_figures/TRD2_prepost_raincloud.png",
    "outputs/main_analyses/10_metabolic_signature_five_metrics/Figure_five_metric_nTRD_TRD_raincloud.png",
]


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def markdown(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source_lines(text)}


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": cell_id,
        "metadata": {}, "outputs": [], "source": source_lines(text.rstrip()),
    }


def inline_results_code(table_specs: list[dict], figure_paths: list[str]) -> str:
    """Return compact, deterministic table and figure previews."""
    return f'''from io import BytesIO
import json as _preview_json
import pandas as _preview_pd
from PIL import Image as PILImage
from IPython.display import Image as NotebookImage, Markdown, display

_inline_table_specs = {table_specs!r}
_inline_figure_paths = {figure_paths!r}

display(Markdown("### Key tables"))
for _table_spec in _inline_table_specs:
    _relative_table = _table_spec["path"]
    _table_path = _repo / _relative_table
    if not _table_path.is_file():
        raise FileNotFoundError(_table_path)
    if _table_path.suffix.lower() == ".csv":
        _preview_table = _preview_pd.read_csv(_table_path)
    elif _table_path.suffix.lower() in {{".xlsx", ".xls"}}:
        _preview_table = _preview_pd.read_excel(
            _table_path, sheet_name=_table_spec.get("sheet", 0)
        )
    elif _table_path.suffix.lower() == ".json":
        _payload = _preview_json.loads(_table_path.read_text(encoding="utf-8"))
        _checks = _payload.get("checks", _payload)
        _preview_table = _preview_pd.DataFrame(
            {{"item": list(_checks), "value": list(_checks.values())}}
        ) if isinstance(_checks, dict) else _preview_pd.json_normalize(_payload)
    else:
        raise ValueError(f"Unsupported table format: {{_table_path.suffix}}")
    if _table_spec.get("query"):
        _preview_table = _preview_table.query(_table_spec["query"]).reset_index(drop=True)
    _max_rows = int(_table_spec.get("max_rows", 40))
    display(Markdown(
        f"**{{_table_spec['title']}}**  \\n"
        f"`{{_relative_table}}` — {{len(_preview_table)}} rows × {{len(_preview_table.columns)}} columns"
    ))
    with _preview_pd.option_context("display.max_columns", 40, "display.max_colwidth", 60):
        display(_preview_table.head(_max_rows))

if _inline_figure_paths:
    display(Markdown("### Figures"))
for _relative_figure in _inline_figure_paths:
    _figure_path = _repo / _relative_figure
    if not _figure_path.is_file():
        raise FileNotFoundError(_figure_path)
    with PILImage.open(_figure_path) as _source_image:
        _preview_image = _source_image.convert("RGB")
        _preview_image.thumbnail((1600, 1200), PILImage.Resampling.LANCZOS)
        _preview_buffer = BytesIO()
        _preview_image.save(_preview_buffer, format="PNG", optimize=True)
    display(Markdown(f"**`{{_relative_figure}}`**"))
    display(NotebookImage(data=_preview_buffer.getvalue(), format="png"))
'''


def preserve_matching_outputs(target: Path, document: dict) -> None:
    """Keep outputs when a regenerated code cell has identical source text."""
    if not target.is_file():
        return
    previous = json.loads(target.read_text(encoding="utf-8"))
    previous_by_id = {
        cell.get("id"): cell
        for cell in previous.get("cells", [])
        if cell.get("cell_type") == "code" and cell.get("id")
    }
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        old_cell = previous_by_id.get(cell.get("id"))
        if old_cell and old_cell.get("source") == cell.get("source"):
            cell["execution_count"] = old_cell.get("execution_count")
            cell["outputs"] = old_cell.get("outputs", [])


def chunks(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^# Source cell (\d+)\s*$", text))
    if not matches:
        return [("analysis", text)]
    output = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        output.append(("preamble", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            output.append((f"source-{match.group(1)}", body))
    return output


def build_notebook(pipeline: str, number: int, specification: tuple[str, str, str]) -> None:
    filename, title, purpose = specification
    dependency_files = (
        [item[0] for item in (MAIN if pipeline == "main" else MWC)[: number - 1]]
    )
    dependency_code = BOOTSTRAP + "".join(
        f'_run_step("{item}")\n'
        for item in dependency_files
    )
    intro = (
        f"# {title}\n\n{purpose}\n\n"
        f"- Formal analysis universe: 12 cortical + 14 limbic ROIs (26 nodes)\n"
        f"- Cross-network connectivity: 12 × 14 = 168 edges\n"
        f"- Pipeline: `{pipeline}`; step {number} of {10 if pipeline == 'main' else 5}\n"
        "- Run mode: standalone; all prerequisite steps load automatically in the same kernel."
    )
    cells = [markdown(intro, "intro"), code(dependency_code, "bootstrap")]
    for chunk_index, (label, body) in enumerate(chunks(STEP_DIR / filename), start=1):
        cells.append(markdown(f"## {label.replace('-', ' ').title()}", f"heading-{chunk_index}"))
        cells.append(code(body, f"code-{chunk_index}"))
    cells.append(markdown(
        "## Results preview\n\nKey numerical tables and figures generated by this step are displayed below. "
        "The full-resolution files remain under `outputs/`.",
        "heading-results-preview",
    ))
    cells.append(code(
        inline_results_code(RESULT_TABLES.get(filename, []), INLINE_FIGURES.get(filename, [])),
        "results-preview",
    ))
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "metafl_pipeline": {
                "pipeline": pipeline, "step": number,
                "dependencies": dependency_files,
                "formal_roi_counts": {"cortical": 12, "limbic": 14, "total": 26, "cross_edges": 168},
            },
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    prefix = f"{number:02d}_{pipeline}_{number:02d}_"
    stem = Path(filename).stem.removeprefix(f"{pipeline}_{number:02d}_")
    target = NOTEBOOK_DIR / f"{prefix}{stem}.ipynb"
    preserve_matching_outputs(target, document)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build_run_all() -> None:
    cells = [
        markdown(
            "# MetaFL — run all formal analysis steps\n\n"
            "Runs MwC 01–05 and then Main 01–10 in one kernel. The formal universe is 12 cortical + 14 limbic ROIs and 168 cross-network edges.",
            "intro",
        ),
        code(BOOTSTRAP, "bootstrap"),
    ]
    for section, steps in [("MwC pipeline", MWC), ("Main pipeline", MAIN)]:
        cells.append(markdown(f"## {section}", f"section-{section.split()[0].lower()}"))
        for index, (filename, title, purpose) in enumerate(steps, start=1):
            safe_id = filename.replace(".", "-")
            cells.append(markdown(f"### {title}\n\n{purpose}", f"heading-{safe_id}"))
            cells.append(code(
                f'_run_step("{filename}")',
                f"run-{safe_id}",
            ))
    cells.append(markdown(
        "## Results preview\n\nSelected tables and figures from the complete workflow are displayed below. "
        "Step-specific notebooks contain the corresponding detailed previews.",
        "heading-results-preview",
    ))
    cells.append(code(
        inline_results_code(RUN_ALL_TABLES, RUN_ALL_FIGURES),
        "results-preview",
    ))
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "metafl_pipeline": {"pipeline": "all", "formal_roi_counts": {"cortical": 12, "limbic": 14, "total": 26, "cross_edges": 168}},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    target = NOTEBOOK_DIR / "00_run_all_steps.ipynb"
    preserve_matching_outputs(target, document)
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def main() -> None:
    for index, item in enumerate(MWC, start=1):
        build_notebook("mwc", index, item)
    for index, item in enumerate(MAIN, start=1):
        build_notebook("main", index, item)
    build_run_all()
    print("Synchronized 5 MwC notebooks, 10 Main notebooks, and run-all notebook.")


if __name__ == "__main__":
    main()
