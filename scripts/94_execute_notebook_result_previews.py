#!/usr/bin/env python3
"""Execute every notebook from top to bottom and store all outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def execute_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=10800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(notebook, path)


def main() -> None:
    paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if len(paths) != 16:
        raise RuntimeError(f"Expected 16 notebooks; found {len(paths)}")
    for path in paths:
        execute_notebook(path)
        print(f"Executed {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
