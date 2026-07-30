"""Static checks for the assignment notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def _load_notebook(name: str) -> dict:
    with (NOTEBOOKS_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_all_notebook_code_cells_compile():
    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        notebook = _load_notebook(path.name)
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")


def test_model_evaluation_notebook_is_reporting_only():
    notebook = _load_notebook("03_model_training.ipynb")
    code = _code_source(notebook)

    forbidden_training_calls = (
        ".fit(",
        ".predict(",
        ".predict_proba(",
        "load_and_clean",
        "make_splits",
        "get_candidate_models",
        "select_best_model",
    )
    for call in forbidden_training_calls:
        assert call not in code

    required_saved_results = (
        "best_model_evaluation.json",
        "validation_comparison.csv",
        "final_test_metrics.csv",
        "permutation_importance.csv",
        "test_curves.json",
        "test_analysis.csv",
    )
    for artifact in required_saved_results:
        assert artifact in code
