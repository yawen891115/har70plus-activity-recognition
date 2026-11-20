"""Run one full-cohort model using the code cells from the original notebook."""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Cell numbers follow the notebook's displayed order, including Markdown cells.
MODELS = {
    "rf": ("classical-models.ipynb", (2, 4, 8), "RF.png"),
    "logistic": ("classical-models.ipynb", (2, 4, 10), "logistic.png"),
    "qda": ("classical-models.ipynb", (2, 4, 12), "QDA.png"),
    "nb": ("classical-models.ipynb", (2, 4, 16), "NB.png"),
    "mlp": ("neural-models.ipynb", (1, 3), "MLP.png"),
    "cnn": ("neural-models.ipynb", (1, 5), "CNN.png"),
    "lstm": ("neural-models.ipynb", (1, 7), "LSTM.png"),
}


def run_cells(model, data_dir, output_dir, *, features=None):
    """Run the archived cells; the sensor ablation may select RF input columns."""
    if features is not None and model != "rf":
        raise ValueError("Sensor selection is supported for the classical RF notebook.")
    data_dir = data_dir.resolve()
    missing = [f"{i}.csv" for i in range(501, 519) if not (data_dir / f"{i}.csv").is_file()]
    if missing:
        raise ValueError(f"Missing inputs in {data_dir}: {', '.join(missing)}. See data/README.md.")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")

    name, positions, figure = MODELS[model]
    notebook = json.loads((HERE / name).read_text(encoding="utf-8"))
    state = {"input_dir": data_dir, "figure_path": output_dir / figure, "__name__": "__main__"}
    for position in positions:
        cell = notebook["cells"][position - 1]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected a code cell in {name}, cell {position}.")
        source = "".join(cell["source"])
        # Resolve only the original file locations. Keep the shared notebook
        # namespace, random states, preprocessing, model and score calculations.
        source = source.replace("f'/root/Project/har70dataset/{i}.csv'", "input_dir / f'{i}.csv'")
        # The archived NB cell saves as QDA.png; give each classifier its own file.
        archived_figure = "QDA.png" if model == "nb" else figure
        source = source.replace(f"'/root/Project/{archived_figure}'", "figure_path")
        print(f"Running {name}, cell {position}", flush=True)
        if model in {"logistic", "qda", "nb"} and position == positions[-1]:
            # Cell 8 reassigns kf after the RF scoring loop. Subsequent classical
            # model cells inherit this unshuffled splitter in notebook order.
            state["kf"] = state["KFold"](n_splits=5)
        if features is not None and position == positions[-1]:
            state["X"] = state["X"].loc[:, features]
        exec(compile(source, f"{name} cell {position}", "exec"), state)
    print(f"Figure: {state['figure_path']}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "paper")
    args = parser.parse_args()

    try:
        run_cells(args.model, args.data_dir, args.output_dir)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
