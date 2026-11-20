"""Launch an archived research script with local data and output paths."""

import argparse
import ast
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ARCHIVED_DESKTOP = "/Users/zhengyawen/Desktop/"
ARCHIVED_INPUTS = (
    ARCHIVED_DESKTOP + "zywhar70/har70dataset/",
    ARCHIVED_DESKTOP + "zywhar70/har70plus/",
)
SUMMARY = "statistical_analysis_results.csv"


class LocalPaths(ast.NodeTransformer):
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for prefix in ARCHIVED_INPUTS:
                if node.value.startswith(prefix):
                    # Keep the slash for paths assembled inside an f-string.
                    node.value = self.data_dir.as_posix().rstrip("/") + "/" + node.value[len(prefix):]
                    return node
            if node.value == ARCHIVED_DESKTOP + SUMMARY:
                node.value = str(self.output_dir / SUMMARY)
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scripts = {path.name: path for path in (PAPER / "scripts").glob("*/*.py")}
    parser.add_argument("script", choices=sorted(scripts))
    parser.add_argument("--data-dir", type=Path, default=PAPER.parent / "data" / "raw",
                        help="directory containing the numbered participant CSVs (default: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=PAPER.parent / "outputs" / "paper",
                        help="destination for the descriptive-statistics CSV (default: outputs/paper)")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"CSV directory does not exist: {data_dir}. See data/README.md.")
    output_dir = args.output_dir.resolve()
    if args.script == "stat descri.py":
        output_dir.mkdir(parents=True, exist_ok=True)
    script = scripts[args.script]
    tree = LocalPaths(data_dir, output_dir).visit(ast.parse(script.read_bytes(), str(script)))
    namespace = {"__name__": "__main__", "__file__": str(script)}
    if args.script == "har70logistic.py":
        # The original uses both names after printing its pooled metrics.
        from sklearn.metrics import classification_report, confusion_matrix
        namespace.update(classification_report=classification_report,
                         confusion_matrix=confusion_matrix)
    exec(compile(tree, str(script), "exec"), namespace)


if __name__ == "__main__":
    main()
