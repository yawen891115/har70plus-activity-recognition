"""Generate deterministic, synthetic six-channel features for inference demos."""
import argparse
from pathlib import Path

import numpy as np


def demo_features(rows=4096):
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, size=(rows, 6)).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination .npy file")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, demo_features(), allow_pickle=False)
