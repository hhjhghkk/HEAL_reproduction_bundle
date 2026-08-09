from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.metrics import load_predictions, score_predictions, write_scores




def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HEAL model responses.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "workspace" / "HEAL_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    scores = score_predictions(args.data_root, predictions)
    summary_path, details_path = write_scores(scores, args.output_dir)
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Sample details: {details_path}")


if __name__ == "__main__":
    main()
