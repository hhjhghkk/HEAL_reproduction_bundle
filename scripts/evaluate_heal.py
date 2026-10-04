from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#从metrics.py导入三个函数，分别是：
#load_predictions：读取predictions.jsonl
#score_predictions：对每条预测打分
#write_scores：把评估结果写入文件
from heal_repro.metrics import load_predictions, score_predictions, write_scores




def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HEAL model responses.")
    #指定评估的预测文件
    parser.add_argument("--predictions", type=Path, required=True)
    #指定HEAL数据集路径
    parser.add_argument("--data-root", type=Path, default=ROOT / "workspace" / "HEAL_dataset")
    #指定EAI源码路径，用于读取VirtualHome的标准node/edge/action goal
    parser.add_argument(
        "--eai-root",
        type=Path,
        default=ROOT / "workspace" / "embodied-agent-interface",
    )
    #指定评估结果保存在哪里
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args()

    #读取预测文件
    predictions = load_predictions(args.predictions)
    #对预测打分
    scores = score_predictions(args.data_root, predictions, eai_root=args.eai_root)
    #把评估结果写入文件
    summary_path, details_path = write_scores(scores, args.output_dir)
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Sample details: {details_path}")


if __name__ == "__main__":
    main()
