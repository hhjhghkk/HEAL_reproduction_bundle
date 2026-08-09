from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import EXPECTED_ROWS, VARIANT_FILES, parse_scene, prompt_from_row, read_rows


DEFAULT_DATA = ROOT / "workspace" / "HEAL_dataset"


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate downloaded HEAL/EAI artifacts.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args()

    report: dict = {
        "heal_revision": git_revision(args.data_root),
        "eai_revision": git_revision(ROOT / "workspace" / "embodied-agent-interface"),
        "datasets": {},
        "paper_public_data_discrepancy": (
            "Paper Table 13 reports 100 BEHAVIOR samples for distractor, synonym, and "
            "contradiction (2574 modified prompts total); public data has 99 each "
            "(2571 total)."
        ),
    }
    errors: list[str] = []
    modified_total = 0
    for environment, expected in EXPECTED_ROWS.items():
        report["datasets"][environment] = {}
        for variant in VARIANT_FILES:
            rows = read_rows(args.data_root, environment, variant)
            parsed_scenes = sum(
                bool(parse_scene(prompt_from_row(row, variant))) for row in rows
            )
            actual = len(rows)
            expected_count = expected[variant]
            report["datasets"][environment][variant] = {
                "rows": actual,
                "expected_public_rows": expected_count,
                "scenes_parsed": parsed_scenes,
            }
            if variant != "baseline":
                modified_total += actual
            if actual != expected_count:
                errors.append(f"{environment}/{variant}: {actual} != {expected_count}")
            if parsed_scenes != actual:
                errors.append(f"{environment}/{variant}: only {parsed_scenes}/{actual} scenes parsed")
    report["modified_prompt_total"] = modified_total
    report["errors"] = errors

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"HEAL revision: {report['heal_revision']}")
        print(f"EAI revision:  {report['eai_revision']}")
        for environment, variants in report["datasets"].items():
            print(f"\n{environment}")
            for variant, values in variants.items():
                print(
                    f"  {variant:27s} rows={values['rows']:4d} "
                    f"parsed={values['scenes_parsed']:4d}"
                )
        print(f"\nModified prompts in public artifact: {modified_total}")
        print(f"Note: {report['paper_public_data_discrepancy']}")
        print("Validation:", "PASS" if not errors else "FAIL")
        for error in errors:
            print(" -", error)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
