from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "workspace" / "HEAL_dataset"

if not DATA.exists():
    raise SystemExit("HEAL_dataset not found. Run: bash scripts/fetch_all.sh")

files = sorted(p for p in DATA.rglob("*") if p.is_file() and ".git" not in p.parts)
print(f"HEAL root: {DATA}")
print(f"Files: {len(files)}")
for p in files:
    print(" -", p.relative_to(DATA), p.stat().st_size, "bytes")

# Report CSV row counts and schemas used by the public HEAL artifact.
for p in files:
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        print(
            f"CSV {p.relative_to(DATA)}: rows={len(rows)}, "
            f"columns={reader.fieldnames}"
        )

# Retain support for a future JSON serialization of the dataset.
for p in files:
    if p.suffix.lower() == ".json":
        print(f"\nPreview JSON: {p.relative_to(DATA)}")
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                print("type=list, length=", len(obj))
                print(json.dumps(obj[:1], ensure_ascii=False, indent=2)[:4000])
            elif isinstance(obj, dict):
                print("type=dict, keys=", list(obj)[:20])
                print(json.dumps(obj, ensure_ascii=False, indent=2)[:4000])
            break
        except Exception as e:
            print("Could not preview:", e)
