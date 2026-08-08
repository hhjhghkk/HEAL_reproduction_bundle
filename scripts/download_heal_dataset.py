from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace" / "HEAL_dataset"
OUT.parent.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    raise SystemExit(
        "Missing huggingface_hub. Install with: pip install -U huggingface_hub"
    ) from e

path = snapshot_download(
    repo_id="Trishna13/HEAL",
    repo_type="dataset",
    local_dir=str(OUT),
)
print(f"HEAL dataset ready: {path}")
