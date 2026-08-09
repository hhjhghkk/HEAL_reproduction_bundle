from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture versions used by this reproduction.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "provenance.json")
    args = parser.parse_args()

    report: dict = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "eai_revision": revision(ROOT / "workspace" / "embodied-agent-interface"),
        "heal_revision": revision(ROOT / "workspace" / "HEAL_dataset"),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "safetensors",
                "huggingface-hub",
                "eai-eval",
            )
        },
    }
    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
    except ImportError:
        report["cuda"] = {"available": False, "torch_cuda_build": None, "device": None}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

