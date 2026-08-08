from __future__ import annotations
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspace"
WS.mkdir(parents=True, exist_ok=True)
URL = "https://files.pythonhosted.org/packages/62/ae/be6df7f55b175dec2564662bd95c2d93fed7373b63cb8f567ee77b8c114b/eai_eval-1.0.5.tar.gz"
SHA256 = "61a13f38b8e60414540b44cbcb51e3b8cddd2fba9150fc0965e089940715dabf"
ARCHIVE = WS / "eai_eval-1.0.5.tar.gz"
TARGET = WS / "embodied-agent-interface"

print(f"Downloading pinned EAI sdist -> {ARCHIVE}")
urllib.request.urlretrieve(URL, ARCHIVE)
h = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
if h != SHA256:
    raise RuntimeError(f"SHA256 mismatch: {h} != {SHA256}")

TMP = WS / "_eai_extract"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()
with tarfile.open(ARCHIVE, "r:gz") as tf:
    tf.extractall(TMP)
roots = [p for p in TMP.iterdir() if p.is_dir()]
if len(roots) != 1:
    raise RuntimeError(f"Unexpected sdist layout: {roots}")
if TARGET.exists():
    shutil.rmtree(TARGET)
shutil.move(str(roots[0]), str(TARGET))
shutil.rmtree(TMP)
print(f"EAI source ready: {TARGET}")
