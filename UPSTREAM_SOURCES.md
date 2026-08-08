# Upstream sources and pinned versions

## HEAL
- Paper: `paper/HEAL_EMNLP2025.pdf`
- Dataset: https://huggingface.co/datasets/Trishna13/HEAL
- The HEAL paper exposes the probing set through Hugging Face; no standalone HEAL GitHub repository is stated in the paper.

## Embodied Agent Interface (EAI)
- GitHub: https://github.com/embodied-agent-interface/embodied-agent-interface
- PyPI package: `eai-eval`
- Pinned source distribution: `eai_eval-1.0.5.tar.gz`
- PyPI release date: 2025-01-14
- SHA256: `61a13f38b8e60414540b44cbcb51e3b8cddd2fba9150fc0965e089940715dabf`
- License: MIT; copy included as `upstream/EAI_LICENSE.txt`.

The download scripts prefer the current GitHub `main` branch. If GitHub cloning fails, they fall back to the pinned PyPI 1.0.5 source distribution for reproducibility.
