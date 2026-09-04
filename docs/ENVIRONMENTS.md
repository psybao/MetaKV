# Environments

Level 0 requires Python 3.10 or newer and no GPU packages. Levels 1–3 use the focused dependencies in `requirements.txt`.

| Platform | Recorded identity | Backend | Notes |
|---|---|---|---|
| RTX 5080 | NVIDIA GeForce RTX 5080 | CUDA/Triton | Frozen P1E platform; consult provenance for exact local environment. |
| RTX 4060 Ti | NVIDIA GeForce RTX 4060 Ti | CUDA/Triton | Cross-GPU returned package; frozen requirements inventory retained in the source audit. |
| AMD gfx942 | AMD Instinct MI308X | ROCm/Triton | gfx942 device. It is not identified as MI300X. |
| H800 | NVIDIA H800 PCIe | CUDA/Triton | PyTorch 2.12.1+cu130, Triton 3.7.1, CUDA runtime 13.0. |

Model experiments require locally obtained model weights. Pass model and output paths through the environment variables or positional arguments expected by each script.
