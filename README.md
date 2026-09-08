# MetaKV

MetaKV studies fault-resilient 32-bit scale metadata for low-precision LLM KV caches. HA-FBMS separates coarse magnitude from local refinement to limit tested single-bit fault amplification without adding metadata bits. Logical remapping and runtime behavior are evaluated separately and remain backend- and characterization-dependent.

## Software dependencies

Python 3.10 or later is recommended. CPU validation uses NumPy, pandas, SciPy, and PyYAML. GPU experiments additionally require a hardware-compatible PyTorch and Triton build. Install the portable analysis dependencies with `python -m pip install -r requirements.txt`.

## Hardware requirements

CPU-only codec, manifest, and result checks require no GPU. Packed-kernel and replay experiments require the platform named by each experiment: NVIDIA RTX 4060 Ti, RTX 5080, or H800 with CUDA, or AMD gfx942-class hardware with ROCm. Qwen replay requires separately obtained model weights; weights are not redistributed.

## Quick start

```bash
python reproducibility/quick_validate.py
python reproducibility/verify_public_results.py
python scripts/reproduce_figures.py --source experiments/frozen_results/figures --output reproduced/figures
python scripts/reproduce_tables.py --source experiments/frozen_results/core/metakv_main_tables_v3.md --output reproduced/tables
```

Platform-specific commands and required environment variables are listed in `experiments/RUN_COMMANDS.md`. Exact historical absolute paths in provenance records are replaced with portable placeholders in this release copy.

## Paper-to-experiment correspondence

- HA-FBMS geometry and single-bit amplification: `src/metakv_codec.py`, `src/fault_screen.py`, and `experiments/frozen_results/core/`.
- Packed deployment: `src/experiments/p1e_multiconfig_generality.py` and frozen deployment results.
- Full-cache and 32-step replay: `experiments/p8_amd_qwen3_32b/`.
- AMD P2D1/P2D2/P2D3 attribution: `scripts/backend_attribution/AMD/` and `experiments/backend_attribution/AMD/`.
- H800 P2N1/P2N2 attribution: `scripts/backend_attribution/H800/` and `experiments/backend_attribution/H800/`.
- RTX 5080 scale validation: `scripts/backend_attribution/RTX5080/` and `experiments/backend_attribution/RTX5080/`.
- DDR/HBM2 mapping analysis: `scripts/mapping_generalization/` and `experiments/mapping_generalization/`.

See `CLAIM_EVIDENCE_MATRIX.md` for file-level traceability. Logical mapping is software-level and does not claim physical DRAM/HBM lane control. MetaKV is complementary to ECC.
