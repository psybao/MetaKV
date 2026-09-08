# Reproduction commands

All commands run from the package root. Historical model weights and third-party HBM2 characterization data are external inputs and are not redistributed.

```bash
python reproducibility/quick_validate.py
python src/fault_screen.py
python scripts/core_experiments/p1e_multiconfig_generality.py
python experiments/p8_amd_qwen3_32b/scripts/p8_formal_v1.py
python scripts/mapping_generalization/p19_mapping_scrambling_sensitivity.py
python scripts/mapping_generalization/p21_ddr_robust_generalization.py
python scripts/backend_attribution/AMD/p2d1_static_isa/p2d1b_real_kernel_dump__p2d1b_compile_once.py
python scripts/backend_attribution/AMD/p2d2_dynamic_profile/p2d2b_r3_rep0025_4m_minimal.py
python scripts/backend_attribution/AMD/p2d3_causal_ab/p2d3a_rep0025_causal_ab.py
python scripts/backend_attribution/H800/p2n1a_h800_sass_microbenchmark.py
python scripts/backend_attribution/H800/p2n1b_h800_latency_ab.py
python scripts/backend_attribution/H800/p2n2b_scale_scan.py
python scripts/backend_attribution/RTX5080/p2n2b_scale_scan_5080.py
```

GPU/model scripts require the environment variables documented in their headers or must be adapted to the local package root. Frozen result verification does not require rerunning GPU experiments.
