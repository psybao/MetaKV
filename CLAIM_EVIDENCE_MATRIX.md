# MetaKV Claim Evidence Matrix

## Paper claim

| Claim | Corresponding experiment | Corresponding code | Corresponding result file |
|---|---|---|---|
| HA-FBMS limits tested logical single-bit fault amplification | N3/P1 fault geometry screens | `code/src/metakv_codec.py`; `code/src/fault_screen.py` | `raw_results/core/`; `raw_results/deployment/` |
| Single-bit validity and amplification differ by representation | exhaustive real-scale screens | `code/src/fault_screen.py`; `code/src/experiments/m55n3b1c_deep_confirmation.py` | `raw_results/core/` |
| Delayed divergence appears in 32-step replay | AMD P8 Qwen3-32B replay | `code/experiments/p8_amd_qwen3_32b/scripts/p8_formal_v1.py` | `code/experiments/p8_amd_qwen3_32b/results/` |
| AMD runtime-permutation cost is associated with VALU pressure | P2D1 static ISA, P2D2 counters, P2D3 causal A/B | `code/scripts/backend_attribution/AMD/` | `raw_results/backend_attribution/AMD/` |
| H800 runtime permutation expands SASS without a material isolated latency penalty | P2N1 microbenchmark and P2N2 SASS | `code/scripts/backend_attribution/H800/` | `raw_results/backend_attribution/H800/` |
| RTX 5080 shows no measurable largest-scale penalty in the audited scan | P2N2B scale scan | `code/scripts/backend_attribution/RTX5080/` | `raw_results/backend_attribution/RTX5080/` |
| Logical remapping depends on positional stability | P19 scrambling and P21 source-aware DDR validation | `code/scripts/mapping_generalization/` | `raw_results/mapping_generalization/` |

The matrix maps software-level logical representation evidence only. It does not assert physical memory-lane control or redistribute model weights or third-party HBM2 raw data.
