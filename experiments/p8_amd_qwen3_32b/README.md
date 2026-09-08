# AMD MI308X Qwen3-32B multi-step replay

This directory contains the passing P8 formal evidence used by the updated MetaKV manuscript. The run used an AMD Instinct MI308X (`gfx942:sramecc+:xnack-`), Qwen3-32B, eight frozen prompts, 24 frozen logical targets, three metadata encodings, and 72 total cases. Each trajectory ran for 32 steps with checkpoints at 1, 8, 16, and 32 in `TEACHER_FORCED` and `FREE_RUNNING` modes. `TARGET_RESELECTION=False`, `FROZEN_LOCATION_TRANSFER=True`, and `AMD_EXACT_REPRODUCTION=False`.

The logical target locations were frozen before the AMD replay. Small clean-state numerical differences existed across backends; therefore the local clean scale, z, and metadata word were reconstructed on AMD at the same frozen logical location before applying the same encoding-specific bit fault.

## Public results

| Metric | Binary32-Z | HA-FBMS |
|---|---:|---:|
| Maximum scale amplification | 512.000000743667 | 1.8660659831 |
| Teacher-forced divergence by step 1 | 0/24 | 0/24 |
| Teacher-forced divergence by step 8 | 4/24 | 2/24 |
| Teacher-forced divergence by step 16 | 8/24 | 4/24 |
| Teacher-forced divergence by step 32 | 8/24 | 5/24 |
| Exact sequence at step 32 | 16/24 | 19/24 |
| Teacher-forced mean token agreement at step 32 | 0.9674479167 | 0.9908854167 |
| Teacher-forced maximum KL | 9.6921710968 | 0.007775790989 |
| Teacher-forced mean-case mean KL | 0.06949804767 | 0.000709157912 |
| Free-running divergence by step 32 | 8/24 | 5/24 |
| Free-running mean token agreement at step 32 | 0.7734375 | 0.8880208333 |
| Free-running maximum KL | 30.496213913 | 28.262727737 |
| Free-running mean-case mean KL | 3.8237588 | 1.9664425364 |

All 24 tested FP32 sign-bit faults produced invalid scale metadata before replay. FP32 multi-step replay is therefore N/A; these cases must not be reported as zero divergence.

## Interpretation and limitations

This is localized logical metadata injection into the initial full-cache representation, not real measured HBM fault injection. It does not estimate HBM BER, absolute hardware fault probability, or simultaneous multi-bit hardware fault probability. The 32-step trajectory is short-horizon replay, not a long-form generation guarantee. Teacher-forced replay provides the cleaner logit-perturbation comparison because both branches use the clean token history. After first divergence, free-running KL is path-dependent because token histories differ; it describes autoregressive propagation rather than pure metadata-fault magnitude.

The results are aggregate observations over a finite frozen target set. HA-FBMS is not guaranteed to outperform Binary32-Z at every target. Scale amplification and downstream KL or token divergence are nonlinear. Logical metadata-bit mapping is not physical DRAM/HBM lane placement. Representation-level resilience complements, rather than replaces, ECC, memory scrubbing, system redundancy, and platform-level protection.

The discarded RTX5080/Qwen3-4B P8 sensitivity experiment is excluded. Existing RTX5080 P1E kernel-performance evidence elsewhere in the artifact remains valid and is retained.
