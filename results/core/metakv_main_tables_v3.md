# Main tables V3

## Table 1. Metadata representations and fault geometry

| Representation | Width | Scale-domain interpretation | Tested fault response | Scope |
|---|---:|---|---|---|
| FP32 | 32 bits | Direct IEEE-754 Binary32 scale metadata | Can become invalid in selected logical-fault screens | Logical single-bit screen |
| Binary32-Z | 32 bits | Uniform unsigned encoding of log2(scale) over the calibrated interval | Maximum amplification 256.0000003305 on 8B/14B and 512.0000007437 on 32B | Corrected P6B2/P7B-R2 only |
| HA-FBMS | 32 bits | Structured Bc/Bf log-scale metadata | Qwen3-32B maximum 1.8660659831; GE2=0 | Audited model-calibrated configurations |

Source metrics: MODEL-Llama-3.1-8B-BMAX, MODEL-Qwen3-14B-BMAX, MODEL-Qwen3-32B-BMAX, Q32-HA-MAX.

## Table 2. DDR/HBM2 characterization-weighted mapping evidence

| Domain or transfer | Metric | Value | Interpretation |
|---|---|---:|---|
| DDR corpus | Strict word-fault rows | 1,067,712 | Corpus scale, not a hardware rate |
| DDR scenarios | Mean H≤3 coverage | 0.7355736886 | Mean across 32 unpooled scenarios |
| HBM2 corpus | Logical-position CV | 0.8834596847 | Non-uniform logical-position observations |
| HBM2 corpus | Top-eight position mass | 0.4761350711 | Descriptive corpus concentration |
| HBM→DDR | Mean domain-reference ratio | 1.00644782 | Relative to the DDR-native reference |
| DDR→HBM | Mean domain-reference ratio | 2.90789646 | Relative to the HBM-native reference |
| Robust selection | Mean maximum reference ratio | 1.00167557 | Eight selected representatives |

Source metrics: M1-ROWS, M2-H3, M3D-CV, M3D-TOP8, M3F-H2D, M3F-D2H, M3G-MAX.

## Table 3. Largest-workload GPU deployment results

| Platform | Representation | Mapped vs. FP32 median (%) | 95% CI (%) | Safe interpretation |
|---|---|---:|---:|---|
| RTX5080 | REP0008 | -1.0226274220 | [-2.1864830127, 1.7472397738] | CI crosses zero |
| RTX4060Ti | REP0026 | 0.1592395459 | [-0.4036587540, 0.4711454213] | CI crosses zero |
| AMD gfx942 | REP0008 | 165.0672031288 | [165.0025061754, 165.0756804176] | Functional, not performance, portability |

The RTX rows show the largest absolute median among the four reported representations on each platform. H800 is discussed as separate authoritative sensitivity evidence because its headline maximum is not part of the P9 machine-readable master-number table. Source metrics: PLAT-RTX5080-REP0008, PLAT-RTX4060Ti-REP0026, PLAT-AMDgfx942-REP0008.

## Table 4. Real-model scale robustness

| Model | Layers | Scale groups | Calibrated interval | Width | Clip count | Corrected Binary32-Z max A | HA-FBMS result |
|---|---:|---:|---|---:|---:|---:|---|
| Llama-3.1-8B | 32 | 276,480 | [-13, 3] | 16 | 0 | 256.0000003305 | Max A < 2; GE2=0 in screen evidence |
| Qwen3-14B | 40 | 317,440 | [-11, 5] | 16 | 0 | 256.0000003305 | Max A < 2; GE2=0 in screen evidence |
| Qwen3-32B | 64 | 507,904 | [-13, 5] | 18 | 0 | 512.0000007437 | Max A=1.8660659831; GE2=0 |

Source metrics: MODEL-Llama-3.1-8B-SCALES, MODEL-Qwen3-14B-SCALES, MODEL-Qwen3-32B-SCALES, corresponding BMAX metrics, and Q32-HA-MAX. Layer, interval, width, and clipping facts come from the audited model-scale table and method facts.

## Table 5. Full-cache clean replay

| Model scope | Replay | Observed result | Interpretation |
|---|---|---|---|
| Llama-3.1-8B and Qwen3-14B | One-step next-token, captured full cache | No observed HA-vs-raw INT4 top-1 change or additional logit perturbation in tested prompts | Does not imply raw INT4 equals BF16 |
| Qwen3-32B | Eight prompt rows | Clean HA-FBMS top-1 changes=0 | One-step replay, not long generation |

Source metric: Q32-CLEAN-TOP1; remaining scope is from the audited model-scale table and claim matrix.

## Table 6. Full-cache localized fault-semantic validation

| Model | Encoding | Selected targets | Top-1 changes | Maximum KL | Maximum amplification |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | Corrected Binary32-Z | 96 | 0 | Not promoted to master table | 256.0000003305 |
| Qwen3-14B | Corrected Binary32-Z | 96 | 2 | median 0.0; p95 0.00041855545714497566; max 11.833429336547852 | 256.0000003305185 |
| Qwen3-32B | Corrected Binary32-Z | 24 | 0 | Not promoted to master table | 512.0000007437 |
| Qwen3-32B | HA-FBMS | 24 | 0 | 0.0015947018 | 1.8660659831 |

The corrected P6B2 Qwen3-14B distribution has median 0.0, p95 0.00041855545714497566, and maximum 11.833429336547852. The low median and p95 alongside an extreme maximum identify rare-outlier, heavy-tail behavior; the maximum is not representative of a typical target. Source metrics: the model BMAX/BTOP1 metrics, Q32-HA-MAX, and Q32-HA-KL.
