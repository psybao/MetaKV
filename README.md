# MetaKV

## Paper

**MetaKV: Fault-Resilient Structured Metadata for Low-Precision LLM KV Caches**

Repository: https://github.com/psybao/MetaKV  
Archived v1.0 artifact: https://doi.org/10.5281/zenodo.22308465

## Core idea

MetaKV treats group-scale metadata as a fault-amplification control point. HA-FBMS is a structured 32-bit log-scale representation with a thermometer-coded coarse field and a binary fine field. It preserves metadata width while bounding the tested logical single-bit scale response for calibrated configurations.

Characterization-weighted mapping is a software logical metadata-bit layout. In historical implementation notation, `mapping[logical] = physical`, where `physical` means a software layout index only. It does not mean a physical HBM lane, DRAM placement, or controller-visible remapping.

## Repository structure

- `code/`: reference codecs, logical mapping, weighting, fault screens, and selected experiment scripts
- `configs/`: public reference configuration
- `results/core/`: processed master numbers and figure/table sources
- `results/deployment/`: authoritative raw blocks, summaries, statuses, and a normalized primary table
- `results/full_cache/`: corrected Binary32-Z P6B2 summary and target rows
- `results/diagnostic/`: failed or diagnostic evidence excluded from primary tables
- `provenance/`: public-safe evidence identities and version notes
- `reproducibility/`: CPU validation and layered reproduction instructions

## Reproduction levels

### CPU-only checks

Run `python reproducibility/quick_validate.py` and `python reproducibility/verify_public_results.py`.

### GPU deployment microbenchmarks

GPU timing scripts require matching PyTorch/Triton environments. Timings are kernel microbenchmarks, not end-to-end LLM latency.

### Model-dependent experiments

Model weights are not redistributed. P3, P4, and P7 use deployment-calibrated configurations in the same HA-FBMS codec family. They are not proven byte-identical historical N3 instances. `HISTORICAL_N0_N3_CODEC_IDENTITY_PROVEN=False`.

### HBM-derived characterization analysis

The HBM2 observations derive from *Read Disturbance in High Bandwidth Memory: A Detailed Experimental Study on HBM2 DRAM Chips*, DSN 2024. MetaKV did not rerun that hardware characterization and does not redistribute the third-party raw corpus.

## Public data and limitations

Fault injection is controlled software logical metadata-fault injection. It does not measure HBM BER, deployed hardware fault probability, or real-event frequency. Full-cache experiments are one-step next-token replay, not full quantized autoregressive generation. Binary32-Z is a corrected same-width conventional baseline: uniform uint32 coding of log2 scale over a calibrated interval. It is not an original N3 representation.

`REPRODUCIBLE_WITH_PUBLIC_ARTIFACTS=PARTIAL` because GPU hardware, model weights, and third-party HBM2 data are external.

## Platform naming

Execution platforms are `rtx5080_windows`, `rtx5080_linux_h800_like`, `rtx4060ti`, `amd_gfx942`, and `h800`. Linux H800-like names the software-stack sensitivity; its execution GPU is NVIDIA GeForce RTX 5080. AMD execution hardware is reported conservatively as an AMD gfx942-class accelerator. H800 R6 is `DIAGNOSTIC_ONLY` and is excluded from primary results.

## Artifact version

This package is the proposed corrected public artifact v1.1-submission. It preserves v1.0-submission and changes packaging, naming, provenance, documentation, and public result coverage. `SCIENTIFIC_RESULT_CHANGE=False`.

## Citation

See `CITATION.cff`. The immutable v1.0 DOI is 10.5281/zenodo.22308465. A new Zenodo version is recommended for this corrected package after author approval.
