# Reproducing MetaKV

## CPU-only checks

From the repository root, run `python reproducibility/quick_validate.py` and `python reproducibility/verify_public_results.py`.

## GPU deployment microbenchmarks

Selected scripts are under `code/experiments/`. Supply model/config inputs and output directories through documented CLI arguments or `METAKV_*` environment variables. Published timings are kernel microbenchmarks and depend on GPU, driver, runtime, and compiler.

## Model-dependent experiments

Llama-3.1-8B, Qwen3-14B, and Qwen3-32B weights are not redistributed. Full-cache evidence is one-step next-token replay, not full autoregressive generation or a production throughput test.

## HBM-derived characterization analysis

MetaKV did not rerun the underlying HBM2 hardware characterization. The third-party DSN 2024 corpus is not redistributed. Only permitted derived metadata is public.

## Reproducibility level

`REPRODUCIBLE_WITH_PUBLIC_ARTIFACTS=PARTIAL`. CPU codec, mapping, processed-table, hash, and public-result checks are reproducible. GPU/model-dependent workflows require external hardware, model weights, and the separately licensed HBM2 source corpus.
