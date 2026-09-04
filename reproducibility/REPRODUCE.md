# Reproducing MetaKV

## Level 0 — CPU-only sanity checks

Run `python reproducibility/quick_validate.py`. This checks HA-FBMS and corrected Binary32-Z encode/decode behavior, FP32 conversion, logical-bit mapping round trips, representative amplification, master-table hashes, and figure-data integrity. No model weights or GPU are required.

## Level 1 — Processed-data reproduction

Use `data/PAPER_MASTER_NUMBERS_P10C.csv`, `figure_data/`, and `tables/` to regenerate tables and figures. These are processed evidence, not raw third-party corpora. Paths must be passed relative to the repository root.

## Level 2 — GPU microbenchmarks

Install a matching PyTorch/Triton backend and run the selected scripts in `code/experiments/`. Supply inputs and output locations via their documented positional arguments or `METAKV_*` environment variables. Results depend on GPU, driver, runtime, and kernel compilation.

## Level 3 — Real-model experiments

Obtain Llama-3.1-8B, Qwen3-14B, and Qwen3-32B from their official providers. Run scale extraction, clean full-cache one-step replay, and localized logical metadata-fault replay using the frozen configurations. Level 3 requires substantial GPU memory and compute. It cannot be reproduced on an ordinary CPU alone.

## Scope

The replay evidence is controlled one-step behavior, not a complete long-sequence generation benchmark. Logical fault injection is conditional-response analysis, not a deployed hardware fault rate.
