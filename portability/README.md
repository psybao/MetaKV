# CPU and Apple M4 portability evidence

This directory contains the source-traceable portability evidence added for the TJS V2 manuscript.

- `intel_cpu/`: Intel Core i7-14700KF x86-64 packed-INT4 timing evidence.
- `amd_cpu/`: AMD Ryzen 7 7840HS x86-64 packed-INT4 timing evidence; this is separate from gfx942 GPU evidence.
- `apple_m4/cpu/`: Apple M4 ARM64 CPU packed-INT4 timing evidence.
- `apple_m4/mps/`: PyTorch MPS functional evidence only, not a timing benchmark.
- `apple_m4/mlx/`: MLX Metal functional scale-sweep evidence only, not a timing benchmark.

Frozen numerical results are copied unchanged. Public provenance uses portable root placeholders where the internal audit recorded private absolute paths.
