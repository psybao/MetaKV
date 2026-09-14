import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


SEED = 20260902
BATCH = 1
HEADS = 4
HEAD_DIM = 32

SEQ_LENGTHS = [
    128,
    512,
    1024,
    2048,
]

MAPPING = [
    7, 22, 1, 28, 13, 4, 19, 31,
    10, 25, 0, 16, 5, 21, 12, 29,
    3, 18, 8, 26, 14, 30, 6, 23,
    11, 27, 2, 20, 15, 9, 24, 17,
]


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def pack_int4(q):
    q = np.asarray(q, dtype=np.int8)

    require(
        np.all((q >= -8) & (q <= 7)),
        "INT4 range failure",
    )

    flat = q.reshape(-1)

    require(
        flat.size % 2 == 0,
        "INT4 count must be even",
    )

    n = (
        flat.astype(np.int16)
        & 0xF
    ).astype(np.uint8)

    return (
        n[0::2]
        | (n[1::2] << 4)
    ).astype(np.uint8)


def mlx_unpack_int4(packed):
    p = packed.astype(mx.uint32)

    lo = p & 0xF
    hi = (p >> 4) & 0xF

    lo = mx.where(
        lo >= 8,
        lo.astype(mx.int32) - 16,
        lo.astype(mx.int32),
    )

    hi = mx.where(
        hi >= 8,
        hi.astype(mx.int32) - 16,
        hi.astype(mx.int32),
    )

    return mx.stack(
        [lo, hi],
        axis=1,
    ).reshape(-1)


def map_numpy(src):
    src = np.asarray(src, dtype=np.uint32)
    dst = np.zeros_like(src)

    for logical, physical in enumerate(MAPPING):
        bit = (
            src
            >> np.uint32(logical)
        ) & np.uint32(1)

        dst |= (
            bit
            << np.uint32(physical)
        )

    return dst


def unmap_numpy(src):
    src = np.asarray(src, dtype=np.uint32)
    dst = np.zeros_like(src)

    for logical, physical in enumerate(MAPPING):
        bit = (
            src
            >> np.uint32(physical)
        ) & np.uint32(1)

        dst |= (
            bit
            << np.uint32(logical)
        )

    return dst


def map_mlx(src):
    dst = mx.zeros(
        src.shape,
        dtype=mx.uint32,
    )

    one = mx.array(
        1,
        dtype=mx.uint32,
    )

    for logical, physical in enumerate(MAPPING):
        dst = dst | (
            (
                (src >> logical)
                & one
            )
            << physical
        )

    return dst


def unmap_mlx(src):
    dst = mx.zeros(
        src.shape,
        dtype=mx.uint32,
    )

    one = mx.array(
        1,
        dtype=mx.uint32,
    )

    for logical, physical in enumerate(MAPPING):
        dst = dst | (
            (
                (src >> physical)
                & one
            )
            << logical
        )

    return dst


def run_case(seq, rng):
    total = (
        BATCH
        * HEADS
        * seq
        * HEAD_DIM
    )

    q = rng.integers(
        -7,
        8,
        size=total,
        dtype=np.int8,
    )

    packed = pack_int4(q)

    packed_gpu = mx.array(
        packed,
        dtype=mx.uint8,
    )

    unpacked = mlx_unpack_int4(
        packed_gpu
    )

    mx.eval(unpacked)

    unpacked_np = np.asarray(
        unpacked,
        dtype=np.int32,
    ).astype(np.int8)

    require(
        np.array_equal(
            q,
            unpacked_np,
        ),
        f"INT4 mismatch seq={seq}",
    )

    metadata = rng.integers(
        0,
        2 ** 32,
        size=max(4096, seq * HEADS),
        dtype=np.uint32,
    )

    mapped_cpu = map_numpy(
        metadata
    )

    recovered_cpu = unmap_numpy(
        mapped_cpu
    )

    require(
        np.array_equal(
            metadata,
            recovered_cpu,
        ),
        f"CPU mapping failure seq={seq}",
    )

    meta_gpu = mx.array(
        metadata,
        dtype=mx.uint32,
    )

    mapped_gpu = map_mlx(
        meta_gpu
    )

    recovered_gpu = unmap_mlx(
        mapped_gpu
    )

    mx.eval(
        mapped_gpu,
        recovered_gpu,
    )

    mapped_gpu_np = np.asarray(
        mapped_gpu,
        dtype=np.uint32,
    )

    recovered_gpu_np = np.asarray(
        recovered_gpu,
        dtype=np.uint32,
    )

    require(
        np.array_equal(
            mapped_cpu,
            mapped_gpu_np,
        ),
        f"CPU/MLX map mismatch seq={seq}",
    )

    require(
        np.array_equal(
            metadata,
            recovered_gpu_np,
        ),
        f"MLX roundtrip failure seq={seq}",
    )

    scales = rng.uniform(
        0.02,
        0.25,
        size=(
            BATCH,
            HEADS,
            seq,
            1,
        ),
    ).astype(np.float32)

    reference = (
        q.astype(np.float32)
        .reshape(
            BATCH,
            HEADS,
            seq,
            HEAD_DIM,
        )
        * scales
    )

    values = (
        unpacked
        .astype(mx.float32)
        .reshape(
            BATCH,
            HEADS,
            seq,
            HEAD_DIM,
        )
    )

    scales_gpu = mx.array(
        scales,
        dtype=mx.float32,
    )

    values = values * scales_gpu

    query = mx.array(
        rng.normal(
            0,
            0.3,
            size=(
                BATCH,
                HEADS,
                1,
                HEAD_DIM,
            ),
        ).astype(np.float32)
    )

    scores = mx.matmul(
        query,
        mx.swapaxes(
            values,
            -1,
            -2,
        ),
    ) / math.sqrt(HEAD_DIM)

    probs = mx.softmax(
        scores,
        axis=-1,
    )

    output = mx.matmul(
        probs,
        values,
    )

    mx.eval(
        values,
        scores,
        probs,
        output,
    )

    values_np = np.asarray(
        values,
        dtype=np.float32,
    )

    output_np = np.asarray(
        output,
        dtype=np.float32,
    )

    max_err = float(
        np.max(
            np.abs(
                values_np
                - reference
            )
        )
    )

    require(
        max_err < 1.0e-6,
        f"dequant error seq={seq}: {max_err}",
    )

    require(
        np.isfinite(
            output_np
        ).all(),
        f"attention output non-finite seq={seq}",
    )

    return {
        "seq_len": seq,
        "int4_values": total,
        "packed_bytes": int(
            packed.size
        ),
        "metadata_words": int(
            metadata.size
        ),
        "int4_bit_exact": True,
        "metadata_cpu_mlx_equal": True,
        "metadata_roundtrip_bit_exact": True,
        "max_dequant_error": max_err,
        "attention_output_finite": True,
        "pass": True,
    }


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(
            "usage: script.py output_dir"
        )

    out = Path(
        sys.argv[1]
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mx.set_default_device(
        mx.gpu
    )

    rng = np.random.default_rng(
        SEED
    )

    results = []

    for seq in SEQ_LENGTHS:
        row = run_case(
            seq,
            rng,
        )

        results.append(row)

        print(
            f"SEQ={seq} "
            f"INT4_BIT_EXACT=True "
            f"MAP_CPU_MLX_EQUAL=True "
            f"MAP_ROUNDTRIP=True "
            f"DEQUANT_MAX_ERR="
            f"{row['max_dequant_error']:.12g} "
            f"ATTENTION_FINITE=True "
            f"PASS=True"
        )

    require(
        all(
            row["pass"]
            for row in results
        ),
        "not all scale cases passed",
    )

    csv_path = (
        out
        / "M4_MLX_KV_SCALE_SWEEP.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    payload = {
        "platform":
            "Apple M4",
        "backend":
            "MLX Metal GPU",
        "seed":
            SEED,
        "sequence_lengths":
            SEQ_LENGTHS,
        "performance_claim":
            False,
        "end_to_end_llm":
            False,
        "physical_memory_mapping":
            False,
        "software_metadata_bit_layout":
            True,
        "all_cases_pass":
            True,
        "results":
            results,
    }

    (
        out
        / "M4_MLX_KV_SCALE_SWEEP.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    status_lines = [
        "METAKV M4 MLX KV SCALE SWEEP",
        "BACKEND=MLX_METAL_GPU",
        "SEQ_LENGTHS=128,512,1024,2048",
        "CASE_COUNT=4",
        "ALL_INT4_BIT_EXACT=True",
        "ALL_METADATA_CPU_MLX_EQUAL=True",
        "ALL_METADATA_ROUNDTRIP_BIT_EXACT=True",
        "ALL_DEQUANT_CORRECT=True",
        "ALL_ATTENTION_OUTPUT_FINITE=True",
        "PERFORMANCE_CLAIM=False",
        "END_TO_END_LLM=False",
        "PHYSICAL_MEMORY_MAPPING=False",
        "M4_MLX_KV_SCALE_SWEEP_PASS=True",
    ]

    (
        out
        / "M4_MLX_KV_SCALE_SWEEP_STATUS.txt"
    ).write_text(
        "\n".join(
            status_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "M4_MLX_KV_SCALE_SWEEP_PASS=True"
    )


if __name__ == "__main__":
    main()
