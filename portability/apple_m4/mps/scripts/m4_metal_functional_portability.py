import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx


SEED = 20260902

BATCH = 1
HEADS = 4
SEQ = 128
HEAD_DIM = 32

BC = 19
BF = 13

# Deterministic compile-time logical->physical bit mapping.
# This is a software metadata bit-layout portability test,
# NOT physical DRAM/HBM lane placement.
MAPPING = [
    7, 22, 1, 28, 13, 4, 19, 31,
    10, 25, 0, 16, 5, 21, 12, 29,
    3, 18, 8, 26, 14, 30, 6, 23,
    11, 27, 2, 20, 15, 9, 24, 17,
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest().upper()


def torch_mps_test(rng):
    require(
        torch.backends.mps.is_built(),
        "PyTorch was not built with MPS support",
    )

    require(
        torch.backends.mps.is_available(),
        "PyTorch MPS backend unavailable",
    )

    shape = (
        BATCH,
        HEADS,
        SEQ,
        HEAD_DIM,
    )

    q_np = rng.normal(
        0.0,
        0.4,
        size=shape,
    ).astype(np.float32)

    k_np = rng.normal(
        0.0,
        0.4,
        size=shape,
    ).astype(np.float32)

    v_np = rng.normal(
        0.0,
        0.4,
        size=shape,
    ).astype(np.float32)

    q_cpu = torch.from_numpy(q_np)
    k_cpu = torch.from_numpy(k_np)
    v_cpu = torch.from_numpy(v_np)

    scale = 1.0 / math.sqrt(HEAD_DIM)

    cpu_scores = torch.matmul(
        q_cpu,
        k_cpu.transpose(-1, -2),
    ) * scale

    cpu_prob = torch.softmax(
        cpu_scores,
        dim=-1,
    )

    cpu_out = torch.matmul(
        cpu_prob,
        v_cpu,
    )

    device = torch.device("mps")

    q_gpu = q_cpu.to(device)
    k_gpu = k_cpu.to(device)
    v_gpu = v_cpu.to(device)

    gpu_scores = torch.matmul(
        q_gpu,
        k_gpu.transpose(-1, -2),
    ) * scale

    gpu_prob = torch.softmax(
        gpu_scores,
        dim=-1,
    )

    gpu_out = torch.matmul(
        gpu_prob,
        v_gpu,
    )

    torch.mps.synchronize()

    gpu_np = gpu_out.cpu().numpy()
    cpu_np = cpu_out.numpy()

    max_abs = float(
        np.max(
            np.abs(
                gpu_np - cpu_np
            )
        )
    )

    mean_abs = float(
        np.mean(
            np.abs(
                gpu_np - cpu_np
            )
        )
    )

    finite = bool(
        np.isfinite(gpu_np).all()
    )

    require(
        finite,
        "MPS output contains NaN/Inf",
    )

    require(
        max_abs < 5.0e-4,
        (
            "MPS-vs-CPU max error too large: "
            f"{max_abs}"
        ),
    )

    return {
        "torch_version": torch.__version__,
        "mps_built": True,
        "mps_available": True,
        "shape": list(shape),
        "max_abs_error_vs_cpu": max_abs,
        "mean_abs_error_vs_cpu": mean_abs,
        "finite": finite,
        "pass": True,
    }


def pack_int4(q):
    q = np.asarray(
        q,
        dtype=np.int8,
    )

    require(
        np.all(
            (q >= -8)
            & (q <= 7)
        ),
        "INT4 values outside [-8,7]",
    )

    flat = q.reshape(-1)

    require(
        flat.size % 2 == 0,
        "INT4 element count must be even",
    )

    nibble = (
        flat.astype(np.int16)
        & 0xF
    ).astype(np.uint8)

    lo = nibble[0::2]
    hi = nibble[1::2]

    packed = (
        lo
        | (hi << 4)
    ).astype(np.uint8)

    return packed


def unpack_int4_numpy(packed):
    p = np.asarray(
        packed,
        dtype=np.uint8,
    )

    lo = (
        p
        & np.uint8(0x0F)
    ).astype(np.int16)

    hi = (
        (p >> np.uint8(4))
        & np.uint8(0x0F)
    ).astype(np.int16)

    lo = np.where(
        lo >= 8,
        lo - 16,
        lo,
    )

    hi = np.where(
        hi >= 8,
        hi - 16,
        hi,
    )

    out = np.empty(
        p.size * 2,
        dtype=np.int8,
    )

    out[0::2] = lo.astype(np.int8)
    out[1::2] = hi.astype(np.int8)

    return out


def map_words_numpy(words):
    src = np.asarray(
        words,
        dtype=np.uint32,
    )

    dst = np.zeros_like(src)

    for logical_bit, physical_bit in enumerate(MAPPING):
        bit = (
            src
            >> np.uint32(logical_bit)
        ) & np.uint32(1)

        dst |= (
            bit
            << np.uint32(physical_bit)
        )

    return dst


def unmap_words_numpy(words):
    src = np.asarray(
        words,
        dtype=np.uint32,
    )

    dst = np.zeros_like(src)

    for logical_bit, physical_bit in enumerate(MAPPING):
        bit = (
            src
            >> np.uint32(physical_bit)
        ) & np.uint32(1)

        dst |= (
            bit
            << np.uint32(logical_bit)
        )

    return dst


def mlx_map_words(words):
    dst = mx.zeros(
        words.shape,
        dtype=mx.uint32,
    )

    one = mx.array(
        1,
        dtype=mx.uint32,
    )

    for logical_bit, physical_bit in enumerate(MAPPING):
        bit = (
            words
            >> logical_bit
        ) & one

        dst = dst | (
            bit
            << physical_bit
        )

    return dst


def mlx_unmap_words(words):
    dst = mx.zeros(
        words.shape,
        dtype=mx.uint32,
    )

    one = mx.array(
        1,
        dtype=mx.uint32,
    )

    for logical_bit, physical_bit in enumerate(MAPPING):
        bit = (
            words
            >> physical_bit
        ) & one

        dst = dst | (
            bit
            << logical_bit
        )

    return dst


def mlx_unpack_int4(packed):
    p = packed.astype(
        mx.uint32
    )

    lo = p & 0xF
    hi = (p >> 4) & 0xF

    lo_signed = mx.where(
        lo >= 8,
        lo.astype(mx.int32) - 16,
        lo.astype(mx.int32),
    )

    hi_signed = mx.where(
        hi >= 8,
        hi.astype(mx.int32) - 16,
        hi.astype(mx.int32),
    )

    stacked = mx.stack(
        [lo_signed, hi_signed],
        axis=1,
    )

    return stacked.reshape(-1)


def mlx_test(rng):
    # Make MLX explicitly use the Metal GPU.
    mx.set_default_device(mx.gpu)

    total_values = (
        BATCH
        * HEADS
        * SEQ
        * HEAD_DIM
    )

    q_int4 = rng.integers(
        -7,
        8,
        size=total_values,
        dtype=np.int8,
    )

    packed_np = pack_int4(q_int4)

    unpacked_cpu = unpack_int4_numpy(
        packed_np
    )

    require(
        np.array_equal(
            unpacked_cpu,
            q_int4,
        ),
        "CPU packed INT4 round-trip failed",
    )

    # Arbitrary metadata words exercise all 32 logical bits.
    metadata_np = rng.integers(
        0,
        2 ** 32,
        size=4096,
        dtype=np.uint32,
    )

    mapped_cpu = map_words_numpy(
        metadata_np
    )

    recovered_cpu = unmap_words_numpy(
        mapped_cpu
    )

    require(
        np.array_equal(
            metadata_np,
            recovered_cpu,
        ),
        "CPU metadata mapping round-trip failed",
    )

    metadata_gpu = mx.array(
        metadata_np,
        dtype=mx.uint32,
    )

    mapped_gpu = mlx_map_words(
        metadata_gpu
    )

    recovered_gpu = mlx_unmap_words(
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
            mapped_gpu_np,
            mapped_cpu,
        ),
        "MLX mapped metadata differs from CPU reference",
    )

    require(
        np.array_equal(
            recovered_gpu_np,
            metadata_np,
        ),
        "MLX metadata mapping round-trip failed",
    )

    packed_gpu = mx.array(
        packed_np,
        dtype=mx.uint8,
    )

    unpacked_gpu = mlx_unpack_int4(
        packed_gpu
    )

    mx.eval(unpacked_gpu)

    unpacked_gpu_np = np.asarray(
        unpacked_gpu,
        dtype=np.int32,
    ).astype(np.int8)

    require(
        np.array_equal(
            unpacked_gpu_np,
            q_int4,
        ),
        "MLX packed INT4 unpack differs from original",
    )

    # Synthetic per-head scale metadata.
    scales_np = rng.uniform(
        0.02,
        0.25,
        size=(
            BATCH,
            HEADS,
            SEQ,
            1,
        ),
    ).astype(np.float32)

    q_float_cpu = (
        q_int4.astype(np.float32)
        .reshape(
            BATCH,
            HEADS,
            SEQ,
            HEAD_DIM,
        )
        * scales_np
    )

    q_gpu_float = (
        unpacked_gpu.astype(mx.float32)
        .reshape(
            BATCH,
            HEADS,
            SEQ,
            HEAD_DIM,
        )
    )

    scales_gpu = mx.array(
        scales_np,
        dtype=mx.float32,
    )

    dequant_gpu = (
        q_gpu_float
        * scales_gpu
    )

    # Small KV-style contraction on Metal.
    query_np = rng.normal(
        0.0,
        0.3,
        size=(
            BATCH,
            HEADS,
            1,
            HEAD_DIM,
        ),
    ).astype(np.float32)

    query_gpu = mx.array(
        query_np,
        dtype=mx.float32,
    )

    scores_gpu = mx.matmul(
        query_gpu,
        mx.swapaxes(
            dequant_gpu,
            -1,
            -2,
        ),
    ) / math.sqrt(HEAD_DIM)

    probs_gpu = mx.softmax(
        scores_gpu,
        axis=-1,
    )

    out_gpu = mx.matmul(
        probs_gpu,
        dequant_gpu,
    )

    mx.eval(
        dequant_gpu,
        scores_gpu,
        probs_gpu,
        out_gpu,
    )

    dequant_gpu_np = np.asarray(
        dequant_gpu,
        dtype=np.float32,
    )

    max_dequant_error = float(
        np.max(
            np.abs(
                dequant_gpu_np
                - q_float_cpu
            )
        )
    )

    require(
        max_dequant_error < 1.0e-6,
        (
            "MLX INT4 dequant error too large: "
            f"{max_dequant_error}"
        ),
    )

    out_np = np.asarray(
        out_gpu,
        dtype=np.float32,
    )

    require(
        np.isfinite(out_np).all(),
        "MLX attention-like output contains NaN/Inf",
    )

    return {
        "default_device": str(
            mx.default_device()
        ),
        "packed_int4_values":
            int(total_values),
        "packed_bytes":
            int(packed_np.size),
        "metadata_words":
            int(metadata_np.size),
        "mapping_width_bits": 32,
        "mapping_logical_to_physical":
            MAPPING,
        "metadata_mapping_matches_cpu":
            True,
        "metadata_roundtrip_bit_exact":
            True,
        "packed_int4_roundtrip_bit_exact":
            True,
        "max_dequant_error_vs_cpu":
            max_dequant_error,
        "attention_output_finite":
            True,
        "pass":
            True,
    }


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(
            "usage: script.py <output_dir>"
        )

    out_dir = Path(
        sys.argv[1]
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        SEED
    )

    result = {
        "stage":
            "M4_METAL_FUNCTIONAL_PORTABILITY",
        "seed":
            SEED,
        "platform":
            platform.platform(),
        "machine":
            platform.machine(),
        "python":
            sys.version.split()[0],
        "claims": {
            "functional_portability_only":
                True,
            "performance_benchmark":
                False,
            "end_to_end_llm":
                False,
            "physical_memory_lane_mapping":
                False,
            "software_metadata_bit_layout":
                True,
            "synthetic_kv_cache_workload":
                True,
        },
    }

    result["torch_mps"] = torch_mps_test(
        rng
    )

    result["mlx_metal"] = mlx_test(
        rng
    )

    result["overall_pass"] = bool(
        result["torch_mps"]["pass"]
        and result["mlx_metal"]["pass"]
    )

    require(
        result["overall_pass"],
        "overall portability gate failed",
    )

    json_path = (
        out_dir
        / "M4_METAL_FUNCTIONAL_RESULT.json"
    )

    json_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    status_path = (
        out_dir
        / "M4_METAL_FUNCTIONAL_STATUS.txt"
    )

    lines = [
        "METAKV APPLE M4 METAL FUNCTIONAL PORTABILITY",
        "ARCH=arm64",
        "WORKLOAD=SYNTHETIC_KV_CACHE",
        "TORCH_MPS_BACKEND_TESTED=True",
        "TORCH_MPS_ATTENTION_LIKE_PASS=True",
        (
            "TORCH_MPS_MAX_ABS_ERROR_VS_CPU="
            f"{result['torch_mps']['max_abs_error_vs_cpu']:.12g}"
        ),
        "MLX_METAL_BACKEND_TESTED=True",
        "MLX_PACKED_INT4_ROUNDTRIP_BIT_EXACT=True",
        "MLX_METADATA_MAPPING_MATCHES_CPU=True",
        "MLX_METADATA_ROUNDTRIP_BIT_EXACT=True",
        (
            "MLX_MAX_DEQUANT_ERROR_VS_CPU="
            f"{result['mlx_metal']['max_dequant_error_vs_cpu']:.12g}"
        ),
        "MLX_ATTENTION_OUTPUT_FINITE=True",
        "METADATA_MAPPING_WIDTH_BITS=32",
        "FUNCTIONAL_PORTABILITY_ONLY=True",
        "PERFORMANCE_BENCHMARK=False",
        "END_TO_END_LLM=False",
        "PHYSICAL_MEMORY_LANE_MAPPING=False",
        "SOFTWARE_METADATA_BIT_LAYOUT=True",
        "M4_METAL_FUNCTIONAL_PORTABILITY_PASS=True",
    ]

    status_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "TORCH_MPS_ATTENTION_LIKE_PASS=True"
    )

    print(
        "TORCH_MPS_MAX_ABS_ERROR_VS_CPU="
        f"{result['torch_mps']['max_abs_error_vs_cpu']:.12g}"
    )

    print(
        "MLX_PACKED_INT4_ROUNDTRIP_BIT_EXACT=True"
    )

    print(
        "MLX_METADATA_MAPPING_MATCHES_CPU=True"
    )

    print(
        "MLX_METADATA_ROUNDTRIP_BIT_EXACT=True"
    )

    print(
        "MLX_MAX_DEQUANT_ERROR_VS_CPU="
        f"{result['mlx_metal']['max_dequant_error_vs_cpu']:.12g}"
    )

    print(
        "MLX_ATTENTION_OUTPUT_FINITE=True"
    )

    print(
        "M4_METAL_FUNCTIONAL_PORTABILITY_PASS=True"
    )


if __name__ == "__main__":
    main()
