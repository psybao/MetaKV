
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

# ======================================================================================
# P2D3A diagnostic causal A/B
# Frozen production kernels are copied verbatim from canonical P2C source.
# No physical DRAM/HBM lane placement is claimed.
# ======================================================================================


@triton.jit
def swar_popcount_u32(x):

    x = (
        x
        -
        (
            (x >> 1)
            &
            0x55555555
        )
    )

    x = (
        (x & 0x33333333)
        +
        (
            (x >> 2)
            &
            0x33333333
        )
    )

    x = (
        x
        +
        (x >> 4)
    ) & 0x0F0F0F0F

    x = x + (x >> 8)
    x = x + (x >> 16)

    return x & 0x3F

@triton.jit
def kernel_identity(
    packed_ptr,
    metadata_ptr,
    out_ptr,

    n_values: tl.constexpr,
    group_size: tl.constexpr,

    BC_: tl.constexpr,
    ZMIN_: tl.constexpr,
    W_: tl.constexpr,
    FMAX_: tl.constexpr,
    COARSE_MASK_: tl.constexpr,

    BLOCK_: tl.constexpr
):

    offs = (
        tl.program_id(0)
        *
        BLOCK_
        +
        tl.arange(0, BLOCK_)
    )

    mask = offs < n_values

    packed = tl.load(
        packed_ptr + offs // 2,
        mask=mask,
        other=0
    ).to(tl.uint8)

    high = offs & 1

    nib = tl.where(
        high != 0,
        (packed >> 4) & 0xF,
        packed & 0xF
    ).to(tl.int32)

    q = tl.where(
        nib >= 8,
        nib - 16,
        nib
    ).to(tl.float32)

    group = offs // group_size

    word = tl.load(
        metadata_ptr + group,
        mask=mask,
        other=0
    ).to(tl.uint32)

    coarse = (
        word
        &
        COARSE_MASK_
    )

    count = swar_popcount_u32(
        coarse
    ).to(tl.float32)

    fine = (
        (
            word >> BC_
        )
        &
        FMAX_
    ).to(tl.float32)

    u = (
        count
        +
        fine / FMAX_
    )

    z = (
        ZMIN_
        +
        W_ / (BC_ + 1) * u
    )

    scale = tl.exp2(z)

    tl.store(
        out_ptr + offs,
        q * scale,
        mask=mask
    )

@triton.jit
def kernel_mapped(
    packed_ptr,
    metadata_ptr,
    out_ptr,

    n_values: tl.constexpr,
    group_size: tl.constexpr,

    BC_: tl.constexpr,
    BF_: tl.constexpr,

    ZMIN_: tl.constexpr,
    W_: tl.constexpr,
    FMAX_: tl.constexpr,

    COARSE_PHYS_MASK_: tl.constexpr,

    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    P4: tl.constexpr,
    P5: tl.constexpr,
    P6: tl.constexpr,
    P7: tl.constexpr,
    P8: tl.constexpr,
    P9: tl.constexpr,
    P10: tl.constexpr,
    P11: tl.constexpr,
    P12: tl.constexpr,
    P13: tl.constexpr,
    P14: tl.constexpr,
    P15: tl.constexpr,
    P16: tl.constexpr,
    P17: tl.constexpr,
    P18: tl.constexpr,

    BLOCK_: tl.constexpr
):

    offs = (
        tl.program_id(0)
        *
        BLOCK_
        +
        tl.arange(0, BLOCK_)
    )

    mask = offs < n_values

    packed = tl.load(
        packed_ptr + offs // 2,
        mask=mask,
        other=0
    ).to(tl.uint8)

    high = offs & 1

    nib = tl.where(
        high != 0,
        (packed >> 4) & 0xF,
        packed & 0xF
    ).to(tl.int32)

    q = tl.where(
        nib >= 8,
        nib - 16,
        nib
    ).to(tl.float32)

    group = offs // group_size

    word = tl.load(
        metadata_ptr + group,
        mask=mask,
        other=0
    ).to(tl.uint32)

    coarse = (
        word
        &
        COARSE_PHYS_MASK_
    )

    count = swar_popcount_u32(
        coarse
    ).to(tl.float32)

    fine = tl.zeros(
        (BLOCK_,),
        dtype=tl.uint32
    )

    if BF_ > 0:
        fine += (((word >> P0) & 1) << 0)
    if BF_ > 1:
        fine += (((word >> P1) & 1) << 1)
    if BF_ > 2:
        fine += (((word >> P2) & 1) << 2)
    if BF_ > 3:
        fine += (((word >> P3) & 1) << 3)
    if BF_ > 4:
        fine += (((word >> P4) & 1) << 4)
    if BF_ > 5:
        fine += (((word >> P5) & 1) << 5)
    if BF_ > 6:
        fine += (((word >> P6) & 1) << 6)
    if BF_ > 7:
        fine += (((word >> P7) & 1) << 7)
    if BF_ > 8:
        fine += (((word >> P8) & 1) << 8)
    if BF_ > 9:
        fine += (((word >> P9) & 1) << 9)
    if BF_ > 10:
        fine += (((word >> P10) & 1) << 10)
    if BF_ > 11:
        fine += (((word >> P11) & 1) << 11)
    if BF_ > 12:
        fine += (((word >> P12) & 1) << 12)
    if BF_ > 13:
        fine += (((word >> P13) & 1) << 13)
    if BF_ > 14:
        fine += (((word >> P14) & 1) << 14)
    if BF_ > 15:
        fine += (((word >> P15) & 1) << 15)
    if BF_ > 16:
        fine += (((word >> P16) & 1) << 16)
    if BF_ > 17:
        fine += (((word >> P17) & 1) << 17)
    if BF_ > 18:
        fine += (((word >> P18) & 1) << 18)

    u = (
        count
        +
        fine.to(tl.float32) / FMAX_
    )

    z = (
        ZMIN_
        +
        W_ / (BC_ + 1) * u
    )

    scale = tl.exp2(z)

    tl.store(
        out_ptr + offs,
        q * scale,
        mask=mask
    )


def logical_to_physical_numpy(logical_words, mapping):
    logical_words = np.asarray(
        logical_words,
        dtype=np.uint32
    )

    out = np.zeros_like(
        logical_words,
        dtype=np.uint32
    )

    for logical in range(32):
        physical = int(mapping[logical])

        bit = (
            logical_words
            >> np.uint32(logical)
        ) & np.uint32(1)

        out |= (
            bit
            << np.uint32(physical)
        )

    return out


def physical_to_logical_numpy(physical_words, mapping):
    physical_words = np.asarray(
        physical_words,
        dtype=np.uint32
    )

    out = np.zeros_like(
        physical_words,
        dtype=np.uint32
    )

    for logical in range(32):
        physical = int(mapping[logical])

        bit = (
            physical_words
            >> np.uint32(physical)
        ) & np.uint32(1)

        out |= (
            bit
            << np.uint32(logical)
        )

    return out


def bootstrap_median_ci(values, n_boot=10000, seed=20260908):
    x = np.asarray(values, dtype=np.float64)

    if x.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    boots = np.empty(
        n_boot,
        dtype=np.float64
    )

    for i in range(n_boot):
        sample = rng.choice(
            x,
            size=x.size,
            replace=True
        )
        boots[i] = np.median(sample)

    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5))
    )


def event_time_ms(fn, launches):
    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(launches):
        fn()

    end.record()
    end.synchronize()

    return (
        float(start.elapsed_time(end))
        / float(launches)
    )



# ======================================================================================
# Frozen REP_0025 experiment constants
# ======================================================================================

REP_ID = "REP_0025"

BC = 19
BF = 13

ZMIN = -5.75
ZMAX = -3.0
W = ZMAX - ZMIN

FMAX = (1 << BF) - 1

GROUP_SIZE = 32
GROUPS = 2 ** 22
VALUES = GROUPS * GROUP_SIZE

BLOCK = 256
NUM_WARPS = 4

WARMUP = 30
PAIRED_BLOCKS = 15
LAUNCHES_PER_TIMED_BLOCK = 100

SEED = 20260908

RESULT_DIR = Path('${WORKSPACE}/metakv_amd_attribution/p2d3_causal_ab/results')
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAP_FILE = Path('${WORKSPACE}/metakv_amd_attribution/inputs/m55m3g_cross_technology_robust_mappings.json')

mapping_json = json.loads(
    MAP_FILE.read_text(
        encoding="utf-8-sig"
    )
)

mapping = list(
    mapping_json[
        REP_ID
    ][
        "selected_mapping"
    ]
)

assert len(mapping) == 32
assert sorted(mapping) == list(range(32))

identity_mapping = list(range(32))

coarse_phys_mask_real = 0

for logical in range(BC):
    coarse_phys_mask_real |= (
        1 << mapping[logical]
    )

coarse_phys_mask_noop = (
    (1 << BC) - 1
)

fine_phys_real = [
    mapping[BC + j]
    if j < BF
    else 0
    for j in range(19)
]

fine_phys_noop = [
    BC + j
    if j < BF
    else 0
    for j in range(19)
]


print("=" * 120)
print("P2D3A REP_0025 CAUSAL A/B")
print("=" * 120)

print(
    f"Python={platform.python_version()}"
)
print(
    f"PyTorch={torch.__version__}"
)
print(
    f"Triton={triton.__version__}"
)
print(
    f"HIP={torch.version.hip}"
)
print(
    f"CUDA={torch.version.cuda}"
)

print(
    f"GPU_AVAILABLE={torch.cuda.is_available()}"
)

try:
    print(
        "GPU_NAME="
        + str(
            torch.cuda.get_device_name(0)
        )
    )
except Exception:
    print("GPU_NAME=")

print(
    "REP_MAPPING="
    + ",".join(
        map(str, mapping)
    )
)

print(
    f"GROUPS={GROUPS}"
)
print(
    f"VALUES={VALUES}"
)
print(
    f"GROUP_SIZE={GROUP_SIZE}"
)
print(
    f"PAIRED_BLOCKS={PAIRED_BLOCKS}"
)
print(
    f"LAUNCHES_PER_TIMED_BLOCK="
    f"{LAUNCHES_PER_TIMED_BLOCK}"
)


# ======================================================================================
# Generate valid true SBL32 metadata
#
# coarse field = thermometer code with 0..BC ones
# fine field   = BF-bit integer
# ======================================================================================

print()
print("[P2D3A] generating packed INT4 payload and valid SBL32 metadata")

rng = np.random.default_rng(SEED)

packed_np = rng.integers(
    0,
    256,
    size=(VALUES + 1) // 2,
    dtype=np.uint8
)

coarse_count = rng.integers(
    0,
    BC + 1,
    size=GROUPS,
    dtype=np.uint32
)

fine_np = rng.integers(
    0,
    FMAX + 1,
    size=GROUPS,
    dtype=np.uint32
)

logical_np = np.zeros(
    GROUPS,
    dtype=np.uint32
)

# Vectorized thermometer generation.
# BC=19, therefore shifts are safe in uint32.
nonzero = coarse_count > 0

logical_np[nonzero] = (
    (
        np.uint32(1)
        << coarse_count[nonzero]
    )
    - np.uint32(1)
)

logical_np |= (
    fine_np
    << np.uint32(BC)
)


# ======================================================================================
# REAL physical software bit layout
# ======================================================================================

t0 = time.perf_counter()

mapped_np = logical_to_physical_numpy(
    logical_np,
    mapping
)

real_mapping_prepare_sec = (
    time.perf_counter() - t0
)


# ======================================================================================
# PREPERMUTED:
# normalize mapped metadata back to logical layout OUTSIDE timed kernel
# ======================================================================================

t0 = time.perf_counter()

prelogical_np = physical_to_logical_numpy(
    mapped_np,
    mapping
)

prepermute_prepare_sec = (
    time.perf_counter() - t0
)

prepermute_roundtrip_exact = bool(
    np.array_equal(
        logical_np,
        prelogical_np
    )
)

print(
    "PREPERMUTE_ROUNDTRIP_EXACT="
    + str(prepermute_roundtrip_exact)
)

print(
    f"REAL_MAPPING_PREPARE_SEC="
    f"{real_mapping_prepare_sec:.6f}"
)

print(
    f"PREPERMUTE_NORMALIZATION_SEC="
    f"{prepermute_prepare_sec:.6f}"
)


# ======================================================================================
# Upload
# Use signed int32 storage container, kernel loads as uint32.
# ======================================================================================

packed_gpu = torch.from_numpy(
    packed_np
).cuda()

logical_gpu = torch.from_numpy(
    logical_np.view(np.int32)
).cuda()

mapped_gpu = torch.from_numpy(
    mapped_np.view(np.int32)
).cuda()

prelogical_gpu = torch.from_numpy(
    prelogical_np.view(np.int32)
).cuda()

out_ref = torch.empty(
    VALUES,
    device="cuda",
    dtype=torch.float32
)

out_test = torch.empty_like(
    out_ref
)

torch.cuda.synchronize()

grid = (
    triton.cdiv(
        VALUES,
        BLOCK
    ),
)


# ======================================================================================
# Launch wrappers
# ======================================================================================

def launch_identity(out):
    kernel_identity[
        grid
    ](
        packed_gpu,
        logical_gpu,
        out,

        n_values=VALUES,
        group_size=GROUP_SIZE,

        BC_=BC,
        ZMIN_=ZMIN,
        W_=W,
        FMAX_=FMAX,
        COARSE_MASK_=(
            (1 << BC) - 1
        ),

        BLOCK_=BLOCK,

        num_warps=NUM_WARPS
    )


def launch_real(out):
    kernel_mapped[
        grid
    ](
        packed_gpu,
        mapped_gpu,
        out,

        n_values=VALUES,
        group_size=GROUP_SIZE,

        BC_=BC,
        BF_=BF,

        ZMIN_=ZMIN,
        W_=W,
        FMAX_=FMAX,

        COARSE_PHYS_MASK_=
            coarse_phys_mask_real,

        P0=fine_phys_real[0],
        P1=fine_phys_real[1],
        P2=fine_phys_real[2],
        P3=fine_phys_real[3],
        P4=fine_phys_real[4],
        P5=fine_phys_real[5],
        P6=fine_phys_real[6],
        P7=fine_phys_real[7],
        P8=fine_phys_real[8],
        P9=fine_phys_real[9],
        P10=fine_phys_real[10],
        P11=fine_phys_real[11],
        P12=fine_phys_real[12],
        P13=fine_phys_real[13],
        P14=fine_phys_real[14],
        P15=fine_phys_real[15],
        P16=fine_phys_real[16],
        P17=fine_phys_real[17],
        P18=fine_phys_real[18],

        BLOCK_=BLOCK,

        num_warps=NUM_WARPS
    )


def launch_noop(out):
    # Same kernel_mapped structure, but identity logical->physical mapping.
    kernel_mapped[
        grid
    ](
        packed_gpu,
        logical_gpu,
        out,

        n_values=VALUES,
        group_size=GROUP_SIZE,

        BC_=BC,
        BF_=BF,

        ZMIN_=ZMIN,
        W_=W,
        FMAX_=FMAX,

        COARSE_PHYS_MASK_=
            coarse_phys_mask_noop,

        P0=fine_phys_noop[0],
        P1=fine_phys_noop[1],
        P2=fine_phys_noop[2],
        P3=fine_phys_noop[3],
        P4=fine_phys_noop[4],
        P5=fine_phys_noop[5],
        P6=fine_phys_noop[6],
        P7=fine_phys_noop[7],
        P8=fine_phys_noop[8],
        P9=fine_phys_noop[9],
        P10=fine_phys_noop[10],
        P11=fine_phys_noop[11],
        P12=fine_phys_noop[12],
        P13=fine_phys_noop[13],
        P14=fine_phys_noop[14],
        P15=fine_phys_noop[15],
        P16=fine_phys_noop[16],
        P17=fine_phys_noop[17],
        P18=fine_phys_noop[18],

        BLOCK_=BLOCK,

        num_warps=NUM_WARPS
    )


def launch_pre(out):
    # Runtime permutation removed.
    # Metadata has already been normalized back to logical layout.
    kernel_identity[
        grid
    ](
        packed_gpu,
        prelogical_gpu,
        out,

        n_values=VALUES,
        group_size=GROUP_SIZE,

        BC_=BC,
        ZMIN_=ZMIN,
        W_=W,
        FMAX_=FMAX,
        COARSE_MASK_=(
            (1 << BC) - 1
        ),

        BLOCK_=BLOCK,

        num_warps=NUM_WARPS
    )


# ======================================================================================
# Correctness gate
# ======================================================================================

print()
print("=" * 120)
print("CORRECTNESS GATE")
print("=" * 120)

launch_identity(
    out_ref
)
torch.cuda.synchronize()

correctness = {}

for name, fn in [
    ("REAL_MAPPED", launch_real),
    ("MAPPED_NOOP", launch_noop),
    ("PREPERMUTED", launch_pre),
]:
    fn(
        out_test
    )
    torch.cuda.synchronize()

    diff = (
        out_test
        - out_ref
    ).abs()

    max_abs = float(
        diff.max().item()
    )

    mean_abs = float(
        diff.mean().item()
    )

    finite = bool(
        torch.isfinite(
            out_test
        ).all().item()
    )

    passed = (
        finite
        and
        max_abs <= 1.0e-6
    )

    correctness[name] = {
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "pass": passed,
    }

    print(
        f"{name}_FINITE={finite} "
        f"MAX_ABS={max_abs:.12g} "
        f"MEAN_ABS={mean_abs:.12g} "
        f"PASS={passed}"
    )


correctness_all = (
    prepermute_roundtrip_exact
    and
    all(
        x["pass"]
        for x in correctness.values()
    )
)

print(
    "CORRECTNESS_ALL_PASS="
    + str(correctness_all)
)


# ======================================================================================
# Warmup
# ======================================================================================

print()
print("=" * 120)
print("WARMUP")
print("=" * 120)

for _ in range(WARMUP):
    launch_identity(
        out_test
    )
    launch_noop(
        out_test
    )
    launch_real(
        out_test
    )
    launch_pre(
        out_test
    )

torch.cuda.synchronize()

print(
    f"WARMUP_ROUNDS={WARMUP}"
)


# ======================================================================================
# Paired-block event timing
# Each variant appears twice per block.
# No profiler attached.
# ======================================================================================

orders = [
    [
        "ID",
        "NOOP",
        "REAL",
        "PRE",
        "PRE",
        "REAL",
        "NOOP",
        "ID",
    ],
    [
        "PRE",
        "REAL",
        "NOOP",
        "ID",
        "ID",
        "NOOP",
        "REAL",
        "PRE",
    ],
]

launchers = {
    "ID":
        lambda: launch_identity(
            out_test
        ),
    "NOOP":
        lambda: launch_noop(
            out_test
        ),
    "REAL":
        lambda: launch_real(
            out_test
        ),
    "PRE":
        lambda: launch_pre(
            out_test
        ),
}

raw_rows = []

id_blocks = []
noop_blocks = []
real_blocks = []
pre_blocks = []

noop_vs_id = []
real_vs_id = []
pre_vs_id = []
real_vs_pre = []

print()
print("=" * 120)
print("PAIRED BLOCK TIMING")
print("=" * 120)

for bi in range(
    1,
    PAIRED_BLOCKS + 1
):

    order = orders[
        (bi - 1) % len(orders)
    ]

    block_values = {
        "ID": [],
        "NOOP": [],
        "REAL": [],
        "PRE": [],
    }

    for variant in order:

        ms = event_time_ms(
            launchers[variant],
            LAUNCHES_PER_TIMED_BLOCK
        )

        block_values[
            variant
        ].append(
            ms
        )

    block_median = {
        k:
            float(
                statistics.median(v)
            )
        for k, v
        in block_values.items()
    }

    id_ms = block_median["ID"]
    noop_ms = block_median["NOOP"]
    real_ms = block_median["REAL"]
    pre_ms = block_median["PRE"]

    p_noop_id = (
        (noop_ms / id_ms) - 1.0
    ) * 100.0

    p_real_id = (
        (real_ms / id_ms) - 1.0
    ) * 100.0

    p_pre_id = (
        (pre_ms / id_ms) - 1.0
    ) * 100.0

    p_real_pre = (
        (real_ms / pre_ms) - 1.0
    ) * 100.0

    id_blocks.append(
        id_ms
    )
    noop_blocks.append(
        noop_ms
    )
    real_blocks.append(
        real_ms
    )
    pre_blocks.append(
        pre_ms
    )

    noop_vs_id.append(
        p_noop_id
    )
    real_vs_id.append(
        p_real_id
    )
    pre_vs_id.append(
        p_pre_id
    )
    real_vs_pre.append(
        p_real_pre
    )

    raw_rows.append({
        "block": bi,
        "identity_ms": id_ms,
        "noop_ms": noop_ms,
        "real_mapped_ms": real_ms,
        "prepermuted_ms": pre_ms,
        "noop_vs_identity_percent": p_noop_id,
        "real_vs_identity_percent": p_real_id,
        "pre_vs_identity_percent": p_pre_id,
        "real_vs_prepermuted_percent": p_real_pre,
    })

    print(
        f"[P2D3A] block={bi:02d}/"
        f"{PAIRED_BLOCKS} "
        f"ID={id_ms:.6f} "
        f"NOOP={noop_ms:.6f} "
        f"REAL={real_ms:.6f} "
        f"PRE={pre_ms:.6f} "
        f"NOOP-ID={p_noop_id:+.3f}% "
        f"REAL-ID={p_real_id:+.3f}% "
        f"PRE-ID={p_pre_id:+.3f}% "
        f"REAL-PRE={p_real_pre:+.3f}%",
        flush=True
    )


# ======================================================================================
# Summary
# ======================================================================================

def med(x):
    return float(
        statistics.median(x)
    )


noop_ci = bootstrap_median_ci(
    noop_vs_id
)

real_ci = bootstrap_median_ci(
    real_vs_id
)

pre_ci = bootstrap_median_ci(
    pre_vs_id
)

real_pre_ci = bootstrap_median_ci(
    real_vs_pre
)


summary = {
    "representation":
        REP_ID,

    "groups":
        GROUPS,

    "values":
        VALUES,

    "group_size":
        GROUP_SIZE,

    "paired_blocks":
        PAIRED_BLOCKS,

    "launches_per_timed_block":
        LAUNCHES_PER_TIMED_BLOCK,

    "identity_median_ms":
        med(id_blocks),

    "noop_median_ms":
        med(noop_blocks),

    "real_mapped_median_ms":
        med(real_blocks),

    "prepermuted_median_ms":
        med(pre_blocks),

    "noop_vs_identity_median_percent":
        med(noop_vs_id),

    "noop_vs_identity_ci95":
        list(noop_ci),

    "real_vs_identity_median_percent":
        med(real_vs_id),

    "real_vs_identity_ci95":
        list(real_ci),

    "pre_vs_identity_median_percent":
        med(pre_vs_id),

    "pre_vs_identity_ci95":
        list(pre_ci),

    "real_vs_prepermuted_median_percent":
        med(real_vs_pre),

    "real_vs_prepermuted_ci95":
        list(real_pre_ci),

    "prepermute_roundtrip_exact":
        prepermute_roundtrip_exact,

    "correctness_all_pass":
        correctness_all,

    "correctness":
        correctness,

    "real_mapping_prepare_sec":
        real_mapping_prepare_sec,

    "prepermute_normalization_sec":
        prepermute_prepare_sec,

    "limitations": [
        "Kernel microbenchmark causal diagnostic only.",
        "PREPERMUTED metadata normalization is outside the timed kernel.",
        "Software-emulated fixed bit layout only.",
        "No physical DRAM/HBM lane placement demonstrated.",
        "Negative latency deltas do not imply acceleration.",
    ],
}


raw_csv = (
    RESULT_DIR
    /
    "P2D3A_RAW_BLOCKS.csv"
)

with raw_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            raw_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        raw_rows
    )


summary_json = (
    RESULT_DIR
    /
    "P2D3A_SUMMARY.json"
)

summary_json.write_text(
    json.dumps(
        summary,
        indent=2
    )
    + "\n",
    encoding="utf-8"
)


print()
print("=" * 120)
print("P2D3A CAUSAL SUMMARY")
print("=" * 120)

print(
    f"IDENTITY_MEDIAN_MS="
    f"{summary['identity_median_ms']:.9f}"
)

print(
    f"NOOP_MEDIAN_MS="
    f"{summary['noop_median_ms']:.9f}"
)

print(
    f"REAL_MAPPED_MEDIAN_MS="
    f"{summary['real_mapped_median_ms']:.9f}"
)

print(
    f"PREPERMUTED_MEDIAN_MS="
    f"{summary['prepermuted_median_ms']:.9f}"
)

print()

print(
    f"NOOP_VS_IDENTITY_PERCENT="
    f"{summary['noop_vs_identity_median_percent']:+.6f} "
    f"CI=[{noop_ci[0]:+.6f},"
    f"{noop_ci[1]:+.6f}]"
)

print(
    f"REAL_VS_IDENTITY_PERCENT="
    f"{summary['real_vs_identity_median_percent']:+.6f} "
    f"CI=[{real_ci[0]:+.6f},"
    f"{real_ci[1]:+.6f}]"
)

print(
    f"PREPERMUTED_VS_IDENTITY_PERCENT="
    f"{summary['pre_vs_identity_median_percent']:+.6f} "
    f"CI=[{pre_ci[0]:+.6f},"
    f"{pre_ci[1]:+.6f}]"
)

print(
    f"REAL_VS_PREPERMUTED_PERCENT="
    f"{summary['real_vs_prepermuted_median_percent']:+.6f} "
    f"CI=[{real_pre_ci[0]:+.6f},"
    f"{real_pre_ci[1]:+.6f}]"
)


# ======================================================================================
# Conservative causal classification
# ======================================================================================

noop_small = (
    abs(
        summary[
            "noop_vs_identity_median_percent"
        ]
    )
    <= 5.0
)

pre_small = (
    abs(
        summary[
            "pre_vs_identity_median_percent"
        ]
    )
    <= 5.0
)

real_large = (
    summary[
        "real_vs_prepermuted_median_percent"
    ]
    >= 10.0
)

causal_supported = (
    correctness_all
    and
    pre_small
    and
    real_large
)

kernel_shape_penalty_supported = (
    correctness_all
    and
    not noop_small
)

print()

print(
    "NOOP_NEAR_IDENTITY="
    + str(noop_small)
)

print(
    "PREPERMUTED_NEAR_IDENTITY="
    + str(pre_small)
)

print(
    "REAL_RUNTIME_MAPPING_PENALTY_LARGE="
    + str(real_large)
)

print(
    "CAUSAL_RUNTIME_PERMUTATION_SUPPORTED="
    + str(causal_supported)
)

print(
    "KERNEL_SHAPE_PENALTY_SUPPORTED="
    + str(kernel_shape_penalty_supported)
)

print(
    "CORRECTNESS_ALL_PASS="
    + str(correctness_all)
)

print(
    "PREPERMUTE_ROUNDTRIP_EXACT="
    + str(prepermute_roundtrip_exact)
)

print(
    "P2D3A_EXECUTION_COMPLETE=True"
)
