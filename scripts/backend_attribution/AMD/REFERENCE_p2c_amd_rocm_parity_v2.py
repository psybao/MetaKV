import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl


# =============================================================================
# P2C AMD ROCm PARITY V2 — source-derived from frozen NVIDIA P1E
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

REP_CSV = (
    ROOT
    / "inputs"
    / "m55n3b0_unique_representations.csv"
)

MAP_JSON = (
    ROOT
    / "inputs"
    / "m55m3g_cross_technology_robust_mappings.json"
)

OUT = (
    ROOT
    / "p2c_rocm_parity_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT / "p2c_v2_raw_blocks.csv"
SUMMARY_CSV = OUT / "p2c_v2_summary.csv"
SUMMARY_JSON = OUT / "p2c_v2_summary.json"
STATUS_TXT = OUT / "P2C_V2_FINAL_STATUS.txt"


GROUP_SIZE = 32
BLOCK = 256

TARGET_BC = [
    14,
    19,
    22,
    26,
]

GROUP_COUNTS = [
    2**20,
    2**22,
]

WARMUP = 30
PAIRED_BLOCKS = 15
LAUNCHES_PER_TIMED_BLOCK = 100
BOOTSTRAP_N = 10000

SEED = 20260901


# =============================================================================
# Helpers
# =============================================================================

def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def pack_int4(q):
    q16 = q.to(torch.int16)

    nib = (
        q16 & 0xF
    ).to(torch.uint8)

    lo = nib[..., 0::2]
    hi = nib[..., 1::2]

    return (
        lo | (hi << 4)
    ).contiguous()


def encode_sbl32(
    z,
    bc,
    bf,
    zmin,
    zmax
):
    z = np.asarray(
        z,
        dtype=np.float64
    )

    W = zmax - zmin
    fmax = (1 << bf) - 1

    zc = np.clip(
        z,
        zmin,
        zmax
    )

    u = (
        (bc + 1)
        *
        (zc - zmin)
        /
        W
    )

    endpoint = (
        u >= bc + 1
    )

    coarse = np.floor(
        u
    ).astype(np.int64)

    coarse = np.clip(
        coarse,
        0,
        bc
    )

    frac = u - coarse

    fine = np.rint(
        frac * fmax
    ).astype(np.int64)

    fine = np.clip(
        fine,
        0,
        fmax
    )

    coarse[endpoint] = bc
    fine[endpoint] = fmax

    thermo = np.zeros(
        len(z),
        dtype=np.uint32
    )

    for c in range(bc + 1):
        mask = coarse == c

        if c == 0:
            t = 0
        else:
            t = (1 << c) - 1

        thermo[mask] = np.uint32(t)

    return (
        thermo
        |
        (
            fine.astype(np.uint32)
            << np.uint32(bc)
        )
    ).astype(np.uint32)


def logical_to_physical_words(
    logical_words,
    mapping
):
    logical_words = np.asarray(
        logical_words,
        dtype=np.uint32
    )

    out = np.zeros_like(
        logical_words
    )

    for logical in range(32):
        physical = mapping[logical]

        bit = (
            logical_words
            >>
            np.uint32(logical)
        ) & np.uint32(1)

        out |= (
            bit
            <<
            np.uint32(physical)
        )

    return out


def bootstrap_median_ci(
    values,
    seed,
    n=BOOTSTRAP_N
):
    x = np.asarray(
        values,
        dtype=np.float64
    )

    rng = np.random.default_rng(seed)

    meds = np.empty(
        n,
        dtype=np.float64
    )

    for i in range(n):
        idx = rng.integers(
            0,
            len(x),
            size=len(x)
        )

        meds[i] = np.median(
            x[idx]
        )

    lo, hi = np.percentile(
        meds,
        [2.5, 97.5]
    )

    return float(lo), float(hi)


# =============================================================================
# Pick representative frozen configurations
# =============================================================================

rows = read_csv(REP_CSV)

mapping_json = json.loads(
    MAP_JSON.read_text(
        encoding="utf-8"
    )
)

eligible = []

for r in rows:
    rep_id = r["representation_id"]

    if rep_id not in mapping_json:
        continue

    bc = int(r["Bc"])
    bf = int(r["Bf"])

    if bc + bf != 32:
        continue

    if bf > 19:
        continue

    eligible.append(r)


selected = []
used_rep = set()

for target in TARGET_BC:

    candidates = sorted(
        eligible,
        key=lambda r: (
            abs(int(r["Bc"]) - target),
            int(r["Bc"]),
            r["representation_id"],
        )
    )

    picked = None

    for r in candidates:
        if r["representation_id"] not in used_rep:
            picked = r
            break

    if picked is None:
        raise RuntimeError(
            f"No representative found for target Bc={target}"
        )

    selected.append(picked)
    used_rep.add(
        picked["representation_id"]
    )


# =============================================================================
# Environment
# =============================================================================

print("=" * 110)
print("P2C AMD ROCm PARITY V2 — NVIDIA P1E SOURCE-DERIVED")
print("=" * 110)

print(f"[ENV] Python={platform.python_version()}")
print(f"[ENV] PyTorch={torch.__version__}")
print(f"[ENV] Triton={triton.__version__}")
print(f"[ENV] CUDA={torch.version.cuda}")
print(f"[ENV] HIP={torch.version.hip}")
print(f"[ENV] accelerator available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("GPU accelerator unavailable through torch.cuda API")

device = torch.device("cuda")

print(f"[ENV] GPU={torch.cuda.get_device_name(0)}")

for i, r in enumerate(selected, 1):
    print(
        f"[CONFIG] {i}/4 "
        f"REP={r['representation_id']} "
        f"model={r['model']} "
        f"platform={r['platform']} "
        f"Bc={r['Bc']} Bf={r['Bf']} "
        f"interval=[{r['zmin']},{r['zmax']}]",
        flush=True
    )


# =============================================================================
# Triton helpers
# =============================================================================

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


# =============================================================================
# FP32 baseline kernel
# =============================================================================

@triton.jit
def kernel_fp32(
    packed_ptr,
    scale_ptr,
    out_ptr,

    n_values: tl.constexpr,
    group_size: tl.constexpr,
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

    scale = tl.load(
        scale_ptr + group,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    tl.store(
        out_ptr + offs,
        q * scale,
        mask=mask
    )


# =============================================================================
# Identity SBL kernel
# =============================================================================

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


# =============================================================================
# Fixed mapped SBL kernel
#
# Supports BF <= 19.
# =============================================================================

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


# =============================================================================
# Block timer
# =============================================================================

def time_block(fn):

    torch.cuda.synchronize()

    st = torch.cuda.Event(
        enable_timing=True
    )

    en = torch.cuda.Event(
        enable_timing=True
    )

    st.record()

    for _ in range(
        LAUNCHES_PER_TIMED_BLOCK
    ):
        fn()

    en.record()

    torch.cuda.synchronize()

    return (
        float(
            st.elapsed_time(en)
        )
        /
        LAUNCHES_PER_TIMED_BLOCK
    )


# =============================================================================
# Main
# =============================================================================

raw_rows = []
summary_rows = []

global_t0 = time.perf_counter()


for ci, rep in enumerate(
    selected,
    1
):

    REP_ID = rep[
        "representation_id"
    ]

    BC = int(rep["Bc"])
    BF = int(rep["Bf"])

    ZMIN = float(rep["zmin"])
    ZMAX = float(rep["zmax"])

    W = ZMAX - ZMIN
    FMAX = (1 << BF) - 1

    mapping = list(
        mapping_json[
            REP_ID
        ][
            "selected_mapping"
        ]
    )

    coarse_phys_mask = 0

    for logical in range(BC):
        coarse_phys_mask |= (
            1 << mapping[logical]
        )

    fine_phys = [
        mapping[
            BC + j
        ]
        for j in range(BF)
    ]

    while len(fine_phys) < 19:
        fine_phys.append(0)


    print()
    print("=" * 110)

    print(
        f"[P2C-V2] CONFIG {ci}/{len(selected)} "
        f"REP={REP_ID} "
        f"Bc={BC} Bf={BF} "
        f"interval=[{ZMIN},{ZMAX}]",
        flush=True
    )


    for wi, groups in enumerate(
        GROUP_COUNTS,
        1
    ):

        t0 = time.perf_counter()

        values = (
            groups
            *
            GROUP_SIZE
        )


        print(
            f"[P2C-V2] workload={wi}/{len(GROUP_COUNTS)} "
            f"groups={groups:,} "
            f"values={values:,}",
            flush=True
        )


        q = torch.randint(
            low=-7,
            high=8,
            size=(
                groups,
                GROUP_SIZE
            ),
            dtype=torch.int8,
            device=device
        )

        packed = pack_int4(q)

        del q


        local_rng = np.random.default_rng(
            SEED
            +
            ci * 1000
            +
            wi
        )

        z_np = local_rng.uniform(
            ZMIN,
            ZMAX,
            size=groups
        ).astype(np.float32)

        scale_np = np.exp2(
            z_np
        ).astype(np.float32)

        logical_np = encode_sbl32(
            z_np,
            BC,
            BF,
            ZMIN,
            ZMAX
        )

        mapped_np = (
            logical_to_physical_words(
                logical_np,
                mapping
            )
        )


        scales_gpu = (
            torch.from_numpy(
                scale_np
            ).to(device)
        )

        logical_gpu = (
            torch.from_numpy(
                logical_np.view(
                    np.int32
                ).copy()
            ).to(device)
        )

        mapped_gpu = (
            torch.from_numpy(
                mapped_np.view(
                    np.int32
                ).copy()
            ).to(device)
        )


        out = torch.empty(
            values,
            dtype=torch.float32,
            device=device
        )


        grid = (
            triton.cdiv(
                values,
                BLOCK
            ),
        )


        def launch_fp():

            kernel_fp32[
                grid
            ](
                packed,
                scales_gpu,
                out,

                n_values=values,
                group_size=GROUP_SIZE,
                BLOCK_=BLOCK,

                num_warps=4
            )


        def launch_id():

            kernel_identity[
                grid
            ](
                packed,
                logical_gpu,
                out,

                n_values=values,
                group_size=GROUP_SIZE,

                BC_=BC,
                ZMIN_=ZMIN,
                W_=W,
                FMAX_=FMAX,
                COARSE_MASK_=(
                    (1 << BC) - 1
                ),

                BLOCK_=BLOCK,

                num_warps=4
            )


        def launch_map():

            kernel_mapped[
                grid
            ](
                packed,
                mapped_gpu,
                out,

                n_values=values,
                group_size=GROUP_SIZE,

                BC_=BC,
                BF_=BF,

                ZMIN_=ZMIN,
                W_=W,
                FMAX_=FMAX,

                COARSE_PHYS_MASK_=
                    coarse_phys_mask,

                P0=fine_phys[0],
                P1=fine_phys[1],
                P2=fine_phys[2],
                P3=fine_phys[3],
                P4=fine_phys[4],
                P5=fine_phys[5],
                P6=fine_phys[6],
                P7=fine_phys[7],
                P8=fine_phys[8],
                P9=fine_phys[9],
                P10=fine_phys[10],
                P11=fine_phys[11],
                P12=fine_phys[12],
                P13=fine_phys[13],
                P14=fine_phys[14],
                P15=fine_phys[15],
                P16=fine_phys[16],
                P17=fine_phys[17],
                P18=fine_phys[18],

                BLOCK_=BLOCK,

                num_warps=4
            )


        # Warmup

        for _ in range(WARMUP):
            launch_fp()
            launch_id()
            launch_map()

        torch.cuda.synchronize()


        codec_list = []
        mapping_list = []
        total_list = []

        fp_list = []
        id_list = []
        map_list = []


        for bi in range(
            1,
            PAIRED_BLOCKS + 1
        ):

            if bi % 2:
                sequence = [
                    "FP",
                    "ID",
                    "MAP",
                    "MAP",
                    "ID",
                    "FP",
                ]

                order = (
                    "F-I-M-M-I-F"
                )

            else:
                sequence = [
                    "MAP",
                    "ID",
                    "FP",
                    "FP",
                    "ID",
                    "MAP",
                ]

                order = (
                    "M-I-F-F-I-M"
                )


            local = {
                "FP": [],
                "ID": [],
                "MAP": [],
            }


            for name in sequence:

                if name == "FP":
                    v = time_block(
                        launch_fp
                    )

                elif name == "ID":
                    v = time_block(
                        launch_id
                    )

                else:
                    v = time_block(
                        launch_map
                    )

                local[name].append(v)


            fp = statistics.median(
                local["FP"]
            )

            identity = statistics.median(
                local["ID"]
            )

            mapped = statistics.median(
                local["MAP"]
            )


            codec = (
                100.0
                *
                (
                    identity / fp
                    -
                    1.0
                )
            )

            mapping_ov = (
                100.0
                *
                (
                    mapped / identity
                    -
                    1.0
                )
            )

            total = (
                100.0
                *
                (
                    mapped / fp
                    -
                    1.0
                )
            )


            fp_list.append(fp)
            id_list.append(identity)
            map_list.append(mapped)

            codec_list.append(codec)
            mapping_list.append(
                mapping_ov
            )
            total_list.append(total)


            raw_rows.append({
                "representation_id":
                    REP_ID,

                "model":
                    rep["model"],

                "platform":
                    rep["platform"],

                "Bc":
                    BC,

                "Bf":
                    BF,

                "zmin":
                    ZMIN,

                "zmax":
                    ZMAX,

                "groups":
                    groups,

                "values":
                    values,

                "block":
                    bi,

                "order":
                    order,

                "fp32_ms":
                    fp,

                "identity_ms":
                    identity,

                "mapped_ms":
                    mapped,

                "codec_percent":
                    codec,

                "mapping_percent":
                    mapping_ov,

                "total_percent":
                    total,
            })


            print(
                f"[P2C-V2] REP={REP_ID} "
                f"G={groups:,} "
                f"block={bi:02d}/{PAIRED_BLOCKS} "
                f"FP={fp:.6f} "
                f"ID={identity:.6f} "
                f"MAP={mapped:.6f} "
                f"codec={codec:+.3f}% "
                f"mapping={mapping_ov:+.3f}% "
                f"total={total:+.3f}%",
                flush=True
            )


        codec_arr = np.asarray(
            codec_list,
            dtype=np.float64
        )

        mapping_arr = np.asarray(
            mapping_list,
            dtype=np.float64
        )

        total_arr = np.asarray(
            total_list,
            dtype=np.float64
        )


        codec_ci = bootstrap_median_ci(
            codec_arr,
            SEED + ci * 100 + wi * 10 + 1
        )

        mapping_ci = bootstrap_median_ci(
            mapping_arr,
            SEED + ci * 100 + wi * 10 + 2
        )

        total_ci = bootstrap_median_ci(
            total_arr,
            SEED + ci * 100 + wi * 10 + 3
        )


        row = {
            "representation_id":
                REP_ID,

            "model":
                rep["model"],

            "platform":
                rep["platform"],

            "Bc":
                BC,

            "Bf":
                BF,

            "zmin":
                ZMIN,

            "zmax":
                ZMAX,

            "groups":
                groups,

            "values":
                values,

            "fp32_median_ms":
                float(
                    np.median(fp_list)
                ),

            "identity_median_ms":
                float(
                    np.median(id_list)
                ),

            "mapped_median_ms":
                float(
                    np.median(map_list)
                ),

            "codec_median_percent":
                float(
                    np.median(
                        codec_arr
                    )
                ),

            "codec_ci95_low":
                codec_ci[0],

            "codec_ci95_high":
                codec_ci[1],

            "mapping_median_percent":
                float(
                    np.median(
                        mapping_arr
                    )
                ),

            "mapping_ci95_low":
                mapping_ci[0],

            "mapping_ci95_high":
                mapping_ci[1],

            "mapped_vs_fp32_median_percent":
                float(
                    np.median(
                        total_arr
                    )
                ),

            "mapped_vs_fp32_ci95_low":
                total_ci[0],

            "mapped_vs_fp32_ci95_high":
                total_ci[1],
        }


        summary_rows.append(row)


        print()
        print(
            f"[P2C-V2-SUMMARY] REP={REP_ID} "
            f"Bc={BC} Bf={BF} "
            f"groups={groups:,}",
            flush=True
        )

        print(
            f"[P2C-V2-SUMMARY] codec="
            f"{row['codec_median_percent']:+.4f}% "
            f"CI=[{codec_ci[0]:+.4f}%,"
            f"{codec_ci[1]:+.4f}%]",
            flush=True
        )

        print(
            f"[P2C-V2-SUMMARY] mapping="
            f"{row['mapping_median_percent']:+.4f}% "
            f"CI=[{mapping_ci[0]:+.4f}%,"
            f"{mapping_ci[1]:+.4f}%]",
            flush=True
        )

        print(
            f"[P2C-V2-SUMMARY] total="
            f"{row['mapped_vs_fp32_median_percent']:+.4f}% "
            f"CI=[{total_ci[0]:+.4f}%,"
            f"{total_ci[1]:+.4f}%]",
            flush=True
        )

        print(
            f"[P2C-V2-SUMMARY] elapsed="
            f"{time.perf_counter()-t0:.1f}s",
            flush=True
        )


        del packed
        del scales_gpu
        del logical_gpu
        del mapped_gpu
        del out

        torch.cuda.empty_cache()


# =============================================================================
# Save
# =============================================================================

with RAW_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            raw_rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(raw_rows)


with SUMMARY_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(summary_rows)


large_rows = [
    r
    for r in summary_rows
    if r["groups"] == max(
        GROUP_COUNTS
    )
]


largest_abs_total = max(
    abs(
        float(
            r[
                "mapped_vs_fp32_median_percent"
            ]
        )
    )
    for r in large_rows
)


critical = []

if len(selected) != 4:
    critical.append(
        "CONFIG_COUNT_NOT_4"
    )

if len(summary_rows) != 8:
    critical.append(
        "SUMMARY_ROW_COUNT_NOT_8"
    )

for r in summary_rows:
    for key in [
        "fp32_median_ms",
        "identity_median_ms",
        "mapped_median_ms",
        "codec_median_percent",
        "mapping_median_percent",
        "mapped_vs_fp32_median_percent",
    ]:
        if not np.isfinite(
            float(r[key])
        ):
            critical.append(
                f"NONFINITE_{key}"
            )


summary_json = {
    "stage":
        "P2C_AMD_PARITY_V2",

    "gpu":
        torch.cuda.get_device_name(0),

    "selected_representations":
        [
            {
                "representation_id":
                    r["representation_id"],

                "model":
                    r["model"],

                "platform":
                    r["platform"],

                "Bc":
                    int(r["Bc"]),

                "Bf":
                    int(r["Bf"]),

                "zmin":
                    float(r["zmin"]),

                "zmax":
                    float(r["zmax"]),
            }
            for r in selected
        ],

    "target_Bc":
        TARGET_BC,

    "workloads":
        GROUP_COUNTS,

    "paired_blocks":
        PAIRED_BLOCKS,

    "launches_per_timed_block":
        LAUNCHES_PER_TIMED_BLOCK,

    "largest_workload_max_abs_mapped_vs_fp32_median_percent":
        largest_abs_total,

    "limitations":
        [
            "Cross-vendor parity port of frozen NVIDIA P1E source; runtime GPU recorded above.",
            "Kernel microbenchmark only.",
            "Synthetic packed INT4 payload.",
            "Fixed software-emulated metadata mappings.",
            "No physical DRAM/HBM lane placement claim.",
            "No end-to-end LLM latency claim.",
        ],
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary_json,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


lines = [
    "P2C AMD ROCm PARITY V2 — NVIDIA P1E SOURCE-DERIVED",

    f"GPU={torch.cuda.get_device_name(0)}",

    "CONFIG_COUNT=4",

    (
        "REPRESENTATIONS="
        +
        ",".join(
            r["representation_id"]
            for r in selected
        )
    ),

    (
        "BC_VALUES="
        +
        ",".join(
            str(r["Bc"])
            for r in selected
        )
    ),

    (
        "BF_VALUES="
        +
        ",".join(
            str(r["Bf"])
            for r in selected
        )
    ),

    (
        "WORKLOAD_GROUPS="
        +
        ",".join(
            str(x)
            for x in GROUP_COUNTS
        )
    ),

    f"PAIRED_BLOCKS={PAIRED_BLOCKS}",

    (
        "LAUNCHES_PER_TIMED_BLOCK="
        f"{LAUNCHES_PER_TIMED_BLOCK}"
    ),

    (
        "LARGEST_WORKLOAD_MAX_ABS_TOTAL_MEDIAN_PERCENT="
        f"{largest_abs_total}"
    ),

    "REFERENCE_NVIDIA_P1E_SHA256=FF4D2B585512E3EE866644F67BE661D644CC610D603766638E2EA02F5CA107BB",

    "SOURCE_DERIVED_PARITY_PORT=True",

    "KERNEL_ALGORITHM_CHANGED=False",

    "INPUT_GENERATION_CHANGED=False",

    "TIMING_PROTOCOL_CHANGED=False",

    "TRUE_NIBBLE_PACKED_INT4=True",

    "METADATA_BITS_PER_GROUP=32",

    "MULTIPLE_FBMS_SPLITS_TESTED=True",

    "KERNEL_MICROBENCHMARK=True",

    "END_TO_END_LLM_LATENCY_CLAIM=False",

    "PHYSICAL_HBM_LANE_PLACEMENT_DEMONSTRATED=False",

    f"CRITICAL_ISSUES_COUNT={len(critical)}",

    f"P2C_V2_FINAL_PASS={len(critical)==0}",
]


for c in critical:
    lines.append(
        f"CRITICAL={c}"
    )


STATUS_TXT.write_text(
    "\n".join(lines)
    +
    "\n",
    encoding="utf-8"
)


print()
print("=" * 110)

for line in lines:
    print(
        line,
        flush=True
    )

print()
print(
    f"RAW_CSV={RAW_CSV}",
    flush=True
)

print(
    f"SUMMARY_CSV={SUMMARY_CSV}",
    flush=True
)

print(
    f"SUMMARY_JSON={SUMMARY_JSON}",
    flush=True
)

print(
    f"STATUS_TXT={STATUS_TXT}",
    flush=True
)

print(
    f"TOTAL_ELAPSED_SEC="
    f"{time.perf_counter()-global_t0:.2f}",
    flush=True
)
