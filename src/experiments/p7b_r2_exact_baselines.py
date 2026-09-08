from pathlib import Path
import csv
import json
import math
import sys

import numpy as np


SCALE14=Path(sys.argv[1])
SCALE32=Path(sys.argv[2])
P4B0=Path(sys.argv[3])
RUN=Path(sys.argv[4])

OUT=RUN/"output"
FREEZE=RUN/"freeze"

OUT.mkdir(parents=True,exist_ok=True)
FREEZE.mkdir(parents=True,exist_ok=True)


BC=19
BF=13
BITS=32

FMAX=(1<<BF)-1
COARSE_MASK=(1<<BC)-1

BINARY_MAX=(1<<32)-1

assert BC+BF == 32


# ======================================================================================
# DATA
# ======================================================================================

def load_z(path):

    if not path.exists():
        raise RuntimeError(
            f"missing scale corpus: {path}"
        )

    z=np.load(
        path
    ).astype(
        np.float64
    ).reshape(-1)

    if not np.isfinite(z).all():
        raise RuntimeError(
            f"nonfinite z corpus: {path}"
        )

    return z


z14=load_z(SCALE14)
z32=load_z(SCALE32)


# ======================================================================================
# EXACT BINARY32-Z FAMILY
#
# Frozen P3B0/P4B0 semantics:
#
#   code = round((z-zmin)/(zmax-zmin) * (2^32-1))
#   decode = zmin + code/(2^32-1)*(zmax-zmin)
#
# This is conventional same-width 32-bit uniform binary coding of z.
# It is NOT IEEE-754 encoding of z.
# ======================================================================================

def binary32_encode(
    z,
    zmin,
    zmax
):
    z=np.asarray(
        z,
        dtype=np.float64
    )

    norm=(
        (z-zmin)
        /
        float(zmax-zmin)
    )

    norm=np.clip(
        norm,
        0.0,
        1.0
    )

    # np.rint matches historical round-to-nearest construction.
    code=np.rint(
        norm
        *
        float(BINARY_MAX)
    ).astype(
        np.uint64
    )

    return (
        code
        &
        np.uint64(
            BINARY_MAX
        )
    ).astype(
        np.uint32
    )


def binary32_decode(
    words,
    zmin,
    zmax
):
    words=np.asarray(
        words,
        dtype=np.uint32
    )

    return (
        float(zmin)
        +
        (
            words.astype(
                np.float64
            )
            /
            float(
                BINARY_MAX
            )
        )
        *
        float(
            zmax-zmin
        )
    )


# ======================================================================================
# HA-FBMS
# ======================================================================================

def ha_encode(
    z,
    zmin,
    zmax
):
    z=np.asarray(
        z,
        dtype=np.float64
    )

    u=(
        (z-zmin)
        *
        (BC+1)
        /
        float(zmax-zmin)
    )

    u=np.clip(
        u,
        0.0,
        float(BC+1)
    )

    coarse=np.floor(
        u
    ).astype(
        np.int64
    )

    endpoint=(
        coarse >= BC+1
    )

    fine=np.rint(
        (u-coarse)
        *
        FMAX
    ).astype(
        np.int64
    )

    coarse[endpoint]=BC
    fine[endpoint]=FMAX

    coarse=np.clip(
        coarse,
        0,
        BC
    )

    fine=np.clip(
        fine,
        0,
        FMAX
    )

    words=np.zeros(
        z.shape,
        dtype=np.uint32
    )

    for c in range(
        BC+1
    ):
        therm=(
            0
            if c==0
            else
            (1<<c)-1
        )

        words[
            coarse==c
        ]=np.uint32(
            therm
        )

    words |= (
        fine.astype(
            np.uint32
        )
        <<
        np.uint32(BC)
    )

    return words


def ha_decode(
    words,
    zmin,
    zmax
):
    words=np.asarray(
        words,
        dtype=np.uint32
    )

    coarse=(
        words
        &
        np.uint32(
            COARSE_MASK
        )
    )

    flat=coarse.reshape(-1)

    count=np.fromiter(
        (
            int(x).bit_count()
            for x in flat
        ),
        dtype=np.int16,
        count=flat.size
    ).reshape(
        coarse.shape
    )

    fine=(
        words
        >>
        np.uint32(BC)
    ) & np.uint32(
        FMAX
    )

    u=(
        count.astype(
            np.float64
        )
        +
        fine.astype(
            np.float64
        )
        /
        float(FMAX)
    )

    return (
        float(zmin)
        +
        (
            float(zmax-zmin)
            /
            float(BC+1)
        )
        *
        u
    )


# ======================================================================================
# FP32 DIRECT SCALE
# ======================================================================================

def fp32_words_from_scale(
    scale
):
    return np.asarray(
        scale,
        dtype=np.float32
    ).view(
        np.uint32
    )


def fp32_scale_from_words(
    words
):
    return np.asarray(
        words,
        dtype=np.uint32
    ).view(
        np.float32
    ).astype(
        np.float64
    )


# ======================================================================================
# CLASSIFICATION
# ======================================================================================

def classify_scale(
    x
):
    x=np.asarray(
        x,
        dtype=np.float64
    )

    finite=np.isfinite(x)

    return {
        "nan":
            int(
                np.count_nonzero(
                    np.isnan(x)
                )
            ),

        "posinf":
            int(
                np.count_nonzero(
                    np.isposinf(x)
                )
            ),

        "neginf":
            int(
                np.count_nonzero(
                    np.isneginf(x)
                )
            ),

        "zero":
            int(
                np.count_nonzero(
                    finite
                    &
                    (x==0)
                )
            ),

        "negative_finite":
            int(
                np.count_nonzero(
                    finite
                    &
                    (x<0)
                )
            ),

        "positive_finite":
            int(
                np.count_nonzero(
                    finite
                    &
                    (x>0)
                )
            ),
    }


def amp_stats(
    clean_scale,
    fault_scale
):
    clean_scale=np.asarray(
        clean_scale,
        dtype=np.float64
    )

    fault_scale=np.asarray(
        fault_scale,
        dtype=np.float64
    )

    valid=(
        np.isfinite(
            fault_scale
        )
        &
        (fault_scale>0)
    )

    invalid=int(
        np.count_nonzero(
            ~valid
        )
    )

    if not np.any(valid):
        return {
            "valid":0,
            "invalid":invalid,
            "max_a":None,
            "ge2":0,
            "ge10":0,
            "ge100":0,
            "ge1000":0,
            "ge10000":0,
            "ge1e6":0,
        }

    ratio=(
        fault_scale[valid]
        /
        clean_scale[valid]
    )

    amp=np.maximum(
        ratio,
        1.0/ratio
    )

    return {
        "valid":
            int(
                np.count_nonzero(
                    valid
                )
            ),

        "invalid":
            invalid,

        "max_a":
            float(
                np.max(
                    amp
                )
            ),

        "ge2":
            int(
                np.count_nonzero(
                    amp>=2.0
                )
            ),

        "ge10":
            int(
                np.count_nonzero(
                    amp>=10.0
                )
            ),

        "ge100":
            int(
                np.count_nonzero(
                    amp>=100.0
                )
            ),

        "ge1000":
            int(
                np.count_nonzero(
                    amp>=1000.0
                )
            ),

        "ge10000":
            int(
                np.count_nonzero(
                    amp>=10000.0
                )
            ),

        "ge1e6":
            int(
                np.count_nonzero(
                    amp>=1e6
                )
            ),
    }


# ======================================================================================
# GENERIC SCREEN
# ======================================================================================

def screen(
    z,
    zmin,
    zmax,
    label
):

    clean_scale=np.exp2(
        z
    )

    fp_words=fp32_words_from_scale(
        clean_scale
    )

    bz_words=binary32_encode(
        z,
        zmin,
        zmax
    )

    ha_words=ha_encode(
        z,
        zmin,
        zmax
    )


    bz_clean_z=binary32_decode(
        bz_words,
        zmin,
        zmax
    )

    bz_clean_max_z_error=float(
        np.max(
            np.abs(
                bz_clean_z-z
            )
        )
    )


    accum={}

    for enc in [
        "FP32",
        "BINARY32_Z",
        "HA_FBMS",
    ]:
        accum[enc]={
            "events":0,
            "valid":0,
            "invalid":0,
            "max_a":1.0,
            "ge2":0,
            "ge10":0,
            "ge100":0,
            "ge1000":0,
            "ge10000":0,
            "ge1e6":0,
            "nan":0,
            "posinf":0,
            "neginf":0,
            "zero":0,
            "negative_finite":0,
            "positive_finite":0,
        }


    bit_rows=[]


    for bit in range(32):

        mask=np.uint32(
            1<<bit
        )


        # --------------------------------------------------------------
        # FP32 direct scale
        # --------------------------------------------------------------

        fw=(
            fp_words
            ^
            mask
        )

        fs=fp32_scale_from_words(
            fw
        )

        st=amp_stats(
            clean_scale,
            fs
        )

        cl=classify_scale(
            fs
        )

        bit_rows.append({
            "dataset":label,
            "encoding":"FP32",
            "bit":bit,
            "events":len(z),
            **st,
            **cl,
        })


        # --------------------------------------------------------------
        # Binary32-Z exact frozen family
        # --------------------------------------------------------------

        bw=(
            bz_words
            ^
            mask
        )

        fault_z=binary32_decode(
            bw,
            zmin,
            zmax
        )

        bs=np.exp2(
            fault_z
        )

        st=amp_stats(
            clean_scale,
            bs
        )

        cl=classify_scale(
            bs
        )

        bit_rows.append({
            "dataset":label,
            "encoding":"BINARY32_Z",
            "bit":bit,
            "events":len(z),
            **st,
            **cl,
        })


        # --------------------------------------------------------------
        # HA
        # --------------------------------------------------------------

        hw=(
            ha_words
            ^
            mask
        )

        hz=ha_decode(
            hw,
            zmin,
            zmax
        )

        hs=np.exp2(
            hz
        )

        st=amp_stats(
            clean_scale,
            hs
        )

        cl=classify_scale(
            hs
        )

        bit_rows.append({
            "dataset":label,
            "encoding":"HA_FBMS",
            "bit":bit,
            "events":len(z),
            **st,
            **cl,
        })


    for row in bit_rows:

        a=accum[
            row[
                "encoding"
            ]
        ]

        a["events"] += row["events"]
        a["valid"] += row["valid"]
        a["invalid"] += row["invalid"]

        if row["max_a"] is not None:
            a["max_a"]=max(
                a["max_a"],
                row["max_a"]
            )

        for k in [
            "ge2",
            "ge10",
            "ge100",
            "ge1000",
            "ge10000",
            "ge1e6",
            "nan",
            "posinf",
            "neginf",
            "zero",
            "negative_finite",
            "positive_finite",
        ]:
            a[k] += row[k]


    return {
        "dataset":
            label,

        "scale_count":
            int(
                len(z)
            ),

        "zmin":
            zmin,

        "zmax":
            zmax,

        "width":
            zmax-zmin,

        "binary_clean_max_z_error":
            bz_clean_max_z_error,

        "global":
            accum,

        "bit_rows":
            bit_rows,
    }


# ======================================================================================
# HISTORICAL QWEN14 REPRODUCTION
# ======================================================================================

r14=screen(
    z14,
    -11.0,
    5.0,
    "QWEN3_14B_REPRODUCTION"
)


b14=r14["global"][
    "BINARY32_Z"
]


print("="*150)
print("P7B-R2 HISTORICAL QWEN3-14B REPRODUCTION")
print("="*150)

print(
    "P4_REPRO_SCALE_COUNT=",
    r14["scale_count"],
    sep=""
)

print(
    "P4_REPRO_BINARY_CLEAN_MAX_Z_ERROR=",
    r14[
        "binary_clean_max_z_error"
    ],
    sep=""
)

print(
    "P4_REPRO_BINARY_INVALID=",
    b14["invalid"],
    sep=""
)

print(
    "P4_REPRO_BINARY_MAX_A=",
    b14["max_a"],
    sep=""
)

print(
    "P4_REPRO_BINARY_GE2=",
    b14["ge2"],
    sep=""
)

print(
    "P4_REPRO_BINARY_GE10=",
    b14["ge10"],
    sep=""
)

print(
    "P4_REPRO_BINARY_GE100=",
    b14["ge100"],
    sep=""
)


# Frozen P4B0 headline invariants.
#
# From authoritative historical run:
# scale count = 317440
# clean max log2 error ≈ 1.862645149230957e-09
# invalid = 0
# max A ≈ 256.000000661
# GE100 = 317440
#
# GE2/GE10 counts are checked from the full output if parsable below.
repro_scale_count_pass=(
    r14["scale_count"]
    ==
    317440
)

repro_clean_error_pass=(
    abs(
        r14["binary_clean_max_z_error"]
        -
        1.862645149230957e-09
    )
    <=
    5e-12
)

repro_invalid_pass=(
    b14["invalid"]==0
)

repro_max_a_pass=(
    abs(
        b14["max_a"]
        -
        256.000000661
    )
    <
    1e-3
)

repro_ge100_pass=(
    b14["ge100"]
    ==
    317440
)


historical_reproduction_pass=all([
    repro_scale_count_pass,
    repro_clean_error_pass,
    repro_invalid_pass,
    repro_max_a_pass,
    repro_ge100_pass,
])


print(
    "P4_REPRO_SCALE_COUNT_PASS=",
    repro_scale_count_pass,
    sep=""
)

print(
    "P4_REPRO_CLEAN_ERROR_PASS=",
    repro_clean_error_pass,
    sep=""
)

print(
    "P4_REPRO_INVALID_PASS=",
    repro_invalid_pass,
    sep=""
)

print(
    "P4_REPRO_MAX_A_PASS=",
    repro_max_a_pass,
    sep=""
)

print(
    "P4_REPRO_GE100_PASS=",
    repro_ge100_pass,
    sep=""
)

print(
    "P4_BINARY32_Z_EXACT_REPRODUCTION_PASS=",
    historical_reproduction_pass,
    sep=""
)


if not historical_reproduction_pass:

    raise RuntimeError(
        "Binary32-Z reconstruction failed frozen P4B0 reproduction gate; "
        "32B comparative result must not be promoted."
    )


# ======================================================================================
# QWEN3-32B AUTHORITATIVE SCREEN
# ======================================================================================

r32=screen(
    z32,
    -13.0,
    5.0,
    "QWEN3_32B"
)


print()
print("="*150)
print("P7B-R2 QWEN3-32B AUTHORITATIVE GLOBAL")
print("="*150)


summary_rows=[]


for enc in [
    "FP32",
    "BINARY32_Z",
    "HA_FBMS",
]:

    a=r32[
        "global"
    ][enc]

    row={
        "encoding":
            enc,

        **a,
    }

    summary_rows.append(
        row
    )


    print(
        f"P7B_R2_GLOBAL "
        f"ENC={enc} "
        f"EVENTS={a['events']} "
        f"VALID={a['valid']} "
        f"INVALID={a['invalid']} "
        f"NAN={a['nan']} "
        f"POSINF={a['posinf']} "
        f"NEGINF={a['neginf']} "
        f"ZERO={a['zero']} "
        f"NEGATIVE_FINITE={a['negative_finite']} "
        f"POSITIVE_FINITE={a['positive_finite']} "
        f"MAX_A={a['max_a']} "
        f"GE2={a['ge2']} "
        f"GE10={a['ge10']} "
        f"GE100={a['ge100']} "
        f"GE1000={a['ge1000']} "
        f"GE10000={a['ge10000']} "
        f"GE1E6={a['ge1e6']}"
    )


print(
    "P7B_R2_BINARY_CLEAN_MAX_Z_ERROR=",
    r32[
        "binary_clean_max_z_error"
    ],
    sep=""
)


# ======================================================================================
# FINAL GATES
# ======================================================================================

fp=r32["global"]["FP32"]
bz=r32["global"]["BINARY32_Z"]
ha=r32["global"]["HA_FBMS"]


event_count_expected=(
    507904*32
)


event_count_pass=all(
    r32["global"][e]["events"]
    ==
    event_count_expected
    for e in [
        "FP32",
        "BINARY32_Z",
        "HA_FBMS",
    ]
)


binary_valid_pass=(
    bz["invalid"]==0
)


ha_geometry_pass=all([
    ha["invalid"]==0,
    ha["ge2"]==0,
    ha["max_a"]<2.0,
])


final_pass=all([
    historical_reproduction_pass,
    r32["scale_count"]==507904,
    event_count_pass,
    binary_valid_pass,
    ha_geometry_pass,
])


# ======================================================================================
# CSV
# ======================================================================================

with (
    OUT
    /
    "p7b_r2_global_summary.csv"
).open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w=csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(
        summary_rows
    )


bit_rows=r32[
    "bit_rows"
]


with (
    OUT
    /
    "p7b_r2_per_bit_summary.csv"
).open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w=csv.DictWriter(
        f,
        fieldnames=list(
            bit_rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(
        bit_rows
    )


payload={
    "phase":
        "P7B_R2",

    "model":
        "Qwen3-32B",

    "authoritative":
        final_pass,

    "historical_p4_binary32_reproduction_pass":
        historical_reproduction_pass,

    "binary32_semantics":
        "uniform_unsigned_32bit_quantization_of_log2_scale_over_calibrated_interval",

    "binary32_is_ieee_float":
        False,

    "real_scale_count":
        r32["scale_count"],

    "events_per_encoding":
        event_count_expected,

    "total_events":
        event_count_expected*3,

    "interval":[
        -13.0,
        5.0,
    ],

    "bc":
        BC,

    "bf":
        BF,

    "binary_clean_max_z_error":
        r32[
            "binary_clean_max_z_error"
        ],

    "global":
        r32["global"],

    "event_count_pass":
        event_count_pass,

    "binary_all_fault_scales_valid":
        binary_valid_pass,

    "ha_fault_geometry_pass":
        ha_geometry_pass,

    "fault_probability_claim":
        False,

    "hardware_ber_claim":
        False,

    "p7b_r2_final_pass":
        final_pass,
}


(
    FREEZE
    /
    "P7B_R2_SUMMARY.json"
).write_text(
    json.dumps(
        payload,
        indent=2
    ),
    encoding="utf-8"
)


status=f"""====================================================================================================
METAKV P7B-R2 EXACT BASELINES / QWEN3-32B AUTHORITATIVE SCREEN FINAL STATUS
====================================================================================================

MODEL=QWEN3-32B

BINARY32_Z_SEMANTICS=
UNIFORM_UNSIGNED_32BIT_QUANTIZATION_OF_LOG2_SCALE_OVER_CALIBRATED_INTERVAL

BINARY32_Z_IS_IEEE_FLOAT=False

P4_BINARY32_Z_EXACT_REPRODUCTION_PASS={historical_reproduction_pass}

REAL_SCALE_COUNT={r32['scale_count']}

DEPLOYMENT_INTERVAL=[-13.0,5.0]
INTERVAL_WIDTH=18.0

EVENTS_PER_ENCODING={event_count_expected}
TOTAL_LOGICAL_FAULT_EVENTS={event_count_expected*3}

BINARY32_CLEAN_MAX_Z_ERROR={r32['binary_clean_max_z_error']}

FP32_INVALID_SCALE_COUNT={fp['invalid']}
FP32_MAX_AMPLIFICATION={fp['max_a']}

BINARY32_INVALID_SCALE_COUNT={bz['invalid']}
BINARY32_MAX_AMPLIFICATION={bz['max_a']}
BINARY32_GE2_COUNT={bz['ge2']}
BINARY32_GE10_COUNT={bz['ge10']}
BINARY32_GE100_COUNT={bz['ge100']}

HA_INVALID_SCALE_COUNT={ha['invalid']}
HA_MAX_AMPLIFICATION={ha['max_a']}
HA_GE2_COUNT={ha['ge2']}
HA_GE10_COUNT={ha['ge10']}
HA_GE100_COUNT={ha['ge100']}

EVENT_COUNT_PASS={event_count_pass}
BINARY32_ALL_FAULT_SCALES_VALID={binary_valid_pass}
HA_FAULT_GEOMETRY_PASS={ha_geometry_pass}

FAULT_PROBABILITY_CLAIM=False
HARDWARE_BER_CLAIM=False

P7B_R2_AUTHORITATIVE={final_pass}
P7B_R2_FINAL_PASS={final_pass}
====================================================================================================
"""


(
    FREEZE
    /
    "P7B_R2_FINAL_STATUS.txt"
).write_text(
    status,
    encoding="utf-8"
)


print()
print(status)
