from pathlib import Path
import csv
import json
import math
import sys

import numpy as np


P3A=Path(sys.argv[1])
P4A=Path(sys.argv[2])
RUN=Path(sys.argv[3])

OUT=RUN/"output"
FREEZE=RUN/"freeze"

OUT.mkdir(parents=True,exist_ok=True)
FREEZE.mkdir(parents=True,exist_ok=True)


# ======================================================================================
# CODEC
# ======================================================================================

BC=19
BF=13
BITS=32
FMAX=(1<<BF)-1
COARSE_MASK=(1<<BC)-1

assert BC+BF == 32


def encode_ha(
    z,
    zmin,
    zmax
):
    z=np.asarray(
        z,
        dtype=np.float64
    )

    width=float(
        zmax-zmin
    )

    u=(
        (z-zmin)
        *
        (BC+1)
        /
        width
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


def decode_ha(
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
        zmin
        +
        (
            float(zmax-zmin)
            /
            float(BC+1)
        )
        *
        u
    )


def theorem_bound(
    zmin,
    zmax
):
    return (
        float(zmax-zmin)
        /
        float(BC+1)
        /
        (
            2.0
            *
            float(FMAX)
        )
    )


# ======================================================================================
# RECONSTRUCT PER-PROMPT SCALE ARRAYS FROM FROZEN P3A/P4A RECORDS
# ======================================================================================

def load_model_corpus(
    root,
    prefix
):
    scale_path=(
        root
        /
        "output"
        /
        f"{prefix}_real_log2_scales.npy"
    )

    record_path=(
        root
        /
        "output"
        /
        f"{prefix}_real_scale_records.csv"
    )

    if not scale_path.exists():
        raise RuntimeError(
            f"missing {scale_path}"
        )

    if not record_path.exists():
        raise RuntimeError(
            f"missing {record_path}"
        )

    scales=np.load(
        scale_path
    ).astype(
        np.float64
    )

    with record_path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        records=list(
            csv.DictReader(f)
        )

    per_prompt={}

    cursor=0

    reconstructed=[]

    for r in records:
        prompt_id=int(
            r["prompt_id"]
        )

        groups=int(
            r["groups"]
        )

        block=scales[
            cursor:cursor+groups
        ]

        if len(block)!=groups:
            raise RuntimeError(
                "record/scale reconstruction mismatch"
            )

        per_prompt.setdefault(
            prompt_id,
            []
        ).append(
            block
        )

        reconstructed.append(
            len(block)
        )

        cursor += groups

    if cursor != len(scales):
        raise RuntimeError(
            f"scale count mismatch "
            f"{cursor} != {len(scales)}"
        )

    merged={
        pid:
            np.concatenate(
                blocks
            ).astype(
                np.float64,
                copy=False
            )
        for pid,blocks in per_prompt.items()
    }

    return scales,records,merged


# ======================================================================================
# SPLIT CONFIG
# ======================================================================================

SPLITS=[
    {
        "split":
            "A_TO_B",

        "calibration_prompts":[
            0,1,2,3
        ],

        "evaluation_prompts":[
            4,5,6,7
        ],
    },
    {
        "split":
            "B_TO_A",

        "calibration_prompts":[
            4,5,6,7
        ],

        "evaluation_prompts":[
            0,1,2,3
        ],
    },
]


MODELS=[
    {
        "model":
            "LLAMA31_8B",

        "root":
            P3A,

        "prefix":
            "p3a",

        "full_interval":[
            -13.0,
            3.0
        ],
    },
    {
        "model":
            "QWEN3_14B",

        "root":
            P4A,

        "prefix":
            "p4a",

        "full_interval":[
            -11.0,
            5.0
        ],
    },
]


summary_rows=[]
fault_bit_rows=[]
prompt_rows=[]


# ======================================================================================
# MODEL / SPLIT LOOP
# ======================================================================================

for mc in MODELS:

    model=mc[
        "model"
    ]

    all_scales,records,per_prompt=load_model_corpus(
        mc["root"],
        mc["prefix"]
    )

    prompt_ids=sorted(
        per_prompt.keys()
    )

    if prompt_ids != list(range(8)):
        raise RuntimeError(
            f"{model}: expected prompt IDs 0..7, "
            f"got {prompt_ids}"
        )

    print()
    print("="*140)
    print(
        f"MODEL={model} "
        f"TOTAL_SCALE_COUNT={len(all_scales)} "
        f"FULL_MIN={float(all_scales.min())} "
        f"FULL_MAX={float(all_scales.max())}"
    )
    print("="*140)

    for pid in prompt_ids:
        arr=per_prompt[
            pid
        ]

        prompt_rows.append({
            "model":
                model,

            "prompt_id":
                pid,

            "scale_count":
                len(arr),

            "z_min":
                float(
                    arr.min()
                ),

            "z_max":
                float(
                    arr.max()
                ),

            "z_mean":
                float(
                    arr.mean()
                ),
        })

        print(
            f"P5B0_PROMPT "
            f"MODEL={model} "
            f"PROMPT={pid} "
            f"N={len(arr)} "
            f"MIN={float(arr.min())} "
            f"MAX={float(arr.max())}"
        )

    for split in SPLITS:

        split_name=split[
            "split"
        ]

        cal_prompts=split[
            "calibration_prompts"
        ]

        eval_prompts=split[
            "evaluation_prompts"
        ]

        cal=np.concatenate([
            per_prompt[p]
            for p in cal_prompts
        ])

        ev=np.concatenate([
            per_prompt[p]
            for p in eval_prompts
        ])

        # ----------------------------------------------------------------------
        # Freeze interval using calibration data ONLY.
        # ----------------------------------------------------------------------

        cal_min=float(
            cal.min()
        )

        cal_max=float(
            cal.max()
        )

        zmin=float(
            math.floor(
                cal_min
            )
        )

        zmax=float(
            math.ceil(
                cal_max
            )
        )

        if not zmax > zmin:
            raise RuntimeError(
                "invalid calibrated interval"
            )

        width=(
            zmax-zmin
        )

        # ----------------------------------------------------------------------
        # HELD-OUT COVERAGE
        # ----------------------------------------------------------------------

        eval_min=float(
            ev.min()
        )

        eval_max=float(
            ev.max()
        )

        clip_low_mask=(
            ev < zmin
        )

        clip_high_mask=(
            ev > zmax
        )

        clip_mask=(
            clip_low_mask
            |
            clip_high_mask
        )

        clip_low=int(
            np.count_nonzero(
                clip_low_mask
            )
        )

        clip_high=int(
            np.count_nonzero(
                clip_high_mask
            )
        )

        clip_count=int(
            np.count_nonzero(
                clip_mask
            )
        )

        clip_rate=(
            clip_count
            /
            len(ev)
        )

        # ----------------------------------------------------------------------
        # CLEAN HELD-OUT CODEC ERROR
        # ----------------------------------------------------------------------

        words=encode_ha(
            ev,
            zmin,
            zmax
        )

        decoded=decode_ha(
            words,
            zmin,
            zmax
        )

        z_abs=np.abs(
            decoded-ev
        )

        in_range=(
            ~clip_mask
        )

        in_range_max_z_error=(
            float(
                z_abs[
                    in_range
                ].max()
            )
            if np.any(
                in_range
            )
            else
            None
        )

        overall_max_z_error=float(
            z_abs.max()
        )

        mean_z_error=float(
            z_abs.mean()
        )

        bound=theorem_bound(
            zmin,
            zmax
        )

        in_range_bound_pass=(
            in_range_max_z_error is not None
            and
            in_range_max_z_error
            <=
            bound*(1+1e-9)
            +
            1e-12
        )

        clean_scale=np.exp2(
            ev
        )

        decoded_scale=np.exp2(
            decoded
        )

        scale_abs=np.abs(
            decoded_scale-clean_scale
        )

        scale_rel=(
            scale_abs
            /
            np.maximum(
                clean_scale,
                np.finfo(
                    np.float64
                ).tiny
            )
        )

        max_scale_rel=float(
            scale_rel.max()
        )

        mean_scale_rel=float(
            scale_rel.mean()
        )

        # ----------------------------------------------------------------------
        # EXHAUSTIVE 32 LOGICAL SINGLE-BIT HELD-OUT FAULT SCREEN
        # ----------------------------------------------------------------------

        fault_max_a=1.0
        fault_invalid=0
        fault_ge2=0
        fault_ge10=0
        fault_ge100=0

        per_bit=[]

        for bit in range(32):

            mutated=(
                words
                ^
                np.uint32(
                    1<<bit
                )
            )

            mz=decode_ha(
                mutated,
                zmin,
                zmax
            )

            ms=np.exp2(
                mz
            )

            valid=(
                np.isfinite(
                    ms
                )
                &
                (
                    ms > 0
                )
            )

            invalid=int(
                np.count_nonzero(
                    ~valid
                )
            )

            fault_invalid += invalid

            if np.any(
                valid
            ):
                ratio=(
                    ms[
                        valid
                    ]
                    /
                    clean_scale[
                        valid
                    ]
                )

                amp=np.maximum(
                    ratio,
                    1.0/ratio
                )

                max_a=float(
                    amp.max()
                )

                ge2=int(
                    np.count_nonzero(
                        amp>=2.0
                    )
                )

                ge10=int(
                    np.count_nonzero(
                        amp>=10.0
                    )
                )

                ge100=int(
                    np.count_nonzero(
                        amp>=100.0
                    )
                )

            else:
                max_a=None
                ge2=0
                ge10=0
                ge100=0

            if max_a is not None:
                fault_max_a=max(
                    fault_max_a,
                    max_a
                )

            fault_ge2 += ge2
            fault_ge10 += ge10
            fault_ge100 += ge100

            row={
                "model":
                    model,

                "split":
                    split_name,

                "bit":
                    bit,

                "eval_scale_count":
                    len(ev),

                "invalid":
                    invalid,

                "max_amplification":
                    max_a,

                "ge2":
                    ge2,

                "ge10":
                    ge10,

                "ge100":
                    ge100,
            }

            fault_bit_rows.append(
                row
            )

        event_count=(
            len(ev)
            *
            32
        )

        # ----------------------------------------------------------------------
        # FULL-CORPUS INTERVAL COMPARISON
        # ----------------------------------------------------------------------

        full_zmin=float(
            mc[
                "full_interval"
            ][0]
        )

        full_zmax=float(
            mc[
                "full_interval"
            ][1]
        )

        same_as_full=(
            zmin==full_zmin
            and
            zmax==full_zmax
        )

        # ----------------------------------------------------------------------
        # SPLIT PASS:
        #
        # Do not require zero clipping blindly.
        # We report clipping and use a conservative operational criterion:
        #
        # <= 0.1% held-out scales clipped
        # no invalid HA fault decodes
        # no A >= 2 events
        # in-range numerical theorem bound passes
        #
        # This is an explicitly new P5B sensitivity criterion.
        # ----------------------------------------------------------------------

        clip_operational_pass=(
            clip_rate
            <=
            0.001
        )

        fault_bound_pass=(
            fault_invalid==0
            and
            fault_ge2==0
        )

        split_pass=all([
            clip_operational_pass,
            in_range_bound_pass,
            fault_bound_pass,
        ])

        row={
            "model":
                model,

            "split":
                split_name,

            "calibration_prompts":
                ",".join(
                    map(
                        str,
                        cal_prompts
                    )
                ),

            "evaluation_prompts":
                ",".join(
                    map(
                        str,
                        eval_prompts
                    )
                ),

            "calibration_scale_count":
                len(cal),

            "evaluation_scale_count":
                len(ev),

            "calibration_observed_min":
                cal_min,

            "calibration_observed_max":
                cal_max,

            "evaluation_observed_min":
                eval_min,

            "evaluation_observed_max":
                eval_max,

            "frozen_zmin":
                zmin,

            "frozen_zmax":
                zmax,

            "interval_width":
                width,

            "same_as_original_full_interval":
                same_as_full,

            "clip_low":
                clip_low,

            "clip_high":
                clip_high,

            "clip_count":
                clip_count,

            "clip_rate":
                clip_rate,

            "theorem_bound":
                bound,

            "in_range_max_z_error":
                in_range_max_z_error,

            "in_range_theorem_bound_pass":
                in_range_bound_pass,

            "overall_max_z_error":
                overall_max_z_error,

            "mean_z_error":
                mean_z_error,

            "max_scale_relative_error":
                max_scale_rel,

            "mean_scale_relative_error":
                mean_scale_rel,

            "fault_event_count":
                event_count,

            "fault_invalid":
                fault_invalid,

            "fault_max_amplification":
                fault_max_a,

            "fault_ge2":
                fault_ge2,

            "fault_ge10":
                fault_ge10,

            "fault_ge100":
                fault_ge100,

            "clip_operational_pass":
                clip_operational_pass,

            "fault_bound_pass":
                fault_bound_pass,

            "split_pass":
                split_pass,
        }

        summary_rows.append(
            row
        )

        print()
        print(
            f"P5B0_SPLIT "
            f"MODEL={model} "
            f"SPLIT={split_name} "
            f"CAL={cal_prompts} "
            f"EVAL={eval_prompts} "
            f"CAL_N={len(cal)} "
            f"EVAL_N={len(ev)} "
            f"CAL_RANGE=[{cal_min},{cal_max}] "
            f"EVAL_RANGE=[{eval_min},{eval_max}] "
            f"FROZEN_INTERVAL=[{zmin},{zmax}] "
            f"WIDTH={width} "
            f"SAME_FULL={same_as_full}"
        )

        print(
            f"P5B0_CLEAN "
            f"MODEL={model} "
            f"SPLIT={split_name} "
            f"CLIP_COUNT={clip_count} "
            f"CLIP_RATE={clip_rate} "
            f"BOUND={bound} "
            f"INRANGE_MAX_ZERR={in_range_max_z_error} "
            f"BOUND_PASS={in_range_bound_pass} "
            f"OVERALL_MAX_ZERR={overall_max_z_error} "
            f"MEAN_ZERR={mean_z_error} "
            f"MAX_SCALE_RELERR={max_scale_rel}"
        )

        print(
            f"P5B0_FAULT "
            f"MODEL={model} "
            f"SPLIT={split_name} "
            f"EVENTS={event_count} "
            f"INVALID={fault_invalid} "
            f"MAX_A={fault_max_a} "
            f"GE2={fault_ge2} "
            f"GE10={fault_ge10} "
            f"GE100={fault_ge100}"
        )

        print(
            f"P5B0_GATE "
            f"MODEL={model} "
            f"SPLIT={split_name} "
            f"CLIP_OPERATIONAL_PASS={clip_operational_pass} "
            f"THEOREM_PASS={in_range_bound_pass} "
            f"FAULT_BOUND_PASS={fault_bound_pass} "
            f"SPLIT_PASS={split_pass}"
        )


# ======================================================================================
# SAVE CSV
# ======================================================================================

def save_csv(
    path,
    rows
):
    if not rows:
        return

    fields=[]

    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        w=csv.DictWriter(
            f,
            fieldnames=fields
        )
        w.writeheader()
        w.writerows(rows)


save_csv(
    OUT/"p5b0_split_summary.csv",
    summary_rows
)

save_csv(
    OUT/"p5b0_fault_bit_summary.csv",
    fault_bit_rows
)

save_csv(
    OUT/"p5b0_prompt_scale_summary.csv",
    prompt_rows
)


# ======================================================================================
# FINAL GATES
# ======================================================================================

expected_rows=4

row_count_pass=(
    len(summary_rows)
    ==
    expected_rows
)

all_theorem_pass=all(
    r[
        "in_range_theorem_bound_pass"
    ]
    for r in summary_rows
)

all_fault_bound_pass=all(
    r[
        "fault_bound_pass"
    ]
    for r in summary_rows
)

all_clip_operational_pass=all(
    r[
        "clip_operational_pass"
    ]
    for r in summary_rows
)

all_split_pass=all(
    r[
        "split_pass"
    ]
    for r in summary_rows
)


max_clip_rate=max(
    r[
        "clip_rate"
    ]
    for r in summary_rows
)

max_fault_a=max(
    r[
        "fault_max_amplification"
    ]
    for r in summary_rows
)


# Core scientific result should remain visible even if our <=0.1% operational
# clip criterion fails.
p5b0_complete=all([
    row_count_pass,
    all_theorem_pass,
    all_fault_bound_pass,
])

p5b0_operational_pass=all([
    p5b0_complete,
    all_clip_operational_pass,
])


payload={
    "phase":
        "P5B0",

    "design":
        "reciprocal_4_prompt_calibration_4_prompt_heldout_evaluation",

    "gpu_rerun":
        False,

    "model_forward":
        False,

    "models":[
        "Llama-3.1-8B",
        "Qwen3-14B",
    ],

    "bc":
        BC,

    "bf":
        BF,

    "metadata_bits":
        BITS,

    "interval_calibration_rule":
        "zmin=floor(calibration_min), zmax=ceil(calibration_max)",

    "evaluation_used_to_calibrate_interval":
        False,

    "clip_operational_threshold":
        0.001,

    "logical_fault_model":
        "all_32_single_logical_metadata_bit_flips",

    "fault_probability_claim":
        False,

    "hardware_hbm_fault_claim":
        False,

    "summary":
        summary_rows,

    "row_count_pass":
        row_count_pass,

    "all_theorem_pass":
        all_theorem_pass,

    "all_fault_bound_pass":
        all_fault_bound_pass,

    "all_clip_operational_pass":
        all_clip_operational_pass,

    "max_clip_rate":
        max_clip_rate,

    "max_fault_amplification":
        max_fault_a,

    "p5b0_complete":
        p5b0_complete,

    "p5b0_operational_pass":
        p5b0_operational_pass,
}


(FREEZE/"P5B0_SUMMARY.json").write_text(
    json.dumps(
        payload,
        indent=2
    ),
    encoding="utf-8"
)


status=f"""====================================================================================================
METAKV P5B0 CALIBRATION / HELD-OUT EVALUATION SPLIT SENSITIVITY FINAL STATUS
====================================================================================================

DESIGN=RECIPROCAL_4_PROMPT_CALIBRATION_4_PROMPT_HELDOUT_EVALUATION

MODELS=LLAMA31_8B,QWEN3_14B

SPLIT_A_CALIBRATION_PROMPTS=0,1,2,3
SPLIT_A_EVALUATION_PROMPTS=4,5,6,7

SPLIT_B_CALIBRATION_PROMPTS=4,5,6,7
SPLIT_B_EVALUATION_PROMPTS=0,1,2,3

INTERVAL_CALIBRATION_RULE=FLOOR_CALIBRATION_MIN_AND_CEIL_CALIBRATION_MAX
EVALUATION_USED_TO_CALIBRATE_INTERVAL=False

BC=19
BF=13
METADATA_BITS=32

GPU_RERUN=False
MODEL_FORWARD=False

LOGICAL_FAULT_MODEL=ALL_32_SINGLE_LOGICAL_METADATA_BIT_FLIPS
FAULT_PROBABILITY_CLAIM=False
HARDWARE_HBM_FAULT_CLAIM=False

SPLIT_RESULT_COUNT={len(summary_rows)}
ROW_COUNT_PASS={row_count_pass}

ALL_IN_RANGE_THEOREM_PASS={all_theorem_pass}
ALL_FAULT_BOUND_PASS={all_fault_bound_pass}

CLIP_OPERATIONAL_THRESHOLD=0.001
ALL_CLIP_OPERATIONAL_PASS={all_clip_operational_pass}

MAX_HELDOUT_CLIP_RATE={max_clip_rate}
MAX_HELDOUT_HA_FAULT_AMPLIFICATION={max_fault_a}

P5B0_COMPLETE={p5b0_complete}
P5B0_OPERATIONAL_PASS={p5b0_operational_pass}
====================================================================================================
"""


(FREEZE/"P5B0_FINAL_STATUS.txt").write_text(
    status,
    encoding="utf-8"
)


print()
print(status)
