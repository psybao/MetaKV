import os
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("METAKV_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
M55 = ROOT / "method_upgrade_m55"

IN_DIR = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3c1_rowpress_union_corpus"
)

BITPOS = (
    IN_DIR
    / "m55m3c1_physical_bit_position_counts.csv"
)

CONDITIONS = (
    IN_DIR
    / "m55m3c1_condition_summary.csv"
)

OUT = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3d_position_heterogeneity"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CHIP_OUT = OUT / "m55m3d_per_chip_metrics.csv"
AGGON_OUT = OUT / "m55m3d_per_chip_aggon_metrics.csv"
CROSS_OUT = OUT / "m55m3d_cross_chip_correlations.csv"
AGGON_CORR_OUT = OUT / "m55m3d_aggon_rank_stability.csv"
BIT_RANK_OUT = OUT / "m55m3d_bit_rank_summary.csv"
SUMMARY = OUT / "m55m3d_summary.json"
STATUS = OUT / "M5.5M3D_FINAL_STATUS.txt"


# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def average_ranks(x):
    """
    Average ranks for ties, ascending.
    Equivalent to standard rankdata(method='average').
    """
    x = np.asarray(x, dtype=np.float64)

    order = np.argsort(
        x,
        kind="mergesort"
    )

    ranks = np.empty(
        len(x),
        dtype=np.float64
    )

    i = 0

    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and
            x[order[j]] == x[order[i]]
        ):
            j += 1

        avg = (
            (i + 1)
            +
            j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg

        i = j

    return ranks


def pearson(x,y):
    x = np.asarray(x,dtype=np.float64)
    y = np.asarray(y,dtype=np.float64)

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(x,y)[0,1]
    )


def spearman(x,y):
    return pearson(
        average_ranks(x),
        average_ranks(y)
    )


def metrics(v):
    v = np.asarray(
        v,
        dtype=np.float64
    )

    if len(v) != 32:
        raise RuntimeError(
            f"Expected 32 positions, got {len(v)}"
        )

    total = float(
        v.sum()
    )

    if total <= 0:
        return {
            "total": 0,
            "zero_positions": 32,
            "mean": 0,
            "std": 0,
            "cv": float("nan"),
            "max_min_nonzero_ratio": float("nan"),
            "max_uniform_ratio": float("nan"),
            "min_uniform_ratio": float("nan"),
            "top1_mass": float("nan"),
            "top4_mass": float("nan"),
            "top8_mass": float("nan"),
            "normalized_entropy": float("nan"),
            "l1_distance_from_uniform": float("nan"),
        }

    p = v / total

    nz = v[
        v > 0
    ]

    mean = float(
        np.mean(v)
    )

    std = float(
        np.std(v)
    )

    cv = (
        std/mean
        if mean > 0
        else float("nan")
    )

    uniform = 1.0/32.0

    sorted_p = np.sort(p)[::-1]

    entropy = float(
        -np.sum(
            p[p > 0]
            *
            np.log2(
                p[p > 0]
            )
        )
    )

    normalized_entropy = (
        entropy
        /
        math.log2(32)
    )

    return {
        "total":
            int(total),

        "zero_positions":
            int(
                np.sum(v == 0)
            ),

        "mean":
            mean,

        "std":
            std,

        "cv":
            float(cv),

        "max_min_nonzero_ratio":
            (
                float(
                    np.max(nz)
                    /
                    np.min(nz)
                )
                if len(nz)
                else float("nan")
            ),

        "max_uniform_ratio":
            float(
                np.max(p)
                /
                uniform
            ),

        "min_uniform_ratio":
            float(
                np.min(p)
                /
                uniform
            ),

        "top1_mass":
            float(
                sorted_p[:1].sum()
            ),

        "top4_mass":
            float(
                sorted_p[:4].sum()
            ),

        "top8_mass":
            float(
                sorted_p[:8].sum()
            ),

        "normalized_entropy":
            float(
                normalized_entropy
            ),

        "l1_distance_from_uniform":
            float(
                np.sum(
                    np.abs(
                        p-uniform
                    )
                )
            ),
    }


def vector_from(counter):
    return np.asarray(
        [
            counter.get(i,0)
            for i in range(32)
        ],
        dtype=np.float64
    )


# ============================================================
# Load
# ============================================================

rows = read_csv(
    BITPOS
)

condition_rows = read_csv(
    CONDITIONS
)

print(
    f"[M3D] BITPOS rows={len(rows)} "
    f"condition rows={len(condition_rows)}",
    flush=True
)


# ============================================================
# Aggregate
# ============================================================

union_chip = defaultdict(
    lambda: defaultdict(int)
)

raw_chip = defaultdict(
    lambda: defaultdict(int)
)

union_chip_aggon = defaultdict(
    lambda: defaultdict(int)
)

raw_chip_aggon = defaultdict(
    lambda: defaultdict(int)
)


for r in rows:
    chip = r["chip"]
    aggon = int(r["aggon"])
    bit = int(
        r["physical_bit32"]
    )

    raw = int(
        r["raw_bit_observation_count"]
    )

    union = int(
        r["union_word_presence_count"]
    )

    if not 0 <= bit < 32:
        raise RuntimeError(
            f"physical_bit32 out of range: {bit}"
        )

    raw_chip[
        chip
    ][bit] += raw

    union_chip[
        chip
    ][bit] += union

    raw_chip_aggon[
        (chip,aggon)
    ][bit] += raw

    union_chip_aggon[
        (chip,aggon)
    ][bit] += union


chips = sorted(
    union_chip.keys()
)

aggons = sorted({
    int(r["aggon"])
    for r in rows
})


print(
    f"[M3D] chips={chips}",
    flush=True
)

print(
    f"[M3D] aggon={aggons}",
    flush=True
)


if len(chips) != 6:
    raise RuntimeError(
        f"Expected 6 chips, got {len(chips)}"
    )

if aggons != list(range(10)):
    raise RuntimeError(
        f"Expected AggOn 0..9, got {aggons}"
    )


# ============================================================
# Per-chip metrics
#
# Primary:
# union-word presence distribution.
#
# Secondary:
# raw bit-observation distribution.
# ============================================================

chip_metric_rows = []


for chip in chips:
    uv = vector_from(
        union_chip[chip]
    )

    rv = vector_from(
        raw_chip[chip]
    )

    um = metrics(uv)
    rm = metrics(rv)

    row = {
        "chip":
            chip,

        "primary_semantics":
            "UNION_WORD_PRESENCE",

        "union_total":
            um["total"],

        "union_zero_positions":
            um["zero_positions"],

        "union_cv":
            um["cv"],

        "union_max_min_nonzero_ratio":
            um["max_min_nonzero_ratio"],

        "union_max_uniform_ratio":
            um["max_uniform_ratio"],

        "union_min_uniform_ratio":
            um["min_uniform_ratio"],

        "union_top1_mass":
            um["top1_mass"],

        "union_top4_mass":
            um["top4_mass"],

        "union_top8_mass":
            um["top8_mass"],

        "union_normalized_entropy":
            um["normalized_entropy"],

        "union_l1_from_uniform":
            um["l1_distance_from_uniform"],

        "raw_total":
            rm["total"],

        "raw_cv":
            rm["cv"],

        "raw_max_min_nonzero_ratio":
            rm["max_min_nonzero_ratio"],

        "raw_top8_mass":
            rm["top8_mass"],

        "raw_normalized_entropy":
            rm["normalized_entropy"],

        "raw_l1_from_uniform":
            rm["l1_distance_from_uniform"],

        "raw_vs_union_spearman":
            spearman(
                rv,
                uv
            ),
    }

    chip_metric_rows.append(
        row
    )

    print(
        f"[M3D] {chip} "
        f"union_cv={um['cv']:.6f} "
        f"max/min={um['max_min_nonzero_ratio']:.6f} "
        f"top8={um['top8_mass']:.6f} "
        f"Hnorm={um['normalized_entropy']:.6f} "
        f"raw_union_rho={row['raw_vs_union_spearman']:.6f}",
        flush=True
    )


# ============================================================
# Per-chip per-AggOn metrics
# ============================================================

aggon_metric_rows = []


for chip in chips:
    for aggon in aggons:
        uv = vector_from(
            union_chip_aggon[
                (chip,aggon)
            ]
        )

        rv = vector_from(
            raw_chip_aggon[
                (chip,aggon)
            ]
        )

        um = metrics(uv)
        rm = metrics(rv)

        aggon_metric_rows.append({
            "chip":
                chip,

            "aggon":
                aggon,

            "union_total":
                um["total"],

            "union_cv":
                um["cv"],

            "union_max_min_nonzero_ratio":
                um["max_min_nonzero_ratio"],

            "union_top8_mass":
                um["top8_mass"],

            "union_normalized_entropy":
                um["normalized_entropy"],

            "union_l1_from_uniform":
                um["l1_distance_from_uniform"],

            "raw_total":
                rm["total"],

            "raw_cv":
                rm["cv"],

            "raw_top8_mass":
                rm["top8_mass"],

            "raw_normalized_entropy":
                rm["normalized_entropy"],

            "raw_vs_union_spearman":
                spearman(
                    rv,
                    uv
                ),
        })


# ============================================================
# Cross-chip correlation
# ============================================================

cross_rows = []

cross_union_rhos = []
cross_raw_rhos = []


for a,b in combinations(
    chips,
    2
):
    ua = vector_from(
        union_chip[a]
    )

    ub = vector_from(
        union_chip[b]
    )

    ra = vector_from(
        raw_chip[a]
    )

    rb = vector_from(
        raw_chip[b]
    )

    urho = spearman(
        ua,
        ub
    )

    rrho = spearman(
        ra,
        rb
    )

    cross_union_rhos.append(
        urho
    )

    cross_raw_rhos.append(
        rrho
    )

    cross_rows.append({
        "chip_a":
            a,

        "chip_b":
            b,

        "union_spearman":
            urho,

        "union_pearson":
            pearson(
                ua,
                ub
            ),

        "raw_spearman":
            rrho,

        "raw_pearson":
            pearson(
                ra,
                rb
            ),
    })


# ============================================================
# AggOn rank stability
#
# Within each chip:
# all 45 pairwise AggOn comparisons.
# ============================================================

aggon_corr_rows = []

within_chip_aggon_rhos = defaultdict(
    list
)


for chip in chips:
    for a,b in combinations(
        aggons,
        2
    ):
        va = vector_from(
            union_chip_aggon[
                (chip,a)
            ]
        )

        vb = vector_from(
            union_chip_aggon[
                (chip,b)
            ]
        )

        rho = spearman(
            va,
            vb
        )

        within_chip_aggon_rhos[
            chip
        ].append(
            rho
        )

        aggon_corr_rows.append({
            "chip":
                chip,

            "aggon_a":
                a,

            "aggon_b":
                b,

            "union_spearman":
                rho,

            "union_pearson":
                pearson(
                    va,
                    vb
                ),
        })


# ============================================================
# Global / bit rank summary
# ============================================================

global_union = defaultdict(int)
global_raw = defaultdict(int)

for chip in chips:
    for bit in range(32):
        global_union[bit] += (
            union_chip[chip][bit]
        )

        global_raw[bit] += (
            raw_chip[chip][bit]
        )


gu = vector_from(
    global_union
)

gr = vector_from(
    global_raw
)

gm = metrics(
    gu
)

grm = metrics(
    gr
)


# Per-bit chip rank stability.
chip_rank_vectors = {}

for chip in chips:
    v = vector_from(
        union_chip[chip]
    )

    # Larger count -> rank 1 means hotter.
    chip_rank_vectors[chip] = (
        33.0
        -
        average_ranks(v)
    )


global_hot_rank = (
    33.0
    -
    average_ranks(gu)
)


bit_rank_rows = []

for bit in range(32):
    vals = np.asarray(
        [
            union_chip[c][bit]
            for c in chips
        ],
        dtype=np.float64
    )

    ranks = np.asarray(
        [
            chip_rank_vectors[c][bit]
            for c in chips
        ],
        dtype=np.float64
    )

    bit_rank_rows.append({
        "physical_bit32":
            bit,

        "global_union_presence":
            int(
                global_union[bit]
            ),

        "global_raw_observations":
            int(
                global_raw[bit]
            ),

        "global_union_mass":
            float(
                global_union[bit]
                /
                gu.sum()
            ),

        "global_hot_rank":
            float(
                global_hot_rank[bit]
            ),

        "mean_chip_hot_rank":
            float(
                np.mean(ranks)
            ),

        "std_chip_hot_rank":
            float(
                np.std(ranks)
            ),

        "min_chip_hot_rank":
            float(
                np.min(ranks)
            ),

        "max_chip_hot_rank":
            float(
                np.max(ranks)
            ),

        "chip_count_cv_for_this_bit":
            float(
                np.std(vals)
                /
                np.mean(vals)
            )
            if np.mean(vals) > 0
            else float("nan"),
    })


# ============================================================
# Save CSVs
# ============================================================

def write_rows(
    path,
    rows
):
    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        w.writeheader()
        w.writerows(rows)


write_rows(
    CHIP_OUT,
    chip_metric_rows
)

write_rows(
    AGGON_OUT,
    aggon_metric_rows
)

write_rows(
    CROSS_OUT,
    cross_rows
)

write_rows(
    AGGON_CORR_OUT,
    aggon_corr_rows
)

write_rows(
    BIT_RANK_OUT,
    bit_rank_rows
)


# ============================================================
# Aggregate descriptive evidence
# ============================================================

union_cvs = np.asarray(
    [
        float(
            r["union_cv"]
        )
        for r in chip_metric_rows
    ],
    dtype=float
)

union_ratios = np.asarray(
    [
        float(
            r[
                "union_max_min_nonzero_ratio"
            ]
        )
        for r in chip_metric_rows
    ],
    dtype=float
)

union_top8 = np.asarray(
    [
        float(
            r["union_top8_mass"]
        )
        for r in chip_metric_rows
    ],
    dtype=float
)

union_entropy = np.asarray(
    [
        float(
            r["union_normalized_entropy"]
        )
        for r in chip_metric_rows
    ],
    dtype=float
)

raw_union_rhos = np.asarray(
    [
        float(
            r["raw_vs_union_spearman"]
        )
        for r in chip_metric_rows
    ],
    dtype=float
)


all_aggon_rhos = np.asarray(
    [
        x
        for chip in chips
        for x in within_chip_aggon_rhos[chip]
        if np.isfinite(x)
    ],
    dtype=float
)

cross_union_rhos_np = np.asarray(
    [
        x
        for x in cross_union_rhos
        if np.isfinite(x)
    ],
    dtype=float
)


top8_bits = [
    int(x)
    for x in np.argsort(
        gu
    )[::-1][:8]
]

bottom8_bits = [
    int(x)
    for x in np.argsort(
        gu
    )[:8]
]


summary = {
    "stage":
        "M5.5M3D",

    "mechanism":
        "HBM2_ROWPRESS",

    "primary_position_metric":
        "UNION_WORD_PRESENCE",

    "secondary_position_metric":
        "RAW_BIT_OBSERVATIONS",

    "chips":
        chips,

    "aggon_values":
        aggons,

    "global_union_total":
        int(
            gu.sum()
        ),

    "global_raw_total":
        int(
            gr.sum()
        ),

    "global_union_metrics":
        gm,

    "global_raw_metrics":
        grm,

    "per_chip_union_cv": {
        "min":
            float(
                np.min(union_cvs)
            ),

        "mean":
            float(
                np.mean(union_cvs)
            ),

        "max":
            float(
                np.max(union_cvs)
            ),
    },

    "per_chip_union_max_min_ratio": {
        "min":
            float(
                np.min(union_ratios)
            ),

        "mean":
            float(
                np.mean(union_ratios)
            ),

        "max":
            float(
                np.max(union_ratios)
            ),
    },

    "per_chip_union_top8_mass": {
        "min":
            float(
                np.min(union_top8)
            ),

        "mean":
            float(
                np.mean(union_top8)
            ),

        "max":
            float(
                np.max(union_top8)
            ),
    },

    "per_chip_union_normalized_entropy": {
        "min":
            float(
                np.min(union_entropy)
            ),

        "mean":
            float(
                np.mean(union_entropy)
            ),

        "max":
            float(
                np.max(union_entropy)
            ),
    },

    "raw_vs_union_spearman": {
        "min":
            float(
                np.min(raw_union_rhos)
            ),

        "mean":
            float(
                np.mean(raw_union_rhos)
            ),

        "max":
            float(
                np.max(raw_union_rhos)
            ),
    },

    "cross_chip_union_spearman": {
        "min":
            float(
                np.min(
                    cross_union_rhos_np
                )
            ),

        "mean":
            float(
                np.mean(
                    cross_union_rhos_np
                )
            ),

        "median":
            float(
                np.median(
                    cross_union_rhos_np
                )
            ),

        "max":
            float(
                np.max(
                    cross_union_rhos_np
                )
            ),
    },

    "within_chip_aggon_union_spearman": {
        "min":
            float(
                np.min(
                    all_aggon_rhos
                )
            ),

        "mean":
            float(
                np.mean(
                    all_aggon_rhos
                )
            ),

        "median":
            float(
                np.median(
                    all_aggon_rhos
                )
            ),

        "max":
            float(
                np.max(
                    all_aggon_rhos
                )
            ),
    },

    "global_top8_physical_bits":
        top8_bits,

    "global_bottom8_physical_bits":
        bottom8_bits,

    "absolute_fault_probability_claim":
        False,

    "simultaneous_event_claim":
        False,

    "internal_silicon_lane_claim":
        False,

    "coordinate_projection_source_confirmed":
        True,
}


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# Status
#
# PASS means audit is internally complete.
# It does NOT predetermine whether heterogeneity is "strong".
# ============================================================

critical = []

if len(chips) != 6:
    critical.append(
        "CHIP_COUNT_NE_6"
    )

if int(gu.sum()) <= 0:
    critical.append(
        "NO_UNION_POSITION_DATA"
    )

if int(gr.sum()) <= 0:
    critical.append(
        "NO_RAW_POSITION_DATA"
    )

if len(cross_rows) != 15:
    critical.append(
        "CROSS_CHIP_PAIR_COUNT_NE_15"
    )

if len(aggon_corr_rows) != 270:
    critical.append(
        "AGGON_PAIR_COUNT_NE_270"
    )


lines = [
    "M5.5M3D HBM2 PHYSICAL-POSITION HETEROGENEITY AUDIT",

    f"CHIPS={len(chips)}",

    f"AGGON_LEVELS={len(aggons)}",

    (
        "GLOBAL_UNION_POSITION_COUNT="
        f"{int(gu.sum())}"
    ),

    (
        "GLOBAL_RAW_BIT_OBSERVATIONS="
        f"{int(gr.sum())}"
    ),

    (
        "GLOBAL_UNION_CV="
        f"{gm['cv']}"
    ),

    (
        "GLOBAL_UNION_MAX_MIN_RATIO="
        f"{gm['max_min_nonzero_ratio']}"
    ),

    (
        "GLOBAL_UNION_TOP8_MASS="
        f"{gm['top8_mass']}"
    ),

    (
        "GLOBAL_UNION_NORMALIZED_ENTROPY="
        f"{gm['normalized_entropy']}"
    ),

    (
        "PER_CHIP_UNION_CV_MIN="
        f"{float(np.min(union_cvs))}"
    ),

    (
        "PER_CHIP_UNION_CV_MEAN="
        f"{float(np.mean(union_cvs))}"
    ),

    (
        "PER_CHIP_UNION_CV_MAX="
        f"{float(np.max(union_cvs))}"
    ),

    (
        "CROSS_CHIP_SPEARMAN_MIN="
        f"{float(np.min(cross_union_rhos_np))}"
    ),

    (
        "CROSS_CHIP_SPEARMAN_MEAN="
        f"{float(np.mean(cross_union_rhos_np))}"
    ),

    (
        "CROSS_CHIP_SPEARMAN_MEDIAN="
        f"{float(np.median(cross_union_rhos_np))}"
    ),

    (
        "CROSS_CHIP_SPEARMAN_MAX="
        f"{float(np.max(cross_union_rhos_np))}"
    ),

    (
        "WITHIN_CHIP_AGGON_SPEARMAN_MIN="
        f"{float(np.min(all_aggon_rhos))}"
    ),

    (
        "WITHIN_CHIP_AGGON_SPEARMAN_MEAN="
        f"{float(np.mean(all_aggon_rhos))}"
    ),

    (
        "WITHIN_CHIP_AGGON_SPEARMAN_MEDIAN="
        f"{float(np.median(all_aggon_rhos))}"
    ),

    (
        "RAW_VS_UNION_SPEARMAN_MEAN="
        f"{float(np.mean(raw_union_rhos))}"
    ),

    (
        "GLOBAL_TOP8_BITS="
        + ",".join(
            map(str,top8_bits)
        )
    ),

    (
        "GLOBAL_BOTTOM8_BITS="
        + ",".join(
            map(str,bottom8_bits)
        )
    ),

    "PRIMARY_METRIC=UNION_WORD_PRESENCE",

    "RAW_COUNTS_SECONDARY=True",

    "ABSOLUTE_FAULT_PROBABILITY_CLAIM=False",

    "SIMULTANEOUS_EVENT_PATTERN_CLAIM=False",

    "INTERNAL_SILICON_LANE_CLAIM=False",

    "COORDINATE_PROJECTION_SOURCE_CONFIRMED=True",

    "RAW_HBM_FILES_MODIFIED=0",

    f"CRITICAL_ISSUES_COUNT={len(critical)}",

    f"M55M3D_FINAL_PASS={len(critical)==0}",
]


for c in critical:
    lines.append(
        f"CRITICAL={c}"
    )


STATUS.write_text(
    "\n".join(lines)
    +
    "\n",
    encoding="utf-8"
)


print()
print("="*110)

for line in lines:
    print(line)

print()
print("CHIP METRICS =", CHIP_OUT)
print("AGGON METRICS =", AGGON_OUT)
print("CROSS CHIP =", CROSS_OUT)
print("AGGON STABILITY =", AGGON_CORR_OUT)
print("BIT RANKS =", BIT_RANK_OUT)
print("SUMMARY =", SUMMARY)
print("STATUS =", STATUS)
