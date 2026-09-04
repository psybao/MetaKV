import os
import csv
import gzip
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("METAKV_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
M55 = ROOT / "method_upgrade_m55"

SEED = 20260901
RNG = random.Random(SEED)

# ============================================================
# Inputs
# ============================================================

REP_CSV = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b0_mapping_input_freeze"
    / "m55n3b0_unique_representations.csv"
)

SELECTED_8 = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b1c_deep_confirmation"
    / "m55n3b1c_selected_representations.csv"
)

DDR_MAP_JSON = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b1_fast_mapping_synthesis"
    / "m55n3b1_fast_mappings.json"
)

HBM_MAP_JSON = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3e_robust_mapping"
    / "m55m3e_robust_mappings.json"
)

M3F_CSV = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3f_cross_technology_transfer"
    / "m55m3f_cross_technology_transfer.csv"
)

PROD = (
    ROOT
    / "results_local_final"
    / "L2_c4_codesign"
    / "raw"
    / "production_scales_master.csv"
)

PATTERN_GZ = (
    M55
    / "m55m_hardware_fault_maps"
    / "m55m2e_h1_h8_sparse_corpus"
    / "m55m2e_global_h1_h8_patterns.csv.gz"
)

M2C = (
    M55
    / "m55m_hardware_fault_maps"
    / "m55m2c_full_multiplicity"
    / "m55m2c_full_multiplicity_distribution.csv"
)

N2B = (
    M55
    / "m55n2_true_risk_engine"
    / "n2b_exposure_weights"
    / "m55n2b_summary.json"
)

BITPOS = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3c1_rowpress_union_corpus"
    / "m55m3c1_physical_bit_position_counts.csv"
)


# ============================================================
# Outputs
# ============================================================

OUT = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3g_cross_technology_robust"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATE_OUT = OUT / "m55m3g_candidate_true_risks.csv"
RESULT_OUT = OUT / "m55m3g_cross_technology_robust_results.csv"
MAP_OUT = OUT / "m55m3g_cross_technology_robust_mappings.json"
SUMMARY = OUT / "m55m3g_summary.json"
STATUS = OUT / "M5.5M3G_FINAL_STATUS.txt"


BYTE_POP = np.asarray(
    [i.bit_count() for i in range(256)],
    dtype=np.uint8
)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def popcount_u32(x):
    x = np.ascontiguousarray(
        x,
        dtype=np.uint32
    )

    b = x.view(
        np.uint8
    ).reshape(
        x.shape + (4,)
    )

    return BYTE_POP[b].sum(
        axis=-1,
        dtype=np.uint16
    )


# ============================================================
# Representation set
# ============================================================

all_reps = read_csv(REP_CSV)

rep_by_id = {
    r["representation_id"]: r
    for r in all_reps
}

selected_rows = read_csv(
    SELECTED_8
)

selected_ids = [
    r["representation_id"]
    for r in selected_rows
]

if len(selected_ids) != 8:
    raise RuntimeError(
        f"Expected 8 selected representations, got {len(selected_ids)}"
    )

for rid in selected_ids:
    if rid not in rep_by_id:
        raise RuntimeError(
            f"Selected representation missing: {rid}"
        )

print(
    "[M3G] selected representations:",
    flush=True
)

for rid in selected_ids:
    rep = rep_by_id[rid]

    print(
        f"[M3G] {rid} "
        f"{rep['model']} "
        f"Bc={rep['Bc']} Bf={rep['Bf']}",
        flush=True
    )


# ============================================================
# Native maps / M3F references
# ============================================================

ddr_map_json = json.loads(
    DDR_MAP_JSON.read_text(
        encoding="utf-8"
    )
)

hbm_map_json = json.loads(
    HBM_MAP_JSON.read_text(
        encoding="utf-8"
    )
)

m3f_rows = read_csv(
    M3F_CSV
)

m3f_by_id = {
    r["representation_id"]: r
    for r in m3f_rows
}


def get_mapping(
    obj,
    keys
):
    for key in keys:
        if key in obj:
            m = list(obj[key])

            if (
                len(m) != 32
                or
                sorted(m) != list(range(32))
            ):
                raise RuntimeError(
                    f"Invalid mapping under key={key}"
                )

            return m

    raise RuntimeError(
        f"No mapping field in object; tried {keys}"
    )


def ddr_native_map(rid):
    return get_mapping(
        ddr_map_json[rid],
        [
            "mapping",
            "selected_mapping",
        ]
    )


def hbm_native_map(rid):
    return get_mapping(
        hbm_map_json[rid],
        [
            "selected_mapping",
            "mapping",
        ]
    )


# ============================================================
# Production scale samples
# ============================================================

z_by_entry = defaultdict(list)

for r in read_csv(PROD):
    z_by_entry[
        (
            r["model"],
            r["platform"],
        )
    ].append(
        float(
            r["log2_scale"]
        )
    )

z_by_entry = {
    k: np.asarray(
        v,
        dtype=np.float64
    )
    for k,v in z_by_entry.items()
}


# ============================================================
# DDR exact pattern corpus
# ============================================================

ddr_masks = []
ddr_h = []
ddr_counts = []

with gzip.open(
    PATTERN_GZ,
    "rt",
    encoding="utf-8",
    newline=""
) as f:

    for r in csv.DictReader(f):
        ddr_masks.append(
            int(r["mask_uint32"])
        )

        ddr_h.append(
            int(r["h"])
        )

        ddr_counts.append(
            int(r["count"])
        )


ddr_masks = np.asarray(
    ddr_masks,
    dtype=np.uint32
)

ddr_h = np.asarray(
    ddr_h,
    dtype=np.uint8
)

ddr_counts = np.asarray(
    ddr_counts,
    dtype=np.int64
)


if len(ddr_masks) != 352657:
    raise RuntimeError(
        f"Unexpected DDR masks={len(ddr_masks)}"
    )


tail_counts = Counter()

for r in read_csv(M2C):
    h = int(r["h"])

    if h >= 9:
        tail_counts[h] += int(
            r["count"]
        )


n2b = json.loads(
    N2B.read_text(
        encoding="utf-8"
    )
)

TOTAL_EXPOSED = int(
    n2b[
        "total_exposed_32bit_words"
    ]
)


# ============================================================
# DDR additive physical-position surrogate
#
# Used ONLY for candidate generation / screening.
# Final DDR selection uses exact multi-bit true codec risk.
# ============================================================

ddr_phys = np.zeros(
    32,
    dtype=np.float64
)

for p in range(32):
    present = (
        (
            (
                ddr_masks
                >>
                np.uint32(p)
            )
            &
            np.uint32(1)
        )
        .astype(bool)
    )

    ddr_phys[p] = float(
        ddr_counts[
            present
        ].sum()
    )


if np.any(ddr_phys <= 0):
    raise RuntimeError(
        "DDR surrogate has unsupported physical position."
    )


ddr_q = (
    ddr_phys
    /
    ddr_phys.sum()
)


# ============================================================
# HBM pooled union-position marginal distribution
# ============================================================

hbm_phys_by_chip = defaultdict(
    lambda:
        np.zeros(
            32,
            dtype=np.float64
        )
)

for r in read_csv(BITPOS):
    chip = r["chip"]

    bit = int(
        r["physical_bit32"]
    )

    cnt = int(
        r["union_word_presence_count"]
    )

    hbm_phys_by_chip[
        chip
    ][bit] += cnt


chips = sorted(
    hbm_phys_by_chip
)

if len(chips) != 6:
    raise RuntimeError(
        f"Expected six HBM chips, got {chips}"
    )


hbm_q_by_chip = {}

for chip in chips:
    v = hbm_phys_by_chip[
        chip
    ]

    if np.any(v <= 0):
        raise RuntimeError(
            f"{chip}: zero support"
        )

    hbm_q_by_chip[
        chip
    ] = v/v.sum()


# IMPORTANT:
# Cross-technology HBM objective must exactly match frozen M3F.
#
# M3F semantics:
#   sum union-position counts across all six chips first,
#   then normalize globally.
#
# Do NOT average six independently normalized chip distributions here.
hbm_phys_pooled = np.sum(
    np.stack(
        [
            hbm_phys_by_chip[c]
            for c in chips
        ],
        axis=0
    ),
    axis=0
)

hbm_q = (
    hbm_phys_pooled
    /
    hbm_phys_pooled.sum()
)


# ============================================================
# Codec
# ============================================================

def codec_params(rep):
    bc = int(rep["Bc"])
    bf = int(rep["Bf"])
    zmin = float(rep["zmin"])
    zmax = float(rep["zmax"])

    return (
        bc,
        bf,
        zmin,
        zmax,
        zmax-zmin,
    )


def encode_z(
    z,
    bc,
    bf,
    zmin,
    zmax,
):
    W = zmax-zmin
    fmax = (1 << bf)-1

    zc = np.clip(
        z,
        zmin,
        zmax
    )

    u = (
        (bc+1)
        *
        (zc-zmin)
        /
        W
    )

    endpoint = (
        u >= bc+1
    )

    c = np.floor(
        u
    ).astype(
        np.int64
    )

    c = np.clip(
        c,
        0,
        bc
    )

    frac = u-c

    fine = np.rint(
        frac*fmax
    ).astype(
        np.int64
    )

    fine = np.clip(
        fine,
        0,
        fmax
    )

    c[endpoint] = bc
    fine[endpoint] = fmax

    thermo = np.asarray(
        [
            (
                (1 << int(x))-1
                if x > 0
                else 0
            )
            for x in c
        ],
        dtype=np.uint32
    )

    return (
        thermo
        |
        (
            fine.astype(
                np.uint32
            )
            <<
            np.uint32(bc)
        )
    )


def decode_words(
    words,
    bc,
    bf,
    zmin,
    zmax,
):
    W = zmax-zmin

    fmax = (
        (1 << bf)-1
    )

    cmask = (
        (1 << bc)-1
    )

    coarse = (
        words
        &
        np.uint32(cmask)
    )

    cc = popcount_u32(
        coarse
    ).astype(
        np.float64
    )

    fine = (
        (
            words
            >>
            np.uint32(bc)
        )
        &
        np.uint32(fmax)
    ).astype(
        np.float64
    )

    u = (
        cc
        +
        fine/fmax
    )

    z = (
        zmin
        +
        W/(bc+1)*u
    )

    return np.clip(
        z,
        zmin,
        zmax
    )


def single_bit_true_costs(
    rep
):
    bc,bf,zmin,zmax,W = (
        codec_params(rep)
    )

    z = z_by_entry[
        (
            rep["model"],
            rep["platform"],
        )
    ]

    cw = encode_z(
        z,
        bc,
        bf,
        zmin,
        zmax
    )

    clean = decode_words(
        cw,
        bc,
        bf,
        zmin,
        zmax
    )

    costs = np.zeros(
        32,
        dtype=np.float64
    )

    bound = (
        W/(bc+1)
    )

    for l in range(32):
        fw = (
            cw
            ^
            (
                np.uint32(1)
                <<
                np.uint32(l)
            )
        )

        fd = decode_words(
            fw,
            bc,
            bf,
            zmin,
            zmax
        )

        A = np.abs(
            fd-clean
        )

        if np.max(A) > bound + 1e-10:
            raise RuntimeError(
                f"single-bit theorem violation "
                f"{rep['representation_id']} l={l}"
            )

        costs[l] = float(
            np.mean(A)
        )

    return costs


# ============================================================
# Mapping helpers
# mapping[logical] = physical
# ============================================================

def validate_mapping(m):
    if (
        len(m) != 32
        or
        sorted(m) != list(range(32))
    ):
        raise RuntimeError(
            "Invalid mapping permutation."
        )


def inverse_mapping(m):
    validate_mapping(m)

    inv = [-1]*32

    for logical,physical in enumerate(m):
        inv[physical] = logical

    return inv


def rearrangement(
    logical_cost,
    physical_exposure
):
    """
    Exact optimum for additive marginal objective:
    highest logical cost -> lowest physical exposure.
    """
    logical_order = sorted(
        range(32),
        key=lambda x:
            logical_cost[x],
        reverse=True
    )

    physical_order = sorted(
        range(32),
        key=lambda x:
            physical_exposure[x]
    )

    m = [-1]*32

    for l,p in zip(
        logical_order,
        physical_order
    ):
        m[l] = p

    validate_mapping(m)

    return m


def additive_risk(
    logical_cost,
    physical_exposure,
    mapping
):
    inv = inverse_mapping(
        mapping
    )

    return float(
        sum(
            physical_exposure[p]
            *
            logical_cost[
                inv[p]
            ]
            for p in range(32)
        )
    )


def perturb(
    base,
    swaps
):
    m = list(base)

    for _ in range(swaps):
        a,b = RNG.sample(
            range(32),
            2
        )

        m[a],m[b] = (
            m[b],
            m[a]
        )

    return m


# ============================================================
# DDR exact risk
# ============================================================

def physical_to_logical_masks(
    mapping
):
    inv = inverse_mapping(
        mapping
    )

    out = np.zeros(
        len(ddr_masks),
        dtype=np.uint32
    )

    for p in range(32):
        present = (
            (
                (
                    ddr_masks
                    >>
                    np.uint32(p)
                )
                &
                1
            )
            .astype(bool)
        )

        out[
            present
        ] |= (
            np.uint32(1)
            <<
            np.uint32(
                inv[p]
            )
        )

    return out


def ddr_true_risk(
    rep,
    mapping
):
    bc,bf,zmin,zmax,W = (
        codec_params(rep)
    )

    z = z_by_entry[
        (
            rep["model"],
            rep["platform"],
        )
    ]

    cw = encode_z(
        z,
        bc,
        bf,
        zmin,
        zmax
    )

    clean = decode_words(
        cw,
        bc,
        bf,
        zmin,
        zmax
    )

    logical_masks = (
        physical_to_logical_masks(
            mapping
        )
    )

    total_exact = 0.0

    BATCH = 4096

    for start in range(
        0,
        len(logical_masks),
        BATCH
    ):
        end = min(
            start+BATCH,
            len(logical_masks)
        )

        fw = (
            logical_masks[
                start:end,
                None
            ]
            ^
            cw[None,:]
        )

        fd = decode_words(
            fw,
            bc,
            bf,
            zmin,
            zmax
        )

        A = np.abs(
            fd-clean[None,:]
        )

        maxA = np.max(
            A,
            axis=1
        )

        theoretical = np.minimum(
            W,
            ddr_h[
                start:end
            ].astype(float)
            *
            W/(bc+1)
        )

        if np.any(
            maxA
            >
            theoretical + 1e-10
        ):
            raise RuntimeError(
                f"DDR theorem violation "
                f"{rep['representation_id']}"
            )

        total_exact += float(
            np.dot(
                np.mean(
                    A,
                    axis=1
                ),
                ddr_counts[
                    start:end
                ]
            )
        )


    tail_bound_total = sum(
        count
        *
        min(
            W,
            h*W/(bc+1)
        )
        for h,count
        in tail_counts.items()
    )


    return (
        total_exact
        +
        tail_bound_total
    ) / TOTAL_EXPOSED


# ============================================================
# Candidate generation
# ============================================================

ALPHAS = np.linspace(
    0.0,
    1.0,
    21
)

RANDOM_PERTURBATIONS = 192

TRUE_FINALISTS = 20


candidate_rows = []
result_rows = []
mapping_out = {}

t0 = time.perf_counter()


for rep_index,rid in enumerate(
    selected_ids,
    1
):
    rep = rep_by_id[
        rid
    ]

    native_ddr = ddr_native_map(
        rid
    )

    native_hbm = hbm_native_map(
        rid
    )

    identity = list(
        range(32)
    )

    costs = single_bit_true_costs(
        rep
    )


    # --------------------------------------------------------
    # Native reference risk comes from frozen M3F.
    # --------------------------------------------------------

    m3f = m3f_by_id[
        rid
    ]

    ddr_ref = float(
        m3f[
            "ddr_native_mapping_risk"
        ]
    )

    hbm_ref = float(
        m3f[
            "hbm_native_mapping_marginal_risk"
        ]
    )

    if (
        ddr_ref <= 0
        or
        hbm_ref <= 0
    ):
        raise RuntimeError(
            f"Invalid native reference {rid}"
        )


    # --------------------------------------------------------
    # Base candidate set
    # --------------------------------------------------------

    candidates = {}


    def add_candidate(
        name,
        mapping
    ):
        validate_mapping(
            mapping
        )

        key = tuple(
            mapping
        )

        if key not in candidates:
            candidates[key] = {
                "mapping":
                    list(mapping),

                "aliases":
                    [name],
            }

        else:
            candidates[
                key
            ][
                "aliases"
            ].append(name)


    add_candidate(
        "IDENTITY",
        identity
    )

    add_candidate(
        "DDR_NATIVE_REFERENCE",
        native_ddr
    )

    add_candidate(
        "HBM_NATIVE_REFERENCE",
        native_hbm
    )


    # Pure additive DDR and pooled HBM optima.
    ddr_additive = rearrangement(
        costs,
        ddr_q
    )

    hbm_pooled = rearrangement(
        costs,
        hbm_q
    )

    add_candidate(
        "DDR_ADDITIVE_REARRANGEMENT",
        ddr_additive
    )

    add_candidate(
        "HBM_POOLED_REARRANGEMENT",
        hbm_pooled
    )


    # HBM per-chip exact marginal optima.
    for chip in chips:
        add_candidate(
            f"HBM_OPT_{chip}",
            rearrangement(
                costs,
                hbm_q_by_chip[
                    chip
                ]
            )
        )


    # --------------------------------------------------------
    # Cross-technology alpha frontier.
    #
    # Both distributions normalized to sum 1.
    # This is candidate generation only.
    # --------------------------------------------------------

    alpha_maps = []

    for alpha in ALPHAS:
        combined = (
            (1.0-alpha)
            *
            ddr_q
            +
            alpha
            *
            hbm_q
        )

        m = rearrangement(
            costs,
            combined
        )

        alpha_maps.append(
            m
        )

        add_candidate(
            f"ALPHA_{alpha:.2f}",
            m
        )


    # --------------------------------------------------------
    # Structured perturbations around native / frontier maps
    # --------------------------------------------------------

    perturb_bases = [
        native_ddr,
        native_hbm,
        ddr_additive,
        hbm_pooled,
    ]

    # Add a few alpha anchors.
    for idx in [
        4,
        8,
        10,
        12,
        16,
    ]:
        perturb_bases.append(
            alpha_maps[idx]
        )


    for n in range(
        RANDOM_PERTURBATIONS
    ):
        base = perturb_bases[
            n
            %
            len(perturb_bases)
        ]

        swaps = (
            1
            +
            (
                n
                //
                len(perturb_bases)
            )
            %
            6
        )

        m = perturb(
            base,
            swaps
        )

        add_candidate(
            f"PERTURB_{n:03d}_S{swaps}",
            m
        )


    # --------------------------------------------------------
    # Cheap dual-domain screening
    #
    # DDR side here is additive surrogate ONLY.
    # HBM side is actual marginal additive objective.
    # --------------------------------------------------------

    ddr_ref_surrogate = additive_risk(
        costs,
        ddr_q,
        native_ddr
    )

    hbm_ref_check = additive_risk(
        costs,
        hbm_q,
        native_hbm
    )

    print(
        f"[M3G] {rid} HBM_REF_CHECK "
        f"calc={hbm_ref_check:.15g} "
        f"M3F={hbm_ref:.15g} "
        f"absdiff={abs(hbm_ref_check-hbm_ref):.3e}",
        flush=True
    )

    if abs(
        hbm_ref_check
        -
        hbm_ref
    ) > max(
        1e-10,
        1e-6*abs(hbm_ref)
    ):
        raise RuntimeError(
            f"HBM native-reference mismatch "
            f"{rid}: calc={hbm_ref_check} "
            f"M3F={hbm_ref}"
        )


    screened = []

    for item in candidates.values():
        m = item[
            "mapping"
        ]

        ddr_surr = additive_risk(
            costs,
            ddr_q,
            m
        )

        hbm_risk = additive_risk(
            costs,
            hbm_q,
            m
        )

        ddr_surr_ratio = (
            ddr_surr
            /
            ddr_ref_surrogate
        )

        hbm_ratio = (
            hbm_risk
            /
            hbm_ref
        )

        screen_score = max(
            ddr_surr_ratio,
            hbm_ratio
        )

        screened.append({
            "mapping":
                m,

            "aliases":
                item[
                    "aliases"
                ],

            "ddr_surrogate":
                ddr_surr,

            "ddr_surrogate_ratio":
                ddr_surr_ratio,

            "hbm_risk":
                hbm_risk,

            "hbm_ratio":
                hbm_ratio,

            "screen_score":
                screen_score,
        })


    screened.sort(
        key=lambda x:
            x[
                "screen_score"
            ]
    )


    finalists = (
        screened[
            :TRUE_FINALISTS
        ]
    )


    # Native references must ALWAYS be true-evaluated.
    required = [
        identity,
        native_ddr,
        native_hbm,
        ddr_additive,
        hbm_pooled,
    ]


    for m in required:
        if not any(
            tuple(x["mapping"])
            ==
            tuple(m)
            for x in finalists
        ):
            item = next(
                x
                for x in screened
                if tuple(
                    x["mapping"]
                )
                ==
                tuple(m)
            )

            finalists.append(
                item
            )


    print()
    print(
        f"[M3G] REP "
        f"{rep_index}/{len(selected_ids)} "
        f"{rid} {rep['model']} "
        f"Bc={rep['Bc']} "
        f"candidates={len(candidates)} "
        f"true_finalists={len(finalists)}",
        flush=True
    )


    # --------------------------------------------------------
    # Final true evaluation
    # --------------------------------------------------------

    evaluated = []


    for ci,item in enumerate(
        finalists,
        1
    ):
        tc0 = time.perf_counter()

        m = item[
            "mapping"
        ]

        ddr_risk = ddr_true_risk(
            rep,
            m
        )

        hbm_risk = item[
            "hbm_risk"
        ]

        ddr_ratio = (
            ddr_risk
            /
            ddr_ref
        )

        hbm_ratio = (
            hbm_risk
            /
            hbm_ref
        )

        robust_score = max(
            ddr_ratio,
            hbm_ratio
        )

        mean_ratio = (
            ddr_ratio
            +
            hbm_ratio
        ) / 2.0


        row = {
            "representation_id":
                rid,

            "model":
                rep["model"],

            "platform":
                rep["platform"],

            "Bc":
                rep["Bc"],

            "Bf":
                rep["Bf"],

            "candidate_aliases":
                "|".join(
                    item[
                        "aliases"
                    ]
                ),

            "ddr_true_risk":
                ddr_risk,

            "ddr_native_reference_risk":
                ddr_ref,

            "ddr_reference_ratio":
                ddr_ratio,

            "hbm_marginal_risk":
                hbm_risk,

            "hbm_native_reference_risk":
                hbm_ref,

            "hbm_reference_ratio":
                hbm_ratio,

            "max_native_reference_ratio":
                robust_score,

            "mean_native_reference_ratio":
                mean_ratio,

            "ddr_surrogate_ratio":
                item[
                    "ddr_surrogate_ratio"
                ],
        }


        evaluated.append(
            {
                "row":
                    row,

                "mapping":
                    m,

                "aliases":
                    item[
                        "aliases"
                    ],
            }
        )

        candidate_rows.append(
            row
        )


        print(
            f"[M3G] {rid} "
            f"TRUE {ci}/{len(finalists)} "
            f"{row['candidate_aliases']} "
            f"DDRratio={ddr_ratio:.6f} "
            f"HBMratio={hbm_ratio:.6f} "
            f"MAX={robust_score:.6f} "
            f"dt={time.perf_counter()-tc0:.1f}s",
            flush=True
        )


    selected = min(
        evaluated,
        key=lambda x:
            (
                x[
                    "row"
                ][
                    "max_native_reference_ratio"
                ],

                x[
                    "row"
                ][
                    "mean_native_reference_ratio"
                ],
            )
    )


    sr = selected[
        "row"
    ]


    # Native candidate rows.
    ddr_native_eval = next(
        x
        for x in evaluated
        if tuple(
            x["mapping"]
        )
        ==
        tuple(
            native_ddr
        )
    )

    hbm_native_eval = next(
        x
        for x in evaluated
        if tuple(
            x["mapping"]
        )
        ==
        tuple(
            native_hbm
        )
    )


    result_rows.append({
        "representation_id":
            rid,

        "model":
            rep["model"],

        "platform":
            rep["platform"],

        "Bc":
            rep["Bc"],

        "Bf":
            rep["Bf"],

        "selected_aliases":
            sr[
                "candidate_aliases"
            ],

        "selected_ddr_reference_ratio":
            sr[
                "ddr_reference_ratio"
            ],

        "selected_hbm_reference_ratio":
            sr[
                "hbm_reference_ratio"
            ],

        "selected_max_native_reference_ratio":
            sr[
                "max_native_reference_ratio"
            ],

        "selected_mean_native_reference_ratio":
            sr[
                "mean_native_reference_ratio"
            ],

        "ddr_native_cross_technology_score":
            ddr_native_eval[
                "row"
            ][
                "max_native_reference_ratio"
            ],

        "hbm_native_cross_technology_score":
            hbm_native_eval[
                "row"
            ][
                "max_native_reference_ratio"
            ],

        "selected_equals_ddr_native":
            tuple(
                selected[
                    "mapping"
                ]
            )
            ==
            tuple(
                native_ddr
            ),

        "selected_equals_hbm_native":
            tuple(
                selected[
                    "mapping"
                ]
            )
            ==
            tuple(
                native_hbm
            ),
    })


    mapping_out[
        rid
    ] = {
        "mapping_semantics":
            "mapping[logical]=physical",

        "selected_mapping":
            selected[
                "mapping"
            ],

        "selected_aliases":
            selected[
                "aliases"
            ],

        "ddr_reference_ratio":
            sr[
                "ddr_reference_ratio"
            ],

        "hbm_reference_ratio":
            sr[
                "hbm_reference_ratio"
            ],

        "max_native_reference_ratio":
            sr[
                "max_native_reference_ratio"
            ],

        "native_references_are_global_optima":
            False,

        "global_cross_technology_optimum_claim":
            False,

        "candidate_generation_uses_surrogate":
            True,

        "final_ddr_evaluation_uses_true_multibit_codec_risk":
            True,

        "final_hbm_evaluation_uses_true_single_bit_marginal_codec_risk":
            True,
    }


    print(
        f"[M3G] DONE {rid} "
        f"selected={sr['candidate_aliases']} "
        f"DDRratio={sr['ddr_reference_ratio']:.6f} "
        f"HBMratio={sr['hbm_reference_ratio']:.6f} "
        f"MAX={sr['max_native_reference_ratio']:.6f} "
        f"elapsed={time.perf_counter()-t0:.1f}s",
        flush=True
    )


# ============================================================
# Save
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
    CANDIDATE_OUT,
    candidate_rows
)

write_rows(
    RESULT_OUT,
    result_rows
)


MAP_OUT.write_text(
    json.dumps(
        mapping_out,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# Aggregate
# ============================================================

max_ratios = np.asarray(
    [
        float(
            r[
                "selected_max_native_reference_ratio"
            ]
        )
        for r in result_rows
    ],
    dtype=np.float64
)

ddr_ratios = np.asarray(
    [
        float(
            r[
                "selected_ddr_reference_ratio"
            ]
        )
        for r in result_rows
    ],
    dtype=np.float64
)

hbm_ratios = np.asarray(
    [
        float(
            r[
                "selected_hbm_reference_ratio"
            ]
        )
        for r in result_rows
    ],
    dtype=np.float64
)

ddr_native_scores = np.asarray(
    [
        float(
            r[
                "ddr_native_cross_technology_score"
            ]
        )
        for r in result_rows
    ],
    dtype=np.float64
)

hbm_native_scores = np.asarray(
    [
        float(
            r[
                "hbm_native_cross_technology_score"
            ]
        )
        for r in result_rows
    ],
    dtype=np.float64
)


summary = {
    "stage":
        "M5.5M3G",

    "representations":
        len(result_rows),

    "selection_set":
        "N3B1C representative eight",

    "objective":
        (
            "minimize maximum native-reference-normalized "
            "DDR/HBM domain loss over evaluated candidate set"
        ),

    "native_references_are_global_optima":
        False,

    "global_cross_technology_optimum_claim":
        False,

    "final_ddr_metric":
        (
            "exposure-normalized true multi-bit "
            "hybrid codec risk"
        ),

    "final_hbm_metric":
        (
            "pooled union-position-weighted "
            "single-bit true-codec marginal risk"
        ),

    "surrogate_role":
        "candidate generation and screening only",

    "selected_max_reference_ratio": {
        "min":
            float(
                max_ratios.min()
            ),

        "mean":
            float(
                max_ratios.mean()
            ),

        "median":
            float(
                np.median(
                    max_ratios
                )
            ),

        "max":
            float(
                max_ratios.max()
            ),
    },

    "selected_ddr_reference_ratio": {
        "min":
            float(
                ddr_ratios.min()
            ),

        "mean":
            float(
                ddr_ratios.mean()
            ),

        "median":
            float(
                np.median(
                    ddr_ratios
                )
            ),

        "max":
            float(
                ddr_ratios.max()
            ),
    },

    "selected_hbm_reference_ratio": {
        "min":
            float(
                hbm_ratios.min()
            ),

        "mean":
            float(
                hbm_ratios.mean()
            ),

        "median":
            float(
                np.median(
                    hbm_ratios
                )
            ),

        "max":
            float(
                hbm_ratios.max()
            ),
    },

    "ddr_native_cross_technology_score_mean":
        float(
            ddr_native_scores.mean()
        ),

    "hbm_native_cross_technology_score_mean":
        float(
            hbm_native_scores.mean()
        ),

    "selected_equals_ddr_native_count":
        sum(
            bool(
                r[
                    "selected_equals_ddr_native"
                ]
            )
            for r in result_rows
        ),

    "selected_equals_hbm_native_count":
        sum(
            bool(
                r[
                    "selected_equals_hbm_native"
                ]
            )
            for r in result_rows
        ),

    "absolute_ddr_hbm_risk_directly_comparable":
        False,

    "hbm_simultaneous_event_claim":
        False,

    "hbm_absolute_fault_probability_claim":
        False,

    "internal_silicon_lane_claim":
        False,
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
# Audit
# ============================================================

critical = []

if len(result_rows) != 8:
    critical.append(
        "REPRESENTATION_COUNT_NE_8"
    )

if np.any(
    max_ratios <= 0
):
    critical.append(
        "INVALID_MAX_REFERENCE_RATIO"
    )

if not all(
    len(
        mapping_out[
            rid
        ][
            "selected_mapping"
        ]
    ) == 32
    for rid in mapping_out
):
    critical.append(
        "INVALID_MAPPING_LENGTH"
    )


lines = [
    "M5.5M3G CROSS-TECHNOLOGY ROBUST MAPPING SYNTHESIS",

    f"REPRESENTATIONS={len(result_rows)}",

    (
        "SELECTED_MAX_REFERENCE_RATIO_MIN="
        f"{float(max_ratios.min())}"
    ),

    (
        "SELECTED_MAX_REFERENCE_RATIO_MEAN="
        f"{float(max_ratios.mean())}"
    ),

    (
        "SELECTED_MAX_REFERENCE_RATIO_MEDIAN="
        f"{float(np.median(max_ratios))}"
    ),

    (
        "SELECTED_MAX_REFERENCE_RATIO_MAX="
        f"{float(max_ratios.max())}"
    ),

    (
        "SELECTED_DDR_REFERENCE_RATIO_MEAN="
        f"{float(ddr_ratios.mean())}"
    ),

    (
        "SELECTED_HBM_REFERENCE_RATIO_MEAN="
        f"{float(hbm_ratios.mean())}"
    ),

    (
        "DDR_NATIVE_CROSS_TECH_SCORE_MEAN="
        f"{float(ddr_native_scores.mean())}"
    ),

    (
        "HBM_NATIVE_CROSS_TECH_SCORE_MEAN="
        f"{float(hbm_native_scores.mean())}"
    ),

    (
        "SELECTED_EQUALS_DDR_NATIVE_COUNT="
        f"{sum(bool(r['selected_equals_ddr_native']) for r in result_rows)}"
    ),

    (
        "SELECTED_EQUALS_HBM_NATIVE_COUNT="
        f"{sum(bool(r['selected_equals_hbm_native']) for r in result_rows)}"
    ),

    "OBJECTIVE=NATIVE_REFERENCE_NORMALIZED_MINIMAX",

    "NATIVE_REFERENCES_ARE_GLOBAL_OPTIMA=False",

    "GLOBAL_CROSS_TECHNOLOGY_OPTIMUM_CLAIM=False",

    "SURROGATE_USED_FOR_CANDIDATE_GENERATION_ONLY=True",

    "FINAL_DDR_USES_TRUE_MULTIBIT_CODEC_RISK=True",

    "FINAL_HBM_USES_SINGLE_BIT_MARGINAL_TRUE_CODEC_RISK=True",

    "ABSOLUTE_DDR_HBM_RISKS_DIRECTLY_COMPARABLE=False",

    "HBM_SIMULTANEOUS_EVENT_CLAIM=False",

    "HBM_ABSOLUTE_FAULT_PROBABILITY_CLAIM=False",

    "INTERNAL_SILICON_LANE_CLAIM=False",

    "REPRESENTATION_FROZEN=True",

    "RAW_DDR_FILES_MODIFIED=0",

    "RAW_HBM_FILES_MODIFIED=0",

    f"CRITICAL_ISSUES_COUNT={len(critical)}",

    f"M55M3G_FINAL_PASS={len(critical)==0}",
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

for x in lines:
    print(x)

print()
print("CANDIDATES =", CANDIDATE_OUT)
print("RESULTS =", RESULT_OUT)
print("MAPPINGS =", MAP_OUT)
print("SUMMARY =", SUMMARY)
print("STATUS =", STATUS)


