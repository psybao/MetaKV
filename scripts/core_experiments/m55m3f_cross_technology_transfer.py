import os
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("METAKV_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
M55 = ROOT / "method_upgrade_m55"

REP_CSV = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b0_mapping_input_freeze"
    / "m55n3b0_unique_representations.csv"
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

OUT = (
    M55
    / "m55m3_hbm2_corpus"
    / "m3f_cross_technology_transfer"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

RESULT = OUT / "m55m3f_cross_technology_transfer.csv"
SUMMARY = OUT / "m55m3f_summary.json"
STATUS = OUT / "M5.5M3F_FINAL_STATUS.txt"


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
# Inputs
# ============================================================

reps = read_csv(REP_CSV)

if len(reps) != 44:
    raise RuntimeError(
        f"Expected 44 reps, got {len(reps)}"
    )

ddr_maps_raw = json.loads(
    DDR_MAP_JSON.read_text(
        encoding="utf-8"
    )
)

hbm_maps_raw = json.loads(
    HBM_MAP_JSON.read_text(
        encoding="utf-8"
    )
)


def get_ddr_mapping(rid):
    obj = ddr_maps_raw[rid]

    if "mapping" in obj:
        return list(obj["mapping"])

    if "selected_mapping" in obj:
        return list(obj["selected_mapping"])

    raise RuntimeError(
        f"DDR mapping field not found: {rid}"
    )


def get_hbm_mapping(rid):
    obj = hbm_maps_raw[rid]

    if "selected_mapping" in obj:
        return list(
            obj["selected_mapping"]
        )

    if "mapping" in obj:
        return list(
            obj["mapping"]
        )

    raise RuntimeError(
        f"HBM mapping field not found: {rid}"
    )


def validate_mapping(m):
    if (
        len(m) != 32
        or
        sorted(m) != list(range(32))
    ):
        raise RuntimeError(
            "invalid permutation"
        )


# ============================================================
# Production scales
# ============================================================

z_by_entry = defaultdict(list)

for r in read_csv(PROD):
    z_by_entry[
        (
            r["model"],
            r["platform"],
        )
    ].append(
        float(r["log2_scale"])
    )

z_by_entry = {
    k: np.asarray(
        v,
        dtype=np.float64
    )
    for k,v in z_by_entry.items()
}


# ============================================================
# DDR patterns
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
        f"DDR masks={len(ddr_masks)}"
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
# HBM pooled marginal position distribution
#
# We use pooled union presence only for cross-technology
# transfer scoring.
#
# M3E robust quality itself remains six-chip max-regret.
# ============================================================

hbm_phys = np.zeros(
    32,
    dtype=np.float64
)

for r in read_csv(BITPOS):
    bit = int(
        r["physical_bit32"]
    )

    hbm_phys[bit] += int(
        r["union_word_presence_count"]
    )

if np.any(hbm_phys <= 0):
    raise RuntimeError(
        "HBM position with zero support"
    )

hbm_q = (
    hbm_phys
    /
    hbm_phys.sum()
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
    ).astype(np.int64)

    c = np.clip(
        c,
        0,
        bc
    )

    frac = u-c

    fine = np.rint(
        frac*fmax
    ).astype(np.int64)

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
            fine.astype(np.uint32)
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
    fmax = (1 << bf)-1
    cmask = (1 << bc)-1

    coarse = (
        words
        &
        np.uint32(cmask)
    )

    cc = popcount_u32(
        coarse
    ).astype(np.float64)

    fine = (
        (
            words
            >>
            np.uint32(bc)
        )
        &
        np.uint32(fmax)
    ).astype(np.float64)

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


def inverse_mapping(mapping):
    inv = [-1]*32

    for l,p in enumerate(mapping):
        if inv[p] != -1:
            raise RuntimeError(
                "non-bijective mapping"
            )

        inv[p] = l

    return inv


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
        sel = (
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

        out[sel] |= (
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

    total = 0.0
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
            fd
            -
            clean[None,:]
        )

        maxA = np.max(
            A,
            axis=1
        )

        bound = np.minimum(
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
            bound + 1e-10
        ):
            raise RuntimeError(
                "DDR theorem violation"
            )

        total += float(
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


    tail = sum(
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
        total + tail
    ) / TOTAL_EXPOSED


def single_bit_costs(
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

    out = np.zeros(
        32,
        dtype=np.float64
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

        out[l] = float(
            np.mean(
                np.abs(
                    fd-clean
                )
            )
        )

    return out


def hbm_marginal_risk(
    costs,
    mapping
):
    inv = inverse_mapping(
        mapping
    )

    return float(
        sum(
            hbm_q[p]
            *
            costs[
                inv[p]
            ]
            for p in range(32)
        )
    )


# ============================================================
# Main
# ============================================================

rows = []


for i,rep in enumerate(
    reps,
    1
):
    rid = rep[
        "representation_id"
    ]

    identity = list(range(32))
    ddr_map = get_ddr_mapping(rid)
    hbm_map = get_hbm_mapping(rid)

    validate_mapping(identity)
    validate_mapping(ddr_map)
    validate_mapping(hbm_map)

    # DDR domain
    ddr_identity = ddr_true_risk(
        rep,
        identity
    )

    ddr_native = ddr_true_risk(
        rep,
        ddr_map
    )

    ddr_hbm_transfer = ddr_true_risk(
        rep,
        hbm_map
    )

    # HBM domain
    costs = single_bit_costs(
        rep
    )

    hbm_identity = hbm_marginal_risk(
        costs,
        identity
    )

    hbm_native = hbm_marginal_risk(
        costs,
        hbm_map
    )

    hbm_ddr_transfer = hbm_marginal_risk(
        costs,
        ddr_map
    )


    ddr_native_gain = (
        100.0
        *
        (
            ddr_identity-ddr_native
        )
        /
        ddr_identity
    )

    ddr_hbm_gain = (
        100.0
        *
        (
            ddr_identity
            -
            ddr_hbm_transfer
        )
        /
        ddr_identity
    )

    hbm_native_gain = (
        100.0
        *
        (
            hbm_identity-hbm_native
        )
        /
        hbm_identity
    )

    hbm_ddr_gain = (
        100.0
        *
        (
            hbm_identity
            -
            hbm_ddr_transfer
        )
        /
        hbm_identity
    )


    hbm_to_ddr_regret = (
        ddr_hbm_transfer
        /
        ddr_native
    )

    ddr_to_hbm_regret = (
        hbm_ddr_transfer
        /
        hbm_native
    )


    same_mapping = (
        tuple(ddr_map)
        ==
        tuple(hbm_map)
    )


    rows.append({
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

        "ddr_identity_risk":
            ddr_identity,

        "ddr_native_mapping_risk":
            ddr_native,

        "ddr_hbm_mapping_transfer_risk":
            ddr_hbm_transfer,

        "ddr_native_gain_percent":
            ddr_native_gain,

        "ddr_hbm_transfer_gain_percent":
            ddr_hbm_gain,

        "hbm_to_ddr_regret":
            hbm_to_ddr_regret,

        "hbm_identity_marginal_risk":
            hbm_identity,

        "hbm_native_mapping_marginal_risk":
            hbm_native,

        "hbm_ddr_mapping_transfer_marginal_risk":
            hbm_ddr_transfer,

        "hbm_native_gain_percent":
            hbm_native_gain,

        "hbm_ddr_transfer_gain_percent":
            hbm_ddr_gain,

        "ddr_to_hbm_regret":
            ddr_to_hbm_regret,

        "ddr_hbm_mapping_identical":
            same_mapping,
    })


    print(
        f"[M3F] {i}/{len(reps)} "
        f"{rid} Bc={rep['Bc']} "
        f"DDR native={ddr_native_gain:.3f}% "
        f"HBMmap->DDR={ddr_hbm_gain:.3f}% "
        f"regret={hbm_to_ddr_regret:.4f} | "
        f"HBM native={hbm_native_gain:.3f}% "
        f"DDRmap->HBM={hbm_ddr_gain:.3f}% "
        f"regret={ddr_to_hbm_regret:.4f}",
        flush=True
    )


# ============================================================
# Output
# ============================================================

with RESULT.open(
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


def arr(field):
    return np.asarray(
        [
            float(r[field])
            for r in rows
        ],
        dtype=np.float64
    )


ddr_native_gain = arr(
    "ddr_native_gain_percent"
)

ddr_transfer_gain = arr(
    "ddr_hbm_transfer_gain_percent"
)

hbm_native_gain = arr(
    "hbm_native_gain_percent"
)

hbm_transfer_gain = arr(
    "hbm_ddr_transfer_gain_percent"
)

h2d_regret = arr(
    "hbm_to_ddr_regret"
)

d2h_regret = arr(
    "ddr_to_hbm_regret"
)


summary = {
    "stage":
        "M5.5M3F",

    "representations":
        len(rows),

    "ddr_domain_metric":
        "exposure-normalized true multi-bit hybrid codec risk",

    "hbm_domain_metric":
        (
            "pooled union-position-weighted "
            "single-bit true-codec marginal risk"
        ),

    "metrics_not_cross-domain_comparable":
        True,

    "relative_mapping_regret_comparable":
        True,

    "ddr_native_gain_percent": {
        "min":
            float(
                ddr_native_gain.min()
            ),

        "mean":
            float(
                ddr_native_gain.mean()
            ),

        "median":
            float(
                np.median(
                    ddr_native_gain
                )
            ),

        "max":
            float(
                ddr_native_gain.max()
            ),
    },

    "hbm_map_on_ddr_gain_percent": {
        "min":
            float(
                ddr_transfer_gain.min()
            ),

        "mean":
            float(
                ddr_transfer_gain.mean()
            ),

        "median":
            float(
                np.median(
                    ddr_transfer_gain
                )
            ),

        "max":
            float(
                ddr_transfer_gain.max()
            ),
    },

    "hbm_native_gain_percent": {
        "min":
            float(
                hbm_native_gain.min()
            ),

        "mean":
            float(
                hbm_native_gain.mean()
            ),

        "median":
            float(
                np.median(
                    hbm_native_gain
                )
            ),

        "max":
            float(
                hbm_native_gain.max()
            ),
    },

    "ddr_map_on_hbm_gain_percent": {
        "min":
            float(
                hbm_transfer_gain.min()
            ),

        "mean":
            float(
                hbm_transfer_gain.mean()
            ),

        "median":
            float(
                np.median(
                    hbm_transfer_gain
                )
            ),

        "max":
            float(
                hbm_transfer_gain.max()
            ),
    },

    "hbm_to_ddr_regret": {
        "min":
            float(
                h2d_regret.min()
            ),

        "mean":
            float(
                h2d_regret.mean()
            ),

        "median":
            float(
                np.median(
                    h2d_regret
                )
            ),

        "max":
            float(
                h2d_regret.max()
            ),
    },

    "ddr_to_hbm_regret": {
        "min":
            float(
                d2h_regret.min()
            ),

        "mean":
            float(
                d2h_regret.mean()
            ),

        "median":
            float(
                np.median(
                    d2h_regret
                )
            ),

        "max":
            float(
                d2h_regret.max()
            ),
    },

    "identical_mapping_count":
        sum(
            bool(
                r[
                    "ddr_hbm_mapping_identical"
                ]
            )
            for r in rows
        ),

    "simultaneous_hbm_event_claim":
        False,

    "absolute_hbm_fault_probability_claim":
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


critical = []

if len(rows) != 44:
    critical.append(
        "REP_COUNT_NE_44"
    )

if np.any(
    ddr_native_gain < -1e-10
):
    critical.append(
        "DDR_NATIVE_MAPPING_REGRESSION"
    )

if np.any(
    hbm_native_gain < -1e-10
):
    critical.append(
        "HBM_NATIVE_MAPPING_REGRESSION"
    )

if np.any(
    h2d_regret <= 0
):
    critical.append(
        "INVALID_HBM_TO_DDR_REGRET"
    )

if np.any(
    d2h_regret <= 0
):
    critical.append(
        "INVALID_DDR_TO_HBM_REGRET"
    )


lines = [
    "M5.5M3F DDR-HBM CROSS-TECHNOLOGY MAPPING TRANSFER",

    f"REPRESENTATIONS={len(rows)}",

    (
        "IDENTICAL_MAPPING_COUNT="
        f"{sum(bool(r['ddr_hbm_mapping_identical']) for r in rows)}"
    ),

    (
        "DDR_NATIVE_GAIN_MEAN_PERCENT="
        f"{float(ddr_native_gain.mean())}"
    ),

    (
        "HBM_MAP_ON_DDR_GAIN_MEAN_PERCENT="
        f"{float(ddr_transfer_gain.mean())}"
    ),

    (
        "HBM_TO_DDR_REGRET_MEAN="
        f"{float(h2d_regret.mean())}"
    ),

    (
        "HBM_TO_DDR_REGRET_MAX="
        f"{float(h2d_regret.max())}"
    ),

    (
        "HBM_NATIVE_GAIN_MEAN_PERCENT="
        f"{float(hbm_native_gain.mean())}"
    ),

    (
        "DDR_MAP_ON_HBM_GAIN_MEAN_PERCENT="
        f"{float(hbm_transfer_gain.mean())}"
    ),

    (
        "DDR_TO_HBM_REGRET_MEAN="
        f"{float(d2h_regret.mean())}"
    ),

    (
        "DDR_TO_HBM_REGRET_MAX="
        f"{float(d2h_regret.max())}"
    ),

    "DDR_DOMAIN_USES_TRUE_MULTIBIT_HYBRID_RISK=True",

    "HBM_DOMAIN_USES_MARGINAL_SINGLE_BIT_RISK=True",

    "ABSOLUTE_DDR_HBM_RISK_VALUES_DIRECTLY_COMPARABLE=False",

    "RELATIVE_MAPPING_REGRET_COMPARABLE=True",

    "HBM_SIMULTANEOUS_EVENT_CLAIM=False",

    "HBM_ABSOLUTE_FAULT_PROBABILITY_CLAIM=False",

    "INTERNAL_SILICON_LANE_CLAIM=False",

    "REPRESENTATION_FROZEN=True",

    "RAW_DDR_FILES_MODIFIED=0",

    "RAW_HBM_FILES_MODIFIED=0",

    f"CRITICAL_ISSUES_COUNT={len(critical)}",

    f"M55M3F_FINAL_PASS={len(critical)==0}",
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
print("RESULT =", RESULT)
print("SUMMARY =", SUMMARY)
print("STATUS =", STATUS)
