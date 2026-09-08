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

REP_CSV = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b0_mapping_input_freeze"
    / "m55n3b0_unique_representations.csv"
)

FAST_RESULT = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b1_fast_mapping_synthesis"
    / "m55n3b1_fast_mapping_results.csv"
)

FAST_MAP = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b1_fast_mapping_synthesis"
    / "m55n3b1_fast_mappings.json"
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

OUT = (
    M55
    / "m55n3_ha_fbms_synthesis"
    / "n3b1c_deep_confirmation"
)

OUT.mkdir(parents=True, exist_ok=True)

SELECTED = OUT / "m55n3b1c_selected_representations.csv"
CANDIDATE_OUT = OUT / "m55n3b1c_true_candidates.csv"
RESULT_OUT = OUT / "m55n3b1c_confirmation_results.csv"
MAP_OUT = OUT / "m55n3b1c_best_mappings.json"
SUMMARY = OUT / "m55n3b1c_summary.json"
STATUS = OUT / "M5.5N3B1C_FINAL_STATUS.txt"


SEED = 20260831
RNG = random.Random(SEED)

N_CONFIRM = 8

# Broad candidate generation, but only selected finalists use true codec.
STRUCTURED_CANDIDATES = 768
TRUE_FINALISTS = 24

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
    b = x.view(np.uint8).reshape(
        x.shape + (4,)
    )
    return BYTE_POP[b].sum(
        axis=-1,
        dtype=np.uint16
    )


reps = read_csv(REP_CSV)
fast = read_csv(FAST_RESULT)
fast_map = json.loads(
    FAST_MAP.read_text(encoding="utf-8")
)

rep_by_id = {
    r["representation_id"]: r
    for r in reps
}

fast_by_id = {
    r["representation_id"]: r
    for r in fast
}

if len(reps) != 44 or len(fast) != 44:
    raise RuntimeError(
        f"Expected 44/44; got reps={len(reps)} fast={len(fast)}"
    )


# ============================================================
# Representative subset selection
# ============================================================

gain_sorted = sorted(
    fast,
    key=lambda r: float(r["risk_reduction_percent"])
)

gain_values = np.asarray(
    [float(r["risk_reduction_percent"]) for r in fast],
    dtype=float
)
gain_median = float(np.median(gain_values))

selected_ids = []


def add(rep_id):
    if rep_id not in selected_ids:
        selected_ids.append(rep_id)


# min / max gain
add(gain_sorted[0]["representation_id"])
add(gain_sorted[-1]["representation_id"])

# closest median
add(
    min(
        fast,
        key=lambda r:
            abs(
                float(r["risk_reduction_percent"])
                -
                gain_median
            )
    )["representation_id"]
)

# min / max Bc
add(
    min(
        reps,
        key=lambda r: int(r["Bc"])
    )["representation_id"]
)
add(
    max(
        reps,
        key=lambda r: int(r["Bc"])
    )["representation_id"]
)


# Maximize model diversity next.
used_models = {
    rep_by_id[x]["model"]
    for x in selected_ids
}

for r in sorted(
    fast,
    key=lambda x:
        float(x["risk_reduction_percent"]),
    reverse=True
):
    rid = r["representation_id"]
    model = rep_by_id[rid]["model"]

    if (
        model not in used_models
        and
        rid not in selected_ids
    ):
        add(rid)
        used_models.add(model)

    if len(selected_ids) >= N_CONFIRM:
        break


# Then maximize Bc diversity if still short.
used_bc = {
    int(rep_by_id[x]["Bc"])
    for x in selected_ids
}

for r in reps:
    rid = r["representation_id"]
    bc = int(r["Bc"])

    if (
        bc not in used_bc
        and
        rid not in selected_ids
    ):
        add(rid)
        used_bc.add(bc)

    if len(selected_ids) >= N_CONFIRM:
        break


# Final deterministic fill.
for r in gain_sorted:
    add(r["representation_id"])
    if len(selected_ids) >= N_CONFIRM:
        break

selected_ids = selected_ids[:N_CONFIRM]

selected_rows = []

for rank,rid in enumerate(selected_ids):
    r = rep_by_id[rid]
    f = fast_by_id[rid]

    selected_rows.append({
        "rank": rank,
        "representation_id": rid,
        "model": r["model"],
        "platform": r["platform"],
        "Bc": r["Bc"],
        "Bf": r["Bf"],
        "zmin": r["zmin"],
        "zmax": r["zmax"],
        "fast_gain_percent":
            f["risk_reduction_percent"],
        "fast_selected_candidate":
            f["selected_candidate"],
    })


with SELECTED.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(selected_rows[0].keys())
    )
    w.writeheader()
    w.writerows(selected_rows)


print("[N3B1C] selected representations:", flush=True)

for x in selected_rows:
    print(
        "[N3B1C]",
        x["representation_id"],
        x["model"],
        f"Bc={x['Bc']}",
        f"fast_gain={x['fast_gain_percent']}%",
        flush=True,
    )


# ============================================================
# Production scales
# ============================================================

z_by_entry = defaultdict(list)

for r in read_csv(PROD):
    z_by_entry[
        (r["model"], r["platform"])
    ].append(
        float(r["log2_scale"])
    )

z_by_entry = {
    k: np.asarray(v, dtype=np.float64)
    for k,v in z_by_entry.items()
}


# ============================================================
# DDR pattern corpus
# ============================================================

masks = []
hs = []
counts = []

with gzip.open(
    PATTERN_GZ,
    "rt",
    encoding="utf-8",
    newline=""
) as f:
    for r in csv.DictReader(f):
        masks.append(int(r["mask_uint32"]))
        hs.append(int(r["h"]))
        counts.append(int(r["count"]))

masks = np.asarray(masks, dtype=np.uint32)
hs = np.asarray(hs, dtype=np.uint8)
counts = np.asarray(counts, dtype=np.int64)

if len(masks) != 352657:
    raise RuntimeError(
        f"Unexpected empirical masks={len(masks)}"
    )


# Weighted physical incidence across ALL empirical h<=8 events.
physical_incidence = np.zeros(32, dtype=np.float64)

for p in range(32):
    sel = (
        ((masks >> np.uint32(p)) & np.uint32(1))
        .astype(bool)
    )
    physical_incidence[p] = float(
        counts[sel].sum()
    )


# h1-only incidence.
h1_incidence = np.zeros(32, dtype=np.float64)

for m,c,h in zip(masks, counts, hs):
    if h == 1:
        p = int(m).bit_length() - 1
        h1_incidence[p] += int(c)


# tail
tail_counts = Counter()

for r in read_csv(M2C):
    h = int(r["h"])
    if h >= 9:
        tail_counts[h] += int(r["count"])


n2b = json.loads(
    N2B.read_text(encoding="utf-8")
)

TOTAL_EXPOSED = int(
    n2b["total_exposed_32bit_words"]
)


# ============================================================
# Codec
# ============================================================

def codec_params(rep):
    bc = int(rep["Bc"])
    bf = int(rep["Bf"])
    zmin = float(rep["zmin"])
    zmax = float(rep["zmax"])
    return bc,bf,zmin,zmax,zmax-zmin


def logical_weights(bc,bf,W):
    fmax = (1 << bf)-1
    step = W/(bc+1)

    w = np.zeros(32, dtype=np.float64)
    w[:bc] = step

    for j in range(bf):
        w[bc+j] = (
            step * (1 << j) / fmax
        )

    return w


def optimal_additive_mapping(lw, incidence):
    # Highest logical sensitivity -> lowest physical incidence.
    logical = sorted(
        range(32),
        key=lambda x: lw[x],
        reverse=True
    )

    physical = sorted(
        range(32),
        key=lambda x: incidence[x]
    )

    m = [-1]*32

    for l,p in zip(logical, physical):
        m[l] = p

    return m


def invert_mapping(mapping):
    inv = [-1]*32

    for l,p in enumerate(mapping):
        if inv[p] != -1:
            raise RuntimeError("non-bijective mapping")
        inv[p] = l

    return inv


def additive_score(mapping,lw,incidence):
    inv = invert_mapping(mapping)

    return float(
        sum(
            incidence[p]
            *
            lw[inv[p]]
            for p in range(32)
        )
    )


def perturb(base, swaps):
    m = list(base)

    for _ in range(swaps):
        a,b = RNG.sample(range(32),2)
        m[a],m[b] = m[b],m[a]

    return m


def encode_z(z,bc,bf,zmin,zmax):
    W = zmax-zmin
    fmax = (1 << bf)-1

    zc = np.clip(z,zmin,zmax)

    u = (bc+1)*(zc-zmin)/W
    endpoint = u >= bc+1

    c = np.floor(u).astype(np.int64)
    c = np.clip(c,0,bc)

    frac = u-c

    fine = np.rint(
        frac*fmax
    ).astype(np.int64)
    fine = np.clip(fine,0,fmax)

    c[endpoint] = bc
    fine[endpoint] = fmax

    thermo = np.asarray(
        [
            (1 << int(x))-1
            if x > 0
            else 0
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


def decode_words(words,bc,bf,zmin,zmax):
    W = zmax-zmin
    fmax = (1 << bf)-1
    cmask = (1 << bc)-1

    coarse = words & np.uint32(cmask)

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

    u = cc + fine/fmax

    z = (
        zmin
        +
        W/(bc+1)*u
    )

    return np.clip(z,zmin,zmax)


def physical_to_logical_masks(mapping):
    inv = invert_mapping(mapping)

    out = np.zeros(
        len(masks),
        dtype=np.uint32
    )

    for p in range(32):
        sel = (
            ((masks >> np.uint32(p)) & 1)
            .astype(bool)
        )

        out[sel] |= (
            np.uint32(1)
            <<
            np.uint32(inv[p])
        )

    return out


def true_risk(rep,mapping):
    bc,bf,zmin,zmax,W = codec_params(rep)

    z = z_by_entry[
        (rep["model"],rep["platform"])
    ]

    cw = encode_z(
        z,bc,bf,zmin,zmax
    )

    clean = decode_words(
        cw,bc,bf,zmin,zmax
    )

    lmasks = physical_to_logical_masks(
        mapping
    )

    total_exact = 0.0

    BATCH = 4096

    for start in range(
        0,len(lmasks),BATCH
    ):
        end = min(
            start+BATCH,
            len(lmasks)
        )

        fm = (
            lmasks[start:end,None]
            ^
            cw[None,:]
        )

        fd = decode_words(
            fm,
            bc,bf,zmin,zmax
        )

        A = np.abs(
            fd-clean[None,:]
        )

        maxA = np.max(A,axis=1)

        bound = np.minimum(
            W,
            hs[start:end].astype(float)
            *
            W/(bc+1)
        )

        if np.any(
            maxA > bound + 1e-10
        ):
            raise RuntimeError(
                "theorem violation"
            )

        total_exact += float(
            np.dot(
                np.mean(A,axis=1),
                counts[start:end]
            )
        )


    tail = sum(
        count
        *
        min(
            W,
            h*W/(bc+1)
        )
        for h,count in tail_counts.items()
    )

    return (
        total_exact + tail
    ) / TOTAL_EXPOSED


# ============================================================
# Deep confirmation
# ============================================================

candidate_rows = []
results = []
best_maps = {}

t0 = time.perf_counter()


for ri,rid in enumerate(selected_ids,1):
    rep = rep_by_id[rid]

    bc,bf,zmin,zmax,W = codec_params(rep)
    lw = logical_weights(bc,bf,W)

    identity = list(range(32))

    fast_selected = list(
        fast_map[rid]["mapping"]
    )

    h1_opt = optimal_additive_mapping(
        lw,
        h1_incidence
    )

    all_event_opt = optimal_additive_mapping(
        lw,
        physical_incidence
    )

    candidates = {
        tuple(identity): "IDENTITY",
        tuple(fast_selected): "FAST_SELECTED",
        tuple(h1_opt): "H1_REARRANGEMENT",
        tuple(all_event_opt): "ALL_EVENT_REARRANGEMENT",
    }


    bases = [
        fast_selected,
        h1_opt,
        all_event_opt,
    ]


    for n in range(STRUCTURED_CANDIDATES):
        base = bases[n % len(bases)]

        # 1..12 random logical swaps.
        swaps = 1 + (
            (n // len(bases))
            % 12
        )

        m = perturb(base,swaps)

        candidates.setdefault(
            tuple(m),
            f"PERTURB_{n:04d}_S{swaps}"
        )


    scored = []

    for mtuple,name in candidates.items():
        m = list(mtuple)

        scored.append({
            "name": name,
            "mapping": m,
            "surrogate":
                additive_score(
                    m,
                    lw,
                    physical_incidence
                ),
        })


    scored.sort(
        key=lambda x: x["surrogate"]
    )


    finalists = scored[:TRUE_FINALISTS]


    # Fast and identity must always be evaluated.
    for required_name,required_map in [
        ("FAST_SELECTED",fast_selected),
        ("IDENTITY",identity),
    ]:
        if not any(
            tuple(x["mapping"])
            ==
            tuple(required_map)
            for x in finalists
        ):
            finalists.append({
                "name": required_name,
                "mapping": required_map,
                "surrogate":
                    additive_score(
                        required_map,
                        lw,
                        physical_incidence
                    )
            })


    print(
        f"\n[N3B1C] REP {ri}/{len(selected_ids)} "
        f"{rid} {rep['model']} "
        f"Bc={bc} finalists={len(finalists)}",
        flush=True
    )


    tr = []

    for ci,c in enumerate(finalists,1):
        tc0 = time.perf_counter()

        risk = true_risk(
            rep,
            c["mapping"]
        )

        dt = time.perf_counter()-tc0

        print(
            f"[N3B1C] {rid} "
            f"TRUE {ci}/{len(finalists)} "
            f"{c['name']} "
            f"risk={risk:.12g} "
            f"dt={dt:.1f}s",
            flush=True
        )

        row = {
            "representation_id": rid,
            "model": rep["model"],
            "platform": rep["platform"],
            "Bc": bc,
            "Bf": bf,
            "candidate": c["name"],
            "surrogate_score":
                c["surrogate"],
            "absolute_true_hybrid_risk":
                risk,
        }

        tr.append(row)
        candidate_rows.append(row)


    fast_row = next(
        x for x in tr
        if x["candidate"] == "FAST_SELECTED"
    )

    identity_row = next(
        x for x in tr
        if x["candidate"] == "IDENTITY"
    )

    best = min(
        tr,
        key=lambda x:
            x["absolute_true_hybrid_risk"]
    )

    fast_risk = fast_row[
        "absolute_true_hybrid_risk"
    ]

    best_risk = best[
        "absolute_true_hybrid_risk"
    ]

    extra_vs_fast = (
        100.0
        *
        (fast_risk-best_risk)
        /
        fast_risk
    )

    gain_vs_identity = (
        100.0
        *
        (
            identity_row[
                "absolute_true_hybrid_risk"
            ]
            -
            best_risk
        )
        /
        identity_row[
            "absolute_true_hybrid_risk"
        ]
    )


    selected_map = next(
        x["mapping"]
        for x in finalists
        if x["name"] == best["candidate"]
    )


    best_maps[rid] = {
        "mapping_semantics":
            "mapping[logical]=physical",
        "mapping":
            selected_map,
        "candidate":
            best["candidate"],
        "fast_risk":
            fast_risk,
        "deep_best_risk":
            best_risk,
        "extra_improvement_vs_fast_percent":
            extra_vs_fast,
        "global_optimality_claim":
            False,
    }


    results.append({
        "representation_id": rid,
        "model": rep["model"],
        "platform": rep["platform"],
        "Bc": bc,
        "Bf": bf,
        "fast_true_risk":
            fast_risk,
        "deep_best_true_risk":
            best_risk,
        "extra_improvement_vs_fast_percent":
            extra_vs_fast,
        "deep_gain_vs_identity_percent":
            gain_vs_identity,
        "deep_selected_candidate":
            best["candidate"],
        "fast_remains_best":
            abs(
                fast_risk-best_risk
            ) <= 1e-15,
    })


    print(
        f"[N3B1C] DONE {rid} "
        f"extra_vs_fast={extra_vs_fast:.6f}% "
        f"gain_vs_identity={gain_vs_identity:.6f}% "
        f"elapsed={time.perf_counter()-t0:.1f}s",
        flush=True
    )


with CANDIDATE_OUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(candidate_rows[0].keys())
    )
    w.writeheader()
    w.writerows(candidate_rows)


with RESULT_OUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    w.writeheader()
    w.writerows(results)


MAP_OUT.write_text(
    json.dumps(
        best_maps,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


extra = np.asarray(
    [
        r["extra_improvement_vs_fast_percent"]
        for r in results
    ],
    dtype=float
)


summary = {
    "stage": "M5.5N3B1C",
    "confirmation_representations":
        len(results),
    "candidate_strategy":
        (
            "broader structured candidate search "
            "+ true-codec final selection"
        ),
    "extra_vs_fast_percent": {
        "min": float(extra.min()),
        "mean": float(extra.mean()),
        "median": float(np.median(extra)),
        "max": float(extra.max()),
    },
    "fast_exact_best_count":
        sum(r["fast_remains_best"] for r in results),
    "global_optimality_claim": False,
    "elapsed_sec":
        time.perf_counter()-t0,
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

if len(results) != 8:
    critical.append(
        "CONFIRMATION_COUNT_NE_8"
    )

if np.any(extra < -1e-10):
    critical.append(
        "DEEP_SEARCH_WORSE_THAN_FAST"
    )


lines = [
    "M5.5N3B1C DDR DEEP MAPPING CONFIRMATION",
    f"CONFIRMATION_REPRESENTATIONS={len(results)}",
    (
        "EXTRA_VS_FAST_MIN_PERCENT="
        f"{float(extra.min())}"
    ),
    (
        "EXTRA_VS_FAST_MEAN_PERCENT="
        f"{float(extra.mean())}"
    ),
    (
        "EXTRA_VS_FAST_MEDIAN_PERCENT="
        f"{float(np.median(extra))}"
    ),
    (
        "EXTRA_VS_FAST_MAX_PERCENT="
        f"{float(extra.max())}"
    ),
    (
        "FAST_EXACT_BEST_COUNT="
        f"{sum(r['fast_remains_best'] for r in results)}"
    ),
    "FINAL_SELECTION_USES_TRUE_CODEC=True",
    "FAST_MAPPING_ALWAYS_INCLUDED=True",
    "GLOBAL_OPTIMALITY_CLAIM=False",
    "REPRESENTATION_FROZEN=True",
    "RAW_N2_FILES_MODIFIED=0",
    "RAW_N3A_FILES_MODIFIED=0",
    f"CRITICAL_ISSUES_COUNT={len(critical)}",
    f"M55N3B1C_FINAL_PASS={len(critical)==0}",
]

for c in critical:
    lines.append(f"CRITICAL={c}")


STATUS.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)


print()
print("="*110)

for x in lines:
    print(x)

print()
print("SELECTED =", SELECTED)
print("RESULT =", RESULT_OUT)
print("CANDIDATES =", CANDIDATE_OUT)
print("MAPPINGS =", MAP_OUT)
print("SUMMARY =", SUMMARY)
print("STATUS =", STATUS)
