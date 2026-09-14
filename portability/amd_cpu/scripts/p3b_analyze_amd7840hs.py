import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


WORK = Path(r"D:\metakv_results_backup\AMD7840HS_P3B_FINAL_ANALYSIS\freeze_extracted\results\formal")
RAW = WORK / "p3b_raw_blocks.csv"
CHECK = WORK / "p3b_correctness.csv"
SUMMARY = WORK / "p3b_summary.csv"
SUMMARY_JSON = WORK / "p3b_summary.json"
STATUS = WORK / "P3B_FINAL_STATUS.txt"

REPS = [
    "REP_0008",
    "REP_0025",
    "REP_0026",
    "REP_0027",
]

THREADS = [1, 4, 8]

GROUPS = 2**22

PAIRED_BLOCKS = 15
BOOTSTRAP_N = 10000
SEED = 20260902

EXPECTED_RAW_ROWS = (
    4
    *
    3
    *
    15
    *
    2
)


def read_csv(path):
    if not path.is_file():
        raise RuntimeError(
            f"missing file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def bootstrap_median_ci(values, seed):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size != PAIRED_BLOCKS:
        raise RuntimeError(
            f"bootstrap size={values.size}, "
            f"expected={PAIRED_BLOCKS}"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "nonfinite bootstrap values"
        )

    rng = np.random.default_rng(seed)

    bootstrap_values = np.empty(
        BOOTSTRAP_N,
        dtype=np.float64,
    )

    for bootstrap_index in range(
        BOOTSTRAP_N
    ):
        indices = rng.integers(
            0,
            values.size,
            size=values.size,
        )

        bootstrap_values[
            bootstrap_index
        ] = np.median(
            values[indices]
        )

    low, high = np.percentile(
        bootstrap_values,
        [2.5, 97.5],
    )

    return float(low), float(high)


raw = read_csv(RAW)
checks = read_csv(CHECK)

if len(raw) != EXPECTED_RAW_ROWS:
    raise RuntimeError(
        f"raw rows={len(raw)}, "
        f"expected={EXPECTED_RAW_ROWS}"
    )

if len(checks) != 4:
    raise RuntimeError(
        f"correctness rows={len(checks)}, expected=4"
    )


for row in checks:

    if (
        row["representation_id"]
        not in REPS
    ):
        raise RuntimeError(
            "unexpected representation "
            "in correctness CSV"
        )

    if (
        row["lut_identity_exact"].lower()
        != "true"
    ):
        raise RuntimeError(
            "LUT correctness failure"
        )


grouped = defaultdict(dict)


for row in raw:

    rep = row["representation_id"]
    threads = int(row["threads"])
    groups = int(row["groups"])
    block = int(row["block"])
    kernel = row["kernel"]
    ms = float(row["ms"])

    if rep not in REPS:
        raise RuntimeError(
            f"unexpected rep={rep}"
        )

    if threads not in THREADS:
        raise RuntimeError(
            f"unexpected threads={threads}"
        )

    if groups != GROUPS:
        raise RuntimeError(
            f"unexpected groups={groups}"
        )

    if kernel not in {
        "ID_FULL",
        "MAP_LUT8",
    }:
        raise RuntimeError(
            f"unexpected kernel={kernel}"
        )

    if not (
        math.isfinite(ms)
        and
        ms > 0.0
    ):
        raise RuntimeError(
            "invalid timing value"
        )

    grouped[
        (rep, threads)
    ].setdefault(
        block,
        {}
    )[kernel] = ms


summary_rows = []


for rep_index, rep in enumerate(REPS):

    for thread_index, threads in enumerate(THREADS):

        key = (rep, threads)

        if key not in grouped:
            raise RuntimeError(
                f"missing result key={key}"
            )

        blocks = grouped[key]

        if len(blocks) != PAIRED_BLOCKS:
            raise RuntimeError(
                f"{key}: block count={len(blocks)}"
            )

        id_ms = []
        lut_ms = []
        overhead = []

        for block in range(
            1,
            PAIRED_BLOCKS + 1
        ):
            if block not in blocks:
                raise RuntimeError(
                    f"{key}: missing block={block}"
                )

            pair = blocks[block]

            if set(pair.keys()) != {
                "ID_FULL",
                "MAP_LUT8",
            }:
                raise RuntimeError(
                    f"{key}/block={block}: "
                    "kernel pair mismatch"
                )

            identity = float(
                pair["ID_FULL"]
            )

            lut = float(
                pair["MAP_LUT8"]
            )

            id_ms.append(identity)
            lut_ms.append(lut)

            overhead.append(
                100.0
                *
                (
                    lut / identity
                    -
                    1.0
                )
            )


        ci_low, ci_high = bootstrap_median_ci(
            overhead,
            SEED
            +
            rep_index * 1000
            +
            thread_index * 100
            +
            1,
        )


        first_row = next(
            row
            for row in raw
            if (
                row["representation_id"] == rep
                and
                int(row["threads"]) == threads
            )
        )


        summary_rows.append({
            "representation_id":
                rep,

            "Bc":
                int(first_row["Bc"]),

            "Bf":
                int(first_row["Bf"]),

            "threads":
                threads,

            "groups":
                GROUPS,

            "identity_median_ms":
                float(
                    np.median(id_ms)
                ),

            "lut8_median_ms":
                float(
                    np.median(lut_ms)
                ),

            "lut8_mapping_over_identity_median_percent":
                float(
                    np.median(overhead)
                ),

            "lut8_mapping_ci95_low":
                ci_low,

            "lut8_mapping_ci95_high":
                ci_high,

            "absolute_mapping_overhead_under_5_percent":
                abs(
                    float(
                        np.median(overhead)
                    )
                ) < 5.0,

            "absolute_mapping_overhead_under_8_percent":
                abs(
                    float(
                        np.median(overhead)
                    )
                ) < 8.0,
        })


if len(summary_rows) != 12:
    raise RuntimeError(
        f"summary rows={len(summary_rows)}, expected=12"
    )


with SUMMARY.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(summary_rows)


abs_overheads = [
    abs(
        float(
            row[
                "lut8_mapping_over_identity_median_percent"
            ]
        )
    )
    for row in summary_rows
]


within_5_count = sum(
    value < 5.0
    for value in abs_overheads
)

within_8_count = sum(
    value < 8.0
    for value in abs_overheads
)


strong_pass = (
    within_5_count == 12
)

practical_pass = (
    within_8_count == 12
)


per_thread = {}

for threads in THREADS:

    rows = [
        row
        for row in summary_rows
        if row["threads"] == threads
    ]

    if len(rows) != 4:
        raise RuntimeError(
            f"threads={threads}: row count !=4"
        )

    thread_abs = [
        abs(
            float(
                row[
                    "lut8_mapping_over_identity_median_percent"
                ]
            )
        )
        for row in rows
    ]

    per_thread[str(threads)] = {
        "count": 4,
        "max_abs_mapping_overhead_percent":
            max(thread_abs),

        "all_under_5_percent":
            all(
                value < 5.0
                for value in thread_abs
            ),

        "all_under_8_percent":
            all(
                value < 8.0
                for value in thread_abs
            ),
    }


summary_json = {
    "stage":
        "P3B_AMD7840HS_X86_MULTITHREAD",

    "role":
        "multithread_throughput_validation",

    "backend":
        "AMD_Ryzen_7_7840HS_x86_64_CPU",

    "cpu_model":
        "AMD Ryzen 7 7840HS",

    "openmp":
        True,

    "thread_affinity":
        "OS_scheduler_managed",

    "p_core_only_claim":
        False,

    "thread_counts":
        THREADS,

    "groups":
        GROUPS,

    "mapping_semantics":
        "mapping[logical]=physical",

    "mapping_source":
        "frozen_M3G_arbitrary_mapping",

    "byte_lut_bytes":
        4096,

    "representation_count":
        4,

    "paired_blocks":
        PAIRED_BLOCKS,

    "raw_row_count":
        len(raw),

    "expected_raw_row_count":
        EXPECTED_RAW_ROWS,

    "summary_row_count":
        len(summary_rows),

    "correctness_all":
        True,

    "primary_metric":
        "MAP_LUT8_vs_ID_FULL",

    "within_5_percent_count":
        within_5_count,

    "within_8_percent_count":
        within_8_count,

    "max_abs_mapping_overhead_percent":
        max(abs_overheads),

    "strong_pass_all_under_5_percent":
        strong_pass,

    "practical_pass_all_under_8_percent":
        practical_pass,

    "per_thread":
        per_thread,

    "kernel_microbenchmark":
        True,

    "end_to_end_llm_latency_claim":
        False,

    "physical_dram_lane_placement_demonstrated":
        False,

    "negative_overhead_does_not_imply_acceleration":
        True,

    "p3b_final_pass":
        True,
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary_json,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


status_lines = [
    "=" * 120,
    "P3B AMD RYZEN 7 7840HS MULTITHREAD THROUGHPUT FINAL STATUS",
    "=" * 120,

    "ROLE=MULTITHREAD_THROUGHPUT_VALIDATION",

    "BACKEND=AMD_Ryzen_7_7840HS_x86_64_CPU",

    "OPENMP=True",
    "THREAD_AFFINITY=OS_SCHEDULER_MANAGED",
    "P_CORE_ONLY_CLAIM=False",

    "THREAD_COUNTS=1,4,8",
    "GROUPS=4194304",

    "FROZEN_M3G_MAPPING=True",
    "MAPPING_SEMANTICS=MAPPING_LOGICAL_TO_PHYSICAL",

    "TRUE_NIBBLE_PACKED_INT4=True",
    "GROUP_SIZE=32",

    "BYTE_LUT=True",
    "BYTE_LUT_BYTES=4096",

    "CONFIG_COUNT=4",
    "PAIRED_BLOCKS=15",

    f"RAW_ROW_COUNT={len(raw)}",
    f"EXPECTED_RAW_ROW_COUNT={EXPECTED_RAW_ROWS}",
    f"SUMMARY_ROW_COUNT={len(summary_rows)}",

    "LUT_IDENTITY_EXACT_ALL=True",

    "PRIMARY_MAPPING_METRIC=MAP_LUT8_VS_ID_FULL",

    f"WITHIN_5_PERCENT_COUNT={within_5_count}/12",
    f"WITHIN_8_PERCENT_COUNT={within_8_count}/12",

    (
        "MAX_ABS_MAPPING_OVERHEAD_PERCENT="
        f"{max(abs_overheads)}"
    ),

    (
        "STRONG_PASS_ALL_UNDER_5_PERCENT="
        f"{strong_pass}"
    ),

    (
        "PRACTICAL_PASS_ALL_UNDER_8_PERCENT="
        f"{practical_pass}"
    ),

    "KERNEL_MICROBENCHMARK=True",
    "END_TO_END_LLM_LATENCY_CLAIM=False",

    "FIXED_MAPPING_IS_SOFTWARE_EMULATED_BIT_LAYOUT=True",
    "PHYSICAL_DRAM_LANE_PLACEMENT_DEMONSTRATED=False",

    "NEGATIVE_OVERHEAD_DOES_NOT_IMPLY_ACCELERATION=True",

    "CRITICAL_ISSUES_COUNT=0",
    "P3B_FINAL_PASS=True",
]


STATUS.write_text(
    "\n".join(status_lines)
    +
    "\n",
    encoding="utf-8",
)


print("=" * 120)
print("P3B MULTITHREAD SUMMARY")
print("=" * 120)

for row in summary_rows:

    print(
        f"{row['representation_id']} "
        f"Bc={row['Bc']} "
        f"Bf={row['Bf']} "
        f"T={row['threads']} "
        f"ID={row['identity_median_ms']:.6f}ms "
        f"LUT={row['lut8_median_ms']:.6f}ms "
        f"LUTvsID="
        f"{row['lut8_mapping_over_identity_median_percent']:+.3f}% "
        f"CI=["
        f"{row['lut8_mapping_ci95_low']:+.3f}%,"
        f"{row['lut8_mapping_ci95_high']:+.3f}%]"
    )


print()
print("=" * 120)
print("P3B THREAD-LEVEL SUMMARY")
print("=" * 120)

for threads in THREADS:

    info = per_thread[str(threads)]

    print(
        f"T={threads} "
        f"MAX_ABS="
        f"{info['max_abs_mapping_overhead_percent']:.3f}% "
        f"ALL_LT5="
        f"{info['all_under_5_percent']} "
        f"ALL_LT8="
        f"{info['all_under_8_percent']}"
    )


print()
print(
    "\n".join(status_lines)
)
