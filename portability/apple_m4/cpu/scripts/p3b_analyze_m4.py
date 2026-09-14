import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPS = ["REP_0008", "REP_0025", "REP_0026", "REP_0027"]
THREADS = [1, 4, 8]
GROUPS = 2 ** 22
PAIRED_BLOCKS = 15
BOOTSTRAP_N = 10000
BASE_SEED = 20260902


def read_csv(path):
    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def finite_float(value, name):
    x = float(value)
    if not math.isfinite(x):
        raise RuntimeError(f"non-finite {name}")
    return x


def bootstrap_median_ci(values, seed):
    values = np.asarray(values, dtype=np.float64)

    if values.shape != (PAIRED_BLOCKS,):
        raise RuntimeError(
            f"bad bootstrap shape={values.shape}"
        )

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        PAIRED_BLOCKS,
        size=(BOOTSTRAP_N, PAIRED_BLOCKS),
    )

    medians = np.median(values[idx], axis=1)

    low, high = np.percentile(
        medians,
        [2.5, 97.5],
    )

    return float(low), float(high)


def main():
    if len(sys.argv) != 4:
        raise RuntimeError(
            "usage: analyzer raw.csv correctness.csv output_dir"
        )

    raw_path = Path(sys.argv[1])
    correct_path = Path(sys.argv[2])
    out = Path(sys.argv[3])

    out.mkdir(parents=True, exist_ok=True)

    raw = read_csv(raw_path)
    correct = read_csv(correct_path)

    if len(raw) != 360:
        raise RuntimeError(
            f"raw rows={len(raw)}, expected=360"
        )

    if len(correct) != 4:
        raise RuntimeError(
            f"correctness rows={len(correct)}, expected=4"
        )

    correct_reps = set()

    for row in correct:
        rep = row["representation_id"]

        if rep not in REPS:
            raise RuntimeError(
                f"unexpected correctness rep={rep}"
            )

        if rep in correct_reps:
            raise RuntimeError(
                f"duplicate correctness rep={rep}"
            )

        correct_reps.add(rep)

        if row["lut_identity_exact"].lower() != "true":
            raise RuntimeError(
                f"correctness failure: {rep}"
            )

        if int(row["groups"]) != GROUPS:
            raise RuntimeError(
                f"correctness groups mismatch: {rep}"
            )

    if correct_reps != set(REPS):
        raise RuntimeError(
            "correctness representation set mismatch"
        )

    keyed = {}

    for row in raw:
        rep = row["representation_id"]
        threads = int(row["threads"])
        groups = int(row["groups"])
        block = int(row["block"])
        kernel = row["kernel"]
        ms = finite_float(row["ms"], "ms")

        if rep not in REPS:
            raise RuntimeError(f"unexpected rep={rep}")

        if threads not in THREADS:
            raise RuntimeError(
                f"unexpected threads={threads}"
            )

        if groups != GROUPS:
            raise RuntimeError(
                f"unexpected groups={groups}"
            )

        if not 1 <= block <= PAIRED_BLOCKS:
            raise RuntimeError(
                f"unexpected block={block}"
            )

        if kernel not in {"ID_FULL", "MAP_LUT8"}:
            raise RuntimeError(
                f"unexpected kernel={kernel}"
            )

        if ms <= 0:
            raise RuntimeError(
                f"non-positive timing={ms}"
            )

        key = (rep, threads, block, kernel)

        if key in keyed:
            raise RuntimeError(
                f"duplicate raw key={key}"
            )

        keyed[key] = ms

    if len(keyed) != 360:
        raise RuntimeError(
            f"unique keys={len(keyed)}, expected=360"
        )

    summary = []

    combo_index = 0

    for rep in REPS:
        for threads in THREADS:
            identity = []
            lut8 = []
            overheads = []

            for block in range(1, PAIRED_BLOCKS + 1):
                id_ms = keyed[
                    (rep, threads, block, "ID_FULL")
                ]

                lut_ms = keyed[
                    (rep, threads, block, "MAP_LUT8")
                ]

                overhead = 100.0 * (
                    lut_ms / id_ms - 1.0
                )

                if not math.isfinite(overhead):
                    raise RuntimeError(
                        "non-finite paired overhead"
                    )

                identity.append(id_ms)
                lut8.append(lut_ms)
                overheads.append(overhead)

            median = float(
                np.median(
                    np.asarray(
                        overheads,
                        dtype=np.float64,
                    )
                )
            )

            seed = BASE_SEED + combo_index
            combo_index += 1

            ci_low, ci_high = bootstrap_median_ci(
                overheads,
                seed,
            )

            row = {
                "representation_id": rep,
                "threads": threads,
                "groups": GROUPS,
                "paired_blocks": PAIRED_BLOCKS,
                "identity_median_ms": float(
                    np.median(identity)
                ),
                "lut8_median_ms": float(
                    np.median(lut8)
                ),
                "lut8_mapping_over_identity_median_percent":
                    median,
                "lut8_mapping_ci95_low":
                    ci_low,
                "lut8_mapping_ci95_high":
                    ci_high,
                "bootstrap_n":
                    BOOTSTRAP_N,
                "bootstrap_seed":
                    seed,
            }

            summary.append(row)

    if len(summary) != 12:
        raise RuntimeError(
            f"summary rows={len(summary)}, expected=12"
        )

    abs_values = [
        abs(
            row[
                "lut8_mapping_over_identity_median_percent"
            ]
        )
        for row in summary
    ]

    within5 = sum(v < 5.0 for v in abs_values)
    within8 = sum(v < 8.0 for v in abs_values)
    max_abs = max(abs_values)

    max_row = max(
        summary,
        key=lambda row: abs(
            row[
                "lut8_mapping_over_identity_median_percent"
            ]
        ),
    )

    csv_path = out / "p3b_m4_summary.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    json_path = out / "p3b_m4_summary.json"

    payload = {
        "platform": "Apple M4 ARM64",
        "role":
            "cross_isa_portability_replication",
        "groups": GROUPS,
        "threads": THREADS,
        "paired_blocks": PAIRED_BLOCKS,
        "bootstrap_n": BOOTSTRAP_N,
        "base_seed": BASE_SEED,
        "configuration_count": 12,
        "within_5_percent_count": within5,
        "within_8_percent_count": within8,
        "max_abs_median_mapping_overhead_percent":
            max_abs,
        "max_abs_configuration": {
            "representation_id":
                max_row["representation_id"],
            "threads":
                max_row["threads"],
            "median_percent":
                max_row[
                    "lut8_mapping_over_identity_median_percent"
                ],
            "ci95_low":
                max_row["lut8_mapping_ci95_low"],
            "ci95_high":
                max_row["lut8_mapping_ci95_high"],
        },
        "analyzer_file_identical_to_x86": False,
        "analyzer_statistical_method_equivalent": True,
        "kernel_microbenchmark": True,
        "end_to_end_llm_latency_claim": False,
        "results": summary,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    status = out / "M4_P3B_BOOTSTRAP_FINAL_STATUS.txt"

    status_lines = [
        "METAKV APPLE M4 ARM64 P3B FINAL",
        "PLATFORM=Apple M4 ARM64",
        "ROLE=CROSS_ISA_PORTABILITY_REPLICATION",
        f"GROUPS={GROUPS}",
        "THREADS=1,4,8",
        f"PAIRED_BLOCKS={PAIRED_BLOCKS}",
        f"BOOTSTRAP_N={BOOTSTRAP_N}",
        f"BOOTSTRAP_BASE_SEED={BASE_SEED}",
        "PRIMARY_METRIC=MAP_LUT8_VS_ID_FULL",
        "ANALYZER_FILE_IDENTICAL_TO_X86=False",
        "ANALYZER_STATISTICAL_METHOD_EQUIVALENT=True",
        f"WITHIN_5_PERCENT_COUNT={within5}/12",
        f"WITHIN_8_PERCENT_COUNT={within8}/12",
        (
            "MAX_ABS_MAPPING_OVERHEAD_PERCENT="
            f"{max_abs:.9f}"
        ),
        (
            "STRONG_ALL_UNDER_5_PERCENT_PASS="
            f"{within5 == 12}"
        ),
        (
            "PRACTICAL_ALL_UNDER_8_PERCENT_PASS="
            f"{within8 == 12}"
        ),
        "KERNEL_MICROBENCHMARK=True",
        "END_TO_END_LLM_LATENCY_CLAIM=False",
        "M4_P3B_BOOTSTRAP_FINAL_PASS=True",
    ]

    status.write_text(
        "\n".join(status_lines) + "\n",
        encoding="utf-8",
    )

    print("APPLE_M4_ARM64_BOOTSTRAP_RESULTS")

    for row in summary:
        print(
            f"{row['representation_id']} "
            f"T={row['threads']} "
            f"LUTvsID="
            f"{row['lut8_mapping_over_identity_median_percent']:+.3f}% "
            f"CI=["
            f"{row['lut8_mapping_ci95_low']:+.3f}%,"
            f"{row['lut8_mapping_ci95_high']:+.3f}%]"
        )

    print("M4_BOOTSTRAP_FINAL")
    print(
        f"WITHIN_5_PERCENT_COUNT={within5}/12"
    )
    print(
        f"WITHIN_8_PERCENT_COUNT={within8}/12"
    )
    print(
        "MAX_ABS_MAPPING_OVERHEAD_PERCENT="
        f"{max_abs:.6f}"
    )
    print(
        "MAX_ABS_CONFIGURATION="
        f"{max_row['representation_id']} "
        f"T={max_row['threads']}"
    )
    print(
        "ANALYZER_STATISTICAL_METHOD_EQUIVALENT=True"
    )
    print(
        "M4_P3B_BOOTSTRAP_FINAL_PASS=True"
    )


if __name__ == "__main__":
    main()
