from __future__ import annotations

from pathlib import Path
import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import struct
import time
from collections import defaultdict

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# ======================================================================================
# P8 FORMAL V1 — FROZEN PROTOCOL
# ======================================================================================

PROTOCOL = "P8_FORMAL_SHORT_HORIZON_V1"

GROUP_SIZE = 32

MAX_HORIZON = 32
CHECKPOINT_HORIZONS = (1, 8, 16, 32)

MODES = (
    "TEACHER_FORCED",
    "FREE_RUNNING",
)

ENCODINGS = (
    ("FP32", "fp32_bit"),
    ("BINARY32_Z", "binary32_bit"),
    ("HA_FBMS", "ha_fbms_bit"),
)

EXPECTED_PROMPTS = 8
EXPECTED_TARGETS = 24

TARGET_RESELECTION = False
FROZEN_LOCATION_TRANSFER = True
AMD_LOCAL_CLEAN_METADATA_RECONSTRUCTION = True
FAULT_INJECTION_ONCE_AT_INITIAL_CACHE = True

GREEDY_DECODING = True
DO_SAMPLE = False

POST_DIVERGENCE_FREE_RUNNING_KL_IS_PURE_FAULT_EFFECT = False

SEED = 20260904

# Exact P7D1 representation constants.
ZMIN = -13.0
ZMAX = 5.0

BC = 19
BF = 13

FMAX = (1 << BF) - 1
COARSE_MASK = (1 << BC) - 1
BINARY_MAX = (1 << 32) - 1

assert BC + BF == 32


# ======================================================================================
# PATHS
# ======================================================================================

def require_path(env_name: str) -> Path:
    value = os.environ.get(env_name)

    if not value:
        raise RuntimeError(
            f"{env_name} is required"
        )

    path = Path(value)

    if not path.exists():
        raise FileNotFoundError(
            f"{env_name}={path}"
        )

    return path


MODEL_PATH = require_path(
    "METAKV_MODEL_PATH"
)

PROMPT_SOURCE = require_path(
    "METAKV_SOURCE_SCRIPT"
)

MANIFEST_PATH = require_path(
    "METAKV_MANIFEST"
)

P7D1_SOURCE = require_path(
    "METAKV_P7D1_SOURCE"
)

OUTROOT = Path(
    os.environ.get(
        "METAKV_OUTROOT",
        "${WORKSPACE}/metakv_overhead_diagnostic/p8_formal_v1",
    )
)

OUTROOT.mkdir(
    parents=True,
    exist_ok=True,
)

ROWS_PATH = (
    OUTROOT /
    "p8_formal_rows.csv"
)

CASES_PATH = (
    OUTROOT /
    "p8_formal_cases.jsonl"
)

SUMMARY_PATH = (
    OUTROOT /
    "P8_FORMAL_SUMMARY.json"
)

STATUS_PATH = (
    OUTROOT /
    "P8_FORMAL_STATUS.txt"
)

PROTOCOL_PATH = (
    OUTROOT /
    "P8_FORMAL_PROTOCOL.json"
)

PROVENANCE_PATH = (
    OUTROOT /
    "P8_FORMAL_PROVENANCE.txt"
)

PROGRESS_PATH = (
    OUTROOT /
    "P8_FORMAL_PROGRESS.json"
)


# ======================================================================================
# UTILITIES
# ======================================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def json_safe_number(x):
    if x is None:
        return None

    try:
        x = float(x)
    except Exception:
        return str(x)

    if math.isfinite(x):
        return x

    return str(x)


def median_or_none(values):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if not values:
        return None

    return float(
        statistics.median(values)
    )


def mean_or_none(values):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if not values:
        return None

    return float(
        statistics.fmean(values)
    )


# ======================================================================================
# FROZEN PROMPT RECOVERY
# ======================================================================================

def recover_frozen_prompts(source: Path):
    text = source.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(text)

    candidates = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        if isinstance(
            node,
            ast.Assign,
        ):

            names = [
                target.id
                for target in node.targets
                if isinstance(
                    target,
                    ast.Name,
                )
            ]

            value = node.value

        else:

            names = (
                [node.target.id]
                if isinstance(
                    node.target,
                    ast.Name,
                )
                else []
            )

            value = node.value

        for name in names:

            if "prompt" not in name.lower():
                continue

            try:
                obj = ast.literal_eval(
                    value
                )
            except Exception:
                continue

            if (
                isinstance(
                    obj,
                    (list, tuple),
                )
                and
                len(obj) >= EXPECTED_PROMPTS
                and
                all(
                    isinstance(x, str)
                    for x in obj
                )
            ):

                candidates.append(
                    (
                        name,
                        list(obj),
                    )
                )

    if not candidates:
        raise RuntimeError(
            "unable to recover frozen P4 prompts"
        )

    candidates.sort(
        key=lambda x: (
            abs(
                len(x[1])
                -
                EXPECTED_PROMPTS
            ),
            x[0],
        )
    )

    prompt_variable, prompts = (
        candidates[0]
    )

    prompts = prompts[
        :EXPECTED_PROMPTS
    ]

    if len(prompts) != EXPECTED_PROMPTS:
        raise RuntimeError(
            "expected exactly 8 frozen prompts"
        )

    return (
        prompt_variable,
        prompts,
    )


PROMPT_VARIABLE, PROMPTS = (
    recover_frozen_prompts(
        PROMPT_SOURCE
    )
)


# ======================================================================================
# MANIFEST
# ======================================================================================

with MANIFEST_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    MANIFEST = list(
        csv.DictReader(f)
    )


if len(MANIFEST) != EXPECTED_TARGETS:
    raise RuntimeError(
        f"expected {EXPECTED_TARGETS} "
        f"targets, got {len(MANIFEST)}"
    )


manifest_prompt_ids = sorted(
    {
        int(row["prompt_id"])
        for row in MANIFEST
    }
)

if manifest_prompt_ids != list(
    range(EXPECTED_PROMPTS)
):
    raise RuntimeError(
        "manifest prompt coverage mismatch: "
        f"{manifest_prompt_ids}"
    )


# ======================================================================================
# LOAD EXACT P7D1 FUNCTION DEFINITIONS
# ======================================================================================

def load_p7d1_functions(
    source: Path,
):
    text = source.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(text)

    namespace = {
        "__name__":
            "_metakv_p7d1_helpers",

        "__file__":
            str(source),

        "Path":
            Path,

        "ast":
            ast,

        "csv":
            csv,

        "gc":
            gc,

        "json":
            json,

        "math":
            math,

        "os":
            os,

        "struct":
            struct,

        "np":
            np,

        "torch":
            torch,

        "DynamicCache":
            DynamicCache,

        "GROUP_SIZE":
            GROUP_SIZE,

        "ZMIN":
            ZMIN,

        "ZMAX":
            ZMAX,

        "BC":
            BC,

        "BF":
            BF,

        "FMAX":
            FMAX,

        "COARSE_MASK":
            COARSE_MASK,

        "BINARY_MAX":
            BINARY_MAX,
    }

    function_nodes = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    ]

    module = ast.Module(
        body=function_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    exec(
        compile(
            module,
            str(source),
            "exec",
        ),
        namespace,
        namespace,
    )

    required = [
        "cache_to_legacy",
        "legacy_to_dynamic",
        "clone_legacy",
        "quantize_group",
        "build_raw_full_cache",
        "fp32_codec",
        "binary32_codec",
        "ha_codec",
        "amplification",
    ]

    missing = [
        name
        for name in required
        if name not in namespace
    ]

    if missing:
        raise RuntimeError(
            "missing authoritative "
            f"P7D1 helpers: {missing}"
        )

    return namespace


P7 = load_p7d1_functions(
    P7D1_SOURCE
)

cache_to_legacy = (
    P7["cache_to_legacy"]
)

legacy_to_dynamic = (
    P7["legacy_to_dynamic"]
)

clone_legacy = (
    P7["clone_legacy"]
)

quantize_group = (
    P7["quantize_group"]
)

build_raw_full_cache = (
    P7["build_raw_full_cache"]
)

fp32_codec = (
    P7["fp32_codec"]
)

binary32_codec = (
    P7["binary32_codec"]
)

ha_codec = (
    P7["ha_codec"]
)

amplification = (
    P7["amplification"]
)


# ======================================================================================
# CACHE PATCH / READBACK
# ======================================================================================

def patch_group(
    raw_legacy,
    layer,
    kv_idx,
    start,
    end,
    values,
):

    pairs = list(
        raw_legacy
    )

    pair = list(
        pairs[layer]
    )

    patched = (
        pair[kv_idx]
        .detach()
        .clone()
        .contiguous()
    )

    patched.view(-1)[
        start:end
    ] = values

    pair[kv_idx] = patched

    pairs[layer] = tuple(
        pair
    )

    return tuple(
        pairs
    )


def readback_group(
    legacy,
    layer,
    kv_idx,
    start,
    end,
):

    cache = legacy_to_dynamic(
        clone_legacy(
            legacy
        )
    )

    got = (
        cache_to_legacy(
            cache
        )[layer][kv_idx]
        .contiguous()
        .view(-1)[
            start:end
        ]
        .detach()
    )

    del cache

    return got


# ======================================================================================
# LOGIT METRICS
# ======================================================================================

def compare_logits(
    baseline_logits,
    fault_logits,
):

    a = baseline_logits.float()
    b = fault_logits.float()

    baseline_token = int(
        torch.argmax(
            a,
            dim=-1,
        ).item()
    )

    fault_token = int(
        torch.argmax(
            b,
            dim=-1,
        ).item()
    )

    finite_a = bool(
        torch.isfinite(
            a
        ).all().item()
    )

    finite_b = bool(
        torch.isfinite(
            b
        ).all().item()
    )

    if finite_a and finite_b:

        log_pa = torch.log_softmax(
            a,
            dim=-1,
        )

        log_pb = torch.log_softmax(
            b,
            dim=-1,
        )

        pa = torch.softmax(
            a,
            dim=-1,
        )

        kl = torch.sum(
            pa
            *
            (
                log_pa
                -
                log_pb
            ),
            dim=-1,
        ).mean()

        diff = (
            a
            -
            b
        ).abs()

        kl_value = float(
            kl.item()
        )

        max_diff = float(
            diff.max().item()
        )

        mean_diff = float(
            diff.mean().item()
        )

    else:

        kl_value = None
        max_diff = None
        mean_diff = None

    return {
        "baseline_token_id":
            baseline_token,

        "fault_token_id":
            fault_token,

        "token_same":
            bool(
                baseline_token
                ==
                fault_token
            ),

        "kl":
            kl_value,

        "max_logit_diff":
            max_diff,

        "mean_logit_diff":
            mean_diff,

        "baseline_logits_finite":
            finite_a,

        "fault_logits_finite":
            finite_b,
    }


# ======================================================================================
# CLEAN 32-STEP TRAJECTORY
# ======================================================================================

def run_clean_trajectory(
    model,
    clean_legacy,
    initial_input_token,
    context_len,
):

    cache = legacy_to_dynamic(
        clone_legacy(
            clean_legacy
        )
    )

    input_token = (
        initial_input_token
        .detach()
        .clone()
    )

    trajectory = []

    for step in range(
        1,
        MAX_HORIZON + 1,
    ):

        position = (
            context_len
            +
            step
            -
            1
        )

        position_ids = torch.tensor(
            [[position]],
            dtype=torch.long,
            device=model.device,
        )

        cache_position = torch.tensor(
            [position],
            dtype=torch.long,
            device=model.device,
        )

        with torch.inference_mode():

            out = model(
                input_ids=input_token,
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )

        cache = (
            out.past_key_values
        )

        logits = (
            out.logits[
                :,
                -1,
                :
            ]
            .detach()
        )

        token = int(
            torch.argmax(
                logits,
                dim=-1,
            ).item()
        )

        trajectory.append({
            "step":
                step,

            "position":
                position,

            "input_token_id":
                int(
                    input_token.item()
                ),

            "output_token_id":
                token,

            "logits":
                logits,
        })

        input_token = torch.tensor(
            [[token]],
            dtype=torch.long,
            device=model.device,
        )

        del out

    del cache

    return trajectory


# ======================================================================================
# FAULT TRAJECTORY
# ======================================================================================

def run_fault_trajectory(
    model,
    fault_legacy,
    initial_input_token,
    context_len,
    clean_trajectory,
    mode,
):

    cache = legacy_to_dynamic(
        clone_legacy(
            fault_legacy
        )
    )

    input_token = (
        initial_input_token
        .detach()
        .clone()
    )

    rows = []

    first_divergence = None

    for step in range(
        1,
        MAX_HORIZON + 1,
    ):

        clean_step = (
            clean_trajectory[
                step - 1
            ]
        )

        position = (
            context_len
            +
            step
            -
            1
        )

        position_ids = torch.tensor(
            [[position]],
            dtype=torch.long,
            device=model.device,
        )

        cache_position = torch.tensor(
            [position],
            dtype=torch.long,
            device=model.device,
        )

        with torch.inference_mode():

            out = model(
                input_ids=input_token,
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )

        cache = (
            out.past_key_values
        )

        fault_logits = (
            out.logits[
                :,
                -1,
                :
            ]
            .detach()
        )

        metrics = compare_logits(
            clean_step["logits"],
            fault_logits,
        )

        if (
            first_divergence
            is None
            and
            not metrics["token_same"]
        ):
            first_divergence = step

        post_divergence = bool(
            first_divergence
            is not None
            and
            step
            >
            first_divergence
        )

        rows.append({
            "step":
                step,

            "position":
                position,

            "clean_input_token_id":
                clean_step[
                    "input_token_id"
                ],

            "fault_input_token_id":
                int(
                    input_token.item()
                ),

            **metrics,

            "post_first_divergence":
                post_divergence,

            "free_running_kl_path_dependent":
                bool(
                    mode
                    ==
                    "FREE_RUNNING"
                    and
                    first_divergence
                    is not None
                    and
                    step
                    >
                    first_divergence
                ),
        })

        clean_next = torch.tensor(
            [[
                clean_step[
                    "output_token_id"
                ]
            ]],
            dtype=torch.long,
            device=model.device,
        )

        fault_next = torch.tensor(
            [[
                metrics[
                    "fault_token_id"
                ]
            ]],
            dtype=torch.long,
            device=model.device,
        )

        if mode == "TEACHER_FORCED":

            # Both branches retain the same clean-token history.
            input_token = (
                clean_next
            )

        elif mode == "FREE_RUNNING":

            # Fault branch feeds back its own greedy token.
            input_token = (
                fault_next
            )

        else:

            raise RuntimeError(
                f"unknown mode={mode}"
            )

        del out
        del fault_logits

    del cache

    return (
        rows,
        first_divergence,
    )


# ======================================================================================
# CASE METRICS
# ======================================================================================

def summarize_mode_rows(
    rows,
    first_divergence,
):

    result = {
        "first_divergence_step":
            first_divergence,

        "top1_change_count_32":
            int(
                sum(
                    not row[
                        "token_same"
                    ]
                    for row in rows
                )
            ),

        "max_kl_32":
            max(
                (
                    row["kl"]
                    for row in rows
                    if row["kl"]
                    is not None
                ),
                default=None,
            ),

        "mean_kl_32":
            mean_or_none(
                [
                    row["kl"]
                    for row in rows
                    if row["kl"]
                    is not None
                ]
            ),

        "max_logit_diff_32":
            max(
                (
                    row[
                        "max_logit_diff"
                    ]
                    for row in rows
                    if row[
                        "max_logit_diff"
                    ]
                    is not None
                ),
                default=None,
            ),

        "any_nonfinite":
            bool(
                any(
                    (
                        not row[
                            "baseline_logits_finite"
                        ]
                    )
                    or
                    (
                        not row[
                            "fault_logits_finite"
                        ]
                    )
                    for row in rows
                )
            ),
    }

    for horizon in CHECKPOINT_HORIZONS:

        subset = rows[
            :horizon
        ]

        agreement_count = sum(
            row["token_same"]
            for row in subset
        )

        result[
            f"diverged_by_{horizon}"
        ] = bool(
            first_divergence
            is not None
            and
            first_divergence
            <=
            horizon
        )

        result[
            f"sequence_exact_match_{horizon}"
        ] = bool(
            agreement_count
            ==
            horizon
        )

        result[
            f"token_agreement_rate_{horizon}"
        ] = float(
            agreement_count
            /
            horizon
        )

    return result


# ======================================================================================
# RESUME SUPPORT
# ======================================================================================

def load_completed_cases():

    completed = {}

    if not CASES_PATH.exists():
        return completed

    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )
            except Exception:
                continue

            key = row.get(
                "case_key"
            )

            if key:
                completed[
                    key
                ] = row

    return completed


def load_existing_row_keys():

    keys = set()

    if not ROWS_PATH.exists():
        return keys

    with ROWS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        for row in csv.DictReader(f):

            keys.add((
                int(row["target_id"]),
                row["encoding"],
                row["mode"],
                int(row["step"]),
            ))

    return keys


ROW_FIELDS = [
    "protocol",
    "prompt_id",
    "target_id",
    "layer_band",
    "layer",
    "kv",
    "local_group",
    "global_group_index",

    "encoding",
    "fault_bit",
    "mode",
    "step",
    "position",

    "clean_input_token_id",
    "fault_input_token_id",

    "baseline_token_id",
    "fault_token_id",
    "token_same",

    "kl",
    "max_logit_diff",
    "mean_logit_diff",

    "baseline_logits_finite",
    "fault_logits_finite",

    "post_first_divergence",
    "free_running_kl_path_dependent",

    "manifest_z",
    "amd_reconstructed_z",
    "abs_z_error",

    "raw_scale",
    "clean_scale",
    "fault_scale",
    "scale_amplification",

    "clean_word",
    "fault_word",

    "clean_equal_raw",
    "clean_source",

    "local_write_changed",
    "cache_readback_pass",
]


def append_rows(
    rows,
    existing_keys,
):

    if not rows:
        return

    write_header = (
        not ROWS_PATH.exists()
        or
        ROWS_PATH.stat().st_size
        ==
        0
    )

    with ROWS_PATH.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=ROW_FIELDS,
        )

        if write_header:
            writer.writeheader()

        for row in rows:

            key = (
                int(
                    row[
                        "target_id"
                    ]
                ),
                row[
                    "encoding"
                ],
                row[
                    "mode"
                ],
                int(
                    row[
                        "step"
                    ]
                ),
            )

            if key in existing_keys:
                continue

            writer.writerow(
                row
            )

            existing_keys.add(
                key
            )

        f.flush()


def append_case(
    case_summary,
):

    with CASES_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                case_summary,
                ensure_ascii=False,
            )
            +
            "\n"
        )

        f.flush()


# ======================================================================================
# PROTOCOL / PROVENANCE
# ======================================================================================

protocol = {
    "protocol":
        PROTOCOL,

    "model":
        "Qwen3-32B",

    "prompt_count":
        EXPECTED_PROMPTS,

    "target_count":
        EXPECTED_TARGETS,

    "encodings":
        [
            x[0]
            for x in ENCODINGS
        ],

    "modes":
        list(
            MODES
        ),

    "max_horizon":
        MAX_HORIZON,

    "checkpoint_horizons":
        list(
            CHECKPOINT_HORIZONS
        ),

    "greedy_decoding":
        True,

    "do_sample":
        False,

    "target_reselection":
        False,

    "frozen_location_transfer":
        True,

    "amd_local_clean_metadata_reconstruction":
        True,

    "fault_injection_once_at_initial_cache":
        True,

    "post_divergence_free_running_kl_is_pure_fault_effect":
        False,

    "teacher_forced_definition":
        (
            "The fault branch receives the clean-branch "
            "greedy token at every subsequent step, "
            "preserving the clean token history."
        ),

    "free_running_definition":
        (
            "The fault branch feeds back its own greedy "
            "token after every step."
        ),

    "horizon_derivation":
        (
            "A single 32-step trajectory is executed; "
            "statistics at 1/8/16/32 are derived from "
            "the same trajectory."
        ),

    "p7d1_frozen_files_modified":
        False,
}


PROTOCOL_PATH.write_text(
    json.dumps(
        protocol,
        indent=2,
        ensure_ascii=False,
    )
    +
    "\n",
    encoding="utf-8",
)


# ======================================================================================
# MODEL LOAD ONCE
# ======================================================================================

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


model_load_t0 = time.time()


tokenizer = (
    AutoTokenizer.from_pretrained(
        str(
            MODEL_PATH
        ),
        trust_remote_code=True,
        local_files_only=True,
    )
)


model = (
    AutoModelForCausalLM.from_pretrained(
        str(
            MODEL_PATH
        ),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
)


model.eval()


model_load_sec = (
    time.time()
    -
    model_load_t0
)


# ======================================================================================
# EXECUTION
# ======================================================================================

completed_cases = (
    load_completed_cases()
)

existing_row_keys = (
    load_existing_row_keys()
)


total_expected_cases = (
    EXPECTED_TARGETS
    *
    len(
        ENCODINGS
    )
)


run_start = time.time()

case_counter = 0


for prompt_id in range(
    EXPECTED_PROMPTS
):

    prompt = (
        PROMPTS[
            prompt_id
        ]
    )

    prompt_targets = sorted(
        [
            row
            for row in MANIFEST
            if int(
                row[
                    "prompt_id"
                ]
            )
            ==
            prompt_id
        ],
        key=lambda row:
            int(
                row[
                    "target_id"
                ]
            ),
    )

    if len(
        prompt_targets
    ) != 3:

        raise RuntimeError(
            f"prompt {prompt_id} "
            f"expected 3 targets, "
            f"got {len(prompt_targets)}"
        )


    # ------------------------------------------------------------------
    # Initial BF16 prompt forward
    # ------------------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = (
        inputs[
            "input_ids"
        ]
        .to(
            model.device
        )
    )

    attention_mask = (
        inputs.get(
            "attention_mask"
        )
    )

    if attention_mask is not None:

        attention_mask = (
            attention_mask.to(
                model.device
            )
        )

    context_len = int(
        input_ids.shape[
            1
        ]
    )


    with torch.inference_mode():

        initial = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )


    original_legacy = (
        cache_to_legacy(
            initial.past_key_values
        )
    )


    next_token = torch.argmax(
        initial.logits[
            :,
            -1,
            :
        ],
        dim=-1,
        keepdim=True,
    ).detach()


    initial_next_token_id = int(
        next_token.item()
    )


    # ------------------------------------------------------------------
    # Raw full-cache INT4 baseline
    # ------------------------------------------------------------------

    (
        raw_legacy,
        raw_stats,
    ) = build_raw_full_cache(
        original_legacy
    )


    # ------------------------------------------------------------------
    # Precompute RAW clean trajectory ONCE per prompt.
    # Most encoding-clean targets equal this after BF16 cast.
    # ------------------------------------------------------------------

    raw_clean_trajectory = (
        run_clean_trajectory(
            model=model,
            clean_legacy=raw_legacy,
            initial_input_token=next_token,
            context_len=context_len,
        )
    )


    for target in prompt_targets:

        target_id = int(
            target[
                "target_id"
            ]
        )

        layer = int(
            target[
                "layer"
            ]
        )

        kv = (
            target[
                "kv"
            ]
            .upper()
        )

        if kv == "K":
            kv_idx = 0

        elif kv == "V":
            kv_idx = 1

        else:
            raise RuntimeError(
                f"unexpected kv={kv}"
            )


        local_group = int(
            target[
                "local_group"
            ]
        )


        original_tensor = (
            original_legacy[
                layer
            ][
                kv_idx
            ]
        )


        original_flat = (
            original_tensor
            .detach()
            .float()
            .contiguous()
            .view(-1)
        )


        start = (
            local_group
            *
            GROUP_SIZE
        )

        end = min(
            start
            +
            GROUP_SIZE,

            int(
                original_flat.numel()
            ),
        )


        if not (
            0
            <=
            start
            <
            end
        ):

            raise RuntimeError(
                f"target group out of range "
                f"id={target_id}"
            )


        q, raw_scale = (
            quantize_group(
                original_flat[
                    start:end
                ]
            )
        )


        amd_z = math.log2(
            raw_scale
        )

        manifest_z = float(
            target[
                "z"
            ]
        )

        z_error = abs(
            amd_z
            -
            manifest_z
        )


        raw_target = (
            raw_legacy[
                layer
            ][
                kv_idx
            ]
            .detach()
            .contiguous()
            .view(-1)[
                start:end
            ]
        )


        raw_target_cast = (
            raw_target.to(
                device=
                    original_tensor.device,

                dtype=
                    original_tensor.dtype,
            )
        )


        for (
            encoding,
            bit_field,
        ) in ENCODINGS:

            case_counter += 1

            case_key = (
                f"P{prompt_id}_"
                f"T{target_id}_"
                f"{encoding}"
            )


            if (
                case_key
                in
                completed_cases
            ):

                print(
                    f"SKIP_COMPLETED "
                    f"{case_key}"
                )

                continue


            bit = int(
                target[
                    bit_field
                ]
            )


            if encoding == "FP32":

                (
                    clean_word,
                    fault_word,
                    clean_scale,
                    fault_scale,
                ) = fp32_codec(
                    raw_scale,
                    bit,
                )


            elif encoding == "BINARY32_Z":

                (
                    clean_word,
                    fault_word,
                    clean_scale,
                    fault_scale,
                ) = binary32_codec(
                    raw_scale,
                    bit,
                )


            elif encoding == "HA_FBMS":

                (
                    clean_word,
                    fault_word,
                    clean_scale,
                    fault_scale,
                ) = ha_codec(
                    raw_scale,
                    bit,
                )


            else:

                raise RuntimeError(
                    encoding
                )


            amp = amplification(
                clean_scale,
                fault_scale,
            )


            fault_scale_valid = bool(
                math.isfinite(
                    fault_scale
                )
                and
                fault_scale
                >
                0
            )


            clean_values = (
                q
                *
                float(
                    clean_scale
                )
            ).to(
                device=
                    original_tensor.device,

                dtype=
                    original_tensor.dtype,
            )


            clean_equal_raw = bool(
                torch.equal(
                    clean_values,
                    raw_target_cast,
                )
            )


            if clean_equal_raw:

                clean_legacy = (
                    raw_legacy
                )

                clean_source = (
                    "FULL_RAW_INT4"
                )

                clean_trajectory = (
                    raw_clean_trajectory
                )


            else:

                clean_legacy = (
                    patch_group(
                        raw_legacy,
                        layer,
                        kv_idx,
                        start,
                        end,
                        clean_values,
                    )
                )

                clean_source = (
                    "ENCODING_CLEAN_TARGET"
                )

                clean_trajectory = (
                    run_clean_trajectory(
                        model=model,
                        clean_legacy=
                            clean_legacy,
                        initial_input_token=
                            next_token,
                        context_len=
                            context_len,
                    )
                )


            base_case = {
                "case_key":
                    case_key,

                "protocol":
                    PROTOCOL,

                "prompt_id":
                    prompt_id,

                "target_id":
                    target_id,

                "layer_band":
                    target[
                        "layer_band"
                    ],

                "layer":
                    layer,

                "kv":
                    kv,

                "local_group":
                    local_group,

                "global_group_index":
                    int(
                        target[
                            "global_group_index"
                        ]
                    ),

                "encoding":
                    encoding,

                "fault_bit":
                    bit,

                "manifest_z":
                    manifest_z,

                "amd_reconstructed_z":
                    amd_z,

                "abs_z_error":
                    z_error,

                "raw_scale":
                    float(
                        raw_scale
                    ),

                "clean_word":
                    int(
                        clean_word
                    ),

                "fault_word":
                    int(
                        fault_word
                    ),

                "clean_scale":
                    json_safe_number(
                        clean_scale
                    ),

                "fault_scale":
                    json_safe_number(
                        fault_scale
                    ),

                "scale_amplification":
                    json_safe_number(
                        amp
                    ),

                "fault_scale_valid":
                    fault_scale_valid,

                "clean_equal_raw":
                    clean_equal_raw,

                "clean_source":
                    clean_source,

                "initial_next_token_id":
                    initial_next_token_id,

                "target_reselection":
                    False,

                "amd_local_clean_metadata":
                    True,

                "fault_injection_once_at_initial_cache":
                    True,
            }


            # ----------------------------------------------------------
            # Invalid fault scale: record, do not execute fault forward.
            # ----------------------------------------------------------

            if not fault_scale_valid:

                case_summary = {
                    **base_case,

                    "status":
                        "INVALID_FAULT_SCALE",

                    "local_write_changed":
                        None,

                    "cache_readback_pass":
                        None,

                    "fault_forward_executed":
                        False,

                    "teacher_forced":
                        None,

                    "free_running":
                        None,
                }


                append_case(
                    case_summary
                )

                completed_cases[
                    case_key
                ] = case_summary


                print(
                    f"CASE_DONE "
                    f"{case_key} "
                    f"INVALID_FAULT_SCALE"
                )


                continue


            # ----------------------------------------------------------
            # Fault write — exact P7D1 semantics.
            # ----------------------------------------------------------

            fault_values = (
                q
                *
                float(
                    fault_scale
                )
            ).to(
                device=
                    original_tensor.device,

                dtype=
                    original_tensor.dtype,
            )


            fault_legacy = (
                patch_group(
                    raw_legacy,
                    layer,
                    kv_idx,
                    start,
                    end,
                    fault_values,
                )
            )


            local_write_changed = bool(
                not torch.equal(
                    fault_values,
                    raw_target_cast,
                )
            )


            readback = readback_group(
                fault_legacy,
                layer,
                kv_idx,
                start,
                end,
            )


            cache_readback_pass = bool(
                torch.equal(
                    readback,
                    fault_values,
                )
            )


            del readback


            if not cache_readback_pass:

                raise RuntimeError(
                    f"cache readback failure "
                    f"target={target_id} "
                    f"encoding={encoding}"
                )


            mode_summaries = {}


            for mode in MODES:

                (
                    mode_rows,
                    first_divergence,
                ) = run_fault_trajectory(
                    model=model,
                    fault_legacy=
                        fault_legacy,
                    initial_input_token=
                        next_token,
                    context_len=
                        context_len,
                    clean_trajectory=
                        clean_trajectory,
                    mode=
                        mode,
                )


                enriched_rows = []


                for row in mode_rows:

                    enriched_rows.append({
                        "protocol":
                            PROTOCOL,

                        "prompt_id":
                            prompt_id,

                        "target_id":
                            target_id,

                        "layer_band":
                            target[
                                "layer_band"
                            ],

                        "layer":
                            layer,

                        "kv":
                            kv,

                        "local_group":
                            local_group,

                        "global_group_index":
                            int(
                                target[
                                    "global_group_index"
                                ]
                            ),

                        "encoding":
                            encoding,

                        "fault_bit":
                            bit,

                        "mode":
                            mode,

                        **row,

                        "manifest_z":
                            manifest_z,

                        "amd_reconstructed_z":
                            amd_z,

                        "abs_z_error":
                            z_error,

                        "raw_scale":
                            float(
                                raw_scale
                            ),

                        "clean_scale":
                            json_safe_number(
                                clean_scale
                            ),

                        "fault_scale":
                            json_safe_number(
                                fault_scale
                            ),

                        "scale_amplification":
                            json_safe_number(
                                amp
                            ),

                        "clean_word":
                            int(
                                clean_word
                            ),

                        "fault_word":
                            int(
                                fault_word
                            ),

                        "clean_equal_raw":
                            clean_equal_raw,

                        "clean_source":
                            clean_source,

                        "local_write_changed":
                            local_write_changed,

                        "cache_readback_pass":
                            cache_readback_pass,
                    })


                append_rows(
                    enriched_rows,
                    existing_row_keys,
                )


                mode_summaries[
                    mode
                ] = summarize_mode_rows(
                    mode_rows,
                    first_divergence,
                )


            case_summary = {
                **base_case,

                "status":
                    "PASS",

                "local_write_changed":
                    local_write_changed,

                "cache_readback_pass":
                    cache_readback_pass,

                "fault_forward_executed":
                    True,

                "teacher_forced":
                    mode_summaries[
                        "TEACHER_FORCED"
                    ],

                "free_running":
                    mode_summaries[
                        "FREE_RUNNING"
                    ],
            }


            append_case(
                case_summary
            )


            completed_cases[
                case_key
            ] = case_summary


            elapsed_sec = (
                time.time()
                -
                run_start
            )


            progress = {
                "completed_cases":
                    len(
                        completed_cases
                    ),

                "expected_cases":
                    total_expected_cases,

                "last_case":
                    case_key,

                "elapsed_sec":
                    elapsed_sec,

                "model_load_sec":
                    model_load_sec,
            }


            PROGRESS_PATH.write_text(
                json.dumps(
                    progress,
                    indent=2,
                )
                +
                "\n",
                encoding="utf-8",
            )


            print(
                f"CASE_DONE "
                f"{case_key} "
                f"TF_FIRST="
                f"{mode_summaries['TEACHER_FORCED']['first_divergence_step']} "
                f"FR_FIRST="
                f"{mode_summaries['FREE_RUNNING']['first_divergence_step']} "
                f"COMPLETED="
                f"{len(completed_cases)}/"
                f"{total_expected_cases}"
            )


            if not clean_equal_raw:

                del clean_trajectory
                del clean_legacy


            del fault_legacy
            del fault_values


            gc.collect()
            torch.cuda.empty_cache()


    # ------------------------------------------------------------------
    # Prompt cleanup
    # ------------------------------------------------------------------

    for item in raw_clean_trajectory:
        del item["logits"]

    del raw_clean_trajectory
    del raw_legacy
    del original_legacy
    del initial
    del input_ids
    del next_token

    gc.collect()
    torch.cuda.empty_cache()


# ======================================================================================
# FINAL AGGREGATION
# ======================================================================================

all_cases = list(
    load_completed_cases().values()
)


by_encoding = {}


for encoding, _ in ENCODINGS:

    subset = [
        case
        for case in all_cases
        if case[
            "encoding"
        ]
        ==
        encoding
    ]


    valid = [
        case
        for case in subset
        if case.get(
            "status"
        )
        ==
        "PASS"
    ]


    invalid = [
        case
        for case in subset
        if case.get(
            "status"
        )
        ==
        "INVALID_FAULT_SCALE"
    ]


    enc_summary = {
        "case_count":
            len(subset),

        "valid_case_count":
            len(valid),

        "invalid_fault_scale_count":
            len(invalid),

        "max_scale_amplification":
            max(
                (
                    float(
                        case[
                            "scale_amplification"
                        ]
                    )
                    for case in valid
                    if isinstance(
                        case.get(
                            "scale_amplification"
                        ),
                        (int, float),
                    )
                ),
                default=None,
            ),
    }


    for mode_key, mode_name in [
        (
            "teacher_forced",
            "TF",
        ),
        (
            "free_running",
            "FR",
        ),
    ]:

        mode_cases = [
            case[
                mode_key
            ]
            for case in valid
            if case.get(
                mode_key
            )
            is not None
        ]


        first_divs = [
            mode[
                "first_divergence_step"
            ]
            for mode in mode_cases
            if mode[
                "first_divergence_step"
            ]
            is not None
        ]


        enc_summary[
            f"{mode_name}_diverged_case_count_32"
        ] = len(
            first_divs
        )


        enc_summary[
            f"{mode_name}_first_divergence_median"
        ] = median_or_none(
            first_divs
        )


        for horizon in CHECKPOINT_HORIZONS:

            enc_summary[
                f"{mode_name}_diverged_by_{horizon}_count"
            ] = int(
                sum(
                    bool(
                        mode[
                            f"diverged_by_{horizon}"
                        ]
                    )
                    for mode in mode_cases
                )
            )


            enc_summary[
                f"{mode_name}_sequence_exact_match_{horizon}_count"
            ] = int(
                sum(
                    bool(
                        mode[
                            f"sequence_exact_match_{horizon}"
                        ]
                    )
                    for mode in mode_cases
                )
            )


            enc_summary[
                f"{mode_name}_mean_token_agreement_rate_{horizon}"
            ] = mean_or_none(
                [
                    mode[
                        f"token_agreement_rate_{horizon}"
                    ]
                    for mode in mode_cases
                ]
            )


        enc_summary[
            f"{mode_name}_max_KL_32"
        ] = max(
            (
                mode[
                    "max_kl_32"
                ]
                for mode in mode_cases
                if mode[
                    "max_kl_32"
                ]
                is not None
            ),
            default=None,
        )


        enc_summary[
            f"{mode_name}_mean_case_mean_KL_32"
        ] = mean_or_none(
            [
                mode[
                    "mean_kl_32"
                ]
                for mode in mode_cases
                if mode[
                    "mean_kl_32"
                ]
                is not None
            ]
        )


        enc_summary[
            f"{mode_name}_nonfinite_case_count"
        ] = int(
            sum(
                bool(
                    mode[
                        "any_nonfinite"
                    ]
                )
                for mode in mode_cases
            )
        )


    by_encoding[
        encoding
    ] = enc_summary


final_elapsed_sec = (
    time.time()
    -
    run_start
)


summary = {
    "protocol":
        PROTOCOL,

    "FINAL_PASS":
        bool(
            len(
                all_cases
            )
            ==
            total_expected_cases
        ),

    "MODEL_LOAD_COUNT":
        1,

    "MODEL_LOAD_SEC":
        model_load_sec,

    "PROMPT_COUNT":
        EXPECTED_PROMPTS,

    "TARGET_COUNT":
        EXPECTED_TARGETS,

    "ENCODING_COUNT":
        len(
            ENCODINGS
        ),

    "EXPECTED_CASE_COUNT":
        total_expected_cases,

    "COMPLETED_CASE_COUNT":
        len(
            all_cases
        ),

    "MAX_HORIZON":
        MAX_HORIZON,

    "CHECKPOINT_HORIZONS":
        list(
            CHECKPOINT_HORIZONS
        ),

    "MODES":
        list(
            MODES
        ),

    "TARGET_RESELECTION":
        False,

    "FROZEN_LOCATION_TRANSFER":
        True,

    "AMD_LOCAL_CLEAN_METADATA_RECONSTRUCTION":
        True,

    "FAULT_INJECTION_ONCE_AT_INITIAL_CACHE":
        True,

    "GREEDY_DECODING":
        True,

    "POST_DIVERGENCE_FREE_RUNNING_KL_IS_PURE_FAULT_EFFECT":
        False,

    "P7D1_FROZEN_FILES_MODIFIED":
        False,

    "ELAPSED_SEC":
        final_elapsed_sec,

    "BY_ENCODING":
        by_encoding,

    "ENVIRONMENT": {
        "torch":
            torch.__version__,

        "hip":
            torch.version.hip,

        "transformers":
            transformers.__version__,

        "gpu":
            (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else None
            ),

        "gcnArchName":
            (
                getattr(
                    torch.cuda.get_device_properties(
                        0
                    ),
                    "gcnArchName",
                    None,
                )
                if torch.cuda.is_available()
                else None
            ),
    },

    "PROVENANCE": {
        "model_path":
            str(
                MODEL_PATH
            ),

        "prompt_source":
            str(
                PROMPT_SOURCE
            ),

        "prompt_source_sha256":
            sha256_file(
                PROMPT_SOURCE
            ),

        "manifest":
            str(
                MANIFEST_PATH
            ),

        "manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "p7d1_source":
            str(
                P7D1_SOURCE
            ),

        "p7d1_source_sha256":
            sha256_file(
                P7D1_SOURCE
            ),
    },
}


SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    +
    "\n",
    encoding="utf-8",
)


status_lines = [
    f"P8_FORMAL_FINAL_PASS={summary['FINAL_PASS']}",
    f"COMPLETED_CASE_COUNT={len(all_cases)}",
    f"EXPECTED_CASE_COUNT={total_expected_cases}",
    "PROMPT_COUNT=8",
    "TARGET_COUNT=24",
    "ENCODINGS=FP32,BINARY32_Z,HA_FBMS",
    "MODES=TEACHER_FORCED,FREE_RUNNING",
    "MAX_HORIZON=32",
    "CHECKPOINT_HORIZONS=1,8,16,32",
    "MODEL_LOAD_COUNT=1",
    "TARGET_RESELECTION=False",
    "FROZEN_LOCATION_TRANSFER=True",
    "AMD_LOCAL_CLEAN_METADATA_RECONSTRUCTION=True",
    "FAULT_INJECTION_ONCE_AT_INITIAL_CACHE=True",
    "P7D1_FROZEN_FILES_MODIFIED=False",
]


for encoding in [
    "FP32",
    "BINARY32_Z",
    "HA_FBMS",
]:

    enc = (
        by_encoding[
            encoding
        ]
    )

    status_lines.extend([
        (
            f"{encoding}_VALID_CASES="
            f"{enc['valid_case_count']}"
        ),
        (
            f"{encoding}_INVALID_FAULT_SCALE="
            f"{enc['invalid_fault_scale_count']}"
        ),
        (
            f"{encoding}_TF_DIVERGED_BY_32="
            f"{enc['TF_diverged_by_32_count']}"
        ),
        (
            f"{encoding}_FR_DIVERGED_BY_32="
            f"{enc['FR_diverged_by_32_count']}"
        ),
        (
            f"{encoding}_TF_MAX_KL_32="
            f"{enc['TF_max_KL_32']}"
        ),
    ])


STATUS_PATH.write_text(
    "\n".join(
        status_lines
    )
    +
    "\n",
    encoding="utf-8",
)


provenance_lines = [
    "METAKV P8 FORMAL V1 PROVENANCE",
    "=" * 100,
    f"PROTOCOL={PROTOCOL}",
    f"MODEL_PATH={MODEL_PATH}",
    f"PROMPT_SOURCE={PROMPT_SOURCE}",
    f"PROMPT_SOURCE_SHA256={sha256_file(PROMPT_SOURCE)}",
    f"MANIFEST={MANIFEST_PATH}",
    f"MANIFEST_SHA256={sha256_file(MANIFEST_PATH)}",
    f"P7D1_SOURCE={P7D1_SOURCE}",
    f"P7D1_SOURCE_SHA256={sha256_file(P7D1_SOURCE)}",
    f"TORCH={torch.__version__}",
    f"HIP={torch.version.hip}",
    f"TRANSFORMERS={transformers.__version__}",
    (
        "GPU="
        +
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "NONE"
        )
    ),
    (
        "GCN_ARCH="
        +
        str(
            getattr(
                torch.cuda.get_device_properties(0),
                "gcnArchName",
                None,
            )
            if torch.cuda.is_available()
            else None
        )
    ),
    "TARGET_RESELECTION=False",
    "AMD_EXACT_REPRODUCTION=False",
    "FROZEN_LOCATION_TRANSFER=True",
    "FAULT_PROBABILITY_CLAIM=False",
    "HARDWARE_BER_CLAIM=False",
    "PHYSICAL_HBM_EVENT_PROBABILITY_CLAIM=False",
    "POST_DIVERGENCE_FREE_RUNNING_KL_IS_PURE_FAULT_EFFECT=False",
]


PROVENANCE_PATH.write_text(
    "\n".join(
        provenance_lines
    )
    +
    "\n",
    encoding="utf-8",
)


print("=" * 100)
print(
    f"P8_FORMAL_FINAL_PASS="
    f"{summary['FINAL_PASS']}"
)
print(
    f"COMPLETED_CASE_COUNT="
    f"{len(all_cases)}"
)
print(
    f"EXPECTED_CASE_COUNT="
    f"{total_expected_cases}"
)

for encoding in [
    "FP32",
    "BINARY32_Z",
    "HA_FBMS",
]:

    enc = (
        by_encoding[
            encoding
        ]
    )

    print(
        f"{encoding}:"
        f"VALID={enc['valid_case_count']},"
        f"INVALID={enc['invalid_fault_scale_count']},"
        f"TF32={enc['TF_diverged_by_32_count']},"
        f"FR32={enc['FR_diverged_by_32_count']},"
        f"TF_MAX_KL={enc['TF_max_KL_32']}"
    )


del model

gc.collect()

torch.cuda.empty_cache()
