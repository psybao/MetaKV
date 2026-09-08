#!/usr/bin/env python3
"""AMD target-transfer audit for the frozen P7D1/P7D0 evidence.

This program deliberately performs clean prompt forwards only.  It transfers the
24 frozen logical locations, reconstructs their clean metadata on the active AMD
backend, and never performs fault injection, a fault forward, replay, or target
reselection.  Frozen source files are read but never modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import logging
import math
import os
import statistics
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


DEFAULT_MANIFEST = Path(
    r"${METAKV_ROOT}\method_upgrade_m55\output\final_evidence_audit\full_evidence_extract"
    r"\h800_final_experiments\P7_QWEN3_32B\P7D1_FULL_CACHE_FAULT_20260904_020635"
    r"\freeze\P7D0_24_LOCATION_MANIFEST_SOURCE.csv"
)
TRANSFER_STATEMENT = (
    "Frozen P7D0 logical target locations are transferred across backend;\n"
    "clean metadata values are reconstructed locally on AMD."
)
REUSED_FUNCTIONS = {
    "quantize_group",
    "build_raw_full_cache",
    "f32_to_word",
    "word_to_f32",
    "fp32_codec",
    "binary32_encode",
    "binary32_decode",
    "binary32_codec",
    "ha_encode",
    "ha_decode",
    "ha_codec",
}
REUSED_CONSTANTS = {
    "GROUP_SIZE", "ZMIN", "ZMAX", "BC", "BF", "FMAX", "COARSE_MASK", "BINARY_MAX"
}
ROW_FIELDS = [
    "target_id", "prompt_id", "layer_band", "layer", "kv", "local_group",
    "manifest_z", "amd_reconstructed_z", "abs_z_error", "raw_scale",
    "fp32_clean_word", "binary32z_clean_word", "ha_fbms_clean_word",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_frozen_prompts(source_path: Path) -> tuple[str, list[str]]:
    """Use the same AST/literal selection rule as frozen P7D1, for all 8 prompts."""
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    candidates: list[tuple[str, list[str]]] = []
    for node in tree.body:
        names: list[str] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            value = node.value
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            names = [node.target.id]
        if value is None:
            continue
        try:
            obj = ast.literal_eval(value)
        except Exception:
            continue
        for name in names:
            if "prompt" not in name.lower() or not isinstance(obj, (list, tuple)):
                continue
            if len(obj) >= 8 and all(isinstance(x, str) for x in obj):
                candidates.append((name, list(obj)))
    if not candidates:
        raise RuntimeError("unable to recover frozen P4 prompts")
    candidates.sort(key=lambda item: (abs(len(item[1]) - 8), item[0]))
    variable, prompts = candidates[0]
    prompts = prompts[:8]
    if len(prompts) != 8:
        raise RuntimeError(f"expected 8 frozen prompts, got {len(prompts)}")
    return variable, prompts


def load_frozen_helpers(source_path: Path) -> dict[str, Any]:
    """Load exact constants/functions from frozen P7D1 without its top-level execution."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected: list[ast.stmt] = []
    found_functions: set[str] = set()
    found_constants: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names: set[str] = set()
            if isinstance(node, ast.Assign):
                names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node.target, ast.Name):
                names = {node.target.id}
            if names & REUSED_CONSTANTS:
                selected.append(node)
                found_constants |= names & REUSED_CONSTANTS
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in REUSED_FUNCTIONS:
            selected.append(node)
            found_functions.add(node.name)
    if found_functions != REUSED_FUNCTIONS or found_constants != REUSED_CONSTANTS:
        raise RuntimeError(
            f"frozen helper extraction incomplete: functions={sorted(found_functions)}, "
            f"constants={sorted(found_constants)}"
        )
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {
        "torch": torch, "np": np, "math": math, "struct": struct,
        "DynamicCache": DynamicCache,
    }
    exec(compile(module, str(source_path), "exec"), namespace, namespace)
    if int(namespace["GROUP_SIZE"]) != 32:
        raise RuntimeError(f"frozen GROUP_SIZE is not 32: {namespace['GROUP_SIZE']}")
    return namespace


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "target_id", "prompt_id", "layer_band", "layer", "kv", "local_group", "z",
        "fp32_bit", "binary32_bit", "ha_fbms_bit",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"manifest columns missing: {sorted(required - set(rows[0] if rows else []))}")
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 frozen targets, got {len(rows)}")
    prompt_ids = [int(row["prompt_id"]) for row in rows]
    if sorted(set(prompt_ids)) != list(range(8)):
        raise RuntimeError(f"expected prompt ids 0..7, got {sorted(set(prompt_ids))}")
    if any(prompt_ids.count(i) != 3 for i in range(8)):
        raise RuntimeError("expected exactly 3 frozen targets per prompt")
    return rows


def gpu_provenance() -> tuple[str, str]:
    props = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name(0)
    gcn_arch = getattr(props, "gcnArchName", None)
    if not gcn_arch:
        gcn_arch = getattr(props, "gcn_arch_name", None)
    return str(gpu_name), str(gcn_arch or "UNAVAILABLE")


def percentile95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    p7d1_source = os.environ.get("METAKV_P7D1_SOURCE")
    if not p7d1_source:
        parser.error("METAKV_P7D1_SOURCE must be set to the frozen P7D1 worker source path")
    parser.set_defaults(source_p7d1=Path(p7d1_source))
    parser.add_argument("--manifest", type=Path, default=Path(os.environ.get("METAKV_MANIFEST", DEFAULT_MANIFEST)))
    parser.add_argument("--p4-prompt-source", type=Path, default=os.environ.get("METAKV_SOURCE_SCRIPT"), required="METAKV_SOURCE_SCRIPT" not in os.environ)
    parser.add_argument("--model-path", type=Path, default=os.environ.get("METAKV_MODEL_PATH"), required="METAKV_MODEL_PATH" not in os.environ)
    parser.add_argument("--outroot", type=Path, default=Path(os.environ.get("METAKV_OUTROOT", Path(__file__).resolve().parent / "p7d1_amd_target_transfer_output")))
    return parser.parse_args()


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    for source in (args.source_p7d1, args.manifest, args.p4_prompt_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if torch.version.hip is None:
        raise RuntimeError("AMD ROCm/HIP PyTorch is required; torch.version.hip is None")

    args.outroot.mkdir(parents=True, exist_ok=True)
    log_path = args.outroot / "p7d1_amd_target_transfer_audit.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    log = logging.getLogger("p7d1_amd_target_transfer")

    helpers = load_frozen_helpers(args.source_p7d1)
    prompt_variable, prompts = recover_frozen_prompts(args.p4_prompt_source)
    manifest_rows = read_manifest(args.manifest)
    targets_by_prompt = {i: [] for i in range(8)}
    for target in manifest_rows:
        targets_by_prompt[int(target["prompt_id"])].append(target)

    gpu_name, gcn_arch = gpu_provenance()
    provenance = {
        "SOURCE_P7D1": str(args.source_p7d1.resolve()),
        "SOURCE_P7D1_SHA256": sha256_file(args.source_p7d1),
        "SOURCE_MANIFEST": str(args.manifest.resolve()),
        "SOURCE_MANIFEST_SHA256": sha256_file(args.manifest),
        "P4_PROMPT_SOURCE": str(args.p4_prompt_source.resolve()),
        "P4_PROMPT_SOURCE_SHA256": sha256_file(args.p4_prompt_source),
        "P4_PROMPT_VARIABLE": prompt_variable,
        "MODEL_PATH": str(args.model_path.resolve()),
        "TORCH_VERSION": torch.__version__,
        "HIP_VERSION": torch.version.hip,
        "TRANSFORMERS_VERSION": transformers.__version__,
        "GPU_NAME": gpu_name,
        "GCN_ARCH_NAME": gcn_arch,
    }
    log.info("provenance=%s", json.dumps(provenance, sort_keys=True))

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    # The sole model construction in this program; it is reused for all 8 prompts.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=True,
    )
    model.eval()

    output_rows: list[dict[str, Any]] = []
    cache_stats: dict[str, Any] = {}
    for prompt_id, prompt in enumerate(prompts):
        log.info("prompt_start prompt_id=%d target_count=%d", prompt_id, len(targets_by_prompt[prompt_id]))
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        clean_output = model(**inputs, use_cache=True, return_dict=True)
        legacy = tuple(
            (key.detach(), value.detach())
            for key, value in clean_output.past_key_values.to_legacy_cache()
        )
        # Invoke the exact frozen P7D1 full-cache construction once per prompt.
        raw_full_cache, stats = helpers["build_raw_full_cache"](legacy)
        cache_stats[str(prompt_id)] = stats

        for target in targets_by_prompt[prompt_id]:
            layer = int(target["layer"])
            kv = target["kv"].strip().lower()
            local_group = int(target["local_group"])
            if layer < 0 or layer >= len(legacy) or kv not in {"k", "v"}:
                raise RuntimeError(f"invalid frozen target location: {target}")
            tensor = legacy[layer][0 if kv == "k" else 1]
            flat = tensor.detach().float().contiguous().view(-1)
            start = local_group * 32
            end = min(start + 32, int(flat.numel()))
            if start < 0 or start >= end:
                raise RuntimeError(f"invalid frozen local_group: {target}")
            _, raw_scale = helpers["quantize_group"](flat[start:end])
            reconstructed_z = math.log2(raw_scale)
            manifest_z = float(target["z"])
            fp32_clean_word = helpers["fp32_codec"](raw_scale, int(target["fp32_bit"]))[0]
            binary_clean_word = helpers["binary32_codec"](raw_scale, int(target["binary32_bit"]))[0]
            ha_clean_word = helpers["ha_codec"](raw_scale, int(target["ha_fbms_bit"]))[0]
            output_rows.append({
                "target_id": target["target_id"], "prompt_id": prompt_id,
                "layer_band": target["layer_band"], "layer": layer, "kv": kv,
                "local_group": local_group, "manifest_z": manifest_z,
                "amd_reconstructed_z": reconstructed_z,
                "abs_z_error": abs(reconstructed_z - manifest_z), "raw_scale": raw_scale,
                "fp32_clean_word": fp32_clean_word,
                "binary32z_clean_word": binary_clean_word,
                "ha_fbms_clean_word": ha_clean_word,
            })
            log.info("target_done id=%s prompt_id=%d zerr=%.12g", target["target_id"], prompt_id, abs(reconstructed_z - manifest_z))

        del raw_full_cache, legacy, clean_output, inputs
        gc.collect()
        torch.cuda.empty_cache()
        log.info("prompt_done prompt_id=%d", prompt_id)

    if len(output_rows) != 24:
        raise RuntimeError(f"internal target count mismatch: {len(output_rows)}")
    errors = [float(row["abs_z_error"]) for row in output_rows]
    summary = {
        "TARGET_COUNT": len(output_rows), "PROMPT_COUNT": len(prompts),
        "MAX_ABS_Z_ERROR": max(errors), "MEAN_ABS_Z_ERROR": statistics.fmean(errors),
        "MEDIAN_ABS_Z_ERROR": statistics.median(errors), "P95_ABS_Z_ERROR": percentile95(errors),
        "COUNT_ZERR_LE_1E-3": sum(x <= 1e-3 for x in errors),
        "COUNT_ZERR_LE_1E-2": sum(x <= 1e-2 for x in errors),
        "COUNT_ZERR_LE_5E-2": sum(x <= 5e-2 for x in errors),
        "TARGET_RESELECTION": False, "FAULT_FORWARD": False,
        "FROZEN_LOCATION_TRANSFER": True, "AMD_EXACT_REPRODUCTION": False,
        "MODEL_LOAD_COUNT": 1, "GROUP_SIZE": 32, "CACHE_STATS_BY_PROMPT": cache_stats,
        "STATEMENT": TRANSFER_STATEMENT, "PROVENANCE": provenance,
    }
    rows_path = args.outroot / "p7d1_amd_target_transfer_rows.csv"
    summary_path = args.outroot / "P7D1_AMD_TARGET_TRANSFER_SUMMARY.json"
    status_path = args.outroot / "P7D1_AMD_TARGET_TRANSFER_STATUS.txt"
    write_csv_atomic(rows_path, output_rows)
    write_text_atomic(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    status_lines = [
        "P7D1_AMD_TARGET_TRANSFER_PASS=True", "TARGET_COUNT=24", "PROMPT_COUNT=8",
        "MODEL_LOAD_COUNT=1", "TARGET_RESELECTION=False", "FAULT_FORWARD=False",
        "FROZEN_LOCATION_TRANSFER=True", "AMD_EXACT_REPRODUCTION=False", TRANSFER_STATEMENT,
    ]
    write_text_atomic(status_path, "\n".join(status_lines) + "\n")
    log.info("audit_complete rows=%s summary=%s status=%s", rows_path, summary_path, status_path)
    print("P7D1_AMD_TARGET_TRANSFER_PASS=True")
    print(f"OUTPUT_ROOT={args.outroot.resolve()}")
    print("TARGET_COUNT=24")
    print("PROMPT_COUNT=8")
    print("MODEL_LOAD_COUNT=1")
    print("TARGET_RESELECTION=False")
    print("FAULT_FORWARD=False")
    print("AMD_EXACT_REPRODUCTION=False")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
