import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL=Path(
    os.environ["METAKV_QWEN_MODEL"]
)

RUNROOT=Path(
    os.environ["METAKV_P4A_ROOT"]
)

OUT=RUNROOT/"output"
FREEZE=RUNROOT/"freeze"

OUT.mkdir(
    parents=True,
    exist_ok=True
)

FREEZE.mkdir(
    parents=True,
    exist_ok=True
)

SEED=20260904

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE="cuda:0"

GROUP_SIZE=32

BC=19
BF=13

assert BC+BF==32

FMAX=(1<<BF)-1

PROMPTS=[
    "Explain why metadata corruption can be more damaging than payload corruption in quantized caches.",
    "Give a concise explanation of attention key-value caching in transformer inference.",
    "What is the difference between numerical error and a catastrophic representation error?",
    "Summarize the trade-off between memory efficiency and numerical robustness in low-precision inference.",
    "A system stores quantization scales alongside packed INT4 values. Explain why scale protection matters.",
    "Write three sentences explaining why fault-tolerant metadata encoding can improve system reliability.",
    "What happens if a quantization scale becomes several orders of magnitude too large?",
    "Explain the purpose of grouped quantization in neural-network inference.",
]

MAX_INPUT_TOKENS=256


def quantize_tensor_groupwise_int4(
    x,
    group_size=32,
):

    x32=(
        x.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
    )

    shape=x32.shape

    flat=x32.reshape(-1)

    n=flat.size

    pad=(-n)%group_size

    if pad:

        flat=np.pad(
            flat,
            (0,pad),
            constant_values=0.0
        )

    groups=flat.reshape(
        -1,
        group_size
    )

    maxabs=np.max(
        np.abs(groups),
        axis=1
    )

    scale=(
        maxabs/7.0
    ).astype(
        np.float32
    )

    scale[
        scale==0
    ]=np.float32(
        2.0**-24
    )

    q=np.rint(
        groups
        /
        scale[:,None]
    )

    q=np.clip(
        q,
        -8,
        7
    ).astype(
        np.int8
    )

    dq=(
        q.astype(np.float32)
        *
        scale[:,None]
    )

    return {
        "shape":shape,
        "q":q,
        "scale":scale,
        "dequant":dq.reshape(-1)[:n].reshape(shape),
        "pad":pad,
    }


def extract_legacy_cache(
    past_key_values
):

    if past_key_values is None:
        raise RuntimeError(
            "past_key_values is None"
        )

    if hasattr(
        past_key_values,
        "to_legacy_cache"
    ):

        try:

            x=past_key_values.to_legacy_cache()

            if x is not None:
                return x

        except Exception:
            pass

    try:
        return tuple(
            past_key_values
        )

    except Exception as e:
        raise RuntimeError(
            f"unsupported cache type {type(past_key_values)}"
        ) from e


def fbms_encode(
    z,
    zmin,
    zmax,
):

    z=np.asarray(
        z,
        dtype=np.float64
    )

    width=zmax-zmin

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
        coarse>=BC+1
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

        mask=(
            coarse==c
        )

        thermometer=(
            0
            if c==0
            else
            (1<<c)-1
        )

        words[mask]=np.uint32(
            thermometer
        )

    words |= (
        fine.astype(np.uint32)
        <<
        np.uint32(BC)
    )

    return words


def fbms_decode(
    words,
    zmin,
    zmax,
):

    words=np.asarray(
        words,
        dtype=np.uint32
    )

    coarse_mask=(
        1<<BC
    )-1

    coarse_bits=(
        words
        &
        np.uint32(coarse_mask)
    )

    flat=coarse_bits.reshape(-1)

    count=np.fromiter(
        (
            int(x).bit_count()
            for x in flat
        ),
        dtype=np.int16,
        count=flat.size
    ).reshape(
        coarse_bits.shape
    )

    fine=(
        words
        >>
        np.uint32(BC)
    ) & np.uint32(FMAX)

    u=(
        count.astype(np.float64)
        +
        fine.astype(np.float64)
        /
        float(FMAX)
    )

    return (
        zmin
        +
        (zmax-zmin)
        /
        float(BC+1)
        *
        u
    )


def theorem_bound(
    zmin,
    zmax,
):

    return (
        (zmax-zmin)
        /
        float(BC+1)
        /
        (
            2.0
            *
            float(FMAX)
        )
    )


print("="*110)
print("METAKV P4A QWEN3-14B REAL-KV CLEAN CALIBRATION")
print("="*110)

print("PYTHON=",sys.version.replace("\n"," "),sep="")
print("PYTORCH=",torch.__version__,sep="")
print("CUDA=",torch.version.cuda,sep="")
print("GPU=",torch.cuda.get_device_name(0),sep="")
print("GROUP_SIZE=",GROUP_SIZE,sep="")
print("BC=",BC,sep="")
print("BF=",BF,sep="")

t0=time.time()

tokenizer=AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
)

model=AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={
        "":DEVICE
    },
    low_cpu_mem_usage=True,
)

model.eval()

print(
    "MODEL_CLASS=",
    model.__class__.__name__,
    sep=""
)

print(
    "NUM_PARAMETERS=",
    sum(
        p.numel()
        for p in model.parameters()
    ),
    sep=""
)

print(
    "MODEL_LOAD_SEC=",
    time.time()-t0,
    sep=""
)

records=[]
all_z=[]

print()
print("="*110)
print("PASS 1 — REAL KV SCALE COLLECTION")
print("="*110)

with torch.inference_mode():

    for prompt_id,prompt in enumerate(
        PROMPTS
    ):

        enc=tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        input_ids=enc[
            "input_ids"
        ].to(
            DEVICE
        )

        attention_mask=enc.get(
            "attention_mask"
        )

        if attention_mask is not None:
            attention_mask=attention_mask.to(
                DEVICE
            )

        out=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        cache=extract_legacy_cache(
            out.past_key_values
        )

        print(
            f"[P4A-PASS1] "
            f"prompt={prompt_id+1}/{len(PROMPTS)} "
            f"tokens={input_ids.shape[1]} "
            f"layers={len(cache)}",
            flush=True
        )

        for layer_idx,layer in enumerate(
            cache
        ):

            if len(layer)<2:
                continue

            for kv,tensor in [
                ("K",layer[0]),
                ("V",layer[1]),
            ]:

                qd=quantize_tensor_groupwise_int4(
                    tensor,
                    GROUP_SIZE
                )

                scale=qd[
                    "scale"
                ].astype(
                    np.float64
                )

                z=np.log2(
                    scale
                )

                if not np.all(
                    np.isfinite(z)
                ):
                    raise RuntimeError(
                        f"nonfinite scale prompt={prompt_id} layer={layer_idx} kv={kv}"
                    )

                all_z.extend(
                    z.tolist()
                )

                records.append({
                    "prompt_id":prompt_id,
                    "layer":layer_idx,
                    "kv":kv,
                    "tensor_shape":
                        json.dumps(
                            list(
                                tensor.shape
                            )
                        ),
                    "groups":len(scale),
                    "scale_min":float(scale.min()),
                    "scale_max":float(scale.max()),
                    "log2_min":float(z.min()),
                    "log2_max":float(z.max()),
                    "log2_mean":float(z.mean()),
                })

        del out
        del cache
        gc.collect()
        torch.cuda.empty_cache()


z=np.asarray(
    all_z,
    dtype=np.float64
)

if z.size==0:
    raise RuntimeError(
        "no scale samples"
    )

observed_min=float(
    z.min()
)

observed_max=float(
    z.max()
)

# Deployment calibration interval:
# conservative integer enclosure.
ZMIN=float(
    math.floor(
        observed_min
    )
)

ZMAX=float(
    math.ceil(
        observed_max
    )
)

if ZMAX<=ZMIN:
    ZMAX=ZMIN+1.0

bound=theorem_bound(
    ZMIN,
    ZMAX
)

print()
print("REAL_SCALE_SAMPLE_COUNT=",z.size,sep="")
print("OBSERVED_LOG2_MIN=",observed_min,sep="")
print("OBSERVED_LOG2_MAX=",observed_max,sep="")
print("DEPLOYMENT_CALIBRATED_ZMIN=",ZMIN,sep="")
print("DEPLOYMENT_CALIBRATED_ZMAX=",ZMAX,sep="")
print("CLEAN_LOG2_THEOREM_BOUND=",bound,sep="")

words=fbms_encode(
    z,
    ZMIN,
    ZMAX
)

decoded_z=fbms_decode(
    words,
    ZMIN,
    ZMAX
)

err=np.abs(
    decoded_z-z
)

real_scale=np.exp2(
    z
)

decoded_scale=np.exp2(
    decoded_z
)

scale_abs=np.abs(
    decoded_scale
    -
    real_scale
)

scale_rel=(
    scale_abs
    /
    np.maximum(
        np.abs(real_scale),
        1e-30
    )
)

max_z_err=float(
    err.max()
)

mean_z_err=float(
    err.mean()
)

max_scale_abs=float(
    scale_abs.max()
)

max_scale_rel=float(
    scale_rel.max()
)

bound_pass=(
    max_z_err
    <=
    bound*(1+1e-9)
    +
    1e-12
)

no_clip=bool(
    np.all(z>=ZMIN)
    and
    np.all(z<=ZMAX)
)

finite_pass=bool(
    np.all(
        np.isfinite(
            decoded_z
        )
    )
)

print()
print("="*110)
print("PASS 2 — CLEAN CODEC")
print("="*110)

print("CODEC_WORD_COUNT=",len(words),sep="")
print("MAX_LOG2_ABS_ERROR=",max_z_err,sep="")
print("MEAN_LOG2_ABS_ERROR=",mean_z_err,sep="")
print("MAX_SCALE_ABS_ERROR=",max_scale_abs,sep="")
print("MAX_SCALE_REL_ERROR=",max_scale_rel,sep="")
print("CLEAN_THEOREM_BOUND_PASS=",bound_pass,sep="")
print("NO_CLIP_PASS=",no_clip,sep="")
print("CODEC_FINITE_PASS=",finite_pass,sep="")


def write_csv(
    path,
    rows,
):

    if not rows:
        return

    fields=list(
        rows[0].keys()
    )

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


write_csv(
    OUT/"p4a_real_scale_records.csv",
    records
)

np.save(
    OUT/"p4a_real_log2_scales.npy",
    z
)

np.save(
    OUT/"p4a_fbms_words.npy",
    words
)

final_pass=all([
    z.size>0,
    bound_pass,
    no_clip,
    finite_pass,
])

summary={
    "project":"MetaKV",
    "phase":"P4A",
    "model":"Qwen3-14B",
    "hardware":
        torch.cuda.get_device_name(0),
    "group_size":GROUP_SIZE,
    "bc":BC,
    "bf":BF,
    "metadata_bits":32,
    "prompt_count":len(PROMPTS),
    "real_scale_sample_count":int(z.size),
    "observed_log2_min":observed_min,
    "observed_log2_max":observed_max,
    "deployment_calibrated_zmin":ZMIN,
    "deployment_calibrated_zmax":ZMAX,
    "clean_log2_theorem_bound":bound,
    "max_log2_abs_error":max_z_err,
    "mean_log2_abs_error":mean_z_err,
    "max_scale_abs_error":max_scale_abs,
    "max_scale_rel_error":max_scale_rel,
    "clean_theorem_bound_pass":bound_pass,
    "no_clip_pass":no_clip,
    "codec_finite_pass":finite_pass,
    "deployment_interval_derived_from_real_qwen_scale_corpus":True,
    "historical_n3_representation_claim":False,
    "global_optimality_claim":False,
    "fault_injection_performed":False,
    "physical_hbm_mapping_applied":False,
    "end_to_end_quantized_inference_claim":False,
    "p4a_final_pass":final_pass,
}

(
    OUT/"p4a_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)

status=f"""====================================================================================================
METAKV P4A QWEN3-14B REAL-KV CLEAN CALIBRATION FINAL STATUS
====================================================================================================

MODEL=Qwen3-14B
GPU={torch.cuda.get_device_name(0)}
BACKEND=NVIDIA_CUDA

PROMPT_COUNT={len(PROMPTS)}

GROUP_SIZE={GROUP_SIZE}

BC={BC}
BF={BF}
METADATA_BITS=32

REAL_SCALE_SAMPLE_COUNT={z.size}

OBSERVED_LOG2_MIN={observed_min}
OBSERVED_LOG2_MAX={observed_max}

DEPLOYMENT_CALIBRATED_ZMIN={ZMIN}
DEPLOYMENT_CALIBRATED_ZMAX={ZMAX}

CLEAN_LOG2_THEOREM_BOUND={bound}

MAX_LOG2_ABS_ERROR={max_z_err}
MEAN_LOG2_ABS_ERROR={mean_z_err}

MAX_SCALE_ABS_ERROR={max_scale_abs}
MAX_SCALE_REL_ERROR={max_scale_rel}

CLEAN_THEOREM_BOUND_PASS={bound_pass}
NO_CLIP_PASS={no_clip}
CODEC_FINITE_PASS={finite_pass}

DEPLOYMENT_INTERVAL_DERIVED_FROM_REAL_QWEN_SCALE_CORPUS=True

HISTORICAL_N3_REPRESENTATION_CLAIM=False
GLOBAL_OPTIMALITY_CLAIM=False

FAULT_INJECTION_PERFORMED=False
PHYSICAL_HBM_MAPPING_APPLIED=False
END_TO_END_QUANTIZED_INFERENCE_CLAIM=False

P4A_FINAL_PASS={final_pass}
====================================================================================================
"""

(
    OUT/"P4A_FINAL_STATUS.txt"
).write_text(
    status,
    encoding="utf-8"
)

print()
print(status)

print(
    "P4A_ELAPSED_SEC=",
    time.time()-t0,
    sep=""
)
