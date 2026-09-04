from pathlib import Path
import ast
import csv
import gc
import hashlib
import json
import math
import sys

import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL_PATH=Path(sys.argv[1])
P4_SOURCE=Path(sys.argv[2])
RUN=Path(sys.argv[3])

OUT=RUN/"output"
FREEZE=RUN/"freeze"

OUT.mkdir(parents=True,exist_ok=True)
FREEZE.mkdir(parents=True,exist_ok=True)


GROUP_SIZE=32

BC=19
BF=13
BITS=32

FMAX=(1<<BF)-1
COARSE_MASK=(1<<BC)-1

assert BC+BF==32


# ======================================================================================
# HASH
# ======================================================================================

def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ======================================================================================
# RECOVER EXACT FROZEN P4 PROMPT LIST
# ======================================================================================

source_text=P4_SOURCE.read_text(
    encoding="utf-8",
    errors="replace"
)

tree=ast.parse(
    source_text
)

prompt_candidates=[]


for node in ast.walk(tree):

    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        )
    ):
        continue

    if isinstance(
        node,
        ast.Assign
    ):

        names=[
            t.id
            for t in node.targets
            if isinstance(
                t,
                ast.Name
            )
        ]

        value=node.value

    else:

        names=(
            [
                node.target.id
            ]
            if isinstance(
                node.target,
                ast.Name
            )
            else
            []
        )

        value=node.value


    for name in names:

        if "prompt" not in name.lower():
            continue

        try:
            obj=ast.literal_eval(
                value
            )
        except Exception:
            continue

        if (
            isinstance(
                obj,
                (list,tuple)
            )
            and
            len(obj)>=8
            and
            all(
                isinstance(x,str)
                for x in obj
            )
        ):

            prompt_candidates.append(
                (
                    name,
                    list(obj)
                )
            )


if not prompt_candidates:

    raise RuntimeError(
        "unable to recover frozen P4 prompt list"
    )


prompt_candidates.sort(
    key=lambda x:(
        abs(
            len(x[1])-8
        ),
        x[0]
    )
)


PROMPT_VARIABLE,PROMPTS=prompt_candidates[0]

PROMPTS=PROMPTS[:8]


if len(PROMPTS)!=8:

    raise RuntimeError(
        f"expected exactly 8 prompts, got {len(PROMPTS)}"
    )


print(
    "P4_PROMPT_SOURCE=",
    P4_SOURCE,
    sep=""
)

print(
    "P4_PROMPT_SOURCE_SHA256=",
    sha256(
        P4_SOURCE
    ),
    sep=""
)

print(
    "PROMPT_VARIABLE=",
    PROMPT_VARIABLE,
    sep=""
)

print(
    "PROMPT_COUNT=",
    len(PROMPTS),
    sep=""
)


for i,p in enumerate(PROMPTS):

    print(
        f"P7A_R1_PROMPT_TEXT "
        f"ID={i} "
        f"TEXT={repr(p)}"
    )


# ======================================================================================
# CACHE ACCESS
# ======================================================================================

def cache_to_legacy(cache):

    if hasattr(
        cache,
        "to_legacy_cache"
    ):

        try:

            legacy=cache.to_legacy_cache()

            if legacy is not None:

                return tuple(
                    (
                        x[0],
                        x[1],
                    )
                    for x in legacy
                )

        except Exception:
            pass


    if hasattr(
        cache,
        "layers"
    ):

        result=[]

        for layer in cache.layers:

            if not (
                hasattr(
                    layer,
                    "keys"
                )
                and
                hasattr(
                    layer,
                    "values"
                )
            ):

                raise RuntimeError(
                    "DynamicCache layer missing keys/values"
                )

            result.append(
                (
                    layer.keys,
                    layer.values,
                )
            )

        return tuple(result)


    if (
        hasattr(
            cache,
            "key_cache"
        )
        and
        hasattr(
            cache,
            "value_cache"
        )
    ):

        return tuple(
            zip(
                cache.key_cache,
                cache.value_cache
            )
        )


    raise RuntimeError(
        f"unsupported cache type {type(cache)}"
    )


# ======================================================================================
# REAL G32 MAXABS/7 SCALE EXTRACTION
# ======================================================================================

@torch.no_grad()
def tensor_group_scales(
    tensor
):

    flat=(
        tensor
        .detach()
        .float()
        .contiguous()
        .view(-1)
    )

    n=int(
        flat.numel()
    )

    scales=[]

    for start in range(
        0,
        n,
        GROUP_SIZE
    ):

        end=min(
            start+GROUP_SIZE,
            n
        )

        x=flat[
            start:end
        ]

        maxabs=float(
            x.abs().max().item()
        )

        if (
            maxabs==0.0
            or
            not math.isfinite(
                maxabs
            )
        ):

            scale=1.0

        else:

            scale=maxabs/7.0

        scales.append(
            scale
        )


    return np.asarray(
        scales,
        dtype=np.float64
    )


# ======================================================================================
# HA-FBMS
# ======================================================================================

def encode_z(
    z,
    zmin,
    zmax
):

    z=np.asarray(
        z,
        dtype=np.float64
    )

    u=(
        (z-zmin)
        *
        (BC+1)
        /
        float(
            zmax-zmin
        )
    )

    u=np.clip(
        u,
        0.0,
        float(
            BC+1
        )
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
        (
            u-coarse
        )
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
        np.uint32(
            BC
        )
    )

    return words


def decode_z(
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
        np.uint32(
            BC
        )
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
        float(
            FMAX
        )
    )

    return (
        zmin
        +
        (
            float(
                zmax-zmin
            )
            /
            float(
                BC+1
            )
        )
        *
        u
    )


# ======================================================================================
# CONFIG PROVENANCE BEFORE MODEL LOAD
# ======================================================================================

cfg=AutoConfig.from_pretrained(
    str(
        MODEL_PATH
    ),
    local_files_only=True,
    trust_remote_code=True,
)


print()
print("="*140)
print("QWEN3-32B CONFIG PROVENANCE")
print("="*140)

print(
    "MODEL_PATH=",
    MODEL_PATH,
    sep=""
)

print(
    "MODEL_TYPE=",
    getattr(
        cfg,
        "model_type",
        None
    ),
    sep=""
)

print(
    "ARCHITECTURES=",
    getattr(
        cfg,
        "architectures",
        None
    ),
    sep=""
)

print(
    "NUM_HIDDEN_LAYERS=",
    getattr(
        cfg,
        "num_hidden_layers",
        None
    ),
    sep=""
)

print(
    "HIDDEN_SIZE=",
    getattr(
        cfg,
        "hidden_size",
        None
    ),
    sep=""
)

print(
    "NUM_ATTENTION_HEADS=",
    getattr(
        cfg,
        "num_attention_heads",
        None
    ),
    sep=""
)

print(
    "NUM_KEY_VALUE_HEADS=",
    getattr(
        cfg,
        "num_key_value_heads",
        None
    ),
    sep=""
)

print(
    "QUANTIZATION_CONFIG=",
    getattr(
        cfg,
        "quantization_config",
        None
    ),
    sep=""
)


if getattr(
    cfg,
    "model_type",
    None
) != "qwen3":

    raise RuntimeError(
        "model_type is not qwen3"
    )


if int(
    getattr(
        cfg,
        "num_hidden_layers",
        -1
    )
) != 64:

    raise RuntimeError(
        "expected Qwen3-32B 64 layers"
    )


if getattr(
    cfg,
    "quantization_config",
    None
) is not None:

    raise RuntimeError(
        "unexpected quantization_config; "
        "P7A-R1 requires non-derived original model"
    )


# ======================================================================================
# LOAD TOKENIZER + MODEL
# ======================================================================================

tokenizer=AutoTokenizer.from_pretrained(
    str(
        MODEL_PATH
    ),
    local_files_only=True,
    trust_remote_code=True,
)


torch.manual_seed(
    20260904
)

torch.cuda.manual_seed_all(
    20260904
)


before_alloc=torch.cuda.memory_allocated(
    0
) / 1024**3


model=AutoModelForCausalLM.from_pretrained(
    str(
        MODEL_PATH
    ),
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={
        "":0
    },
    low_cpu_mem_usage=True,
)


model.eval()


after_alloc=torch.cuda.memory_allocated(
    0
) / 1024**3


parameter_count=sum(
    p.numel()
    for p in model.parameters()
)


print()
print("="*140)
print("MODEL LOAD")
print("="*140)

print(
    "MODEL_CLASS=",
    model.__class__.__name__,
    sep=""
)

print(
    "PARAMETER_COUNT=",
    parameter_count,
    sep=""
)

print(
    "GPU=",
    torch.cuda.get_device_name(
        0
    ),
    sep=""
)

print(
    "GPU_TOTAL_GIB=",
    torch.cuda.get_device_properties(
        0
    ).total_memory/1024**3,
    sep=""
)

print(
    "GPU_ALLOC_BEFORE_GIB=",
    before_alloc,
    sep=""
)

print(
    "GPU_ALLOC_AFTER_MODEL_GIB=",
    after_alloc,
    sep=""
)


# ======================================================================================
# REAL KV — MATCHED 8 PROMPTS
# ======================================================================================

all_z=[]

prompt_rows=[]

tensor_rows=[]


for pid,prompt in enumerate(
    PROMPTS
):

    inputs=tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids=inputs[
        "input_ids"
    ].to(
        model.device
    )

    attention_mask=inputs.get(
        "attention_mask"
    )

    if attention_mask is not None:

        attention_mask=attention_mask.to(
            model.device
        )


    with torch.inference_mode():

        result=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )


    if result.past_key_values is None:

        raise RuntimeError(
            f"prompt {pid}: no KV cache"
        )


    legacy=cache_to_legacy(
        result.past_key_values
    )


    if len(legacy)!=64:

        raise RuntimeError(
            f"prompt {pid}: expected 64 layers, "
            f"got {len(legacy)}"
        )


    prompt_z=[]


    for layer,(k,v) in enumerate(
        legacy
    ):

        for kv,tensor in (
            ("K",k),
            ("V",v),
        ):

            if not torch.isfinite(
                tensor.float()
            ).all():

                raise RuntimeError(
                    f"nonfinite KV "
                    f"prompt={pid} "
                    f"layer={layer} "
                    f"kv={kv}"
                )


            scales=tensor_group_scales(
                tensor
            )

            z=np.log2(
                scales
            )


            if not np.isfinite(
                z
            ).all():

                raise RuntimeError(
                    f"nonfinite z "
                    f"prompt={pid} "
                    f"layer={layer} "
                    f"kv={kv}"
                )


            prompt_z.append(
                z
            )

            all_z.append(
                z
            )


            tensor_rows.append({
                "prompt_id":
                    pid,

                "layer":
                    layer,

                "kv":
                    kv,

                "shape":
                    str(
                        tuple(
                            tensor.shape
                        )
                    ),

                "values":
                    int(
                        tensor.numel()
                    ),

                "groups":
                    int(
                        len(
                            z
                        )
                    ),

                "z_min":
                    float(
                        z.min()
                    ),

                "z_max":
                    float(
                        z.max()
                    ),
            })


    pz=np.concatenate(
        prompt_z
    )


    prompt_rows.append({
        "prompt_id":
            pid,

        "input_tokens":
            int(
                input_ids.shape[-1]
            ),

        "layer_count":
            len(
                legacy
            ),

        "scale_count":
            int(
                len(
                    pz
                )
            ),

        "z_min":
            float(
                pz.min()
            ),

        "z_max":
            float(
                pz.max()
            ),
    })


    print(
        f"P7A_R1_PROMPT "
        f"PROMPT={pid} "
        f"TOKENS={int(input_ids.shape[-1])} "
        f"LAYERS={len(legacy)} "
        f"SCALES={len(pz)} "
        f"ZMIN={float(pz.min())} "
        f"ZMAX={float(pz.max())}"
    )


    del result
    del legacy
    del pz

    gc.collect()
    torch.cuda.empty_cache()


# ======================================================================================
# MERGED SCALE CORPUS
# ======================================================================================

z=np.concatenate(
    all_z
).astype(
    np.float64
)


scale_count=int(
    z.size
)


observed_min=float(
    z.min()
)

observed_max=float(
    z.max()
)


zmin=float(
    math.floor(
        observed_min
    )
)

zmax=float(
    math.ceil(
        observed_max
    )
)


width=float(
    zmax-zmin
)


clip_count=int(
    np.count_nonzero(
        (z<zmin)
        |
        (z>zmax)
    )
)


# ======================================================================================
# CLEAN HA-FBMS
# ======================================================================================

words=encode_z(
    z,
    zmin,
    zmax
)


decoded=decode_z(
    words,
    zmin,
    zmax
)


zerr=np.abs(
    decoded-z
)


theorem_bound=(
    width
    /
    float(
        BC+1
    )
    /
    (
        2.0
        *
        float(
            FMAX
        )
    )
)


max_z_error=float(
    zerr.max()
)

mean_z_error=float(
    zerr.mean()
)


theorem_pass=(
    max_z_error
    <=
    theorem_bound*(1+1e-9)
    +
    1e-12
)


real_scale=np.exp2(
    z
)

decoded_scale=np.exp2(
    decoded
)


scale_abs=np.abs(
    decoded_scale-real_scale
)


scale_rel=(
    scale_abs
    /
    np.maximum(
        real_scale,
        np.finfo(
            np.float64
        ).tiny
    )
)


max_scale_abs=float(
    scale_abs.max()
)

mean_scale_abs=float(
    scale_abs.mean()
)

max_scale_rel=float(
    scale_rel.max()
)

mean_scale_rel=float(
    scale_rel.mean()
)


# ======================================================================================
# LOGICAL ONE-HOT 32-BIT HA-FAULT GEOMETRY
# ======================================================================================

fault_event_count=(
    scale_count*32
)


fault_invalid=0

fault_max_a=1.0

fault_ge2=0
fault_ge10=0
fault_ge100=0


for bit in range(
    32
):

    mutated=(
        words
        ^
        np.uint32(
            1<<bit
        )
    )


    fault_z=decode_z(
        mutated,
        zmin,
        zmax
    )


    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore"
    ):

        fault_scale=np.exp2(
            fault_z
        )


    valid=(
        np.isfinite(
            fault_scale
        )
        &
        (
            fault_scale>0
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
            fault_scale[
                valid
            ]
            /
            real_scale[
                valid
            ]
        )


        amp=np.maximum(
            ratio,
            1.0/ratio
        )


        fault_max_a=max(
            fault_max_a,
            float(
                amp.max()
            )
        )


        fault_ge2 += int(
            np.count_nonzero(
                amp>=2.0
            )
        )

        fault_ge10 += int(
            np.count_nonzero(
                amp>=10.0
            )
        )

        fault_ge100 += int(
            np.count_nonzero(
                amp>=100.0
            )
        )


# ======================================================================================
# SAVE CORPUS
# ======================================================================================

np.save(
    OUT/"p7a_r1_real_log2_scales.npy",
    z
)


with (
    OUT
    /
    "p7a_r1_prompt_summary.csv"
).open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w=csv.DictWriter(
        f,
        fieldnames=list(
            prompt_rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(
        prompt_rows
    )


with (
    OUT
    /
    "p7a_r1_real_scale_records.csv"
).open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w=csv.DictWriter(
        f,
        fieldnames=list(
            tensor_rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(
        tensor_rows
    )


# ======================================================================================
# VALIDITY
# ======================================================================================

prompt_count_pass=(
    len(
        prompt_rows
    )==8
)


all_64_layers=all(
    x[
        "layer_count"
    ]==64
    for x in prompt_rows
)


tensor_record_count_expected=(
    8*64*2
)


tensor_record_count_pass=(
    len(
        tensor_rows
    )
    ==
    tensor_record_count_expected
)


zero_clip=(
    clip_count==0
)


ha_geometry_pass=all([
    fault_invalid==0,
    fault_ge2==0,
])


final_pass=all([
    prompt_count_pass,
    all_64_layers,
    tensor_record_count_pass,
    zero_clip,
    theorem_pass,
    ha_geometry_pass,
])


# ======================================================================================
# FINAL ARTIFACTS
# ======================================================================================

payload={
    "phase":
        "P7A_R1",

    "model":
        "Qwen3-32B",

    "model_path":
        str(
            MODEL_PATH
        ),

    "model_type":
        getattr(
            cfg,
            "model_type",
            None
        ),

    "architectures":
        getattr(
            cfg,
            "architectures",
            None
        ),

    "parameter_count":
        parameter_count,

    "num_hidden_layers":
        getattr(
            cfg,
            "num_hidden_layers",
            None
        ),

    "quantization_config":
        getattr(
            cfg,
            "quantization_config",
            None
        ),

    "prompt_source":
        str(
            P4_SOURCE
        ),

    "prompt_source_sha256":
        sha256(
            P4_SOURCE
        ),

    "prompt_variable":
        PROMPT_VARIABLE,

    "matched_p4_prompt_corpus":
        True,

    "prompt_count":
        len(
            prompt_rows
        ),

    "tensor_record_count":
        len(
            tensor_rows
        ),

    "group_size":
        GROUP_SIZE,

    "int4_range":
        "[-8,7]",

    "real_scale_count":
        scale_count,

    "observed_z_min":
        observed_min,

    "observed_z_max":
        observed_max,

    "deployment_zmin":
        zmin,

    "deployment_zmax":
        zmax,

    "interval_width":
        width,

    "clip_count":
        clip_count,

    "bc":
        BC,

    "bf":
        BF,

    "metadata_bits":
        BITS,

    "theorem_bound":
        theorem_bound,

    "max_z_error":
        max_z_error,

    "mean_z_error":
        mean_z_error,

    "max_scale_abs_error":
        max_scale_abs,

    "mean_scale_abs_error":
        mean_scale_abs,

    "max_scale_relative_error":
        max_scale_rel,

    "mean_scale_relative_error":
        mean_scale_rel,

    "theorem_bound_pass":
        theorem_pass,

    "logical_single_bit_events":
        fault_event_count,

    "fault_invalid_scale_count":
        fault_invalid,

    "fault_max_amplification":
        fault_max_a,

    "fault_ge2_count":
        fault_ge2,

    "fault_ge10_count":
        fault_ge10,

    "fault_ge100_count":
        fault_ge100,

    "hardware_fault_probability_claim":
        False,

    "hardware_ber_claim":
        False,

    "p7a_r1_final_pass":
        final_pass,
}


(
    FREEZE
    /
    "P7A_R1_SUMMARY.json"
).write_text(
    json.dumps(
        payload,
        indent=2,
        default=str
    ),
    encoding="utf-8"
)


status=f"""====================================================================================================
METAKV P7A-R1 QWEN3-32B REAL-KV SCALE-UP FINAL STATUS
====================================================================================================

MODEL=QWEN3-32B
MODEL_PATH={MODEL_PATH}

MODEL_TYPE={getattr(cfg,'model_type',None)}
ARCHITECTURES={getattr(cfg,'architectures',None)}
PARAMETER_COUNT={parameter_count}

NUM_HIDDEN_LAYERS={getattr(cfg,'num_hidden_layers',None)}
QUANTIZATION_CONFIG={getattr(cfg,'quantization_config',None)}

MATCHED_P4_PROMPT_CORPUS=True
PROMPT_SOURCE={P4_SOURCE}
PROMPT_SOURCE_SHA256={sha256(P4_SOURCE)}
PROMPT_VARIABLE={PROMPT_VARIABLE}

PROMPT_COUNT={len(prompt_rows)}
PROMPT_COUNT_PASS={prompt_count_pass}

ALL_PROMPTS_64_LAYERS={all_64_layers}

TENSOR_RECORD_COUNT={len(tensor_rows)}
EXPECTED_TENSOR_RECORD_COUNT={tensor_record_count_expected}
TENSOR_RECORD_COUNT_PASS={tensor_record_count_pass}

GROUP_SIZE={GROUP_SIZE}
INT4_RANGE=[-8,7]

REAL_SCALE_COUNT={scale_count}

OBSERVED_Z_MIN={observed_min}
OBSERVED_Z_MAX={observed_max}

DEPLOYMENT_INTERVAL=[{zmin},{zmax}]
INTERVAL_WIDTH={width}

SCALE_CLIP_COUNT={clip_count}
ZERO_SCALE_CLIP={zero_clip}

BC={BC}
BF={BF}
METADATA_BITS={BITS}

THEOREM_BOUND={theorem_bound}
MAX_Z_ERROR={max_z_error}
MEAN_Z_ERROR={mean_z_error}

MAX_SCALE_ABS_ERROR={max_scale_abs}
MEAN_SCALE_ABS_ERROR={mean_scale_abs}

MAX_SCALE_RELATIVE_ERROR={max_scale_rel}
MEAN_SCALE_RELATIVE_ERROR={mean_scale_rel}

THEOREM_BOUND_PASS={theorem_pass}

LOGICAL_SINGLE_BIT_EVENTS={fault_event_count}

FAULT_INVALID_SCALE_COUNT={fault_invalid}
FAULT_MAX_AMPLIFICATION={fault_max_a}

FAULT_GE2_COUNT={fault_ge2}
FAULT_GE10_COUNT={fault_ge10}
FAULT_GE100_COUNT={fault_ge100}

HA_FAULT_GEOMETRY_PASS={ha_geometry_pass}

FAULT_PROBABILITY_CLAIM=False
HARDWARE_BER_CLAIM=False

P7A_R1_FINAL_PASS={final_pass}
====================================================================================================
"""


(
    FREEZE
    /
    "P7A_R1_FINAL_STATUS.txt"
).write_text(
    status,
    encoding="utf-8"
)


print()
print(status)


del model
gc.collect()

torch.cuda.empty_cache()
