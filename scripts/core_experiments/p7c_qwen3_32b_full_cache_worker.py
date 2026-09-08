from pathlib import Path
import ast
import gc
import json
import math
import os

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache


MODEL_PATH=Path(
    os.environ["METAKV_MODEL_PATH"]
)

SOURCE_SCRIPT=Path(
    os.environ["METAKV_SOURCE_SCRIPT"]
)

PROMPT_ID=int(
    os.environ["METAKV_PROMPT_ID"]
)

OUTROOT=Path(
    os.environ["METAKV_OUTROOT"]
)


GROUP_SIZE=32

ZMIN=-13.0
ZMAX=5.0

BC=19
BF=13

FMAX=(1<<BF)-1
COARSE_MASK=(1<<BC)-1

assert BC+BF==32

OUTROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# RECOVER EXACT MATCHED P4 PROMPT CORPUS
# ======================================================================================

source_text=SOURCE_SCRIPT.read_text(
    encoding="utf-8",
    errors="replace"
)

tree=ast.parse(
    source_text
)

candidates=[]


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
            [node.target.id]
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

            candidates.append(
                (
                    name,
                    list(obj)
                )
            )


if not candidates:

    raise RuntimeError(
        "unable to recover frozen P4 prompts"
    )


candidates.sort(
    key=lambda x:(
        abs(
            len(x[1])-8
        ),
        x[0]
    )
)


PROMPT_VARIABLE,PROMPTS=candidates[0]

PROMPTS=PROMPTS[:8]


if len(PROMPTS)!=8:

    raise RuntimeError(
        "expected eight prompts"
    )


PROMPT=PROMPTS[
    PROMPT_ID
]


# ======================================================================================
# CACHE
# ======================================================================================

def cache_to_legacy(
    cache
):

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

        return tuple(
            (
                layer.keys,
                layer.values,
            )
            for layer in cache.layers
        )


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
                cache.value_cache,
            )
        )


    raise RuntimeError(
        f"unsupported cache type {type(cache)}"
    )


def legacy_to_dynamic(
    legacy
):

    if hasattr(
        DynamicCache,
        "from_legacy_cache"
    ):

        try:

            return DynamicCache.from_legacy_cache(
                tuple(
                    legacy
                )
            )

        except Exception:
            pass


    cache=DynamicCache()

    for layer_idx,(k,v) in enumerate(
        legacy
    ):

        cache.update(
            k,
            v,
            layer_idx,
        )

    return cache


def clone_legacy(
    legacy
):

    return tuple(
        (
            k.detach().clone().contiguous(),
            v.detach().clone().contiguous(),
        )
        for k,v in legacy
    )


# ======================================================================================
# HA-FBMS
# ======================================================================================

def ha_encode_z(
    z
):

    u=(
        (z-ZMIN)
        *
        (BC+1)
        /
        float(
            ZMAX-ZMIN
        )
    )

    u=min(
        max(
            u,
            0.0
        ),
        float(
            BC+1
        )
    )

    coarse=int(
        math.floor(
            u
        )
    )

    if coarse>=BC+1:

        coarse=BC
        fine=FMAX

    else:

        fine=int(
            np.rint(
                (
                    u-coarse
                )
                *
                FMAX
            )
        )

        coarse=min(
            max(
                coarse,
                0
            ),
            BC
        )

        fine=min(
            max(
                fine,
                0
            ),
            FMAX
        )


    therm=(
        0
        if coarse==0
        else
        (1<<coarse)-1
    )


    return (
        int(
            therm
        )
        |
        (
            int(
                fine
            )
            <<
            BC
        )
    )


def ha_decode_word(
    word
):

    coarse=(
        int(
            word
        )
        &
        COARSE_MASK
    )

    count=int(
        coarse
    ).bit_count()

    fine=(
        int(
            word
        )
        >>
        BC
    ) & FMAX


    u=(
        float(
            count
        )
        +
        float(
            fine
        )
        /
        float(
            FMAX
        )
    )


    z=(
        ZMIN
        +
        (
            float(
                ZMAX-ZMIN
            )
            /
            float(
                BC+1
            )
        )
        *
        u
    )


    return (
        z,
        2.0**z
    )


# ======================================================================================
# FULL CACHE QUANT-DEQUANT
# ======================================================================================

@torch.no_grad()
def quant_dequant_tensor(
    tensor,
    ha_scale
):

    original_shape=tensor.shape
    dtype=tensor.dtype
    device=tensor.device


    flat=(
        tensor
        .detach()
        .float()
        .contiguous()
        .view(-1)
    )


    out=torch.empty_like(
        flat
    )


    stats={
        "values":0,
        "groups":0,
        "scale_clip":0,
        "z_min":math.inf,
        "z_max":-math.inf,
        "max_scale_rel_error":0.0,
        "q_min":999,
        "q_max":-999,
    }


    n=int(
        flat.numel()
    )


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


        z=math.log2(
            scale
        )


        stats[
            "z_min"
        ]=min(
            stats[
                "z_min"
            ],
            z
        )


        stats[
            "z_max"
        ]=max(
            stats[
                "z_max"
            ],
            z
        )


        if (
            z<ZMIN
            or
            z>ZMAX
        ):

            stats[
                "scale_clip"
            ] += 1


        used_scale=scale


        if ha_scale:

            word=ha_encode_z(
                z
            )

            _z2,used_scale=ha_decode_word(
                word
            )


            rel=abs(
                used_scale-scale
            ) / max(
                scale,
                np.finfo(
                    np.float64
                ).tiny
            )


            stats[
                "max_scale_rel_error"
            ]=max(
                stats[
                    "max_scale_rel_error"
                ],
                rel
            )


        # Payload quantization always uses original maxabs/7 scale.
        q=torch.round(
            x/scale
        ).clamp(
            -8,
            7
        )


        if q.numel()>0:

            stats[
                "q_min"
            ]=min(
                stats[
                    "q_min"
                ],
                int(
                    q.min().item()
                )
            )

            stats[
                "q_max"
            ]=max(
                stats[
                    "q_max"
                ],
                int(
                    q.max().item()
                )
            )


        out[
            start:end
        ]=(
            q
            *
            float(
                used_scale
            )
        )


        stats[
            "groups"
        ] += 1

        stats[
            "values"
        ] += (
            end-start
        )


    return (
        out
        .view(
            original_shape
        )
        .to(
            device=device,
            dtype=dtype
        )
        .contiguous(),
        stats,
    )


@torch.no_grad()
def build_full_cache(
    legacy,
    ha_scale
):

    result=[]

    total={
        "layers":len(legacy),
        "kv_tensors":0,
        "groups":0,
        "values":0,
        "scale_clip":0,
        "z_min":math.inf,
        "z_max":-math.inf,
        "max_scale_rel_error":0.0,
        "q_min":999,
        "q_max":-999,
    }


    for k,v in legacy:

        pair=[]


        for tensor in (
            k,
            v
        ):

            dq,s=quant_dequant_tensor(
                tensor,
                ha_scale
            )


            if not torch.isfinite(
                dq.float()
            ).all():

                raise RuntimeError(
                    "nonfinite replay tensor"
                )


            pair.append(
                dq
            )


            total[
                "kv_tensors"
            ] += 1

            total[
                "groups"
            ] += s[
                "groups"
            ]

            total[
                "values"
            ] += s[
                "values"
            ]

            total[
                "scale_clip"
            ] += s[
                "scale_clip"
            ]

            total[
                "z_min"
            ]=min(
                total[
                    "z_min"
                ],
                s[
                    "z_min"
                ]
            )

            total[
                "z_max"
            ]=max(
                total[
                    "z_max"
                ],
                s[
                    "z_max"
                ]
            )

            total[
                "max_scale_rel_error"
            ]=max(
                total[
                    "max_scale_rel_error"
                ],
                s[
                    "max_scale_rel_error"
                ]
            )

            total[
                "q_min"
            ]=min(
                total[
                    "q_min"
                ],
                s[
                    "q_min"
                ]
            )

            total[
                "q_max"
            ]=max(
                total[
                    "q_max"
                ],
                s[
                    "q_max"
                ]
            )


        result.append(
            tuple(
                pair
            )
        )


    return (
        tuple(
            result
        ),
        total,
    )


# ======================================================================================
# METRICS
# ======================================================================================

def metrics(
    ref_logits,
    test_logits
):

    ref=ref_logits.float()
    test=test_logits.float()


    finite=bool(
        torch.isfinite(
            test
        ).all().item()
    )


    if not finite:

        return {
            "finite":False,
            "top1_same":False,
            "kl":None,
            "max_abs":None,
            "mean_abs":None,
        }


    ref_logp=torch.log_softmax(
        ref,
        dim=-1
    )

    test_logp=torch.log_softmax(
        test,
        dim=-1
    )

    p=torch.exp(
        ref_logp
    )


    kl=float(
        torch.sum(
            p
            *
            (
                ref_logp-test_logp
            ),
            dim=-1
        ).mean().item()
    )


    delta=(
        test-ref
    ).abs()


    return {
        "finite":True,

        "top1_same":
            bool(
                torch.equal(
                    ref.argmax(
                        dim=-1
                    ),
                    test.argmax(
                        dim=-1
                    )
                )
            ),

        "kl":
            kl,

        "max_abs":
            float(
                delta.max().item()
            ),

        "mean_abs":
            float(
                delta.mean().item()
            ),
    }


# ======================================================================================
# LOAD MODEL
# ======================================================================================

torch.manual_seed(
    20260904
)

torch.cuda.manual_seed_all(
    20260904
)


tokenizer=AutoTokenizer.from_pretrained(
    str(
        MODEL_PATH
    ),
    local_files_only=True,
    trust_remote_code=True,
)


model=AutoModelForCausalLM.from_pretrained(
    str(
        MODEL_PATH
    ),
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"":0},
    low_cpu_mem_usage=True,
)


model.eval()


# ======================================================================================
# PROMPT FORWARD
# ======================================================================================

inputs=tokenizer(
    PROMPT,
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

    initial=model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )


legacy=cache_to_legacy(
    initial.past_key_values
)


if len(legacy)!=64:

    raise RuntimeError(
        f"expected 64 layers, got {len(legacy)}"
    )


next_token=initial.logits[
    :,
    -1:,
    :
].argmax(
    dim=-1
)


past_len=int(
    legacy[0][0].shape[-2]
)


position_ids=torch.tensor(
    [[past_len]],
    device=model.device,
    dtype=torch.long
)


cache_position=torch.tensor(
    [past_len],
    device=model.device,
    dtype=torch.long
)


# ======================================================================================
# BF16 REFERENCE
# ======================================================================================

with torch.inference_mode():

    ref_cache=legacy_to_dynamic(
        clone_legacy(
            legacy
        )
    )

    ref=model(
        input_ids=next_token,
        past_key_values=ref_cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    )


ref_logits=ref.logits[
    :,
    -1,
    :
].detach()


del ref_cache
del ref

torch.cuda.empty_cache()


# ======================================================================================
# RAW INT4 FULL CACHE
# ======================================================================================

raw_legacy,raw_stats=build_full_cache(
    legacy,
    ha_scale=False
)


with torch.inference_mode():

    raw_cache=legacy_to_dynamic(
        raw_legacy
    )

    raw_out=model(
        input_ids=next_token,
        past_key_values=raw_cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    )


raw_logits=raw_out.logits[
    :,
    -1,
    :
].detach()


raw_metrics=metrics(
    ref_logits,
    raw_logits
)


del raw_cache
del raw_out
del raw_legacy

torch.cuda.empty_cache()


# ======================================================================================
# HA INT4 FULL CACHE
# ======================================================================================

ha_legacy,ha_stats=build_full_cache(
    legacy,
    ha_scale=True
)


with torch.inference_mode():

    ha_cache=legacy_to_dynamic(
        ha_legacy
    )

    ha_out=model(
        input_ids=next_token,
        past_key_values=ha_cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    )


ha_logits=ha_out.logits[
    :,
    -1,
    :
].detach()


ha_metrics=metrics(
    ref_logits,
    ha_logits
)


ha_vs_raw=metrics(
    raw_logits,
    ha_logits
)


# ======================================================================================
# FINAL
# ======================================================================================

expected_kv_tensors=64*2


worker_pass=all([
    raw_metrics["finite"],
    ha_metrics["finite"],
    ha_vs_raw["finite"],

    raw_stats["layers"]==64,
    ha_stats["layers"]==64,

    raw_stats["kv_tensors"]==expected_kv_tensors,
    ha_stats["kv_tensors"]==expected_kv_tensors,

    raw_stats["scale_clip"]==0,
    ha_stats["scale_clip"]==0,

    raw_stats["q_min"]>=-8,
    raw_stats["q_max"]<=7,

    ha_stats["q_min"]>=-8,
    ha_stats["q_max"]<=7,
])


payload={
    "model":"Qwen3-32B",

    "prompt_id":
        PROMPT_ID,

    "prompt_variable":
        PROMPT_VARIABLE,

    "input_tokens":
        int(
            input_ids.shape[-1]
        ),

    "layer_count":
        len(
            legacy
        ),

    "next_token_id":
        int(
            next_token.item()
        ),

    "group_size":
        GROUP_SIZE,

    "int4_range":
        "[-8,7]",

    "interval":[
        ZMIN,
        ZMAX
    ],

    "raw_stats":
        raw_stats,

    "ha_stats":
        ha_stats,

    "raw_vs_bf16":
        raw_metrics,

    "ha_vs_bf16":
        ha_metrics,

    "ha_vs_raw":
        ha_vs_raw,

    "full_cache_replay":
        True,

    "all_layers":
        True,

    "all_kv_tensors":
        True,

    "fault_injection":
        False,

    "worker_pass":
        worker_pass,
}


(
    OUTROOT
    /
    "P7C_RESULT.json"
).write_text(
    json.dumps(
        payload,
        indent=2
    ),
    encoding="utf-8"
)


status=f"""====================================================================================================
METAKV P7C QWEN3-32B FULL-CACHE INT4 CLEAN WORKER STATUS
====================================================================================================

MODEL=QWEN3-32B
PROMPT_ID={PROMPT_ID}

INPUT_TOKENS={int(input_ids.shape[-1])}

LAYER_COUNT={len(legacy)}
KV_TENSOR_COUNT={ha_stats['kv_tensors']}

GROUP_SIZE={GROUP_SIZE}
INT4_RANGE=[-8,7]

HA_INTERVAL=[{ZMIN},{ZMAX}]

FULL_CACHE_REPLAY=True
ALL_LAYERS=True
ALL_KV_TENSORS=True

RAW_GROUP_COUNT={raw_stats['groups']}
RAW_SCALE_CLIP={raw_stats['scale_clip']}
RAW_Q_MIN={raw_stats['q_min']}
RAW_Q_MAX={raw_stats['q_max']}

HA_GROUP_COUNT={ha_stats['groups']}
HA_SCALE_CLIP={ha_stats['scale_clip']}
HA_Q_MIN={ha_stats['q_min']}
HA_Q_MAX={ha_stats['q_max']}
HA_MAX_SCALE_REL_ERROR={ha_stats['max_scale_rel_error']}

RAW_FINITE={raw_metrics['finite']}
RAW_TOP1_SAME={raw_metrics['top1_same']}
RAW_KL={raw_metrics['kl']}
RAW_MAXLOGIT={raw_metrics['max_abs']}

HA_FINITE={ha_metrics['finite']}
HA_TOP1_SAME={ha_metrics['top1_same']}
HA_KL={ha_metrics['kl']}
HA_MAXLOGIT={ha_metrics['max_abs']}

HA_RAW_FINITE={ha_vs_raw['finite']}
HA_RAW_TOP1_SAME={ha_vs_raw['top1_same']}
HA_RAW_KL={ha_vs_raw['kl']}
HA_RAW_MAXLOGIT={ha_vs_raw['max_abs']}

FAULT_INJECTION=False

P7C_WORKER_PASS={worker_pass}
====================================================================================================
"""


(
    OUTROOT
    /
    "P7C_WORKER_STATUS.txt"
).write_text(
    status,
    encoding="utf-8"
)


print(
    f"P7C_RESULT "
    f"PROMPT={PROMPT_ID} "
    f"LAYERS={len(legacy)} "
    f"GROUPS={ha_stats['groups']} "
    f"CLIP={ha_stats['scale_clip']} "
    f"RAW_TOP1={raw_metrics['top1_same']} "
    f"RAW_KL={raw_metrics['kl']} "
    f"RAW_MAXLOGIT={raw_metrics['max_abs']} "
    f"HA_TOP1={ha_metrics['top1_same']} "
    f"HA_KL={ha_metrics['kl']} "
    f"HA_MAXLOGIT={ha_metrics['max_abs']} "
    f"HA_RAW_TOP1={ha_vs_raw['top1_same']} "
    f"HA_RAW_KL={ha_vs_raw['kl']} "
    f"HA_RAW_MAXLOGIT={ha_vs_raw['max_abs']} "
    f"PASS={worker_pass}"
)


print(
    "P7C_WORKER_PASS=",
    worker_pass,
    sep=""
)


del model
gc.collect()

torch.cuda.empty_cache()
