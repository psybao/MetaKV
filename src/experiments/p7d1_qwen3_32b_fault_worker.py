from pathlib import Path
import ast
import csv
import gc
import json
import math
import os
import struct

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache


MODEL_PATH=Path(os.environ["METAKV_MODEL_PATH"])
SOURCE_SCRIPT=Path(os.environ["METAKV_SOURCE_SCRIPT"])
MANIFEST=Path(os.environ["METAKV_MANIFEST"])
PROMPT_ID=int(os.environ["METAKV_PROMPT_ID"])
OUTROOT=Path(os.environ["METAKV_OUTROOT"])

OUTROOT.mkdir(parents=True,exist_ok=True)

GROUP_SIZE=32

ZMIN=-13.0
ZMAX=5.0

BC=19
BF=13

FMAX=(1<<BF)-1
COARSE_MASK=(1<<BC)-1
BINARY_MAX=(1<<32)-1

assert BC+BF==32


# ======================================================================================
# PROMPTS
# ======================================================================================

text=SOURCE_SCRIPT.read_text(
    encoding="utf-8",
    errors="replace"
)

tree=ast.parse(text)

candidates=[]

for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign,ast.AnnAssign)
    ):
        continue

    if isinstance(node,ast.Assign):

        names=[
            t.id
            for t in node.targets
            if isinstance(t,ast.Name)
        ]

        value=node.value

    else:

        names=(
            [node.target.id]
            if isinstance(node.target,ast.Name)
            else
            []
        )

        value=node.value

    for name in names:

        if "prompt" not in name.lower():
            continue

        try:
            obj=ast.literal_eval(value)
        except Exception:
            continue

        if (
            isinstance(obj,(list,tuple))
            and
            len(obj)>=8
            and
            all(isinstance(x,str) for x in obj)
        ):
            candidates.append(
                (name,list(obj))
            )

if not candidates:
    raise RuntimeError(
        "unable to recover frozen P4 prompts"
    )

candidates.sort(
    key=lambda x:(
        abs(len(x[1])-8),
        x[0]
    )
)

PROMPT_VARIABLE,PROMPTS=candidates[0]

PROMPTS=PROMPTS[:8]

if len(PROMPTS)!=8:
    raise RuntimeError(
        "expected 8 frozen prompts"
    )

PROMPT=PROMPTS[PROMPT_ID]


# ======================================================================================
# TARGETS
# ======================================================================================

with MANIFEST.open(
    newline="",
    encoding="utf-8-sig"
) as f:

    all_targets=list(
        csv.DictReader(f)
    )

targets=[
    x
    for x in all_targets
    if int(x["prompt_id"])==PROMPT_ID
]

if len(targets)!=3:
    raise RuntimeError(
        f"prompt {PROMPT_ID}: expected 3 targets, got {len(targets)}"
    )


# ======================================================================================
# CACHE
# ======================================================================================

def cache_to_legacy(cache):

    if hasattr(cache,"to_legacy_cache"):

        try:

            x=cache.to_legacy_cache()

            if x is not None:

                return tuple(
                    (a[0],a[1])
                    for a in x
                )

        except Exception:
            pass


    if hasattr(cache,"layers"):

        return tuple(
            (layer.keys,layer.values)
            for layer in cache.layers
        )


    if (
        hasattr(cache,"key_cache")
        and
        hasattr(cache,"value_cache")
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


def legacy_to_dynamic(legacy):

    if hasattr(
        DynamicCache,
        "from_legacy_cache"
    ):

        try:

            return DynamicCache.from_legacy_cache(
                tuple(legacy)
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
            layer_idx
        )

    return cache


def clone_legacy(legacy):

    return tuple(
        (
            k.detach().clone().contiguous(),
            v.detach().clone().contiguous()
        )
        for k,v in legacy
    )


# ======================================================================================
# INT4 FULL-CACHE RAW BACKGROUND
# ======================================================================================

@torch.no_grad()
def quantize_group(x):

    xf=(
        x.detach()
        .float()
        .contiguous()
        .view(-1)
    )

    maxabs=float(
        xf.abs().max().item()
    )

    if (
        maxabs==0.0
        or
        not math.isfinite(maxabs)
    ):
        scale=1.0
    else:
        scale=maxabs/7.0

    q=torch.round(
        xf/scale
    ).clamp(
        -8,
        7
    )

    return q,scale


@torch.no_grad()
def build_raw_full_cache(legacy):

    out=[]

    total_groups=0
    total_values=0
    scale_clip=0
    zmin_seen=math.inf
    zmax_seen=-math.inf


    for k,v in legacy:

        pair=[]

        for tensor in (k,v):

            flat=(
                tensor.detach()
                .float()
                .contiguous()
                .view(-1)
            )

            dq=torch.empty_like(flat)

            for start in range(
                0,
                int(flat.numel()),
                GROUP_SIZE
            ):

                end=min(
                    start+GROUP_SIZE,
                    int(flat.numel())
                )

                q,scale=quantize_group(
                    flat[start:end]
                )

                z=math.log2(scale)

                zmin_seen=min(
                    zmin_seen,z
                )

                zmax_seen=max(
                    zmax_seen,z
                )

                if (
                    z<ZMIN
                    or
                    z>ZMAX
                ):
                    scale_clip+=1

                dq[start:end]=q*scale

                total_groups+=1
                total_values+=end-start


            pair.append(
                dq.view(
                    tensor.shape
                ).to(
                    device=tensor.device,
                    dtype=tensor.dtype
                ).contiguous()
            )


        out.append(tuple(pair))


    return tuple(out),{
        "groups":total_groups,
        "values":total_values,
        "scale_clip":scale_clip,
        "z_min":zmin_seen,
        "z_max":zmax_seen,
    }


# ======================================================================================
# FP32 SCALE
# ======================================================================================

def f32_to_word(x):

    return struct.unpack(
        "<I",
        struct.pack(
            "<f",
            np.float32(x)
        )
    )[0]


def word_to_f32(w):

    return float(
        struct.unpack(
            "<f",
            struct.pack(
                "<I",
                int(w)&0xffffffff
            )
        )[0]
    )


def fp32_codec(scale,bit):

    clean_word=f32_to_word(scale)

    fault_word=(
        clean_word
        ^
        (1<<bit)
    )

    clean_scale=word_to_f32(
        clean_word
    )

    fault_scale=word_to_f32(
        fault_word
    )

    return (
        clean_word,
        fault_word,
        clean_scale,
        fault_scale
    )


# ======================================================================================
# EXACT BINARY32-Z
# ======================================================================================

def binary32_encode(z):

    norm=(
        (z-ZMIN)
        /
        float(ZMAX-ZMIN)
    )

    norm=min(
        max(norm,0.0),
        1.0
    )

    code=int(
        np.rint(
            norm
            *
            float(BINARY_MAX)
        )
    )

    return (
        code
        &
        BINARY_MAX
    )


def binary32_decode(word):

    return (
        ZMIN
        +
        (
            float(
                int(word)&BINARY_MAX
            )
            /
            float(BINARY_MAX)
        )
        *
        float(ZMAX-ZMIN)
    )


def binary32_codec(scale,bit):

    z=math.log2(scale)

    clean_word=binary32_encode(z)

    fault_word=(
        clean_word
        ^
        (1<<bit)
    )

    clean_z=binary32_decode(
        clean_word
    )

    fault_z=binary32_decode(
        fault_word
    )

    return (
        clean_word,
        fault_word,
        2.0**clean_z,
        2.0**fault_z
    )


# ======================================================================================
# HA-FBMS
# ======================================================================================

def ha_encode(z):

    u=(
        (z-ZMIN)
        *
        (BC+1)
        /
        float(ZMAX-ZMIN)
    )

    u=min(
        max(u,0.0),
        float(BC+1)
    )

    coarse=int(
        math.floor(u)
    )


    if coarse>=BC+1:

        coarse=BC
        fine=FMAX

    else:

        fine=int(
            np.rint(
                (u-coarse)
                *
                FMAX
            )
        )

        coarse=min(
            max(coarse,0),
            BC
        )

        fine=min(
            max(fine,0),
            FMAX
        )


    therm=(
        0
        if coarse==0
        else
        (1<<coarse)-1
    )


    return (
        therm
        |
        (fine<<BC)
    )


def ha_decode(word):

    coarse=(
        int(word)
        &
        COARSE_MASK
    )

    count=int(coarse).bit_count()

    fine=(
        int(word)
        >>
        BC
    ) & FMAX

    u=(
        float(count)
        +
        float(fine)
        /
        float(FMAX)
    )

    z=(
        ZMIN
        +
        (
            float(ZMAX-ZMIN)
            /
            float(BC+1)
        )
        *
        u
    )

    return z


def ha_codec(scale,bit):

    z=math.log2(scale)

    clean_word=ha_encode(z)

    fault_word=(
        clean_word
        ^
        (1<<bit)
    )

    clean_z=ha_decode(
        clean_word
    )

    fault_z=ha_decode(
        fault_word
    )

    return (
        clean_word,
        fault_word,
        2.0**clean_z,
        2.0**fault_z
    )


# ======================================================================================
# METRICS
# ======================================================================================

def metrics(ref_logits,test_logits):

    ref=ref_logits.float()
    test=test_logits.float()

    finite=bool(
        torch.isfinite(test)
        .all()
        .item()
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
                ref_logp
                -
                test_logp
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
                    ref.argmax(dim=-1),
                    test.argmax(dim=-1)
                )
            ),

        "kl":kl,

        "max_abs":
            float(
                delta.max().item()
            ),

        "mean_abs":
            float(
                delta.mean().item()
            ),
    }


def amplification(
    clean_scale,
    fault_scale
):

    if not (
        math.isfinite(clean_scale)
        and
        clean_scale>0
        and
        math.isfinite(fault_scale)
        and
        fault_scale>0
    ):
        return None


    r=(
        fault_scale
        /
        clean_scale
    )

    return max(
        r,
        1.0/r
    )


# ======================================================================================
# MODEL
# ======================================================================================

torch.manual_seed(20260904)
torch.cuda.manual_seed_all(20260904)


tokenizer=AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True,
)


model=AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"":0},
    low_cpu_mem_usage=True,
)

model.eval()


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


# ======================================================================================
# PROMPT FORWARD
# ======================================================================================

with torch.inference_mode():

    initial=model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True
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
    dtype=torch.long,
    device=model.device
)

cache_position=torch.tensor(
    [past_len],
    dtype=torch.long,
    device=model.device
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

    ref_out=model(
        input_ids=next_token,
        past_key_values=ref_cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True
    )


bf16_logits=ref_out.logits[
    :,
    -1,
    :
].detach()


del ref_cache
del ref_out

torch.cuda.empty_cache()


# ======================================================================================
# FULL RAW INT4 CLEAN BACKGROUND
# ======================================================================================

raw_legacy,raw_stats=build_raw_full_cache(
    legacy
)

if raw_stats["scale_clip"]!=0:

    raise RuntimeError(
        f"unexpected full-cache scale clips: {raw_stats['scale_clip']}"
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
        return_dict=True
    )


raw_logits=raw_out.logits[
    :,
    -1,
    :
].detach()


raw_vs_bf16=metrics(
    bf16_logits,
    raw_logits
)


del raw_cache
del raw_out

torch.cuda.empty_cache()


# ======================================================================================
# FAULT REPLAY
# ======================================================================================

ENCODINGS=[
    (
        "FP32",
        "fp32_bit",
    ),
    (
        "BINARY32_Z",
        "binary32_bit",
    ),
    (
        "HA_FBMS",
        "ha_fbms_bit",
    ),
]


rows=[]

max_z_error=0.0
readback_fail=0
adaptive_clean_forwards=0
fault_forwards=0


for target in targets:

    target_id=int(
        target["target_id"]
    )

    layer=int(
        target["layer"]
    )

    kv=str(
        target["kv"]
    ).upper()

    local_group=int(
        target["local_group"]
    )

    band=str(
        target["layer_band"]
    )

    manifest_z=float(
        target["z"]
    )


    kv_idx=(
        0
        if kv=="K"
        else
        1
        if kv=="V"
        else
        -1
    )


    if kv_idx<0:

        raise RuntimeError(
            f"invalid KV={kv}"
        )


    original_tensor=legacy[
        layer
    ][
        kv_idx
    ]


    original_flat=(
        original_tensor
        .detach()
        .float()
        .contiguous()
        .view(-1)
    )


    start=(
        local_group
        *
        GROUP_SIZE
    )

    end=min(
        start+GROUP_SIZE,
        int(original_flat.numel())
    )


    if not (
        0<=start<end
    ):

        raise RuntimeError(
            f"group out of range target={target_id}"
        )


    q,raw_scale=quantize_group(
        original_flat[start:end]
    )


    reconstructed_z=math.log2(
        raw_scale
    )


    zerr=abs(
        reconstructed_z
        -
        manifest_z
    )


    max_z_error=max(
        max_z_error,
        zerr
    )


    if zerr>1e-6:

        raise RuntimeError(
            f"target z mismatch "
            f"id={target_id} "
            f"manifest={manifest_z} "
            f"reconstructed={reconstructed_z}"
        )


    raw_target=(
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


    for encoding,bit_field in ENCODINGS:

        bit=int(
            target[
                bit_field
            ]
        )


        if encoding=="FP32":

            (
                clean_word,
                fault_word,
                clean_scale,
                fault_scale
            )=fp32_codec(
                raw_scale,
                bit
            )


        elif encoding=="BINARY32_Z":

            (
                clean_word,
                fault_word,
                clean_scale,
                fault_scale
            )=binary32_codec(
                raw_scale,
                bit
            )


        else:

            (
                clean_word,
                fault_word,
                clean_scale,
                fault_scale
            )=ha_codec(
                raw_scale,
                bit
            )


        amp=amplification(
            clean_scale,
            fault_scale
        )


        fault_scale_valid=bool(
            math.isfinite(
                fault_scale
            )
            and
            fault_scale>0
        )


        # ------------------------------------------------------------------
        # Encoding-clean reference.
        # Usually equal to raw INT4 after BF16 cast.
        # ------------------------------------------------------------------

        clean_values=(
            q
            *
            float(
                clean_scale
            )
        ).to(
            device=original_tensor.device,
            dtype=original_tensor.dtype
        )


        raw_target_cast=raw_target.to(
            device=original_tensor.device,
            dtype=original_tensor.dtype
        )


        clean_equal_raw=bool(
            torch.equal(
                clean_values,
                raw_target_cast
            )
        )


        clean_logits=raw_logits
        clean_source="FULL_RAW_INT4"


        if not clean_equal_raw:

            adaptive_clean_forwards+=1

            clean_pairs=list(
                raw_legacy
            )

            pair=list(
                clean_pairs[
                    layer
                ]
            )

            patched=(
                pair[
                    kv_idx
                ]
                .detach()
                .clone()
                .contiguous()
            )

            patched.view(-1)[
                start:end
            ]=clean_values

            pair[
                kv_idx
            ]=patched

            clean_pairs[
                layer
            ]=tuple(pair)


            with torch.inference_mode():

                clean_cache=legacy_to_dynamic(
                    tuple(
                        clean_pairs
                    )
                )

                clean_out=model(
                    input_ids=next_token,
                    past_key_values=clean_cache,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=False,
                    return_dict=True
                )


            clean_logits=clean_out.logits[
                :,
                -1,
                :
            ].detach()


            clean_source="ENCODING_CLEAN_TARGET"


            del clean_cache
            del clean_out

            torch.cuda.empty_cache()


        # ------------------------------------------------------------------
        # Fault cache
        # ------------------------------------------------------------------

        fault_pairs=list(
            raw_legacy
        )

        pair=list(
            fault_pairs[
                layer
            ]
        )


        fault_tensor=(
            pair[
                kv_idx
            ]
            .detach()
            .clone()
            .contiguous()
        )


        fault_values=(
            q
            *
            float(
                fault_scale
            )
        ).to(
            device=original_tensor.device,
            dtype=original_tensor.dtype
        )


        fault_tensor.view(-1)[
            start:end
        ]=fault_values


        local_write_changed=bool(
            not torch.equal(
                fault_tensor.view(-1)[
                    start:end
                ],
                raw_target_cast
            )
        )


        pair[
            kv_idx
        ]=fault_tensor

        fault_pairs[
            layer
        ]=tuple(pair)


        fault_cache=legacy_to_dynamic(
            tuple(
                fault_pairs
            )
        )


        readback=cache_to_legacy(
            fault_cache
        )[
            layer
        ][
            kv_idx
        ].contiguous().view(-1)[
            start:end
        ]


        readback_pass=bool(
            torch.equal(
                readback,
                fault_values
            )
        )


        if not readback_pass:

            readback_fail+=1

            raise RuntimeError(
                f"cache readback failure "
                f"target={target_id} "
                f"encoding={encoding}"
            )


        with torch.inference_mode():

            fault_out=model(
                input_ids=next_token,
                past_key_values=fault_cache,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                return_dict=True
            )


        fault_forwards+=1


        fault_logits=fault_out.logits[
            :,
            -1,
            :
        ].detach()


        vs_clean=metrics(
            clean_logits,
            fault_logits
        )


        vs_raw=metrics(
            raw_logits,
            fault_logits
        )


        vs_bf16=metrics(
            bf16_logits,
            fault_logits
        )


        rows.append({
            "model":
                "QWEN3_32B",

            "prompt_id":
                PROMPT_ID,

            "target_id":
                target_id,

            "layer_band":
                band,

            "layer":
                layer,

            "kv":
                kv,

            "local_group":
                local_group,

            "manifest_z":
                manifest_z,

            "reconstructed_z":
                reconstructed_z,

            "z_abs_error":
                zerr,

            "encoding":
                encoding,

            "bit":
                bit,

            "clean_word":
                int(clean_word),

            "fault_word":
                int(fault_word),

            "raw_scale":
                raw_scale,

            "clean_scale":
                clean_scale,

            "fault_scale":
                fault_scale,

            "fault_scale_valid":
                fault_scale_valid,

            "scale_amplification":
                amp,

            "clean_group_equal_raw":
                clean_equal_raw,

            "clean_reference_source":
                clean_source,

            "local_write_changed":
                local_write_changed,

            "cache_readback_pass":
                readback_pass,

            "fault_logits_finite":
                vs_clean["finite"],

            "fault_vs_clean_top1_same":
                vs_clean["top1_same"],

            "fault_vs_clean_kl":
                vs_clean["kl"],

            "fault_vs_clean_maxlogit":
                vs_clean["max_abs"],

            "fault_vs_clean_meanlogit":
                vs_clean["mean_abs"],

            "fault_vs_raw_top1_same":
                vs_raw["top1_same"],

            "fault_vs_raw_kl":
                vs_raw["kl"],

            "fault_vs_raw_maxlogit":
                vs_raw["max_abs"],

            "fault_vs_bf16_top1_same":
                vs_bf16["top1_same"],

            "fault_vs_bf16_kl":
                vs_bf16["kl"],

            "fault_vs_bf16_maxlogit":
                vs_bf16["max_abs"],
        })


        del fault_cache
        del fault_out

        torch.cuda.empty_cache()


# ======================================================================================
# WORKER GATES
# ======================================================================================

expected_faults=9

count_pass=(
    len(rows)
    ==
    expected_faults
    ==
    fault_forwards
)


ha_rows=[
    r
    for r in rows
    if r["encoding"]=="HA_FBMS"
]


ha_geometry_pass=all(
    r["fault_scale_valid"]
    and
    r["scale_amplification"] is not None
    and
    r["scale_amplification"]<2.0
    for r in ha_rows
)


all_readback=all(
    r["cache_readback_pass"]
    for r in rows
)


worker_pass=all([
    count_pass,
    all_readback,
    readback_fail==0,
    max_z_error<=1e-6,
    len(ha_rows)==3,
    ha_geometry_pass,
])


with (
    OUTROOT
    /
    "P7D1_RESULTS.csv"
).open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w=csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        )
    )

    w.writeheader()
    w.writerows(rows)


payload={
    "model":
        "Qwen3-32B",

    "prompt_id":
        PROMPT_ID,

    "target_count":
        len(targets),

    "fault_forward_count":
        fault_forwards,

    "expected_fault_forward_count":
        expected_faults,

    "adaptive_clean_forwards":
        adaptive_clean_forwards,

    "full_cache_raw_stats":
        raw_stats,

    "raw_vs_bf16":
        raw_vs_bf16,

    "target_z_max_abs_error":
        max_z_error,

    "all_cache_readback_pass":
        all_readback,

    "ha_fault_geometry_pass":
        ha_geometry_pass,

    "worker_pass":
        worker_pass,
}


(
    OUTROOT
    /
    "P7D1_SUMMARY.json"
).write_text(
    json.dumps(
        payload,
        indent=2
    ),
    encoding="utf-8"
)


print(
    f"P7D1_WORKER_RESULT "
    f"PROMPT={PROMPT_ID} "
    f"TARGETS={len(targets)} "
    f"FAULTS={fault_forwards} "
    f"ADAPTIVE_CLEAN={adaptive_clean_forwards} "
    f"GROUPS={raw_stats['groups']} "
    f"CLIP={raw_stats['scale_clip']} "
    f"ZERR={max_z_error} "
    f"READBACK={all_readback} "
    f"HA_GEOMETRY={ha_geometry_pass} "
    f"PASS={worker_pass}"
)


print(
    "P7D1_WORKER_PASS=",
    worker_pass,
    sep=""
)


del model
gc.collect()
torch.cuda.empty_cache()
