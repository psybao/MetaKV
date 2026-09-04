from pathlib import Path
import ast
import csv
import gc
import json
import math
import os
import struct
import sys
import warnings

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache


warnings.filterwarnings("ignore")


# ======================================================================================
# INPUT
# ======================================================================================

MODEL_TAG=os.environ["METAKV_MODEL_TAG"]

MODEL_PATH=Path(
    os.environ["METAKV_MODEL_PATH"]
)

SOURCE_SCRIPT=Path(
    os.environ["METAKV_SOURCE_SCRIPT"]
)

MANIFEST=Path(
    os.environ["METAKV_MANIFEST"]
)

PROMPT_ID=int(
    os.environ["METAKV_PROMPT_ID"]
)

OUTROOT=Path(
    os.environ["METAKV_OUTROOT"]
)

ZMIN=float(
    os.environ["METAKV_ZMIN"]
)

ZMAX=float(
    os.environ["METAKV_ZMAX"]
)



CORRECTED_BINARY_ONLY=True
BINARY32_Z_IS_IEEE_FLOAT=False
BINARY32_Z_SEMANTICS="UNIFORM_UNSIGNED_32BIT_LOG2_SCALE"

GROUP_SIZE=32

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
# PROMPTS — recover exact frozen list
# ======================================================================================

source_text=SOURCE_SCRIPT.read_text(
    encoding="utf-8",
    errors="replace"
)

tree=ast.parse(
    source_text
)


def recover_prompts():

    candidates=[]

    for node in ast.walk(
        tree
    ):

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
                x.id
                for x in node.targets
                if isinstance(
                    x,
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
                candidates.append(
                    (
                        name,
                        list(obj)
                    )
                )

    if not candidates:

        raise RuntimeError(
            "frozen prompt list not recovered"
        )

    candidates.sort(
        key=lambda x:(
            abs(
                len(x[1])-8
            ),
            x[0]
        )
    )

    return candidates[0]


PROMPT_VARIABLE,PROMPTS=recover_prompts()

if len(PROMPTS)<8:
    raise RuntimeError(
        "frozen prompt list < 8"
    )

PROMPT=PROMPTS[
    PROMPT_ID
]


# ======================================================================================
# MANIFEST
# ======================================================================================

with MANIFEST.open(
    newline="",
    encoding="utf-8-sig"
) as f:

    manifest_all=list(
        csv.DictReader(f)
    )


targets=[
    r
    for r in manifest_all
    if int(
        r["prompt_id"]
    )==PROMPT_ID
]


if len(targets)!=12:

    raise RuntimeError(
        f"expected 12 frozen targets, got {len(targets)}"
    )


required_fields=[
    "prompt_id",
    "layer",
    "kv",
    "local_group",
    "severity_class",
    "z",
    "fp32_bit",
    "binary32_bit",
    "ha_fbms_bit",
]


for field in required_fields:

    if field not in targets[0]:

        raise RuntimeError(
            f"missing manifest field: {field}"
        )


# ======================================================================================
# CACHE HELPERS
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

        result=[]

        for layer in cache.layers:

            result.append(
                (
                    layer.keys,
                    layer.values,
                )
            )

        return tuple(
            result
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
# INT4
# ======================================================================================

def quant_group(
    x
):

    xf=(
        x
        .detach()
        .float()
        .contiguous()
        .view(-1)
    )

    maxabs=float(
        xf.abs().max().item()
    )

    if (
        not math.isfinite(
            maxabs
        )
        or
        maxabs==0.0
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

    return (
        q,
        scale,
    )


@torch.no_grad()
def full_raw_int4_cache(
    legacy
):

    out=[]

    total_groups=0
    total_values=0

    zmin_seen=math.inf
    zmax_seen=-math.inf


    for k,v in legacy:

        pair=[]

        for tensor in (
            k,
            v
        ):

            flat=(
                tensor
                .detach()
                .float()
                .contiguous()
                .view(-1)
            )

            dq=torch.empty_like(
                flat
            )

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

                q,scale=quant_group(
                    flat[
                        start:end
                    ]
                )

                z=math.log2(
                    scale
                )

                zmin_seen=min(
                    zmin_seen,
                    z
                )

                zmax_seen=max(
                    zmax_seen,
                    z
                )

                dq[
                    start:end
                ]=q*scale

                total_groups += 1
                total_values += (
                    end-start
                )


            dq=(
                dq
                .view(
                    tensor.shape
                )
                .to(
                    device=tensor.device,
                    dtype=tensor.dtype
                )
                .contiguous()
            )

            pair.append(
                dq
            )

        out.append(
            tuple(
                pair
            )
        )


    return tuple(out),{
        "group_count":
            total_groups,

        "value_count":
            total_values,

        "z_min":
            zmin_seen,

        "z_max":
            zmax_seen,
    }


# ======================================================================================
# CODECS
# ======================================================================================

def float32_to_word(
    value
):

    return struct.unpack(
        "<I",
        struct.pack(
            "<f",
            np.float32(
                value
            )
        )
    )[0]


def word_to_float32(
    word
):

    return float(
        struct.unpack(
            "<f",
            struct.pack(
                "<I",
                int(
                    word
                ) & 0xFFFFFFFF
            )
        )[0]
    )


def fp32_scale_codec(
    scale,
    bit
):

    clean_word=float32_to_word(
        scale
    )

    fault_word=(
        clean_word
        ^
        (
            1<<bit
        )
    )

    clean_scale=word_to_float32(
        clean_word
    )

    fault_scale=word_to_float32(
        fault_word
    )

    return (
        clean_word,
        fault_word,
        clean_scale,
        fault_scale,
    )



def binary32_z_codec(
    scale,
    bit
):

    BINARY_MAX=(1<<32)-1

    z=math.log2(
        scale
    )

    norm=(
        (z-ZMIN)
        /
        float(
            ZMAX-ZMIN
        )
    )

    norm=min(
        max(
            norm,
            0.0
        ),
        1.0
    )

    clean_word=int(
        np.rint(
            norm
            *
            float(
                BINARY_MAX
            )
        )
    ) & BINARY_MAX

    fault_word=(
        clean_word
        ^
        (
            1<<bit
        )
    ) & BINARY_MAX

    clean_z=(
        ZMIN
        +
        (
            float(
                clean_word
            )
            /
            float(
                BINARY_MAX
            )
        )
        *
        float(
            ZMAX-ZMIN
        )
    )

    fault_z=(
        ZMIN
        +
        (
            float(
                fault_word
            )
            /
            float(
                BINARY_MAX
            )
        )
        *
        float(
            ZMAX-ZMIN
        )
    )

    clean_scale=float(
        np.exp2(
            np.float64(
                clean_z
            )
        )
    )

    fault_scale=float(
        np.exp2(
            np.float64(
                fault_z
            )
        )
    )

    return (
        clean_word,
        fault_word,
        clean_scale,
        fault_scale,
    )


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

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore"
    ):

        scale=float(
            np.exp2(
                np.float64(
                    z
                )
            )
        )

    return (
        z,
        scale,
    )


def ha_codec(
    scale,
    bit
):

    z=math.log2(
        scale
    )

    clean_word=ha_encode_z(
        z
    )

    fault_word=(
        clean_word
        ^
        (
            1<<bit
        )
    )

    clean_z,clean_scale=ha_decode_word(
        clean_word
    )

    fault_z,fault_scale=ha_decode_word(
        fault_word
    )

    return (
        clean_word,
        fault_word,
        clean_scale,
        fault_scale,
    )


# ======================================================================================
# METRICS
# ======================================================================================

def metric(
    reference_logits,
    test_logits
):

    ref=reference_logits.float()
    test=test_logits.float()


    test_finite=bool(
        torch.isfinite(
            test
        ).all().item()
    )


    if not test_finite:

        return {
            "finite":
                False,

            "top1_same":
                False,

            "kl":
                None,

            "max_abs":
                None,

            "mean_abs":
                None,
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
        "finite":
            True,

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


def amplification(
    clean_scale,
    fault_scale
):

    if not (
        math.isfinite(
            clean_scale
        )
        and
        clean_scale>0
        and
        math.isfinite(
            fault_scale
        )
        and
        fault_scale>0
    ):

        return None


    ratio=(
        fault_scale
        /
        clean_scale
    )


    return max(
        ratio,
        1.0/ratio
    )


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
    device_map={
        "":0
    },
)

model.eval()


inputs=tokenizer(
    PROMPT,
    return_tensors="pt",
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
        return_dict=True,
    )


original_legacy=cache_to_legacy(
    initial.past_key_values
)

layer_count=len(
    original_legacy
)


next_token=initial.logits[
    :,
    -1:,
    :
].argmax(
    dim=-1
)


past_len=int(
    original_legacy[0][0].shape[-2]
)


position_ids=torch.tensor(
    [[past_len]],
    dtype=torch.long,
    device=model.device,
)


cache_position=torch.tensor(
    [past_len],
    dtype=torch.long,
    device=model.device,
)


# ======================================================================================
# BF16 REFERENCE
# ======================================================================================

with torch.inference_mode():

    bf16_cache=legacy_to_dynamic(
        clone_legacy(
            original_legacy
        )
    )

    bf16_out=model(
        input_ids=next_token,
        past_key_values=bf16_cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    )


bf16_logits=bf16_out.logits[
    :,
    -1,
    :
].detach()


# ======================================================================================
# FULL-CACHE RAW INT4 CLEAN BACKGROUND
# ======================================================================================

raw_legacy,raw_stats=full_raw_int4_cache(
    original_legacy
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


clean_vs_bf16=metric(
    bf16_logits,
    raw_logits
)


# ======================================================================================
# TARGET REPLAY
# ======================================================================================

rows=[]

location_z_max_error=0.0
readback_fail_count=0
adaptive_clean_forwards=0
fault_forward_count=0


ENCODINGS=[
    (
        "BINARY32_Z",
        "binary32_bit",
    ),
]


for location_id,target in enumerate(
    targets
):

    layer=int(
        target[
            "layer"
        ]
    )

    kv=str(
        target[
            "kv"
        ]
    ).upper()

    local_group=int(
        target[
            "local_group"
        ]
    )

    severity=str(
        target[
            "severity_class"
        ]
    )

    manifest_z=float(
        target[
            "z"
        ]
    )


    if not (
        0 <= layer < layer_count
    ):

        raise RuntimeError(
            f"layer out of range: {layer}"
        )


    kv_index=(
        0
        if kv=="K"
        else
        1
        if kv=="V"
        else
        -1
    )


    if kv_index<0:

        raise RuntimeError(
            f"bad kv field: {kv}"
        )


    original_tensor=original_legacy[
        layer
    ][
        kv_index
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
        int(
            original_flat.numel()
        )
    )


    if not (
        0 <= start < end
    ):

        raise RuntimeError(
            f"group out of bounds "
            f"layer={layer} kv={kv} "
            f"group={local_group} "
            f"numel={original_flat.numel()}"
        )


    original_group=original_flat[
        start:end
    ]


    q,raw_scale=quant_group(
        original_group
    )

    raw_z=math.log2(
        raw_scale
    )


    z_error=abs(
        raw_z
        -
        manifest_z
    )

    location_z_max_error=max(
        location_z_max_error,
        z_error
    )


    if z_error > 1e-6:

        raise RuntimeError(
            f"frozen target scale mismatch "
            f"location={location_id} "
            f"manifest_z={manifest_z} "
            f"reconstructed_z={raw_z} "
            f"error={z_error}"
        )


    raw_tensor=raw_legacy[
        layer
    ][
        kv_index
    ]


    raw_group=(
        raw_tensor
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


        if not (
            0 <= bit < 32
        ):

            raise RuntimeError(
                f"bad bit {bit}"
            )


        if encoding=="FP32":

            (
                clean_word,
                fault_word,
                encoded_clean_scale,
                fault_scale,
            )=fp32_scale_codec(
                raw_scale,
                bit
            )

        elif encoding=="BINARY32_Z":

            (
                clean_word,
                fault_word,
                encoded_clean_scale,
                fault_scale,
            )=binary32_z_codec(
                raw_scale,
                bit
            )

        else:

            (
                clean_word,
                fault_word,
                encoded_clean_scale,
                fault_scale,
            )=ha_codec(
                raw_scale,
                bit
            )


        amp=amplification(
            encoded_clean_scale,
            fault_scale
        )


        scale_valid=bool(
            math.isfinite(
                fault_scale
            )
            and
            fault_scale>0
        )


        # ------------------------------------------------------------------
        # Clean encoded target group.
        # If BF16 replay values are exactly identical to raw-scale baseline,
        # RAW full-cache logits are a valid clean baseline.
        #
        # Otherwise run an adaptive clean-encoding forward so fault-vs-clean
        # never conflates clean codec quantization error.
        # ------------------------------------------------------------------

        clean_values=(
            q
            *
            float(
                encoded_clean_scale
            )
        ).to(
            dtype=original_tensor.dtype,
            device=original_tensor.device
        )


        raw_group_cast=raw_group.to(
            dtype=original_tensor.dtype,
            device=original_tensor.device
        )


        clean_group_equal_raw=bool(
            torch.equal(
                clean_values,
                raw_group_cast
            )
        )


        clean_reference_logits=raw_logits

        clean_reference_source="FULL_RAW_INT4"


        if not clean_group_equal_raw:

            adaptive_clean_forwards += 1

            clean_pairs=list(
                raw_legacy
            )

            pair=list(
                clean_pairs[
                    layer
                ]
            )

            patched_clean=pair[
                kv_index
            ].detach().clone().contiguous()

            patched_flat=patched_clean.view(
                -1
            )

            patched_flat[
                start:end
            ]=clean_values

            pair[
                kv_index
            ]=patched_clean

            clean_pairs[
                layer
            ]=tuple(
                pair
            )


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
                    return_dict=True,
                )


            clean_reference_logits=clean_out.logits[
                :,
                -1,
                :
            ].detach()


            clean_reference_source=(
                "ENCODING_CLEAN_TARGET"
            )


        # ------------------------------------------------------------------
        # FAULT TARGET GROUP
        # ------------------------------------------------------------------

        fault_pairs=list(
            raw_legacy
        )

        pair=list(
            fault_pairs[
                layer
            ]
        )

        fault_tensor=pair[
            kv_index
        ].detach().clone().contiguous()

        fault_flat=fault_tensor.view(
            -1
        )


        fault_values=(
            q
            *
            float(
                fault_scale
            )
        ).to(
            dtype=original_tensor.dtype,
            device=original_tensor.device
        )


        fault_flat[
            start:end
        ]=fault_values


        # local write gate
        local_write_changed=bool(
            not torch.equal(
                fault_flat[
                    start:end
                ],
                raw_group_cast
            )
        )


        pair[
            kv_index
        ]=fault_tensor

        fault_pairs[
            layer
        ]=tuple(
            pair
        )


        fault_cache=legacy_to_dynamic(
            tuple(
                fault_pairs
            )
        )


        # Readback gate
        rb=cache_to_legacy(
            fault_cache
        )[
            layer
        ][
            kv_index
        ].contiguous().view(-1)[
            start:end
        ]


        readback_pass=bool(
            torch.equal(
                rb,
                fault_values
            )
        )


        if not readback_pass:

            readback_fail_count += 1

            raise RuntimeError(
                f"fault cache readback failed "
                f"location={location_id} "
                f"encoding={encoding}"
            )


        with torch.inference_mode():

            fault_out=model(
                input_ids=next_token,
                past_key_values=fault_cache,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                return_dict=True,
            )


        fault_forward_count += 1


        fault_logits=fault_out.logits[
            :,
            -1,
            :
        ].detach()


        fault_vs_clean=metric(
            clean_reference_logits,
            fault_logits
        )


        fault_vs_raw=metric(
            raw_logits,
            fault_logits
        )


        fault_vs_bf16=metric(
            bf16_logits,
            fault_logits
        )


        row={
            "model":
                MODEL_TAG,

            "prompt_id":
                PROMPT_ID,

            "location_id":
                location_id,

            "layer":
                layer,

            "kv":
                kv,

            "local_group":
                local_group,

            "severity":
                severity,

            "magnitude_class":
                target.get(
                    "magnitude_class"
                ),

            "layer_band":
                target.get(
                    "layer_band"
                ),

            "manifest_z":
                manifest_z,

            "reconstructed_z":
                raw_z,

            "z_reconstruction_abs_error":
                z_error,

            "encoding":
                encoding,

            "bit":
                bit,

            "clean_word":
                int(
                    clean_word
                ),

            "fault_word":
                int(
                    fault_word
                ),

            "raw_scale":
                raw_scale,

            "encoded_clean_scale":
                encoded_clean_scale,

            "fault_scale":
                fault_scale,

            "fault_scale_valid":
                scale_valid,

            "scale_amplification":
                amp,

            "clean_group_equal_raw":
                clean_group_equal_raw,

            "clean_reference_source":
                clean_reference_source,

            "local_write_changed":
                local_write_changed,

            "cache_readback_pass":
                readback_pass,

            "fault_logits_finite":
                fault_vs_clean[
                    "finite"
                ],

            "fault_vs_clean_top1_same":
                fault_vs_clean[
                    "top1_same"
                ],

            "fault_vs_clean_kl":
                fault_vs_clean[
                    "kl"
                ],

            "fault_vs_clean_maxlogit":
                fault_vs_clean[
                    "max_abs"
                ],

            "fault_vs_clean_meanlogit":
                fault_vs_clean[
                    "mean_abs"
                ],

            "fault_vs_raw_top1_same":
                fault_vs_raw[
                    "top1_same"
                ],

            "fault_vs_raw_kl":
                fault_vs_raw[
                    "kl"
                ],

            "fault_vs_raw_maxlogit":
                fault_vs_raw[
                    "max_abs"
                ],

            "fault_vs_bf16_top1_same":
                fault_vs_bf16[
                    "top1_same"
                ],

            "fault_vs_bf16_kl":
                fault_vs_bf16[
                    "kl"
                ],

            "fault_vs_bf16_maxlogit":
                fault_vs_bf16[
                    "max_abs"
                ],
        }


        rows.append(
            row
        )


        print(
            f"P6B1_FAULT "
            f"MODEL={MODEL_TAG} "
            f"PROMPT={PROMPT_ID} "
            f"LOC={location_id} "
            f"SEV={severity} "
            f"LAYER={layer} "
            f"KV={kv} "
            f"GROUP={local_group} "
            f"ENC={encoding} "
            f"BIT={bit} "
            f"VALID_SCALE={scale_valid} "
            f"A={amp} "
            f"CLEAN_EQ_RAW={clean_group_equal_raw} "
            f"READBACK={readback_pass} "
            f"FINITE={fault_vs_clean['finite']} "
            f"TOP1_SAME={fault_vs_clean['top1_same']} "
            f"KL={fault_vs_clean['kl']} "
            f"MAXLOGIT={fault_vs_clean['max_abs']}"
        )


# ======================================================================================
# SAVE WORKER
# ======================================================================================

result_count=len(
    rows
)

expected_count=12


result_count_pass=(
    result_count
    ==
    expected_count
)


all_readback=all(
    r[
        "cache_readback_pass"
    ]
    for r in rows
)


ha_rows=[
    r
    for r in rows
    if r[
        "encoding"
    ]=="HA_FBMS"
]


ha_all_scale_valid=all(
    r[
        "fault_scale_valid"
    ]
    for r in ha_rows
)


ha_max_a=max(
    (
        r[
            "scale_amplification"
        ]
        for r in ha_rows
        if r[
            "scale_amplification"
        ]
        is not None
    ),
    default=None
)


ha_ge2=sum(
    1
    for r in ha_rows
    if (
        r[
            "scale_amplification"
        ]
        is not None
        and
        r[
            "scale_amplification"
        ]>=2.0
    )
)


worker_pass=all([
    result_count_pass,
    all_readback,
    readback_fail_count==0,
    location_z_max_error<=1e-6,
    ha_all_scale_valid,
    ha_ge2==0,
])


with (
    OUTROOT
    /
    "P6B1_RESULTS.csv"
).open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fields=list(
        rows[0].keys()
    )

    w=csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()
    w.writerows(
        rows
    )


payload={
    "model":
        MODEL_TAG,

    "prompt_id":
        PROMPT_ID,

    "prompt_variable":
        PROMPT_VARIABLE,

    "target_count":
        len(
            targets
        ),

    "fault_forward_count":
        fault_forward_count,

    "expected_fault_forward_count":
        expected_count,

    "result_count_pass":
        result_count_pass,

    "adaptive_clean_forwards":
        adaptive_clean_forwards,

    "layer_count":
        layer_count,

    "full_cache_raw_stats":
        raw_stats,

    "clean_full_cache_vs_bf16":
        clean_vs_bf16,

    "location_z_max_error":
        location_z_max_error,

    "all_readback":
        all_readback,

    "ha_all_scale_valid":
        ha_all_scale_valid,

    "ha_max_amplification":
        ha_max_a,

    "ha_ge2":
        ha_ge2,

    "worker_pass":
        worker_pass,
}


(
    OUTROOT
    /
    "P6B1_SUMMARY.json"
).write_text(
    json.dumps(
        payload,
        indent=2
    ),
    encoding="utf-8"
)


status=f"""====================================================================================================
METAKV P6B1 FULL-CACHE STRATIFIED FAULT WORKER STATUS
====================================================================================================

MODEL={MODEL_TAG}
PROMPT_ID={PROMPT_ID}

FROZEN_TARGET_COUNT={len(targets)}
ENCODINGS=FP32,BINARY32_Z,HA_FBMS

EXPECTED_FAULT_FORWARDS={expected_count}
ACTUAL_FAULT_FORWARDS={fault_forward_count}

RESULT_COUNT_PASS={result_count_pass}

FULL_CACHE_INT4_BACKGROUND=True
GROUP_SIZE={GROUP_SIZE}

CLEAN_FULL_CACHE_LOGITS_FINITE={clean_vs_bf16['finite']}
CLEAN_FULL_CACHE_TOP1_SAME_BF16={clean_vs_bf16['top1_same']}
CLEAN_FULL_CACHE_KL_BF16={clean_vs_bf16['kl']}

TARGET_Z_MAX_ABS_ERROR={location_z_max_error}

CACHE_READBACK_ALL_PASS={all_readback}
READBACK_FAIL_COUNT={readback_fail_count}

ADAPTIVE_CLEAN_ENCODING_FORWARDS={adaptive_clean_forwards}

HA_ALL_FAULT_SCALES_VALID={ha_all_scale_valid}
HA_MAX_SCALE_AMPLIFICATION={ha_max_a}
HA_SCALE_GE2_COUNT={ha_ge2}

LOCALIZED_TARGET_FAULT=True
FULL_CACHE_INT4_CONTEXT=True

FAULT_PROBABILITY_CLAIM=False
HARDWARE_BER_CLAIM=False

P6B1_WORKER_PASS={worker_pass}
====================================================================================================
"""


(
    OUTROOT
    /
    "P6B1_WORKER_STATUS.txt"
).write_text(
    status,
    encoding="utf-8"
)


print()
print(
    f"P6B1_WORKER_RESULT "
    f"MODEL={MODEL_TAG} "
    f"PROMPT={PROMPT_ID} "
    f"FAULTS={fault_forward_count} "
    f"ADAPTIVE_CLEAN={adaptive_clean_forwards} "
    f"ZERR={location_z_max_error} "
    f"HA_MAX_A={ha_max_a} "
    f"HA_GE2={ha_ge2} "
    f"PASS={worker_pass}"
)

print(
    "P6B1_WORKER_PASS=",
    worker_pass,
    sep=""
)


del model
gc.collect()

torch.cuda.empty_cache()
