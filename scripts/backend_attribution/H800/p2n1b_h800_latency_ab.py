import torch
import triton
import triton.language as tl
from pathlib import Path


ROOT=Path("${WORKSPACE}/metakv_h800_attribution")

OUT=ROOT/"p2n1_static_sass"

OUT.mkdir(parents=True,exist_ok=True)


BLOCK=1024


@triton.jit
def identity_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE:tl.constexpr
):
    pid=tl.program_id(0)
    offs=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)

    mask=offs<n

    x=tl.load(x_ptr+offs,mask=mask)
    tl.store(y_ptr+offs,x,mask=mask)



@triton.jit
def runtime_perm_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE:tl.constexpr
):

    pid=tl.program_id(0)

    offs=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)

    mask=offs<n

    x=tl.load(x_ptr+offs,mask=mask)

    # emulate runtime metadata permutation
    b0=x & 0x55555555
    b1=x & (-1431655766)

    b0=b0<<1
    b1=b1>>1

    y=b0|b1

    tl.store(y_ptr+offs,y,mask=mask)



@triton.jit
def preperm_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE:tl.constexpr
):

    pid=tl.program_id(0)

    offs=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)

    mask=offs<n

    x=tl.load(x_ptr+offs,mask=mask)

    # already normalized layout
    y=x

    tl.store(y_ptr+offs,y,mask=mask)



def launch(fn):

    n=1024*1024

    x=torch.randint(
        0,
        2**30,
        (n,),
        device="cuda",
        dtype=torch.int32
    )

    y=torch.empty_like(x)


    fn[((n+BLOCK-1)//BLOCK,)](
        x,
        y,
        n,
        BLOCK_SIZE=BLOCK
    )

    torch.cuda.synchronize()



for name,fn in [
    ("identity",identity_kernel),
    ("runtime_perm",runtime_perm_kernel),
    ("preperm",preperm_kernel)
]:

    print("COMPILE",name)

    launch(fn)

    print("PASS",name)


print("P2N1A_MICROBENCH_PASS=True")



import torch
import numpy as np


def benchmark_latency(fn, name):

    # warmup
    for _ in range(100):
        launch(fn)

    torch.cuda.synchronize()

    times=[]

    for _ in range(1000):

        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)

        start.record()

        launch(fn)

        end.record()

        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    med=float(np.median(times))

    print(
        name,
        "MEDIAN_MS=",
        med
    )

    return med


print("================================================")
print("H800 LATENCY A/B")
print("================================================")


identity_ms = benchmark_latency(
    identity_kernel,
    "IDENTITY"
)

runtime_ms = benchmark_latency(
    runtime_perm_kernel,
    "RUNTIME_PERM"
)

preperm_ms = benchmark_latency(
    preperm_kernel,
    "PREPERM"
)


print()
print(
"RUNTIME_VS_IDENTITY_PERCENT=",
(runtime_ms/identity_ms-1)*100
)

print(
"PREPERM_VS_IDENTITY_PERCENT=",
(preperm_ms/identity_ms-1)*100
)

print(
"RUNTIME_VS_PREPERM_PERCENT=",
(runtime_ms/preperm_ms-1)*100
)

print()
print("P2N1B_H800_LATENCY_AB_PASS=True")
