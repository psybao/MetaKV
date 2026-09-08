import torch
import triton
import triton.language as tl


@triton.jit
def kernel_identity(x_ptr,n,BLOCK:tl.constexpr):

    pid=tl.program_id(0)
    offs=pid*BLOCK+tl.arange(0,BLOCK)
    mask=offs<n

    x=tl.load(x_ptr+offs,mask=mask)

    tl.store(x_ptr+offs,x,mask=mask)



@triton.jit
def kernel_perm(x_ptr,n,BLOCK:tl.constexpr):

    pid=tl.program_id(0)
    offs=pid*BLOCK+tl.arange(0,BLOCK)
    mask=offs<n

    x=tl.load(x_ptr+offs,mask=mask)

    mask_a=0x55555555
    mask_b=-1431655766

    a=x & mask_a
    b=x & mask_b

    y=(a<<1)|(b>>1)

    tl.store(x_ptr+offs,y,mask=mask)



def bench(fn,n):

    x=torch.randint(
        0,
        2**30,
        (n,),
        device="cuda",
        dtype=torch.int32
    )

    grid=lambda meta:(triton.cdiv(n,meta["BLOCK"]),)

    fn[grid](x,n,BLOCK=1024)

    torch.cuda.synchronize()


    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(100):
        fn[grid](x,n,BLOCK=1024)

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end)/100



print("="*70)
print("RTX 5080 SCALE SCAN")
print("="*70)

print("GPU=",torch.cuda.get_device_name(0))
print("TORCH=",torch.__version__)
print("CUDA=",torch.version.cuda)


for n in [
    10000,
    100000,
    1000000,
    10000000,
    100000000
]:

    identity=bench(kernel_identity,n)
    perm=bench(kernel_perm,n)

    print(
        f"N={n} "
        f"IDENTITY_MS={identity:.6f} "
        f"RUNTIME_MS={perm:.6f} "
        f"OVERHEAD={(perm/identity-1)*100:.3f}%"
    )
