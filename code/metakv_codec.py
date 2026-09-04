"""CPU reference codecs used by MetaKV. Project-authored, MIT licensed."""
from __future__ import annotations
import math
import struct

U32_MAX = (1 << 32) - 1

def _check(zmin: float, zmax: float) -> None:
    if not math.isfinite(zmin) or not math.isfinite(zmax) or zmax <= zmin:
        raise ValueError("require finite zmin < zmax")

def ha_fbms_encode(z: float, zmin: float, zmax: float, coarse_bits: int, fine_bits: int) -> int:
    _check(zmin, zmax)
    if coarse_bits + fine_bits != 32 or coarse_bits < 1 or fine_bits < 1:
        raise ValueError("coarse_bits + fine_bits must equal 32")
    u = min(max((z - zmin) / (zmax - zmin) * (coarse_bits + 1), 0.0), float(coarse_bits + 1))
    fmax = (1 << fine_bits) - 1
    if u >= coarse_bits + 1:
        coarse, fine = coarse_bits, fmax
    else:
        coarse = min(int(math.floor(u)), coarse_bits)
        fine = int(round((u - coarse) * fmax))
        fine = min(max(fine, 0), fmax)
    thermometer = (1 << coarse) - 1
    return (thermometer | (fine << coarse_bits)) & U32_MAX

def ha_fbms_decode(word: int, zmin: float, zmax: float, coarse_bits: int, fine_bits: int) -> float:
    _check(zmin, zmax)
    coarse_mask = (1 << coarse_bits) - 1
    fine_mask = (1 << fine_bits) - 1
    coarse = (int(word) & coarse_mask).bit_count()
    fine = (int(word) >> coarse_bits) & fine_mask
    return zmin + (zmax - zmin) / (coarse_bits + 1) * (coarse + fine / fine_mask)

def binary32z_encode(z: float, zmin: float, zmax: float) -> int:
    _check(zmin, zmax)
    u = min(max((z - zmin) / (zmax - zmin), 0.0), 1.0)
    return int(round(u * U32_MAX))

def binary32z_decode(word: int, zmin: float, zmax: float) -> float:
    _check(zmin, zmax)
    return zmin + (int(word) & U32_MAX) / U32_MAX * (zmax - zmin)

def fp32_encode(value: float) -> int:
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]

def fp32_decode(word: int) -> float:
    return struct.unpack('<f', struct.pack('<I', int(word) & U32_MAX))[0]

def amplification(scale: float, fault_scale: float) -> float:
    if scale <= 0 or fault_scale <= 0 or not math.isfinite(scale) or not math.isfinite(fault_scale):
        return math.inf
    return max(fault_scale / scale, scale / fault_scale)
