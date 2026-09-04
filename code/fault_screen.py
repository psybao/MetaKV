"""Small logical one-bit fault screen for MetaKV codecs."""
from metakv_codec import ha_fbms_decode, binary32z_decode, amplification

def screen(word, decode, *args):
    base = 2.0 ** decode(word, *args)
    rows = []
    for bit in range(32):
        fault = 2.0 ** decode(word ^ (1 << bit), *args)
        rows.append((bit, amplification(base, fault)))
    return rows

def screen_ha(word, zmin, zmax, coarse_bits, fine_bits):
    return screen(word, ha_fbms_decode, zmin, zmax, coarse_bits, fine_bits)

def screen_binary32z(word, zmin, zmax):
    return screen(word, binary32z_decode, zmin, zmax)
