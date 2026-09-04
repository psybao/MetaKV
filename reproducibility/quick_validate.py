#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

codec=load('metakv_codec','code/metakv_codec.py')
mapping=load('mapping','code/mapping.py')
codec_pass=binary_pass=ha_pass=mapping_pass=master_pass=figure_pass=False
try:
    assert codec.fp32_decode(codec.fp32_encode(1.25)) == 1.25
    codec_pass=True
    for bc,bf,zmin,zmax in [(14,18,-5.0,-0.5),(19,13,-5.75,-3.0),(22,10,-5.75,-3.0),(26,6,-5.75,-3.0)]:
        for z in [zmin,(zmin+zmax)/2,zmax]:
            w=codec.ha_fbms_encode(z,zmin,zmax,bc,bf)
            z2=codec.ha_fbms_decode(w,zmin,zmax,bc,bf)
            assert zmin-1e-12 <= z2 <= zmax+1e-12
            assert codec.ha_fbms_encode(z2,zmin,zmax,bc,bf)==w
    ha_pass=True
    for z in [-11.0,-3.0,5.0]:
        q=codec.binary32z_encode(z,-11.0,5.0)
        assert codec.binary32z_encode(codec.binary32z_decode(q,-11.0,5.0),-11.0,5.0)==q
    assert codec.amplification(1.0,256.0)==256.0
    binary_pass=True
    perm=list(reversed(range(32)))
    for w in [0,1,0x12345678,0xffffffff]:
        assert mapping.physical_to_logical(mapping.logical_to_physical(w,perm),perm)==w
    mapping_pass=True
    expected=json.loads((ROOT/'reproducibility/quick_validation_expected.json').read_text(encoding='utf-8'))
    master_pass=sha256(ROOT/expected['master']['path'])==expected['master']['sha256']
    figure_pass=all(sha256(ROOT/p)==h for p,h in expected['figure_data'].items())
except Exception:
    pass
final=all([codec_pass,binary_pass,ha_pass,mapping_pass,master_pass,figure_pass])
print('METAKV QUICK VALIDATION')
print(f'CODEC_PASS={codec_pass}')
print(f'BINARY32Z_PASS={binary_pass}')
print(f'HA_FBMS_PASS={ha_pass}')
print(f'MAPPING_PASS={mapping_pass}')
print(f'MASTER_HASH_PASS={master_pass}')
print(f'FIGURE_DATA_PASS={figure_pass}')
print(f'FINAL_PASS={final}')
raise SystemExit(0 if final else 1)
