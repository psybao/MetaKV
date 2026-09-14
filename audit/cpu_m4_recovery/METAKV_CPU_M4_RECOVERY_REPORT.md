# MetaKV CPU / M4 backup recovery report

## Discovery

- Backup root: `${METAKV_SOURCE_ARCHIVE_ROOT}`
- Files recursively inventoried: 101
- Archives read-only listed: 18
- Original archives modified: no
- Audit extraction root: `${METAKV_ROOT}\output\cpu_m4_recovery_audit\tmp`

## Recovered authoritative evidence

Intel Core i7-14700KF, AMD Ryzen 7 7840HS, and Apple M4 each have a frozen CPU codec package with environment metadata, source, raw/summary results, paired-block timing, 10,000-resample bootstrap intervals, final PASS gates, and SHA256 provenance. Their maximum absolute mapped-versus-identity median differences over the 12 controlled configurations are 5.276852405917%, 6.303494066428%, and 6.712926515948%, respectively; all 36 evaluated configurations are within 8%.

The AMD Ryzen CPU evidence is distinct from the AMD gfx942 accelerator evidence. No CPU-versus-GPU latency comparison is made.

Apple M4 additionally has authoritative functional-only PyTorch MPS and MLX Metal checks. MPS and MLX are kept separate from the CPU timing result and are not characterized as performance benchmarks.

## TJS reconciliation

The P15 manuscript contained Intel-only CPU wording based on earlier evidence and omitted recovered AMD CPU and Apple M4 evidence. V2 therefore applies the allowed minimal revision: one restrained cross-ISA paragraph in Section 6.3 and one detailed Supplement S10 table. GPU, fault, codec, mapping, equation, reference, archive, and limitation content is otherwise unchanged.

INTEL_CPU_AUTHORITATIVE=True
AMD_CPU_AUTHORITATIVE=True
M4_CPU_AUTHORITATIVE=True
M4_MPS_AUTHORITATIVE=True_FUNCTIONAL_ONLY
M4_MLX_AUTHORITATIVE=True_FUNCTIONAL_ONLY
TJS_REVISION_REQUIRED=True
