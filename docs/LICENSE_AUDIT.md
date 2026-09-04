# License audit

| File/path | Origin | License | Redistributable | Included | Notes |
|---|---|---|---|---|---|
| code/metakv_codec.py | MetaKV project-authored reference | MIT | Yes | Yes | Derived from frozen formula traceability. |
| code/mapping.py, code/fault_screen.py, code/weighting.py | MetaKV project-authored reference | MIT | Yes | Yes | No third-party source embedded. |
| code/experiments/ | MetaKV project experiment scripts | MIT for project-authored portions | Yes after path/privacy scan | Yes | Selected authoritative/final branches only. |
| data/, figure_data/, tables/ | MetaKV-derived outputs | MIT package grant | Yes | Yes | No model weights or third-party raw corpus. |
| figures/ | MetaKV-generated publication figures | MIT package grant | Yes | Yes | Derived from included processed sources. |
| HBM2 raw characterization corpus | Third party | Not established here | No | No | Cite and obtain from the original source. |
| Model weights/tokenizers | Meta/Qwen providers | Provider-specific | No package permission asserted | No | Obtain separately. |
| Python/CUDA/ROCm environments | Third parties | Component-specific | Not bundled | No | Install from official providers. |

LICENSE_AUDIT_PASS=True
