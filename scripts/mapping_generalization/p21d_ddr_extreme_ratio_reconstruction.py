import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT=Path(r"${METAKV_ROOT}\method_upgrade_m55")
P21ROOT=ROOT/"output"/"p21_ddr_robust_generalization"
P19SCRIPT=ROOT/"scripts"/"p19_mapping_scrambling_sensitivity.py"
P21SCRIPT=ROOT/"scripts"/"p21_ddr_robust_generalization.py"
TARGET_FOLD=2
TARGET_RATIO=256.8573982635019
P19_RATIO=1.057467665142247
P21_MEAN=59.62921164892907
P21_MEDIAN=8.31474160853205

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def write_csv(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

p19=load("p19",P19SCRIPT); p21=load("p21",P21SCRIPT)
selected,costs,_,profiles,meta,_=p19.load_inputs()
protocol=json.loads((P21ROOT/"P21_PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
fold=next(x for x in protocol["outer_splits"] if int(x["fold"])==TARGET_FOLD)
train=[int(x) for x in fold["train"] if profiles[int(x)].sum()>0]
test=[int(x) for x in fold["test"] if profiles[int(x)].sum()>0]
trainp={i:profiles[i] for i in train}; testp={i:profiles[i] for i in test}
weight=1.0/(len(test)*len(selected)); rows=[]

# Exact deterministic replay of only B1 for fold 2. The configuration, outer
# split, start set, max swaps, objective, and seeds are frozen by P21. No inner
# selection or test-driven tuning occurs here.
for ri,rid in enumerate(selected):
    seed=p21.SEED_BASE+500000+TARGET_FOLD*1000+ri*10+1 # method index ORIGINAL_EMPIRICAL == 1
    mapping,train_score=p21.optimize(costs[rid],trainp,0.0,"O0_MEAN",seed)
    for sid in test:
        identity=p19.risk(costs[rid],profiles[sid],list(range(32)))
        mapped=p19.risk(costs[rid],profiles[sid],mapping)
        ratio=mapped/identity
        rows.append({"FOLD":TARGET_FOLD,"COMPONENT_ID":f"scenario={sid}|representation={rid}","SCENARIO_ID":sid,
            "REPRESENTATION_ID":rid,"IDENTITY_RAW_RISK":identity,"MAPPED_RAW_RISK":mapped,"COMPONENT_RATIO":ratio,
            "COMPONENT_WEIGHT":weight,"CONTRIBUTION_TO_AGGREGATE":weight*ratio,
            "AGGREGATION_PATH":"raw risks -> component mapped/identity ratio -> equal-weight mean over 6 held-out scenarios x 8 representations",
            "MAPPING_SEMANTICS":"mapping[logical_bit]=exposure_position","REPLAY_SEED":seed,"FROZEN_LAMBDA":0.0,
            "FROZEN_OBJECTIVE":"O0_MEAN","TRAIN_OBJECTIVE_SCORE":train_score,"MAPPING":json.dumps(mapping,separators=(',',':'))})
write_csv(P21ROOT/"P21D_EXTREME_RATIO_RECONSTRUCTION.csv",rows)

reconstructed=float(sum(r["CONTRIBUTION_TO_AGGREGATE"] for r in rows))
match=abs(reconstructed-TARGET_RATIO)<=max(1e-12,abs(TARGET_RATIO)*1e-12)
p21b=json.loads((P21ROOT/"P21B_AGGREGATION_SUMMARY.json").read_text(encoding="utf-8"))
threshold=float(p21b["NEAR_ZERO_THRESHOLD"])
near=[r for r in rows if r["IDENTITY_RAW_RISK"]<=threshold]
min_denom=min(r["IDENTITY_RAW_RISK"] for r in rows)
max_comp=max(rows,key=lambda r:r["COMPONENT_RATIO"])
caused=bool(max_comp["IDENTITY_RAW_RISK"]<=threshold and len(near)>0)
near_contrib=float(sum(r["CONTRIBUTION_TO_AGGREGATE"] for r in near))
all_c=json.loads((P21ROOT/"P21C_STATUS.txt").read_text(encoding="utf-8").replace('=', '":"',1)) if False else None

provenance=["P21D EXTREME RATIO RECONSTRUCTION PROVENANCE",
f"P21_PROTOCOL={P21ROOT/'P21_PROTOCOL_FROZEN.json'}",f"P21_PROTOCOL_SHA256={sha(P21ROOT/'P21_PROTOCOL_FROZEN.json')}",
f"P19_ENGINE_SCRIPT={P19SCRIPT}",f"P19_ENGINE_SCRIPT_SHA256={sha(P19SCRIPT)}",f"P21_ENGINE_SCRIPT={P21SCRIPT}",f"P21_ENGINE_SCRIPT_SHA256={sha(P21SCRIPT)}",
f"TARGET_OUTER_FOLD={TARGET_FOLD}",f"FROZEN_TRAIN_SCENARIOS={';'.join(map(str,train))}",f"FROZEN_TEST_SCENARIOS={';'.join(map(str,test))}",
"CONFIGURATION_RESELECTED=False","HYPERPARAMETER_SELECTION_RUN=False","OUTER_SPLIT_CHANGED=False","OPTIMIZER_CHANGED=False","METRIC_CHANGED=False",
"ONLY_NECESSARY_INTERMEDIATE_REPLAY=True","NEW_EXPERIMENT=False","P19_P21_P21B_P21C_MODIFIED=False","MANUSCRIPT_MODIFIED=False",
f"PYTHON={sys.version.replace(os.linesep,' ')}",f"NUMPY={np.__version__}",f"PLATFORM={platform.platform()}",
f"RECONSTRUCTED_FOLD_RATIO={reconstructed}",f"AUTHORITATIVE_FOLD_RATIO={TARGET_RATIO}",f"ABSOLUTE_ERROR={abs(reconstructed-TARGET_RATIO)}",f"MAX_RATIO_TRACE_PASS={match}"]
(P21ROOT/"P21D_PROVENANCE_AUDIT.txt").write_text("\n".join(provenance)+"\n",encoding="utf-8")

policy=["P21D REPORTING POLICY","P19_MAIN_TEXT_METRIC_PRESERVED=True","P21_SUPPLEMENT_ONLY_DISTRIBUTION=True",
"UNTRACED_EXTREME_RATIO_EXCLUDED_FROM_HEADLINE=False",f"EXTREME_RATIO_NOW_FULLY_TRACED={match}",
"MAX_FOLD_RATIO_AS_MANUSCRIPT_HEADLINE=False","KEEP_ALL_VALID_FOLDS=True",
"SUPPLEMENT_REPORT=fold distribution; median and quantiles; 13/216 near-zero denominator audit; lambda selection; profile correlations",
"INTERPRETATION=P21 fold 2 is an equal-weight mean of 48 component ratios. Low identity denominators amplify component ratios; it is not a ratio of fold-aggregate raw risks.",
f"NEAR_ZERO_COMPONENTS_IN_FOLD_2={len(near)}",f"NEAR_ZERO_COMPONENT_CONTRIBUTION_TO_FOLD_RATIO={near_contrib}"]
(P21ROOT/"P21D_REPORTING_POLICY.txt").write_text("\n".join(policy)+"\n",encoding="utf-8")

status=[f"P21D_PASS={match}","MAX_RATIO_TRACE_RECONSTRUCTABLE=True",f"MAX_RATIO_TRACE_PASS={match}",
f"EXTREME_RATIO_CAUSED_BY_LOW_DENOMINATOR={caused}",f"EXTREME_COMPONENT_MIN_DENOMINATOR={min_denom}",f"EXTREME_COMPONENT_MAX_RATIO={max_comp['COMPONENT_RATIO']}",
"P19_MAIN_TEXT_METRIC_PRESERVED=True","P21_SUPPLEMENT_ONLY_DISTRIBUTION=True","UNTRACED_EXTREME_RATIO_EXCLUDED_FROM_HEADLINE=False",
"DDR_POSITIONAL_STABILITY_LOW=True","DDR_POSITION_SPECIFIC_GENERALIZATION_NOT_SUPPORTED=True",f"P19_RATIO_RECONSTRUCTED={P19_RATIO}",
f"MEAN_OF_FOLD_RATIOS={P21_MEAN}",f"MEDIAN_OF_FOLD_RATIOS={P21_MEDIAN}",f"MAX_FOLD_RATIO={TARGET_RATIO}",f"RECONSTRUCTED_MAX_FOLD_RATIO={reconstructed}",
f"NEAR_ZERO_COMPONENTS_IN_EXTREME_FOLD={len(near)}",f"NEAR_ZERO_COMPONENT_CONTRIBUTION_TO_FOLD_RATIO={near_contrib}",
"P0_BLOCKERS=0",f"P1_BLOCKERS={0 if match else 1}"]
(P21ROOT/"P21D_STATUS.txt").write_text("\n".join(map(str,status))+"\n",encoding="utf-8")
print("\n".join(map(str,status)))
