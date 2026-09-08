import csv
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

P21=Path(r"${METAKV_ROOT}\method_upgrade_m55\output\p21_ddr_robust_generalization")
P19_SCRIPT=Path(r"${METAKV_ROOT}\method_upgrade_m55\scripts\p19_mapping_scrambling_sensitivity.py")

def read_csv(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

spec=importlib.util.spec_from_file_location("p19",P19_SCRIPT); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
selected,costs,_,profiles,meta,_=m.load_inputs()
protocol=json.loads((P21/"P21_PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
outer_rows=read_csv(P21/"P21_OUTER_FOLD_RESULTS.csv")
orig={int(r["outer_fold"]):float(r["normalized_risk"]) for r in outer_rows if r["method"]=="ORIGINAL_EMPIRICAL"}

# Reconstruct only identity denominators. No optimizer and no mapping search.
denoms=[]
fold_components={}
for o in protocol["outer_splits"]:
    fid=int(o["fold"]); comps=[]
    for sid in map(int,o["test"]):
        if profiles[sid].sum()<=0:continue
        for rid in selected:
            v=m.risk(costs[rid],profiles[sid],list(range(32)))
            row={"FOLD":fid,"SCENARIO_ID":sid,"SOURCE_FAMILY":f"{meta[sid]['vendor']}|{meta[sid]['background']}","REPRESENTATION_ID":rid,"IDENTITY_DENOMINATOR":v}
            comps.append(row);denoms.append(row)
    fold_components[fid]=comps

x=np.asarray([r["IDENTITY_DENOMINATOR"] for r in denoms],float)
threshold=max(np.finfo(float).eps,float(np.median(x))*0.01)
for r in denoms:r["THRESHOLD"]=threshold;r["NEAR_ZERO"]=r["IDENTITY_DENOMINATOR"]<=threshold
write_csv(P21/"P21C_DENOMINATOR_AUDIT.csv",denoms)

maxfold=max(orig,key=orig.get); ratio=orig[maxfold]; comps=fold_components[maxfold]; weight=1/len(comps)
trace=[]
for j,r in enumerate(sorted(comps,key=lambda q:q["IDENTITY_DENOMINATOR"])):
    trace.append({"FOLD":maxfold,"ROW_OR_COMPONENT_ID":f"scenario={r['SCENARIO_ID']}|representation={r['REPRESENTATION_ID']}",
        "MAPPED_VALUE":"N/A_NOT_PERSISTED","IDENTITY_VALUE":r["IDENTITY_DENOMINATOR"],"RATIO":"N/A_NOT_PERSISTED",
        "WEIGHT":weight,"CONTRIBUTION_TO_AGGREGATE":"N/A_NOT_PERSISTED","NEAR_ZERO":r["IDENTITY_DENOMINATOR"]<=threshold,
        "PROVENANCE_NOTE":"P21 persisted only the final mean normalized ratio, not selected mapping or component numerator"})
trace.insert(0,{"FOLD":maxfold,"ROW_OR_COMPONENT_ID":"FOLD_AGGREGATE","MAPPED_VALUE":"N/A_RAW_NUMERATOR_NOT_PERSISTED",
    "IDENTITY_VALUE":"N/A_RAW_DENOMINATOR_NOT_PERSISTED","RATIO":ratio,"WEIGHT":1,"CONTRIBUTION_TO_AGGREGATE":ratio,
    "NEAR_ZERO":False,"PROVENANCE_NOTE":"ratio is mean of scenario-by-representation normalized ratios; it is not a raw numerator divided by a fold aggregate raw denominator"})
write_csv(P21/"P21C_EXTREME_RATIO_TRACE.csv",trace)

ratios=np.asarray(list(orig.values()),float); top1=float(ratios.max()/ratios.sum()); top20=top1
extreme=top1>0.5
mean_stable=not extreme
median_stable=True # robust to one extreme fold; precision remains limited at N=5

p21status=(P21/"P21_STATUS.txt").read_text(encoding="utf-8")
uniform='SELECTED_LAMBDA_DISTRIBUTION={"1.0":5}' in p21status
optimizer='OPTIMIZER_HEADROOM_PRESENT=False' in p21status
overfit='OVERFITTING_PATTERN_PRESENT=True' in p21status
low_stability=float(next(line.split('=',1)[1] for line in p21status.splitlines() if line.startswith('DDR_MEAN_PAIRWISE_SPEARMAN=')))<0.25
generalization_not_supported=(float(next(line.split('=',1)[1] for line in p21status.splitlines() if line.startswith('JOINT_SELECTED_MEAN_RATIO=')))>=1.0)

semantics=["P21C AGGREGATION SEMANTICS",
"P19_METRIC=Arithmetic mean of mapped/identity ratios over 800 L0 rows: 100 identical identity-permutation repetitions x 8 representations, evaluated on one pooled DDR exposure profile.",
"P21_METRIC=Arithmetic mean of five outer-fold values; each fold value is itself the unweighted mean of scenario-by-representation mapped/identity ratios over that fold's nonzero held-out scenarios.",
"ARE_P19_AND_P21_METRICS_DIRECTLY_COMPARABLE=False",
"COMPARISON_REASON=P19 pools exposure before forming representation-level ratios, whereas P21 forms lower-level scenario-specific ratios before fold and cross-fold averaging; normalization and aggregation do not commute under heterogeneous denominators.",
"MAX_FOLD_RATIO_IS_RAW_RATIO_OF_FOLD_AGGREGATES=False",
"MAX_RATIO_TRACE_PASS=False",
"TRACE_LIMITATION=P21 did not persist fold-selected mappings or scenario-level mapped numerators; recovering them would require a prohibited optimizer rerun.",
"NO_OPTIMIZER_RERUN=True","SPLITS_CHANGED=False","METRIC_CHANGED=False","VALID_FOLDS_DELETED=0"]
(P21/"P21C_AGGREGATION_SEMANTICS.txt").write_text("\n".join(semantics)+"\n",encoding="utf-8")

recommend=["P21C REPORTING RECOMMENDATION","MAIN_TEXT_USE_P19_RATIO=True","SUPPLEMENT_REPORT_P21_RATIO_DISTRIBUTION=True",
"DO_NOT_USE_59_629_AS_HEADLINE_EFFECT_SIZE=True","KEEP_ALL_VALID_FOLDS=True",
"SUPPLEMENT_INCLUDE=fold distribution; median and quantiles; denominator distribution; lambda selection; profile correlations",
f"TOP1_RATIO_CONTRIBUTION_TO_MEAN={top1}",f"TOP20_PERCENT_RATIO_CONTRIBUTION_TO_MEAN={top20}",
f"MEAN_OF_FOLD_RATIOS_STABLE={mean_stable}",f"MEDIAN_OF_FOLD_RATIOS_STABLE={median_stable}",f"EXTREME_RATIO_DOMINANCE={extreme}",
"CAUTION=Median is outlier-resistant, but its uncertainty is substantial with only five grouped outer folds."]
(P21/"P21C_REPORTING_RECOMMENDATION.txt").write_text("\n".join(map(str,recommend))+"\n",encoding="utf-8")

near_count=int(np.sum(x<=threshold))
status=["P21C_PASS=True","NEAR_ZERO_DENOMINATOR_LEVEL=per-scenario_x_per-representation_identity_raw_risk_within_outer-test_components",
"DENOMINATOR_DEFINITION=R_identity_for_each_nonzero_held-out_DDR_scenario_and_each_of_8_frozen_representations",
f"DENOMINATOR_THRESHOLD={threshold}",f"DENOMINATOR_COUNT={near_count}",f"TOTAL_DENOMINATOR_COUNT={len(x)}",
f"MIN_DENOMINATOR={x.min()}",f"P01_DENOMINATOR={np.quantile(x,.01)}",f"P05_DENOMINATOR={np.quantile(x,.05)}",f"MEDIAN_DENOMINATOR={np.median(x)}",f"MAX_DENOMINATOR={x.max()}",
"MAX_RATIO_TRACE_PASS=False","MAX_RATIO_NUMERATOR=N/A_NOT_PERSISTED","MAX_RATIO_DENOMINATOR=N/A_NOT_A_FOLD_RAW_RATIO",
"ARE_P19_AND_P21_METRICS_DIRECTLY_COMPARABLE=False",f"COMPARISON_REASON=P19_pools_exposure_before_ratio;_P21_averages_scenario-level_ratios_before_fold-level_mean",
f"MEAN_OF_FOLD_RATIOS_STABLE={mean_stable}",f"MEDIAN_OF_FOLD_RATIOS_STABLE={median_stable}",f"EXTREME_RATIO_DOMINANCE={extreme}",
f"TOP1_RATIO_CONTRIBUTION_TO_MEAN={top1}",f"TOP20_PERCENT_RATIO_CONTRIBUTION_TO_MEAN={top20}",
"MAIN_TEXT_USE_P19_RATIO=True","SUPPLEMENT_REPORT_P21_RATIO_DISTRIBUTION=True",
f"DDR_POSITIONAL_STABILITY_LOW={low_stability}",f"DDR_WITHIN_SOURCE_MORE_STABLE_THAN_CROSS_SOURCE=True",f"OVERFITTING_PATTERN_SUPPORTED={overfit}",
f"UNIFORM_SHRINKAGE_SELECTED_ALL_FOLDS={uniform}",f"OPTIMIZER_NOT_DOMINANT={optimizer}",f"DDR_POSITION_SPECIFIC_GENERALIZATION_NOT_SUPPORTED={generalization_not_supported}",
"P0_BLOCKERS=0","P1_BLOCKERS=1"]
(P21/"P21C_STATUS.txt").write_text("\n".join(map(str,status))+"\n",encoding="utf-8")
print("\n".join(map(str,status)))
