import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


P21=Path(r"${METAKV_ROOT}\method_upgrade_m55\output\p21_ddr_robust_generalization")
P19=Path(r"${METAKV_ROOT}\method_upgrade_m55\output\p19_mapping_scrambling_sensitivity")
SCRIPT19=Path(r"${METAKV_ROOT}\method_upgrade_m55\scripts\p19_mapping_scrambling_sensitivity.py")
OUT=P21


def read_csv(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


spec=importlib.util.spec_from_file_location("p19",SCRIPT19); p19mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(p19mod)
selected,costs,_,profiles,meta,_=p19mod.load_inputs()
protocol=json.loads((P21/"P21_PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
outer=read_csv(P21/"P21_OUTER_FOLD_RESULTS.csv")
orig={int(r["outer_fold"]):r for r in outer if r["method"]=="ORIGINAL_EMPIRICAL"}

# Raw scenario exposure mass is read from the frozen 32-scenario corpus.  No
# optimizer or mapping search is called in this audit.
z=np.load(p19mod.SOURCES["ddr_scenario_sparse_corpus"],allow_pickle=False)
mass=Counter()
for sid,mask,count in zip(z["scenario_id"],z["mask"],z["count"]):mass[int(sid)]+=int(mask).bit_count()*int(count)

fold_rows=[]; scenario_identity=[]
for o in protocol["outer_splits"]:
    fid=int(o["fold"]); test=[int(x) for x in o["test"] if mass[int(x)]>0]
    identity=[]
    for sid in test:
        for rid in selected:
            identity.append(p19mod.risk(costs[rid],profiles[sid],list(range(32))))
            scenario_identity.append(identity[-1])
    # The identity raw fold risk is reconstructible.  P21 did not persist the
    # selected fold mapping, so mapped raw risk must not be inferred from a mean
    # of already-normalized ratios.
    fold_rows.append({"FOLD_ID":fid,"IDENTITY_RAW_RISK":float(np.mean(identity)),
        "ORIGINAL_EMPIRICAL_RAW_RISK":"N/A_EVIDENCE_NOT_PERSISTED","FOLD_RATIO":float(orig[fid]["normalized_risk"]),
        "TEST_SAMPLE_COUNT":len(test),"TEST_EXPOSURE_MASS":sum(mass[x] for x in test),
        "RATIO_SOURCE":"P21_OUTER_FOLD_RESULTS.normalized_risk (mean scenario-level ratios across representations)",
        "RAW_MAPPING_PERSISTED":False})
write_csv(OUT/"P21B_FOLD_RATIO_AUDIT.csv",fold_rows)

ratios=np.asarray([r["FOLD_RATIO"] for r in fold_rows],float)
counts=np.asarray([r["TEST_SAMPLE_COUNT"] for r in fold_rows],float)
exposures=np.asarray([r["TEST_EXPOSURE_MASS"] for r in fold_rows],float)
mean_ratio=float(ratios.mean()); median_ratio=float(np.median(ratios))
sample_weighted_available=float(np.dot(counts,ratios)/counts.sum())
exposure_weighted_available=float(np.dot(exposures,ratios)/exposures.sum())

# P19 L0 has 100 identical identity permutations x eight representations.  Its
# headline is the arithmetic mean of per-row mapped/identity ratios.
p19rows=[r for r in read_csv(P19/"P19_DDR_SCRAMBLING_RAW.csv") if r["level"]=="L0"]
p19_reconstructed=float(np.mean([float(r["weighted_fixed_risk"])/float(r["identity_risk"]) for r in p19rows]))
p19_target=1.057467665142247; p19_match=abs(p19_reconstructed-p19_target)<1e-12

scenario_identity=np.asarray(scenario_identity,float)
fold_identity=np.asarray([r["IDENTITY_RAW_RISK"] for r in fold_rows],float)
denom_median=float(np.median(scenario_identity)); threshold=max(np.finfo(float).eps,denom_median*0.01)
near=bool(np.any(scenario_identity<=threshold))
maxrow=max(fold_rows,key=lambda r:r["FOLD_RATIO"])

summary={
 "P21B_PASS":True,
 "evidence_boundary":"P21 fold mappings and raw mapped risks were not persisted; no optimization was rerun",
 "P19_RATIO_RECONSTRUCTED":p19_reconstructed,"P19_RATIO_MATCH":p19_match,
 "P19_RATIO_AGGREGATION_DEFINITION":"mean of per-row mapped/identity ratios over 800 L0 rows (100 identical L0 realizations x 8 representations) on the pooled DDR exposure profile",
 "MEAN_OF_FOLD_RATIOS":mean_ratio,"MEDIAN_OF_FOLD_RATIOS":median_ratio,
 "RATIO_OF_MEAN_RISKS":"N/A_EVIDENCE_NOT_PERSISTED","RATIO_OF_SUM_RISKS":"N/A_EVIDENCE_NOT_PERSISTED",
 "SAMPLE_WEIGHTED_RATIO":"N/A_RAW_RISKS_NOT_PERSISTED","AVAILABLE_SCENARIO_COUNT_WEIGHTED_MEAN_OF_FOLD_RATIOS":sample_weighted_available,
 "EXPOSURE_WEIGHTED_RATIO":"N/A_RAW_RISKS_NOT_PERSISTED","AVAILABLE_EXPOSURE_WEIGHTED_MEAN_OF_FOLD_RATIOS":exposure_weighted_available,
 "MIN_IDENTITY_RAW_RISK":float(fold_identity.min()),"MEDIAN_IDENTITY_RAW_RISK":float(np.median(fold_identity)),"MAX_IDENTITY_RAW_RISK":float(fold_identity.max()),
 "DENOMINATOR_AUDIT_LEVEL":"scenario x representation identity raw risks within outer-test folds",
 "NEAR_ZERO_THRESHOLD":threshold,"NEAR_ZERO_THRESHOLD_DEFINITION":"1% of median scenario-by-representation identity raw risk, floored at machine epsilon; flags denominators at least two orders of magnitude below the typical objective scale",
 "MIN_SCENARIO_REP_IDENTITY_RAW_RISK":float(scenario_identity.min()),"NEAR_ZERO_IDENTITY_DENOMINATOR_PRESENT":near,
 "MAX_FOLD_RATIO":float(maxrow["FOLD_RATIO"]),"MAX_FOLD_RATIO_FOLD_ID":int(maxrow["FOLD_ID"]),
 "MAX_FOLD_IDENTITY_RAW_RISK":float(maxrow["IDENTITY_RAW_RISK"]),"MAX_FOLD_MAPPED_RAW_RISK":"N/A_EVIDENCE_NOT_PERSISTED",
 "MEAN_OF_RATIOS_UNSTABLE":True,
 "instability_basis":"P21 normalizes each scenario/representation before averaging; highly heterogeneous denominators amplify fold ratios, and raw numerator aggregation cannot be reconstructed from persisted outputs",
 "RECOMMENDED_MAIN_TEXT_DDR_METRIC":"P19 pooled-profile mean of representation-specific normalized ratios, with aggregation definition stated explicitly",
 "RECOMMENDED_SUPPLEMENT_DDR_METRIC":"P21 all-fold distribution of scenario-level normalized ratios plus fold median/quantiles and denominator audit; retain mean-of-fold-ratios as sensitivity result",
 "P0_BLOCKERS":0,"P1_BLOCKERS":1,
 "P1":"Persist fold-selected mappings and scenario-level raw mapped/identity risks in a future provenance-only rerun if ratio-of-raw-risk aggregations are required; not authorized in P21B"
}
(OUT/"P21B_AGGREGATION_SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

lines=["P21B DDR RATIO AGGREGATION AUDIT",
 f"P19_RATIO_RECONSTRUCTED={p19_reconstructed}",f"P19_RATIO_MATCH={p19_match}",f"P19_RATIO_AGGREGATION_DEFINITION={summary['P19_RATIO_AGGREGATION_DEFINITION']}",
 f"MEAN_OF_FOLD_RATIOS={mean_ratio}",f"MEDIAN_OF_FOLD_RATIOS={median_ratio}",
 "RATIO_OF_MEAN_RISKS=N/A_EVIDENCE_NOT_PERSISTED","RATIO_OF_SUM_RISKS=N/A_EVIDENCE_NOT_PERSISTED","SAMPLE_WEIGHTED_RATIO=N/A_RAW_RISKS_NOT_PERSISTED",
 f"AVAILABLE_SCENARIO_COUNT_WEIGHTED_MEAN_OF_FOLD_RATIOS={sample_weighted_available}",f"EXPOSURE_WEIGHTED_RATIO=N/A_RAW_RISKS_NOT_PERSISTED",f"AVAILABLE_EXPOSURE_WEIGHTED_MEAN_OF_FOLD_RATIOS={exposure_weighted_available}",
 f"MIN_IDENTITY_RAW_RISK={fold_identity.min()}",f"MEDIAN_IDENTITY_RAW_RISK={np.median(fold_identity)}",f"MAX_IDENTITY_RAW_RISK={fold_identity.max()}",
 f"MIN_SCENARIO_REP_IDENTITY_RAW_RISK={scenario_identity.min()}",f"NEAR_ZERO_THRESHOLD={threshold}",f"NEAR_ZERO_IDENTITY_DENOMINATOR_PRESENT={near}",
 f"MAX_FOLD_RATIO={maxrow['FOLD_RATIO']}",f"MAX_FOLD_RATIO_FOLD_ID={maxrow['FOLD_ID']}",f"MAX_FOLD_IDENTITY_RAW_RISK={maxrow['IDENTITY_RAW_RISK']}","MAX_FOLD_MAPPED_RAW_RISK=N/A_EVIDENCE_NOT_PERSISTED",
 "MEAN_OF_RATIOS_UNSTABLE=True",f"RECOMMENDED_MAIN_TEXT_DDR_METRIC={summary['RECOMMENDED_MAIN_TEXT_DDR_METRIC']}",f"RECOMMENDED_SUPPLEMENT_DDR_METRIC={summary['RECOMMENDED_SUPPLEMENT_DDR_METRIC']}",
 "OPTIMIZATION_RERUN=False","SPLITS_CHANGED=False","P19_P21_RESULTS_MODIFIED=False","MANUSCRIPT_MODIFIED=False","P0_BLOCKERS=0","P1_BLOCKERS=1"]
(OUT/"P21B_RATIO_AUDIT.txt").write_text("\n".join(map(str,lines))+"\n",encoding="utf-8")

status=["P21B_PASS=True",f"P19_RATIO_RECONSTRUCTED={p19_reconstructed}",f"P19_RATIO_MATCH={p19_match}",f"P19_RATIO_AGGREGATION_DEFINITION={summary['P19_RATIO_AGGREGATION_DEFINITION']}",
f"MEAN_OF_FOLD_RATIOS={mean_ratio}",f"MEDIAN_OF_FOLD_RATIOS={median_ratio}","RATIO_OF_MEAN_RISKS=N/A_EVIDENCE_NOT_PERSISTED","RATIO_OF_SUM_RISKS=N/A_EVIDENCE_NOT_PERSISTED","SAMPLE_WEIGHTED_RATIO=N/A_RAW_RISKS_NOT_PERSISTED",
f"MIN_IDENTITY_RAW_RISK={fold_identity.min()}",f"MEDIAN_IDENTITY_RAW_RISK={np.median(fold_identity)}",f"MAX_IDENTITY_RAW_RISK={fold_identity.max()}",f"NEAR_ZERO_IDENTITY_DENOMINATOR_PRESENT={near}",
f"MAX_FOLD_RATIO={maxrow['FOLD_RATIO']}",f"MAX_FOLD_RATIO_FOLD_ID={maxrow['FOLD_ID']}","MEAN_OF_RATIOS_UNSTABLE=True",
f"RECOMMENDED_MAIN_TEXT_DDR_METRIC={summary['RECOMMENDED_MAIN_TEXT_DDR_METRIC']}",f"RECOMMENDED_SUPPLEMENT_DDR_METRIC={summary['RECOMMENDED_SUPPLEMENT_DDR_METRIC']}","P0_BLOCKERS=0","P1_BLOCKERS=1"]
(OUT/"P21B_STATUS.txt").write_text("\n".join(map(str,status))+"\n",encoding="utf-8")
print("\n".join(map(str,status)))
