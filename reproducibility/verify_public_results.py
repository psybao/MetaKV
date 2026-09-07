from pathlib import Path
import csv,hashlib
root=Path(__file__).resolve().parents[1]
p=root/'results/deployment/deployment_primary_results.csv'
rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
assert len(rows)==20
linux=[r for r in rows if r['execution_platform']=='RTX5080_LINUX_H800_LIKE']
assert len(linux)==4
assert abs(max(abs(float(r['mapped_vs_fp32_median_percent'])) for r in linux)-0.16855494236021062)<1e-12
assert all(r['status']=='FROZEN_PASS' for r in rows)
assert not any('r6' in r['execution_platform'].lower() for r in rows)
for d,summary,raw in [
 ('rtx5080_windows','p1e_summary.csv','p1e_raw_blocks.csv'),
 ('rtx5080_linux_h800_like','p1e_summary_execution_fixed.csv','p1e_raw_blocks.csv'),
 ('rtx4060ti','p1e_summary.csv','p1e_raw_blocks.csv'),
 ('amd_gfx942','p2c_summary.csv','p2c_raw_blocks.csv'),
 ('h800','p1e_summary.csv','p1e_raw_blocks.csv')]:
 base=root/'results/deployment'/d
 assert sum(1 for _ in csv.DictReader((base/summary).open(encoding='utf-8-sig')))==8,(d,'summary')
 assert sum(1 for _ in csv.DictReader((base/raw).open(encoding='utf-8-sig')))==120,(d,'raw')
print('PUBLIC_RESULT_VERIFY_PASS=True')
