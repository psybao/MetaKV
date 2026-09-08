from pathlib import Path
import argparse, shutil

p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output',required=True); a=p.parse_args()
src=Path(a.source); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
svgs=sorted(src.glob('figure_*.svg'))
if len(svgs) != 6: raise SystemExit(f'expected 6 SVG figure sources, found {len(svgs)}')
for f in svgs: shutil.copy2(f,out/f.name)
print(f'FIGURE_SOURCE_REPRODUCTION_PASS=True COUNT={len(svgs)}')
