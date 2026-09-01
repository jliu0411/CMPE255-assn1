from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'artifacts/results.json'
if not p.exists():raise SystemExit('Run scripts/run_lab.py first.')
d=json.loads(p.read_text());rows=d['demonstrations'];errors=[]
if len(rows)!=46:errors.append(f'Expected 46 demonstrations, found {len(rows)}')
if len({r['skill'] for r in rows})!=46:errors.append('Skill names are not unique')
for r in rows:
    for key in ['dataset','kaggle_slug','scenario','category','evidence','status']:
        if not r.get(key):errors.append(f"{r.get('skill')}: missing {key}")
    if not (ROOT/r['evidence']).exists():errors.append(f"{r['skill']}: evidence missing")
    if r['status']!='complete':errors.append(f"{r['skill']}: incomplete")
if errors:print('\n'.join(errors));sys.exit(1)
print('Coverage gate passed: 46/46 skills have complete, traceable demonstrations.')
