"""Flag repeated patterns for listening review; never labels predictions true/false."""
import argparse
import collections
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('tracks',type=Path);a=p.parse_args()
s=json.loads(a.tracks.read_text(encoding='utf-8'));rows=[]
for tr in s['sources']:
 times=[e['time_s'] for e in tr['onsets']]
 gaps=[round(b-a,4) for a,b in zip(times,times[1:])]
 durations=[round(e['end_s']-e['start_s'],4) for e in tr['active_intervals']]
 rows.append({'id':tr['id'],'onsets':len(times),'inter_onset_gaps_s':dict(collections.Counter(gaps)),'activity_durations_s':dict(collections.Counter(durations)),'flags':['Exact repeated spacing/duration needs audio verification; periodic music may legitimately have these patterns.'],'not_measured':['missed onsets','false onsets','identity correctness','actual timing error']})
report={'input':str(a.tracks),'sources':rows,'ground_truth_available':False}
out=a.tracks.with_name('pattern-audit.json');out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
