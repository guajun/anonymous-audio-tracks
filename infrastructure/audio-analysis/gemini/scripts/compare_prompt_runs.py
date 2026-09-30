"""Compare saved experiment outputs, including invalid responses, without repairing them."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze  # noqa: E402

RUNS = [
    ('v1', '20260929-001042-747776'),
    ('v2', '20260929-001533-524108'),
    ('v2 + SAM auxiliary', '20260929-001802-124793'),
]
HISTORY = ROOT / 'history'
rows = []
for label, run_id in RUNS:
    run = HISTORY / 'runs' / run_id
    raw = (run / 'response.txt').read_text(encoding='utf-8').strip()
    if raw.startswith('```'):
        raw = '\n'.join(raw.splitlines()[1:]).rsplit('```', 1)[0].strip()
    data = json.loads(raw)
    try:
        analyze.validate(data)
        valid = True
    except (AssertionError, KeyError, TypeError, ValueError):
        valid = False
    source = data['sources'][0]
    times = [event['time_s'] for event in source['onsets']]
    invalid_confidence = [event for event in source['onsets'] if not 0 <= event['confidence'] <= 1]
    rows.append(dict(label=label, run=str(run), passes_current_validator=valid,
                     invalid_onset_confidence=invalid_confidence,
                     description=source['description_zh'], count=len(times), onsets_s=times,
                     gaps_s=dict(collections.Counter(round(b-a, 4) for a,b in zip(times,times[1:])))))
report = dict(ground_truth_available=False,
              warning='Descriptive comparison only. Single sample per condition; no accuracy or causal claim. Invalid response is not repaired.',
              runs=rows)
out = HISTORY / 'runs' / 'prompt-comparison-20260929.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(out)
