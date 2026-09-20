"""Read small live campaign artifacts; never launches or changes cloud jobs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime as dt
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name('generals-policy-checkpoints')
output = ROOT / 'runs/published-20260919'
output.mkdir(exist_ok=True)
jobs = []
for manifest in sorted((ROOT / 'runs').glob('published-20260919-*/launches.json')):
    jobs.extend(json.loads(manifest.read_text())['jobs'])


def read(job):
    result = dict(job)
    for name in ('run.json', 'iteration-timing.jsonl'):
        try:
            content = b''.join(volume.read_file(f"{job['name']}/{name}"))
            (output / f"{job['name']}-{name}").write_bytes(content)
            if name == 'run.json':
                record = json.loads(content)
                result.update({key: record.get(key) for key in ('status', 'gpu', 'gpu_count', 'benchmark', 'error')})
            else:
                rows = [json.loads(line) for line in content.splitlines()]
                result['iterations_finished'] = len(rows)
                if rows:
                    result['last_iteration'] = rows[-1]
        except Exception as exc:
            result[name + '_unavailable'] = str(exc)[:160]
    if job.get('cancelled_at'):
        result.update(status='cancelled', cancelled_at=job['cancelled_at'], reason=job.get('reason'))
    return result


with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(read, jobs))
report = dict(observed_at=dt.datetime.now(dt.timezone.utc).isoformat(), runs=results)
(output / 'status.json').write_text(json.dumps(report, indent=2)+'\n')
for row in results:
    compact = {k: v for k, v in row.items() if not k.endswith('_unavailable')}
    if compact.get('last_iteration'):
        compact['last_iteration'] = {k: v for k, v in compact['last_iteration'].items() if k != 'gpu_memory'}
    print(json.dumps(compact))
