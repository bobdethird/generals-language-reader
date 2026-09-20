"""Compare four/eight B200s on the SAME native four-replica workload."""
from pathlib import Path
import datetime as dt
import json
import math
import os
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'.cache'))
selected = {}
provenance = []
for path in sorted((ROOT/'runs').glob('published-20260919-native4-*/*gpu-bench.json')):
    record = json.loads(path.read_text())
    if record.get('status') != 'completed':
        continue
    assert record['batch_mode'] == 'native4' and record['gpu'] == 'B200'
    assert record['gpu_count'] in (4, 8)
    assert (record['global_envs'], record['global_minibatch'], record['rollout_steps']) == (2048, 4096, 512)
    assert record['iterations'] == 24 and record['adam_updates_per_iteration'] == 128
    start = record['resume']['iteration']
    assert record['progress']['iteration'] == start+24
    timing = path.with_name(path.stem+'-iteration-timing.jsonl')
    rows = [json.loads(line) for line in timing.read_text().splitlines()]
    assert [r['iteration'] for r in rows] == list(range(start+1, start+25))
    assert all(r['samples'] == 2097152 for r in rows)
    intervals = [(b, b['timestamp']-a['timestamp']) for a, b in zip(rows[1:-1], rows[2:])]
    assert len(intervals) == 22 and all(math.isfinite(t) and t > 0 for _, t in intervals)
    wall = statistics.mean(t for _, t in intervals)
    median = statistics.median(t for _, t in intervals)
    assert abs(wall-record['benchmark']['mean_iteration_wall_seconds']) < 1e-6
    row = dict(name=record['name'], gpu_count=record['gpu_count'],
               start_iteration=start, end_iteration=start+24,
               mean_wall_seconds=wall, median_wall_seconds=median,
               mean_compute_seconds=record['benchmark']['mean_update_seconds'],
               rollout_seconds=record['benchmark']['mean_rollout_seconds'],
               ppo_seconds=record['benchmark']['mean_ppo_seconds'],
               samples_per_second=2097152/wall,
               iterations_per_hour=3600/wall,
               steady_iterations_per_hour=3600/median,
               steady_hours_to_target=(30000-(start+24))*median/3600,
               time_to_first_update_seconds=rows[0]['timestamp']-dt.datetime.fromisoformat(record['started_at']).timestamp(),
               total_benchmark_seconds=record['wall_seconds'],
               gpu_usd_per_hour=record['gpu_count']*.001736*3600,
               stages=sorted({r['stage'] for r in rows}),
               source=str(path), support_sha256=record['support_sha256'],
               checkpoint_sha256=record['resume']['sha256'],
               long_intervals=[dict(iteration=r['iteration'], stage=r['stage'], seconds=t)
                               for r, t in intervals if t > median*2])
    selected[record['gpu_count']] = row
    provenance.append(record)
rows = [selected[k] for k in sorted(selected)]
if 4 in selected and 8 in selected:
    four, eight = selected[4], selected[8]
    assert four['checkpoint_sha256'] == eight['checkpoint_sha256']
    assert four['support_sha256'] == eight['support_sha256']
    eight['speedup_mean'] = four['mean_wall_seconds']/eight['mean_wall_seconds']
    eight['speedup_median'] = four['median_wall_seconds']/eight['median_wall_seconds']
    eight['gpu_cost_per_iteration_ratio'] = 2/eight['speedup_mean']
    eight['steady_gpu_cost_per_iteration_ratio'] = 2/eight['speedup_median']
report = dict(observed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    global_envs=2048, global_minibatch=4096, player_samples_per_iteration=2097152,
    adam_updates_per_iteration=128, iteration_target=30000,
    deadline='2026-09-20T04:19:03+00:00',
    method='24 updates from the same saved learner; first two updates excluded from timing. All wall intervals retained; median also shown to expose evaluation/compilation pauses.',
    semantics='Four GPUs use upstream PPO unchanged. Eight GPUs split each of four logical replicas across a GPU pair; top-k and masked gradient averaging retain four-replica semantics. Random trajectories and floating-point reductions are not bit-identical.',
    pricing_source='https://modal.com/pricing', pricing_date='2026-09-19',
    limitations=['One short run per hardware allocation; this establishes throughput, not policy strength.',
                 'Hardware clock differences and stage transitions can affect these timings.',
                 'Target-time estimates use steady median intervals and exclude future evaluations, compilation, checkpointing and allocation delays.',
                 'GPU prices exclude CPU, RAM and startup.'], results=rows, records=provenance)
(ROOT/'runs/published-20260919/native-scaling.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(rows, indent=2))
if len(rows) != 2:
    raise SystemExit(0)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.2), layout='constrained')
fig.set_facecolor('#101923'); ax.set_facecolor('#101923')
labels = [f"{r['gpu_count']} × B200" for r in rows]
x = list(range(len(rows)))
width = .3
ax.barh([v-width/2 for v in x], [r['median_wall_seconds'] for r in rows], height=width,
        label='Median iteration', color='#49ceb4')
ax.barh([v+width/2 for v in x], [r['mean_wall_seconds'] for r in rows], height=width,
        label='Mean incl. stage-change pause', color='#5ba9d9')
for i, r in enumerate(rows):
    for offset, key in [(-width/2, 'median_wall_seconds'), (width/2, 'mean_wall_seconds')]:
        ax.text(r[key]+.12, i+offset, f"{r[key]:.2f}s", va='center', color='#edf4fa')
ax.set_yticks(x, labels); ax.invert_yaxis()
ax.set_xlim(0, max(r['mean_wall_seconds'] for r in rows)*1.2)
ax.set_xlabel('Seconds per iteration — lower is faster', color='#c3ceda')
ax.tick_params(colors='#c3ceda')
for spine in ax.spines.values(): spine.set_visible(False)
ax.legend(facecolor='#101923', labelcolor='#c3ceda', frameon=False)
fig.suptitle('Same training workload: four vs eight B200s', color='#f3f7fb', fontsize=16)
ax.set_title('2,048 games · 512 steps · global minibatch 4,096', color='#aab9c9', fontsize=10)
fig.supxlabel('24 updates from the same checkpoint; first two excluded. Both tests include one curriculum recompilation.\nMedian reflects ordinary updates. GPU count changes; batch size stays fixed.',
              fontsize=9, color='#8e9dae')
output = ROOT/'runs/plots/native-four-vs-eight-b200.png'
fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
