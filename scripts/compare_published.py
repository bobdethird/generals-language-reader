"""Validate and summarize full-recipe hardware benchmarks; never launches jobs."""
from pathlib import Path
import datetime as dt
import json
import math
import os
import statistics

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
RATES={'H100':.001097,'H200':.001261,'B200':.001736,'B300':.001972}
SHA='d86515a3a368c22fb8a2015d2f087cc003af5ce94fe2b7de717678ecd947d4e0'
records=[]
selected={}
for path in sorted((ROOT/'runs').glob('published-20260919-*/*gpu-bench.json')):
    record=json.loads(path.read_text())
    if record.get('batch_mode', 'fixed_global') != 'fixed_global':
        continue  # Different workloads belong in a separate comparison.
    records.append(dict(path=str(path), **record))
    if record['status']!='completed':
        continue
    assert record['config_sha256']==SHA
    assert (record['global_envs'],record['global_minibatch'],record['rollout_steps'])==(512,1024,512)
    assert record['progress']['iteration']==record['iterations']==24
    timing=path.with_name(path.stem+'-iteration-timing.jsonl')
    rows=[json.loads(line) for line in timing.read_text().splitlines()]
    assert [r['iteration'] for r in rows]==list(range(1,25))
    assert all(r['samples']==524288 and r['stage']==0 for r in rows)
    intervals=[b['timestamp']-a['timestamp'] for a,b in zip(rows[1:-1],rows[2:])]
    assert len(intervals)==22 and all(math.isfinite(v) and v>0 for v in intervals)
    wall=statistics.mean(intervals)
    assert abs(wall-record['benchmark']['mean_iteration_wall_seconds'])<1e-6
    row=dict(name=record['name'],gpu=record['gpu'],gpu_count=record['gpu_count'],
        started_at=record['started_at'],mean_iteration_seconds=wall,
        median_iteration_seconds=statistics.median(intervals),
        update_seconds=record['benchmark']['mean_update_seconds'],
        player_samples_per_second=524288/wall,
        gpu_usd_per_hour=3600*record['gpu_count']*RATES[record['gpu']],
        source=str(path),support_sha256=record['support_sha256'])
    key=(row['gpu'],row['gpu_count'])
    if key not in selected or row['started_at']>selected[key]['started_at']:
        selected[key]=row
order={'H100':0,'H200':1,'B200':2,'B300':3}
rows=sorted(selected.values(),key=lambda r:(order[r['gpu']],r['gpu_count']))
baseline=selected[('H100',1)]['mean_iteration_seconds']
for row in rows:
    row['speedup_vs_h100']=baseline/row['mean_iteration_seconds']
report=dict(observed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    recipe_sha256=SHA,parameters=15351761,global_envs=512,rollout_steps=512,
    global_minibatch=1024,player_samples_per_iteration=524288,adam_updates_per_iteration=128,
    pricing_date='2026-09-19',pricing_source='https://modal.com/pricing',
    method='24 iterations on each configuration; first two excluded. Mean intervals between completed updates include pool regeneration, EMA, logging and other loop overhead.',
    limitations=[
        'Early curriculum only; initialization, recurring evaluation and later-stage recompilation are not represented in these warm intervals.',
        'This measures training throughput, not final strength or Elo. Random trajectories and floating-point reductions differ across hardware.',
        'Instrumentation revisions fixed retained rollout memory and a JAX distributed top-k compiler crash; hashes and failed attempts are retained below.',
        'GPU hourly prices exclude CPU, memory, startup and actual billing adjustments.',
    ],successful_comparisons=rows,attempts=records)
out=ROOT/'runs/published-20260919/comparison.json'
out.write_text(json.dumps(report,indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
fig.set_facecolor('#101923');ax.set_facecolor('#101923')
colors={'H100':'#8997aa','H200':'#a6b4c6','B200':'#5ba9d9','B300':'#49ceb4'}
labels=[f"{r['gpu_count']} × {r['gpu']}" for r in rows]
values=[r['mean_iteration_seconds'] for r in rows]
ax.barh(labels,values,color=[colors[r['gpu']] for r in rows],height=.62)
ax.invert_yaxis();ax.set_xlim(0,max(values)*1.34)
for i,row in enumerate(rows):
    ax.text(values[i]+.18,i,f"{values[i]:.2f}s  ·  {row['speedup_vs_h100']:.2f}×",va='center',color='#edf4fa',fontsize=11)
ax.set_xlabel('Seconds per completed iteration — lower is faster',color='#c3ceda',labelpad=12)
ax.tick_params(colors='#c3ceda',labelsize=11)
for spine in ax.spines.values():spine.set_visible(False)
ax.set_axisbelow(True);ax.grid(axis='x',alpha=.12,color='white')
fig.suptitle('AverageJoe: earlier fixed-batch speed tests',color='#f3f7fb',fontsize=18,ha='left',x=.03)
ax.set_title('15.35M parameters · 512 games · 512 steps · global minibatch 1,024',color='#aab9c9',fontsize=10,loc='left',pad=16)
fig.supxlabel('24 iterations/run; first 2 excluded. Early curriculum; startup and periodic evaluations excluded.',fontsize=9,color='#8e9dae')
plots=ROOT/'runs/plots';plots.mkdir(exist_ok=True)
fig.savefig(plots/'published-gpu-comparison.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(plots/'published-gpu-comparison.svg',facecolor=fig.get_facecolor())
plt.close(fig)
print(json.dumps(rows,indent=2))
