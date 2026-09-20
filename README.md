# Generals language reader

An experiment with two parallel outputs from a game policy: the move, and a
free-form text readout of its learned representations. Text is never an input
to the player. The language reader consumes exported, detached activations.

This repository contains the source code, configurations, tests, and pinned
dependency manifests. Generated runs, training data, model weights, downloaded
upstream checkouts, local environments, and credentials are excluded from Git.
Paths under `runs/` below refer to local or Modal artifacts, not files included
in this repository. Use the setup instructions to fetch the pinned dependencies.

## Current implementation

- Pinned Average Joe and generals-bots sources, with local self-play training.
- A small 688,337-parameter policy on 9×9 maps. This is a development baseline,
  not the paper's superhuman checkpoint or a reproduction of its results.
- A completed H100 pilot with a 1,128,145-parameter policy on 11–12 tile boards;
  its final EMA checkpoint won 113 of 128 fresh-map games against random play.
- The full 15,351,761-parameter published policy and distance curriculum through
  17–28, with a six-hour Modal campaign and measured multi-GPU scaling below.
- A local human-versus-checkpoint game using the upstream GUI, with fog of war,
  queued moves, half-army moves, and a paused start.
- An activation exporter that verifies its readout against the original policy
  and value heads. Independent map-generation/reset seeds for train/validation;
  both players and all frames of each game stay in the same split.
- A frozen Qwen3-0.6B language decoder with a trainable 279,936-parameter adapter.
  The adapter converts activation tokens into language-model prefix embeddings.
- A bounded warm-start experiment using observable-fact captions. The decoder
  generates vocabulary tokens; there is no classification head selecting labels.
- Validation with real, shuffled, and zeroed activations.
- A second experiment that trains the free-form reader using move-prediction
  rewards. A frozen, independent predictor sees the board/history plus generated
  words and reconstructs the player's move distribution. The player never sees
  the words. An additional predictor audits the result without supplying rewards.
- Local Apple GPU support via PyTorch MPS, verified with the actual language
  model's generation and backward pass. GPU access requires execution outside
  the desktop sandbox on this machine.

**Not yet implemented:** text-to-activation reconstruction, future-trajectory
rewards, a strong player, a replay UI, Stockfish support, or validated causal
explanations. Warm-start captions describe facts, not the causes of an action.

## First local results

Both initial experiments completed on the CPU of an Apple M5 Max with 128 GB RAM.
PyTorch reported MPS unavailable inside the sandbox. A later check with local
device access succeeded; the behavior experiment below uses the GPU. These
initial reader experiments did not use cloud resources.

- Self-play pilot: 200 iterations, 204,800 environment transitions / 409,600
  player samples. Last in-training evaluation: 6 wins, 16 losses, 42 draws against
  a random opponent (64 games, capped at 128 turns). The player is still weak.
- Activation dataset: 1,024 training and 512 validation snapshots. Original and
  readout policy/value outputs agree. The reader uses the pilot's EMA checkpoint.
- Reader warm-start: 100 updates of the adapter, with game and language weights
  frozen. On 64 held-out snapshots, mean caption token loss went from 2.463 to
  0.232 nats. After training, shuffled activations scored 0.253 and zero inputs
  1.970. The real-vs-shuffled gap is small; this is an initial signal, not an
  interpretability result. Generated captions still contain factual mistakes.
- Five targeted tests passed, including float32/bfloat16 readout agreement and
  stopping reader gradients. Installed dependencies passed `pip check`.

Detailed evidence is in `runs/local-pilot/training.log`,
`runs/activations-pilot/manifest.json`, and `runs/reader-warmup/report.json`.
The next experiment adds behavior rewards, while preserving this initial run.

## Local setup

Python 3.12 is used for the tested environment. Create an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python scripts/fetch_upstream.py
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.lock
.venv/bin/python -m pip install --cache-dir .cache/pip --no-deps -e vendor/generals-bots
.venv/bin/python scripts/download_reader.py
```

`fetch_upstream.py --git /absolute/path/to/git` selects a non-system Git if the
Apple tools are unavailable. Sources are pinned in `upstream-lock.json`; language
weights are pinned in `reader-model.json`. Downloads stay in the project cache.
The language model download is about 1.2 GB. Existing upstream checkouts are
verified and never silently reset.

On this Mac, the working bootstrap Python and Git came from Codex's bundled
runtime; the Apple command-line tools currently require Xcode license acceptance.
No license acceptance or system configuration was performed by this project.

## Play against iteration 2000

`scripts/play_checkpoint.py` opens the upstream human-game window locally on
CPU. It loads the frozen EMA model with the published architecture and uses the
final curriculum's 17–28 general distance, 0.21 mountain density, and 23×23 map
(padded to 24 for the network). Playing does not update the model or cloud run.

The weights are stored in the private Modal Volume, not GitHub. After the local
setup above and Modal authentication, download them once:

```sh
mkdir -p runs/play-2000
.venv-modal/bin/python -m modal volume get generals-policy-checkpoints \
  published-20260919-native4-dense100-eight-b200-8gpu-train/checkpoints/L_7d_gae90/L_7d_gae90_ema_2000.eqx \
  runs/play-2000/L_7d_gae90_ema_2000.eqx
.venv/bin/python scripts/play_checkpoint.py
```

The verified checkpoint SHA-256 is
`37198a4851baf222942395870de9b3e7e6f2d46dc60829af137bfb867f780ad5`.
The first game opens paused. You control red; press **P** to start or pause.
Click a tile and use **W/A/S/D** to queue moves. **Shift + W/A/S/D** sends half
the army. **E** undoes the last move, **Q** clears the queue, **Space** deselects,
and **[ / ]** changes speed. **V** reveals the map as a debugging cheat.

Use `--player 1` to play blue, `--seed` for a repeatable map sequence,
`--grid-size` for a trained size from 17 to 23, or `--checkpoint` for another
compatible network-only EMA file. Logs and the checkpoint hash are recorded
under `runs/play-2000/`. The window scales to the desktop and can be resized.

```sh
.venv/bin/python scripts/play_checkpoint.py --smoke-test --seed 2092026
```

This check loads the actual weights and runs eight simulator turns without a
window. It passed locally with finite weights and about 9 ms per warmed-up
decision; interactive human play was also confirmed.

## Run a local experiment

The wrapper selects the CPU explicitly, disables external experiment logging,
and records configuration, exit status, and console output. A fresh output name
is required, so existing runs cannot be overwritten accidentally.

```sh
.venv/bin/python scripts/run_local.py --name smoke-1
.venv/bin/python scripts/run_local.py --config configs/local-pilot.yaml --name pilot-1
.venv/bin/python scripts/collect_activations.py \
  --config configs/local-pilot.yaml \
  --checkpoint runs/pilot-1/checkpoints/local_pilot/local_pilot_ema_200.eqx \
  --output runs/activations-1
.venv/bin/python scripts/train_reader.py \
  --data runs/activations-1 --output runs/reader-1 \
  --steps 100 --batch-size 4 --eval-samples 64 --device cpu
.venv/bin/python -m pytest -q tests
```

PyTorch's MPS backend may be selected explicitly with `--device mps` when
available. The initial experiments used the local CPU; later experiments use
local MPS and Modal as described below. Announce a compute-location change before
launching.

## Train descriptions from behavior feedback

```sh
.venv/bin/python scripts/collect_activations.py \
  --config configs/local-pilot.yaml \
  --checkpoint runs/local-pilot/checkpoints/local_pilot/local_pilot_ema_200.eqx \
  --output runs/behavior-data-v1 --steps 512 --stride 8 --envs 16 \
  --seed 6044 --include-test
.venv/bin/python scripts/train_behavior_reader.py \
  --data runs/behavior-data-v1 --output runs/behavior-gpu-v1 --device mps \
  --warmup-steps 200 --predictor-steps 400 --rl-steps 40 \
  --bootstrap-texts 128 --eval-samples 96 --max-tokens 80
```

Use new output names when repeating an experiment. `--device cpu` is supported
explicitly; there is no silent GPU fallback. The simulator still runs on CPU.
All model weights are loaded from the existing cache; no external API is called.

The run has three stages:

1. Warm-start the reader's adapter on automatically generated visible facts,
   including coarse troop/general locations. These captions do not consult the
   selected move or policy logits. The language output is ordinary vocabulary
   generation, not a fixed strategy-label head.
2. Fit board-only and board-plus-text predictors on the frozen player's move
   probabilities. Text comes from automatic facts and the reader's own generated
   descriptions. Select predictor checkpoints on validation maps, then freeze
   them. An independently initialized auditor never provides a training reward.
3. Sample three descriptions for each of two training positions per update.
   Reward descriptions for reducing movement-prediction loss versus missing
   text. A leave-one-out policy-gradient baseline compares candidates for the
   same position. Retain a small visible-fact training loss and a penalty for
   drifting from the warm-start reader. Only the activation adapter changes.

The predictor receives visible board/history, a legal-action mask, and text. It
never receives the player's activations, logits, or sampled action as inputs.
Logits serve only as training/scoring targets. All 81 equivalent pass encodings
are combined into one action. Separate conditional-movement KL and top-choice
agreement prevent a pass-only predictor from appearing successful. Because the
player is stochastic, matching its probability distribution is a more suitable
training objective than guessing its random sampled action exactly.

Final evaluation uses 96 positions from independent test maps, with real,
missing, and other-game descriptions. Reported intervals resample whole games.
A matched factual-training-only branch receives the same extra grounding
examples without behavior rewards. This checks whether gains actually require
the new reward. The auditor has separate predictor weights but shares the frozen
English text encoder; it is not an independent judge of semantic truth.

Artifacts include `before_rl.pt`, `after_rl.pt`, `anchor_only.pt`, three frozen
predictors, configuration/code hashes, per-update metrics, sampled descriptions,
and `report.json` with before/after/control results and test examples. The script
checks that predictor parameters and the original player checkpoint stay frozen.

This is a small experiment on immediate move probabilities. A gain could reflect
predictor exploitation or better factual descriptions; it is not proof of a
faithful explanation. Future-move checks and causal interventions remain work
for a later stage.

### First behavior-feedback result

`runs/behavior-gpu-v1` completed on the Apple GPU in 244 seconds: 200 grounding
updates, 400 updates per predictor, 40 behavior-RL updates, and a matched 40-update
grounding-only control. Evaluation covered 96 test positions, including 90 with
legal troop moves, from 32 games contributing to movement metrics.

There was **no meaningful improvement from behavior RL**. On the frozen reward
predictor, conditional movement KL was 0.131055 for both real and shuffled text
(rounded to six decimals), versus 0.130096 with no text. Its before-to-after
improvement was -0.000000745 nats, with a game-bootstrap 95% interval spanning
zero. The independent auditor also showed no reliable before-to-after or
RL-over-grounding-only gain. The reader produced only seven unique greedy test
descriptions; 77 of 96 repeated the same upper-left description. These words
should not be treated as faithful explanations.

The GPU training machinery and frozen-weight checks worked, and all 11 tests
passed. The priority is now a competent game-playing checkpoint trained on
Modal, then interpretation of that frozen checkpoint. This pilot's lack of
useful explanations does not establish whether stronger play alone will solve
the reader's grounding problem.

## Published recipe and GPU campaign

**Current target: 30,000 total iterations**, including already completed updates,
or the extended **2026-09-20 07:00:00 UTC / 03:00 Eastern** save-and-stop deadline, whichever comes
first. The unchanged upstream YAML is retained for provenance; the runtime target
is 30,000 and resuming subtracts the checkpoint's completed iterations.

At iteration 946 the learner was checkpointed and resumed on the same eight-B200
configuration with **100-iteration checkpoint saves**, as requested for denser
evaluation. Production overrides `ckpt_every` and `save_every` to 100; the
published YAML remains unchanged. Weights, optimizer, EMA and curriculum resume
from that checkpoint; game environments start fresh. The target and original
deadline remained unchanged at that handoff.

The user subsequently extended the cutoff to **3 a.m. Eastern on September 20**.
The fixed process timer requires a safe restart: the dense-checkpoint run saved
iteration **2143**, including weights, Adam, EMA, and final curriculum stage 4.
The continuation uses the same eight-B200 workload and 100-iteration save cadence:

```sh
.venv-modal/bin/python -m modal run --detach modal_published.py \
  --gpus B200:8 --batch-mode native4 --iterations 30000 \
  --campaign published-20260920-3am-eight \
  --cache-from published-20260919-native4-dense100-eight-b200-8gpu-train \
  --resume-from published-20260919-native4-dense100-eight-b200-8gpu-train \
  --deadline 2026-09-20T03:00:00-04:00
```

Active learner: `published-20260920-3am-eight-b200-8gpu-train`
([Modal run](https://modal.com/apps/admin-23601/main/ap-zXvzitoUHwZXO72PeoaVVV)).
Fresh game environments start on resume. Both dashboards retain their original
W&B run IDs and ancestry; their replacement workers use the extended deadline.

**Earlier-run caveat identified September 19:** through iteration 240 the multi-GPU runner
preserved the *one-GPU* global batch, not the repository's native four-GPU batch.
Upstream `train/ppo.py` allocates `cfg.num_envs` on every device and uses
`cfg.minibatch_size` per device. With the unchanged YAML on four GPUs, that is
2,048 total games and a global minibatch of 4,096. Our earlier override used 512 total
games and a global minibatch of 1,024, with global top-advantage routing. This is
a deliberate hardware-scaling benchmark adaptation but was incorrectly described
as an exact reproduction of the reported four-GPU training setup.

The [paper](https://arxiv.org/html/2606.23348v1) reports four days on **four H200s**;
its single-H200 figures are simulator benchmarks. The YAML's 100,000-iteration
limit does not establish the final paper agent's training length. Its ladder
nickname includes `30k_ema`, suggesting a 30,000-iteration checkpoint, but that
name alone does not prove the exact training budget. Runtime projections from
our adapted run must not be presented as a matched paper-reproduction budget.

The user then requested native four-GPU training plus an eight-B200 comparison
that accelerates the **same** workload. `--batch-mode native4` now uses 2,048
total games, 512 rollout steps, 2,097,152 player samples per iteration, a global
minibatch of 4,096, and 128 Adam steps per iteration on either four or eight GPUs.
On four GPUs, upstream PPO, per-device top-k selection and loss reductions are
unmodified. On eight, each of the four logical replicas is split across a GPU
pair: top-k selection stays within the original logical replica and masked
gradients retain upstream's four-replica averaging. This preserves the training
workload and minibatch sampling distribution, not identical random trajectories
or floating-point results.

Twenty tests passed on eight virtual CPU devices, including selection across
GPU pairs and agreement of weights, Adam state and statistics with the original
four-device PPO update. A separate small eight-device end-to-end run completed
rollouts, PPO, EMA and checkpoint saving. Modal comparisons start from the same
iteration-240 checkpoint and curriculum stage 1. They retain the existing
deadline. Eight GPUs use 256 games and minibatch 512 per physical GPU; four use
the original 512 and 1,024. Use `scripts/compare_native_scaling.py` to compare
these runs; `compare_published.py` retains the older, smaller-workload results.
The earlier 240 updates remain part of the model's training history, so this is
a continuation with corrected settings rather than a reproduction from scratch.

The completed native-workload comparison measured median full iteration times of
**9.319 s on four B200s** and **5.755 s on eight**, a **1.619× steady speedup**.
Both performed 24 updates from the same iteration-240 learner, passed the random
opponent threshold, and saved iteration 264 at stage 2 (distance 6–13). The means
over the 22 measured intervals were 25.653 s and 21.595 s, respectively, because
both included a roughly six-minute evaluation/curriculum-compilation interval.
These short-run means should not be extrapolated as ordinary iteration speed.
Time from container start to the first completed update was 267 s and 660 s;
queue time is additional. Modal's workspace metrics showed a 10-GPU concurrent
limit, so the eight-GPU test waited while the four-GPU test was active. Full results and limitations are in
`runs/published-20260919/native-scaling.json`; the plot is
`runs/plots/native-four-vs-eight-b200.png`.

The selected eight-B200 production launch is
`published-20260919-native4-production-eight-b200-8gpu-train`
([Modal run](https://modal.com/apps/admin-23601/main/ap-oCQjU79lAgydtHdEwq1EcH)).
It resumes the eight-GPU benchmark's iteration-264 checkpoint with weights, Adam,
EMA and stage 2 preserved, reuses its full compilation cache, and targets
**29,736 additional updates**, reaching **30,000 total**. The original deadline
still takes precedence. At the measured steady speed alone, reaching the target
would take about 47.5 additional hours, excluding future pauses; this campaign's
remaining window cannot finish 30,000. Four targeted checks verified resume
accounting and that either stop condition saves all learner state before exit.
The production run was verified through iteration **273/30000** with finite
losses. Its first ordinary iterations took approximately six seconds including
loop overhead; actual timing can vary between allocations.

`configs/averagejoe-published.yaml` is a byte-for-byte copy of pinned upstream
`configs/custom/L_7d_gae90.yaml` (SHA-256
`d86515a3a368c22fb8a2015d2f087cc003af5ce94fe2b7de717678ecd947d4e0`).
It uses the 15,351,761-parameter model, 17–23 tile maps padded to 24, the original
200,000-map pool, and distance stages 2–6, 4–8, 6–13, 11–17, and 17–28. The
small-model checkpoints are shape-incompatible; these experiments start fresh.

The September 19 campaign originally had a six-hour deadline of
**2026-09-20 04:19:03 UTC / 00:19:03 Eastern**, including benchmark time; the user
extended it to **07:00 UTC / 03:00 Eastern** as documented above. A completed
update near the deadline saves current weights, Adam, EMA, curriculum stage and
iteration before exiting. A parent-process deadline also prevents a hung training
step from exceeding the window. It does not mark a timed-out run as completed.

The following command documents the earlier fixed-global-batch comparison:

```sh
.venv-modal/bin/python -m modal run --detach modal_published.py \
  --gpus H100:1,H200:1,B200:1,B300:1 --iterations 24 \
  --batch-mode fixed_global \
  --campaign published-20260919-single --deadline 2026-09-20T04:19:03+00:00
```

In that earlier mode, benchmark length is the only learning-configuration override
on one GPU. Multiple GPUs divide `num_envs` and `minibatch_size` by the device count, preserving 512
total games, 512 rollout steps, 524,288 player samples per iteration, 1,024 examples
per optimizer update, 128 Adam updates per iteration, and the original schedules.
Because upstream selects top advantages separately per device, the distributed
adapter instead selects the top 25% globally and routes selected examples with a
reduce-scatter. It also weights masked gradients by valid-example counts.
Random trajectories and floating-point reductions can still differ across GPUs;
this is equivalent training semantics, not a bit-exact replay.

The original reference checkpoint bank is not public. Its configured directory
is empty, so upstream skips that Elo benchmark. Production launches explicitly
set `ref_eval_every=0` to also avoid building its unused map pool during startup;
the four/eight-GPU benchmarks retain the original empty-bank setup. Self-play, random-opponent
evaluation and curriculum advancement remain active. Runtime compatibility fixes
also refresh JAX map-generator caches at curriculum transitions; otherwise the
simulator can silently reuse the previous distance settings.

`modal_published.py` checks actual GPU models, config hashes and finite losses,
and records effective overrides and per-iteration timings in the existing Modal
volume. Warm timing excludes the first two updates; intervals include evaluation,
pool resets and logging. Short benchmarks do not measure later curriculum stages
or establish policy strength. Full training uses the user's 30,000-iteration
total target subject to the explicit campaign deadline. Use a new campaign name
and an explicit future deadline for a later experiment.

The four-GPU comparison completed 24 iterations per configuration with finite
losses. Mean wall time over updates 3–24 was **4.685 s on 4× H100**, **3.494 s on
4× B200**, and **3.215 s on 4× B300**. One H100 took 12.332 s. These intervals
include ordinary loop overhead and a pool reset, but not startup or the regular
50-iteration evaluations. The full comparison and plot are in
`runs/published-20260919/comparison.json` and
`runs/plots/published-gpu-comparison.png`.

For the earlier smaller-batch run, the user selected **4× B200**. The preceding one-B300 main run checkpointed at
iteration **115**, and `published-20260919-production-b2004-b200-4gpu-train`
resumed its current weights, Adam state, EMA and curriculum stage. It subsequently
saved and stopped at iteration 240 to begin the native-four-replica comparison.
Games restart on new workers. The 04:19:03 UTC deadline remains unchanged.

The resumed run was verified through iteration 145 with finite losses. Its first
steady wall intervals were about **4.77 s**, slower than the 3.49 s benchmark on
a different allocation. One allocated B200 reported a persistent 1,155 MHz SM
clock while the others reported 1,965 MHz, without thermal or power throttling
flags. Restoring default clocks was denied by the container's GPU permissions.
Those updates remain valid; use live timings rather than assuming benchmark
throughput for a different node.

GPU-only list prices on September 19 were $15.7968/hour for four H100s,
$24.9984/hour for four B200s and $28.3968/hour for four B300s, plus CPU and memory
([Modal pricing](https://modal.com/pricing)). B200 was chosen for more training
within the fixed time window; H100 costs less per completed iteration.

## Live Weights & Biases dashboard

`modal_wandb.py` mirrors the existing learner's scalar logs on a separate
CPU-only Modal function. It does not restart training or load model weights.
The training volume is mounted read-only. Store a W&B key as `WANDB_API_KEY`
inside a Modal secret named `wandb-secret` in the same workspace/environment.
Do not put the key in source code or Git.

```sh
.venv-modal/bin/python -m modal run modal_wandb.py --check-only
.venv-modal/bin/python -m modal run --detach modal_wandb.py \
  --source published-20260920-3am-eight-b200-8gpu-train \
  --tracking-source published-20260919-native4-production-eight-b200-8gpu-train
```

The uploader uses the key's default W&B entity, or `--entity YOUR_TEAM`, and
creates `generals-language-reader` with private visibility. It refuses to send
metrics to an existing public project. The launch prints the dashboard URL.
Run only one uploader per source run. `--inspect-only` reads its server status.

History follows the selected checkpoint's ancestry, with each parent clipped
at the resumed iteration; sibling benchmarks and earlier small models are
excluded. Charts use the original **iteration** axis. Separate evaluation
records at the same iteration are preserved. Per-record IDs allow a replacement
uploader to deduplicate against W&B history when resuming its stable run ID.
After a training handoff, `--source` selects the resumed learner while
`--tracking-source` preserves the original W&B run and dashboard URL. Stop the
old uploader before starting its replacement so each dashboard has one writer.

The uploader checks for new logs every 30 seconds; Modal volume commits and
W&B ingestion can add delay. It stops after observing a terminal trainer state
and a final one-minute sync window, or ten minutes after the trainer's saved
deadline. This does not change the learner's deadline or iteration target.

Useful panels include `train/total_loss`, `train/policy_loss`,
`train/value_loss`, `eval/win_rate`, `curriculum/stage`, and
`performance/iterations_per_second`. Throughput uses consecutive learner
timestamps, including evaluation and compilation pauses. The run configuration
records hardware and batch transitions (the current lineage increases its global
batch at iteration 241). Evaluation win rate is against random play at the
current curriculum difficulty; self-play `train/win_rate` is not a strength rating.
Uploader host/system metrics and automatic code/environment metadata capture
are disabled. Only selected scalar logs and training configuration are uploaded.

```sh
.venv/bin/python -m pytest -q tests/test_wandb_sync.py
```

### Wins against older checkpoints

`modal_checkpoint_eval.py` watches the active run's numbered EMA checkpoints.
It retains iteration 500 as a fixed reference, then adds 1000, 1500, and each
subsequent 500-step checkpoint. **Candidates are evaluated every 100 iterations**
against all earlier references: 600, 700, 800, 900 and 1000 versus 500; 1100 versus
500 and 1000; 1600 versus 500, 1000 and 1500; and so on. The initial 500-versus-500
control checks paired-game accounting and is displayed separately from progress.
The first campaign saved only iteration 500 before the change, so 600–900 cannot
be reconstructed. Dense saving begins after the safe handoff at iteration 946.

```sh
.venv-modal/bin/python -m modal run --detach modal_checkpoint_eval.py \
  --source published-20260920-3am-eight-b200-8gpu-train \
  --tracking-source published-20260919-native4-production-eight-b200-8gpu-train \
  --entity bobdethird
```

The CPU watcher requests one **H100** only when a candidate is ready. The learner
continues on its existing GPUs; its volume is mounted read-only by the evaluator.
Numbered snapshots must be confirmed by the trainer's completed checkpoint marker
before evaluation. Mutable `published_latest` files are never used as references.
Start only one watcher for a given tracking source. A resumed watcher reuses
completed result files and deduplicates W&B events. It follows the selected
checkpoint ancestry to retain references from before a training handoff, with
each ancestor clipped at the iteration actually resumed.

Each real comparison uses **512 games / 256 paired maps**, with identical seeds
and both player positions. The fixed test environment uses 17–23 sized boards,
generals 17–28 tiles apart, the published terrain/city settings, greedy actions,
and a 2,048-turn limit. All candidates and opponents use their EMA weights.
These test seeds are separate from the training seed. The initial symmetric
control uses 128 games and must have equal win and loss counts.

The private W&B project contains a separate **EMA checkpoint comparisons** run
with `checkpoint/win_rate_vs_500`, `checkpoint/draw_rate_vs_500`,
`checkpoint/loss_rate_vs_500`, and equivalent series for each later reference.
`checkpoint/score_vs_500` counts a draw as half a point; this differs from the win
rate, which keeps draws in the denominator. A fixed set of older opponents helps
detect regressions that testing only against random play can miss. These results
are relative to our own checkpoint bank, not a human-player rating.

Results, seeds, per-batch counts, checkpoint hashes, and logs persist in the
`generals-checkpoint-evaluations` Modal Volume under the source run and protocol
hash. Model weights are not uploaded to W&B. The watcher launches no new work in
the last two minutes of the saved campaign deadline, and an active H100 subprocess
is terminated before that deadline. Any unfinished comparisons remain listed in
the persisted watcher state for a later explicitly resumed campaign.

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/test_checkpoint_evaluation.py tests/test_upstream_training.py \
  -k 'checkpoint or paired_matches or match_executor'
```

## Earlier small-model Modal runs

The small-map job below was stopped and superseded by the published-recipe
campaign. Its checkpoints through cumulative iteration 4,000 remain on the Modal
volume. This section documents the earlier run, not the current training setup.

The next stage uses AverageJoe's existing curriculum and reference-evaluation
hooks. Launch it explicitly with the `curriculum` preset:

```sh
.venv-modal/bin/python -m modal run --detach modal_policy.py \
  --preset curriculum --gpu H100 --name modal-h100-curriculum-full-v1
```

This resumes cumulative iteration 1,500 (including Adam and EMA) for 98,500
additional iterations, reaching the repository's 100,000-iteration total target.
The user's stop condition is reaching that target: there is no application
elapsed-time watchdog (`seconds=0`, the default). Modal requires a per-call timeout;
it is set to the platform maximum of 24 hours. An infrastructure timeout/failure
would require resuming saved checkpoints; it is not considered training completion.
Checkpoints are saved every 500 iterations, following the repository's cadence.
It retains the existing 1,128,145-parameter model, 11–12-cell
maps and PPO settings. General-distance stages are 2–6, 4–8, 6–10 and 8–12; the
upstream curriculum advances when random-opponent evaluation reaches 60% wins.
Evaluation uses 128 games every 50 iterations. This is the upstream method adapted
to our pilot, not the original full-size seven-layer agent or training budget.

Frozen EMA checkpoints at cumulative iterations 300 and 1,500 form the reference
set. Every 250 iterations (plus the initial baseline), both the current model and
EMA play each reference on 64 maps with player positions swapped: 128 games per
opponent. Reference tests always use 12×12 maps, general distances 4–10, 512-step
games, and the simulator's default mountain density of 0.18–0.26. The reference
models also play each other once to establish the matrix used by upstream's
relative Elo calculation. These are internal ratings, not generals.io ladder Elo.

`reader/upstream_training.py` installs opt-in compatibility changes for this preset:

- Recompile map generation and initial-state sampling when curriculum attributes
  change. The pinned simulator otherwise caches the old distances because its
  environment object is a static JAX argument.
- Reuse the upstream compiled match loop across checkpoint weights. Reference
  evaluation swaps player positions on identical maps; upstream tests candidates
  only as player zero.
- Save scalar metrics to `metrics.jsonl` and reference matrices/results to
  `reference-evaluations/`. Self-play, rewards, the network and PPO remain upstream.

The runner validates source checkpoint/config hashes before training, records
curriculum transitions and the runtime support hash, and evaluates the final EMA
on the final attained curriculum stage. Resume configurations must preserve that
stage when continuing a later run; upstream checkpoints omit curriculum state,
environment state and RNG state. Seven targeted tests cover paired outcomes,
reference matrix reuse, actual map distances after a stage change, match executor
reuse, optional process time limits, and absence of a watchdog for iteration-based runs.

Hardware and rollout choice: **one H100 with the original `pilot` batch settings**
(256 parallel games, 256-turn rollouts, optimizer minibatch 1,024). The launcher
defaults to the `curriculum` continuation above; use `--preset pilot` to start a
fresh short experiment. The H100 request is exact, with no automatic upgrade. The
original comparison baseline is
`runs/modal-h100-pilot-v1/checkpoints/modal_pilot/modal_pilot_ema_300.eqx`,
which scored 113W/3L/12D in the held-out random-opponent evaluation. Scaling
variants remain available as explicit experiments. The `continue` preset restores
the baseline's training weights, Adam optimizer and EMA after verifying their
checksums. It runs up to 1,200 additional iterations with schedule offset 300 and
a new rollout seed. Environments and RNG state restart; checkpoint filenames and
logs count additional iterations from one. The `pilot` preset starts fresh.

```sh
.venv-modal/bin/python -m modal run --detach modal_policy.py \
  --preset continue --gpu H100 --name modal-h100-continue-v1 --seconds 900
```

Detached training survives client disconnection. Checkpoints persist in the Modal
Volume; the client also downloads artifacts if it stays connected through completion.

The continuation completed all 1,200 additional iterations (1,500 cumulative).
Its final EMA checkpoint scored 127W/0L/1D on the same 128-game held-out
random-opponent evaluation, compared with 113W/3L/12D for the pilot.
Plot the recorded losses and evaluations locally, without starting another GPU run:

```sh
.venv/bin/python scripts/plot_training.py \
  runs/modal-h100-pilot-v1 runs/modal-h100-continue-v1 \
  --output runs/plots/h100-loss-curves
```

This writes PNG/SVG charts, the raw metrics as CSV, and evaluation provenance as
JSON. The plots use cumulative iterations and a trailing 25-iteration mean that
restarts at the resume boundary. Self-play loss is not a direct measure of playing
strength; random-opponent results do not establish strength against skilled players.

The cloud client uses a separate environment so installing it does not change
the local PyTorch/JAX environment:

```sh
.venv/bin/python -m venv .venv-modal
.venv-modal/bin/python -m pip install -r requirements-modal.txt
.venv-modal/bin/python -m modal token new
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset smoke --name modal-h100-smoke-v1 --seconds 300
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset pilot --name modal-h100-pilot-v1 --seconds 900
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset pilot --gpu H200 --name modal-h200-pilot-v1 --seconds 900
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset pilot --gpu B200 --name modal-b200-pilot-v1 --seconds 900
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset pilot --gpu B300 --name modal-b300-pilot-v1 --seconds 900
```

Run the pilot after the GPU smoke test succeeds. Announce hardware and estimated
cost before launching. The user's minimum is H100; `--gpu` selects one
H100, H200, B200, or B300 (default H100), or `H100:2` for two H100s with the
`dual` preset. H100 requests use `H100!` to prevent
automatic upgrades during hardware comparisons. Jobs cap CPU at two physical cores
and host memory at 16 GiB, and has no configured retries or warm containers.
An application wall-clock watchdog is enabled only when an explicit positive
`--seconds` is supplied; normal training stops at its iteration target. Modal has
a separate infrastructure timeout. At September 19, 2026 listed prices, twenty minutes at
these maximum compute resources is approximately $1.39, excluding image builds,
startup, storage, and any platform-level recovery. Runtime/resource limits are
not an account-wide dollar spending limit; account budgets live in Modal.

`configs/modal-pilot.yaml` starts a fresh, larger policy on 11–12 tile boards:
three transformer layers, 128-wide activations, 256 parallel games, 256-turn
rollouts, and up to 300 iterations (39,321,600 player samples if completed).
It evaluates against random play every 50
iterations and saves optimizer/EMA checkpoints every 25. This first cloud run
measures throughput and learning; it is not expected to reproduce superhuman
Average Joe. Consistent play against random and scripted opponents, and separate
held-out evaluation, should precede interpreting a frozen checkpoint.
If at least 60 seconds remain in the same runtime allowance after training,
the latest EMA checkpoint is evaluated on 128 paired games with a separate seed.

The image pins CUDA-enabled JAX and fetches the two upstream repos by SHA.
Only project source/configuration files are uploaded, not local caches or runs.
The job fails if JAX cannot find the GPU. Logs, resolved configuration, package
versions, and checkpoints persist in the `generals-policy-checkpoints` Modal
Volume and are downloaded into `runs/<name>` when the call returns. A timeout
preserves earlier checkpoints; an interrupted checkpoint write may be incomplete
and must be load-validated before reuse. There is no automatic resume loop.
The wrapper verifies the actual GPU name and uses a fresh temporary JAX
compilation cache per run. B300 requires CUDA 13.1 or newer; the tested image's
resolved CUDA packages are 13.4. Each GPU uses the same pilot configuration.

Useful recovery command if the local client disconnects:

```sh
.venv-modal/bin/python -m modal volume get generals-policy-checkpoints \
  modal-h100-pilot-v1 runs/modal-h100-pilot-recovered
```

The test suite includes deadline enforcement for a silent subprocess
and preservation of successful process logs. Cloud dependencies are recorded
separately in `requirements-cloud.txt`; resolved transitive versions are saved
in each cloud run's `run.json`.

### Completed H100 pilot

On September 19, 2026, `runs/modal-h100-pilot-v1` completed all 300 iterations on
an NVIDIA H100 80GB: 39,321,600 player samples in 334 seconds including training
compilation. Training and the separate evaluation took 395 seconds overall.
The final EMA checkpoint scored **113 wins, 3 losses, and 12 draws** over 128
fresh-map games (64 maps played from both seats), an 88.3% win rate against the
random legal-action opponent. Maps were 11–12 tiles across and games were capped
at 512 turns. This does not establish strength against humans or competent bots.

All artifacts were downloaded from Modal. The checkpoint
`runs/modal-h100-pilot-v1/checkpoints/modal_pilot/modal_pilot_ema_300.eqx` was
loaded locally, verified finite, and matched the evaluation's SHA-256:
`29c377bb4c5f03a4ecd07d314e30ac2411a53572cee94cf18826ad2385fa4e4c`.
See `heldout-evaluation.json` and `run.json` in that run directory for evidence.

Once matched pilots have finished, compare their saved artifacts without
launching additional jobs:

```sh
.venv/bin/python scripts/compare_gpu_runs.py \
  runs/modal-h100-pilot-v1 runs/modal-h200-pilot-v1 \
  runs/modal-b200-pilot-v1 runs/modal-b300-pilot-v1 \
  --output runs/gpu-comparison-v1.json
```

The comparison verifies configuration and source revisions and checkpoint
hashes, then reports warm throughput, total runtime, evaluation outcomes, and
cost estimates using dated Modal prices. Warm throughput uses iterations 11–300
including periodic map-pool resets, and excludes evaluation/checkpoint writes.
Total runtime includes compilation and evaluation but excludes container startup,
image builds, volume commit, and downloads. Cost estimates are not Modal bills.
The earlier H100 pilot used a shared cache; warm throughput is the more useful
hardware comparison. One seed per GPU and a small model do not establish either
statistical performance rankings or speedups for full-size Average Joe training.

The September 19 comparison completed all 300 iterations and the 128-game
evaluation on every GPU. Warm throughput was 367,851 player samples/second on
H100, 379,568 on H200 (+3.2%), 412,699 on B200 (+12.2%), and 396,419 on B300
(+7.8%). B200 was fastest in this run; H100 provided the most warm training
samples per GPU dollar at the listed rates. All final EMA checkpoints loaded
locally with finite weights and hashes matching their cloud evaluations.

Final EMA evaluations scored 113W/3L/12D (H100), 103W/3L/22D (H200),
94W/4L/30D (B200), and 112W/3L/13D (B300). These are random-opponent results
from one training seed per device, not evidence that a particular GPU learns a
better policy. The three new runs' measured function runtime corresponds to
about $2.00 of GPU compute, or $2.07 including the configured maximum CPU/memory
rates; startup, image builds, storage, and other unmeasured overhead are extra.
The complete measurements and limitations are in `runs/gpu-comparison-v1.json`.

### Parallel games, batches, and two GPUs

The baseline already vectorizes 256 games on one GPU. The `wide` preset uses
1,024 games on one H100 while preserving the 1,024-example optimizer minibatch.
`wide-batch` also increases that minibatch to 4,096. The `dual` preset distributes
1,024 games across two H100s (512 each), with a 2,048-example minibatch per GPU
and thus a 4,096-example global batch. Upstream `pmap` averages gradients across
devices on each optimizer update; both GPUs train one shared policy.

```sh
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset wide --gpu H100 --name modal-h100-wide-v1 --seconds 720
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset wide-batch --gpu H100 --name modal-h100-wide-batch-v1 --seconds 720
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset dual --gpu H100:2 --name modal-h100-dual-v1 --seconds 720
.venv-modal/bin/python -m modal run modal_policy.py \
  --preset wide-short --gpu H100 --name modal-h100-wide-short-v1 --seconds 720
.venv/bin/python scripts/compare_gpu_runs.py --scaling \
  runs/modal-h100-pilot-v1 runs/modal-h100-wide-v1 \
  runs/modal-h100-wide-batch-v1 runs/modal-h100-dual-v1 runs/modal-h100-wide-short-v1 \
  --output runs/scaling-comparison-v1.json
```

The first three scaling runs use the same model, game distribution, rollout length, and
39,321,600 total player samples, collected in 75 rather than 300 rollout rounds.
EMA decay is rescaled to `0.99^4` to preserve its sample-based averaging horizon.
This is a throughput experiment, not mathematically identical training: larger
minibatches reduce optimizer updates; the policy refreshes less often per
sample; entropy scheduling is still based on rollout rounds; evaluation and
checkpoint cadence differ. Top-advantage filtering happens separately on each
GPU in the dual run. Keep held-out evaluations with speed measurements before
choosing a configuration for a longer learning run.

The initial results exposed this tradeoff. `wide` reached 483,493 player
samples/second (1.31× baseline), `wide-batch` 542,827 (1.48×), and `dual`
875,738 (2.38×). Their final EMA checkpoints won 68, 60, and 69 games out of
128 respectively, compared with 113 for the baseline. This does not isolate
which setting caused the learning regression, but it rules out treating higher
throughput as demonstrated faster learning in this experiment.

The `wide-short` refinement keeps 1,024 parallel games but collects only 64
turns per game before updating. Its total collection batch, optimizer batch,
19,200 optimizer updates, 300 rollout rounds, EMA decay, and schedules match
the baseline. Its shorter GAE horizon and broader game sample diversity still
change training. The completed refinement reached 425,795 player samples/second
(1.16× baseline), but scored 90W/6L/32D against random play. The original H100
checkpoint retains the highest held-out score among these runs; none of these
throughput gains has yet demonstrated faster improvement in playing strength.

The two-GPU run was 1.61× faster than the comparable single-GPU `wide-batch`
configuration, or 2.38× faster than the original small-batch baseline. The latter
gain combines batching and adding a GPU, rather than measuring GPU scaling alone.
All four new checkpoints loaded locally with finite weights and matching hashes;
`local-validation.json` records each check. All jobs completed and stopped.
Their measured runtime corresponds to approximately $1.88 of compute including
maximum configured host resources, excluding startup and storage. Complete
measurements and caveats are saved in `runs/scaling-comparison-v1.json`.

The wrapper checks both actual GPU model and device count. A targeted test
verifies that gradient averaging keeps replicas synchronized even when they see
different examples. Run it locally on two virtual CPU devices:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .venv/bin/python -m pytest -q tests/test_readout.py -k 'replicated or data_parallel'
```

### Upstream pretrained weights

A September 19, 2026 check found no publicly downloadable Average Joe checkpoint
in the upstream repository tree, its releases/tags, or the default trees and
releases of its four public forks. The author's Hugging Face account
`strakammm` listed replay data but no public models. The paper links training
code, and the README describes generating checkpoints locally; neither provided
a weight download. This search does not establish that no other copy exists.

The preferred next input is the author's deployed EMA checkpoint corresponding
to `L_7d_gae90_30k_ema`, together with its exact YAML configuration and compatible
source revision. No author contact or further cloud training was initiated
during this search.

## Artifacts and what they mean

- `runs/<policy>/training.log`: evaluation and PPO progress. `SPS` counts both
  players' samples, so it is twice the number of environment transitions/second.
- `runs/<policy>/checkpoints/...`: exact resolved config and weights. Use an EMA
  network-only checkpoint for activation export; `_final.eqx` also stores the
  optimizer and is intended for policy training resumption.
- `runs/<dataset>/manifest.json`: checkpoint hash, upstream revisions, split
  seeds, counts, and readout agreement. `.npz` files contain no privileged full
  simulator state, only the player's inputs and readouts.
- `runs/<reader>/adapter.pt`: only adapter weights and provenance. The game
  policy and language model remain frozen.
- `runs/<reader>/report.json`: held-out token losses and generated examples.
  Lower loss with real rather than shuffled activations is evidence of useful
  activation information, not proof that the words identify decision causes.

The first caption targets concern army balance, visible enemies, discovery of
the enemy general, and territory balance. They teach the initial connection to
English. They deliberately avoid guessed intent or privileged fog-of-war facts.
The behavior experiment adds coarse spatial facts for English grounding and
then rewards free-form text for helping reconstruct move probabilities. Further
work includes activation reconstruction and testing whether specific claims
predict responses to controlled interventions.

## Compatibility and reproducibility

The released source is run unchanged. `reader/runtime.py` provides two narrow
compatibility shims: JAX's documented replacement for `device_put_replicated`,
and the simulator's renamed `num_cities_range`/`num_castles_range` property.
Tests cover their behavior and matching readouts in float32 and bfloat16.

Training follows the pinned code, including its expansion-prior regularizer and
absolute-advantage filtering. Those details differ from the paper's prose; this
project does not claim an exact replication of the paper.

Dependencies, models, runs, and third-party sources are ignored by Git. The
fetch script and lock files recreate them. Source is backed up in the private
[GitHub repository](https://github.com/bobdethird/generals-language-reader).

## References

- [Average Joe](https://github.com/strakam/AverageJoe)
- [Generals simulator](https://github.com/strakam/generals-bots)
- [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Self-CTRL](https://arxiv.org/abs/2606.18327) — related behavior-predictive
  explanation training; adapting that idea to game-policy activations is this
  project's experiment, not a reproduction of the paper.
