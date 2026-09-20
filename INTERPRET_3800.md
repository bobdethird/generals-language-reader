# Interpretation of checkpoint 3800

The active objective is a separate language model that reports the frozen EMA
player's chosen move and gives tested evidence for why that choice depends on
the information it sees. Neither text nor reader gradients enter the actor.

## Existing work being reused

The ongoing iteration-2000 readout comparison, fresh audit, and conditional
iteration-3800 transfer are recorded in `runs/spatial-campaign-v1.json`. Do not
duplicate or restart live jobs. The reader uses a frozen Qwen3-0.6B backbone with
a trainable activation adapter. Its copied frozen policy head supplies a
continuous language prefix for accurate move reporting. Its original and
held-out-map accuracy must be checked on checkpoint 3800 before delivery.

## Added evidence for why

`reader/counterfactuals.py` defines ten probes with two strengths each: decrease
or increase troops at the move source, destination, nearest visible enemy to
our general, or our general; and remove recent friendly/enemy troop-change cues.
Current troop edits preserve geometry, ownership, and action legality and update
the redundant observation channels, current public total, and newest history
delta. Older history is held fixed. Hidden cells are never edited. History
removal is explicitly an input ablation, not a feasible alternative game.

The snapshot collector stores observations as float32, but the upstream rollout
passes bfloat16 observations into normalization. Probing must restore that input
dtype; simply reloading the float32 values changed logits and near-tied actions
in the initial pilot. Larger runs additionally compare explicit identity probes
and archived logits. If the original choice changes under a no-change replay,
the explanation is inconclusive. Claimed effects must exceed 0.15 plus four
times a conservative bound on the measured null drift. Exact agreement is
reported where observed, not assumed for every GPU execution.

Each probe reruns the actual frozen player. The evidence records the change in
the original action's margin over its strongest alternative, and whether greedy
play changes. Pass encodings use the maximum logit, matching deployed greedy
play. A claimed effect must have the same sign at both strengths and exceed
0.15 logit units at each. When no probe meets that rule, the reader must say
that the tests are inconclusive. These measurements establish local policy
sensitivity; they do not recover a long-term strategic intention or prove that
the move improves future outcomes.

`reader/why_language.py` trains a continuous-prefix adapter from detached actor
activations and the numerical probe effects to the frozen language decoder.
Probe names and rationale labels are training targets, not input text. At
inference it runs the probes again; its words are verified against the resulting
measurements. This is a post-hoc interpreter and adds inference cost.

## Completed results — September 20, 2026

The selected readers are `iter3800-decision-readout-v1` (step 3584) and
`iter3800-why-reader-v4` (step 1024). The latter's shared influence scorer was
selected using validation accuracy: 95.70%, versus 92.77% for the structured
comparison and 77.69% for the tokenwise baseline. The selected model's test
results were opened after that choice. All models reuse frozen Qwen3-0.6B,
revision `c1899de289a04d12100db370d81485cdf75e47ca`. Only adapters learn.
Neither reader trains or controls the game player. Supervision is generated
from the policy and controlled tests; no human annotation was required.

| Held-out check | Result |
| --- | --- |
| Exact move, including source, direction, and amount | 1,024/1,024 correct across 375 maps |
| All move-description sentences correct | 91.11% |
| Half-army moves | 29/29 correct; this subgroup is small |
| Copied action head versus original actor | 100% greedy agreement; maximum logit difference 0.0625 |
| Strongest-probe explanation, including effect and action change | 96.07% of 4,096 positions across 373 maps |
| Map-bootstrap 95% interval for explanation accuracy | 95.49–96.67% |
| Explanation accuracy where a consistent influence exists | 95.85% of 3,877 positions |
| Correctly inconclusive | 219/219 |
| Explanation accuracy with evidence shuffled across maps | 9.20% |
| Explanation accuracy with evidence removed | 0%; outputs failed strict parsing |

Independent verification found 99.24% of generated explanations made supported
claims; 96.66% named a strongest influence allowing physically identical tied
probes. These are **separate secondary metrics**. The original 96.07% metric
still requires the canonical strongest-probe choice, its sign, and switch claim.
A valid but weaker influence is not counted as a canonical success.

The independent observer was trained on board/history and semantic context.
With the explicit source/direction/amount sentence removed, generated context
raised moving-action accuracy from 51.39% to 60.47%. The gain was 9.08 percentage
points, with a map-bootstrap 95% interval of 6.31–11.97 points. This supports the
usefulness of the descriptive context; it does not validate a strategic motive.

Training/validation/test maps were checked disjoint. The move dataset contains
131,072/8,192/8,192 positions; its final reported test is a seeded subset of
1,024. The why dataset contains 16,384/2,048/4,096 positions. Across all 22,528
probe positions, 21,362 had a qualifying influence and none changed selected
action under the no-change numerical controls. Maximum archived-versus-replayed
logit drift was 0.234375. The numerical controls and conservative thresholds
remain necessary. Player and language-backbone hashes were unchanged.

The full repository suite passed **204 tests, with 8 environment-dependent tests
skipped**. Combined local CPU inference succeeded on four real test positions:
0 (wait/inconclusive), 71 (full move), 192 (full move), and 339 (half-army move).
Both readers' raw outputs and the fresh policy tests are saved in each JSON.
These examples are demonstrations, not a new population accuracy estimate.

## Limits and retained errors

The why reader explains measured local input sensitivity, not a recovered
thought process. It receives ten predefined probes, each at two strengths,
along with detached actor activations. It learns which effect to verbalize,
but the measurements are supplied by the probe pipeline. It is not a general
strategic narrator that reads hidden activations alone. Current-state edits
need not represent reachable game histories; history-cue ablations are explicitly
input tests. Neither proves that a move improves a future outcome.

The move reader directly reuses the frozen action head. Therefore its move
accuracy measures faithful verbalization, not an independent discovery of why
the policy acts. Extra destination/general-relative sentences still contain
errors, which the local verifier flags without substituting reference text.
Rare probe categories have weaker estimates: nearest-enemy increase scored
88% on only 50 positions. No claims of universal accuracy follow from these tests.

For example, held-out B200 position 192 generated the contradictory claim:
“increasing source troops strengthens the preferred action in both tests” and
“the stronger test changes the preferred action only in the first one.” The
strict verifier rejected it. Its raw output remains in `why-report.json` and
`audit.json`; it has not been repaired. Recomputing this same position on CPU
produced a correct explanation. Generation can differ with backend, batching,
and freshly measured effects, so local examples and cloud audit results retain
separate provenance.

The player SHA-256 is
`3ead46cee8d76e624179126bb7c3a8be54b00a24a75ae6f9e83b49e4978e3929`.
The selected what adapter is
`44a91ec163934740d16918fdd357feee1d62eecdefca0eac0f7a7a7e1e90ef2c`;
the why adapter is
`f15c1dd4ad12fb4a6ec8e673cf5e18a7a69d1672096be7f9d675b9f17e88aec2`.
A compact result record is committed at `reports/interpreter-3800.json`; full
weights, evidence, reports, and example images stay in ignored `runs/` and the
private Modal Volume `generals-reader-experiments`.

## Reproduce the completed interpreter

After the repository setup and Modal authentication, download the selected
adapters, player, reports, and one real held-out shard:

```bash
.venv-modal/bin/python scripts/download_interpreter.py
.venv/bin/python scripts/audit_interpreter.py
.venv/bin/python scripts/explain_3800.py \
  --player-checkpoint runs/interpreter-3800/player-3800.eqx \
  --what-reader runs/interpreter-3800/what.pt \
  --why-reader runs/interpreter-3800/why.pt \
  --evidence-shard runs/interpreter-3800/sample \
  --index 339 --device cpu --output runs/interpreter-3800/example-339.json
.venv/bin/python scripts/render_explanation.py \
  --result runs/interpreter-3800/example-339.json \
  --samples runs/interpreter-3800/sample/evidence.npz \
  --output runs/interpreter-3800/example-339.png
```

The local command recomputes the original policy and intervention effects before
asking the frozen language models to decode them. It preserves raw text, marks
unsupported outputs unverified, and does not insert reference text as a repair.
The image renderer adds a plain-English display layer: it shows the actual
selected move and simplifies verified explanation claims. Concrete before/after
troop counts replace logit-margin jargon. The unchanged reader outputs remain
underneath, with errors flagged. This display wording is not new model output
and does not change the reported reader accuracy.
CPU inference was verified end to end; training and population accuracy were
measured on B200. No training job needs to be restarted to use the interpreter.

The current reproducible input contract is an exported self-play observation:
bfloat16 before the player's normalization, stored losslessly as float32. The
existing human-play script supplies float32 before normalization and is not yet
wired to this interpreter. Do not present the exported-snapshot audit as proof
of accuracy for that other input convention.
