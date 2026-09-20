# Spatial language-reader experiment

The reader translates detached activations from the frozen iteration-2000 game
player into four automatically supervised facts: friendly troop location, own
general location, visible enemy troop location, and army balance. The language
model is frozen too; only the adapter learns.

## Spatial adapter

The player exports three global/history tokens and an 8×8 grid of board-patch
tokens, each representing a 3×3 group of cells. `reader/spatial.py` gives the
patches explicit row/column coordinates, local convolutions, and two attention
layers. A continuous residual adds spatial context to the original adapter's
language embeddings. Its output starts at zero, so both comparison branches
start with exactly the same language prefix.

Training-only heads learn the locations of the general and largest friendly and
visible enemy forces, plus army balance. Targets come directly from game
observations, including ties. These heads do not supply predicted labels to the
language decoder. At inference the reader needs only the player's activations.

The spatial adapter has 3,268,638 trainable parameters, including the original
379,008-parameter adapter. The game player still chooses its action independently
of the reader.

## Expanded data

`modal_spatial_collection.py` schedules four B200 collection workers. Dataset
`iter2000-spatial-data-v3` contains 131,072 training snapshots, 8,192 validation
snapshots, and 8,192 test snapshots. Each shard uses independent map seeds and
partitions the map pool between simultaneous games. Initial-map fingerprints are
checked for overlap between splits before the assembled dataset is published.

Collection uses final-curriculum maps with generals 17–28 tiles apart, both player
perspectives, and one sample every 32 rollout steps. Hidden activations retain
their original float32 precision. Policy/value outputs from the instrumented
player must agree with the original player. All shards are retained in the
`generals-reader-experiments` Modal volume.

## Matched comparison

`modal_spatial_training.py` waits for successful dataset assembly, then schedules
one B200 for each branch:

| Setting | Original adapter | Spatial adapter |
| --- | --- | --- |
| Initialization | Prior H100 reader's `best.pt` | Same adapter plus zero spatial residual |
| Training snapshots | 131,072 | Same snapshots and batch order |
| Training | 4 passes, batch 128, 4,096 updates | Same |
| Factual word loss weight | 5× | 5× |
| Spatial auxiliary loss | None | Weight 0.1 |
| Player and language backbone | Frozen | Frozen |

Each branch receives 524,288 snapshot presentations. Generated descriptions are
checked every 512 updates against the same 512 validation examples. Selection
uses the proportion of descriptions with all four facts correct, then individual
fact accuracy, then caption loss. A separate 1,024-example final test is opened
after selection. Evaluation favors different maps and groups uncertainty
estimates by map. Shuffled-activation controls check whether text depends on the
correct game state. Frozen-weight hashes are checked at the end.

This comparison measures the combined spatial architecture and auxiliary
supervision against the original adapter under the larger-data recipe. It does
not separately isolate each architectural addition. These scores measure factual
grounding, not game wins or whether a description explains the chosen move. The
description-to-action experiment comes after this grounding check.

## Run artifacts

The active campaign's app IDs and function-call IDs are recorded in
`runs/spatial-campaign-v1.json`. Both cloud coordinators run independently of the
local terminal. The training outputs are
`iter2000-spatial-comparison-v1-tokenwise` and
`iter2000-spatial-comparison-v1-spatial` in the Modal volume. Each stores metrics,
generated validation text, selected/latest adapter weights, configuration hashes,
and a final report.

Tests cover exact initial-prefix preservation, spatial cell ordering and ties,
map-pool separation, factual-token weighting, original activation precision, and
loss equivalence with a real small Qwen forward pass while keeping its weights
frozen.
