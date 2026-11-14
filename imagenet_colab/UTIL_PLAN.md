# ImageNet EP-KD (R34->R18, F1+F2) on Colab TPU v6e1: state, fixes, and next optimizations

Status 2026-11-12 11:20 PKT. Account balance almost spent; run continues on a top-up or a new account.

## Where the run is
- F1 (80 epochs, stages 112/140/168/196 px, 224 px anchors) at epoch 29, 140 px. Best val top-1 65.88% (epoch 23-27 range 64.3-65.9).
- Verified checkpoint: `backup_full/state_e27.tgz` (latest.pt epoch 27, finite, optimizer state). `backup_full/state.tgz` = newest
  per-epoch backup (driver). Precompute cache: `backup_full/cache_inc_001..029.tgz` (complete, verified).
- Resume on any account: `colab/drive_full.sh` (new VM -> push code -> restore cache + state -> download ImageNet -> resume).

## What limited speed, and what fixed it (all measured on the v6e1)
| Change | Effect |
|---|---|
| torch_tpu `batch_transfer` -> fresh-tensor int32 upload | correctness: batch_transfer corrupted inputs |
| ghost-BN pad rows -> 0 | correctness: NaN at every epoch end |
| tcmalloc preload | +20% |
| **huge-page (THP) host step buffers + int32 upload** | upload 0.33 -> 4.44 GB/s; 2,050 -> ~4,500 rows/s |
| foreach SGD | optimizer 8.9 -> 2.4 ms/step, bit-identical |
| 10 decode procs per loader worker (after the THP fix) | +~8% (~5,000 rows/s) |
| prefetch depth 2/3 | slower, keep 1 |
| decode lookahead 3 batches per worker (was 1) | neutral (~4,900 rows/s, slightly shallower dips); kept |
| loader prep: per-window edge clamping instead of full-batch `np.pad`; `valid_extent` scans inward on the unpadded image | prepare_microbatch 7.0 -> 3.2 ms (bit-identical, 150 random trials incl. black edges) |

TPU now: duty cycle ~80-84%, TensorCore ~26% (was ~50% / ~9%).
Device-only ceiling (batch already on the TPU, 140 px + 224 anchors): 85 ms/step = ~6,000 rows/s.

## Next optimizations, by expected gain
1. **Loader (was the limiter; trainer waited ~35% of each step; prep 2.2x faster since epoch 30, see table).**
   Worker profile: active 58%, of which prepare_microbatch ~61% (now halved), decode waits ~10%. Neither the faster prep
   nor 3-batch decode lookahead moved throughput (~4,900-5,000 rows/s), so the limiter is the main thread: it gathers
   16 micro-batches into the step buffer serially between steps (gather 12-15 s per 200 steps).
   **Tested 11-12 on the v6e1, REJECTED:** `--gather-thread 1` keeps the loss bit-identical (26/26 progress points of
   epoch 32 vs the old run) but is ~4-5% SLOWER (same VM, steady state: e33 thread on 4,336 rows/s median, gather 23 s/200
   steps; e34 thread off 4,510 rows/s, gather 13 s). Likely GIL contention with the main thread. Leave it off (default 0;
   /content/work/GATHER_THREAD=0). Other options if it is not enough: The main thread gathers 16 micro-batches from 6 worker
   rings into the step buffer (~50 ms/step of copies). Options: workers write straight into the THP step buffer
   (shared memory) so gather is a pointer swap; or a gather thread so it overlaps the step. Expected: up to ~6,000 rows/s.
2. **Device efficiency (TensorCore ~26%).** Many small eager ops per step (loss, ghost-BN bookkeeping). Compiling the
   loss + stats with the model (forward only; backward stays eager: torch_tpu drops gradients when backward runs in a
   compiled region) would cut dispatches. Validate with bench/tpu_update_probe.py (OK at fp32 and bf16) + loss trajectory.
3. Stage changes: 168/196 px views make each step bigger; per-step time grows mostly on the device (anchors stay 224).
   All later stages + F2 passed 200-step crash checks (colab/perf_bench.sh).

## Rough cost to finish (v6e1, 5.09 units/h, ~5,000 rows/s now)
| Part | Epochs | ~min/epoch | ~hours |
|---|---|---|---|
| F1 140 px | 30-40 | 13 | 2.5 |
| F1 168 px | 41-60 | 16 | 5.3 |
| F1 196 px | 61-80 | 19 | 6.3 |
| F2 224 px | 81-100 | 12 | 4.0 |
| VM restarts (download + restore) | | | ~1-2 |

About 19-20 h, about 100 units (later stages are estimates, ±25%). Loader work (item 1) would cut roughly 15-20%.

## Rules learned (keep)
- Always check the loss, not just rows/s, in any TPU test.
- Restart only at epoch boundaries, killing the session, pipeline and trainer **by PID**: a script whose command line
  contains a pgrep pattern kills itself.
- `colab sessions` lists server-side assignments; a timed-out `colab new` can leave a billing orphan (adopt it via its
  notebook hash, see scratchpad adopt_orphan.py).

## R50 -> MNv1 (pair E), 11-13
Device-only step at 112 px views + 224 px anchors: 137 ms (teacher views 5.5, teacher anchors 25.1, student views fwd+bwd 17.9,
student anchors fwd+bwd 102.0). The MobileNetV1 student on the 224 px anchors is the bottleneck, not the ResNet-50 teacher.
Tried, all slower: depthwise 3x3 as 9 shifted multiply-adds (epkd_colab/dwconv.py) 339 ms; channels_last 678 ms.
Training gets ~2,650 rows/s vs the 3,726 device ceiling (host gather + wait ~45 ms/step) -> gather thread A/B at epoch 3.
py-spy on the live trainer (11-14): the main thread spent 66.6% in Thread.join (the next batch's upload cannot finish before
the current step finishes on the TPU), then gathered the next batch with nothing queued on the device. Fix (train.py,
EPKD_COMPUTE_FIRST, default on): dispatch step N first, then gather + upload N+1. E1 epoch 5: ~2,480 -> ~2,950 rows/s (+19%).
Fast ghost BN (EPKD_GHOST_IMPL=fast, default on): -4% device step, same maths. Remaining gap to the 133 ms device step:
the upload of N+1 still waits for step N (~15 ms) plus dispatch.
