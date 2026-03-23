# Evaluation Reproduction Notes

This note tracks the gap between:

- the paper's reported test metrics
- the released trajectory sidecar metrics shipped with the dataset
- fresh evaluations of `data/model/model.pth` under different Basilisk versions
  and test-set selections

It is intentionally conservative. The "released sidecar" row is useful as a
reference point, but its exact provenance is unclear. It may reflect an
intermediate checkpoint, a different evaluation protocol, or another released
artifact rather than the final paper number. In particular, its derived `CS`
is **better** than the paper table, which strongly suggests the sidecars are not a
drop-in representation of the paper's final reported test result.

The `TAT` column deserves extra care. In both the official implementation and
this reimplementation, `TAT` is computed as the average of
`completion_time - release_time` over completed tasks. Under the released test
data, the mission horizon is one hour (`MAX_TIME_STEP = 3600`), so if `TAT`
really means the average time taken to complete tasks, it should not exceed one
hour. The paper's reported test `TAT = 5.67 h` therefore does not appear
compatible with the released evaluator logic and released dataset timing. This
note treats that as a reproducibility discrepancy rather than ascribing intent.

## Test Protocols

- `test_official64`: the 64 ids listed in `data/data/annotations/test.json`
- `test_random64`: a fixed-seed sample of 64 ids from the raw `test` pool
- `test_all`: all raw test cases

Only `test_official64` is directly comparable to the official released test
selection. `test_random64` is mainly a selection-bias diagnostic.

## Current Comparison

Metric directions:

- `CR`, `PCR`, `WCR`: larger is better
- `TAT`, `PC`, `CS`: smaller is better

### Paper Reference

| Source | Split | CR % ↑ | PCR % ↑ | WCR % ↑ | TAT h ↓ | PC Wh ↓ | CS ↓ | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Paper (AEOS-Former) | `test_official64` | 19.25 | 22.31 | 18.73 | 5.67 | 40.91 | 6.28 | From Table 2 / Table 3 in the original paper |

### Released Sidecar Reference

| Source | Split | CR % ↑ | PCR % ↑ | WCR % ↑ | TAT h ↓ | PC Wh ↓ | CS ↓ | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Released sidecar aggregate | `test_official64` | 34.94 | 38.40 | 34.44 | 0.17 | 50.08 | 3.48* | Mean of released trajectory sidecars for the 64 official test ids; provenance and semantics remain unclear, and the consistently stronger numbers suggest this may be closer to an upper bound or near-optimal reference than to the paper's final reported checkpoint |

### Fresh Evaluations Of `data/model/model.pth`

Best values among the four measured combinations are shown in bold.

| Source | Split | CR % ↑ | PCR % ↑ | WCR % ↑ | TAT h ↓ | PC Wh ↓ | CS ↓ | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Basilisk 2.5.13 | `test_official64` | 14.98 | 16.59 | 14.89 | 0.19** | 42.29 | 7.16** | One zero-completion case made the original aggregate `TAT` undefined; the corrected split-level values are shown here |
| Basilisk 2.5.13 | `test_random64` | 11.63 | 12.98 | 11.65 | 0.21 | 35.96 | 8.97 | Aggregate is finite after the split-level `TAT` fix, despite zero-completion cases |
| Basilisk 2.9.1 | `test_official64` | **23.24** | **24.76** | **23.20** | **0.18** | 41.46 | **4.84** | Current repo default environment |
| Basilisk 2.9.1 | `test_random64` | 18.45 | 19.49 | 18.40 | 0.19 | **34.85** | 5.90 | Lower than `test_official64`, consistent with selection bias |

\* `CS` here is the repository's provisional formula applied to the released
sidecar means. The released sidecars do not ship a split-level `CS`.

\** The original `bsk2513 × test_official64` aggregate still contained `inf`
because it was produced before the split-level `TAT` aggregation fix. The
table uses corrected values reconstructed from the per-scenario results:
`TAT = 699.15 s = 0.19 h`, `CS = 7.16`.

## Released Sidecar Aggregate

You can reproduce the released-sidecar row with:

```bash
uv run python scripts/aggregate_sidecar_metrics.py --split test
```

At the time of writing, this returns:

```json
{
  "count": 64,
  "display_metrics": {
    "CR_percent": 34.94006430702518,
    "CS_provisional": 3.4821173343413307,
    "PC_Wh": 50.08115687052409,
    "PCR_percent": 38.397219046601094,
    "TAT_hours": 0.16689796129862466,
    "WCR_percent": 34.43985547928605
  },
  "epoch": 1,
  "raw_metrics": {
    "CR": 0.34940064307025176,
    "CS_provisional": 3.4821173343413307,
    "PC_watt_seconds": 180292.16473388672,
    "PCR": 0.38397219046601094,
    "TAT_seconds": 600.8326606750488,
    "WCR": 0.3443985547928605
  },
  "split": "test"
}
```

## Run Commands

Normal repo environment, Basilisk `2.9.1`:

```bash
uv run python scripts/eval.py configs/eval/official_aeosformer.yaml data/model/model.pth --split test_official64 --output-dir outputs/eval/repro/bsk291/test_official64
uv run python scripts/eval.py configs/eval/official_aeosformer.yaml data/model/model.pth --split test_random64 --output-dir outputs/eval/repro/bsk291/test_random64
```

Separate `.venv-bsk2513` environment, Basilisk `2.5.13`:

```bash
python scripts/eval.py configs/eval/official_aeosformer.yaml data/model/model.pth --split test_official64 --output-dir outputs/eval/repro/bsk2513/test_official64
python scripts/eval.py configs/eval/official_aeosformer.yaml data/model/model.pth --split test_random64 --output-dir outputs/eval/repro/bsk2513/test_random64
```

See [`docs/bsk2513.md`](./bsk2513.md) for the setup of the `2.5.13`
environment.

## Current Takeaways

- The paper row and the released sidecar row are not consistent. The released
  sidecars are substantially better than the paper on `CR`, `PCR`, `WCR`, and
  provisional `CS`, so they should be treated as a separate reference artifact,
  not as a direct serialization of the paper table.
- The paper's reported `TAT/h` values do not appear compatible with the
  released evaluator logic or the released test timing. On the released data,
  the mission horizon is one hour, and the released sidecars remain in the
  `~600 s` (`~0.17 h`) regime.
- On this reimplementation, `Basilisk 2.9.1` performs much better than
  `Basilisk 2.5.13` on both `test_official64` and `test_random64`. That means
  Basilisk version is not a minor nuisance variable here; it materially changes
  the reported metrics.
- For both Basilisk versions, `test_random64` is harder than `test_official64`
  on the current evaluator. That supports the concern that the official 64-case
  selection is not interchangeable with an arbitrary 64-case sample from the
  raw test pool.
- The remaining `2.5.13` zero-completion anomalies around cases such as `281`
  and `913` still deserve investigation. The split-level aggregation bug is now
  fixed, but the per-scenario behavior gap is likely deeper than reporting.
