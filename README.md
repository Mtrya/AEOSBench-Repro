# AEOSBench-Repro

This repository is the **unofficial, community maintained, re-implemented** version of the paper "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology", [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515). This is re-implemented from scratch and is **NOT** the official implementation. **Not affiliated with the original authors.**

## Installation

```bash
uv sync
```

## Data

### Download all data (to reproduce the paper)

```bash
hf download MessianX/AEOS-dataset --repo-type dataset --local-dir ./data
```

### Download specific fragments only

Download only model checkpoints:
```bash
hf download MessianX/AEOS-dataset model.tar --repo-type dataset --local-dir ./data
```

Download only `trajectories.1/`:
```bash
hf download MessianX/AEOS-dataset trajectories.1/ --repo-type dataset --local-dir ./data
```

### Extract all archives (if already downloaded)

If you downloaded data without extracting:
```bash
find ./data -type f -name '*.tar' -exec sh -c 'tar -xf "$1" -C "$(dirname "$1")" && rm "$1"' _ {} \;
```

## Steps

### 1. Confirm the data

The right file tree should look like this：

```
data/                               # ~340 GB total
├── trajectories.1/                 #  ~79 GB
│   ├── test/
│   ├── train/
│   │   ├── 00/         # contains pth and json files per case
│   │   ├── 01/
│   │   ├── ...
│   ├── val_seen/
│   └── val_unseen/
├── trajectories.2/                 # ~100 GB
├── trajectories.3/                 # ~155 GB
├── model/                          # ~370 MB
│   ├── callbacks.pth
│   ├── meta.pth
│   ├── model.pth
│   ├── optim.pth
│   └── strategy.pth
└── data/                           #   ~6 GB
    ├── annotations/                #  ~136 KB
    │   ├── test.json
    │   ├── train.json
    │   ├── val_seen.json
    │   └── val_unseen.json
    ├── constellations/             #   ~3.7 GB
    │   ├── test/
    │   │   └── 00/     # contains json files per case (e.g. 00000.json)
    │   ├── train/
    │   │   ├── 00/
    │   │   ├── 01/
    │   │   ├── ...
    │   ├── val_seen/
    │   └── val_unseen/
    ├── orbits/                     #   ~20 MB, json files per orbit
    ├── satellites/                 #   ~12 MB
    │   ├── test/
    │   ├── train/
    │   ├── val_seen/
    │   └── val_unseen/
    └── tasksets/                   #   ~2.4 GB
        ├── mrp.json
        ├── test/
        ├── train/
        ├── val_seen/
        └── val_unseen/
```

If your dataset is not in the default `./data` location, set
`AEOS_DATA_ROOT=/path/to/data_root` for the commands below.

## Supervised Training

Paper-default supervised pretraining:

```bash
python scripts/train_sl.py configs/train_sl/paper_default.yaml --device cuda
```

Fast smoke run:

```bash
python scripts/train_sl.py configs/train_sl/tiny.yaml --device cpu
```

Useful flags:

- `--work-dir path/to/run_dir` to override the default timestamped directory under `outputs/train_sl/`
- `--resume latest` or `--resume path/to/checkpoints/iter_N`
- `--load-model-from path/to/model.pth` to overlay weights before training

Checkpoints are written under `checkpoints/iter_N/`. Evaluate a produced checkpoint with:

```bash
python scripts/eval.py configs/train_sl/paper_default.yaml outputs/train_sl/<run>/checkpoints/iter_N/model.pth --split val_seen --limit 1
```

## Evaluation

Use an explicit model config plus one or more checkpoints:

```bash
python scripts/eval.py configs/eval/official_aeosformer.yaml data/model/model.pth --split val_seen
```

Start with a subset:

```bash
python scripts/eval.py configs/eval/official_aeosformer.yaml data/model/model.pth --split val_seen --limit 4
```

Multiple splits or checkpoints are allowed and are reported separately:

```bash
python scripts/eval.py \
  configs/eval/official_aeosformer.yaml \
  data/model/model.pth \
  path/to/another-model.pth \
  --split val_seen --split val_unseen
```

Useful flags:

- `--device cuda` to force GPU when your local PyTorch can see it
- `--format terminal --format json --format md` to limit outputs
- `--output-dir path/to/output_dir` to override the default timestamped directory under `outputs/eval/`
- `--no-progress` to disable the scenario-level progress bar

Note that the evaluation is extremely time consuming due to simulation overhead.

Default outputs are a terminal table plus `summary.json` and `summary.md`.
