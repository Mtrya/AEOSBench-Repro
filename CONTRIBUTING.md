# Contributing

This repository aims to be community-maintained reimplementation of
AEOSBench and AEOS-Former. Contributions are welcome both for faithful
reproduction and for pushing the benchmark forward.

## Main Contribution Directions

### 1. Reproduction

- verify how closely this repo reproduces the paper's reported results
- compare paper numbers, official released artifacts, and repo-trained
  checkpoints
- investigate and document discrepancies

### 2. Algorithmic Improvements

- improve model quality, training stability, optimization, or robustness
- paper-faithfulness is valuable, but improvement-oriented contributions are
  also welcome

### 3. Baselines

- implement missing baselines
- document primary sources and AEOSBench adaptation assumptions
- keep baseline evaluation under the same reporting contract

### 4. Bug Fixes

- fix correctness issues in evaluation, training, RL, dataset generation, and
  reporting
- include regression tests whenever practical

### 5. Leaderboard

- contribute stronger checkpoints or better methods
- help maintain a fair, auditable leaderboard
- provide enough metadata for others to reproduce claims

### 6. Documentation And Investigation

- clarify paper and official-code divergences
- improve setup, usage, and design documentation
- document confirmed behaviors and unresolved ambiguities

### 7. Performance And Infrastructure

- speed up evaluation or training
- improve logging, resume behavior, progress reporting, and multi-GPU
  scalability
- improve tooling for experiment comparison and reproducibility

## Reproduction vs Improvement

There are two broad categories of contributions:

- reproduction contributions aim to match the paper or explain why results
  differ
- improvement contributions aim to outperform the paper or make the system
  more practical

Both are welcome. Contributions should state clearly which category they
belong to.

## Expectations For Result Claims

- do not cherry-pick checkpoints per case
- report exact configs, checkpoint paths, and splits
- prefer machine-readable evaluation outputs
- if claiming improved results, include enough detail for others to rerun the
  experiment

## Code And Test Expectations

- keep scripts thin; put logic in `aeosbench/`
- prefer explicit modules over framework-heavy abstractions
- use YAML configs
- put tests under `./tests`
- prefer small unit tests before heavy simulator-backed tests when possible

## Small Contributions

Small contributions are also useful, including:

- typo fixes
- README improvements
- config comments
- test cleanups
- reproducibility tooling
