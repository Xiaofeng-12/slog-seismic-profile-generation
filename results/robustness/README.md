# Checkpoint and ranking robustness results

This directory contains the checkpoint-wise robustness outputs used to assess whether the Australian SLoG / LoG / Canny comparison depends on a particular late training checkpoint.

The analysis is based on the three independently trained GAN seeds:

```text
35, 39, 42
```

for each structural prior.

## Common late-checkpoint protocol

For direct same-step comparison, only checkpoints available for all nine prior × training-seed runs are retained.

The common late window contains 22 checkpoints:

```text
10000, 10500, 11000, ..., 20000, 20500
```

with a 500-step interval.

All discrepancy metrics use the convention:

```text
lower is better
```

The common-step comparison is intended as a checkpoint-robustness analysis rather than as a replacement for the final held-out evaluation reported elsewhere in the repository.

## Files

### `development_metrics_clean.csv`

Cleaned checkpoint-wise development metrics used as the input to the robustness analysis.

Each row corresponds to one structural prior, one GAN training seed, and one evaluated checkpoint.

The file contains all available late-window checkpoint measurements after applying the minimum-step criterion.

### `common_step_rankings.csv`

Prior rankings at the 22 checkpoints shared by all nine prior × training-seed runs.

For each checkpoint and metric, the file reports:

* mean across the three GAN training seeds;
* standard deviation across the three training seeds;
* number of training seeds;
* rank among SLoG, LoG, and Canny.

The analyzed metrics are:

```text
total phi FD/RR
GEO KID
SWD
StructTex phi FD/RR
Spectrum phi FD/RR
Energy phi FD/RR
```

### `common_step_winner_frequency.csv`

Summary of how often each prior obtains rank 1 across the 22 common checkpoints.

The main late-window counts are:

```text
Metric                  SLoG     LoG     Canny
------------------------------------------------
Total phi FD/RR         22/22     0/22     0/22
GEO KID                 22/22     0/22     0/22
StructTex phi FD/RR     22/22     0/22     0/22
Spectrum phi FD/RR      22/22     0/22     0/22
SWD                     11/22    10/22     1/22
Energy phi FD/RR         9/22     2/22    11/22
```

Thus, SLoG retains the lowest discrepancy at every common checkpoint for total phi FD/RR, GEO KID, StructTex, and Spectrum, whereas SWD and Energy show mixed prior rankings.

### `selected_steps_wide.csv`

Checkpoint choices obtained under alternative checkpoint-selection rules.

This file is retained to document sensitivity to the checkpoint-selection criterion and to show which checkpoints would be selected under different development-stage rules.

## Interpretation

The late-window analysis supports the stability of the relative Australian ranking across training progress for several major discrepancy measures.

It should not be interpreted as a leakage-free cross-survey validation because the Australian development and evaluation profiles originate from the same 3D seismic volume.

The matched design nevertheless applies the same source-profile sampling, preprocessing, model architecture, training budget, checkpoint window, and evaluator to all three structural priors.

