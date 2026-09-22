# F3 external-field evaluation results

This directory contains the final three-training-seed external F3 synthesis evaluation used in the manuscript.

## Experimental protocol

The F3 experiment uses:

```text
image size:                 128 × 128
GAN training seeds:         35, 39, 42
evaluated checkpoint:       25,000 steps
evaluation edge:            LoG
evaluation seeds:           43, 44, 45, 46, 47
repeats per evaluation seed: 30
evaluation runs per model:  150
```

To isolate the direct contribution of the structural-prior channel, the edge contribution to the fault-likelihood channel is removed in this experiment.

The condition layout remains:

```text
fault + AGC + edge
```

while the direct edge representation is SLoG, LoG, or Canny.

## Files

### `f3_manifest.csv`

Records the protocol for each of the nine independently trained generators, including:

* structural prior;
* GAN training seed;
* result filename;
* checkpoint step;
* image size;
* batch size;
* training epochs;
* condition edge type;
* shared evaluation edge;
* evaluation seeds;
* number of repeats;
* number of evaluation runs.

### `*_metrics_test_summary.json`

The nine generator-specific files correspond to:

```text
SLoG × seeds 35, 39, 42
LoG  × seeds 35, 39, 42
Canny × seeds 35, 39, 42
```

Each file contains the complete set of 150 matched evaluation runs together with within-model summary statistics.

### `seed_level_summary.csv`

Reports one row for each independently trained prior × seed model.

The reported mean for each row is computed from that model's 150 evaluation runs. The accompanying evaluation standard deviation describes within-model resampling variability and is not independent-training uncertainty.

### `model_level_summary.csv`

Reports the final model-level mean and sample standard deviation across the three independently trained GANs for each structural prior.

The final total phi FD/RR values are:

```text
SLoG   2.5710 ± 0.4045
LoG   11.7568 ± 3.6821
Canny 16.2937 ± 2.9796
```

Here the standard deviation is computed across GAN training seeds `35`, `39`, and `42`.

## Interpretation

The F3 dataset used for this synthesis evaluation is label-free.

Therefore, these results quantify distributional agreement between generated and real seismic profiles and should not be interpreted as direct fault-label accuracy.

The F3 experiment serves as an external-field stress test complementary to the controlled Australian within-volume comparison.

