# CRACKS cross-survey downstream evaluation results

This directory contains the final Australian-to-CRACKS downstream fault-segmentation evaluation and bootstrap analysis used in the manuscript.

## Experimental protocol

A common U-Net fault-segmentation model is trained independently using seismic profiles from four training sources:

```text
Real
SLoG synthetic
LoG synthetic
Canny synthetic
```

Three independent U-Net training seeds are used:

```text
35, 39, 42
```

The external test set contains:

```text
40 expert-annotated CRACKS seismic sections
native image size: 255 × 701
prediction threshold: 0.5
tolerance radius: 3 pixels
```

The primary confident-label protocol interprets the CRACKS mask as:

```text
class 3 = confident fault
class 1 = confident non-fault
class 0 = ignored
class 2 = ignored / uncertain
```

A sensitivity analysis additionally treats uncertain class 2 together with class 3 as positive fault labels.

## Primary three-training-seed results

The table below reports the mean and sample standard deviation across the three independently trained U-Net models.

| Training source |            Dice |             IoU |       Precision |          Recall |     Tolerant F1 |
| --------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| Real            | 0.0954 ± 0.0139 | 0.0501 ± 0.0076 | 0.0907 ± 0.0196 | 0.1049 ± 0.0273 | 0.1184 ± 0.0152 |
| SLoG            | 0.1504 ± 0.0269 | 0.0814 ± 0.0157 | 0.1468 ± 0.0308 | 0.1554 ± 0.0265 | 0.1833 ± 0.0357 |
| LoG             | 0.1461 ± 0.0396 | 0.0791 ± 0.0233 | 0.1056 ± 0.0353 | 0.2508 ± 0.0513 | 0.1630 ± 0.0458 |
| Canny           | 0.1492 ± 0.0139 | 0.0806 ± 0.0081 | 0.1117 ± 0.0082 | 0.2279 ± 0.0427 | 0.1689 ± 0.0131 |

These results indicate a precision-recall trade-off across structural priors. SLoG gives higher mean precision and tolerant F1 than LoG under the primary confident-label protocol, whereas LoG gives higher recall.

The CRACKS experiment is therefore interpreted as a downstream selectivity/sensitivity comparison rather than as evidence that one structural prior is uniformly superior for every segmentation metric.

## Files

### `aggregate_metrics.csv`

Contains the aggregated downstream metrics for each:

```text
label scheme
× training seed
× training source
```

This file retains the seed-specific results used to construct the model-level summaries.

### `training_seed_summary.csv`

Reports the mean, sample standard deviation, minimum, and maximum across the three independently trained U-Net seeds for each metric.

It includes both:

```text
primary
inclusive_uncertain
```

label schemes.

The primary rows correspond to the main CRACKS results reported in the manuscript.

### `per_section_metrics_and_counts.csv`

Contains section-level results for the 40 expert-annotated CRACKS sections.

These section-level observations provide the basis for the paired bootstrap analyses and preserve the correspondence between methods evaluated on the same test sections.

### `paired_bootstrap_results.json`

Contains the paired bootstrap and hierarchical paired-bootstrap analyses comparing SLoG and LoG.

The bootstrap protocol uses:

```text
bootstrap replicates: 10,000
bootstrap seed:       20260821
test sections:        40
training runs:        3
```

The file includes both the primary confident-label protocol and the inclusive uncertain-label sensitivity analysis.

## Statistical interpretation

Training seeds `35`, `39`, and `42` are treated as independent model-training replicates.

The 40 CRACKS test sections are matched across methods, so paired section-level comparisons preserve the common test-section structure.

The hierarchical bootstrap additionally accounts for variation across both training runs and test sections.

The uncertainty-label sensitivity analysis is reported separately because treating class 2 as positive changes the operational definition of a fault target and can alter the precision-recall balance.

## Scope

CRACKS is used as an external downstream evaluation rather than as a synthesis-distribution metric.

The experiment tests whether structural differences in the synthetic training profiles transfer to fault-segmentation behavior on an expert-annotated external dataset.

It should therefore be interpreted together with, rather than as a replacement for, the label-free F3 synthesis evaluation and the controlled Australian within-volume comparison.

