# Structural Conditioning for Source-Associated Seismic Resynthesis

Official code, experiment, and result repository for the manuscript:

> **Structural Conditioning for Source-Associated Seismic Resynthesis: Multi-Seed and Cross-Survey Evaluation of Structural Priors**  
> Bensheng Yun, Xiaofeng Zhang, Yang Xiang, and Jie Shen  
> Manuscript prepared for submission to **Acta Geophysica** (2026)

This repository accompanies a controlled study of **structural-condition representation** in seismic resynthesis. The primary scientific question is not whether a particular GAN architecture is universally superior, but how the **spatial support, continuity, orientation information, and selectivity of the supplied structural condition** affect strongly conditioned seismic synthesis when the network backbone and training protocol are held fixed.

The common synthesis backbone is a StyleGAN2-inspired conditional generator with bounded residual FiLM conditioning. Three structural priors are compared under matched conditions:

- **SLoG**: structure-tensor-guided smoothing-based Laplacian of Gaussian; an orientation-aware continuous structural prior;
- **LoG**: isotropic Laplacian of Gaussian;
- **Canny**: binary edge skeleton.

The main condition layout is:

```text
fault + AGC + edge
```

where `fault` is a label-free fault-likelihood channel, `AGC` is the automatic-gain-controlled seismic profile, and `edge` is SLoG, LoG, or Canny.

> **Interpretation:** because the conditions are extracted from the corresponding source seismic profiles, the task is treated as **source-associated seismic resynthesis**, not unconstrained generation of independent geological realizations.

---

## 1. Scientific scope

The study uses a common StyleGAN2–FiLM backbone as a controlled test bed for a geophysical representation question:

> **What form of structural support is most effective for preserving reflector continuity and fault-related discontinuities when condition layout, architecture, optimization, and evaluation are matched?**

The working hypothesis is that a continuous, orientation-aware structural representation can provide more useful spatial support than either an isotropic graded response or a compact binary edge skeleton.

The repository supports the following analyses:

- matched SLoG / LoG / Canny conditional synthesis;
- three independent GAN training seeds;
- late-window checkpoint robustness;
- deterministic U-Net-L1 and U-Net-Struct controls;
- fixed-condition latent-sensitivity analysis;
- 500-profile structural-prior characterization;
- paired residual analysis;
- within-survey downstream fault segmentation;
- label-free external F3 synthesis evaluation;
- zero-shot Australian-to-CRACKS fault-transfer evaluation.

---

## 2. Main Australian synthesis results

### 2.1 Model-level three-seed comparison

For each structural prior, three generators are trained independently using training seeds:

```text
35, 39, 42
```

Each selected generator is evaluated with 150 matched latent/evaluation resamples. Final values below are the mean ± sample standard deviation across the three independently trained generators (`n = 3`).

The final Australian FD/RR values use `T = 1000` real-real reference partitions.

| Prior | SWD | GEO KID | Total phi FD/RR | StructTex FD/RR | Spectrum FD/RR | Energy FD/RR |
|---|---:|---:|---:|---:|---:|---:|
| **SLoG** | **0.1176 ± 0.0001** | **0.5770 ± 0.2118** | **0.9150 ± 0.0848** | **1.0377 ± 0.4505** | **0.2552 ± 0.0847** | 1.4354 ± 0.1169 |
| LoG | 0.1185 ± 0.0003 | 4.1668 ± 1.9808 | 1.0898 ± 0.1480 | 3.6926 ± 0.9721 | 0.5410 ± 0.0679 | 1.1735 ± 0.1422 |
| Canny | 0.1202 ± 0.0005 | 5.5309 ± 2.7172 | 1.5245 ± 0.1934 | 8.3345 ± 3.0062 | 0.8986 ± 0.2979 | **1.0531 ± 0.0975** |

Lower values are better.

The main interpretation is therefore **not** that SLoG is universally superior for every seismic attribute. SLoG gives the lowest overall, structural, and spectral discrepancies, whereas the Energy subspace more often favors Canny.

### 2.2 Seed-level total phi FD/RR

| Prior | Seed 35 | Seed 39 | Seed 42 |
|---|---:|---:|---:|
| **SLoG** | **0.858** | **1.012** | **0.875** |
| LoG | 1.200 | 1.148 | 0.922 |
| Canny | 1.454 | 1.743 | 1.376 |

The SLoG–LoG–Canny ordering is retained for total phi FD/RR at all three independent training seeds.

The seed-specific SLoG / LoG / Canny result files already stored under `results/` correspond to these independently trained **Australian** generators when they use the 512×512 Australian protocol and independently selected checkpoints. See the reproducibility notes below for how to distinguish them from the F3 runs.

### 2.3 Late-window robustness

A common late-training interval contains 22 checkpoints shared by all nine Australian prior–seed runs:

```text
10,000 to 20,500 steps
```

Across these 22 common checkpoints:

- SLoG has the lowest mean **total phi FD/RR** at 22/22 checkpoints;
- SLoG has the lowest mean **GEO KID** at 22/22 checkpoints;
- SLoG has the lowest mean **StructTex FD/RR** at 22/22 checkpoints;
- SLoG has the lowest mean **Spectrum FD/RR** at 22/22 checkpoints;
- SWD is mixed: SLoG is lowest at 11 checkpoints, LoG at 10, and Canny at 1;
- Energy FD/RR is mixed: Canny is lowest at 11 checkpoints, SLoG at 9, and LoG at 2.

This analysis is used to test whether the cross-prior ordering is restricted to a single selected checkpoint.

---

## 3. External F3 synthesis evaluation

A separate field-data stress test uses cropped real seismic images from the Netherlands F3 survey.

Key settings:

```text
image size:             128 × 128
training seeds:         35, 39, 42
training budget:        25,000 steps
evaluated checkpoint:   25,000-step EMA checkpoint for every run
evaluation resamples:   150 per trained generator
```

To isolate the direct structural-prior contribution, the edge contribution to the fault-likelihood channel is removed in this experiment:

```text
fault edge weight        = 0
coherence-reduction      = 0.6470588
orientation-change       = 0.3529412
```

Thus the fault and AGC channels are shared across SLoG, LoG, and Canny; the direct structural-prior channel is the controlled difference.

### F3 model-level results

| Prior | SWD | GEO MMD | GEO W | phi FD | phi FD/RR | phi KID |
|---|---:|---:|---:|---:|---:|---:|
| **SLoG** | **0.1080 ± 0.0064** | **0.3036 ± 0.0253** | **1.0413 ± 0.0669** | **96.11 ± 5.70** | **2.5779 ± 0.1990** | **1.8279 ± 0.2245** |
| LoG | 0.1091 ± 0.0066 | 0.6292 ± 0.0303 | 1.6144 ± 0.0960 | 323.84 ± 12.18 | 8.6844 ± 0.5036 | 20.3779 ± 1.7458 |
| Canny | 0.1146 ± 0.0070 | 0.6668 ± 0.0193 | 2.1333 ± 0.0949 | 506.69 ± 17.00 | 13.5909 ± 0.8071 | 54.6218 ± 4.8891 |

The F3 images used for this synthesis comparison are label-free. These values therefore describe distributional agreement, not fault-label accuracy.

---

## 4. Zero-shot Australian-to-CRACKS evaluation

Downstream cross-survey transfer is evaluated on the expert-annotated CRACKS resource from the Netherlands North Sea F3 volume.

The CRACKS expert subset used in the manuscript contains:

```text
40 matched seismic-section / mask pairs
native image size: 255 × 701
U-Net training seeds: 35, 39, 42
prediction threshold: 0.5
tolerance radius: 3 pixels
```

Primary confident-label evaluation uses:

```text
class 3 = fault
class 1 = non-fault
class 0 = ignored
class 2 = ignored / uncertain
```

A sensitivity analysis additionally treats class 2 as fault.

### Primary confident-label results

| Training source | Dice/F1 | IoU | Precision | Recall | Tolerant F1 |
|---|---:|---:|---:|---:|---:|
| Real | 0.0954 ± 0.0139 | 0.0501 ± 0.0076 | 0.0907 ± 0.0196 | 0.1049 ± 0.0273 | 0.1184 ± 0.0152 |
| **SLoG synthetic** | **0.1504 ± 0.0269** | **0.0814 ± 0.0157** | **0.1468 ± 0.0308** | 0.1554 ± 0.0265 | **0.1833 ± 0.0357** |
| LoG synthetic | 0.1461 ± 0.0396 | 0.0791 ± 0.0233 | 0.1056 ± 0.0353 | **0.2508 ± 0.0513** | 0.1630 ± 0.0458 |
| Canny synthetic | 0.1492 ± 0.0139 | 0.0806 ± 0.0081 | 0.1117 ± 0.0082 | 0.2279 ± 0.0427 | 0.1689 ± 0.0131 |

The low absolute values indicate substantial cross-survey domain shift. The experiment is interpreted as a precision–recall trade-off rather than as evidence of universal downstream superiority.

CRACKS v2:
https://doi.org/10.5281/zenodo.13926822

---

## 5. Deterministic controls

The deterministic baselines use the same source-derived conditional tensor as the adversarial model.

| Model | SWD | GEO KID | Total phi FD/RR | StructTex | Spectrum | Energy |
|---|---:|---:|---:|---:|---:|---:|
| U-Net-L1 | 0.1187 ± 0.0165 | 0.0919 ± 0.0352 | 0.6762 ± 0.0423 | 0.6120 ± 0.1351 | 0.0559 ± 0.0043 | 1.1012 ± 0.1374 |
| U-Net-Struct | 0.1187 ± 0.0163 | 0.0836 ± 0.0326 | 0.5228 ± 0.0335 | 0.5289 ± 0.1236 | 0.0211 ± 0.0040 | 0.8527 ± 0.0946 |

These controls estimate how much source-associated agreement can be recovered directly from the condition tensor without a latent input or adversarial discriminator.

They are **not** used to claim that the GAN is intrinsically superior to deterministic reconstruction. The StyleGAN2–FiLM model is used as a common conditional synthesis backbone in which structural-representation differences can be isolated.

---

## 6. Fixed-condition latent sensitivity

For the reported SLoG latent-sensitivity experiment, the condition tensor and synthesis-noise realization are held fixed while the latent code varies.

```text
fixed conditions:        20
latent codes/condition:  50
latent seed:             46000
synthesis-noise seed:    2026
```

| Measure | Value |
|---|---:|
| Mean pixel variance | 4.1127 × 10^-6 |
| Mean pairwise LPIPS | 3.1337 × 10^-4 |
| Active-variance pixel fraction | 0.03836% |
| GEO Structure within/between ratio | 8.1812 × 10^-4 |
| GEO Texture within/between ratio | 1.3305 × 10^-3 |
| phi StructTex within/between ratio | 1.3246 × 10^-3 |
| Fault-geometry within/between ratio | 1.6094 × 10^-3 |

These results indicate that the system is strongly condition dominated and has limited latent-driven diversity under the tested protocol.

---

## 7. Repository structure

```text
slog-seismic-profile-generation/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── run_conditional_stylegan_experiment.py
├── slog_parameter_search.py
├── unet_regression_baseline.py
├── seismic_edge_metrics.py
├── downstream_fault_segmentation.py
│
└── results/
    ├── stylegan_slog_summary.json
    ├── stylegan_log_summary.json
    ├── stylegan_canny_summary.json
    ├── unet_l1_summary.json
    ├── unet_struct_summary.json
    ├── unet_l1_paired_metrics.json
    ├── unet_struct_paired_metrics.json
    ├── stylegan_slog_latent_sensitivity.json
    ├── algorithm_summary.csv
    ├── seismic_edge_metrics.csv
    ├── effective_parameters.json
    └── seed-specific SLoG / LoG / Canny result files
```

The final repository snapshot should retain the seed-specific result files for training seeds 35, 39, and 42 because the manuscript treats the **trained generator**, rather than metric resampling alone, as the model-level independent unit.

Historical files may use `glog` as the internal alias for SLoG:

```text
glog == SLoG
```

---

## 8. Main programs

### `run_conditional_stylegan_experiment.py`

Main implementation for:

- conditional StyleGAN2 training;
- SLoG / LoG / Canny condition construction;
- multi-scale FiLM conditioning;
- EMA checkpoint evaluation;
- GEO and phi feature extraction;
- FD/RR, KID-style polynomial MMD², PRDC, and SWD;
- downstream synthetic-profile generation;
- fixed-condition latent-sensitivity analysis.

### `slog_parameter_search.py`

Parameter-search and preprocessing-analysis program used to examine SLoG and alternative structural-prior settings.

### `unet_regression_baseline.py`

Deterministic condition-to-seismic U-Net implementation used for U-Net-L1, U-Net-Struct, paired reconstruction metrics, and distributional evaluation.

### `seismic_edge_metrics.py`

Structural-prior characterization for SLoG / LoG / Canny, including edge density, directional continuity, break rate, fragmentation, short-response ratio, orientation entropy, and low-coherence association.

### `downstream_fault_segmentation.py`

Common U-Net fault-segmentation implementation used for comparisons among real, SLoG-synthetic, LoG-synthetic, and Canny-synthetic training sources.

---

## 9. Environment

The manuscript experiments were implemented under:

```text
Ubuntu 22.04 LTS
Python 3.9
PyTorch 1.12
NVIDIA RTX A6000, 48 GB VRAM
batch size = 16 for the main GAN experiments
```

Install:

```bash
git clone https://github.com/Xiaofeng-12/slog-seismic-profile-generation.git
cd slog-seismic-profile-generation

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Conda:

```bash
conda create -n slog-seismic python=3.9 -y
conda activate slog-seismic
pip install -r requirements.txt
```

If the pinned CUDA build is not appropriate for the local GPU/driver, install a compatible PyTorch build first and then install the remaining dependencies.

---

## 10. Australian dataset and partition

The main dataset is the public interpreted 3-D seismic volume released by:

> An et al. (2021), *A gigabyte interpreted seismic dataset for automatic fault recognition*, Data in Brief, 37, 107219.

Data article:
https://doi.org/10.1016/j.dib.2021.107219

Dataset:
https://doi.org/10.7910/DVN/YBYGBK

The original seismic data are not redistributed here.

### Preparation

1. Two-dimensional profiles are extracted from the original 3-D volume.
2. The original `3174 × 1537` planes are proportionally resampled to `2000 × 968`.
3. The extraction produces 1,603 2-D profiles.
4. A fixed subset of 1,200 profiles is retained and ordered by source index.
5. Four hundred profiles sampled at regular index intervals are used for model development.
6. The remaining 800 profiles form the within-volume evaluation partition.

### Important limitation

The 400 development profiles and 800 evaluation profiles come from the **same 3-D seismic volume** and may contain spatially adjacent sections.

Therefore:

- the Australian evaluation is a **matched within-volume comparison**;
- local spatial dependence may make absolute discrepancy values more optimistic than a spatially blocked evaluation;
- SLoG, LoG, and Canny use exactly the same source indices, preprocessing, training budget, and evaluator;
- the split therefore supports controlled **relative comparison** among the priors, but it should not be interpreted as a leakage-free estimate of cross-survey generalization;
- the independent F3 experiments provide complementary evidence under field-domain shift.

The manuscript does **not** claim that the prior ranking is mathematically guaranteed to remain unchanged under a spatially blocked split.

---

## 11. Preprocessing and condition construction

The principal preprocessing sequence is:

```text
grayscale
  ↓
non-local means denoising
  ↓
structure-guided smoothing
  ↓
automatic gain control
  ↓
structural-prior construction
```

Australian synthesis images are cropped/resampled to:

```text
512 × 512
```

and normalized to:

```text
[-1, 1]
```

### Main preprocessing parameters

| Parameter | Value |
|---|---:|
| NLMeans strength `h` | 0.8 |
| SGS `sigma` | 1.5 |
| SGS anisotropy | 2.2 |
| SGS iterations | 1 |
| SGS diffusion step `gamma` | 0.06 |
| AGC window | 31 |
| AGC lower percentile `p1` | 1.0 |

### Structural-prior parameters

| Prior | Settings |
|---|---|
| SLoG | `sigma=0.4`, `anisotropy=2.1`, `nangles=8`, `sharpness=14`, `alpha=0.92` |
| LoG | `sigma=1.1` |
| Canny | `sigma=1.7`, `low=0.1`, `high=0.9` |

See:

```text
results/effective_parameters.json
```

for the cleaned parameter record associated with the structural-prior characterization.

---

## 12. Australian StyleGAN2 training

Display available options:

```bash
python run_conditional_stylegan_experiment.py --help
```

### Example: SLoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir /path/to/train_400 \
  --out_dir ./outputs/stylegan_slog_seed35 \
  --edge_type slog \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --seed 35 \
  --amp
```

Repeat with:

```text
--seed 39
--seed 42
```

For LoG:

```text
--edge_type log
```

For Canny:

```text
--edge_type canny
```

If a historical script version uses:

```text
--edge_type glog
```

then `glog` denotes SLoG.

The main models are trained for 25,000 optimization steps, with EMA checkpoints used for evaluation.

For the original seed-42 Australian runs, the selected checkpoints were:

| Prior | Seed-42 selected step |
|---|---:|
| SLoG | 17,500 |
| LoG | 23,500 |
| Canny | 24,500 |

Seeds 35 and 39 use their own independently selected checkpoints according to the same total phi FD/RR criterion.

---

## 13. Shared-evaluator testing

The final cross-prior Australian comparison uses a common LoG-based GEO / phi evaluator.

This distinction is important:

```text
training/conditioning edge type != evaluation edge type
```

For example:

```text
condition edge = SLoG
evaluation edge = LoG
```

A generic test command is:

```bash
python run_conditional_stylegan_experiment.py \
  --profile australian \
  --mode test \
  --data_dir /path/to/evaluation_800 \
  --out_dir ./outputs/test_slog_seed35 \
  --ckpt_path /path/to/selected_seed35_checkpoint.pth \
  --device cuda:0 \
  --edge_type slog \
  --eval_edge_type log \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --seeds 43,44,45,46,47 \
  --repeats 30 \
  --test_mixing_prob 0.0 \
  --eval_n 256 \
  --rr_fd_trials 1000 \
  --rr_fd_percentile 95
```

For the final Australian results reported in the manuscript, `--rr_fd_trials 1000` and `--rr_fd_percentile 95` must be retained. The default `rr_fd_trials` value in the main script is intended for faster exploratory evaluation and does not reproduce the final FD/RR reference estimation used in the paper.

The final paper treats the three independently trained generators (`35`, `39`, `42`) as the model-level independent units. Repeated metric/latent evaluation within one trained generator is **not** reported as independent-training uncertainty.

---

## 14. How to interpret the result files

The `results/` directory contains both cleaned summary outputs and seed-specific evaluation outputs.

### Australian seed files

A seed-specific SLoG / LoG / Canny file is an **Australian three-seed result** if it corresponds to:

```text
image size = 512 × 512
training seed = 35, 39, or 42
Australian 400/800 partition
selected checkpoint determined independently for that training run
shared LoG-based evaluator
```

The expected seed-level total phi FD/RR values are:

```text
SLoG:  seed35 = 0.858, seed39 = 1.012, seed42 = 0.875
LoG:   seed35 = 1.200, seed39 = 1.148, seed42 = 0.922
Canny: seed35 = 1.454, seed39 = 1.743, seed42 = 1.376
```

These are the files used to form the manuscript's final Australian model-level means.

### F3 seed files

F3 files are distinguishable because they correspond to:

```text
image size = 128 × 128
training seed = 35, 39, or 42
common evaluated checkpoint = 25,000 steps
edge contribution removed from the fault-likelihood channel
external F3 field-image dataset
```

Therefore, files labeled only by `SLoG/LoG/Canny + seed35/39/42` should not automatically be assumed to be F3. The dataset, image size, checkpoint step, and preprocessing/fault-channel configuration should be checked.

### Legacy summary files

The original files:

```text
stylegan_slog_summary.json
stylegan_log_summary.json
stylegan_canny_summary.json
```

may contain the earlier metric/latent-resampling summaries from a single selected generator. These are useful auxiliary records but should not be confused with the final **three-independent-training-seed model-level uncertainty** reported in the current manuscript.

---

## 15. Evaluation spaces

### GEO

The GEO representation contains 28 domain-oriented attributes emphasizing interpretable seismic structure and texture.

### phi

The phi representation contains 160 dimensions:

```text
StructTex = 32
Spectrum  = 64
Energy    = 64
```

### FD/RR

FD/RR is a feature-space-specific normalized Fréchet-distance measure using a real-real reference distribution.

Lower values indicate closer agreement in that feature space.

It should not be interpreted as an absolute measure of geological validity.

### KID terminology

The reported KID values are KID-style unbiased polynomial-kernel MMD² estimates in domain-specific feature spaces.

They are **not** ImageNet Inception KID.

Likewise, domain-specific FD values should not be interpreted as ImageNet FID.

---

## 16. Reproducibility notes

- GAN training is stochastic.
- Training seeds 35, 39, and 42 represent independently trained generators.
- Evaluation seeds 43–47 and repeated resampling are used within each trained model and should not be treated as independent model replicates.
- Final Australian model-level uncertainty is computed across the three independently trained generators.
- The final Australian cross-prior comparison must use the same LoG-based shared evaluator.
- Australian checkpoint selection is performed independently for each training seed using the same criterion.
- The late-window analysis uses 22 common checkpoints from 10,000 to 20,500 steps.
- The F3 comparison evaluates every prior–seed model at the same 25,000-step checkpoint.
- U-Net evaluation uses EMA weights.
- Historical files may use `glog` as an alias for SLoG.
- The fixed-condition latent experiment holds synthesis noise fixed and isolates latent-code sensitivity.
- Numerical values can vary slightly across CUDA, GPU, driver, and library versions.

---

## 17. Data and code availability

### Australian seismic dataset

Harvard Dataverse:

https://doi.org/10.7910/DVN/YBYGBK

### CRACKS expert annotations

Zenodo:

https://doi.org/10.5281/zenodo.13926822

### Source code

GitHub:

https://github.com/Xiaofeng-12/slog-seismic-profile-generation

For the journal submission snapshot, a versioned GitHub Release and an archival DOI (for example through Zenodo) are recommended so that the exact code/results corresponding to the submitted manuscript remain permanently identifiable.

---

## 18. Citation

Before publication, please cite the manuscript as:

```bibtex
@unpublished{Yun2026StructuralConditioning,
  title  = {Structural Conditioning for Source-Associated Seismic Resynthesis:
            Multi-Seed and Cross-Survey Evaluation of Structural Priors},
  author = {Yun, Bensheng and Zhang, Xiaofeng and Xiang, Yang and Shen, Jie},
  year   = {2026},
  note   = {Manuscript prepared for submission to Acta Geophysica}
}
```

After publication, replace this entry with the final journal citation, volume/article number, and DOI.

---

## 19. License

This repository is released under the **MIT License**.

The original seismic datasets and third-party software remain subject to their own licenses and terms of use.

---

## 20. Contact

**Bensheng Yun**  
Zhejiang University of Science and Technology  
No. 318 Liuhe Road, Hangzhou 310023, Zhejiang, China  
Email: yunbsh@zust.edu.cn  
ORCID: 0009-0003-3075-2684
