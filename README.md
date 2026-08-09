# Condition-driven seismic synthesis with orientation-aware structural priors

Official code and experiment repository for the manuscript:

> **Condition-driven seismic synthesis with orientation-aware structural priors**  
> Bensheng Yun, Xiaofeng Zhang, Yang Xiang, and Jie Shen  
> Journal of Applied Geophysics submission, 2026

This repository contains the main implementation, deterministic baselines, structural-prior characterization, downstream fault-segmentation experiment, and cleaned quantitative result files associated with the paper.

---

## 1. Overview

The study investigates **source-associated conditional seismic synthesis** rather than unconstrained generation of independent geological realizations.

The main synthesis model is a StyleGAN2-inspired conditional generator using three source-derived channels:

```text
fault + AGC + edge
```

where:

- `fault` is a label-free fault-likelihood prior derived from coherence reduction, local orientation change, and the selected structural edge response;
- `AGC` is the automatic-gain-controlled seismic profile;
- `edge` is one of the structural priors: **SLoG**, **LoG**, or **Canny**.

The proposed **SLoG** prior is an orientation-aware continuous structural representation. It combines eight anisotropic Laplacian-of-Gaussian responses using local orientation, coherence, and diffusion-tensor anisotropy.

The source-derived conditions are injected into the generator through a multi-scale condition encoder and bounded residual feature-wise linear modulation (FiLM).

The paper additionally uses:

- deterministic **U-Net-L1** and **U-Net-Struct** baselines;
- fixed-condition latent-sensitivity analysis;
- 500-profile structural-prior characterization;
- paired residual analysis;
- synthetic-only transfer to a common real-data fault-segmentation test set.

> **Important:** Because the condition channels are extracted from the corresponding source seismic profiles, the task is treated as **source-associated resynthesis**. The results should not be interpreted as independent geological realization generation.

---

## 2. Main findings reproduced by the repository

### 2.1 Matched StyleGAN2 comparison

The three final `fault + AGC + edge` pipelines use matched channel roles, architecture, optimization settings, checkpoint-selection criterion, and a shared LoG-based output evaluator.

| Prior | SWD | GEO KID | Total φ FD/RR | StructTex | Spectrum | Energy |
|---|---:|---:|---:|---:|---:|---:|
| SLoG | **0.1176 ± 0.0115** | **0.4849 ± 0.2880** | **0.9032 ± 0.0913** | **0.8139 ± 0.1912** | **0.3614 ± 0.0461** | 1.4195 ± 0.2219 |
| LoG | 0.1182 ± 0.0117 | 2.9099 ± 0.4959 | 0.9511 ± 0.0873 | 2.9270 ± 0.7176 | 0.5221 ± 0.0687 | **1.0757 ± 0.1505** |
| Canny | 0.1198 ± 0.0119 | 3.3957 ± 0.3891 | 1.4201 ± 0.1329 | 6.0575 ± 1.4350 | 1.2123 ± 0.1515 | 1.2112 ± 0.1723 |

These values describe metric and latent resampling under the reported protocol and should not be interpreted as independent-training uncertainty.

### 2.2 Deterministic baselines

| Model | SWD | GEO KID | Total φ FD/RR | StructTex | Spectrum | Energy |
|---|---:|---:|---:|---:|---:|---:|
| U-Net-L1 | 0.1187 ± 0.0165 | 0.0919 ± 0.0352 | 0.6762 ± 0.0423 | 0.6120 ± 0.1351 | 0.0559 ± 0.0043 | 1.1012 ± 0.1374 |
| U-Net-Struct | 0.1187 ± 0.0163 | **0.0836 ± 0.0326** | **0.5228 ± 0.0335** | **0.5289 ± 0.1236** | **0.0211 ± 0.0040** | **0.8527 ± 0.0946** |
| StyleGAN2-SLoG | **0.1176 ± 0.0115** | 0.4849 ± 0.2880 | 0.9032 ± 0.0913 | 0.8139 ± 0.1912 | 0.3614 ± 0.0461 | 1.4195 ± 0.2219 |

The deterministic baselines quantify how much source-associated agreement is directly recoverable from the same conditional tensor without a latent input or adversarial discriminator.

### 2.3 Fixed-condition latent sensitivity

For StyleGAN2-SLoG, with both the condition tensor and synthesis-noise realization held fixed:

| Measure | Value |
|---|---:|
| Mean pixel variance | 4.1127 × 10⁻⁶ |
| Mean pairwise LPIPS | 3.1337 × 10⁻⁴ |
| Active-variance pixel fraction | 0.03836% |
| GEO Structure within/between ratio | 8.1812 × 10⁻⁴ |
| GEO Texture within/between ratio | 1.3305 × 10⁻³ |
| φ StructTex within/between ratio | 1.3246 × 10⁻³ |
| Fault-geometry within/between ratio | 1.6094 × 10⁻³ |

The experiment isolates **latent-code sensitivity under fixed synthesis noise**. It does not measure variability caused by changing layer-wise synthesis-noise realizations.

---

## 3. Repository structure

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
    └── effective_parameters.json
```

### `run_conditional_stylegan_experiment.py`

Main implementation for:

- conditional StyleGAN2 training;
- checkpoint testing;
- SLoG / LoG / Canny condition construction;
- GEO and φ feature extraction;
- FD/RR, GEO KID-style MMD², PRDC, and SWD evaluation;
- synthetic-profile generation for downstream experiments;
- fixed-condition latent-sensitivity analysis.

### `slog_parameter_search.py`

Parameter-search and preprocessing-analysis program used to examine SLoG and alternative structural-prior configurations.

### `unet_regression_baseline.py`

Deterministic condition-to-seismic U-Net implementation used for:

- **U-Net-L1**;
- **U-Net-Struct**;
- paired reconstruction metrics;
- shared GEO / φ / SWD distributional evaluation.

### `seismic_edge_metrics.py`

Structural-prior characterization used for the 500-profile SLoG / LoG / Canny comparison, including:

- edge density;
- directional continuity;
- break rate;
- fragmentation;
- short-response ratio;
- local orientation entropy;
- low-coherence overlap and distance measures.

### `downstream_fault_segmentation.py`

Common U-Net fault-segmentation experiment for training on:

- real profiles;
- SLoG-conditioned synthetic profiles;
- LoG-conditioned synthetic profiles;
- Canny-conditioned synthetic profiles.

The same initialization, real validation set, and real test set are used across the four training sources.

---

## 4. Environment

The paper experiments were implemented under:

- Ubuntu 22.04 LTS
- Python 3.9
- PyTorch 1.12
- NVIDIA RTX A6000, 48 GB VRAM

The provided `requirements.txt` pins the PyTorch / torchvision versions to the CUDA 11.6 wheel combination corresponding to the paper environment and constrains the main scientific Python dependencies to compatible ranges.

### Installation

```bash
git clone https://github.com/Xiaofeng-12/slog-seismic-profile-generation.git
cd slog-seismic-profile-generation

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Conda users:

```bash
conda create -n slog-seismic python=3.9 -y
conda activate slog-seismic
pip install -r requirements.txt
```

If CUDA 11.6 wheels are not appropriate for your machine, install a compatible PyTorch build separately and then install the remaining dependencies.

---

## 5. Dataset

The experiments use the public interpreted seismic dataset released by:

> An et al. (2021), **A gigabyte interpreted seismic dataset for automatic fault recognition**, *Data in Brief*, 37, 107219.

- Data article: https://doi.org/10.1016/j.dib.2021.107219
- Dataset repository: https://doi.org/10.7910/DVN/YBYGBK

The original seismic data are **not redistributed** in this repository.

### 5.1 Synthesis partition

The paper uses the following preparation protocol:

1. Two-dimensional seismic profiles are extracted from the original 3-D seismic volume.
2. The extracted 3174 × 1537 planes are proportionally resampled to 2000 × 968 pixels.
3. The extraction produces 1,603 two-dimensional profiles.
4. A subset of 1,200 profiles is retained and ordered by source index.
5. From the ordered subset:
   - 400 profiles are sampled at regular index intervals for model development;
   - the remaining 800 profiles form the within-volume evaluation partition.

Because the model-development and evaluation profiles originate from the same 3-D seismic volume, the reported synthesis results characterize **within-volume performance**, not generalization to an independent survey.

### 5.2 Downstream fault-segmentation partition

A separate labeled set contains 500 profile-mask pairs:

```text
350 training
50 validation
100 real test
```

The interpreted masks are **not** used to train or condition the seismic synthesis models.

For the synthetic-only downstream experiment, each fixed StyleGAN2 checkpoint generates one synthetic profile for each of the 350 training-source profiles, and the generated profile is paired with the mask of its source profile.

---

## 6. Preprocessing and condition construction

The principal preprocessing sequence is:

```text
grayscale
  ↓
non-local means denoising
  ↓
structure-guided smoothing (SGS)
  ↓
automatic gain control (AGC)
  ↓
structural-prior extraction
```

Each final crop is:

```text
512 × 512
```

and is normalized to:

```text
[-1, 1]
```

The main three-channel layout is:

```text
fault,agc,edge
```

### Final preprocessing parameters

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

The complete parameter record used by the structural-prior characterization is provided in:

```text
results/effective_parameters.json
```

### Historical `glog` alias

Some historical experiment outputs and command-line configurations use:

```text
glog
```

as the internal name for the proposed SLoG implementation.

Therefore:

```text
glog == SLoG
```

in those historical records. New commands may use `slog` when supported by the current script.

---

## 7. Structural-prior characterization

Run:

```bash
python seismic_edge_metrics.py \
  --input_dir /path/to/prior_characterization_profiles \
  --output_dir ./outputs/edge_metrics \
  --device cuda:0
```

The script produces per-profile and aggregated measurements for SLoG, LoG, and Canny.

The cleaned paper-associated outputs are provided in:

```text
results/seismic_edge_metrics.csv
results/algorithm_summary.csv
results/effective_parameters.json
```

The analysis is intended to characterize structural-prior support and topology. Low-coherence overlap is used as a structural proxy and should not be interpreted as supervised fault-detection accuracy.

---

## 8. Conditional StyleGAN2

Display all options:

```bash
python run_conditional_stylegan_experiment.py --help
```

### 8.1 Train SLoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir /path/to/train_400 \
  --out_dir ./outputs/stylegan_slog \
  --edge_type slog \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --seed 42 \
  --amp
```

If the current script retains the historical name only, use:

```bash
--edge_type glog
```

instead of `--edge_type slog`.

### 8.2 Train LoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir /path/to/train_400 \
  --out_dir ./outputs/stylegan_log \
  --edge_type log \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --seed 42 \
  --amp
```

### 8.3 Train Canny

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir /path/to/train_400 \
  --out_dir ./outputs/stylegan_canny \
  --edge_type canny \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --seed 42 \
  --amp
```

The paper trains the synthesis models for 25,000 optimization steps with batch size 16 and uses EMA checkpoints for evaluation.

---

## 9. Shared-evaluator testing

The final SLoG / LoG / Canny comparison uses a **common LoG-based GEO / φ evaluator**.

This is important:

```text
condition edge type != evaluation edge type
```

For example, SLoG testing uses:

```text
condition edge: SLoG
evaluation edge: LoG
```

### Example: SLoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode test \
  --data_dir /path/to/evaluation_800 \
  --out_dir ./outputs/test_slog \
  --ckpt_path ./checkpoints/slog_best_total_step17500.pth \
  --device cuda:0 \
  --edge_type slog \
  --eval_edge_type log \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --seeds 43,44,45,46,47 \
  --repeats 3 \
  --test_mixing_prob 0.0 \
  --eval_n 256 \
  --kid_subsets 20 \
  --prdc_ks 3 \
  --prdc_boot 200 \
  --rr_fd_trials 20 \
  --rr_fd_percentile 95
```

Use the corresponding edge type and checkpoint for LoG and Canny.

### Selected StyleGAN2 checkpoints

| Prior | Selected step |
|---|---:|
| SLoG | 17,500 |
| LoG | 23,500 |
| Canny | 24,500 |

When a checkpoint contains both `G_ema` and `G`, testing should use `G_ema`.

---

## 10. Deterministic U-Net baselines

The deterministic U-Net reuses the same condition-construction and shared-evaluation implementation from:

```text
run_conditional_stylegan_experiment.py
```

### 10.1 U-Net-L1

```bash
python unet_regression_baseline.py \
  --mode train \
  --source_code ./run_conditional_stylegan_experiment.py \
  --data_dir /path/to/train_400 \
  --out_dir ./outputs/unet_l1 \
  --edge_type slog \
  --eval_edge_type log \
  --tex_layout fault,agc,edge \
  --max_steps 25000 \
  --lambda_l1 1.0 \
  --lambda_spec 0 \
  --lambda_grad 0 \
  --lambda_grad_dir 0 \
  --lambda_st_ori 0 \
  --amp
```

### 10.2 U-Net-Struct

```bash
python unet_regression_baseline.py \
  --mode train \
  --source_code ./run_conditional_stylegan_experiment.py \
  --data_dir /path/to/train_400 \
  --out_dir ./outputs/unet_struct \
  --edge_type slog \
  --eval_edge_type log \
  --tex_layout fault,agc,edge \
  --max_steps 25000 \
  --lambda_l1 1.0 \
  --lambda_spec 0.15 \
  --lambda_grad 0.20 \
  --lambda_grad_dir 0.03 \
  --lambda_st_ori 0.02 \
  --amp
```

### 10.3 U-Net testing

```bash
python unet_regression_baseline.py \
  --mode test \
  --source_code ./run_conditional_stylegan_experiment.py \
  --ckpt_path ./checkpoints/unet_struct_ema_step25000.pth \
  --eval_pool_path /path/to/fixed_test_pool_256.pt \
  --out_dir ./outputs/unet_struct_test \
  --edge_type slog \
  --eval_edge_type log \
  --tex_layout fault,agc,edge \
  --seeds 43,44,45,46,47 \
  --repeats 3
```

Paper-associated cleaned result files:

```text
results/unet_l1_summary.json
results/unet_struct_summary.json
results/unet_l1_paired_metrics.json
results/unet_struct_paired_metrics.json
```

---

## 11. Fixed-condition latent-sensitivity analysis

The reported SLoG latent experiment uses:

```text
20 fixed conditions
50 latent codes per condition
latent seed = 46000
fixed synthesis-noise seed = 2026
```

Example:

```bash
python run_conditional_stylegan_experiment.py \
  --mode diversity \
  --ckpt_path ./checkpoints/slog_best_total_step17500.pth \
  --eval_pool_path /path/to/fixed_test_pool_256.pt \
  --out_dir ./outputs/latent_sensitivity \
  --edge_type slog \
  --eval_edge_type log \
  --tex_layout fault,agc,edge \
  --diversity_n_conditions 20 \
  --diversity_n_latents 50 \
  --diversity_seed 46000 \
  --diversity_noise_seed 2026
```

The cleaned result is provided in:

```text
results/stylegan_slog_latent_sensitivity.json
```

---

## 12. Downstream fault segmentation

The downstream experiment compares four training sources:

```text
real
SLoG synthetic
LoG synthetic
Canny synthetic
```

with the same segmentation architecture, initialization, augmentation, optimization procedure, real validation set, and real test set.

Example:

```bash
python downstream_fault_segmentation.py \
  --experiments real,slog,log,canny \
  --output_root ./outputs/downstream \
  --device cuda:0 \
  --real_train_images /path/to/real/train/images \
  --real_train_labels /path/to/real/train/labels \
  --real_val_images /path/to/real/val/images \
  --real_val_labels /path/to/real/val/labels \
  --real_test_images /path/to/real/test/images \
  --real_test_labels /path/to/real/test/labels \
  --slog_train_images /path/to/slog/images \
  --slog_train_labels /path/to/slog/labels \
  --log_train_images /path/to/log/images \
  --log_train_labels /path/to/log/labels \
  --canny_train_images /path/to/canny/images \
  --canny_train_labels /path/to/canny/labels \
  --expected_train_count 350 \
  --expected_val_count 50 \
  --expected_test_count 100 \
  --epochs 100 \
  --amp
```

The manuscript reports the following single-run real-test Dice/F1 values:

| Training source | Dice/F1 | IoU | Precision | Recall | Tolerant F1 |
|---|---:|---:|---:|---:|---:|
| Real | 0.5129 | 0.3449 | 0.4604 | 0.5787 | 0.7261 |
| SLoG synthetic | 0.5198 | 0.3511 | 0.4500 | 0.6152 | 0.7281 |
| LoG synthetic | 0.5200 | 0.3514 | 0.4392 | 0.6374 | 0.7262 |
| Canny synthetic | 0.4883 | 0.3231 | 0.4133 | 0.5966 | 0.6950 |

These results are an application-level information-retention check. They do **not** establish a statistically significant segmentation advantage of SLoG over LoG.

---

## 13. Evaluation spaces and metrics

### GEO feature space

The GEO representation contains 28 domain-oriented attributes emphasizing interpretable structure and texture.

### φ feature space

The φ representation contains 160 dimensions:

```text
StructTex: 32
Spectrum:  64
Energy:    64
```

### FD/RR

FD/RR is a feature-space-specific normalized Fréchet-distance measure using a real-real reference distribution.

Lower values indicate closer agreement in the specified feature space.

FD/RR should **not** be interpreted as an absolute measure of geological validity.

### KID terminology

The repository reports a KID-style unbiased polynomial-kernel MMD² estimator in domain-specific feature spaces.

It is **not** ImageNet Inception KID.

### FID terminology

The reported FD values are computed from domain-specific seismic features and should **not** be interpreted as ImageNet FID.

---

## 14. Results directory

The cleaned files under `results/` provide the numerical values used in the manuscript.

| File | Content |
|---|---|
| `stylegan_slog_summary.json` | Final SLoG shared-evaluator results |
| `stylegan_log_summary.json` | Final LoG shared-evaluator results |
| `stylegan_canny_summary.json` | Final Canny shared-evaluator results |
| `unet_l1_summary.json` | U-Net-L1 distributional evaluation |
| `unet_struct_summary.json` | U-Net-Struct distributional evaluation |
| `unet_l1_paired_metrics.json` | U-Net-L1 paired reconstruction metrics |
| `unet_struct_paired_metrics.json` | U-Net-Struct paired reconstruction metrics |
| `stylegan_slog_latent_sensitivity.json` | Fixed-condition SLoG latent-sensitivity analysis |
| `algorithm_summary.csv` | Aggregated 500-profile structural-prior statistics |
| `seismic_edge_metrics.csv` | Per-profile structural-prior measurements |
| `effective_parameters.json` | Final preprocessing / prior parameters |

The cleaned public result files omit local workstation paths and other environment-specific metadata.

---

## 15. Pretrained checkpoints

Paper-associated checkpoints are distributed through:

https://github.com/Xiaofeng-12/slog-seismic-profile-generation/releases

Place downloaded checkpoints under:

```text
checkpoints/
```

The deterministic U-Net release assets use the following names:

```text
unet_l1_ema_step25000.pth
unet_struct_ema_step25000.pth
```

The selected StyleGAN2 checkpoints correspond to steps 17,500, 23,500, and 24,500 for SLoG, LoG, and Canny, respectively.

If the release contains renamed public inference checkpoints, use the exact release asset names in the `--ckpt_path` argument.

---

## 16. Reproducibility notes

- GAN training is stochastic.
- Numerical values may vary slightly across GPU architectures, CUDA versions, drivers, and library builds.
- Final synthesis comparisons should use the same shared LoG-based output evaluator.
- The final 256-profile comparison pool should not be used for checkpoint selection.
- U-Net evaluation should use EMA weights.
- Historical result files may use `glog` as an alias for SLoG.
- The fixed-condition latent experiment holds the synthesis-noise realization constant and therefore isolates latent-code sensitivity only.
- Downstream segmentation values in the paper are from one fixed training run per training source.

---

## 17. Data and code availability

The seismic dataset is publicly available from Harvard Dataverse:

https://doi.org/10.7910/DVN/YBYGBK

The source code, experiment implementation, cleaned result files, and paper-associated checkpoints are provided in this repository and its Releases page.

---

## 18. Citation

If you use this code or the associated results, please cite the manuscript:

```bibtex
@article{Yun2026ConditionDrivenSeismic,
  title   = {Condition-driven seismic synthesis with orientation-aware structural priors},
  author  = {Yun, Bensheng and Zhang, Xiaofeng and Xiang, Yang and Shen, Jie},
  journal = {Journal of Applied Geophysics},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

The BibTeX entry should be updated after publication with the final volume, pages/article number, and DOI.

---

## 19. License

This repository is released under the MIT License.

The original seismic dataset and third-party software remain subject to their respective licenses and terms of use.

---

## 20. Contact

**Bensheng Yun**  
Zhejiang University of Science and Technology  
No. 318 Liuhe Road, Hangzhou 310023, Zhejiang, China  
Email: yunbsh@zust.edu.cn  
ORCID: 0009-0003-3075-2684
