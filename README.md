# Structural Conditioning for Source-Associated Seismic Resynthesis

Official code and experiment repository for the manuscript:

> **Structural Conditioning for Source-Associated Seismic Resynthesis: A Comparative Evaluation of Edge Detection Algorithms**  
> Bensheng Yun, Xiaofeng Zhang, Yang Xiang, and Jie Shen  
> Journal of Applied Geophysics submission, 2026

This repository contains the implementation and paper-associated experiment code for comparing three structural priors—**SLoG**, **LoG**, and **Canny**—under matched conditional seismic-resynthesis pipelines.

The proposed SLoG operator is the **structure-tensor-guided smoothing-based Laplacian of Gaussian (SLoG)** prior. The study treats the representation supplied to the conditional model as the main methodological question. StyleGAN2 with FiLM is used as a common synthesis backbone rather than as the principal contribution.

> **Task framing.** The conditioning channels are extracted from corresponding observed seismic profiles. The experiments therefore evaluate **source-associated seismic resynthesis**, not unconstrained generation of independent geological realizations.

---

## 1. Main experimental design

The principal conditional layout is:

```text
fault + AGC + edge
```

where:

- `fault` is a label-free fault-likelihood channel derived from coherence reduction, local orientation change, and, in the Australian pipeline, the selected structural edge response;
- `AGC` is the automatic-gain-controlled seismic profile;
- `edge` is one of SLoG, isotropic LoG, or binary Canny.

Source-derived conditions are encoded at multiple scales and injected into the StyleGAN2-inspired generator through bounded residual FiLM modulation.

The repository also includes code for:

- three-training-seed Australian evaluation;
- final `T=1000` real-real FD/RR evaluation;
- late-window common-checkpoint ranking robustness;
- 500-profile structural-prior characterization;
- deterministic U-Net controls;
- fixed-condition latent-sensitivity analysis;
- within-survey fault segmentation;
- zero-shot Australian-to-CRACKS fault segmentation;
- paired hierarchical-bootstrap and uncertain-fault sensitivity analysis;
- an external F3 synthesis stress test using an edge-independent fault channel.

---

## 2. Main Australian results

For each prior, three generators are trained independently using training seeds:

```text
35, 39, 42
```

Final evaluation uses:

```text
evaluation base seeds = 43, 44, 45, 46, 47
repetitions per base seed = 30
evaluation runs per trained generator = 150
fixed evaluation conditions = 256
shared evaluation edge = LoG
final RR partitions T = 1000
RR percentile = 95
```

Each trained generator is first summarized over its 150 matched evaluation runs. The values below are then reported as **mean ± sample standard deviation across the three independently trained generators (n=3)**.

| Prior | SWD | GEO KID | Total phi FD/RR | StructTex | Spectrum | Energy |
|---|---:|---:|---:|---:|---:|---:|
| **SLoG** | **0.1176 ± 0.0001** | **0.5770 ± 0.2118** | **0.9150 ± 0.0848** | **1.0377 ± 0.4505** | **0.2552 ± 0.0847** | 1.4354 ± 0.1169 |
| LoG | 0.1185 ± 0.0003 | 4.1668 ± 1.9808 | 1.0898 ± 0.1480 | 3.6926 ± 0.9721 | 0.5410 ± 0.0679 | 1.1735 ± 0.1422 |
| Canny | 0.1202 ± 0.0005 | 5.5309 ± 2.7172 | 1.5245 ± 0.1934 | 8.3345 ± 3.0062 | 0.8986 ± 0.2979 | **1.0531 ± 0.0975** |

The same total-phi-FD/RR ordering is retained at training seeds 35, 39, and 42. The result is not uniform across every attribute: SWD is close between SLoG and LoG, while the Energy subspace favors Canny at the model level.

The raw final-test JSONL files used for this analysis are provided under:

```text
results/australian/
```

Each public JSONL should contain **150 unique evaluation runs**.

---

## 3. Late-window ranking robustness

To test whether the cross-prior ordering is an artifact of prior-specific selected checkpoints, the paper also evaluates checkpoints shared by all nine Australian prior-seed runs.

The manuscript analysis uses:

```text
10,000 to 20,500 optimization steps
500-step interval
22 common checkpoints
3 priors × 3 training seeds
```

Across the 22 common checkpoints, SLoG has the lowest mean:

- total phi FD/RR at 22/22 checkpoints;
- GEO KID at 22/22 checkpoints;
- StructTex FD/RR at 22/22 checkpoints;
- Spectrum FD/RR at 22/22 checkpoints.

Complementary measures are mixed: SWD is nearly split between SLoG and LoG, while Energy more often favors Canny.

Run the analysis with:

```bash
python ranking_robustness.py \
  --input /path/to/australian_training_logs_or_directory \
  --out ./outputs/ranking_robustness \
  --min-step 10000
```

The paper result is based on the 22 common checkpoints from 10,000 through 20,500 steps.

---

## 4. Repository structure

Recommended public layout:

```text
slog-seismic-profile-generation/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── run_conditional_stylegan_experiment.py
├── slog_parameter_search.py
├── seismic_edge_metrics.py
├── unet_regression_baseline.py
├── downstream_fault_segmentation.py
│
├── final_rr_multiseed_analysis.py
├── ranking_robustness.py
├── plot_prior_support_characterization_500.py
├── downstream_crosssurvey_unet.py
├── analyze_crosssurvey_bootstrap_sensitivity.py
│
└── results/
    ├── australian/
    │   ├── multiseed_manifest.csv
    │   ├── slog35.jsonl
    │   ├── slog39.jsonl
    │   ├── slog42.jsonl
    │   ├── log35.jsonl
    │   ├── log39.jsonl
    │   ├── log42.jsonl
    │   ├── canny35.jsonl
    │   ├── canny39.jsonl
    │   └── canny42.jsonl
    │
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

The older cleaned result files may be retained because they document deterministic controls, latent diagnostics, and structural-prior characterization. The `results/australian/` directory contains the raw final three-training-seed Australian evaluation records used for the current main comparison.

---

## 5. Environment

Paper experiments were implemented with:

- Ubuntu 22.04 LTS
- Python 3.9
- PyTorch 1.12
- NVIDIA RTX A6000, 48 GB VRAM

Install the repository dependencies with:

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

If the PyTorch build specified in `requirements.txt` is not appropriate for the local CUDA environment, install a compatible PyTorch build first and then install the remaining dependencies.

---

## 6. Datasets

### 6.1 Australian seismic dataset

The main Australian experiments use the public dataset described by:

> An et al. (2021), *A gigabyte interpreted seismic dataset for automatic fault recognition*, Data in Brief, 37, 107219.

- Data article: https://doi.org/10.1016/j.dib.2021.107219
- Dataset: https://doi.org/10.7910/DVN/YBYGBK

The original seismic data are not redistributed in this repository.

The synthesis preparation used in the paper is:

1. Extract 1,603 two-dimensional profiles from the source volume.
2. Randomly retain 1,200 profiles using **random seed 42**.
3. Restore/order the retained profiles by source index.
4. Select 400 profiles at regular index intervals for model development.
5. Use the remaining 800 profiles as the within-volume evaluation partition.

Because both partitions originate from the same 3-D volume, the Australian comparison is interpreted as **within-volume source-associated resynthesis**, not spatially independent or cross-survey generalization.

A separate labeled set contains:

```text
350 training pairs
50 validation pairs
100 real test pairs
```

for the within-survey segmentation experiment.

### 6.2 Netherlands F3 synthesis data

The external synthesis stress test uses cropped real F3 seismic images prepared by the data source cited in the manuscript (Choi et al., 2025). These images are used as a label-free synthesis target.

F3 synthesis uses three independent GAN training seeds:

```text
35, 39, 42
```

and the same fixed 25,000-step EMA checkpoint for every prior-seed run.

### 6.3 CRACKS expert annotations

The zero-shot downstream experiment uses the separately released CRACKS v2 expert annotations from the Netherlands North Sea F3 volume:

- https://doi.org/10.5281/zenodo.13926822

The expert subset used in the paper contains 40 matched seismic-section/mask pairs at native `255 × 701` geometry.

Primary evaluation uses:

```text
class 3 = confident fault
class 1 = confident non-fault
classes 0 and 2 = ignored
```

A sensitivity analysis additionally treats class 2 as fault.

No CRACKS image or expert label is used for Australian U-Net training, validation, threshold selection, early stopping, or parameter tuning.

---

## 7. Unified conditional StyleGAN2 program

The public implementation uses one main program:

```text
run_conditional_stylegan_experiment.py
```

instead of maintaining separate near-duplicate Australian, F3, and final-test scripts.

Display all options:

```bash
python run_conditional_stylegan_experiment.py --help
```

The program supports:

```text
--profile australian
--profile f3
```

and:

```text
--mode train
--mode test
--mode generate
--mode diversity
```

### 7.1 Australian profile

The Australian profile reproduces the 512 × 512 within-volume protocol.

Important profile defaults include:

| Parameter | Australian |
|---|---:|
| Image size | 512 |
| SGS sigma | 1.5 |
| SGS anisotropy | 2.2 |
| SGS gamma | 0.06 |
| AGC window | 31 |
| AGC p1 | 1 |
| Fault coherence weight | 0.55 |
| Fault orientation weight | 0.30 |
| Fault edge weight | 0.15 |
| Discriminator patch scales | 1.0, 0.5, 0.25 |

Example SLoG training run:

```bash
python run_conditional_stylegan_experiment.py \
  --profile australian \
  --mode train \
  --data_dir /path/to/development_400 \
  --out_dir ./outputs/australian/slog_seed35 \
  --edge_type slog \
  --seed 35 \
  --batch 16 \
  --amp
```

Repeat with training seeds 39 and 42, and replace `--edge_type` with `log` or `canny` for the alternative priors.

### 7.2 F3 profile

The F3 profile reproduces the 128 × 128 external-survey synthesis stress-test protocol.

Important profile defaults include:

| Parameter | F3 |
|---|---:|
| Image size | 128 |
| SGS sigma | 2.5 |
| SGS anisotropy | 2.5 |
| SGS gamma | 0.04 |
| AGC window | 11 |
| AGC p1 | 3 |
| Fault coherence weight | 0.6470588 |
| Fault orientation weight | 0.3529412 |
| **Fault edge weight** | **0.0** |
| Discriminator patch scales | 1.0, 0.5 |

The zero edge weight is essential: fault and AGC conditions are identical across SLoG, LoG, and Canny in the F3 stress test, so only the direct structural-prior channel changes.

Example:

```bash
python run_conditional_stylegan_experiment.py \
  --profile f3 \
  --mode train \
  --data_dir /path/to/f3_train_images \
  --out_dir ./outputs/f3/slog_seed35 \
  --edge_type slog \
  --seed 35 \
  --batch 16 \
  --amp
```

Repeat for seeds 39 and 42 and for LoG/Canny.

---

## 8. Final Australian testing

The final Australian comparison uses a shared LoG-based GEO/phi evaluator and a fixed 256-condition evaluation pool.

Example:

```bash
python run_conditional_stylegan_experiment.py \
  --profile australian \
  --mode test \
  --data_dir /path/to/evaluation_800 \
  --out_dir ./outputs/final_test/slog_seed35 \
  --ckpt_path /path/to/selected_slog_seed35_checkpoint.pth \
  --eval_pool_path /path/to/fixed_test_pool_256.pt \
  --edge_type slog \
  --eval_edge_type log \
  --seeds 43,44,45,46,47 \
  --repeats 30 \
  --eval_n 256 \
  --eval_cond_num 256 \
  --rr_fd_trials 1000 \
  --rr_fd_percentile 95 \
  --test_mixing_prob 0.0 \
  --device cuda:0
```

Important distinctions:

- training seed and evaluation seed are different concepts;
- training seeds are 35, 39, and 42;
- evaluation base seeds are 43–47;
- each base seed is repeated 30 times;
- the final Australian comparison uses `T=1000` RR partitions;
- development/checkpoint-screening evaluations retain the lower-cost `T=20` protocol unless otherwise specified;
- evaluation uses EMA generator weights when available.

---

## 9. Reproduce the Australian multi-seed summary

The nine raw final-test files are mapped by:

```text
results/australian/multiseed_manifest.csv
```

Run from the **repository root**:

```bash
python final_rr_multiseed_analysis.py \
  --manifest results/australian/multiseed_manifest.csv \
  --out_dir results/australian/summary \
  --hier_boot 10000
```

The manifest records:

```text
3 priors × 3 independent training seeds
rr_trials = 1000
```

Expected raw files:

```text
results/australian/slog35.jsonl
results/australian/slog39.jsonl
results/australian/slog42.jsonl
results/australian/log35.jsonl
results/australian/log39.jsonl
results/australian/log42.jsonl
results/australian/canny35.jsonl
results/australian/canny39.jsonl
results/australian/canny42.jsonl
```

Each JSONL should contain 150 unique matched evaluation runs.

The analysis script produces model-level and seed-level summaries together with paired hierarchical-bootstrap results.

---

## 10. Structural-prior characterization

The 500-profile characterization compares response support and topology for SLoG, LoG, and Canny.

Existing structural metrics are produced with:

```text
seismic_edge_metrics.py
```

The additional paper figure is generated with:

```bash
python plot_prior_support_characterization_500.py \
  --metrics_csv results/seismic_edge_metrics.csv \
  --params_json results/effective_parameters.json \
  --image_dir /path/to/the_same_500_profiles \
  --out_dir ./outputs/prior_support
```

The analysis includes:

- response-value distributions;
- threshold-survival support curves;
- valid-signal response density;
- fragmentation.

These measurements characterize representation support and topology. They are not supervised fault-detection accuracy measures.

---

## 11. Deterministic controls and latent sensitivity

### Deterministic U-Net controls

The existing:

```text
unet_regression_baseline.py
```

implements U-Net-L1 and U-Net-Struct controls using the same source-derived condition tensor and shared seismic evaluator.

These controls show how much source-associated agreement is recoverable deterministically from the condition tensor without a latent code or adversarial discriminator.

### Fixed-condition latent sensitivity

The unified main program includes:

```text
--mode diversity
```

The reported experiment uses:

```text
20 fixed SLoG conditions
50 shared latent codes per condition
latent seed = 46000
fixed synthesis-noise seed = 2026
```

Example:

```bash
python run_conditional_stylegan_experiment.py \
  --profile australian \
  --mode diversity \
  --data_dir /path/to/evaluation_data \
  --out_dir ./outputs/latent_sensitivity \
  --ckpt_path /path/to/slog_checkpoint.pth \
  --eval_pool_path /path/to/fixed_test_pool_256.pt \
  --edge_type slog \
  --eval_edge_type log \
  --diversity_n_conditions 20 \
  --diversity_n_latents 50 \
  --diversity_seed 46000 \
  --diversity_noise_seed 2026
```

The experiment holds the condition tensor and synthesis-noise realization fixed and therefore isolates latent-code sensitivity.

---

## 12. Within-survey fault segmentation

The existing:

```text
downstream_fault_segmentation.py
```

compares four training sources:

```text
real
SLoG synthetic
LoG synthetic
Canny synthetic
```

All four use the same segmentation architecture, initialization, augmentation, optimizer, validation set, and real test set.

The paper treats this as a controlled application-level information-retention test rather than evidence that one prior universally improves segmentation.

---

## 13. Zero-shot Australian-to-CRACKS evaluation

The external downstream experiment is implemented by:

```text
downstream_crosssurvey_unet.py
```

For each training seed 35, 39, and 42, the four U-Nets are trained using Australian data and selected only with the Australian validation set. CRACKS is used only for final testing.

Example structure:

```bash
python downstream_crosssurvey_unet.py \
  --seed 35 \
  --output_root ./outputs/cracks/seed35 \
  --device cuda:0 \
  --real_train_images /path/to/real/train/images \
  --real_train_labels /path/to/real/train/labels \
  --real_val_images /path/to/real/val/images \
  --real_val_labels /path/to/real/val/labels \
  --real_test_images /path/to/cracks/images \
  --real_test_labels /path/to/cracks/expert_masks \
  --slog_train_images /path/to/slog/train/images \
  --slog_train_labels /path/to/slog/train/labels \
  --log_train_images /path/to/log/train/images \
  --log_train_labels /path/to/log/train/labels \
  --canny_train_images /path/to/canny/train/images \
  --canny_train_labels /path/to/canny/train/labels \
  --expected_train_count 350 \
  --expected_val_count 50 \
  --expected_test_count 40 \
  --epochs 100 \
  --threshold 0.5 \
  --tolerance_radius 3 \
  --amp
```

Repeat for seeds 39 and 42.

Primary CRACKS results reported in the manuscript are:

| Training source | Dice/F1 | IoU | Precision | Recall | Tolerant F1 |
|---|---:|---:|---:|---:|---:|
| Real | 0.0954 ± 0.0139 | 0.0501 ± 0.0076 | 0.0907 ± 0.0196 | 0.1049 ± 0.0273 | 0.1184 ± 0.0152 |
| **SLoG** | **0.1504 ± 0.0269** | **0.0814 ± 0.0157** | **0.1468 ± 0.0308** | 0.1554 ± 0.0265 | **0.1833 ± 0.0357** |
| LoG | 0.1461 ± 0.0396 | 0.0791 ± 0.0233 | 0.1056 ± 0.0353 | **0.2508 ± 0.0513** | 0.1630 ± 0.0458 |
| Canny | 0.1492 ± 0.0139 | 0.0806 ± 0.0081 | 0.1117 ± 0.0082 | 0.2279 ± 0.0427 | 0.1689 ± 0.0131 |

These results show a precision-recall trade-off rather than a universal SLoG advantage.

---

## 14. CRACKS bootstrap and uncertain-label sensitivity

Post-hoc paired analysis is implemented by:

```text
analyze_crosssurvey_bootstrap_sensitivity.py
```

It reloads the saved U-Net checkpoints and evaluates:

1. the primary confident-label definition;
2. an inclusive definition in which uncertain class-2 pixels are treated as faults.

It also supports hierarchical paired bootstrap across training runs and matched CRACKS sections.

Example:

```bash
python analyze_crosssurvey_bootstrap_sensitivity.py \
  --base_code ./downstream_crosssurvey_unet.py \
  --run_dirs ./outputs/cracks/seed35,./outputs/cracks/seed39,./outputs/cracks/seed42 \
  --f3_images /path/to/cracks/images \
  --f3_labels /path/to/cracks/expert_masks \
  --bootstrap_n 10000 \
  --output_dir ./outputs/cracks/bootstrap
```

---

## 15. External F3 synthesis stress test

The F3 synthesis experiment uses the same unified StyleGAN2 code with:

```text
--profile f3
```

The common 25,000-step EMA checkpoint is evaluated for every prior-seed run.

Final model-level results are:

| Prior | SWD | GEO MMD | GEO W | phi FD | phi FD/RR | phi KID |
|---|---:|---:|---:|---:|---:|---:|
| **SLoG** | **0.1080 ± 0.0064** | **0.3036 ± 0.0253** | **1.0413 ± 0.0669** | **96.11 ± 5.70** | **2.5779 ± 0.1990** | **1.8279 ± 0.2245** |
| LoG | 0.1091 ± 0.0066 | 0.6292 ± 0.0303 | 1.6144 ± 0.0960 | 323.84 ± 12.18 | 8.6844 ± 0.5036 | 20.3779 ± 1.7458 |
| Canny | 0.1146 ± 0.0070 | 0.6668 ± 0.0193 | 2.1333 ± 0.0949 | 506.69 ± 17.00 | 13.5909 ± 0.8071 | 54.6218 ± 4.8891 |

The F3 images used for synthesis evaluation are label-free. This experiment therefore evaluates external-survey synthesis-distribution behavior rather than fault-detection accuracy or zero-shot generator generalization.

---

## 16. Evaluation spaces and terminology

### GEO

The GEO representation contains 28 domain-oriented attributes emphasizing seismic structure and texture.

### phi

The phi representation contains 160 dimensions:

```text
StructTex = 32
Spectrum  = 64
Energy    = 64
```

### FD/RR

FD/RR is the domain-feature Fréchet distance between real and generated samples normalized by a real-real reference:

```text
FD(real, generated) / Q95[FD(real subset 1, real subset 2)]
```

Lower values indicate smaller discrepancy relative to the real-real reference.

### Important terminology

- **FD is not ImageNet FID.**
- The reported **GEO KID** is a KID-style unbiased polynomial-kernel MMD² estimator in domain-specific seismic features; it is not ImageNet Inception KID.
- GEO and phi are internally defined diagnostic spaces used to compare the matched experiments and are not universal seismic-quality scales.

---

## 17. Historical `glog` alias

Some historical files use:

```text
glog
```

as the internal name for SLoG.

For those records:

```text
glog == SLoG
```

The current unified main program accepts both `slog` and the historical `glog` alias.

---

## 18. Reproducibility notes

- GAN training is stochastic; training seeds are explicitly reported.
- The final Australian model-level uncertainty is calculated across three independent GAN training seeds, not across individual evaluation resamples.
- The 256-profile final Australian evaluation pool is separate from the model-development pool and should not be used for checkpoint selection.
- Development/checkpoint-screening RR evaluation normally uses `T=20`; the final Australian comparison uses `T=1000`.
- All cross-prior final output comparisons use the same LoG-based GEO/phi evaluator.
- When a checkpoint contains both `G_ema` and `G`, evaluation uses EMA weights.
- The Australian 400/800 partitions come from the same 3-D volume and may contain spatially adjacent sections.
- The F3 synthesis stress test is label-free.
- CRACKS contains 40 expert-annotated sections and is used only as an external downstream test set.
- Numerical values may vary slightly across GPU architectures, CUDA versions, and library builds.

---

## 19. Data and code availability

Australian seismic data:

- https://doi.org/10.7910/DVN/YBYGBK

CRACKS v2:

- https://doi.org/10.5281/zenodo.13926822

Repository:

- https://github.com/Xiaofeng-12/slog-seismic-profile-generation

The source seismic datasets and CRACKS annotations are not redistributed in this repository.

---

## 20. Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{Yun2026StructuralConditioning,
  title   = {Structural Conditioning for Source-Associated Seismic Resynthesis:
             A Comparative Evaluation of Edge Detection Algorithms},
  author  = {Yun, Bensheng and Zhang, Xiaofeng and Xiang, Yang and Shen, Jie},
  journal = {Journal of Applied Geophysics},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

Update the BibTeX entry with the final DOI, volume, and article number after publication.

---

## 21. License

This repository is released under the MIT License.

The original seismic datasets and third-party software remain subject to their respective licenses and terms of use.

---

## 22. Contact

**Bensheng Yun**  
Zhejiang University of Science and Technology  
No. 318 Liuhe Road, Hangzhou 310023, Zhejiang, China  
Email: yunbsh@zust.edu.cn  
ORCID: 0009-0003-3075-2684
