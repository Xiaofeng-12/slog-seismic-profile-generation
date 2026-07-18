# SLoG-Conditioned Seismic Profile Generation

Official implementation of a conditional StyleGAN2 framework for seismic-profile generation using a structure-tensor-guided smoothing-based Laplacian of Gaussian (SLoG) prior.

**Associated paper:**  
Bensheng Yun, Xiaofeng Zhang, Yang Xiang, and Jie Shen,  
“Interpretable structural-prior-guided seismic profile generation under limited data using conditional StyleGAN2,”  
submitted to *Computers & Geosciences*.

## Overview

This repository provides the main code used to train and evaluate a conditional StyleGAN2 model for grayscale seismic-profile generation.

The framework constructs label-free geophysical condition channels directly from seismic images and injects them into the generator through a multi-scale condition encoder and feature-wise linear modulation (FiLM).

The main condition layout is:

```text
fault,agc,edge
```

where:

- `fault` is a label-free fault-probability prior derived from coherence reduction, local orientation change, and edge response;
- `agc` is the automatic-gain-controlled seismic image;
- `edge` is an SLoG, LoG, or Canny structural prior.

In the source code, the command-line value `glog` refers to the proposed SLoG implementation.

## Repository structure

```text
slog-seismic-profile-generation/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run_conditional_stylegan_experiment.py
├── slog_parameter_search.py
└── results/
    ├── slog_test_runs.jsonl
    └── slog_test_summary.json
```

### `run_conditional_stylegan_experiment.py`

Main program for:

- training the conditional StyleGAN2 model;
- testing a saved checkpoint;
- constructing geophysical condition channels;
- computing GEO and Phi feature representations;
- calculating FD/RR, KID, PRDC, SWD, and related evaluation results;
- saving generated images, checkpoints, and metric logs.

### `slog_parameter_search.py`

Program for searching and comparing preprocessing and edge-prior parameters, including SLoG, LoG, Sobel, and Canny, together with perceptual, structural, spectral, and edge-response measurements.

## Requirements

Main dependencies:

```text
torch
torchvision
numpy
scipy
scikit-image
opencv-python
pillow
tqdm
pandas
matplotlib
lpips
```

Create a Python 3.9 environment and install the dependencies with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The provided `requirements.txt` installs PyTorch 1.12.1 with its official CUDA 11.6 runtime build. This runtime can operate with a newer NVIDIA driver that reports CUDA 12.6 support.

### Tested environment

The code was developed and tested under Ubuntu 22.04 LTS using Python 3.9 and PyTorch 1.12. Experiments were conducted on a workstation equipped with an NVIDIA RTX A6000 GPU with 48 GB of VRAM and an Intel Xeon W5-3435X CPU. The NVIDIA driver and CUDA environment reported CUDA 12.6.

## Dataset

The experiments use the public seismic dataset associated with:

> An et al. (2021), “A gigabyte interpreted seismic dataset for automatic fault recognition,” *Data in Brief*, 37, 107219.

- Data article: https://doi.org/10.1016/j.dib.2021.107219
- Dataset: https://doi.org/10.7910/DVN/YBYGBK

The original dataset is not redistributed in this repository. Download it from the official data repository and comply with the original data terms.

### Dataset preparation and partitioning

Two-dimensional seismic profiles were extracted from the `3174 × 1537` planes of the original three-dimensional seismic volume and proportionally resampled to `2000 × 968` pixels.

The data preparation and partitioning procedure was:

1. A total of 1,603 two-dimensional seismic profiles were extracted using a Python-based procedure.
2. From these profiles, 1,200 images were randomly selected.
3. The selected images were reordered according to their original profile indices to preserve the overall spatial variation of the seismic sequence.
4. From the ordered set, 400 profiles were selected at regular intervals for training.
5. The remaining 800 profiles were used as the test set.

Before prior-channel construction, a sequence-dependent dynamic cropping strategy is applied because the extent of the blank region on the right side varies along the ordered profile sequence. Each final crop has a size of `512 × 512` pixels and is normalized to `[-1, 1]`.

Vertically, the cropping window is randomly sampled within the bottom 600 pixels of the original profile. Horizontally, the candidate range is adjusted according to the relative order of the profile. The initial maximum right boundary is 900 pixels, corresponding to an initial left-boundary range of `[0, 388]`; the candidate range then progressively expands to the right along the ordered sequence.

The training program reads images directly from a folder. Supported extensions are `.png`, `.jpg`, and `.bmp`, and the files are sorted by filename before loading.

Recommended folder structure:

```text
data/
├── train/
│   ├── profile_0001.png
│   ├── profile_0002.png
│   └── ...
└── test/
    ├── profile_0003.png
    ├── profile_0004.png
    └── ...
```

## Paper parameter settings

The preprocessing settings used in the paper are:

| Parameter | Value |
|---|---:|
| NLMeans strength, `h` | 0.8 |
| SGS scale, `sigma` | 1.5 |
| SGS anisotropy | 2.2 |
| SGS iterations | 1 |
| SGS diffusion step, `gamma` | 0.06 |
| AGC window | 31 |
| AGC lower percentile, `p1` | 1 |

The final edge-prior settings are:

| Method | Settings |
|---|---|
| SLoG (`glog`) | `sigma=0.4`, `anisotropy=2.1`, `nangles=8`, `sharpness=14`, `alpha=0.92` |
| LoG | `sigma=1.1` |
| Canny | `sigma=1.7`, `low=0.1`, `high=0.9` |

## Usage

Display all available arguments:

```bash
python run_conditional_stylegan_experiment.py --help
```

### Training with SLoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir ./data/train \
  --out_dir ./outputs/slog \
  --edge_type glog \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --epochs 1000 \
  --seed 42 \
  --amp
```

### Training with LoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir ./data/train \
  --out_dir ./outputs/log \
  --edge_type log \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --epochs 1000 \
  --seed 42 \
  --amp
```

### Training with Canny

```bash
python run_conditional_stylegan_experiment.py \
  --mode train \
  --data_dir ./data/train \
  --out_dir ./outputs/canny \
  --edge_type canny \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
  --epochs 1000 \
  --seed 42 \
  --amp
```

The current training code uses the first CUDA device visible to PyTorch when a GPU is available.

## Pretrained checkpoints

The pretrained checkpoints used for the paper are distributed through the repository's [GitHub Releases](https://github.com/Xiaofeng-12/slog-seismic-profile-generation/releases), rather than being stored directly in the Git repository.

Download the following files and place them in a local `checkpoints/` directory:

| Method | Release asset |
|---|---|
| SLoG | `slog_best_total_step17500.pth` |
| LoG | `log_best_total_step23500.pth` |
| Canny | `canny_best_total_step24500.pth` |

These filenames identify both the edge prior and the selected training step. The original training files may be renamed to the filenames above when they are uploaded as release assets.

## Testing the selected checkpoints

When a checkpoint contains both `G_ema` and `G`, the program uses `G_ema` for testing.

The reported test results use five base seeds (`43–47`) and three repeats per base seed, giving 15 runs for each method. In the current implementation, the three effective seeds for a base seed `s` are `s`, `s + 100000`, and `s + 200000`. Each run evaluates 256 samples drawn from the 800-image test set.

The selected checkpoints are:

| Method | Condition layout | Selected step |
|---|---|---:|
| SLoG | `fault,agc,edge` | 17,500 |
| LoG | `fault,agc,edge` | 23,500 |
| Canny | `fault,agc,edge` | 24,500 |

The following commands assume that the checkpoint files are stored in `./checkpoints/`.

### Testing SLoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode test \
  --data_dir ./data/test \
  --out_dir ./outputs/test_slog \
  --ckpt_path ./checkpoints/slog_best_total_step17500.pth \
  --device cuda:0 \
  --edge_type glog \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
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

### Testing LoG

```bash
python run_conditional_stylegan_experiment.py \
  --mode test \
  --data_dir ./data/test \
  --out_dir ./outputs/test_log \
  --ckpt_path ./checkpoints/log_best_total_step23500.pth \
  --device cuda:0 \
  --edge_type log \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
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

### Testing Canny

```bash
python run_conditional_stylegan_experiment.py \
  --mode test \
  --data_dir ./data/test \
  --out_dir ./outputs/test_canny \
  --ckpt_path ./checkpoints/canny_best_total_step24500.pth \
  --device cuda:0 \
  --edge_type canny \
  --tex_layout fault,agc,edge \
  --img_size 512 \
  --batch 16 \
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

## Condition channels

The `--tex_layout` argument accepts a comma-separated list of the following channel names:

| Channel | Description |
|---|---|
| `fault` | Label-free fault-probability prior |
| `agc` | Automatic-gain-controlled image |
| `edge` | SLoG, LoG, or Canny structural prior |
| `sgs` | Structure-guided smoothed image |
| `hp` | High-pass residual |
| `coherence` | Structure-tensor coherence |
| `ori_sin` | Sine encoding of local orientation |
| `ori_cos` | Cosine encoding of local orientation |
| `gray` | Normalized grayscale image |

Main layout:

```bash
--tex_layout fault,agc,edge
```

For an orientation-conditioned experiment, use the actual channel names:

```bash
--tex_layout fault,ori_sin,ori_cos,edge
```

## SLoG parameter search

Display all parameter-search arguments:

```bash
python slog_parameter_search.py --help
```

Run the final SLoG parameter configuration and save intermediate images:

```bash
python slog_parameter_search.py \
  --input_dir ./data/parameter_search \
  --out_dir ./outputs/slog_parameter_search \
  --grid_params glog_sigma,glog_anisotropy,glog_nangles,glog_sharpness,glog_alpha,glog_dist_mode,sigma,anisotropy,iterations,gamma,nl_h,window,p1 \
  --detector_params '{"enabled":["generalized_log"]}' \
  --save_images \
  --per_detector_mes
```

The program writes `metrics_summary_all_configs.csv` and can also save intermediate preprocessing and detector images.

## Evaluation

### GEO feature space

The GEO representation contains 28 dimensions:

- Structure: 15 dimensions;
- Texture: 6 dimensions;
- Spectrum: 4 dimensions;
- Energy: 3 dimensions.

### Phi feature space

The Phi representation contains 160 dimensions:

- StructTex: 32 dimensions;
- Spectrum: 64 dimensions;
- Energy: 64 dimensions.

The implementation calculates:

- GEO MMD and Wasserstein distance;
- GEO KID and PRDC;
- Phi-space Fréchet distance and FD/RR;
- Phi KID and PRDC;
- multi-scale SWD.

During periodic training evaluation, 256 conditions are fixed to improve comparability across checkpoints. KID is calculated from 20 repeated subset estimates. The reported PRDC results use `k = 3`, with 200 repeated resampling calculations and 95% confidence intervals.

The reported KID values are KID-style unbiased polynomial-kernel MMD² estimates computed in the domain-specific GEO and Phi feature spaces, rather than in an Inception feature space.

FD in this repository is computed from domain-specific Phi features and should not be interpreted as ImageNet FID.

## Output files

Training outputs are saved directly in the selected `--out_dir`, including:

```text
train.log
metrics_train.jsonl
metrics_eval.jsonl
checkpoint_<step>.pth
sample_<step>.png
best_structure.pth
best_total.pth
eval_geo_radar_step<step>.npz
```

Testing outputs include:

```text
test.log
metrics_test_runs.jsonl
metrics_test_summary.json
test_fake_seed<seed>.png
test_real_seed<seed>.png
single_pairs/
```

## Reproducibility notes

GAN training is stochastic, and exact numerical results may vary across GPU models, CUDA runtimes, drivers, and PyTorch versions.

For the final paper-associated release, the repository should provide:

- the final training and test image lists;
- the three selected checkpoint files through GitHub Releases;
- the raw metric output files used to construct the paper tables.

## Citation

Please cite the associated paper when using this code.

```bibtex
@article{YunSLoGSeismic,
  title   = {Interpretable structural-prior-guided seismic profile generation under limited data using conditional StyleGAN2},
  author  = {Yun, Bensheng and Zhang, Xiaofeng and Xiang, Yang and Shen, Jie},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## License

This repository is released under the [MIT License](LICENSE).

Third-party libraries and the original seismic dataset are subject to their own licenses and terms of use.

## Contact

For questions about the code or the associated paper, please contact:

**Bensheng Yun**  
Zhejiang University of Science and Technology  
No. 318 Liuhe Road, Hangzhou 310023, Zhejiang, China  
Email: yunbsh@zust.edu.cn  
ORCID: 0009-0003-3075-2684
