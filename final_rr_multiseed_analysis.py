"""
final_rr_multiseed_analysis.py

Purpose
-------
Post-hoc statistical analysis for the Australian multi-training-seed experiment.

This script intentionally separates TWO roles of RR-FD:

1) Training-time RR-FD
   - Keep the original training protocol unchanged (e.g. rr_fd_trials=20).
   - It may participate in adaptive loss weighting and checkpoint screening.
   - DO NOT use this script to alter an already running training job.

2) Final-reporting RR-FD
   - Re-run the final selected checkpoints with the original test() routine using:
         --rr_fd_trials 1000
         --rr_fd_percentile 95
     on the SAME fixed evaluation pool and the SAME matched evaluation seeds.
   - This is the PRIMARY FD/RR result because each numerator and denominator are
     calculated in the same per-run phi standardization coordinate system.
   - The JSONL files from those final tests are summarized here.

In addition, this script can compute an INDEPENDENT fixed-pool RR stability
diagnostic:
    fixed real pool -> shared LoG phi160 -> standardize on the full fixed pool
    -> 1000 real-real splits -> q95 -> 10,000 bootstrap resamples of q95.

IMPORTANT:
The fixed-pool q95 is a stability/sensitivity diagnostic. It is NOT used as the
primary denominator for the per-run phi FD values produced by the current
test() function, because test() fits phi standardization separately within each
evaluation run.

Expected final experiment structure
-----------------------------------
For example, 3 independent GAN training seeds per prior:
    SLoG: 35, 39, 42
    LoG : 35, 39, 42
    Canny:35, 39, 42

Each selected checkpoint should be re-tested using the same:
    eval_pool_path
    eval_edge_type=log
    eval_cond_num=256
    eval_n=256
    seeds=43,44,45,46,47
    repeats=30
    rr_fd_trials=1000
    rr_fd_percentile=95

The current test() uses:
    run_seed = base_seed + repeat_index * 100000
so rows can be paired by the saved "seed" field.

Manifest CSV
------------
Create a CSV such as:

prior,training_seed,jsonl,rr_trials
SLoG,35,/path/slog_seed35/metrics_test_runs.jsonl,1000
SLoG,39,/path/slog_seed39/metrics_test_runs.jsonl,1000
SLoG,42,/path/slog_seed42/metrics_test_runs.jsonl,1000
LoG,35,/path/log_seed35/metrics_test_runs.jsonl,1000
LoG,39,/path/log_seed39/metrics_test_runs.jsonl,1000
LoG,42,/path/log_seed42/metrics_test_runs.jsonl,1000
Canny,35,/path/canny_seed35/metrics_test_runs.jsonl,1000
Canny,39,/path/canny_seed39/metrics_test_runs.jsonl,1000
Canny,42,/path/canny_seed42/metrics_test_runs.jsonl,1000

Examples
--------
A) Write a manifest template:
python final_rr_multiseed_analysis.py \
    --write_manifest_template ./multiseed_manifest.csv

B) Compute fixed-pool RR stability diagnostic only:
python final_rr_multiseed_analysis.py \
    --pool /path/fixed_test_pool_256.pt \
    --out_dir ./final_rr_analysis \
    --device cuda:0 \
    --rr_trials 1000 \
    --rr_boot 10000

C) Summarize final test JSONLs only:
python final_rr_multiseed_analysis.py \
    --manifest ./multiseed_manifest.csv \
    --out_dir ./final_rr_analysis \
    --hier_boot 10000

D) Do both:
python final_rr_multiseed_analysis.py \
    --pool /path/fixed_test_pool_256.pt \
    --manifest ./multiseed_manifest.csv \
    --out_dir ./final_rr_analysis \
    --device cuda:0 \
    --rr_trials 1000 \
    --rr_boot 10000 \
    --hier_boot 10000

Outputs
-------
fixed_pool_phi160.npy
fixed_pool_phi160_standardized.npy
fixed_pool_phi_valid_mask.npy
fixed_pool_rr_splits.csv
fixed_pool_rr_baseline.json
seed_level_summary.csv
model_level_summary.csv
paired_differences_seed_level.csv
hierarchical_bootstrap.csv
all_runs_with_sensitivity_ratios.csv
analysis_metadata.json

Dependencies
------------
numpy
pandas
torch
scipy
Pillow
scikit-image (recommended; falls back to Gaussian denoising if unavailable)

This file contains the exact shared-LoG phi construction needed for the fixed
pool stability analysis, matching the current Australian evaluator definitions:
    SGS sigma=1.5
    SGS anisotropy=2.2
    SGS iterations=1
    SGS gamma=0.06
    NLMeans h=0.8
    AGC window=31
    LoG sigma=1.1
    phi = StructTex(32) + Spectrum(64) + Energy(64) = 160D
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.linalg import sqrtm

try:
    from skimage import restoration
    _HAVE_SKIMAGE = True
except Exception:
    restoration = None
    _HAVE_SKIMAGE = False


# =============================================================================
# Shared evaluator definitions copied from the current experiment code
# =============================================================================

PHI_LAYOUT = {
    "StructTex": slice(0, 32),
    "Spectrum": slice(32, 96),
    "Energy": slice(96, 160),
}
PHI_DIM = 160


def call_with_channel_arg(func, *args, channel_axis=None, multichannel=False, **kwargs):
    try:
        return func(*args, channel_axis=channel_axis, **kwargs)
    except TypeError:
        return func(*args, multichannel=multichannel, **kwargs)


def estimate_sigma_compat(img: np.ndarray) -> float:
    if _HAVE_SKIMAGE:
        try:
            return float(call_with_channel_arg(
                restoration.estimate_sigma,
                img,
                channel_axis=None,
                multichannel=False,
            ))
        except Exception:
            pass
    return float(np.median(np.abs(img - np.median(img))) * 1.4826)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        return (
            img[..., 0] * 0.2989
            + img[..., 1] * 0.5870
            + img[..., 2] * 0.1140
        )
    return img


def normalize_map(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(m))
    mx = float(np.max(m))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx - mn < 1e-12):
        return np.zeros_like(m, dtype=np.float32)
    m = (m - mn) / (mx - mn + 1e-12)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def denoise_nlmeans(
    img: np.ndarray,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: float = 0.08,
) -> np.ndarray:
    if _HAVE_SKIMAGE:
        sigma_est = estimate_sigma_compat(img)
        sigma_val = float(np.mean(sigma_est)) if np.ndim(sigma_est) else float(sigma_est)
        den = call_with_channel_arg(
            restoration.denoise_nl_means,
            img,
            h=h * sigma_val,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=True,
            channel_axis=None,
            multichannel=False,
            preserve_range=True,
        )
        return np.clip(den, 0.0, 1.0)
    return np.clip(gaussian_filter(img, sigma=0.8), 0.0, 1.0)


def _torch_construct_diffusion_tensor_from_ori(
    coh: torch.Tensor,
    ori: torch.Tensor,
    lambda_parallel: float = 1.0,
    lambda_perp: float = 1e-3,
    coherence_influence: float = 1.0,
    eps: float = 1e-12,
):
    orig_shape = ori.shape
    orig_ndim = len(orig_shape)

    if ori.dim() == 2:
        t_ori = ori.unsqueeze(0).unsqueeze(0)
    else:
        t_ori = ori

    if coh.dim() == 2:
        t_coh = coh.unsqueeze(0).unsqueeze(0)
    else:
        t_coh = coh

    vx = torch.cos(t_ori)
    vy = torch.sin(t_ori)

    lam_par = float(lambda_parallel)
    lam_per = float(lambda_perp)
    lam_per_map = lam_per * (1.0 - coherence_influence * t_coh)
    lam_per_map = torch.clamp(lam_per_map, min=1e-8)

    Dxx = lam_par * (vx * vx) + lam_per_map * (vy * vy)
    Dyy = lam_par * (vy * vy) + lam_per_map * (vx * vx)
    Dxy = (lam_par - lam_per_map) * (vx * vy)

    if orig_ndim == 2:
        return (
            Dxx.squeeze(0).squeeze(0),
            Dxy.squeeze(0).squeeze(0),
            Dyy.squeeze(0).squeeze(0),
        )
    return Dxx, Dxy, Dyy


def weickert_tensor_diffusion_torch(
    img: np.ndarray,
    coherence: np.ndarray,
    orientation: np.ndarray,
    niter: int = 1,
    gamma: float = 0.06,
    lambda_parallel: float = 1.0,
    lambda_perp: float = 1e-3,
    coherence_influence: float = 1.0,
    device: torch.device = None,
    return_tensor_components: bool = False,
    prefer_fp16: bool = False,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_img = (
        torch.from_numpy(img.astype(np.float32))
        .to(device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    t_coh = (
        torch.from_numpy(coherence.astype(np.float32))
        .to(device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    t_ori = (
        torch.from_numpy(orientation.astype(np.float32))
        .to(device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    use_fp16 = prefer_fp16 and (device.type == "cuda")
    dtype = torch.float16 if use_fp16 else torch.float32
    t_img = t_img.to(dtype=dtype)
    t_coh = t_coh.to(dtype=dtype)
    t_ori = t_ori.to(dtype=dtype)

    kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-0.5], [0.0], [0.5]]]], device=device, dtype=dtype)

    I = t_img
    Dxx = Dxy = Dyy = None

    for _ in range(int(niter)):
        Dxx, Dxy, Dyy = _torch_construct_diffusion_tensor_from_ori(
            t_coh,
            t_ori,
            lambda_parallel=lambda_parallel,
            lambda_perp=lambda_perp,
            coherence_influence=coherence_influence,
        )
        Dxx = Dxx.to(device=device, dtype=dtype)
        Dxy = Dxy.to(device=device, dtype=dtype)
        Dyy = Dyy.to(device=device, dtype=dtype)

        if Dxx.dim() == 2:
            Dxx = Dxx.unsqueeze(0).unsqueeze(0)
            Dxy = Dxy.unsqueeze(0).unsqueeze(0)
            Dyy = Dyy.unsqueeze(0).unsqueeze(0)

        I_p_x = F.pad(I, (1, 1, 0, 0), mode="replicate")
        gx = F.conv2d(I_p_x, kx, padding=0)

        I_p_y = F.pad(I, (0, 0, 1, 1), mode="replicate")
        gy = F.conv2d(I_p_y, ky, padding=0)

        Jx = Dxx * gx + Dxy * gy
        Jy = Dxy * gx + Dyy * gy

        Jx_p = F.pad(Jx, (1, 1, 0, 0), mode="replicate")
        dJx_dx = F.conv2d(Jx_p, kx, padding=0)

        Jy_p = F.pad(Jy, (0, 0, 1, 1), mode="replicate")
        dJy_dy = F.conv2d(Jy_p, ky, padding=0)

        div = dJx_dx + dJy_dy
        I = I + float(gamma) * div

        if use_fp16:
            I = torch.clamp(I, -1e3, 1e3)
        I = torch.nan_to_num(I, nan=0.0, posinf=1e9, neginf=-1e9)

    out = I.squeeze(0).squeeze(0).to(dtype=torch.float32, device="cpu").numpy()

    if return_tensor_components:
        def _to_np(t):
            if isinstance(t, torch.Tensor):
                t = t.squeeze().detach().to(dtype=torch.float32, device="cpu").numpy()
            return np.array(t, dtype=np.float32)

        return out, _to_np(Dxx), _to_np(Dxy), _to_np(Dyy)

    return out


def structure_guided_smoothing(
    img: np.ndarray,
    sigma: float = 1.5,
    anisotropy: float = 2.2,
    iterations: int = 1,
    base_kappa: float = 10.0,
    gamma: float = 0.06,
    option: int = 1,
    lambda_perp_base: float = 1e-3,
    lambda_par_base: float = 1.0,
    coherence_influence: float = 1.0,
    device: torch.device = None,
    prefer_fp16: bool = False,
):
    # anisotropy/base_kappa/option are retained for compatibility with the
    # current experiment signature; current implementation uses the tensor
    # diffusion parameters below.
    gradx = ndimage.sobel(img, axis=1).astype(np.float32)
    grady = ndimage.sobel(img, axis=0).astype(np.float32)

    Jxx = gaussian_filter(gradx * gradx, sigma=sigma)
    Jxy = gaussian_filter(gradx * grady, sigma=sigma)
    Jyy = gaussian_filter(grady * grady, sigma=sigma)

    tmp = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2))
    lambda1 = 0.5 * (Jxx + Jyy + tmp)
    lambda2 = 0.5 * (Jxx + Jyy - tmp)

    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-12)
    coherence = np.clip(coherence, 0.0, 1.0)

    orientation = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy + 1e-12))

    res = weickert_tensor_diffusion_torch(
        img.astype(np.float32),
        coherence.astype(np.float32),
        orientation.astype(np.float32),
        niter=int(iterations),
        gamma=float(gamma),
        lambda_parallel=float(lambda_par_base),
        lambda_perp=float(lambda_perp_base),
        coherence_influence=coherence_influence,
        device=device,
        prefer_fp16=prefer_fp16,
        return_tensor_components=True,
    )

    img_sm, Dxx_np, Dxy_np, Dyy_np = res
    img_sm = img_sm - float(np.nanmin(img_sm))
    mx = float(np.nanmax(img_sm))
    if mx > 0:
        img_sm = img_sm / mx
    else:
        img_sm = img_sm * 0.0

    return (
        img_sm.astype(np.float32),
        coherence.astype(np.float32),
        orientation.astype(np.float32),
        np.asarray(Dxx_np, dtype=np.float32),
        np.asarray(Dxy_np, dtype=np.float32),
        np.asarray(Dyy_np, dtype=np.float32),
    )


def agc_local_rms(
    img: np.ndarray,
    window: int = 31,
    eps: float = 1e-8,
    p_low: int = 1,
) -> np.ndarray:
    p_low = int(np.clip(int(p_low), 0, 100))
    p_high = 100 - p_low

    img2 = img ** 2
    kernel = np.ones((window, window), dtype=np.float64) / (window * window)
    local_mean_sq = ndimage.convolve(img2, kernel, mode="reflect")
    local_rms = np.sqrt(local_mean_sq + eps)
    out = img / (local_rms + eps)

    p_low_val, p_high_val = np.percentile(out, (float(p_low), float(p_high)))
    if p_high_val - p_low_val <= 1e-12:
        p_low_val, p_high_val = np.min(out), np.max(out)

    out = (out - p_low_val) / (p_high_val - p_low_val + 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_image_once(
    img: np.ndarray,
    sgs_params: Dict[str, Any],
    agc_params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    gray = to_gray(img).astype(np.float32)
    band = gray.copy().astype(np.float32)

    nl_h = float(sgs_params.get("nl_h", 0.8))
    nl = denoise_nlmeans(
        band,
        patch_size=5,
        patch_distance=6,
        h=nl_h,
    ).astype(np.float32)

    sgs_res, coherence_map, orientation_map, Dxx_map, Dxy_map, Dyy_map = (
        structure_guided_smoothing(
            nl,
            sigma=float(sgs_params.get("sigma", 1.5)),
            anisotropy=float(sgs_params.get("anisotropy", 2.2)),
            iterations=int(sgs_params.get("iterations", 1)),
            base_kappa=float(sgs_params.get("base_kappa", 10.0)),
            gamma=float(sgs_params.get("gamma", 0.06)),
            device=sgs_params.get("device", None),
            prefer_fp16=sgs_params.get("prefer_fp16", False),
        )
    )

    agc_res = agc_local_rms(
        sgs_res,
        window=int(agc_params.get("window", 31)),
        eps=float(agc_params.get("eps", 1e-8)),
        p_low=int(agc_params.get("p1", 1)),
    ).astype(np.float32)

    return {
        "gray": gray,
        "sgs": sgs_res.astype(np.float32),
        "coherence": coherence_map.astype(np.float32),
        "orientation": orientation_map.astype(np.float32),
        "agc": agc_res,
        "Dxx": Dxx_map.astype(np.float32),
        "Dxy": Dxy_map.astype(np.float32),
        "Dyy": Dyy_map.astype(np.float32),
    }


def compute_log_edge_from_preprocessed(
    pp: Dict[str, np.ndarray],
    det_params: Dict[str, Any],
) -> np.ndarray:
    agc = pp["agc"]
    sigma = float(det_params.get("log_sigma", 1.1))
    log_resp = gaussian_laplace(agc, sigma=sigma)
    return normalize_map(np.abs(log_resp).astype(np.float32))


def _to_gray01_from_tensor(imgs: torch.Tensor) -> np.ndarray:
    x = imgs.detach().float().cpu()
    if x.dim() != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(x.shape)}")
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    x = x[:, 0].numpy()
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def phi_features_single(
    gray01: np.ndarray,
    sgs_params: Dict[str, Any],
    agc_params: Dict[str, Any],
    det_params: Dict[str, Any],
    n_spec: int = 64,
    n_hist: int = 64,
) -> np.ndarray:
    pp = preprocess_image_once(gray01, sgs_params, agc_params)
    coh = pp["coherence"].astype(np.float32)
    ori = pp["orientation"].astype(np.float32)
    agc = np.clip(pp["agc"].astype(np.float32), 0.0, 1.0)

    # Final Australian shared evaluator = LoG.
    edge = compute_log_edge_from_preprocessed(pp, det_params)

    coh_mean = float(coh.mean())
    coh_std = float(coh.std())
    coh_q = np.percentile(coh, [10, 50, 90]).astype(np.float32)

    ori_clip = np.clip(ori, -np.pi / 2, np.pi / 2)
    sin_m = float(np.sin(ori_clip).mean())
    cos_m = float(np.cos(ori_clip).mean())

    edge_thr = float(det_params.get("edge_thr_eval", 0.05))
    edge_density = float((edge > edge_thr).mean())
    edge_mean = float(edge.mean())
    edge_q = np.percentile(edge, [50, 90]).astype(np.float32)

    gx = np.gradient(agc, axis=1)
    gy = np.gradient(agc, axis=0)
    gmag = np.sqrt(gx * gx + gy * gy + 1e-12).astype(np.float32)
    g_mean = float(gmag.mean())
    g_std = float(gmag.std())

    lap = gaussian_laplace(agc, sigma=1.0).astype(np.float32)
    lap_energy = float(np.mean(np.abs(lap)))

    base_16 = np.array([
        coh_mean,
        coh_std,
        coh_q[0],
        coh_q[1],
        coh_q[2],
        sin_m,
        cos_m,
        edge_density,
        edge_mean,
        edge_q[0],
        edge_q[1],
        g_mean,
        g_std,
        lap_energy,
        float(np.percentile(gmag, 90)),
        float(np.percentile(agc, 90)),
    ], dtype=np.float32)

    extra_16 = np.array([
        float(np.percentile(coh, 25)),
        float(np.percentile(coh, 75)),
        float(np.mean(np.abs(ori_clip))),
        float(np.std(ori_clip)),
        float(np.percentile(edge, 10)),
        float(np.percentile(edge, 95)),
        float(np.percentile(gmag, 50)),
        float(np.percentile(gmag, 95)),
        float(np.mean(gx * gx)),
        float(np.mean(gy * gy)),
        float(np.mean(np.abs(gx))),
        float(np.mean(np.abs(gy))),
        float(np.percentile(agc, 10)),
        float(np.percentile(agc, 50)),
        float(np.mean(np.abs(lap))),
        float(np.std(lap)),
    ], dtype=np.float32)

    struct_tex = np.concatenate([base_16, extra_16], axis=0).astype(np.float32)

    x = agc - agc.mean()
    spec = np.fft.rfft(x, axis=0)
    amp = (np.abs(spec).mean(axis=1) + 1e-12).astype(np.float32)

    if amp.shape[0] >= n_spec:
        spec64 = amp[:n_spec]
    else:
        spec64 = np.pad(amp, (0, n_spec - amp.shape[0]), mode="constant")
    spec64 = spec64 / (spec64.sum() + 1e-12)

    hist, _ = np.histogram(agc, bins=n_hist, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)

    phi = np.concatenate([struct_tex, spec64, hist], axis=0).astype(np.float32)
    if phi.shape[0] != PHI_DIM:
        raise ValueError(f"Unexpected phi dimension {phi.shape[0]}; expected {PHI_DIM}")
    return phi


def extract_fixed_pool_phi(
    real_crops: torch.Tensor,
    device: torch.device,
    progress_every: int = 16,
) -> np.ndarray:
    if real_crops.dim() != 4:
        raise ValueError(f"Expected fixed pool NCHW tensor, got {tuple(real_crops.shape)}")

    gray = _to_gray01_from_tensor(real_crops)

    sgs_params = {
        "sigma": 1.5,
        "anisotropy": 2.2,
        "iterations": 1,
        "base_kappa": 10.0,
        "gamma": 0.06,
        "device": device,
        # Final reporting should use stable FP32 preprocessing.
        "prefer_fp16": False,
        "nl_h": 0.8,
    }
    agc_params = {
        "window": 31,
        "eps": 1e-8,
        "p1": 1,
    }
    det_params = {
        "log_sigma": 1.1,
        "edge_thr_eval": 0.05,
    }

    feats = []
    n = gray.shape[0]
    for i in range(n):
        feats.append(
            phi_features_single(
                gray[i],
                sgs_params=sgs_params,
                agc_params=agc_params,
                det_params=det_params,
            )
        )
        if progress_every > 0 and ((i + 1) % progress_every == 0 or (i + 1) == n):
            print(f"[phi] {i + 1}/{n}", flush=True)

    return np.stack(feats, axis=0).astype(np.float64)


# =============================================================================
# FD / RR statistics
# =============================================================================

def standardize_real_only(
    real_feat: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(real_feat, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != PHI_DIM:
        raise ValueError(f"Expected (N,{PHI_DIM}) real phi, got {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError("Real phi contains NaN/Inf")

    mu = R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True)
    valid = np.isfinite(sd[0]) & (sd[0] > float(eps))
    if not np.any(valid):
        raise ValueError("No valid phi dimensions")

    scale = np.where(valid[None, :], sd + float(eps), 1.0)
    Rn = (R - mu) / scale
    Rn[:, ~valid] = 0.0
    return Rn, valid


def frechet_distance(
    mu1,
    cov1,
    mu2,
    cov2,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    cov1 = np.atleast_2d(cov1).astype(np.float64)
    cov2 = np.atleast_2d(cov2).astype(np.float64)

    cov1 = cov1 + np.eye(cov1.shape[0]) * eps
    cov2 = cov2 + np.eye(cov2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    val = float(diff.dot(diff) + np.trace(cov1 + cov2 - 2.0 * covmean))
    return val


def _fd_matrix(A: np.ndarray, B: np.ndarray) -> float:
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"Invalid FD matrices: {A.shape}, {B.shape}")
    if A.shape[1] == 0:
        return float("nan")
    return frechet_distance(
        A.mean(axis=0),
        np.cov(A, rowvar=False),
        B.mean(axis=0),
        np.cov(B, rowvar=False),
    )


def compute_rr_split_values(
    phi_real_standardized: np.ndarray,
    valid_mask: np.ndarray,
    trials: int = 1000,
    min_per_split: int = 64,
    seed: int = 20260829,
) -> Dict[str, np.ndarray]:
    R = np.asarray(phi_real_standardized, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)

    N = R.shape[0]
    if N < 2 * int(min_per_split):
        raise ValueError(
            f"Pool has N={N}, but requires >= {2 * int(min_per_split)} "
            f"for min_per_split={min_per_split}"
        )

    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    out = {k: [] for k in ["Total", "StructTex", "Spectrum", "Energy"]}

    for t in range(int(trials)):
        idx = rng.permutation(N)

        nA = max(int(min_per_split), N // 2)
        nB = N - nA
        if nB < int(min_per_split):
            nB = int(min_per_split)
            nA = N - nB

        A = R[idx[:nA]]
        B = R[idx[nA:nA + nB]]

        out["Total"].append(_fd_matrix(A[:, valid], B[:, valid]))

        for name, sl in PHI_LAYOUT.items():
            group_valid = valid[sl]
            out[name].append(
                _fd_matrix(
                    A[:, sl][:, group_valid],
                    B[:, sl][:, group_valid],
                )
            )

        if (t + 1) % max(1, int(trials) // 10) == 0:
            print(f"[RR] {t + 1}/{trials}", flush=True)

    return {
        k: np.asarray(v, dtype=np.float64)
        for k, v in out.items()
    }


def bootstrap_quantile(
    values: np.ndarray,
    q: float = 95.0,
    n_boot: int = 10000,
    ci: float = 95.0,
    seed: int = 20260830,
) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("Too few finite RR values for bootstrap")

    point = float(np.percentile(x, q))
    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    boot = np.empty(int(n_boot), dtype=np.float64)

    n = x.size
    for b in range(int(n_boot)):
        sample = x[rng.randint(0, n, size=n)]
        boot[b] = np.percentile(sample, q)

    alpha = (100.0 - float(ci)) / 2.0
    lo = float(np.percentile(boot, alpha))
    hi = float(np.percentile(boot, 100.0 - alpha))

    return {
        "q_percentile": float(q),
        "threshold": point,
        "bootstrap_ci_percent": float(ci),
        "bootstrap_lo": lo,
        "bootstrap_hi": hi,
        "rr_mean": float(np.mean(x)),
        "rr_std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "rr_median": float(np.median(x)),
        "n_rr_trials": int(x.size),
        "n_bootstrap": int(n_boot),
    }


# =============================================================================
# JSONL multi-seed aggregation
# =============================================================================

DEFAULT_METRICS = [
    "swd_total",
    "geo_mmd_total",
    "geo_w_total",
    "geo_kid_mean",
    "phi_fd_total",
    "phi_fd_structtex",
    "phi_fd_spectrum",
    "phi_fd_energy",
    "rr_fd_ratio_total",
    "rr_fd_ratio_structtex",
    "rr_fd_ratio_spectrum",
    "rr_fd_ratio_energy",
    "phi_kid_mean",
]


def canonical_prior(x: str) -> str:
    t = str(x).strip().lower()
    if t in ("slog", "glog"):
        return "SLoG"
    if t == "log":
        return "LoG"
    if t == "canny":
        return "Canny"
    return str(x).strip()


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def load_manifest(manifest_path: Path, min_rr_trials: int = 1000) -> pd.DataFrame:
    m = pd.read_csv(manifest_path)
    required = {"prior", "training_seed", "jsonl"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    if "rr_trials" not in m.columns:
        raise ValueError(
            "Manifest must contain rr_trials. "
            "For primary final reporting, re-run test with rr_fd_trials=1000 "
            "and write rr_trials=1000 for each row."
        )

    m = m.copy()
    m["prior"] = m["prior"].map(canonical_prior)
    m["training_seed"] = m["training_seed"].astype(int)
    m["rr_trials"] = m["rr_trials"].astype(int)

    bad = m[m["rr_trials"] < int(min_rr_trials)]
    if len(bad):
        raise ValueError(
            "Some manifest rows were produced with too few RR trials for the "
            "primary final analysis:\n" + bad.to_string(index=False)
        )

    return m


def collect_runs(
    manifest: pd.DataFrame,
    fixed_rr_baseline: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    frames = []

    for _, spec in manifest.iterrows():
        path = Path(str(spec["jsonl"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        df = read_jsonl(path)
        if "seed" not in df.columns:
            raise ValueError(f"{path} does not contain saved evaluation run seed")

        df = df.copy()
        df["prior"] = str(spec["prior"])
        df["training_seed"] = int(spec["training_seed"])
        df["rr_trials_final"] = int(spec["rr_trials"])
        df["source_jsonl"] = str(path)

        # Sensitivity-only ratios using ONE fixed-pool q95.
        # These are NOT the primary ratios because current test() standardizes
        # phi separately within each run.
        if fixed_rr_baseline is not None:
            mapping = {
                "Total": "phi_fd_total",
                "StructTex": "phi_fd_structtex",
                "Spectrum": "phi_fd_spectrum",
                "Energy": "phi_fd_energy",
            }
            for group, fd_col in mapping.items():
                if fd_col in df.columns and group in fixed_rr_baseline:
                    denom = float(fixed_rr_baseline[group]["threshold"])
                    df[f"fixedpool_sensitivity_ratio_{group.lower()}"] = (
                        pd.to_numeric(df[fd_col], errors="coerce") / (denom + 1e-12)
                    )

        frames.append(df)

    all_runs = pd.concat(frames, ignore_index=True)
    return all_runs


def validate_matched_run_seeds(all_runs: pd.DataFrame) -> None:
    # Within each training seed, all three priors should have the same run seeds.
    for train_seed, g in all_runs.groupby("training_seed"):
        seed_sets = {}
        for prior, gp in g.groupby("prior"):
            seed_sets[prior] = set(pd.to_numeric(gp["seed"], errors="raise").astype(int))

        if len(seed_sets) < 2:
            continue

        reference_prior = sorted(seed_sets)[0]
        reference = seed_sets[reference_prior]

        for prior, sset in seed_sets.items():
            if sset != reference:
                only_ref = sorted(reference - sset)[:10]
                only_cur = sorted(sset - reference)[:10]
                raise ValueError(
                    f"Evaluation run seeds are not matched for training_seed={train_seed}: "
                    f"{reference_prior} vs {prior}; "
                    f"missing_in_{prior}={only_ref}, extra_in_{prior}={only_cur}"
                )


def available_metrics(all_runs: pd.DataFrame) -> List[str]:
    metrics = []
    for m in DEFAULT_METRICS:
        if m in all_runs.columns:
            vals = pd.to_numeric(all_runs[m], errors="coerce")
            if np.isfinite(vals.to_numpy(dtype=float)).any():
                metrics.append(m)
    return metrics


def seed_level_summary(
    all_runs: pd.DataFrame,
    metrics: List[str],
) -> pd.DataFrame:
    rows = []
    for (prior, training_seed), g in all_runs.groupby(["prior", "training_seed"], sort=True):
        row = {
            "prior": prior,
            "training_seed": int(training_seed),
            "n_eval_runs": int(len(g)),
            "run_seed_min": int(pd.to_numeric(g["seed"]).min()),
            "run_seed_max": int(pd.to_numeric(g["seed"]).max()),
        }
        for metric in metrics:
            x = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            row[f"{metric}_mean"] = float(np.mean(x)) if x.size else float("nan")
            row[f"{metric}_std_within"] = (
                float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def model_level_summary(
    seed_summary: pd.DataFrame,
    metrics: List[str],
) -> pd.DataFrame:
    rows = []
    for prior, g in seed_summary.groupby("prior", sort=True):
        row = {
            "prior": prior,
            "n_training_seeds": int(len(g)),
            "training_seeds": ",".join(str(int(x)) for x in sorted(g["training_seed"])),
        }

        for metric in metrics:
            col = f"{metric}_mean"
            x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            row[f"{metric}_model_mean"] = float(np.mean(x)) if x.size else float("nan")
            row[f"{metric}_model_sd"] = (
                float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
            )

        rows.append(row)
    return pd.DataFrame(rows)


def paired_seed_level_differences(
    all_runs: pd.DataFrame,
    metrics: List[str],
    pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    rows = []

    for A, B in pairs:
        for train_seed in sorted(set(all_runs["training_seed"])):
            ga = all_runs[
                (all_runs["prior"] == A)
                & (all_runs["training_seed"] == train_seed)
            ]
            gb = all_runs[
                (all_runs["prior"] == B)
                & (all_runs["training_seed"] == train_seed)
            ]
            if ga.empty or gb.empty:
                continue

            ma = ga.set_index("seed")
            mb = gb.set_index("seed")
            common = sorted(set(ma.index) & set(mb.index))
            if not common:
                continue

            for metric in metrics:
                xa = pd.to_numeric(ma.loc[common, metric], errors="coerce").to_numpy(dtype=float)
                xb = pd.to_numeric(mb.loc[common, metric], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(xa) & np.isfinite(xb)
                d = xa[ok] - xb[ok]

                if d.size:
                    rows.append({
                        "prior_A": A,
                        "prior_B": B,
                        "training_seed": int(train_seed),
                        "metric": metric,
                        "n_matched_eval_runs": int(d.size),
                        "mean_delta_A_minus_B": float(np.mean(d)),
                        "sd_delta_within_seed": (
                            float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
                        ),
                    })

    return pd.DataFrame(rows)


def hierarchical_bootstrap_pair(
    all_runs: pd.DataFrame,
    prior_a: str,
    prior_b: str,
    metric: str,
    n_boot: int = 10000,
    ci: float = 95.0,
    seed: int = 20260831,
) -> Dict[str, Any]:
    """
    Cluster bootstrap:
      1. resample training-seed clusters with replacement;
      2. within each selected cluster, resample matched evaluation-run pairs
         with replacement;
      3. average within cluster;
      4. average cluster means equally.

    Delta = prior_a - prior_b. For lower-is-better metrics, negative favors A.
    """
    clusters = {}

    common_train_seeds = sorted(
        set(all_runs.loc[all_runs["prior"] == prior_a, "training_seed"])
        & set(all_runs.loc[all_runs["prior"] == prior_b, "training_seed"])
    )

    for ts in common_train_seeds:
        ga = all_runs[
            (all_runs["prior"] == prior_a)
            & (all_runs["training_seed"] == ts)
        ].set_index("seed")

        gb = all_runs[
            (all_runs["prior"] == prior_b)
            & (all_runs["training_seed"] == ts)
        ].set_index("seed")

        common_runs = sorted(set(ga.index) & set(gb.index))
        if not common_runs:
            continue

        xa = pd.to_numeric(ga.loc[common_runs, metric], errors="coerce").to_numpy(dtype=float)
        xb = pd.to_numeric(gb.loc[common_runs, metric], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(xa) & np.isfinite(xb)
        diff = xa[ok] - xb[ok]
        if diff.size:
            clusters[int(ts)] = diff

    if len(clusters) < 2:
        raise ValueError(
            f"Need >=2 matched training-seed clusters for {prior_a} vs {prior_b}, "
            f"metric={metric}; found {len(clusters)}"
        )

    cluster_ids = np.array(sorted(clusters), dtype=int)
    observed_cluster_means = np.array(
        [np.mean(clusters[int(c)]) for c in cluster_ids],
        dtype=np.float64,
    )
    observed = float(np.mean(observed_cluster_means))

    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    boot = np.empty(int(n_boot), dtype=np.float64)

    for b in range(int(n_boot)):
        sampled_clusters = rng.choice(
            cluster_ids,
            size=len(cluster_ids),
            replace=True,
        )
        cluster_means = []

        for c in sampled_clusters:
            d = clusters[int(c)]
            sampled_d = d[rng.randint(0, len(d), size=len(d))]
            cluster_means.append(float(np.mean(sampled_d)))

        boot[b] = float(np.mean(cluster_means))

    alpha = (100.0 - float(ci)) / 2.0
    lo = float(np.percentile(boot, alpha))
    hi = float(np.percentile(boot, 100.0 - alpha))

    return {
        "prior_A": prior_a,
        "prior_B": prior_b,
        "metric": metric,
        "n_training_seed_clusters": int(len(clusters)),
        "training_seeds": ",".join(str(x) for x in sorted(clusters)),
        "observed_delta_A_minus_B": observed,
        "bootstrap_mean_delta": float(np.mean(boot)),
        "ci_percent": float(ci),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_bootstrap": int(n_boot),
        "interpretation_lower_is_better": (
            f"{prior_a} favored" if hi < 0
            else f"{prior_b} favored" if lo > 0
            else "interval includes zero"
        ),
    }


# =============================================================================
# I/O workflows
# =============================================================================

def load_fixed_pool(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "real_crops" not in obj:
            raise KeyError(f"{path} is a dict but has no 'real_crops'")
        x = obj["real_crops"]
    elif torch.is_tensor(obj):
        x = obj
    else:
        raise TypeError(f"Unsupported pool object: {type(obj)}")

    if not torch.is_tensor(x):
        raise TypeError("real_crops is not a torch.Tensor")
    if x.dim() != 4:
        raise ValueError(f"Expected pool NCHW, got {tuple(x.shape)}")
    if x.shape[0] < 128:
        raise ValueError(
            f"Pool has only {x.shape[0]} samples; current RR protocol requires >=128"
        )

    return x.float().cpu()


def run_fixed_pool_rr(
    pool_path: Path,
    out_dir: Path,
    device: torch.device,
    rr_trials: int,
    rr_percentile: float,
    rr_min_n: int,
    rr_seed: int,
    rr_boot: int,
    rr_boot_seed: int,
    ci: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_fixed_pool(pool_path)
    print(f"[pool] loaded {tuple(pool.shape)} from {pool_path}")

    phi = extract_fixed_pool_phi(pool, device=device)
    np.save(out_dir / "fixed_pool_phi160.npy", phi)

    phi_n, valid = standardize_real_only(phi, eps=1e-8)
    np.save(out_dir / "fixed_pool_phi160_standardized.npy", phi_n)
    np.save(out_dir / "fixed_pool_phi_valid_mask.npy", valid)

    rr = compute_rr_split_values(
        phi_n,
        valid_mask=valid,
        trials=rr_trials,
        min_per_split=rr_min_n,
        seed=rr_seed,
    )

    rr_df = pd.DataFrame(rr)
    rr_df.insert(0, "trial", np.arange(1, len(rr_df) + 1))
    rr_df.to_csv(out_dir / "fixed_pool_rr_splits.csv", index=False)

    baseline = {}
    for i, group in enumerate(["Total", "StructTex", "Spectrum", "Energy"]):
        baseline[group] = bootstrap_quantile(
            rr[group],
            q=rr_percentile,
            n_boot=rr_boot,
            ci=ci,
            seed=rr_boot_seed + i * 1000,
        )

    payload = {
        "purpose": (
            "fixed-pool RR stability diagnostic; not the primary denominator "
            "for current per-run-standardized test phi FD"
        ),
        "pool_path": str(pool_path.resolve()),
        "pool_shape": list(pool.shape),
        "phi_dim": int(PHI_DIM),
        "valid_phi_dimensions": int(np.sum(valid)),
        "evaluation_edge_type": "log",
        "rr_trials": int(rr_trials),
        "rr_percentile": float(rr_percentile),
        "rr_min_per_split": int(rr_min_n),
        "rr_seed": int(rr_seed),
        "bootstrap_replicates": int(rr_boot),
        "bootstrap_ci": float(ci),
        "baseline": baseline,
    }

    with (out_dir / "fixed_pool_rr_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[fixed-pool RR baseline diagnostic]")
    for group, stats in baseline.items():
        print(
            f"{group:10s}: q{rr_percentile:g}={stats['threshold']:.6f} "
            f"CI[{stats['bootstrap_lo']:.6f}, {stats['bootstrap_hi']:.6f}]"
        )

    return baseline


def run_multiseed_summary(
    manifest_path: Path,
    out_dir: Path,
    fixed_rr_baseline: Optional[Dict[str, Any]],
    min_rr_trials: int,
    hier_boot: int,
    hier_seed: int,
    ci: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path, min_rr_trials=min_rr_trials)
    all_runs = collect_runs(manifest, fixed_rr_baseline=fixed_rr_baseline)
    validate_matched_run_seeds(all_runs)

    metrics = available_metrics(all_runs)
    if not metrics:
        raise ValueError("No supported metrics found in JSONLs")

    all_runs.to_csv(
        out_dir / "all_runs_with_sensitivity_ratios.csv",
        index=False,
    )

    seed_summary = seed_level_summary(all_runs, metrics)
    seed_summary.to_csv(out_dir / "seed_level_summary.csv", index=False)

    model_summary = model_level_summary(seed_summary, metrics)
    model_summary.to_csv(out_dir / "model_level_summary.csv", index=False)

    pairs = [
        ("SLoG", "LoG"),
        ("SLoG", "Canny"),
        ("LoG", "Canny"),
    ]

    paired_seed = paired_seed_level_differences(
        all_runs,
        metrics,
        pairs,
    )
    paired_seed.to_csv(
        out_dir / "paired_differences_seed_level.csv",
        index=False,
    )

    # Primary bootstrap targets: especially total phi FD/RR.
    bootstrap_metrics = [
        m for m in [
            "rr_fd_ratio_total",
            "rr_fd_ratio_structtex",
            "rr_fd_ratio_spectrum",
            "rr_fd_ratio_energy",
            "swd_total",
            "geo_kid_mean",
            "phi_fd_total",
        ]
        if m in metrics
    ]

    boot_rows = []
    counter = 0
    for A, B in pairs:
        for metric in bootstrap_metrics:
            try:
                res = hierarchical_bootstrap_pair(
                    all_runs=all_runs,
                    prior_a=A,
                    prior_b=B,
                    metric=metric,
                    n_boot=hier_boot,
                    ci=ci,
                    seed=hier_seed + counter * 1000,
                )
                boot_rows.append(res)
            except ValueError as exc:
                print(f"[bootstrap skip] {exc}")
            counter += 1

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "hierarchical_bootstrap.csv", index=False)

    metadata = {
        "manifest": str(manifest_path.resolve()),
        "primary_rr_policy": (
            f"JSONL rows must come from final test runs with rr_fd_trials >= {min_rr_trials}. "
            "Primary rr_fd_ratio_* values are retained from test(), where numerator "
            "and RR denominator share the same per-run phi standardization."
        ),
        "fixed_pool_sensitivity_ratio_policy": (
            "fixedpool_sensitivity_ratio_* columns, when present, are secondary "
            "sensitivity diagnostics only."
        ),
        "metrics_summarized": metrics,
        "hierarchical_bootstrap_replicates": int(hier_boot),
        "bootstrap_ci": float(ci),
        "training_seed_is_independent_unit": True,
        "evaluation_runs_are_not_independent_model_trainings": True,
    }
    with (out_dir / "analysis_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n[model-level summary]")
    cols = [
        c for c in [
            "prior",
            "n_training_seeds",
            "training_seeds",
            "rr_fd_ratio_total_model_mean",
            "rr_fd_ratio_total_model_sd",
            "swd_total_model_mean",
            "swd_total_model_sd",
            "geo_kid_mean_model_mean",
            "geo_kid_mean_model_sd",
        ]
        if c in model_summary.columns
    ]
    print(model_summary[cols].to_string(index=False))

    if len(boot_df):
        print("\n[hierarchical bootstrap: rr_fd_ratio_total]")
        q = boot_df[boot_df["metric"] == "rr_fd_ratio_total"]
        if len(q):
            print(q.to_string(index=False))


def write_manifest_template(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for prior, prefix in [("SLoG", "slog"), ("LoG", "log"), ("Canny", "canny")]:
        for seed in [35, 39, 42]:
            rows.append({
                "prior": prior,
                "training_seed": seed,
                "jsonl": f"/path/{prefix}_seed{seed}/metrics_test_runs.jsonl",
                "rr_trials": 1000,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote manifest template: {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Final RR stability + Australian multi-training-seed analysis"
    )

    p.add_argument("--pool", type=str, default="",
                   help="fixed_test_pool_256.pt; enables fixed-pool RR stability analysis")
    p.add_argument("--manifest", type=str, default="",
                   help="CSV mapping prior/training_seed/jsonl/rr_trials")
    p.add_argument("--out_dir", type=str, default="./final_rr_multiseed_analysis")

    p.add_argument("--device", type=str, default="cuda:0",
                   help="device for fixed-pool phi extraction; e.g. cuda:0 or cpu")

    p.add_argument("--rr_trials", type=int, default=1000)
    p.add_argument("--rr_percentile", type=float, default=95.0)
    p.add_argument("--rr_min_n", type=int, default=64)
    p.add_argument("--rr_seed", type=int, default=20260829)
    p.add_argument("--rr_boot", type=int, default=10000)
    p.add_argument("--rr_boot_seed", type=int, default=20260830)

    p.add_argument("--min_final_rr_trials", type=int, default=1000,
                   help="minimum rr_trials accepted from final-test manifest rows")

    p.add_argument("--hier_boot", type=int, default=10000)
    p.add_argument("--hier_seed", type=int, default=20260831)
    p.add_argument("--ci", type=float, default=95.0)

    p.add_argument("--write_manifest_template", type=str, default="",
                   help="write a 35/39/42 x SLoG/LoG/Canny manifest template and exit")

    return p.parse_args()


def main():
    args = parse_args()

    if args.write_manifest_template:
        write_manifest_template(Path(args.write_manifest_template).expanduser())
        return

    if not args.pool and not args.manifest:
        raise SystemExit(
            "Provide --pool and/or --manifest, or use --write_manifest_template."
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_rr_baseline = None

    if args.pool:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {args.device}, but CUDA is unavailable. Use --device cpu."
            )
        device = torch.device(args.device)
        fixed_rr_baseline = run_fixed_pool_rr(
            pool_path=Path(args.pool).expanduser().resolve(),
            out_dir=out_dir,
            device=device,
            rr_trials=args.rr_trials,
            rr_percentile=args.rr_percentile,
            rr_min_n=args.rr_min_n,
            rr_seed=args.rr_seed,
            rr_boot=args.rr_boot,
            rr_boot_seed=args.rr_boot_seed,
            ci=args.ci,
        )

    if args.manifest:
        run_multiseed_summary(
            manifest_path=Path(args.manifest).expanduser().resolve(),
            out_dir=out_dir,
            fixed_rr_baseline=fixed_rr_baseline,
            min_rr_trials=args.min_final_rr_trials,
            hier_boot=args.hier_boot,
            hier_seed=args.hier_seed,
            ci=args.ci,
        )

    print(f"\nDone. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
