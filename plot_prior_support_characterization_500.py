"""
plot_prior_support_characterization_500.py

Purpose
-------
Create the proposed 2x2 structural-prior characterization figure for the same
500 Australian seismic profiles already used in seismic_edge_metrics.csv.

The script uses:
    1) seismic_edge_metrics.csv
       - EXACT sample manifest: 500 unique image names x 3 algorithms
       - existing per-profile topology/support metrics for panels (c) and (d)

    2) effective_parameters.json
       - the deployed preprocessing and SLoG/LoG/Canny parameters

    3) the same 500 original 512x512 images
       - used to recompute continuous/binary response maps only for panels
         (a) and (b), because the existing CSV does not contain per-pixel maps.

Figure panels
-------------
(a) Response-value distribution
    x = normalized response value [0,1]
    y = mean pixel proportion per histogram bin
    By default y uses a logarithmic scale because Canny is binary and otherwise
    its two endpoint spikes compress the continuous SLoG/LoG distributions.

(b) Response-support survival curve
    x = response threshold t
    y = P(E > t)
    Main line = profile-wise mean
    Shaded band = 25th-75th percentile across the 500 profiles
    This describes how much spatial support remains as the response threshold
    increases. It is a representation descriptor, NOT an "information density"
    metric and NOT a quality score.

(c) Valid-signal support density
    Directly reads edge_density_valid_signal from seismic_edge_metrics.csv.
    Violin + boxplot; no recomputation.

(d) Fragmentation
    Directly reads fragmentation_components_per_100px from the CSV.
    Violin + boxplot; lower values mean fewer connected components per
    100 skeleton pixels.

Important interpretation
------------------------
The figure is designed to show:
    response amount + response distribution + structural organization

It should NOT be interpreted as:
    "more response pixels = better prior".

Canny is intrinsically binary, while SLoG and LoG are continuous. Therefore
panel (a) explicitly compares response representations; panel (b) describes
support persistence over threshold; panels (c) and (d) provide complementary
spatial/topological characterization.

Notes on reproducibility
------------------------
The response implementation below follows the previously used
seismic_edge_metrics_compare_complete.py definitions:
    input robust normalization: 1st-99th percentile
    NLMeans -> SGS -> AGC
    SLoG response:
        sigma=0.4, anisotropy=2.1, 8 angles, sharpness=14, alpha=0.92,
        aniso_clip=4, gamma_scale=1, relative alignment
    LoG:
        sigma=1.1
    Canny:
        sigma=1.7, low=0.1, high=0.9, uint8 AGC input
All deployed values are loaded from effective_parameters.json when present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from PIL import Image

from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter

from skimage import feature, restoration
from skimage.util import img_as_ubyte

import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


EPS = 1e-8
ALGORITHMS = ("SLoG", "LoG", "Canny")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PreprocessConfig:
    nl_h: float = 0.8
    nlmeans_patch_size: int = 5
    nlmeans_patch_distance: int = 6

    sgs_sigma: float = 1.5
    sgs_anisotropy: float = 2.2
    sgs_iterations: int = 1
    sgs_base_kappa: float = 10.0
    sgs_gamma: float = 0.06
    sgs_option: int = 1
    lambda_perp_base: float = 1e-3
    lambda_par_base: float = 1.0
    coherence_influence: float = 1.0

    agc_window: int = 31
    agc_eps: float = 1e-8
    agc_p1: float = 1.0

    percentile_low: float = 1.0
    percentile_high: float = 99.0

    device: str = "cuda:0"
    prefer_fp16: bool = False


@dataclass
class DetectorConfig:
    tex_layout: str = "fault,agc,edge"

    glog_sigma: float = 0.4
    glog_anisotropy: float = 2.1
    glog_nangles: int = 8
    glog_sharpness: float = 14.0
    glog_alpha: float = 0.92
    glog_aniso_clip: float = 4.0
    glog_gamma_scale: float = 1.0
    glog_dist_mode: str = "relative"
    glog_coherence_influence: float = 1.0
    glog_size_factor: float = 6.0
    glog_use_absolute_sigma: bool = False
    glog_absolute_response: bool = False

    edge_thr: float = 0.35

    log_sigma: float = 1.1

    canny_sigma: float = 1.7
    canny_low: float = 0.1
    canny_high: float = 0.9
    canny_use_uint8_input: bool = True


def load_configs(params_json: Path, device_override: str) -> Tuple[PreprocessConfig, DetectorConfig, Dict[str, Any]]:
    with params_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    sgs = raw.get("sgs_params", {})
    agc = raw.get("agc_params", {})
    det = raw.get("det_params", {})

    pre_cfg = PreprocessConfig(
        nl_h=float(sgs.get("nl_h", 0.8)),
        sgs_sigma=float(sgs.get("sigma", 1.5)),
        sgs_anisotropy=float(sgs.get("anisotropy", 2.2)),
        sgs_iterations=int(sgs.get("iterations", 1)),
        sgs_base_kappa=float(sgs.get("base_kappa", 10.0)),
        sgs_gamma=float(sgs.get("gamma", 0.06)),
        sgs_option=int(sgs.get("option", 1)),
        lambda_perp_base=float(sgs.get("lambda_perp_base", 1e-3)),
        lambda_par_base=float(sgs.get("lambda_par_base", 1.0)),
        coherence_influence=float(sgs.get("coherence_influence", 1.0)),
        agc_window=int(agc.get("window", 31)),
        agc_eps=float(agc.get("eps", 1e-8)),
        agc_p1=float(agc.get("p1", 1.0)),
        device=str(device_override),
        # The supplied effective_parameters.json has prefer_fp16=false.
        prefer_fp16=bool(sgs.get("prefer_fp16", False)),
    )

    det_cfg = DetectorConfig(
        tex_layout=str(det.get("tex_layout", "fault,agc,edge")),
        glog_sigma=float(det.get("glog_sigma", 0.4)),
        glog_anisotropy=float(det.get("glog_anisotropy", 2.1)),
        glog_nangles=int(det.get("glog_nangles", 8)),
        glog_sharpness=float(det.get("glog_sharpness", 14.0)),
        glog_alpha=float(det.get("glog_alpha", 0.92)),
        glog_aniso_clip=float(det.get("glog_aniso_clip", 4.0)),
        glog_gamma_scale=float(det.get("glog_gamma_scale", 1.0)),
        glog_dist_mode=str(det.get("glog_dist_mode", "relative")),
        glog_coherence_influence=float(det.get("glog_coherence_influence", 1.0)),
        glog_size_factor=float(det.get("glog_size_factor", 6.0)),
        glog_use_absolute_sigma=bool(det.get("glog_use_absolute_sigma", False)),
        edge_thr=float(det.get("edge_thr", 0.35)),
        log_sigma=float(det.get("log_sigma", 1.1)),
        canny_sigma=float(det.get("canny_sigma", 1.7)),
        canny_low=float(det.get("canny_low", 0.1)),
        canny_high=float(det.get("canny_high", 0.9)),
    )
    return pre_cfg, det_cfg, raw


# =============================================================================
# Original 500-profile preprocessing / detector implementation
# =============================================================================

def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def percentile_normalize(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= EPS:
        return normalize_map(arr)
    return np.clip((arr - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def load_grayscale_image(path: Path, cfg: PreprocessConfig) -> np.ndarray:
    """Load 8/16-bit grayscale or RGB image and robustly normalize to [0,1]."""
    with Image.open(path) as im:
        arr = np.asarray(im)

    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = (
                0.2989 * arr[..., 0].astype(np.float32)
                + 0.5870 * arr[..., 1].astype(np.float32)
                + 0.1140 * arr[..., 2].astype(np.float32)
            )
        else:
            arr = arr[..., 0].astype(np.float32)
    else:
        arr = arr.astype(np.float32)

    return percentile_normalize(arr, cfg.percentile_low, cfg.percentile_high)


def odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def denoise_nlmeans(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    try:
        try:
            sigma_est = restoration.estimate_sigma(img, channel_axis=None)
        except TypeError:
            sigma_est = restoration.estimate_sigma(img, multichannel=False)
        sigma_val = float(np.mean(sigma_est))
    except Exception:
        high_pass = img.astype(np.float32) - gaussian_filter(
            img.astype(np.float32), sigma=1.0
        )
        median = float(np.median(high_pass))
        sigma_val = float(
            np.median(np.abs(high_pass - median)) / 0.67448975
        )

    h = max(float(cfg.nl_h) * sigma_val, 1e-6)
    kwargs = dict(
        h=h,
        patch_size=int(cfg.nlmeans_patch_size),
        patch_distance=int(cfg.nlmeans_patch_distance),
        fast_mode=True,
        preserve_range=True,
    )
    try:
        den = restoration.denoise_nl_means(img, channel_axis=None, **kwargs)
    except TypeError:
        den = restoration.denoise_nl_means(img, multichannel=False, **kwargs)

    return np.clip(den, 0.0, 1.0).astype(np.float32)


def build_structure_tensor_maps(
    img: np.ndarray,
    sigma: float,
    lambda_parallel: float,
    lambda_perp: float,
    coherence_influence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = ndimage.sobel(img, axis=1, mode="reflect").astype(np.float32)
    gy = ndimage.sobel(img, axis=0, mode="reflect").astype(np.float32)

    jxx = gaussian_filter(gx * gx, sigma=sigma, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=sigma, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=sigma, mode="reflect")

    delta = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    eig1 = 0.5 * (jxx + jyy + delta)
    eig2 = 0.5 * (jxx + jyy - delta)

    coherence = np.clip(
        (eig1 - eig2) / (eig1 + eig2 + EPS),
        0.0,
        1.0,
    )
    orientation = 0.5 * np.arctan2(
        2.0 * jxy,
        jxx - jyy + EPS,
    )

    vx = np.cos(orientation)
    vy = np.sin(orientation)

    lam_per_map = float(lambda_perp) * (
        1.0 - float(coherence_influence) * coherence
    )
    lam_per_map = np.clip(lam_per_map, 1e-8, None)
    lam_par = float(lambda_parallel)

    dxx = lam_par * vx * vx + lam_per_map * vy * vy
    dyy = lam_par * vy * vy + lam_per_map * vx * vx
    dxy = (lam_par - lam_per_map) * vx * vy

    return (
        coherence.astype(np.float32),
        orientation.astype(np.float32),
        dxx.astype(np.float32),
        dxy.astype(np.float32),
        dyy.astype(np.float32),
    )


def resolve_torch_device(device_name: str):
    if torch is None:
        return None
    name = str(device_name).lower()
    if name == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    if name.startswith("cuda") and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested but unavailable; using CPU."
        )
        return torch.device("cpu")
    return torch.device(device_name)


def tensor_diffusion_torch(
    img: np.ndarray,
    dxx: np.ndarray,
    dxy: np.ndarray,
    dyy: np.ndarray,
    iterations: int,
    gamma: float,
    device_name: str,
    prefer_fp16: bool = False,
) -> np.ndarray:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is unavailable")

    device = resolve_torch_device(device_name)
    use_fp16 = bool(prefer_fp16) and device.type == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    image = (
        torch.from_numpy(img.astype(np.float32))[None, None]
        .to(device=device, dtype=dtype)
    )
    tdxx = torch.from_numpy(dxx)[None, None].to(
        device=device, dtype=dtype
    )
    tdxy = torch.from_numpy(dxy)[None, None].to(
        device=device, dtype=dtype
    )
    tdyy = torch.from_numpy(dyy)[None, None].to(
        device=device, dtype=dtype
    )

    kx = torch.tensor(
        [[[[-0.5, 0.0, 0.5]]]],
        device=device,
        dtype=dtype,
    )
    ky = torch.tensor(
        [[[[-0.5], [0.0], [0.5]]]],
        device=device,
        dtype=dtype,
    )

    out = image
    for _ in range(max(0, int(iterations))):
        gx = F.conv2d(
            F.pad(out, (1, 1, 0, 0), mode="replicate"),
            kx,
        )
        gy = F.conv2d(
            F.pad(out, (0, 0, 1, 1), mode="replicate"),
            ky,
        )

        jx = tdxx * gx + tdxy * gy
        jy = tdxy * gx + tdyy * gy

        djx = F.conv2d(
            F.pad(jx, (1, 1, 0, 0), mode="replicate"),
            kx,
        )
        djy = F.conv2d(
            F.pad(jy, (0, 0, 1, 1), mode="replicate"),
            ky,
        )

        out = torch.nan_to_num(
            out + float(gamma) * (djx + djy),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

    return (
        out[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def tensor_diffusion_numpy(
    img: np.ndarray,
    dxx: np.ndarray,
    dxy: np.ndarray,
    dyy: np.ndarray,
    iterations: int,
    gamma: float,
) -> np.ndarray:
    out = img.astype(np.float32).copy()

    kernel = np.array(
        [-0.5, 0.0, 0.5],
        dtype=np.float32,
    )

    for _ in range(max(0, int(iterations))):
        gx = ndimage.convolve1d(
            out,
            kernel,
            axis=1,
            mode="nearest",
        )
        gy = ndimage.convolve1d(
            out,
            kernel,
            axis=0,
            mode="nearest",
        )

        jx = dxx * gx + dxy * gy
        jy = dxy * gx + dyy * gy

        djx = ndimage.convolve1d(
            jx,
            kernel,
            axis=1,
            mode="nearest",
        )
        djy = ndimage.convolve1d(
            jy,
            kernel,
            axis=0,
            mode="nearest",
        )

        out = np.nan_to_num(
            out + float(gamma) * (djx + djy),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

    return out.astype(np.float32)


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
    device: str = "auto",
    prefer_fp16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # These arguments are retained exactly for experiment compatibility.
    sigma = float(sigma)
    anisotropy = float(anisotropy)
    iterations = int(iterations)
    base_kappa = float(base_kappa)
    gamma = float(gamma)
    option = int(option)
    lambda_perp_base = float(lambda_perp_base)
    lambda_par_base = float(lambda_par_base)
    coherence_influence = float(coherence_influence)

    if sigma <= 0:
        raise ValueError(f"sgs sigma must be > 0, got {sigma}")
    if anisotropy <= 0:
        raise ValueError(
            f"sgs anisotropy must be > 0, got {anisotropy}"
        )
    if iterations < 0:
        raise ValueError(
            f"sgs iterations must be >= 0, got {iterations}"
        )
    if base_kappa <= 0:
        raise ValueError(
            f"sgs base_kappa must be > 0, got {base_kappa}"
        )

    coherence, orientation, dxx, dxy, dyy = (
        build_structure_tensor_maps(
            img,
            sigma=sigma,
            lambda_parallel=lambda_par_base,
            lambda_perp=lambda_perp_base,
            coherence_influence=coherence_influence,
        )
    )

    if torch is not None:
        try:
            smooth = tensor_diffusion_torch(
                img,
                dxx,
                dxy,
                dyy,
                iterations=iterations,
                gamma=gamma,
                device_name=device,
                prefer_fp16=prefer_fp16,
            )
        except Exception as exc:
            logging.warning(
                "Torch diffusion failed (%s); using NumPy fallback.",
                exc,
            )
            smooth = tensor_diffusion_numpy(
                img,
                dxx,
                dxy,
                dyy,
                iterations,
                gamma,
            )
    else:
        smooth = tensor_diffusion_numpy(
            img,
            dxx,
            dxy,
            dyy,
            iterations,
            gamma,
        )

    smooth = normalize_map(smooth)
    return (
        smooth,
        coherence,
        orientation,
        dxx,
        dxy,
        dyy,
    )


def agc_local_rms(
    img: np.ndarray,
    window: int,
    low_percentile: float,
    eps: float = 1e-8,
) -> np.ndarray:
    window = odd_at_least_three(window)

    local_mean_sq = uniform_filter(
        img.astype(np.float32) ** 2,
        size=window,
        mode="reflect",
    )
    local_rms = np.sqrt(
        np.maximum(local_mean_sq, 0.0) + eps
    )
    out = img / (local_rms + eps)

    p = float(
        np.clip(low_percentile, 0.0, 49.9)
    )
    lo, hi = np.percentile(
        out,
        [p, 100.0 - p],
    )

    if hi - lo <= EPS:
        return normalize_map(out)

    return np.clip(
        (out - lo) / (hi - lo + EPS),
        0.0,
        1.0,
    ).astype(np.float32)


def preprocess_image(
    img: np.ndarray,
    cfg: PreprocessConfig,
) -> Dict[str, np.ndarray]:
    nl = denoise_nlmeans(img, cfg)

    sgs, coherence, orientation, dxx, dxy, dyy = (
        structure_guided_smoothing(
            nl,
            sigma=cfg.sgs_sigma,
            anisotropy=cfg.sgs_anisotropy,
            iterations=cfg.sgs_iterations,
            base_kappa=cfg.sgs_base_kappa,
            gamma=cfg.sgs_gamma,
            option=cfg.sgs_option,
            lambda_perp_base=cfg.lambda_perp_base,
            lambda_par_base=cfg.lambda_par_base,
            coherence_influence=cfg.coherence_influence,
            device=cfg.device,
            prefer_fp16=cfg.prefer_fp16,
        )
    )

    agc = agc_local_rms(
        sgs,
        cfg.agc_window,
        cfg.agc_p1,
        eps=cfg.agc_eps,
    )

    return {
        "gray": img.astype(np.float32),
        "nlmeans": nl,
        "sgs": sgs,
        "agc": agc,
        "coherence": coherence,
        "orientation": orientation,
        "Dxx": dxx,
        "Dxy": dxy,
        "Dyy": dyy,
    }


def make_anisotropic_gaussian_kernel(
    sx: float,
    sy: float,
    angle_rad: float,
    size: int,
) -> np.ndarray:
    half = size // 2
    y, x = np.mgrid[
        -half : half + 1,
        -half : half + 1,
    ].astype(np.float32)

    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)

    xr = ca * x + sa * y
    yr = -sa * x + ca * y

    g = np.exp(
        -0.5
        * (
            (xr / (sx + EPS)) ** 2
            + (yr / (sy + EPS)) ** 2
        )
    )
    return (
        g / (np.sum(g) + EPS)
    ).astype(np.float32)


def laplacian_of_kernel(
    kernel: np.ndarray,
) -> np.ndarray:
    lap = ndimage.laplace(
        kernel,
        mode="reflect",
    )
    lap = lap - np.mean(lap)
    return (
        lap / (np.sum(np.abs(lap)) + EPS)
    ).astype(np.float32)


def slog_response(
    img: np.ndarray,
    orientation_map: np.ndarray,
    coherence_map: np.ndarray,
    dxx_map: np.ndarray,
    dxy_map: np.ndarray,
    dyy_map: np.ndarray,
    cfg: DetectorConfig,
) -> np.ndarray:
    """
    Structure-tensor-guided smoothing-based LoG response.

    This follows the implementation used for the supplied 500-profile
    characterization result.
    """
    image = img.astype(np.float32)

    angles = np.linspace(
        0.0,
        np.pi,
        int(cfg.glog_nangles),
        endpoint=False,
    )

    max_aniso = max(
        1.0,
        float(cfg.glog_anisotropy),
        float(cfg.glog_aniso_clip),
    )

    kernel_size = int(
        max(
            3,
            math.ceil(
                cfg.glog_size_factor
                * cfg.glog_sigma
                * max_aniso
            )
            * 2
            + 1,
        )
    )
    if kernel_size % 2 == 0:
        kernel_size += 1

    a = dxx_map.astype(np.float32)
    b = dxy_map.astype(np.float32)
    c = dyy_map.astype(np.float32)

    delta = np.sqrt(
        (a - c) ** 2
        + 4.0 * b * b
        + 1e-18
    )

    eig1 = np.clip(
        0.5 * (a + c + delta),
        EPS,
        None,
    )
    eig2 = np.clip(
        0.5 * (a + c - delta),
        EPS,
        None,
    )

    tensor_orientation = 0.5 * np.arctan2(
        2.0 * b,
        a - c + EPS,
    )

    anisotropy_ratio = np.clip(
        np.sqrt(eig1 / eig2),
        1.0,
        float(cfg.glog_aniso_clip),
    )

    ratio_scaled = np.power(
        anisotropy_ratio,
        float(cfg.glog_gamma_scale),
    )

    sigma_parallel = (
        float(cfg.glog_sigma)
        * ratio_scaled
    )
    sigma_perpendicular = (
        float(cfg.glog_sigma)
        / (ratio_scaled + EPS)
    )

    if cfg.glog_use_absolute_sigma:
        sigma_parallel = np.abs(sigma_parallel)
        sigma_perpendicular = np.abs(
            sigma_perpendicular
        )

    fused = np.zeros_like(
        image,
        dtype=np.float32,
    )

    if cfg.glog_dist_mode.lower() == "relative":
        max_alignment = np.zeros_like(
            image,
            dtype=np.float32,
        )
        for angle in angles:
            max_alignment = np.maximum(
                max_alignment,
                np.abs(
                    np.cos(
                        float(angle)
                        - tensor_orientation
                    )
                ),
            )
    else:
        max_alignment = np.ones_like(
            image,
            dtype=np.float32,
        )

    for angle in angles:
        kernel = make_anisotropic_gaussian_kernel(
            sx=float(cfg.glog_sigma)
            * float(cfg.glog_anisotropy),
            sy=float(cfg.glog_sigma),
            angle_rad=float(angle),
            size=kernel_size,
        )

        log_kernel = laplacian_of_kernel(
            kernel
        )

        response = ndimage.convolve(
            image,
            log_kernel,
            mode="reflect",
        ).astype(np.float32)

        alignment = np.abs(
            np.cos(
                float(angle)
                - tensor_orientation
            )
        )

        if cfg.glog_dist_mode.lower() == "relative":
            alignment = (
                alignment
                / (max_alignment + 1e-6)
            )

        if float(cfg.glog_sharpness) != 1.0:
            alignment = np.power(
                alignment,
                float(cfg.glog_sharpness),
            )

        coherence_weight = np.clip(
            float(
                cfg.glog_coherence_influence
            )
            * coherence_map,
            0.0,
            1.0,
        )

        base_weight = (
            (1.0 - float(cfg.glog_alpha))
            + float(cfg.glog_alpha)
            * coherence_weight
            * alignment
        )

        local_scale_ratio = (
            sigma_parallel
            / (
                float(cfg.glog_sigma)
                + EPS
            )
        )

        anisotropy_boost = (
            1.0
            + (
                local_scale_ratio
                - 1.0
            )
            * alignment
        )

        fused += (
            response
            * base_weight
            * anisotropy_boost
        )

    local_scale = (
        np.sqrt(
            np.maximum(
                sigma_parallel,
                EPS,
            )
            * np.maximum(
                sigma_perpendicular,
                EPS,
            )
        )
        / (
            float(cfg.glog_sigma)
            + EPS
        )
    )

    fused *= np.clip(
        local_scale,
        1e-6,
        1e3,
    )

    if cfg.glog_absolute_response:
        fused = np.abs(fused)

    fused = normalize_map(fused)

    # Source-compatible second normalization.
    fused = percentile_normalize(
        fused,
        low=5.0,
        high=99.0,
    )

    return fused.astype(np.float32)


def log_response(
    img: np.ndarray,
    sigma: float,
) -> np.ndarray:
    smooth = gaussian_filter(
        img.astype(np.float32),
        sigma=float(sigma),
        mode="reflect",
    )
    response = np.abs(
        ndimage.laplace(
            smooth,
            mode="reflect",
        )
    )
    return normalize_map(response)


def run_detectors(
    pre: Mapping[str, np.ndarray],
    cfg: DetectorConfig,
) -> Dict[str, np.ndarray]:
    agc = pre["agc"]

    sr = slog_response(
        agc,
        pre["orientation"],
        pre["coherence"],
        pre["Dxx"],
        pre["Dxy"],
        pre["Dyy"],
        cfg,
    )

    lr = log_response(
        agc,
        cfg.log_sigma,
    )

    canny_input = (
        img_as_ubyte(
            np.clip(
                agc,
                0.0,
                1.0,
            )
        )
        if cfg.canny_use_uint8_input
        else agc.astype(np.float32)
    )

    cb = feature.canny(
        canny_input,
        sigma=float(cfg.canny_sigma),
        low_threshold=float(cfg.canny_low),
        high_threshold=float(cfg.canny_high),
    )

    return {
        "SLoG": sr.astype(np.float32),
        "LoG": lr.astype(np.float32),
        "Canny": cb.astype(np.float32),
    }


# =============================================================================
# Input audit and exact 500-profile manifest
# =============================================================================

def canonical_algorithm(x: str) -> str:
    t = str(x).strip().lower()
    if t in {"slog", "glog"}:
        return "SLoG"
    if t == "log":
        return "LoG"
    if t == "canny":
        return "Canny"
    return str(x).strip()


def audit_metrics_csv(
    metrics_csv: Path,
    expected_profiles: int = 500,
) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(metrics_csv)

    required = {
        "image",
        "image_path",
        "algorithm",
        "height",
        "width",
        "edge_density_valid_signal",
        "fragmentation_components_per_100px",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"metrics CSV missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["algorithm"] = (
        df["algorithm"]
        .map(canonical_algorithm)
    )

    unknown = sorted(
        set(df["algorithm"])
        - set(ALGORITHMS)
    )
    if unknown:
        raise ValueError(
            f"Unexpected algorithms: {unknown}"
        )

    counts = (
        df.groupby("algorithm")
        .size()
        .to_dict()
    )

    unique_images = (
        df["image"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print("\n[CSV audit]")
    print(f"rows             : {len(df)}")
    print(f"unique profiles  : {len(unique_images)}")
    print(f"algorithm counts : {counts}")

    if expected_profiles > 0:
        if len(unique_images) != expected_profiles:
            raise ValueError(
                f"Expected {expected_profiles} unique profiles, "
                f"found {len(unique_images)}"
            )

        for alg in ALGORITHMS:
            if counts.get(alg, 0) != expected_profiles:
                raise ValueError(
                    f"{alg}: expected {expected_profiles} rows, "
                    f"found {counts.get(alg, 0)}"
                )

    # Verify exactly one row per image/algorithm.
    dup = (
        df.groupby(
            ["image", "algorithm"]
        )
        .size()
    )
    bad = dup[dup != 1]
    if len(bad):
        raise ValueError(
            "Some image/algorithm pairs are duplicated or missing:\n"
            + bad.head(20).to_string()
        )

    # Check expected image shape stored in old results.
    shapes = (
        df[["height", "width"]]
        .drop_duplicates()
    )
    print(
        "stored image shapes:",
        [tuple(x) for x in shapes.to_numpy()],
    )

    return df, unique_images


def resolve_image_paths(
    df: pd.DataFrame,
    image_names: Sequence[str],
    image_dir: Optional[Path],
) -> List[Path]:
    # Map image name -> historical path from CSV.
    first_rows = (
        df.drop_duplicates("image")
        .set_index("image")
    )

    resolved = []
    missing = []

    for name in image_names:
        name = str(name)

        candidates = []

        if image_dir is not None:
            candidates.append(
                image_dir / name
            )

        if name in first_rows.index:
            old_path = first_rows.loc[
                name,
                "image_path",
            ]
            if (
                isinstance(old_path, str)
                and old_path.strip()
            ):
                candidates.append(
                    Path(old_path)
                )

        found = None
        for p in candidates:
            if p.is_file():
                found = p
                break

        if found is None:
            missing.append(
                (
                    name,
                    [str(p) for p in candidates],
                )
            )
        else:
            resolved.append(found)

    if missing:
        preview = "\n".join(
            f"  {name}: {cands}"
            for name, cands in missing[:20]
        )
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} images. "
            f"First missing entries:\n{preview}"
        )

    if len(resolved) != len(image_names):
        raise RuntimeError(
            "Resolved image count does not match manifest."
        )

    print(
        f"[image audit] resolved "
        f"{len(resolved)}/{len(image_names)} profiles"
    )
    return resolved


# =============================================================================
# Response distributions
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def exact_survival_curve(
    response: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Exact P(E > t) using one sort + searchsorted.
    This is far faster than comparing every pixel against every threshold.
    """
    flat = np.asarray(
        response,
        dtype=np.float32,
    ).ravel()

    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.full(
            thresholds.shape,
            np.nan,
            dtype=np.float32,
        )

    flat = np.clip(
        flat,
        0.0,
        1.0,
    )

    sorted_flat = np.sort(flat)
    idx = np.searchsorted(
        sorted_flat,
        thresholds,
        side="right",
    )
    surv = (
        sorted_flat.size - idx
    ) / float(sorted_flat.size)

    return surv.astype(np.float32)


def response_profile_statistics(
    response: np.ndarray,
) -> Dict[str, float]:
    x = np.asarray(
        response,
        dtype=np.float32,
    ).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            k: float("nan")
            for k in [
                "mean",
                "std",
                "nonzero_fraction",
                "q25",
                "q50",
                "q75",
                "q90",
                "q95",
            ]
        }

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "nonzero_fraction": float(
            np.mean(x > 0.0)
        ),
        "q25": float(
            np.percentile(x, 25)
        ),
        "q50": float(
            np.percentile(x, 50)
        ),
        "q75": float(
            np.percentile(x, 75)
        ),
        "q90": float(
            np.percentile(x, 90)
        ),
        "q95": float(
            np.percentile(x, 95)
        ),
    }


def compute_response_curves(
    image_paths: Sequence[Path],
    image_names: Sequence[str],
    pre_cfg: PreprocessConfig,
    det_cfg: DetectorConfig,
    hist_bins: int,
    support_points: int,
    progress_every: int,
) -> Dict[str, Any]:
    bin_edges = np.linspace(
        0.0,
        1.0,
        int(hist_bins) + 1,
        dtype=np.float64,
    )
    bin_centers = (
        0.5
        * (
            bin_edges[:-1]
            + bin_edges[1:]
        )
    )

    thresholds = np.linspace(
        0.0,
        1.0,
        int(support_points),
        dtype=np.float64,
    )

    n = len(image_paths)
    hists = {
        alg: np.zeros(
            (n, hist_bins),
            dtype=np.float32,
        )
        for alg in ALGORITHMS
    }
    survival = {
        alg: np.zeros(
            (n, support_points),
            dtype=np.float32,
        )
        for alg in ALGORITHMS
    }

    per_profile_rows = []

    for i, (name, path) in enumerate(
        zip(image_names, image_paths)
    ):
        gray = load_grayscale_image(
            path,
            pre_cfg,
        )

        pre = preprocess_image(
            gray,
            pre_cfg,
        )

        responses = run_detectors(
            pre,
            det_cfg,
        )

        for alg in ALGORITHMS:
            response = np.asarray(
                responses[alg],
                dtype=np.float32,
            )
            response = np.nan_to_num(
                response,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            response = np.clip(
                response,
                0.0,
                1.0,
            )

            counts, _ = np.histogram(
                response.ravel(),
                bins=bin_edges,
            )
            hists[alg][i] = (
                counts.astype(np.float64)
                / float(response.size)
            ).astype(np.float32)

            survival[alg][i] = (
                exact_survival_curve(
                    response,
                    thresholds,
                )
            )

            stats = (
                response_profile_statistics(
                    response
                )
            )
            row = {
                "image": str(name),
                "image_path": str(path),
                "algorithm": alg,
                **stats,
            }
            per_profile_rows.append(
                row
            )

        if (
            (i + 1) % max(1, progress_every)
            == 0
            or (i + 1) == n
        ):
            print(
                f"[responses] {i + 1}/{n}",
                flush=True,
            )

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "thresholds": thresholds,
        "hists": hists,
        "survival": survival,
        "per_profile": pd.DataFrame(
            per_profile_rows
        ),
    }


def save_response_cache(
    path: Path,
    data: Dict[str, Any],
    image_names: Sequence[str],
    params_sha256: str,
) -> None:
    np.savez_compressed(
        path,
        image_names=np.asarray(
            list(image_names),
            dtype=str,
        ),
        params_sha256=np.asarray(
            [params_sha256],
            dtype=str,
        ),
        bin_edges=data["bin_edges"],
        bin_centers=data["bin_centers"],
        thresholds=data["thresholds"],
        hist_SLoG=data["hists"]["SLoG"],
        hist_LoG=data["hists"]["LoG"],
        hist_Canny=data["hists"]["Canny"],
        survival_SLoG=data["survival"]["SLoG"],
        survival_LoG=data["survival"]["LoG"],
        survival_Canny=data["survival"]["Canny"],
    )


def load_response_cache(
    path: Path,
    expected_image_names: Sequence[str],
    expected_params_sha256: str,
) -> Dict[str, Any]:
    z = np.load(
        path,
        allow_pickle=False,
    )

    cached_names = (
        z["image_names"]
        .astype(str)
        .tolist()
    )
    if cached_names != list(
        map(str, expected_image_names)
    ):
        raise ValueError(
            "Cache image manifest does not match "
            "the current metrics CSV."
        )

    if "params_sha256" in z.files:
        cached_hash = str(
            z["params_sha256"][0]
        )
        if cached_hash != expected_params_sha256:
            raise ValueError(
                "Cache was created with a different "
                "effective_parameters.json."
            )

    return {
        "bin_edges": z["bin_edges"],
        "bin_centers": z[
            "bin_centers"
        ],
        "thresholds": z["thresholds"],
        "hists": {
            "SLoG": z["hist_SLoG"],
            "LoG": z["hist_LoG"],
            "Canny": z["hist_Canny"],
        },
        "survival": {
            "SLoG": z[
                "survival_SLoG"
            ],
            "LoG": z[
                "survival_LoG"
            ],
            "Canny": z[
                "survival_Canny"
            ],
        },
        "per_profile": None,
    }


# =============================================================================
# Summaries
# =============================================================================

def curve_summary(
    x: np.ndarray,
    arrays: Dict[str, np.ndarray],
    x_name: str,
) -> pd.DataFrame:
    rows = []

    for alg in ALGORITHMS:
        arr = np.asarray(
            arrays[alg],
            dtype=np.float64,
        )

        mean = np.nanmean(
            arr,
            axis=0,
        )
        std = np.nanstd(
            arr,
            axis=0,
            ddof=1,
        )
        q25 = np.nanpercentile(
            arr,
            25,
            axis=0,
        )
        median = np.nanpercentile(
            arr,
            50,
            axis=0,
        )
        q75 = np.nanpercentile(
            arr,
            75,
            axis=0,
        )

        for j, xv in enumerate(x):
            rows.append(
                {
                    "algorithm": alg,
                    x_name: float(xv),
                    "mean": float(mean[j]),
                    "std": float(std[j]),
                    "q25": float(q25[j]),
                    "median": float(
                        median[j]
                    ),
                    "q75": float(q75[j]),
                }
            )

    return pd.DataFrame(rows)


def existing_metric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "edge_density_valid_signal":
            "Valid-signal edge density",
        "fragmentation_components_per_100px":
            "Fragments per 100 skeleton pixels",
        "local_directional_continuity":
            "Directional continuity",
        "breaks_per_100px":
            "Breaks per 100 skeleton pixels",
        "isolated_response_ratio":
            "Isolated-response ratio",
    }

    rows = []
    for alg in ALGORITHMS:
        g = df[
            df["algorithm"] == alg
        ]

        for col, label in metrics.items():
            if col not in g.columns:
                continue

            x = pd.to_numeric(
                g[col],
                errors="coerce",
            ).to_numpy(dtype=float)
            x = x[np.isfinite(x)]

            rows.append(
                {
                    "algorithm": alg,
                    "metric": col,
                    "metric_label": label,
                    "n": int(x.size),
                    "mean": float(
                        np.mean(x)
                    ),
                    "std": float(
                        np.std(
                            x,
                            ddof=1,
                        )
                    )
                    if x.size > 1
                    else float("nan"),
                    "median": float(
                        np.median(x)
                    ),
                    "q25": float(
                        np.percentile(
                            x,
                            25,
                        )
                    ),
                    "q75": float(
                        np.percentile(
                            x,
                            75,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def configure_matplotlib(
    font_size: float = 9.0,
) -> None:
    # Arial first; fallbacks keep the script portable on Linux.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def draw_violin_box(
    ax,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    data = []
    for alg in ALGORITHMS:
        vals = pd.to_numeric(
            df.loc[
                df["algorithm"] == alg,
                metric,
            ],
            errors="coerce",
        ).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)

    positions = np.arange(
        1,
        len(ALGORITHMS) + 1,
    )

    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Keep Matplotlib defaults; use transparency only.
    for body in violin["bodies"]:
        body.set_alpha(0.35)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.2,
        },
        whiskerprops={
            "linewidth": 0.9,
        },
        capprops={
            "linewidth": 0.9,
        },
        boxprops={
            "linewidth": 0.9,
        },
    )

    means = [
        float(np.mean(v))
        for v in data
    ]
    ax.scatter(
        positions,
        means,
        marker="D",
        s=18,
        zorder=5,
        label="Mean",
    )

    ax.set_xticks(
        positions,
        ALGORITHMS,
    )
    ax.set_ylabel(
        ylabel
    )
    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )


def make_figure(
    df: pd.DataFrame,
    response_data: Dict[str, Any],
    out_dir: Path,
    hist_y_scale: str,
    dpi: int,
) -> None:
    configure_matplotlib(
        font_size=9.0
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------
    # (a) response-value distribution
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    x = response_data[
        "bin_centers"
    ]

    for alg in ALGORITHMS:
        arr = response_data[
            "hists"
        ][alg]
        mean = np.nanmean(
            arr,
            axis=0,
        )

        if hist_y_scale == "log":
            y = np.where(
                mean > 0,
                mean,
                np.nan,
            )
        else:
            y = mean

        ax.step(
            x,
            y,
            where="mid",
            linewidth=1.4,
            label=alg,
        )

    if hist_y_scale == "log":
        ax.set_yscale("log")
        ax.set_ylabel(
            "Mean pixel proportion per bin\n(log scale)"
        )
    else:
        ax.set_ylabel(
            "Mean pixel proportion per bin"
        )

    ax.set_xlabel(
        "Normalized response value"
    )
    ax.set_xlim(
        0.0,
        1.0,
    )
    ax.grid(
        alpha=0.20,
        linewidth=0.6,
    )
    ax.legend(
        frameon=False,
        ncol=1,
    )
    ax.text(
        0.02,
        0.98,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # (b) response-support survival curve
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    t = response_data[
        "thresholds"
    ]

    for alg in ALGORITHMS:
        arr = response_data[
            "survival"
        ][alg]

        mean = np.nanmean(
            arr,
            axis=0,
        )
        q25 = np.nanpercentile(
            arr,
            25,
            axis=0,
        )
        q75 = np.nanpercentile(
            arr,
            75,
            axis=0,
        )

        line, = ax.plot(
            t,
            mean,
            linewidth=1.5,
            label=alg,
        )
        ax.fill_between(
            t,
            q25,
            q75,
            alpha=0.15,
        )

    ax.set_xlabel(
        "Response threshold, t"
    )
    ax.set_ylabel(
        "Response support fraction, P(E > t)"
    )
    ax.set_xlim(
        0.0,
        1.0,
    )
    ax.set_ylim(
        0.0,
        1.0,
    )
    ax.grid(
        alpha=0.20,
        linewidth=0.6,
    )
    ax.text(
        0.02,
        0.98,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # (c) valid-signal edge density
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    draw_violin_box(
        ax,
        df,
        "edge_density_valid_signal",
        "Valid-signal edge density",
    )
    ax.text(
        0.02,
        0.98,
        "(c)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # (d) fragmentation
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    draw_violin_box(
        ax,
        df,
        "fragmentation_components_per_100px",
        "Fragments per 100 skeleton pixels",
    )
    ax.text(
        0.02,
        0.98,
        "(d)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # Figure-level note, kept concise.
    fig.suptitle(
        "Structural support characteristics of SLoG, LoG, and Canny",
        fontsize=9.5,
    )

    for suffix in [
        "png",
        "pdf",
        "tiff",
    ]:
        out = (
            out_dir
            / f"figure2_prior_support.{suffix}"
        )
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches="tight",
        )
        print(
            f"[saved] {out}"
        )

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot 500-profile SLoG/LoG/Canny "
            "structural-support characterization."
        )
    )

    p.add_argument("--metrics_csv",type=str,required=True,)
    p.add_argument("--params_json",type=str,required=True,)
    p.add_argument("--image_dir",type=str,required=True,help=("Directory containing the same 500 images. The CSV image_path is used as a fallback."),)
    p.add_argument("--out_dir",type=str,required=True,)

    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device used only for SGS tensor diffusion. "
            "Use cpu if desired."
        ),
    )

    p.add_argument(
        "--expected_profiles",
        type=int,
        default=500,
    )
    p.add_argument(
        "--hist_bins",
        type=int,
        default=100,
    )
    p.add_argument(
        "--support_points",
        type=int,
        default=101,
    )
    p.add_argument(
        "--hist_y_scale",
        type=str,
        default="log",
        choices=[
            "log",
            "linear",
        ],
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--reuse_cache",
        action="store_true",
        help=(
            "Reuse response_curve_cache.npz "
            "after verifying profile names and parameter hash."
        ),
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(message)s"
        ),
    )

    metrics_csv = Path(
        args.metrics_csv
    ).expanduser().resolve()
    params_json = Path(
        args.params_json
    ).expanduser().resolve()

    image_dir = (
        Path(args.image_dir)
        .expanduser()
        .resolve()
        if args.image_dir
        else None
    )

    out_dir = (
        Path(args.out_dir)
        .expanduser()
        .resolve()
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not metrics_csv.is_file():
        raise FileNotFoundError(
            metrics_csv
        )
    if not params_json.is_file():
        raise FileNotFoundError(
            params_json
        )

    df, image_names = audit_metrics_csv(
        metrics_csv,
        expected_profiles=int(
            args.expected_profiles
        ),
    )

    pre_cfg, det_cfg, raw_params = (
        load_configs(
            params_json,
            device_override=args.device,
        )
    )

    params_hash = sha256_file(
        params_json
    )

    print("\n[effective parameters]")
    print(
        json.dumps(
            raw_params,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nSGS compute device override: "
        f"{pre_cfg.device}"
    )

    cache_path = (
        out_dir
        / "response_curve_cache.npz"
    )

    response_data = None

    if (
        args.reuse_cache
        and cache_path.is_file()
    ):
        print(
            f"[cache] loading {cache_path}"
        )
        response_data = (
            load_response_cache(
                cache_path,
                expected_image_names=image_names,
                expected_params_sha256=params_hash,
            )
        )

    if response_data is None:
        image_paths = resolve_image_paths(
            df,
            image_names,
            image_dir=image_dir,
        )

        response_data = (
            compute_response_curves(
                image_paths=image_paths,
                image_names=image_names,
                pre_cfg=pre_cfg,
                det_cfg=det_cfg,
                hist_bins=int(
                    args.hist_bins
                ),
                support_points=int(
                    args.support_points
                ),
                progress_every=int(
                    args.progress_every
                ),
            )
        )

        save_response_cache(
            cache_path,
            response_data,
            image_names=image_names,
            params_sha256=params_hash,
        )
        print(
            f"[cache] saved {cache_path}"
        )

        response_data[
            "per_profile"
        ].to_csv(
            out_dir
            / "response_per_profile_summary.csv",
            index=False,
        )

    # Save curve summaries.
    hist_summary = curve_summary(
        response_data["bin_centers"],
        response_data["hists"],
        x_name="response_bin_center",
    )
    hist_summary.to_csv(
        out_dir
        / "response_histogram_summary.csv",
        index=False,
    )

    survival_summary = curve_summary(
        response_data["thresholds"],
        response_data["survival"],
        x_name="threshold",
    )
    survival_summary.to_csv(
        out_dir
        / "response_survival_summary.csv",
        index=False,
    )

    metric_summary = (
        existing_metric_summary(
            df
        )
    )
    metric_summary.to_csv(
        out_dir
        / "figure2_metric_summary.csv",
        index=False,
    )

    # Print the two panel metrics for immediate audit.
    print("\n[panel c/d descriptive statistics]")
    print(
        metric_summary[
            metric_summary["metric"].isin(
                [
                    "edge_density_valid_signal",
                    "fragmentation_components_per_100px",
                ]
            )
        ][
            [
                "algorithm",
                "metric",
                "n",
                "mean",
                "std",
                "median",
                "q25",
                "q75",
            ]
        ].to_string(
            index=False
        )
    )

    make_figure(
        df=df,
        response_data=response_data,
        out_dir=out_dir,
        hist_y_scale=args.hist_y_scale,
        dpi=int(args.dpi),
    )

    metadata = {
        "metrics_csv": str(
            metrics_csv
        ),
        "params_json": str(
            params_json
        ),
        "params_sha256": params_hash,
        "image_dir": (
            str(image_dir)
            if image_dir is not None
            else None
        ),
        "n_unique_profiles": int(
            len(image_names)
        ),
        "algorithms": list(
            ALGORITHMS
        ),
        "hist_bins": int(
            args.hist_bins
        ),
        "support_points": int(
            args.support_points
        ),
        "hist_y_scale": str(
            args.hist_y_scale
        ),
        "dpi": int(
            args.dpi
        ),
        "device_override": str(
            args.device
        ),
        "interpretation": {
            "panel_a": (
                "Response representation distribution; "
                "Canny is intrinsically binary."
            ),
            "panel_b": (
                "Profile-wise response-support persistence "
                "P(E>t); shaded band is IQR across profiles."
            ),
            "panel_c": (
                "Existing valid-signal edge density "
                "from the original 500-profile CSV."
            ),
            "panel_d": (
                "Existing fragmentation components per "
                "100 skeleton pixels from the original CSV."
            ),
            "warning": (
                "Response amount alone is not treated "
                "as a quality or information-density score."
            ),
        },
    }

    with (
        out_dir
        / "run_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nDone. Outputs: {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
