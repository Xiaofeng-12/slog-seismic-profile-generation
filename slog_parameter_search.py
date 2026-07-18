import os
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any
import itertools
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from scipy import fftpack, ndimage
from scipy.ndimage import gaussian_filter

from skimage import restoration, feature,img_as_ubyte
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.ndimage import uniform_filter

import torch
import torchvision
from torchvision import models
import torch.nn.functional as F

import lpips

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def call_with_channel_arg(func, *args, channel_axis=None, multichannel=False, **kwargs):
    try:
        return func(*args, channel_axis=channel_axis, **kwargs)
    except TypeError:
        return func(*args, multichannel=multichannel, **kwargs)
  
def estimate_sigma_compat(img: np.ndarray) -> np.ndarray:
    return call_with_channel_arg(restoration.estimate_sigma, img, channel_axis=None, multichannel=False)
    
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def sanitize_filename(s: str, maxlen: int = 200) -> str:
    if not isinstance(s, str):
        s = str(s)
    safe = "".join([c if (c.isalnum() or c in "._=-+") else '_' for c in s])
    if len(safe) > maxlen:
        safe = safe[:maxlen]
    return safe

def load_image(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    return arr

def save_image(path: str, arr: np.ndarray):
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0, 1)
    im = Image.fromarray((arr * 255).astype(np.uint8))
    im.save(path)

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        return img[..., 0] * 0.2989 + img[..., 1] * 0.5870 + img[..., 2] * 0.1140
    return img

def denoise_nlmeans(img: np.ndarray, patch_size: int = 5, patch_distance: int = 6, h: float = 0.08) -> np.ndarray:
    sigma_est = estimate_sigma_compat(img)
    try:
        sigma_val = float(np.mean(sigma_est))
    except Exception:
        sigma_val = float(sigma_est)
    den = call_with_channel_arg(
        restoration.denoise_nl_means,
        img,
        h=h * sigma_val,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None,
        multichannel=False,
        preserve_range=True
    )
    den = np.clip(den, 0, 1)
    return den

def _torch_construct_diffusion_tensor_from_ori(coh: torch.Tensor,
                                              ori: torch.Tensor,
                                              lambda_parallel: float = 1.0,
                                              lambda_perp: float = 1e-3,
                                              coherence_influence: float = 1.0,
                                              eps: float = 1e-12):
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

    vx = torch.cos(ori)
    vy = torch.sin(ori)

    lam_par = float(lambda_parallel)
    lam_per = float(lambda_perp)
    lam_per_map = lam_per * (1.0 - coherence_influence * coh)
    lam_per_map = torch.clamp(lam_per_map, min=1e-8)

    Dxx = lam_par * (vx * vx) + lam_per_map * (vy * vy)
    Dyy = lam_par * (vy * vy) + lam_per_map * (vx * vx)
    Dxy = (lam_par - lam_per_map) * (vx * vy)

    if orig_ndim == 2:
        return Dxx.squeeze(0).squeeze(0), Dxy.squeeze(0).squeeze(0), Dyy.squeeze(0).squeeze(0)
    else:
        return Dxx, Dxy, Dyy

def weickert_tensor_diffusion_torch(img: np.ndarray,
                                    coherence: np.ndarray,
                                    orientation: np.ndarray,
                                    niter: int = 1,
                                    gamma: float = 0.06,
                                    lambda_parallel: float = 1.0,
                                    lambda_perp: float = 1e-3,
                                    coherence_influence: float = 1.0,
                                    device: torch.device = None,
                                    return_tensor_components: bool = False,
                                    prefer_fp16: bool = False) -> np.ndarray:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t_img = torch.from_numpy(img.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    t_coh = torch.from_numpy(coherence.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)
    t_ori = torch.from_numpy(orientation.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)

    use_fp16 = prefer_fp16 and (device.type == "cuda")
    dtype = torch.float16 if use_fp16 else torch.float32
    t_img = t_img.to(dtype=dtype)
    t_coh = t_coh.to(dtype=dtype)
    t_ori = t_ori.to(dtype=dtype)

    kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device, dtype=dtype)  # (1,1,1,3)
    ky = torch.tensor([[[[-0.5],[0.0],[0.5]]]], device=device, dtype=dtype)  # (1,1,3,1)

    I = t_img
    for it in range(int(niter)):
        Dxx, Dxy, Dyy = _torch_construct_diffusion_tensor_from_ori(t_coh, t_ori,
                                                                    lambda_parallel=lambda_parallel,
                                                                    lambda_perp=lambda_perp,
                                                                    coherence_influence=coherence_influence)
        Dxx = Dxx.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0) if Dxx.dim()==2 else Dxx.to(device=device, dtype=dtype)
        Dxy = Dxy.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0) if Dxy.dim()==2 else Dxy.to(device=device, dtype=dtype)
        Dyy = Dyy.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0) if Dyy.dim()==2 else Dyy.to(device=device, dtype=dtype)

        I_p_x = F.pad(I, (1,1,0,0), mode='replicate')  # pad W dims
        gx = F.conv2d(I_p_x, kx, padding=0)  # 1x1xHxW

        I_p_y = F.pad(I, (0,0,1,1), mode='replicate')
        gy = F.conv2d(I_p_y, ky, padding=0)  # 1x1xHxW

        Jx = Dxx * gx + Dxy * gy
        Jy = Dxy * gx + Dyy * gy

        Jx_p = F.pad(Jx, (1,1,0,0), mode='replicate')
        dJx_dx = F.conv2d(Jx_p, kx, padding=0)
        Jy_p = F.pad(Jy, (0,0,1,1), mode='replicate')
        dJy_dy = F.conv2d(Jy_p, ky, padding=0)

        div = dJx_dx + dJy_dy
        I = I + float(gamma) * div

        if use_fp16:
            I = torch.clamp(I, -1e3, 1e3)
        I = torch.nan_to_num(I, nan=0.0, posinf=1e9, neginf=-1e9)

    out = I.squeeze(0).squeeze(0).to(dtype=torch.float32, device='cpu').numpy()
    if return_tensor_components:
        def _to_np(t):
            tt = t
            if isinstance(tt, torch.Tensor):
                tt = tt.squeeze().detach().to(dtype=torch.float32, device='cpu').numpy()
            return np.array(tt, dtype=np.float32)

        Dxx_np = _to_np(Dxx)
        Dxy_np = _to_np(Dxy)
        Dyy_np = _to_np(Dyy)
        return out, Dxx_np, Dxy_np, Dyy_np
    return out

def structure_guided_smoothing(img: np.ndarray,
                               sigma: float = 0.4,
                               anisotropy: float = 2.0,
                               iterations: int = 1,
                               base_kappa: float = 10.0,
                               gamma: float = 0.06,
                               option: int = 1,
                               lambda_perp_base: float = 1e-3,
                               lambda_par_base: float = 1.0,
                               coherence_influence: float = 1.0,
                               device: torch.device = None,
                               prefer_fp16: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    
    res = weickert_tensor_diffusion_torch(img.astype(np.float32),
                                         coherence.astype(np.float32),
                                         orientation.astype(np.float32),
                                         niter=int(iterations),
                                         gamma=float(gamma),
                                         lambda_parallel=float(lambda_par_base),
                                         lambda_perp=float(lambda_perp_base),
                                         coherence_influence=float(coherence_influence),
                                         device=device,
                                         prefer_fp16=prefer_fp16,
                                         return_tensor_components=True)
    if isinstance(res, tuple) and len(res) == 4:
        img_sm, Dxx_np, Dxy_np, Dyy_np = res
    else:
        img_sm = res
        Dxx_np = np.zeros_like(img_sm)
        Dxy_np = np.zeros_like(img_sm)
        Dyy_np = np.zeros_like(img_sm)
    img_sm = img_sm - float(np.nanmin(img_sm))
    mx = float(np.nanmax(img_sm))
    if mx > 0:
        img_sm = img_sm / mx
    else:
        img_sm = img_sm * 0.0
    try:
        Dxx_np = Dxx_np.astype(np.float32)
        Dxy_np = Dxy_np.astype(np.float32)
        Dyy_np = Dyy_np.astype(np.float32)
        if Dxx_np.shape != img_sm.shape:
            Dxx_np = Dxx_np[:img_sm.shape[0], :img_sm.shape[1]]
            Dxy_np = Dxy_np[:img_sm.shape[0], :img_sm.shape[1]]
            Dyy_np = Dyy_np[:img_sm.shape[0], :img_sm.shape[1]]
    except Exception:
        Dxx_np = np.zeros_like(img_sm, dtype=np.float32)
        Dxy_np = np.zeros_like(img_sm, dtype=np.float32)
        Dyy_np = np.zeros_like(img_sm, dtype=np.float32)

    return img_sm.astype(np.float32), coherence.astype(np.float32), orientation.astype(np.float32),Dxx_np, Dxy_np, Dyy_np

def agc_local_rms(img: np.ndarray, window: int = 131, eps: float = 1e-8,p_low: int = 1) -> np.ndarray:
    try:
        p_low = int(p_low)
    except Exception:
        logging.warning(f"agc_local_rms: invalid p_low={p_low}, resetting to 1")
        p_low = 1
    if p_low < 0:
        logging.warning(f"agc_local_rms: p_low {p_low} < 0, clamping to 0")
        p_low = 0
    if p_low > 100:
        logging.warning(f"agc_local_rms: p_low {p_low} > 100, clamping to 100")
        p_low = 100
        
    p_high = 100 - p_low
        
    img2 = img ** 2
    kernel = np.ones((window, window)) / (window * window)
    local_mean_sq = ndimage.convolve(img2, kernel, mode='reflect')
    local_rms = np.sqrt(local_mean_sq + eps)
    out = img / (local_rms + eps)
    try:
        p_low_val, p_high_val = np.percentile(out, (float(p_low), float(p_high)))
    except Exception:
        logging.warning("agc_local_rms: percentile computation failed; falling back to 1/99 percentiles")
        p_low_val, p_high_val = np.percentile(out, (1.0, 99.0))
        
    if p_high_val - p_low_val <= 1e-12:
        p_low_val, p_high_val = np.min(out), np.max(out)
    out = (out - p_low_val) / (p_high_val - p_low_val + 1e-12)
    out = np.clip(out, 0, 1)
    return out

def _make_anisotropic_gaussian_kernel(sx: float, sy: float, angle_rad: float, size: int):
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    xr =  ca * x + sa * y
    yr = -sa * x + ca * y
    g = np.exp(-0.5 * ((xr / (sx + 1e-12))**2 + (yr / (sy + 1e-12))**2))
    g = g / (np.sum(g) + 1e-12)
    return g

def _laplacian_of_kernel(k):
    lap = ndimage.laplace(k, mode='reflect')
    lap = lap - np.mean(lap)
    return lap

def generalized_oriented_log_factory(orientation_map: np.ndarray = None,
                                     coherence_map: np.ndarray = None,
                                     dxx_map: np.ndarray = None,
                                     dxy_map: np.ndarray = None,
                                     dyy_map: np.ndarray = None,
                                     sigma: float = 1.0,
                                     anisotropy: float = 2.0,
                                     coherence_influence: float = 1.0,
                                     n_angles: int = 16,
                                     size_factor: float = 6.0,
                                     sharpness: float = 4.0,
                                     alpha: float = 0.9,
                                     normalize_out: bool = True,
                                     aniso_clip: float = 4.0,
                                     gamma_scale: float = 1.0,
                                     use_absolute_sigma: bool = False,
                                     dist_mode: str = "relative"):
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    max_aniso = max(1.0, anisotropy, aniso_clip)
    ksize = int(max(3, math.ceil(size_factor * sigma * max_aniso) * 2 + 1))

    base_kernels = []
    for ang in angles:
        sx = float(sigma) * float(anisotropy)
        sy = float(sigma)
        g = _make_anisotropic_gaussian_kernel(sx=sx, sy=sy, angle_rad=float(ang), size=ksize)
        logk = _laplacian_of_kernel(g)
        norm = np.sum(np.abs(logk)) + 1e-12
        logk = (logk / norm).astype(np.float32)
        base_kernels.append(logk)

    def detector(img: np.ndarray) -> np.ndarray:
        I = img.astype(np.float32)
        H, W = I.shape[:2]
        coh = coherence_map if coherence_map is not None else np.zeros_like(I)
        ori = orientation_map if orientation_map is not None else np.zeros_like(I)

        if (dxx_map is not None) and (dxy_map is not None) and (dyy_map is not None):
            a = dxx_map.astype(np.float32)
            b = dxy_map.astype(np.float32)
            c = dyy_map.astype(np.float32)
            tmp = np.sqrt((a - c)**2 + 4.0 * (b**2) + 1e-18)
            eig1 = 0.5 * (a + c + tmp)  
            eig2 = 0.5 * (a + c - tmp)   
            eig1 = np.clip(eig1, 1e-12, None)
            eig2 = np.clip(eig2, 1e-12, None)

            tensor_ori = 0.5 * np.arctan2(2.0 * b, (a - c + 1e-12))

            r = np.sqrt(eig1 / eig2)
            r = np.clip(r, 1.0, float(aniso_clip))
            r_sigma = np.power(r, float(gamma_scale))
            sigma_par_map = float(sigma) * r_sigma
            sigma_perp_map = float(sigma) / (r_sigma + 1e-12)

            eff_ori = tensor_ori
            use_tensor = True
        else:
            eff_ori = ori
            r_sigma = 1.0 + (anisotropy - 1.0) * (coherence_influence * coh)
            r_sigma = np.clip(r_sigma, 1.0, float(aniso_clip))
            sigma_par_map = float(sigma) * r_sigma
            sigma_perp_map = float(sigma) / (r_sigma + 1e-12)
            use_tensor = False

        responses = []
        for k in base_kernels:
            try:
                resp_k = ndimage.convolve(I, k, mode='reflect')
            except Exception:
                from scipy.signal import fftconvolve
                resp_k = fftconvolve(I, k, mode='same')
            responses.append(resp_k .astype(np.float32))
        responses = np.stack(responses, axis=0)  

        angle_rad = angles  # (n_angles,)
        delta = angle_rad[:, None, None] - eff_ori[None, :, :]
        align = np.abs(np.cos(delta))
        if dist_mode.lower() == "relative":
            max_align = np.max(align, axis=0, keepdims=True)
            align = align / (max_align + 1e-6)
        if sharpness != 1.0:
            align = np.power(align, float(sharpness))

        base_weight = ((1.0 - alpha) + alpha * (coh[None, :, :] * align))
        local_scale_ratio = (sigma_par_map[None,:,:] / (sigma + 1e-12))
        aniso_boost = (1.0 + ((local_scale_ratio - 1.0) * (align)))
        weight = base_weight * aniso_boost

        fused = np.sum(responses * weight, axis=0)

        local_scale = np.sqrt(np.maximum(sigma_par_map, 1e-12) * np.maximum(sigma_perp_map, 1e-12)) / (float(sigma) + 1e-12)
        local_scale = np.clip(local_scale, 1e-6, 1e3)
        fused = fused * local_scale

        if normalize_out:
            fused = fused - np.nanmin(fused)
            mx = np.nanmax(fused)
            if mx > 0:
                fused = fused / (mx + 1e-12)
            else:
                fused = fused * 0.0
        return fused.astype(np.float32)

    return detector

def normalize_map(m: np.ndarray) -> np.ndarray:

    m = np.asarray(m, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(np.min(m))
    mx = float(np.max(m))
    denom = mx - mn

    if denom > 0:
        out = (m - mn) / (denom + 1e-12)
        return out.astype(np.float32)
    else:
        return np.zeros_like(m, dtype=np.float32)

def compute_mes(edge_map: np.ndarray) -> float:
    return float(np.mean(edge_map))

def _compute_semblance_map(img: np.ndarray, win_t: int = 11, win_x: int = 11, eps: float = 1e-9) -> np.ndarray:

    I = img.astype(np.float32)
    kernel = np.ones((int(win_t), int(win_x)), dtype=np.float32)
    sum1 = ndimage.convolve(I, kernel, mode='reflect')
    sum2 = ndimage.convolve(I * I, kernel, mode='reflect')
    N = float(win_t * win_x)
    s = (sum1 * sum1) / (N * sum2 + eps)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.clip(s, 0.0, 1.0)
    return s.astype(np.float32)

def compute_semblance_stats(img: np.ndarray, win_t: int = 11, win_x: int = 11) -> Tuple[float, float, np.ndarray]:
    sm = _compute_semblance_map(img, win_t, win_x)
    meanv = float(np.nanmean(sm))
    p25 = float(np.nanpercentile(sm, 25.0))
    return meanv, p25, sm

def compute_orientation_consistency_from_map(orientation_map: np.ndarray, win: int = 11) -> Tuple[float, np.ndarray]:

    ori = np.array(orientation_map, dtype=np.float32)
    cos2 = np.cos(2.0 * ori)
    sin2 = np.sin(2.0 * ori)
    mean_cos2 = uniform_filter(cos2, size=win)
    mean_sin2 = uniform_filter(sin2, size=win)
    R = np.sqrt(mean_cos2 * mean_cos2 + mean_sin2 * mean_sin2)
    R = np.clip(R, 0.0, 1.0)
    meanR = float(np.nanmean(R))
    return meanR, R.astype(np.float32)

def _binarize_saliency_map(
    smap: np.ndarray,
    mask: np.ndarray = None,
    pct: float = 92.0,
    min_positive: float = 1e-6
) -> np.ndarray:

    s = np.asarray(smap, dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        m = mask.astype(bool)
        vals = s[m]
    else:
        vals = s.reshape(-1)

    vals_pos = vals[vals > float(min_positive)]
    if vals_pos.size > 0:
        vals_use = vals_pos
    else:
        vals_use = vals

    if vals_use.size == 0:
        return np.zeros_like(s, dtype=np.uint8)

    thr = float(np.nanpercentile(vals_use, float(pct)))
    if not np.isfinite(thr):
        return np.zeros_like(s, dtype=np.uint8)

    bw = (s >= thr)

    if mask is not None:
        bw = bw & mask.astype(bool)

    bw = ndimage.binary_opening(bw, structure=np.ones((3, 3)))
    bw = ndimage.binary_closing(bw, structure=np.ones((3, 3)))
    return bw.astype(np.uint8)

def compute_fault_saliency_iou(
    pre: Dict[str, np.ndarray],
    proc_det_map: np.ndarray,
    merged_det_cfg: Dict[str, Any] = None,
    mask: np.ndarray = None,
    pct_threshold: float = 92.0
) -> float:

    if merged_det_cfg is None:
        merged_det_cfg = {}

    pm = normalize_map(np.asarray(proc_det_map, dtype=np.float32))

    try:
        agc = pre.get("agc", None)
        if agc is None:
            agc = pre.get("sgs", pre.get("gray"))
        ori = pre.get("orientation", None)
        coh = pre.get("coherence", None)
        Dxx = pre.get("Dxx", None)
        Dxy = pre.get("Dxy", None)
        Dyy = pre.get("Dyy", None)

        ref_det = apply_detector_generalized_log(agc, ori, coh, Dxx, Dxy, Dyy, merged_det_cfg)
        rm = normalize_map(np.asarray(ref_det, dtype=np.float32))
    except Exception:
        return float(np.nan)

    bw_ref = _binarize_saliency_map(rm, mask=mask, pct=pct_threshold)
    bw_proc = _binarize_saliency_map(pm, mask=mask, pct=pct_threshold)

    if mask is not None:
        m = mask.astype(bool)
        inter = float(np.logical_and(bw_ref, bw_proc)[m].sum())
        union = float(np.logical_or(bw_ref, bw_proc)[m].sum())
    else:
        inter = float(np.logical_and(bw_ref, bw_proc).sum())
        union = float(np.logical_or(bw_ref, bw_proc).sum())

    if union <= 0:
        return float(np.nan)
    return float(inter / union)

def compute_autocorr_length_mean(img: np.ndarray, mask: np.ndarray = None, max_lag: int = 128, threshold: float = 1.0/np.e, eps: float = 1e-12) -> float:
    I = img.astype(np.float32)
    H, W = I.shape
    result_lags = []

    if mask is None:
        mask_cols = np.ones(W, dtype=bool)
    else:
        mask_cols = np.array([mask[:, c].sum() > 0 for c in range(W)], dtype=bool)

    for c in range(W):
        if not mask_cols[c]:
            continue
        tr = I[:, c].astype(np.float32)
        tr = tr - np.mean(tr)
        var = np.dot(tr, tr)
        if var <= eps:
            result_lags.append(0.0)
            continue
        acf = np.correlate(tr, tr, mode='full')[H-1:]
        acf = acf / (acf[0] + eps)
        # find first lag where <= threshold
        lag_idx = np.where(acf <= threshold)[0]
        if lag_idx.size == 0:
            lag = float(min(max_lag, H-1))
        else:
            lag = float(min(int(lag_idx[0]), max_lag))
        result_lags.append(lag)
    if len(result_lags) == 0:
        return float(np.nan)
    return float(np.mean(result_lags))

def local_background_zscore(ref: np.ndarray, proc: np.ndarray, mask: np.ndarray = None, patch_size: int = 64):
    H, W = ref.shape
    ref_norm = np.zeros_like(ref, dtype=np.float32)
    proc_norm = np.zeros_like(proc, dtype=np.float32)
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            ys, ye = y, min(H, y + patch_size)
            xs, xe = x, min(W, x + patch_size)
            r_patch = ref[ys:ye, xs:xe].astype(np.float32)
            p_patch = proc[ys:ye, xs:xe].astype(np.float32)
            if mask is not None:
                m_patch = mask[ys:ye, xs:xe]
                if np.sum(m_patch) < max(4, 0.05 * m_patch.size):
                    # not enough texture pixels -> skip aligning this patch, just normalize globally
                    m_patch = None
            else:
                m_patch = None
            if m_patch is not None:
                ref_bg = r_patch[m_patch.astype(bool)]
                proc_bg = p_patch[m_patch.astype(bool)]
            else:
                ref_bg = r_patch.flatten()
                proc_bg = p_patch.flatten()
            med_r = np.median(ref_bg) if ref_bg.size > 0 else np.median(r_patch)
            med_p = np.median(proc_bg) if proc_bg.size > 0 else np.median(p_patch)
            mad_r = np.median(np.abs(ref_bg - med_r)) if ref_bg.size > 0 else np.median(np.abs(r_patch - med_r))
            mad_p = np.median(np.abs(proc_bg - med_p)) if proc_bg.size > 0 else np.median(np.abs(p_patch - med_p))
            mad_r = max(mad_r, 1e-6)
            mad_p = max(mad_p, 1e-6)
            ref_norm_patch = (r_patch - med_r) / mad_r
            proc_norm_patch = (p_patch - med_p) / mad_p
            # scale to roughly 0..1 range (optional)
            ref_norm[ys:ye, xs:xe] = (ref_norm_patch - np.percentile(ref_norm_patch, 1)) / (np.percentile(ref_norm_patch, 99) - np.percentile(ref_norm_patch, 1) + 1e-12)
            proc_norm[ys:ye, xs:xe] = (proc_norm_patch - np.percentile(proc_norm_patch, 1)) / (np.percentile(proc_norm_patch, 99) - np.percentile(proc_norm_patch, 1) + 1e-12)
    ref_norm = np.clip(ref_norm, 0, 1)
    proc_norm = np.clip(proc_norm, 0, 1)
    return ref_norm.astype(np.float32), proc_norm.astype(np.float32)


def masked_structural_component(ref: np.ndarray, proc: np.ndarray, mask: np.ndarray = None, win: int = 11, eps: float = 1e-12) -> float:

    # use uniform_filter to compute local means and variances
    K = win
    mu1 = uniform_filter(ref, size=K)
    mu2 = uniform_filter(proc, size=K)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(ref * ref, size=K) - mu1_sq
    sigma2_sq = uniform_filter(proc * proc, size=K) - mu2_sq
    sigma12 = uniform_filter(ref * proc, size=K) - mu1_mu2

    denom = np.sqrt(np.maximum(sigma1_sq, 0.0) * np.maximum(sigma2_sq, 0.0)) + eps
    s_map = (sigma12 + 1e-6) / denom  # small constant to stabilize
    s_map = np.clip(s_map, -1.0, 1.0)
    if mask is None:
        return float(np.nanmean((s_map + 1.0) / 2.0))  # map to 0..1
    else:
        m = mask.astype(bool)
        if m.sum() == 0:
            return float(np.nanmean((s_map + 1.0) / 2.0))
        return float(np.nanmean(((s_map + 1.0) / 2.0)[m]))


def edge_preservation_f1(ref: np.ndarray, proc: np.ndarray, mask: np.ndarray = None, canny_sigma: float = 1.0, tol: int = 2) -> Dict[str, float]:

    try:
        ref_e = feature.canny(ref.astype(np.float32), sigma=canny_sigma)
    except Exception:
        # fallback to Sobel thresholding
        grad_ref = np.hypot(ndimage.sobel(ref, 0), ndimage.sobel(ref, 1))
        ref_e = grad_ref > np.percentile(grad_ref, 90)
    try:
        proc_e = feature.canny(proc.astype(np.float32), sigma=canny_sigma)
    except Exception:
        grad_proc = np.hypot(ndimage.sobel(proc, 0), ndimage.sobel(proc, 1))
        proc_e = grad_proc > np.percentile(grad_proc, 90)

    if tol > 0:
        structure = np.ones((2 * tol + 1, 2 * tol + 1))
        ref_dil = ndimage.binary_dilation(ref_e, structure=structure)
        proc_dil = ndimage.binary_dilation(proc_e, structure=structure)
    else:
        ref_dil = ref_e
        proc_dil = proc_e

    if mask is not None:
        m = mask.astype(bool)
        # only evaluate edges within mask bounding box
        # create masked edge arrays
        ref_eval = ref_e & m
        proc_eval = proc_e & m
        ref_dil_eval = ref_dil & m
        proc_dil_eval = proc_dil & m
    else:
        ref_eval = ref_e
        proc_eval = proc_e
        ref_dil_eval = ref_dil
        proc_dil_eval = proc_dil

    tp = float(np.logical_and(proc_eval, ref_dil_eval).sum())
    fp = float(np.logical_and(proc_eval, np.logical_not(ref_dil_eval)).sum())
    fn = float(np.logical_and(ref_eval, np.logical_not(proc_dil_eval)).sum())

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"edge_prec": float(prec), "edge_rec": float(rec), "edge_f1": float(f1)}


def lbp_hist_chi2(ref: np.ndarray, proc: np.ndarray, P: int = 8, R: int = 1, n_bins: int = 256, mask: np.ndarray = None) -> float:

    ref_u8 = img_as_ubyte(np.clip(ref, 0, 1))
    proc_u8 = img_as_ubyte(np.clip(proc, 0, 1))
    lbp_r = local_binary_pattern(ref_u8, P=P, R=R, method='uniform')
    lbp_p = local_binary_pattern(proc_u8, P=P, R=R, method='uniform')
    if mask is not None:
        m = mask.astype(bool)
        hist_r, _ = np.histogram(lbp_r[m], bins=n_bins, range=(0, n_bins))
        hist_p, _ = np.histogram(lbp_p[m], bins=n_bins, range=(0, n_bins))
    else:
        hist_r, _ = np.histogram(lbp_r, bins=n_bins, range=(0, n_bins))
        hist_p, _ = np.histogram(lbp_p, bins=n_bins, range=(0, n_bins))
    hist_r = hist_r.astype(np.float32)
    hist_p = hist_p.astype(np.float32)
    hist_r /= (hist_r.sum() + 1e-12)
    hist_p /= (hist_p.sum() + 1e-12)
    chi2 = 0.5 * np.sum(((hist_r - hist_p) ** 2) / (hist_r + hist_p + 1e-12))
    # turn into similarity in 0..1 (simple mapping)
    sim = 1.0 / (1.0 + chi2)
    return float(sim)


def glcm_stats(ref: np.ndarray, proc: np.ndarray, mask: np.ndarray = None, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels: int = 32) -> Dict[str, float]:

    def _quantize(a, levels):
        a_u8 = img_as_ubyte(np.clip(a, 0, 1))
        # scale to [0, levels-1]
        q = (a_u8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)
        return q

    r_q = _quantize(ref, levels)
    p_q = _quantize(proc, levels)

    # extract GLCM on masked area by cropping to bbox of mask (if provided)
    if mask is not None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            r_patch = r_q
            p_patch = p_q
        else:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            r_patch = r_q[y0:y1, x0:x1]
            p_patch = p_q[y0:y1, x0:x1]
    else:
        r_patch = r_q
        p_patch = p_q

    try:
        glcm_r = graycomatrix(r_patch, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        glcm_p = graycomatrix(p_patch, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        props = ['contrast', 'energy', 'correlation', 'dissimilarity']
        out = {}
        for prop in props:
            pr = graycoprops(glcm_r, prop).mean()
            pp = graycoprops(glcm_p, prop).mean()
            # similarity measure: 1 - normalized abs diff
            out[f'glcm_{prop}_sim'] = float(1.0 - abs(pr - pp) / (abs(pr) + abs(pp) + 1e-12))
        return out
    except Exception:
        return {f'glcm_{p}_sim': float(np.nan) for p in ['contrast', 'energy', 'correlation', 'dissimilarity']}


def radial_profile(M: np.ndarray, nbins: int = 40):

    h, w = M.shape
    cy, cx = h//2, w//2
    y, x = np.indices((h, w))
    r = np.hypot(y - cy, x - cx)
    r_flat = r.flatten()
    M_flat = M.flatten()
    maxr = r_flat.max()
    bins = np.linspace(0, maxr, nbins+1)
    inds = np.digitize(r_flat, bins) - 1
    prof = np.zeros(nbins, dtype=np.float32)
    for i in range(nbins):
        sel = inds == i
        if sel.sum() > 0:
            prof[i] = M_flat[sel].sum()
    if prof.sum() > 0:
        prof = prof / (prof.sum() + 1e-12)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof


def radial_profile_similarity(img1: np.ndarray, img2: np.ndarray, nbins: int = 40):

    M1 = compute_fft_magnitude(img1)
    M2 = compute_fft_magnitude(img2)
    _, p1 = radial_profile(M1, nbins=nbins)
    _, p2 = radial_profile(M2, nbins=nbins)
    # use Pearson correlation as similarity
    if np.all(p1 == 0) or np.all(p2 == 0):
        return 0.0
    corr = np.corrcoef(p1, p2)[0,1]
    return float(np.clip((corr + 1.0)/2.0, 0.0, 1.0))

def _normalize_by_percentiles(val, p_low=1.0, p_high=99.0):

    try:
        arr = np.array(val, dtype=np.float32)
        if arr.size == 1:
            # single scalar: treat as value with implicit distribution unknown -> map via identity with clamp
            return float(np.clip(arr.item(), 0.0, 1.0))
        low = np.nanpercentile(arr, p_low)
        high = np.nanpercentile(arr, p_high)
        if high - low < 1e-12:
            # fallback to min/max
            low = np.nanmin(arr)
            high = np.nanmax(arr)
        if high - low < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        out = (arr - low) / (high - low)
        out = np.clip(out, 0.0, 1.0)
        return out
    except Exception:
        try:
            v = float(val)
            return float(np.clip(v, 0.0, 1.0))
        except Exception:
            return 0.0

def compute_fft_magnitude(img: np.ndarray) -> np.ndarray:
    F = fftpack.fftshift(fftpack.fft2(img.astype(np.float32)))
    M = np.abs(F)
    M = M - M.min()
    denom = (M.max() + 1e-12)
    if denom > 0:
        M = M / denom
    return M

# --------------------------- Perceptual (LPIPS) & VGG Gram helpers -------------
def preprocess_image_once(img_rgb: np.ndarray,
                          sgs_params: Dict[str,Any],
                          agc_params: Dict[str,Any]) -> Dict[str, np.ndarray]:
    gray = to_gray(img_rgb).astype(np.float32)
    band = gray.copy().astype(np.float32)
        
    nl_h = float(sgs_params.get("nl_h", 0.8))
    nl = denoise_nlmeans(band, patch_size=5, patch_distance=6, h=nl_h).astype(np.float32)

    sgs_res, coherence_map, orientation_map, Dxx_map, Dxy_map, Dyy_map = structure_guided_smoothing(
        nl,
        sigma=float(sgs_params.get("sigma", 0.4)),
        anisotropy=float(sgs_params.get("anisotropy", 2.0)),
        iterations=int(sgs_params.get("iterations", 1)),
        base_kappa=float(sgs_params.get("base_kappa", 10.0)),
        gamma=float(sgs_params.get("gamma", 0.06)),
        device=sgs_params.get("device", None),
        prefer_fp16=sgs_params.get("prefer_fp16", False)
    )
    
    sgs_res = sgs_res.astype(np.float32)
    coherence_map = coherence_map.astype(np.float32)
    orientation_map = orientation_map.astype(np.float32)

    agc_res = agc_local_rms(sgs_res,
                            window=int(agc_params.get("window", 131)),
                            eps=float(agc_params.get("eps", 1e-8)),
                            p_low=int(agc_params.get("p1", 1))).astype(np.float32)

    return {
        "gray": gray,
        "sgs": sgs_res,
        "coherence": coherence_map,
        "orientation": orientation_map,
        "agc": agc_res,
        "Dxx": Dxx_map,
        "Dxy": Dxy_map,
        "Dyy": Dyy_map,
        "nl_h": nl_h
    }

def soften_for_lpips(emap: np.ndarray, blur_sigma: float = 0.3):
    emap = np.nan_to_num(emap, nan=0.0)
    soft = gaussian_filter(emap.astype(np.float32), sigma=blur_sigma)
    return normalize_map(soft)

def apply_detector_generalized_log_A(agc: np.ndarray, ori: np.ndarray, coh: np.ndarray, Dxx: np.ndarray, Dxy: np.ndarray, Dyy: np.ndarray, merged_det: Dict[str,Any]) -> np.ndarray:
    det_fn = generalized_oriented_log_factory(
        orientation_map=ori,
        coherence_map=coh,
        dxx_map=Dxx,
        dxy_map=Dxy,
        dyy_map=Dyy,
        sigma=float(merged_det.get("glog_sigma", 0.7)),
        anisotropy=float(merged_det.get("glog_anisotropy", 1.6)),
        n_angles=int(merged_det.get("glog_nangles", 8)),
        sharpness=float(merged_det.get("glog_sharpness", 4.0)),
        alpha=float(merged_det.get("glog_alpha", 0.97)),
        normalize_out=True,
        dist_mode=str(merged_det.get("glog_dist_mode", "relative"))
    )
    try:
        res = det_fn(agc.astype(np.float32))
        return normalize_map(res).astype(np.float32)
    except Exception as e:
        logging.exception(f"generalized_log detector failed: {e}")
        return np.zeros_like(agc, dtype=np.float32)
    
def apply_detector_generalized_log(agc: np.ndarray,ori: np.ndarray,coh: np.ndarray,Dxx: np.ndarray,Dxy: np.ndarray,Dyy: np.ndarray,merged_det: Dict[str, Any]) -> np.ndarray:
    det_fn = generalized_oriented_log_factory(
        orientation_map=ori,
        coherence_map=coh,
        dxx_map=Dxx,
        dxy_map=Dxy,
        dyy_map=Dyy,
        sigma=float(merged_det.get("glog_sigma", 0.7)),
        anisotropy=float(merged_det.get("glog_anisotropy", 1.6)),
        n_angles=int(merged_det.get("glog_nangles", 8)),
        sharpness=float(merged_det.get("glog_sharpness", 4.0)),
        alpha=float(merged_det.get("glog_alpha", 0.97)),
        normalize_out=True,
        dist_mode=str(merged_det.get("glog_dist_mode", "relative"))
    )

    try:
        res = det_fn(agc.astype(np.float32))
        res_np = np.asarray(res, dtype=np.float32)

        if res_np.ndim == 4 and res_np.shape[1] == 1:
            res_np = res_np[:, 0, ...]
        if res_np.ndim == 3 and res_np.shape[0] == 1:
            res_np = res_np[0]
        if res_np.ndim == 3 and res_np.shape[-1] == 1:
            res_np = res_np[..., 0]

        img_back = res_np.astype(np.float32)
        img_back = np.nan_to_num(img_back, nan=0.0, posinf=0.0, neginf=0.0)

        lo = float(np.nanpercentile(img_back, 5))    
        hi = float(np.nanpercentile(img_back, 99))    

        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-12:
            return np.zeros_like(agc, dtype=np.float32)

        img_back = (img_back - lo) / (hi - lo + 1e-12)
        img_back = np.clip(img_back, 0.0, 1.0)

        T = float(merged_det.get("edge_thr", 0.35))
        if np.isfinite(T) and T > 0:
            img_back[img_back < T] = 0.0
        img_back = np.nan_to_num(img_back, nan=0.0, posinf=0.0, neginf=0.0)
        return img_back.astype(np.float32)

    except Exception as e:
        logging.exception(f"generalized_log detector failed: {e}")
        return np.zeros_like(agc, dtype=np.float32)
    
def apply_detector_log(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:

    sigma_log = float(cfg.get("log_sigma", 1.0))
    sm = gaussian_filter(img.astype(np.float32), sigma=sigma_log)
    lap = ndimage.laplace(sm)
    lap = np.abs(lap)
    return normalize_map(lap).astype(np.float32)

def apply_detector_sobel(img: np.ndarray) -> np.ndarray:

    gx = ndimage.sobel(img.astype(np.float32), axis=1)
    gy = ndimage.sobel(img.astype(np.float32), axis=0)
    mag = np.hypot(gx, gy)
    return normalize_map(mag).astype(np.float32)


def apply_detector_canny(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:

    sigma_c = float(cfg.get("canny_sigma", 1.0))
    low = float(cfg.get("canny_low", 0.1))
    high = float(cfg.get("canny_high", 0.3))
    try:
        edges = feature.canny(img_as_ubyte(np.clip(img, 0, 1)),
                              sigma=sigma_c,
                              low_threshold=low,
                              high_threshold=high)
    except Exception:
        edges = feature.canny(img_as_ubyte(np.clip(img, 0, 1)), sigma=sigma_c)
    return edges.astype(np.float32)

# ---- batch LPIPS & VGG Gram computation (一次 LPIPS forward + 一次 VGG forward) ----
def compute_lpips_and_vgg_gram_metrics_batch(ref_gray: np.ndarray,
                                             emap_list: List[np.ndarray],
                                             lpips_model,
                                             vgg_model,
                                             vgg_layer_indices,
                                             device: torch.device,
                                             target_size: int = 224,
                                             precomputed_ref_vgg: torch.Tensor = None,
                                             precomputed_feats_ref: List[torch.Tensor] = None) -> List[Dict[str, float]]:
    B = len(emap_list)
    if B == 0:
        return []

    def np_list_to_tensor3(np_arrs: List[np.ndarray]):
        proc3 = []
        for a in np_arrs:
            if a.ndim == 2:
                a3 = np.stack([a, a, a], axis=2)
            elif a.ndim == 3 and a.shape[2] == 3:
                a3 = a
            else:
                raise ValueError("emap shape unsupported")
            proc3.append(a3)
        arr = np.stack(proc3, axis=0)  # B x H x W x 3
        t = torch.from_numpy(arr.transpose(0,3,1,2)).to(dtype=torch.float32, device=device, non_blocking=True)
        if t.shape[-2] != target_size or t.shape[-1] != target_size:
            t = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return t

    batch_proc = np_list_to_tensor3(emap_list)  # Bx3xHxW
    ref3 = np.stack([ref_gray, ref_gray, ref_gray], axis=2) if ref_gray.ndim == 2 else ref_gray
    ref_t = torch.from_numpy(ref3.transpose(2,0,1)).unsqueeze(0).to(dtype=torch.float32, device=device, non_blocking=True)
    if ref_t.shape[-2] != target_size or ref_t.shape[-1] != target_size:
        ref_t_resized = F.interpolate(ref_t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    else:
        ref_t_resized = ref_t

    with torch.no_grad():
        t_ref_lpips_rep = (ref_t_resized * 2.0 - 1.0).repeat(batch_proc.shape[0], 1, 1, 1)  # Bx3xH'xW'
        t_proc_lpips = (batch_proc * 2.0 - 1.0)
        try:
            lpips_out = lpips_model(t_ref_lpips_rep, t_proc_lpips)
            lpips_vals = lpips_out.view(-1).cpu().numpy().tolist()
        except Exception as e:
            logging.exception(f"LPIPS batch forward failed: {e}")
            lpips_vals = [float("nan")] * batch_proc.shape[0]

    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        t_proc_vgg = (batch_proc - mean) / std  # B x 3 x H' x W'
        if precomputed_feats_ref is not None:
            feats_ref = precomputed_feats_ref
        else:
            if precomputed_ref_vgg is not None:
                t = precomputed_ref_vgg.to(device=device)
                if (t.min() >= 0.0) and (t.max() <= 1.05):
                    t_ref_vgg = (t - mean) / std
                else:
                    t_ref_vgg = t
            else:
                t_ref_vgg = (ref_t_resized - mean) / std
            feats_ref = extract_vgg_features(t_ref_vgg, vgg_model, vgg_layer_indices)
        feats_proc = extract_vgg_features(t_proc_vgg, vgg_model, vgg_layer_indices)

    per_layer_dists = []  
    for lidx, (fr, fp) in enumerate(zip(feats_ref, feats_proc)):
        try:
            Cr, hr, wr = fr.shape[1], fr.shape[2], fr.shape[3]
            Cp, hp, wp = fp.shape[1], fp.shape[2], fp.shape[3]
            frf = fr.view(1, Cr, hr * wr)  # 1 x C x N
            fpf = fp.view(fp.shape[0], Cp, hp * wp)  # B x C x N2

            Gr = torch.bmm(frf, frf.transpose(1,2)) / (Cr * hr * wr + 1e-12)  # 1xCxC
            Gp = torch.bmm(fpf, fpf.transpose(1,2)) / (Cp * hp * wp + 1e-12)   # BxCxC
            Gr_rep = Gr.repeat(Gp.shape[0], 1, 1)

            diff = Gp - Gr_rep
            dists = torch.norm(diff.view(diff.shape[0], -1), dim=1)  # B
            per_layer_dists.append(dists.cpu().numpy())
        except Exception as e:
            logging.exception(f"VGG Gram batch compute failed at layer {lidx}: {e}")
            per_layer_dists.append(np.array([np.nan] * batch_proc.shape[0]))

    metrics_list = []
    for i in range(batch_proc.shape[0]):
        md = {}
        md["lpips"] = float(lpips_vals[i]) if lpips_vals is not None and len(lpips_vals) > i else float("nan")
        layer_vals = []
        for li, arr in enumerate(per_layer_dists):
            key = f"conv{li+1}_gram_frob"
            md[key] = float(arr[i]) if (arr is not None and len(arr) > i) else float("nan")
            layer_vals.append(md[key] if not np.isnan(md[key]) else None)
        valid = [v for v in layer_vals if v is not None]
        md["vgg_gram_avg"] = float(np.mean(valid)) if valid else float("nan")
        metrics_list.append(md)

    return metrics_list

def prepare_perceptual_models(device: torch.device = None,prefer_fp16: bool = False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        lpips_model = lpips.LPIPS(net='vgg')  
        lpips_model.to(device)
        lpips_model.eval()

        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False

        if prefer_fp16 and device.type == "cuda":
            try:
                lpips_model.half()
                vgg.half()
            except Exception:
                logging.warning("prefer_fp16 requested but model.fp16 conversion failed; continuing in fp32")

    except RuntimeError as e:
        logging.exception(f"GPU model load failed (OOM?). Falling back to CPU: {e}")
        device = torch.device("cpu")
        lpips_model = lpips.LPIPS(net='vgg').to(device)
        lpips_model.eval()
        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False

    layer_indices = [0, 5, 10, 19]
    return lpips_model, vgg, layer_indices, device

def extract_vgg_features(x: torch.Tensor, vgg_model: torch.nn.Module, layer_indices: List[int]):
    out_feats = []
    cur = x
    for i, layer in enumerate(vgg_model):
        cur = layer(cur)
        if i in layer_indices:
            out_feats.append(cur.clone())
        if i > max(layer_indices):
            break
    return out_feats

GRID_RANGES = {
    "glog_sigma": lambda: [0.4],
    "glog_anisotropy": lambda: [2.1],
    "glog_nangles": lambda: [8],
    "glog_sharpness": lambda: [14],
    "glog_alpha": lambda: [0.92],
    "glog_dist_mode": lambda: ["relative"],
    
    "sigma": lambda: [1.5],
    "anisotropy": lambda: [2.2],
    "iterations": lambda: [1],
    "gamma": lambda: [0.06],

    "nl_h": lambda: [0.8],

    "window": lambda: [31],
    "p1": lambda: [1],

    "log_sigma": lambda: [1.1],
    "canny_sigma": lambda: [1.7],
    "canny_low":   lambda: [0.1],
    "canny_high":  lambda: [0.9],
}

def generate_grid_configs(params_to_grid: List[str], default_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    vals = {}
    for k in default_config.keys():
        vals[k] = [default_config[k]]

    for p in params_to_grid:
        if p in GRID_RANGES:
            vals[p] = GRID_RANGES[p]()
        elif p in default_config:
            vals[p] = [default_config[p]]
        else:
            raise ValueError(f"No range defined for grid parameter: {p}")

    keys = list(vals.keys())
    combos = []
    for prod in itertools.product(*(vals[k] for k in keys)):
        cfg = {k: prod[i] for i, k in enumerate(keys)}
        combos.append(cfg)
    return combos

def config_id_from_cfg(cfg: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, float):
            vs = f"{v}".replace('.', 'p')
        elif isinstance(v, (list, dict)):
            vs = json.dumps(v, separators=(',', ':'))
            vs = vs.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
            vs = vs.replace(',', '-').replace(':', '-').replace(' ', '')
        else:
            vs = str(v)
        parts.append(f"{k}={vs}")
    cid = "__".join(parts)
    return cid

def compute_and_record_metrics(pre: Dict[str, np.ndarray],
                               proc_img: np.ndarray,
                               out_dir: str,
                               record: Dict[str, Any] = None,
                               save_images: bool = True,
                               rms_window: int = 31,
                               fft_dc_radius: int = 6,
                               merged_det: Dict[str, Any] = None) -> Dict[str, Any]:
    ensure_dir(out_dir)
    rec = {} if record is None else dict(record)

    gray = pre.get("gray", None)
    if gray is None:
        raise ValueError("compute_and_record_metrics requires pre['gray']")
    proc_img = proc_img.astype(np.float32)
    if proc_img.ndim == 3:
        proc_img = to_gray(proc_img)
    gray = gray.astype(np.float32)
    if gray.ndim == 3:
        gray = to_gray(gray)

    coh_map = pre.get("coherence", None)
    if coh_map is not None:
        mask = (coh_map > 0.15).astype(np.uint8)
    else:
        local_var = ndimage.generic_filter(gray, np.var, size=7)
        thresh = np.percentile(local_var, 60)
        mask = (local_var > thresh).astype(np.uint8)
    
    ref_norm, proc_norm = local_background_zscore(gray, proc_img, mask=mask, patch_size=64)

    try:
        m_ss = masked_structural_component(ref_norm, proc_norm, mask=mask, win=11)
        rec["masked_ssim_struct"] = float(m_ss)
    except Exception as e:
        logging.exception(f"masked_structural_component failed: {e}")
        rec["masked_ssim_struct"] = float(np.nan)

    try:
        e_metrics = edge_preservation_f1(ref_norm, proc_norm, mask=mask, canny_sigma=1.0, tol=2)
        rec.update({f"{k}": float(v) for k, v in e_metrics.items()})
    except Exception as e:
        logging.exception(f"edge_preservation_f1 failed: {e}")
        rec.update({"edge_prec": float(np.nan), "edge_rec": float(np.nan), "edge_f1": float(np.nan)})

    try:
        lbp_sim = lbp_hist_chi2(ref_norm, proc_norm, P=8, R=1, n_bins=64, mask=mask)
        rec["lbp_sim"] = float(lbp_sim)
    except Exception as e:
        logging.exception(f"lbp_hist_chi2 failed: {e}")
        rec["lbp_sim"] = float(np.nan)

    try:
        glcm_sim = glcm_stats(ref_norm, proc_norm, mask=mask, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32)
        for k, v in glcm_sim.items():
            rec[k] = float(v)
    except Exception as e:
        logging.exception(f"glcm_stats failed: {e}")

    try:
        radial_sim = radial_profile_similarity(gray, proc_img, nbins=40)
        rec["radial_spec_sim"] = float(radial_sim)
        if save_images:
            try:
                Mg = compute_fft_magnitude(gray)
                Mp = compute_fft_magnitude(proc_img)
                centers, pg = radial_profile(Mg, nbins=40)
                _, pp = radial_profile(Mp, nbins=40)
                plt.figure(figsize=(4,2))
                plt.plot(centers, pg, label='orig')
                plt.plot(centers, pp, label='proc')
                plt.legend()
                plt.title("Radial FFT profile")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "radial_profile.png"), dpi=150)
                plt.close()
            except Exception:
                pass
    except Exception as e:
        logging.exception(f"radial_profile_similarity failed: {e}")
        rec["radial_spec_sim"] = float(np.nan)
    try:
        smean, sp25, s_map = compute_semblance_stats(ref_norm, win_t=11, win_x=11)
        rec["semblance_mean"] = float(smean)
        rec["semblance_p25"] = float(sp25)
        if save_images:
            try:
                plt.imsave(os.path.join(out_dir, "semblance_map.png"), normalize_map(s_map), cmap='gray')
            except Exception:
                pass
    except Exception as e:
        logging.exception(f"compute_semblance_stats failed: {e}")
        rec["semblance_mean"] = float(np.nan)
        rec["semblance_p25"] = float(np.nan)

    try:
        ori = pre.get("orientation", None)
        if ori is None:
            gradx = ndimage.sobel(ref_norm, axis=1).astype(np.float32)
            grady = ndimage.sobel(ref_norm, axis=0).astype(np.float32)
            Jxx = gaussian_filter(gradx * gradx, sigma=0.5)
            Jxy = gaussian_filter(gradx * grady, sigma=0.5)
            Jyy = gaussian_filter(grady * grady, sigma=0.5)
            ori = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy + 1e-12))
        o_cons_mean, o_cons_map = compute_orientation_consistency_from_map(ori, win=11)
        rec["orientation_consistency"] = float(o_cons_mean)
        if save_images:
            try:
                plt.imsave(os.path.join(out_dir, "orientation_consistency.png"), normalize_map(o_cons_map), cmap='viridis')
            except Exception:
                pass
    except Exception as e:
        logging.exception(f"compute_orientation_consistency failed: {e}")
        rec["orientation_consistency"] = float(np.nan)

    try:
        proc_det_candidate = apply_detector_generalized_log(
            pre.get("agc", pre.get("sgs", pre.get("gray"))),
            pre.get("orientation"), pre.get("coherence"),
            pre.get("Dxx"), pre.get("Dxy"), pre.get("Dyy"),
            merged_det or {}
        )
        if proc_det_candidate.ndim == 2 and np.nanmax(proc_det_candidate) <= 1.0 and np.nanmin(proc_det_candidate) >= 0.0:
            pass
        iou_val = compute_fault_saliency_iou(pre, det_map, merged_det_cfg=merged_det, mask=mask, pct_threshold=92.0)
        rec["fault_saliency_iou"] = float(iou_val) if (iou_val is not None) else float(np.nan)
    except Exception as e:
        logging.exception(f"compute_fault_saliency_iou failed: {e}")
        rec["fault_saliency_iou"] = float(np.nan)

    try:
        acl = compute_autocorr_length_mean(ref_norm, mask=mask, max_lag=128, threshold=1.0/np.e)
        rec["autocorr_length_mean"] = float(acl)
    except Exception as e:
        logging.exception(f"compute_autocorr_length_mean failed: {e}")
        rec["autocorr_length_mean"] = float(np.nan)

    if save_images:
        try:
            if "coherence" in pre:
                save_image(os.path.join(out_dir, "coherence_map.png"), normalize_map(pre["coherence"]))
            if "orientation" in pre:
                ori_vis = (pre["orientation"] + (math.pi / 2.0)) / math.pi
                save_image(os.path.join(out_dir, "orientation_vis.png"), np.clip(ori_vis, 0, 1))
        except Exception:
            pass
    return rec

def batch_process(input_dir: str,
                  out_dir: str,
                  base_sgs_config: Dict[str, Any],
                  grid_params: List[str],
                  detector_params: Dict[str, Any] = None,
                  grid_total_batches: int = 1,
                  grid_batch_index: int = 1,
                  save_images: bool = False,
                  per_detector_mes: bool = False):
    ensure_dir(out_dir)

    configs = generate_grid_configs(grid_params, base_sgs_config)
    grid_total_batches = max(1, grid_total_batches)
    grid_batch_index = max(1, grid_batch_index)

    configs_selected = [cfg for idx, cfg in enumerate(configs) if (idx % grid_total_batches) == (grid_batch_index - 1)]
    img_paths = sorted([str(p) for p in Path(input_dir).glob("*")
                        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]])

    if detector_params is not None and isinstance(detector_params, dict):
        enabled_detectors = detector_params.get("enabled", ["generalized_log"])
    else:
        enabled_detectors = ["generalized_log"]
    enabled_detectors = list(dict.fromkeys(enabled_detectors))
    if len(enabled_detectors) != 1:
        raise ValueError(f"本脚本设计为一次只跑一个算法，但当前 enabled_detectors = {enabled_detectors}")
    current_detector = enabled_detectors[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_model, vgg_model, vgg_layer_indices, device = prepare_perceptual_models(device)

    fixed_agc = {
        "window": int(base_sgs_config.get("window", 131)),
        "eps": float(base_sgs_config.get("eps", 1e-8)),
        "p1": int(base_sgs_config.get("p1", 1))
    }

    all_records = []

    for cfg in tqdm(configs_selected, desc="Grid configs (heavy pass)"):
        cid = config_id_from_cfg(cfg)
        safe_cid = sanitize_filename(cid)
        cfg_out_dir = Path(out_dir) / safe_cid
        ensure_dir(str(cfg_out_dir))
                
        # merged detector params (only glog-related kept)
        merged_det = {}
        for k in cfg.keys():
            if k.startswith(("glog_", "log_", "canny_", "sobel_")):
                merged_det[k] = cfg[k]

        for p in tqdm(img_paths, desc=f"Images [{cid}]", leave=False):
            try:
                img = load_image(p)
                base = Path(p).stem
                save_dir = cfg_out_dir / base
                ensure_dir(str(save_dir))

                # preprocess
                sgs_tuple = (float(cfg.get("sigma", base_sgs_config["sigma"])),
                             float(cfg.get("anisotropy", base_sgs_config["anisotropy"])),
                             int(cfg.get("iterations", base_sgs_config["iterations"])),
                             float(cfg.get("gamma", base_sgs_config["gamma"])))
                cache_key = (p, sgs_tuple)
                # no global preprocess cache in this simplified script (could add if needed)
                sgs_params_local = {
                    "sigma": float(cfg.get("sigma", base_sgs_config.get("sigma", 0.4))),
                    "anisotropy": float(cfg.get("anisotropy", base_sgs_config.get("anisotropy", 2.0))),
                    "iterations": int(cfg.get("iterations", base_sgs_config.get("iterations", 1))),
                    "gamma": float(cfg.get("gamma", base_sgs_config.get("gamma", 0.06))),
                    "nl_h": float(cfg.get("nl_h", base_sgs_config.get("nl_h", 0.8))),
                    "device": base_sgs_config.get("device", None),
                    "prefer_fp16": base_sgs_config.get("prefer_fp16", False)
                }

                agc_local = {
                    "window": int(cfg.get("window", base_sgs_config.get("window", 131))),
                    "eps": float(cfg.get("eps", base_sgs_config.get("eps", 1e-8))),
                    "p1": int(cfg.get("p1", base_sgs_config.get("p1", 1)))
                }

                pre = preprocess_image_once(img,
                                            sgs_params=sgs_params_local,
                                            agc_params=agc_local)
                gray = pre.get("gray", to_gray(img).astype(np.float32))
                agc = pre.get("agc", None)
                if agc is None:
                    agc = agc_local_rms(gray, window=agc_local["window"], eps=agc_local["eps"], p_low=agc_local["p1"])

                # save some pre images
                if save_images:
                    try:
                        save_image(str(save_dir / "gray.png"), gray)
                        save_image(str(save_dir / "sgs.png"), pre.get("sgs", gray))
                        save_image(str(save_dir / "agc.png"), agc)
                        if "coherence" in pre:
                            save_image(str(save_dir / "coherence.png"), normalize_map(pre["coherence"]))
                        if "orientation" in pre:
                            ori_vis = (pre["orientation"] + (math.pi / 2.0)) / math.pi
                            save_image(str(save_dir / "orientation.png"), np.clip(ori_vis, 0, 1))
                    except Exception as e:
                        logging.warning(f"Failed to save preprocess images for {base}: {e}")

                ori = pre.get("orientation", None)
                coh = pre.get("coherence", None)
                Dxx = pre.get("Dxx", None)
                Dxy = pre.get("Dxy", None)
                Dyy = pre.get("Dyy", None)
                det_maps: Dict[str, np.ndarray] = {}

                if "generalized_log" in enabled_detectors:
                    det_glog = apply_detector_generalized_log(agc, ori, coh, Dxx, Dxy, Dyy, merged_det)
                    det_maps["generalized_log"] = det_glog

                if "log" in enabled_detectors:
                    det_log = apply_detector_log(agc, cfg)
                    det_maps["log"] = det_log

                if "sobel" in enabled_detectors:
                    det_sobel = apply_detector_sobel(agc)
                    det_maps["sobel"] = det_sobel

                if "canny" in enabled_detectors:
                    det_canny = apply_detector_canny(agc, cfg)
                    det_maps["canny"] = det_canny

                if current_detector not in det_maps:
                    raise ValueError(f"Detector {current_detector} 没有生成对应的 det_map，请检查上面的 if 分支。")
                det_map = det_maps[current_detector]

                if save_images:
                    for name, dmap in det_maps.items():
                        try:
                            save_image(str(save_dir / f"{name}.png"), normalize_map(dmap))
                            np.save(str(save_dir / f"{name}.npy"), dmap.astype(np.float32))
                        except Exception as e:
                            logging.warning(f"Failed to save {name} images: {e}")

                emap_list = [soften_for_lpips(det_map, blur_sigma=0.3)]
                try:
                    ref3 = np.stack([agc, agc, agc], axis=2) if agc.ndim == 2 else agc
                    ref_t = torch.from_numpy(ref3.transpose(2,0,1)).unsqueeze(0).to(dtype=torch.float32, device=device, non_blocking=True)
                    if ref_t.shape[-2] != 224 or ref_t.shape[-1] != 224:
                        ref_t_resized = F.interpolate(ref_t, size=(224,224), mode='bilinear', align_corners=False)
                    else:
                        ref_t_resized = ref_t
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
                    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
                    t_ref_vgg = (ref_t_resized - mean) / std
                    with torch.no_grad():
                        feats_ref = extract_vgg_features(t_ref_vgg, vgg_model, vgg_layer_indices)
                except Exception:
                    ref_t_resized = None
                    feats_ref = None

                try:
                    batch_metrics = compute_lpips_and_vgg_gram_metrics_batch(
                        agc, emap_list, lpips_model, vgg_model, vgg_layer_indices, device, target_size=224,
                        precomputed_ref_vgg=ref_t_resized, precomputed_feats_ref=feats_ref
                    )
                    proc_metrics = batch_metrics[0] if batch_metrics else {}
                except Exception as e:
                    logging.exception(f"Perceptual metrics failed for {base} cfg {cid}: {e}")
                    proc_metrics = {}

                rec = {
                    "image": base,
                    "grid_config": json.dumps(cfg, ensure_ascii=False),
                    "detector_variant_config": json.dumps(merged_det, ensure_ascii=False),
                    "algorithms": json.dumps(["generalized_log"], ensure_ascii=False),
                    "mes_generalized_log": float(compute_mes(det_map)),
                }
                rec.update({f"lpips_generalized_log": float(proc_metrics.get("lpips", np.nan)),
                            f"vgg_gram_avg_generalized_log": float(proc_metrics.get("vgg_gram_avg", np.nan))})
                for li in range(len(vgg_layer_indices)):
                    rec[f"conv{li+1}_gram_frob_generalized_log"] = float(proc_metrics.get(f"conv{li+1}_gram_frob", np.nan))

                try:
                    rec = compute_and_record_metrics(pre, det_map, str(save_dir), rec, save_images=save_images,
                                 rms_window=31, fft_dc_radius=6, merged_det=merged_det)
                except Exception as e:
                    logging.exception(f"compute_and_record_metrics failed for {base} cfg {cid}: {e}")

                for pk, pv in merged_det.items():
                    try:
                        if isinstance(pv, (list, dict)):
                            rec[f"param_{pk}"] = json.dumps(pv, ensure_ascii=False)
                        else:
                            rec[f"param_{pk}"] = pv
                    except Exception:
                        rec[f"param_{pk}"] = str(pv)
                try:
                    rec["nl_h"] = float(sgs_params_local.get("nl_h", np.nan))
                    rec["sgs_sigma"] = float(sgs_params_local.get("sigma", np.nan))
                    rec["sgs_anisotropy"] = float(sgs_params_local.get("anisotropy", np.nan))
                    rec["sgs_iterations"] = int(sgs_params_local.get("iterations", -1))
                    rec["sgs_gamma"] = float(sgs_params_local.get("gamma", np.nan))

                    rec["agc_window"] = int(agc_local.get("window", -1))
                    rec["agc_eps"] = float(agc_local.get("eps", np.nan))
                    rec["agc_p1"] = int(agc_local.get("p1", -1))

                    if "nl_h" in pre:
                        try:
                            rec["pre_nl_h"] = float(pre.get("nl_h", np.nan))
                        except Exception:
                            rec["pre_nl_h"] = pre.get("nl_h")
                except Exception:
                    logging.exception("Failed to attach grid params into rec")
                    
                all_records.append(rec)

            except Exception as e:
                logging.exception(f"Failed processing {p} under config {cid}: {e}")

    if all_records:
        df_all = pd.DataFrame.from_records(all_records)
        def sanitize_colname(c):
            return "".join(ch if (ch.isalnum() or ch in "._") else "_" for ch in str(c))
        df_all.columns = [sanitize_colname(c) for c in df_all.columns]
        df_all.to_csv(Path(out_dir) / "metrics_summary_all_configs.csv", index=False)
    print(f"Done. Output at {out_dir}. {len(configs_selected)} configs processed.")
    
def parse_list_arg(s: str) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Seismic edge detection pipeline with LPIPS & VGG Gram per-detector metrics (GPU friendly)")
    parser.add_argument("--input_dir", required=True, help="Folder with RGB seismic images")
    parser.add_argument("--out_dir", required=True, help="Output folder")
    parser.add_argument("--grid_params", default="glog_sigma,glog_anisotropy,glog_nangles,glog_sharpness,glog_alpha,glog_dist_mode,sigma,anisotropy,iterations,gamma,nl_h,window,p1",
    help=("Comma-separated list of params to grid search. Available detector names: "
          "glog_sigma,glog_anisotropy,glog_nangles,glog_sharpness,glog_alpha,glog_dist_mode,"
          "sigma,anisotropy,iterations,gamma,nl_h,window,p1,"
          "log_sigma,canny_sigma,canny_low,canny_high,"))

    parser.add_argument("--grid_total_batches", type=int, default=1, help="Split all configs into this many batches")
    parser.add_argument("--grid_batch_index", type=int, default=1, help="Run only this batch index (1-based)")
    parser.add_argument(
        "--detector_params",
        default={"enabled":["generalized_log"]},
        help='JSON string or dict for detector params. Example:{"enabled":["generalized_log","log","sobel","canny"].Run one detector at a time.}'
    )
    parser.add_argument("--save_images",action="store_true", help="Save intermediate images (gray,nlmeans, sgs, agc).")
    parser.add_argument("--per_detector_mes", action="store_true", help="Save per-detector MES columns in CSV (can be many). Default False")
    args = parser.parse_args()

    grid_params = parse_list_arg(args.grid_params)
    base_sgs_config = {
        "base_kappa": 10.0,
        "anisotropy": 2.0,
        "iterations": 1,
        "sigma": 0.4,
        "gamma": 0.06,
        "nl_h": 0.8,
        "window": 131,
        "eps": 1e-8,
        "p1": 1
    }
    det_params = None
    if hasattr(args, "detector_params") and args.detector_params:
        if isinstance(args.detector_params, str):
            try:
                det_params = json.loads(args.detector_params)
            except Exception as e:
                parser.error(f"Invalid detector_params JSON: {e}")
        elif isinstance(args.detector_params, dict):
            det_params = args.detector_params
        else:
            # argparse may give a string repr of dict, try json.loads fallback
            try:
                det_params = json.loads(str(args.detector_params))
            except Exception:
                det_params = None
    batch_process(args.input_dir, args.out_dir,base_sgs_config, grid_params,
                  detector_params=det_params,
                  grid_total_batches=args.grid_total_batches,
                  grid_batch_index=args.grid_batch_index,
                  save_images=args.save_images,
                  per_detector_mes=args.per_detector_mes)

if __name__ == "__main__":
    main()
