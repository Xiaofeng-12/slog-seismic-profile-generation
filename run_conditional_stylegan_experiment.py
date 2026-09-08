"""
Main implementation for conditional seismic-profile synthesis.

Modes
-----
train:
    Train the conditional StyleGAN2 model.
test:
    Evaluate a trained checkpoint using the shared domain evaluator.
generate:
    Generate source-associated synthetic data for downstream segmentation.
diversity:
    Evaluate latent-code sensitivity under fixed conditions and fixed
    synthesis noise.

Experiment profiles
-------------------
--profile australian reproduces the 512x512 Australian within-volume protocol.
--profile f3 reproduces the 128x128 edge-independent F3 stress-test protocol.
The model implementation is shared; only survey-specific preprocessing,
conditioning, augmentation geometry, and discriminator patch scales differ.
"""
import os, math, random, argparse, time,json,statistics
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.transforms import functional as TF
from PIL import ImageOps
from typing import List, Dict, Tuple, Callable, Any,Optional
from scipy import ndimage
from scipy.ndimage import gaussian_filter,gaussian_laplace
import logging
from datetime import datetime
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp
import copy
import csv
import shutil
try:
    import lpips
    _have_lpips = True
except ImportError:
    lpips = None
    _have_lpips = False
    
def _dump_pt(x, name, step, dump_dir="nan_dumps"):
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"{step}_{name}_{int(time.time())}.pt")
    torch.save(x.detach().cpu(), path)
    print(f"[dump] saved: {path}")

def assert_finite_torch(x, name, step):
    if not torch.isfinite(x).all():
        print(f"[NaN/Inf] {name} at step={step} | shape={tuple(x.shape)} "
              f"| min/max={torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).min().item():.4g}/"
              f"{torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).max().item():.4g}")
        _dump_pt(x, name, step)
        raise FloatingPointError(f"{name} is NaN/Inf")
def _resolve_kid_gamma(args):
     try:
         g = float(getattr(args, "kid_gamma", -1.0))
     except Exception:
         g = -1.0
     return None if g < 0 else float(g)

def apply_g_bounds_(G, args):
    cs_min = float(getattr(args, "cond_scale_min", 0.0))
    cs_max = float(getattr(args, "cond_scale_max", 1.0))
    ns_max = float(getattr(args, "noise_strength_max", 2.0))
    gmax   = float(getattr(args, "film_gamma_max", 3.0))
    bmax   = float(getattr(args, "film_beta_max",  3.0))

    for m in G.modules():
        if isinstance(m, SynthesisBlock):
            m.cond_scale_min = cs_min
            m.cond_scale_max = cs_max
            m.noise_strength_max = ns_max
            m.film_gamma_max = gmax
            m.film_beta_max  = bmax


@torch.no_grad()
def clamp_g_params_(G, args):
    cs_min = float(getattr(args, "cond_scale_min", 0.0))
    cs_max = float(getattr(args, "cond_scale_max", 1.0))
    ns_max = float(getattr(args, "noise_strength_max", 2.0))

    for m in G.modules():
        if isinstance(m, SynthesisBlock):
            m.cond_scale.clamp_(cs_min, cs_max)
            m.noise1.clamp_(-ns_max, ns_max)
            m.noise2.clamp_(-ns_max, ns_max)

    wclip = float(getattr(args, "adapter_w_clip", 0.0))
    bclip = float(getattr(args, "adapter_b_clip", 0.0))
    if hasattr(G, "cond_adapters") and (wclip > 0 or bclip > 0):
        for a in G.cond_adapters:
            if wclip > 0:
                a.weight.clamp_(-wclip, wclip)
            if (a.bias is not None) and (bclip > 0):
                a.bias.clamp_(-bclip, bclip)
    
def setup_logger(out_dir: str, name: str = "train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    os.makedirs(out_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(out_dir, f"{name}.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, obj: dict):
        obj = dict(obj)
        obj["ts"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
try:
    from torchvision.transforms import InterpolationMode
    ROT_KW = {'interpolation': InterpolationMode.BILINEAR}
except Exception:
    ROT_KW = {'resample': Image.BILINEAR}

try:
    import cv2
    _have_cv2 = True
except Exception:
    _have_cv2 = False

try:
    from skimage.feature import canny as sk_canny
    from skimage import img_as_float,restoration
    _have_skimage = True
except Exception:
    _have_skimage = False

def call_with_channel_arg(func, *args, channel_axis=None, multichannel=False, **kwargs):
    try:
        return func(*args, channel_axis=channel_axis, **kwargs)
    except TypeError:
        return func(*args, multichannel=multichannel, **kwargs)

def estimate_sigma_compat(img: np.ndarray) -> float:
    if _have_skimage:
        try:
            return float(call_with_channel_arg(
                restoration.estimate_sigma,
                img,
                channel_axis=None,
                multichannel=False,
            ))
        except Exception as exc:
            logging.warning(
                "skimage estimate_sigma failed (%s); using robust MAD fallback",
                exc,
            )
    try:
        return float(np.median(np.abs(img - np.median(img))) * 1.4826)
    except Exception:
        return float(np.std(img))

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        return img[..., 0] * 0.2989 + img[..., 1] * 0.5870 + img[..., 2] * 0.1140
    return img

def normalize_map(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)

    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(np.min(m))
    mx = float(np.max(m))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx - mn < 1e-12):
        return np.zeros_like(m, dtype=np.float32)

    m = (m - mn) / (mx - mn + 1e-12)
    m = np.clip(m, 0.0, 1.0).astype(np.float32)
    return m

def denoise_nlmeans(img: np.ndarray, patch_size: int = 5, patch_distance: int = 6, h: float = 0.08) -> np.ndarray:
    if _have_skimage:
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
            preserve_range=True
        )
        den = np.clip(den, 0, 1)
        return den
    return np.clip(gaussian_filter(img, sigma=0.8), 0.0, 1.0)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_img = torch.from_numpy(img.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)
    t_coh = torch.from_numpy(coherence.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)
    t_ori = torch.from_numpy(orientation.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)

    use_fp16 = prefer_fp16 and (device.type == "cuda")
    dtype = torch.float16 if use_fp16 else torch.float32
    t_img = t_img.to(dtype=dtype)
    t_coh = t_coh.to(dtype=dtype)
    t_ori = t_ori.to(dtype=dtype)

    kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-0.5],[0.0],[0.5]]]], device=device, dtype=dtype)

    I = t_img
    for it in range(int(niter)):
        Dxx, Dxy, Dyy = _torch_construct_diffusion_tensor_from_ori(t_coh, t_ori,
                                                                    lambda_parallel=lambda_parallel,
                                                                    lambda_perp=lambda_perp,
                                                                    coherence_influence=coherence_influence)
        Dxx = Dxx.to(device=device, dtype=dtype) if isinstance(Dxx, torch.Tensor) else torch.tensor(Dxx, device=device, dtype=dtype)
        Dxy = Dxy.to(device=device, dtype=dtype) if isinstance(Dxy, torch.Tensor) else torch.tensor(Dxy, device=device, dtype=dtype)
        Dyy = Dyy.to(device=device, dtype=dtype) if isinstance(Dyy, torch.Tensor) else torch.tensor(Dyy, device=device, dtype=dtype)

        if Dxx.dim() == 2:
            Dxx = Dxx.unsqueeze(0).unsqueeze(0)
            Dxy = Dxy.unsqueeze(0).unsqueeze(0)
            Dyy = Dyy.unsqueeze(0).unsqueeze(0)

        I_p_x = F.pad(I, (1,1,0,0), mode='replicate')
        gx = F.conv2d(I_p_x, kx, padding=0)

        I_p_y = F.pad(I, (0,0,1,1), mode='replicate')
        gy = F.conv2d(I_p_y, ky, padding=0)

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
                               prefer_fp16: bool = False):
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
                                         coherence_influence=coherence_influence,
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

# AGC
def agc_local_rms(img: np.ndarray, window: int = 31, eps: float = 1e-8,p_low: int = 1) -> np.ndarray:
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

def _make_anisotropic_gaussian_kernel_np(sx, sy, angle_rad, size):
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
    ca = math.cos(angle_rad); sa = math.sin(angle_rad)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    g = np.exp(-0.5 * ((xr / (sx + 1e-12))**2 + (yr / (sy + 1e-12))**2))
    g = g / (g.sum() + 1e-12)
    return g.astype(np.float32)

def _laplacian_of_kernel_np(k):
    from scipy.ndimage import laplace
    lap = laplace(k, mode='reflect')
    lap = lap - lap.mean()
    return lap.astype(np.float32)

def generalized_oriented_log_factory_gpu(
                sigma: float = 1.1,
                anisotropy: float = 2.3,
                n_angles: int = 16,
                size_factor: float = 6.0,
                sharpness: float = 8.0,
                alpha: float = 0.86,
                aniso_clip: float = 4.0,
                gamma_scale: float = 1.0,
                normalize_out: bool = True,
                device: torch.device = None,
                dtype=torch.float32,
                fp16: bool = False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype_t = torch.float16 if (fp16 and device.type == 'cuda') else dtype

    max_aniso = max(1.0, anisotropy, aniso_clip)
    ksize = int(max(3, math.ceil(size_factor * sigma * max_aniso) * 2 + 1))
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    kernel_list = []
    for ang in angles:
        sx = float(sigma) * float(anisotropy)
        sy = float(sigma)
        g = _make_anisotropic_gaussian_kernel_np(sx=sx, sy=sy, angle_rad=float(ang), size=ksize)
        logk = _laplacian_of_kernel_np(g)
        norm = np.sum(np.abs(logk)) + 1e-12
        logk = (logk / norm).astype(np.float32)
        kernel_list.append(logk)
    kernels_np = np.stack(kernel_list, axis=0)
    weight = torch.from_numpy(kernels_np).to(device=device, dtype=dtype_t).unsqueeze(1)
    angle_rad_t = torch.tensor(angles, device=device, dtype=dtype_t)

    def _to_torch_and_broadcast(x, N, name=None):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x.astype(np.float32)).to(device=device, dtype=torch.float32)
        elif torch.is_tensor(x):
            t = x.to(device=device, dtype=torch.float32)
        else:
            t = torch.tensor(x, device=device, dtype=torch.float32)
        if t.dim() == 2:
            t = t.unsqueeze(0).repeat(N, 1, 1)
        elif t.dim() == 3 and t.shape[0] != N:
            if t.shape[0] == 1:
                t = t.repeat(N, 1, 1)
        return t

    def detector(batch_or_image,
                 orientation_map=None, coherence_map=None,
                 dxx=None, dxy=None, dyy=None,
                 sigma_param=sigma, anisotropy_param=anisotropy,
                 dist_mode='relative', alpha_param=alpha,
                 sharpness_param=sharpness, gamma_scale_param=gamma_scale):
        if isinstance(batch_or_image, np.ndarray):
            arr = batch_or_image.astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim == 3:
                t = torch.from_numpy(arr).to(device=device, dtype=dtype_t)
                t = t.unsqueeze(1)
            elif arr.ndim == 4:
                t = torch.from_numpy(arr).to(device=device, dtype=dtype_t)
            else:
                raise ValueError("batch_or_image np.ndarray must be 2/3/4 dims")
        elif torch.is_tensor(batch_or_image):
            t = batch_or_image.to(device=device, dtype=dtype_t)
            if t.dim() == 3:
                t = t.unsqueeze(1)
            if t.dim() == 4 and t.shape[1] != 1:
                t = t.mean(dim=1, keepdim=True)
            if t.dim() != 4:
                raise ValueError("unsupported tensor shape")
        else:
            raise ValueError("unsupported batch_or_image type")

        N, C, H, W = t.shape
        assert C == 1

        ori = _to_torch_and_broadcast(orientation_map, N)
        coh = _to_torch_and_broadcast(coherence_map, N)
        a = _to_torch_and_broadcast(dxx, N)
        b = _to_torch_and_broadcast(dxy, N)
        c = _to_torch_and_broadcast(dyy, N)

        use_tensor = (a is not None) and (b is not None) and (c is not None)

        if use_tensor:
            tmp = torch.sqrt((a - c) ** 2 + 4.0 * (b ** 2) + 1e-18)
            eig1 = 0.5 * (a + c + tmp)
            eig2 = 0.5 * (a + c - tmp)
            eig1 = torch.clamp(eig1, min=1e-12)
            eig2 = torch.clamp(eig2, min=1e-12)
            tensor_ori = 0.5 * torch.atan2(2.0 * b, (a - c + 1e-12))
            r = torch.sqrt(eig1 / eig2)
            r = torch.clamp(r, 1.0, float(aniso_clip))
            r_sigma = torch.pow(r, float(gamma_scale_param))
            sigma_par_map = float(sigma_param) * r_sigma
            sigma_perp_map = float(sigma_param) / (r_sigma + 1e-12)
            eff_ori = tensor_ori
        else:
            eff_ori = ori if ori is not None else torch.zeros((N, H, W), device=device, dtype=torch.float32)
            if coh is None:
                coh = torch.zeros((N, H, W), device=device, dtype=torch.float32)
            r_sigma = 1.0 + (anisotropy_param - 1.0) * (coh if coh is not None else 0.0)
            r_sigma = torch.clamp(r_sigma, 1.0, float(aniso_clip))
            sigma_par_map = float(sigma_param) * r_sigma
            sigma_perp_map = float(sigma_param) / (r_sigma + 1e-12)

        weight_local = weight.to(device=device, dtype=dtype_t)
        responses = F.conv2d(t, weight_local, padding=pad)  # (N, n_angles, H, W)

        angle_rad = angle_rad_t.view(1, -1, 1, 1)
        eff = eff_ori.unsqueeze(1)
        delta = angle_rad - eff
        align = torch.abs(torch.cos(delta))

        if dist_mode.lower() == 'relative':
            max_align, _ = align.max(dim=1, keepdim=True)
            align = align / (max_align + 1e-6)

        if float(sharpness_param) != 1.0:
            align = torch.pow(align, float(sharpness_param))

        coh_map = coh.unsqueeze(1) if coh is not None else torch.zeros_like(align)
        base_weight = ((1.0 - float(alpha_param)) + float(alpha_param) * (coh_map * align))

        local_scale_ratio = (sigma_par_map.unsqueeze(1) / (sigma_param + 1e-12))
        aniso_boost = (1.0 + ((local_scale_ratio - 1.0) * (align)))
        weight_map = base_weight * aniso_boost

        fused = torch.sum(responses * weight_map.to(dtype=responses.dtype), dim=1)

        local_scale = torch.sqrt(torch.clamp(sigma_par_map, min=1e-12) * torch.clamp(sigma_perp_map, min=1e-12)) / (sigma_param + 1e-12)
        fused = fused * local_scale
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        if normalize_out:
            fused = fused - fused.view(N, -1).min(dim=1)[0].view(N, 1, 1)
            mx = fused.view(N, -1).max(dim=1)[0].view(N, 1, 1)
            fused = fused / (mx + 1e-12)

        return fused.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    return detector

def apply_detector_generalized_log(agc: np.ndarray, ori: np.ndarray, coh: np.ndarray, Dxx: np.ndarray, Dxy: np.ndarray, Dyy: np.ndarray, merged_det: Dict[str,Any]) -> np.ndarray:
    device = merged_det.get("device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(str(device))
    det = generalized_oriented_log_factory_gpu(
        sigma=float(merged_det.get("glog_sigma", 1.1)),
        anisotropy=float(merged_det.get("glog_anisotropy", 2.3)),
        n_angles=int(merged_det.get("glog_nangles", 16)),
        size_factor=6.0,
        sharpness=float(merged_det.get("glog_sharpness", 8.0)),
        alpha=float(merged_det.get("glog_alpha", 0.86)),
        aniso_clip=float(merged_det.get("glog_aniso_clip", 4.0)) if merged_det.get("glog_aniso_clip") is not None else 4.0,
        gamma_scale=float(merged_det.get("glog_gamma_scale", 1.0)) if merged_det.get("glog_gamma_scale") is not None else 1.0,
        normalize_out=True,
        device=device,
        fp16=False
    )
    try:
        res_t = det(agc.astype(np.float32),
                    orientation_map=ori.astype(np.float32) if ori is not None else None,
                    coherence_map=coh.astype(np.float32) if coh is not None else None,
                    dxx=Dxx.astype(np.float32) if Dxx is not None else None,
                    dxy=Dxy.astype(np.float32) if Dxy is not None else None,
                    dyy=Dyy.astype(np.float32) if Dyy is not None else None,
                    sigma_param=float(merged_det.get("glog_sigma", 1.1)),
                    anisotropy_param=float(merged_det.get("glog_anisotropy", 2.3)),
                    dist_mode=str(merged_det.get("glog_dist_mode", "relative")),
                    alpha_param=float(merged_det.get("glog_alpha", 0.86)),
                    sharpness_param=float(merged_det.get("glog_sharpness", 8.0)),
                    gamma_scale_param=float(merged_det.get("glog_gamma_scale", 1.0))
                    )
        if torch.is_tensor(res_t):
            res_np = res_t.detach().cpu().numpy()
        else:
            res_np = np.array(res_t, dtype=np.float32)

        res_np = np.array(res_np, dtype=np.float32)

        # (N,1,H,W) -> (N,H,W)
        if res_np.ndim == 4 and res_np.shape[1] == 1:
            res_np = res_np[:, 0, ...]  # (N,H,W)

        # (1,H,W) -> (H,W)
        if res_np.ndim == 3 and res_np.shape[0] == 1:
            res_np = res_np[0]

        if res_np.ndim == 3 and res_np.shape[-1] == 1:
            res_np = res_np[..., 0]   
        img_back = res_np.astype(np.float32)
        img_back = np.nan_to_num(img_back, nan=0.0, posinf=1.0, neginf=0.0)
        lo = np.nanpercentile(img_back, 5)
        hi = np.nanpercentile(img_back, 99)  
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < 1e-6):
            return np.zeros_like(agc, dtype=np.float32)
        img_back = (img_back - lo) / (hi - lo + 1e-12)
        img_back = np.clip(img_back, 0.0, 1.0)
        img_back = np.nan_to_num(img_back, nan=0.0, posinf=1.0, neginf=0.0)

        thr = float(merged_det.get("edge_thr", 0.35))
        k   = float(merged_det.get("edge_soft_k", 12.0))   
        img_back = 1.0 / (1.0 + np.exp(-k * (img_back - thr)))
        img_back = np.clip(img_back, 0.0, 1.0)
        return img_back.astype(np.float32)
    except Exception as e:
        logging.exception(f"generalized_log detector failed: {e}")
        return np.zeros_like(agc, dtype=np.float32)
    
def preprocess_image_once(img: np.ndarray,
                          sgs_params: Dict[str,Any],
                          agc_params: Dict[str,Any]) -> Dict[str, np.ndarray]:
    gray = to_gray(img).astype(np.float32)
    band = gray.copy().astype(np.float32)

    nl_h = float(sgs_params.get("nl_h", 0.8))
    nl = denoise_nlmeans(band, patch_size=5, patch_distance=6, h=nl_h).astype(np.float32)

    sgs_res, coherence_map, orientation_map, Dxx_map, Dxy_map, Dyy_map = structure_guided_smoothing(
        nl,
        sigma=float(sgs_params.get("sigma", 1.5)),
        anisotropy=float(sgs_params.get("anisotropy", 2.2)),
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
                            window=int(agc_params.get("window", 31)),
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

def compute_edge_map_from_preprocessed(
     pp: Dict[str, np.ndarray],
     det_params: Dict[str, Any],
     edge_type: str = "glog",
 ) -> np.ndarray:

     gray = pp["gray"]
     sgs  = pp["sgs"]
     agc  = pp["agc"]
     coh  = pp["coherence"]
     ori  = pp["orientation"]
     Dxx  = pp["Dxx"]
     Dxy  = pp["Dxy"]
     Dyy  = pp["Dyy"]

     et = edge_type.strip().lower()

     if et in ("glog", "slog"):
         merged_det_b = dict(det_params)
         merged_det_b.update({
             "glog_sigma":       float(det_params.get("glog_sigma", 0.8)),
             "glog_anisotropy":  float(det_params.get("glog_anisotropy", 2.0)),
             "glog_nangles":     int(det_params.get("glog_nangles", 16)),
             "glog_sharpness":   float(det_params.get("glog_sharpness", 7.0)),
             "glog_alpha":       float(det_params.get("glog_alpha", 0.8)),
         })
         edge_map = apply_detector_generalized_log(
             agc, ori, coh, Dxx, Dxy, Dyy, merged_det_b
         )
     elif et == "log":
         log_resp = ndimage.gaussian_laplace(agc, sigma=float(det_params.get("log_sigma", 1.1)))
         edge_map = normalize_map(np.abs(log_resp).astype(np.float32))
     elif et == "canny":
         sigma = float(det_params.get("canny_sigma", 1.7))
         low_t = float(det_params.get("canny_low", 0.1))
         high_t= float(det_params.get("canny_high", 0.9))
         if _have_skimage:
             edge_bool = sk_canny(
                 agc.astype(np.float32),
                 sigma=sigma,
                 low_threshold=low_t,
                 high_threshold=high_t,
             )
             edge_map = edge_bool.astype(np.float32)
         elif _have_cv2:
             im8 = (np.clip(agc, 0.0, 1.0) * 255.0).astype(np.uint8)
             t1 = int(np.clip(low_t, 0.0, 1.0) * 255)
             t2 = int(np.clip(high_t,0.0, 1.0) * 255)
             edge_u8 = cv2.Canny(im8, t1, t2)
             edge_map = (edge_u8 > 0).astype(np.float32)
         else:
             gx = np.gradient(agc, axis=1)
             gy = np.gradient(agc, axis=0)
             edge_map = normalize_map(np.sqrt(gx * gx + gy * gy).astype(np.float32))
     else:
         raise ValueError(f"Unknown edge_type: {edge_type}")

     return np.clip(edge_map, 0.0, 1.0).astype(np.float32)

def compute_fault_map_from_preprocessed(
    pp: Dict[str, np.ndarray],
    edge_map: np.ndarray,
    det_params: Dict[str, Any],
) -> np.ndarray:

    coh = np.clip(pp["coherence"].astype(np.float32), 0.0, 1.0)  # [0,1]
    ori = pp["orientation"].astype(np.float32)  # radians, roughly [-pi/2, pi/2]
    ori = np.clip(ori, -np.pi/2, np.pi/2)
    edge = np.clip(edge_map.astype(np.float32), 0.0, 1.0)

    sx = np.sin(2.0 * ori)
    cx = np.cos(2.0 * ori)
    gx_s = ndimage.sobel(sx, axis=1).astype(np.float32)
    gy_s = ndimage.sobel(sx, axis=0).astype(np.float32)
    gx_c = ndimage.sobel(cx, axis=1).astype(np.float32)
    gy_c = ndimage.sobel(cx, axis=0).astype(np.float32)

    ori_jump = np.sqrt(gx_s*gx_s + gy_s*gy_s + gx_c*gx_c + gy_c*gy_c).astype(np.float32)
    ori_jump = normalize_map(ori_jump)

    coh_drop = (1.0 - coh).astype(np.float32)  
    coh_drop = np.clip(coh_drop, 0.0, 1.0)

    a = float(det_params.get("fault_w_coh", 0.55))
    b = float(det_params.get("fault_w_ori", 0.30))
    c = float(det_params.get("fault_w_edge", 0.15))
    score = a * coh_drop + b * ori_jump + c * edge
    score = np.clip(score, 0.0, 1.0).astype(np.float32)

    thr = float(det_params.get("fault_thr", 0.45))
    k   = float(det_params.get("fault_k", 12.0))
    fault_prob = 1.0 / (1.0 + np.exp(-k * (score - thr)))
    fault_prob = np.clip(fault_prob.astype(np.float32), 0.0, 1.0)

    seed_thr = float(det_params.get("fault_seed_thr", 0.65))
    sigma_px = float(det_params.get("fault_band_sigma", 3.5))  
    seed = (fault_prob >= seed_thr).astype(np.uint8)

    if seed.sum() > 0 and sigma_px > 0:
        # distance to nearest seed pixel
        dist = ndimage.distance_transform_edt(1 - seed).astype(np.float32)
        band = np.exp(-(dist * dist) / (2.0 * sigma_px * sigma_px)).astype(np.float32)
        fault = np.clip(0.6 * fault_prob + 0.4 * band, 0.0, 1.0).astype(np.float32)
    else:
        fault = fault_prob.astype(np.float32)

    smooth = float(det_params.get("fault_smooth_sigma", 0.8))
    if smooth > 0:
        fault = gaussian_filter(fault, sigma=smooth).astype(np.float32)
        fault = np.clip(fault, 0.0, 1.0)

    return fault.astype(np.float32)

def degrade_leaky_channel(x01: np.ndarray, down: int = 4, noise_std: float = 0.03,blur_sigma: float = 0.0,quant_bits: int = 0,) -> np.ndarray:
    """
    x01: (H,W) in [0,1]
    down: >1 => downsample then upsample (reduces pixel-level leakage)
    noise_std: Gaussian noise std in [0,1] scale
    """
    x = x01.astype(np.float32)
    H, W = x.shape
    if down is not None and int(down) > 1:
        d = int(down)
        h2 = max(2, H // d)
        w2 = max(2, W // d)
        # area downsample -> bilinear upsample
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        x_t = F.interpolate(x_t, size=(h2, w2), mode='area')
        x_t = F.interpolate(x_t, size=(H, W), mode='bilinear', align_corners=False)
        x = x_t[0, 0].numpy().astype(np.float32)

    try:
        bs = float(blur_sigma)
    except Exception:
        bs = 0.0
    if bs > 0:
        x = gaussian_filter(x, sigma=bs).astype(np.float32)

    try:
        qb = int(quant_bits)
    except Exception:
        qb = 0
    if qb is not None and qb > 0:
        levels = float((1 << qb) - 1)
        if levels > 1:
            x = np.round(np.clip(x, 0.0, 1.0) * levels) / levels

    if noise_std is not None:
        try:
            ns = float(noise_std)
        except Exception:
            ns = 0.0
        if ns > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * ns

    return np.clip(x, 0.0, 1.0).astype(np.float32)

def compute_texture_maps_single(
    img_np: np.ndarray,
    sgs_params: Dict[str, Any],
    agc_params: Dict[str, Any],
    det_params: Dict[str, Any],
    edge_type: str = "glog",
) -> np.ndarray:

    pp = preprocess_image_once(img_np, sgs_params, agc_params)
    gray = normalize_map(pp["gray"])
    sgs  = normalize_map(pp["sgs"])
    agc  = normalize_map(pp["agc"])
    edge_map = compute_edge_map_from_preprocessed(pp, det_params, edge_type=edge_type)
    fault_map = compute_fault_map_from_preprocessed(pp, edge_map=edge_map, det_params=det_params)


    hp = normalize_map(np.abs(gray - sgs).astype(np.float32))

    coherence = np.clip(pp["coherence"].astype(np.float32), 0.0, 1.0)
    #coherence = normalize_map(pp["coherence"])

    ori = pp["orientation"].astype(np.float32)
    ori = np.clip(ori, -np.pi/2, np.pi/2)
    ori_sin = (np.sin(2.0 * ori) + 1.0) * 0.5  # [0,1]
    ori_cos = (np.cos(2.0 * ori) + 1.0) * 0.5  # [0,1]
    # --- coherence gating: suppress unreliable orientation ---
    tau = float(det_params.get("ori_coh_tau", 0.35))     
    gate = np.clip((coherence - tau) / max(1e-6, (1.0 - tau)), 0.0, 1.0).astype(np.float32)

    ori_sin = (ori_sin * gate + 0.5 * (1.0 - gate)).astype(np.float32)
    ori_cos = (ori_cos * gate + 0.5 * (1.0 - gate)).astype(np.float32)

    base_down = int(det_params.get("leak_down", 4))
    base_nz   = float(det_params.get("leak_noise", 0.03))

    raw_sgs_down = det_params.get("sgs_leak_down", 0)
    sgs_down = int(raw_sgs_down) if (raw_sgs_down is not None and int(raw_sgs_down) > 0) else max(base_down, base_down * 2)

    raw_sgs_nz = det_params.get("sgs_leak_noise", -1.0)
    sgs_nz = float(raw_sgs_nz) if (raw_sgs_nz is not None and float(raw_sgs_nz) >= 0.0) else (base_nz * 2.0)

    raw_sgs_blur = det_params.get("sgs_leak_blur", 0.0)
    sgs_blur = float(raw_sgs_blur) if (raw_sgs_blur is not None and float(raw_sgs_blur) > 0.0) else 0.0

    raw_sgs_qbits = det_params.get("sgs_leak_qbits", 0)
    sgs_qbits = int(raw_sgs_qbits) if (raw_sgs_qbits is not None and int(raw_sgs_qbits) > 0) else 0

    sgs_d = degrade_leaky_channel(
        sgs, down=sgs_down, noise_std=sgs_nz, blur_sigma=sgs_blur, quant_bits=sgs_qbits
    )

    agc_d = degrade_leaky_channel(agc, down=base_down, noise_std=base_nz)
    hp_d  = degrade_leaky_channel(hp,  down=base_down, noise_std=base_nz)

    f_down = int(det_params.get("fault_leak_down", base_down))
    f_nz   = float(det_params.get("fault_leak_noise", base_nz))
    fault_d = degrade_leaky_channel(fault_map, down=f_down, noise_std=f_nz)
    
    layout = str(det_params.get("tex_layout", "sgs,agc,edge")).strip().lower()
    keys = [k.strip() for k in layout.split(",") if k.strip()]
    chan_map = {
        "gray": gray,
        "sgs":  sgs_d,
        "agc":  agc_d,
        "edge": edge_map,
        "hp":   hp_d,
        "coherence": coherence,
        "ori_sin": ori_sin,
        "ori_cos": ori_cos,
        "fault": fault_d,
    }
    tex_list = []
    for k in keys:
        if k not in chan_map:
            raise ValueError(f"Unknown tex_layout key '{k}'. Valid: gray,sgs,agc,edge,hp,coherence,ori_sin,ori_cos,fault")
        tex_list.append(chan_map[k])
    tex = np.stack(tex_list, axis=0).astype(np.float32)  # (C_tex,H,W)
    return tex

def fit_valid_standardizer(
    features: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)

    mean = features.mean(axis=0, keepdims=True)
    std_raw = features.std(axis=0, keepdims=True)

    valid = (
        np.isfinite(std_raw[0])
        & (std_raw[0] > float(eps))
    )

    if not np.any(valid):
        raise ValueError(
            "No valid non-constant feature dimensions remain."
        )

    std_safe = np.where(
        valid.reshape(1, -1),
        std_raw,
        1.0,
    )

    return mean, std_safe, valid

def compute_texture_maps_batch(
    imgs: torch.Tensor,
    sgs_params: Dict[str, Any],
    agc_params: Dict[str, Any],
    det_params: Dict[str, Any],
    device=None,
    edge_type: str = "glog",
) -> torch.Tensor:

    if device is None:
        device = imgs.device

    imgs_cpu = imgs.detach().cpu()
    N, C, H, W = imgs_cpu.shape
    if C > 1:
        imgs_gray = imgs_cpu.mean(dim=1, keepdim=True)
    else:
        imgs_gray = imgs_cpu

    tex_list = []
    for i in range(N):
        img_np = imgs_gray[i,0].numpy()
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        tex_np = compute_texture_maps_single(img_np, sgs_params, agc_params, det_params, edge_type=edge_type)
        tex_list.append(tex_np)

    tex_np_batch = np.stack(tex_list, axis=0)  # (N,C_tex,H,W)
    tex_t = torch.from_numpy(tex_np_batch)     # 0..1
    tex_t = tex_t * 2.0 - 1.0                  # -> [-1,1]
    return tex_t.to(device)

class ImageFolderDataset(Dataset):
    def __init__(self, root, crop_size=128, start_max_right=900):
        self.paths = sorted([p for p in Path(root).glob("*") if p.suffix.lower() in (".jpg",".bmp",".png")])
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.crop = int(crop_size)
        self.n = len(self.paths)
        self.start_max_right = int(start_max_right)
        self.start_max_left = max(0, self.start_max_right - self.crop)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("L")
        W, H = img.size

        # If a field image is already exactly the requested size (the F3
        # protocol uses 128x128 prepared images), keep it unchanged.  For the
        # Australian source profiles, retain the original dynamic crop logic.
        if W == self.crop and H == self.crop:
            crop = img
        else:
            if W < self.crop or H < self.crop:
                scale = max(self.crop / W, self.crop / H)
                new_w = int(round(W * scale))
                new_h = int(round(H * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                W, H = img.size

            ratio = idx / max(1, (self.n - 1))
            full_max_left = max(0, W - self.crop)
            curr_max_left = int(round(self.start_max_left * (1 - ratio) + full_max_left * ratio))
            left = random.randint(0, curr_max_left)

            y_min = max(0, H - 600)
            y_max = max(0, H - self.crop)
            if y_min > y_max:
                y_min = 0
                y_max = max(0, H - self.crop)
            top = random.randint(y_min, y_max)

            right = left + self.crop
            bottom = top + self.crop
            crop = img.crop((left, top, right, bottom))
        t = T.ToTensor()(crop)  # (1,H,W)
        t = T.Normalize([0.5], [0.5])(t)  # 1ch
        return t

class PairedGenerationDataset(Dataset):
    """
    专门用于按文件名一一对应生成下游实验数据。

    要求：
    1. 图像已经裁剪为 img_size × img_size；
    2. 文件名为纯数字，例如 0.png、37.png；
    3. 图像与标签使用相同数字文件名。
    """

    IMAGE_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".bmp",
        ".tif", ".tiff", ".webp",
    }

    def __init__(
        self,
        image_dir,
        label_dir,
        img_size=512,
        expected_num=350,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = int(img_size)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_dir}"
            )

        if not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory does not exist: {self.label_dir}"
            )

        image_paths = [
            p for p in self.image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        label_paths = [
            p for p in self.label_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        # 要求纯数字命名
        invalid_images = [
            p.name for p in image_paths
            if not p.stem.isdigit()
        ]
        invalid_labels = [
            p.name for p in label_paths
            if not p.stem.isdigit()
        ]

        if invalid_images:
            raise ValueError(
                f"Image filenames must be numeric: {invalid_images[:10]}"
            )

        if invalid_labels:
            raise ValueError(
                f"Label filenames must be numeric: {invalid_labels[:10]}"
            )

        # 按数字自然排序
        image_map = {
            int(p.stem): p
            for p in image_paths
        }
        label_map = {
            int(p.stem): p
            for p in label_paths
        }

        image_ids = set(image_map)
        label_ids = set(label_map)

        missing_labels = sorted(image_ids - label_ids)
        missing_images = sorted(label_ids - image_ids)

        if missing_labels:
            raise RuntimeError(
                f"Images without labels: {missing_labels[:20]}"
            )

        if missing_images:
            raise RuntimeError(
                f"Labels without images: {missing_images[:20]}"
            )

        common_ids = sorted(image_ids & label_ids)

        if expected_num > 0 and len(common_ids) != expected_num:
            raise ValueError(
                f"Expected {expected_num} image-label pairs, "
                f"but found {len(common_ids)}."
            )

        self.records = [
            {
                "id": image_id,
                "image_path": image_map[image_id],
                "label_path": label_map[image_id],
            }
            for image_id in common_ids
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = record["image_path"]
        label_path = record["label_path"]

        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        if image.size != (self.img_size, self.img_size):
            raise ValueError(
                f"{image_path.name}: expected "
                f"{self.img_size}×{self.img_size}, "
                f"but got {image.size}."
            )

        if label.size != (self.img_size, self.img_size):
            raise ValueError(
                f"{label_path.name}: expected "
                f"{self.img_size}×{self.img_size}, "
                f"but got {label.size}."
            )

        image_tensor = T.ToTensor()(image)
        image_tensor = T.Normalize(
            [0.5],
            [0.5],
        )(image_tensor)

        return {
            "image": image_tensor,
            "image_id": int(record["id"]),
            "image_path": str(image_path),
            "label_path": str(label_path),
        }

def make_latent_batch(
    image_ids,
    z_dim,
    base_seed,
    device,
):
    """
    为每个原始图像编号生成固定潜变量。

    同一个 image_id 在SLoG、LoG、Canny实验中会得到相同z。
    """
    latent_list = []
    latent_seeds = []

    for image_id in image_ids:
        image_id = int(image_id)
        latent_seed = int(base_seed) + image_id

        generator = torch.Generator(device="cpu")
        generator.manual_seed(latent_seed)

        z = torch.randn(
            int(z_dim),
            generator=generator,
            dtype=torch.float32,
        )

        latent_list.append(z)
        latent_seeds.append(latent_seed)

    z_batch = torch.stack(latent_list, dim=0).to(device)

    return z_batch, latent_seeds

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1.0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None)

# Mapping network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(z_dim if i==0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        z = z / torch.sqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)  # PixelNorm
        return self.net(z)

# ModulatedConv2d
class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, style_dim,
                 upsample=False, demodulate=True, anti_alias=False):
        super().__init__()
        self.kernel = kernel
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample
        self.demodulate = demodulate
        self.anti_alias = anti_alias

        fan_in = in_ch * kernel * kernel
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel) / math.sqrt(fan_in))
        self.style_fc = EqualizedLinear(style_dim, in_ch, bias=True)
        self.eps = 1e-8

    def forward(self, x, style):
        N, C, H, W = x.shape
        style = self.style_fc(style)  # (N, in_ch)

        style32 = style.float().view(N, 1, C, 1, 1)
        w32 = self.weight.float() * style32  # (N, out, in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(w32.pow(2).sum((2, 3, 4)) + self.eps)  # (N, out)
            w32 = w32 * demod.view(N, self.out_ch, 1, 1, 1)

        w32 = torch.nan_to_num(w32, nan=0.0, posinf=0.0, neginf=0.0)

        weight = w32.to(dtype=x.dtype)
        weight = weight.view(N * self.out_ch, C, self.kernel, self.kernel)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            if self.anti_alias:
                k = torch.tensor([1., 2., 1.], device=x.device, dtype=x.dtype)
                k = k[:, None] * k[None, :]
                k = k / k.sum()
                k = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
                x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                x = F.conv2d(x, k, padding=0, groups=C)

        x = x.view(1, N * C, x.shape[-2], x.shape[-1])
        out = F.conv2d(x, weight, padding=self.kernel // 2, groups=N)
        out = out.view(N, self.out_ch, out.shape[-2], out.shape[-1])
        return out

class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, upsample=False, cond_mode='film'):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, style_dim, upsample=upsample, anti_alias=True)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, 3, style_dim, upsample=False)
        self.noise1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.noise2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)
        self.cond_mode = cond_mode
        self.cond_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, w, cond: torch.Tensor = None):
        cond_scale_min = float(getattr(self, "cond_scale_min", 0.0))
        cond_scale_max = float(getattr(self, "cond_scale_max", 1.0))
        noise_strength_max = float(getattr(self, "noise_strength_max", 2.0))
        film_gamma_max = float(getattr(self, "film_gamma_max", 3.0))
        film_beta_max  = float(getattr(self, "film_beta_max", 3.0))
        cond_scale = self.cond_scale.clamp(cond_scale_min, cond_scale_max)
        noise1 = self.noise1.clamp(-noise_strength_max, noise_strength_max)
        noise2 = self.noise2.clamp(-noise_strength_max, noise_strength_max)
        
        out = self.conv1(x, w)
        noise_map = torch.randn(out.size(0), 1, out.size(2), out.size(3), device=out.device,dtype=out.dtype)
        out = out + 0.1 * noise1.to(out.dtype) * noise_map

        if cond is not None:
            C = out.shape[1]
            if self.cond_mode == 'film':
                gamma = cond[:, :C, :, :]
                beta  = cond[:, C:2*C, :, :]
                gamma = torch.tanh(gamma) * film_gamma_max
                beta  = torch.tanh(beta)  * film_beta_max

                gamma = gamma.to(out.dtype)
                beta  = beta.to(out.dtype)
                out = out * (1.0 + gamma * 0.1 * cond_scale.to(out.dtype)) + (beta * 0.1 * cond_scale.to(out.dtype))
            else:
                out = out + cond.to(out.dtype) * (1e-2 * cond_scale.to(out.dtype))
        out = self.act(out)

        out = self.conv2(out, w)
        noise_map2 = torch.randn(out.size(0), 1, out.size(2), out.size(3), device=out.device,dtype=out.dtype)
        out = out + 0.1 * noise2.to(out.dtype) * noise_map2
        out = self.act(out)
        return out

class EdgeEncoder(nn.Module):
    def __init__(self, in_ch=3, enc_ch=64, num_levels=6, img_size=128):
        super().__init__()
        self.in_ch = in_ch
        self.enc_ch = enc_ch
        self.num_levels = num_levels
        self.img_size = img_size

        blocks = []
        ch_in = in_ch
        for i in range(num_levels):
            ch_out = enc_ch
            block = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            blocks.append(block)
            ch_in = ch_out

        self.blocks = nn.ModuleList(blocks)
        self.down = nn.AvgPool2d(2)

    def forward(self, cond):
        N, C, H, W = cond.shape
        x = cond
        pyramid = {}
        cur_H = H
        for i, block in enumerate(self.blocks):
            h = block(x)           # (N,enc_ch,cur_H,cur_H)
            pyramid[cur_H] = h
            if i < len(self.blocks) - 1:
                x = self.down(h)
                cur_H = x.shape[2]
        return pyramid

class GeneratorWithTexture(nn.Module):
    def __init__(self,
                 z_dim=512,
                 w_dim=512,
                 base_channels=512,
                 img_size=128,
                 out_channels=1,
                 cond_channels=3,
                 enc_ch=64,
                 cond_mode='film'):
        super().__init__()
        self.img_size = img_size
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        self.num_stages = int(math.log2(img_size) - 2) + 1
        in_ch = base_channels
        self.blocks = nn.ModuleList()
        self.block_out_ch = []
        self.block_res = []
        for i in range(self.num_stages):
            out_ch = max(base_channels // (2**i), 16)
            up = (i != 0)
            b = SynthesisBlock(in_ch, out_ch, w_dim, upsample=up, cond_mode=cond_mode)
            self.blocks.append(b)
            self.block_out_ch.append(out_ch)
            res = 4 * (2 ** i)
            self.block_res.append(res)
            in_ch = out_ch

        self.out_channels = out_channels
        self.to_rgb = nn.Conv2d(in_ch, self.out_channels, 1)
        self.cond_mode = cond_mode

        # EdgeEncoder
        self.edge_encoder = EdgeEncoder(
            in_ch=cond_channels,
            enc_ch=enc_ch,
            num_levels=self.num_stages,
            img_size=img_size,
        )

        self.cond_adapters = nn.ModuleList()
        for out_ch in self.block_out_ch:
            if self.cond_mode == 'film':
                adapter = nn.Conv2d(enc_ch, out_ch * 2, kernel_size=1, stride=1, padding=0)
            else:
                adapter = nn.Conv2d(enc_ch, out_ch, kernel_size=1, stride=1, padding=0)
            nn.init.normal_(adapter.weight, 0.0, 0.01)
            if adapter.bias is not None:
                nn.init.zeros_(adapter.bias)
            self.cond_adapters.append(adapter)

        for b in self.blocks:
            b.cond_scale.data.fill_(0.03)

    def forward(self, z, cond_tex=None, mixing_prob=0.0, z2=None):
        device = z.device
        N = z.shape[0]
        w1 = self.mapping(z)

        # style mixing
        if float(mixing_prob) <= 0.0 or self.num_stages <= 1:
            styles = [w1] * self.num_stages
        else:
            mix_mask = (torch.rand(N, device=device) < float(mixing_prob))
            if mix_mask.any():
                if z2 is None:
                    z2_full = torch.randn(N, z.shape[1], device=device)
                else:
                    if isinstance(z2, torch.Tensor) and z2.shape[0] == N:
                        z2_full = z2.to(device)
                    else:
                        z2_full = torch.randn(N, z.shape[1], device=device)
                w2 = self.mapping(z2_full)
                cutoffs = torch.randint(1, self.num_stages, (N,), device=device)
                styles = []
                for layer_idx in range(self.num_stages):
                    mask_w1 = (layer_idx < cutoffs) | (~mix_mask)
                    mask_w1 = mask_w1.view(N, 1)
                    style_i = torch.where(mask_w1, w1, w2)
                    styles.append(style_i)
            else:
                styles = [w1] * self.num_stages

        cond_pyr = None
        if cond_tex is not None:
            cond_pyr = self.edge_encoder(cond_tex)  # dict: {res: (N,enc_ch,res,res)}

        x = self.const.repeat(N, 1, 1, 1)
        for idx, (block, style, adapter, res) in enumerate(
            zip(self.blocks, styles, self.cond_adapters, self.block_res)
        ):
            cond = None
            if cond_pyr is not None:
                if res in cond_pyr:
                    feat = cond_pyr[res]
                else:
                    all_res = sorted(cond_pyr.keys())
                    nearest = min(all_res, key=lambda r: abs(r-res))
                    feat = cond_pyr[nearest]
                    if feat.shape[2] != res:
                        feat = F.interpolate(feat, size=(res, res), mode='bilinear', align_corners=False)
                cond = adapter(feat)  # (N,2*out_ch,res,res) for FiLM
            x = block(x, style, cond=cond)

        img = self.to_rgb(x)
        img = torch.tanh(img)
        return img

# Minibatch StdDev
class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        g = min(self.group_size, N)
        if N % g != 0:
            g = 1
        if g == 1:
            std = x.std(dim=0, unbiased=False, keepdim=True)
            std_mean = std.mean(dim=1, keepdim=True)
            std_map = std_mean.repeat(N, 1, 1, 1)
            return torch.cat([x, std_map], dim=1)

        M = N // g
        y = x.view(g, M, C, H, W)
        mean = y.mean(dim=1, keepdim=True)
        var = ((y - mean) ** 2).mean(dim=1, keepdim=False)
        std = torch.sqrt(var + self.eps)
        std_mean = std.mean(dim=1, keepdim=True)
        std_mean = std_mean.repeat(1, M, 1, 1, 1).view(N, 1, H, W)
        return torch.cat([x, std_mean], dim=1)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=4, use_mbstd=False):
        super().__init__()
        self.use_mbstd = use_mbstd
        layers = []
        nf = base_channels

        layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, n_layers):
            prev_nf = nf
            nf = min(base_channels * (2 ** i), 512)
            stride = 2 if i < n_layers - 1 else 1
            layers.append(nn.utils.spectral_norm(nn.Conv2d(prev_nf, nf, kernel_size=4, stride=stride, padding=1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

        if self.use_mbstd:
            self.mbstd = MinibatchStdDev(group_size=4)
            self.final_conv = nn.utils.spectral_norm(nn.Conv2d(nf + 1, nf, kernel_size=3, stride=1, padding=1))
            self.final_act = nn.LeakyReLU(0.2, inplace=True)
            out_in_ch = nf
        else:
            self.mbstd = None
            self.final_conv = None
            self.final_act = None
            out_in_ch = nf

        self.out_conv = nn.utils.spectral_norm(nn.Conv2d(out_in_ch, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        h = self.model(x)
        if self.use_mbstd:
            h = self.mbstd(h)
            h = self.final_act(self.final_conv(h))
        out = self.out_conv(h)
        return out

class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=6, use_mbstd=True):
        super().__init__()
        self.use_mbstd = use_mbstd
        
        layers = []
        nf = base_channels

        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, n_layers):
            prev_nf = nf
            nf = min(base_channels * (2 ** i), 512)
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(prev_nf, nf, kernel_size=4, stride=2, padding=1)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

        if self.use_mbstd:
            self.mbstd = MinibatchStdDev(group_size=4)
            self.final_conv = nn.utils.spectral_norm(
                nn.Conv2d(nf + 1, nf, kernel_size=3, stride=1, padding=1)
            )
            self.final_act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.mbstd = None
            self.final_conv = None
            self.final_act = None

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_conv = nn.utils.spectral_norm(
            nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        h = self.model(x)
        if self.use_mbstd:
            h = self.mbstd(h)
            h = self.final_act(self.final_conv(h))
        h = self.pool(h)
        return self.out_conv(h)  # (N,1,1,1)

class CombinedDiscriminator(nn.Module):
    """Combine multi-scale patch logits + one global logit."""
    def __init__(self, D_patch: nn.Module, D_global: nn.Module):
        super().__init__()
        self.D_patch = D_patch
        self.D_global = D_global

    def forward(self, x):
        outs = []
        outs.extend(self.D_patch(x))     # list of patch logits at multiple scales
        outs.append(self.D_global(x))   # one global logit
        return outs
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, base_D: nn.Module, scales=(1.0, 0.5)):
        super().__init__()
        self.D = base_D
        self.scales = scales

    def forward(self, x):
        outs = []
        for s in self.scales:
            if s == 1.0:
                xs = x
            else:
                H, W = x.shape[2], x.shape[3]
                xs = F.interpolate(x, size=(int(H*s), int(W*s)), mode='bilinear', align_corners=False)
            outs.append(self.D(xs))
        return outs  # list of logits

# 损失
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()
    return real_loss + fake_loss

def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def multiscale_fft_l1_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    scales=(1, 2, 4),
    use_log: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert fake.dim() == 4 and real.dim() == 4, "expect NCHW"
    def _spec_mag(x: torch.Tensor) -> torch.Tensor:
        # rfft2 over last 2 dims -> complex
        X = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        mag = torch.abs(X)  # |FFT|
        if use_log:
            mag = torch.log1p(mag)
        # per-sample normalize to reduce energy-scale ambiguity
        mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + eps)
        return mag

    loss = fake.new_tensor(0.0)
    n = 0
    for s in scales:
        if s > 1:
            # avg pool downsample
            f = F.avg_pool2d(fake, kernel_size=s, stride=s)
            r = F.avg_pool2d(real, kernel_size=s, stride=s)
        else:
            f, r = fake, real

        loss = loss + (_spec_mag(f) - _spec_mag(r)).abs().mean()
        n += 1

    return loss / max(1, n)

def _ramp(step: int, final_value: float, warmup: int, ramp: int) -> float:
    if final_value <= 0:
        return 0.0
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return float(final_value)
    t = (step - warmup) / float(ramp)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return float(final_value) * t

def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """x: (N,C,H,W) -> grad magnitude (N,C,H,W)"""
    # Sobel kernels
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    # depthwise conv
    C = x.shape[1]
    kx = kx.repeat(C,1,1,1)
    ky = ky.repeat(C,1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def multiscale_sobel_grad_l1_masked(fake: torch.Tensor,
                                   real: torch.Tensor,
                                   scales=(1,2,4),
                                   hard_q: float = 0.85,
                                   hard_min: float = 0.0,
                                   eps: float = 1e-8) -> torch.Tensor:
    if fake.shape[1] > 1:
        fake_g = fake.mean(dim=1, keepdim=True)
        real_g = real.mean(dim=1, keepdim=True)
    else:
        fake_g, real_g = fake, real

    total = fake.new_tensor(0.0)
    n = 0
    for s in scales:
        if s > 1:
            f = F.avg_pool2d(fake_g, kernel_size=s, stride=s)
            r = F.avg_pool2d(real_g, kernel_size=s, stride=s)
        else:
            f, r = fake_g, real_g

        mask = hard_grad_mask_from_real(r, q=hard_q, min_thr=hard_min, eps=eps)  # (N,1,h,w)

        gx_f, gy_f = _sobel_xy(f)
        gx_r, gy_r = _sobel_xy(r)
        mag_f = torch.sqrt(gx_f*gx_f + gy_f*gy_f + eps)
        mag_r = torch.sqrt(gx_r*gx_r + gy_r*gy_r + eps)

        diff = (mag_f - mag_r).abs() * mask
        loss_s = diff.sum() / (mask.sum() + eps)
        total = total + loss_s
        n += 1

    return total / max(1, n)

# ======================
# New: differentiable orientation losses
# ======================
def _sobel_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: (N,C,H,W) -> (gx, gy), depthwise Sobel."""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    C = x.shape[1]
    kx = kx.repeat(C,1,1,1)
    ky = ky.repeat(C,1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy

def _quantile_per_sample(x: torch.Tensor, q: float) -> torch.Tensor:
    assert x.dim() == 4
    N = x.shape[0]
    flat = x.reshape(N, -1)
    M = flat.shape[1]
    q = float(q)
    q = 0.0 if q < 0 else (1.0 if q > 1 else q)
    k = int(round(q * (M - 1))) + 1          # kth smallest
    k = max(1, min(k, M))
    thr = flat.kthvalue(k, dim=1).values     # (N,)
    return thr.view(N, 1, 1, 1)

def hard_grad_mask_from_real(real_gray: torch.Tensor,
                            q: float = 0.85,
                            min_thr: float = 0.0,
                            eps: float = 1e-8) -> torch.Tensor:
    gx, gy = _sobel_xy(real_gray)
    mag = torch.sqrt(gx * gx + gy * gy + eps)          # (N,1,H,W)
    thr = _quantile_per_sample(mag.detach(), q=q)      # (N,1,1,1)
    if min_thr and float(min_thr) > 0:
        thr = torch.maximum(thr, mag.new_tensor(float(min_thr)))
    mask = (mag.detach() >= thr).to(dtype=real_gray.dtype)
    return mask

def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    # 6*sigma rule; at least 3 and odd
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    k = int(max(3, round(6.0 * float(sigma)) + 1))
    if k % 2 == 0:
        k += 1
    half = k // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / (float(sigma) + 1e-12))**2)
    g = g / (g.sum() + 1e-12)
    return g

def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise separable Gaussian blur, differentiable."""
    if sigma <= 0:
        return x
    g = _gaussian_kernel1d(float(sigma), device=x.device, dtype=x.dtype)
    C = x.shape[1]
    gh = g.view(1,1,1,-1).repeat(C,1,1,1)
    gv = g.view(1,1,-1,1).repeat(C,1,1,1)
    pad = g.numel() // 2
    x = F.conv2d(x, gh, padding=(0, pad), groups=C)
    x = F.conv2d(x, gv, padding=(pad, 0), groups=C)
    return x

def grad_dir_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    scales: Tuple[int, ...] = (1, 2, 4),
    eps: float = 1e-8,
    mag_power: float = 1.0,
    mag_thresh: float = 0.0,
    use_hard_mask: bool = True,
    hard_q: float = 0.85,
    hard_min: float = 0.0,
) -> torch.Tensor:
    if fake.shape[1] > 1:
        fake_g = fake.mean(dim=1, keepdim=True)
        real_g = real.mean(dim=1, keepdim=True)
    else:
        fake_g, real_g = fake, real

    total = fake.new_tensor(0.0)
    n = 0
    for s in scales:
        if s > 1:
            f = F.avg_pool2d(fake_g, kernel_size=s, stride=s)
            r = F.avg_pool2d(real_g, kernel_size=s, stride=s)
        else:
            f, r = fake_g, real_g

        gx_f, gy_f = _sobel_xy(f)
        gx_r, gy_r = _sobel_xy(r)

        mag_f = torch.sqrt(gx_f * gx_f + gy_f * gy_f + eps)
        mag_r = torch.sqrt(gx_r * gx_r + gy_r * gy_r + eps)

        ux_f = gx_f / mag_f
        uy_f = gy_f / mag_f
        ux_r = gx_r / mag_r
        uy_r = gy_r / mag_r

        dot = ux_f * ux_r + uy_f * uy_r
        ang_err = 1.0 - dot.abs()

        w = (mag_r.detach() ** float(mag_power))

        if mag_thresh > 0:
            w = w * (mag_r.detach() > float(mag_thresh)).to(dtype=w.dtype)

        if use_hard_mask:
            mask = hard_grad_mask_from_real(r, q=hard_q, min_thr=hard_min, eps=eps)
            w = w * mask

        loss_s = (ang_err * w).sum() / (w.sum() + eps)
        total = total + loss_s
        n += 1

    return total / max(1, n)

def structure_tensor_orientation_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    scales: Tuple[int, ...] = (1, 2, 4),
    sigmas: Tuple[float, ...] = (1.0,),
    eps: float = 1e-8,
    coh_power: float = 1.0,
    use_hard_mask: bool = True,
    hard_q: float = 0.85,
    hard_min: float = 0.0,
) -> torch.Tensor:
    if fake.shape[1] > 1:
        fake_g = fake.mean(dim=1, keepdim=True)
        real_g = real.mean(dim=1, keepdim=True)
    else:
        fake_g, real_g = fake, real

    def _st_vec(x: torch.Tensor, sigma: float):
        gx, gy = _sobel_xy(x)
        jxx = gx * gx
        jxy = gx * gy
        jyy = gy * gy
        if sigma > 0:
            jxx = gaussian_blur2d(jxx, sigma)
            jxy = gaussian_blur2d(jxy, sigma)
            jyy = gaussian_blur2d(jyy, sigma)

        a = (jxx - jyy)
        b = (2.0 * jxy)
        denom = torch.sqrt(a * a + b * b + eps)
        v0 = a / denom
        v1 = b / denom

        tmp = torch.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2) + eps)
        trace = (jxx + jyy + eps)
        coh = torch.clamp(tmp / trace, 0.0, 1.0)
        return v0, v1, coh

    total = fake.new_tensor(0.0)
    n = 0
    for s in scales:
        if s > 1:
            f = F.avg_pool2d(fake_g, kernel_size=s, stride=s)
            r = F.avg_pool2d(real_g, kernel_size=s, stride=s)
        else:
            f, r = fake_g, real_g

        for sigma in sigmas:
            sigma_eff = float(sigma) * float(s)
            vf0, vf1, _ = _st_vec(f, sigma_eff)
            vr0, vr1, coh_r = _st_vec(r, sigma_eff)

            dot = vf0 * vr0 + vf1 * vr1
            ang_err = 1.0 - dot.abs()

            w = (coh_r.detach() ** float(coh_power))

            if use_hard_mask:
                mask = hard_grad_mask_from_real(r, q=hard_q, min_thr=hard_min, eps=eps)
                w = w * mask

            loss_sr = (ang_err * w).sum() / (w.sum() + eps)
            total = total + loss_sr
            n += 1

    return total / max(1, n)

def _to_gray01_from_tensor(imgs: torch.Tensor) -> np.ndarray:
    """
    imgs: (N,C,H,W) in [-1,1]
    return: (N,H,W) float32 in [0,1]
    """
    x = imgs.detach().cpu()
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=False)
    else:
        x = x[:, 0, :, :]
    return x.numpy().astype(np.float32)

def _hilbert_envelope_approx(img: np.ndarray) -> np.ndarray:
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    env = np.sqrt(gx * gx + gy * gy + 1e-12)
    env = gaussian_filter(env, sigma=1.0)
    return env.astype(np.float32)

def _spectral_features_time_axis(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    x = img - img.mean()
    spec = np.fft.rfft(x, axis=0)  # (Hf, W)
    amp = np.abs(spec).mean(axis=1) + 1e-12  # (Hf,)
    freqs = np.fft.rfftfreq(H, d=1.0)

    centroid = float((freqs * amp).sum() / amp.sum())

    bw = float(np.sqrt(((freqs - centroid) ** 2 * amp).sum() / amp.sum()))

    split = max(1, len(amp) // 4)
    lo = float(amp[:split].sum())
    hi = float(amp[split:].sum())
    hi_lo = float(hi / (lo + 1e-12))

    y = np.log(amp)
    x_f = freqs
    if len(x_f) >= 4:
        A = np.vstack([x_f, np.ones_like(x_f)]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        slope = float(slope)
    else:
        slope = 0.0

    return np.array([centroid, bw, hi_lo, slope], dtype=np.float32)

def _geo_features_single(gray01: np.ndarray,
                         sgs_params: Dict[str,Any],
                         agc_params: Dict[str,Any],
                         det_params: Dict[str,Any],
                         edge_type: str = "glog") -> np.ndarray:

    pp = preprocess_image_once(gray01, sgs_params, agc_params)
    coh = pp["coherence"]
    ori = pp["orientation"]
    agc = pp["agc"]

    edge = compute_edge_map_from_preprocessed(pp, det_params, edge_type=edge_type)

    # coherence: mean/std/quantiles
    coh_mean = float(np.mean(coh))
    coh_std  = float(np.std(coh))
    coh_q10, coh_q50, coh_q90 = np.percentile(coh, [10, 50, 90]).astype(np.float32)

    # orientation histogram (8 bins in [-pi/2, pi/2])
    ori_clip = np.clip(ori, -np.pi/2, np.pi/2)
    hist, _ = np.histogram(ori_clip, bins=8, range=(-np.pi/2, np.pi/2), density=True)
    hist = hist.astype(np.float32)

    # edge density / edge strength
    edge_thr = float(det_params.get("edge_thr_eval", 0.05))
    edge_density = float((edge > edge_thr).mean())
    edge_mean = float(edge.mean())

    # gradient magnitude stats on agc
    gx = np.gradient(agc, axis=1)
    gy = np.gradient(agc, axis=0)
    gmag = np.sqrt(gx*gx + gy*gy + 1e-12).astype(np.float32)
    g_mean = float(gmag.mean())
    g_std  = float(gmag.std())
    g_q90  = float(np.percentile(gmag, 90))

    # Laplacian energy (proxy for sharpness/texture)
    lap = gaussian_laplace(agc, sigma=1.0).astype(np.float32)
    lap_energy = float(np.mean(np.abs(lap)))

    # local variance (texture roughness)
    mu = gaussian_filter(agc, sigma=1.0)
    mu2 = gaussian_filter(agc*agc, sigma=1.0)
    lvar = (mu2 - mu*mu).astype(np.float32)
    lvar_mean = float(lvar.mean())
    lvar_q90  = float(np.percentile(lvar, 90))

    spec_feat = _spectral_features_time_axis(agc)

    rms = float(np.sqrt(np.mean((agc - agc.mean())**2) + 1e-12))
    # envelope proxy
    env = _hilbert_envelope_approx(agc)
    env_mean = float(env.mean())
    env_q90  = float(np.percentile(env, 90))

    feat = np.concatenate([
        np.array([coh_mean, coh_std, coh_q10, coh_q50, coh_q90], dtype=np.float32),
        hist,  # 8 dims
        np.array([edge_density, edge_mean], dtype=np.float32),
        np.array([g_mean, g_std, g_q90, lap_energy, lvar_mean, lvar_q90], dtype=np.float32),
        spec_feat,  # 4 dims
        np.array([rms, env_mean, env_q90], dtype=np.float32),
    ], axis=0)

    return feat.astype(np.float32)

def geo_features_batch(real_imgs: torch.Tensor,
                       fake_imgs: torch.Tensor,
                       sgs_params: Dict[str,Any],
                       agc_params: Dict[str,Any],
                       det_params: Dict[str,Any],
                       edge_type: str = "glog") -> Tuple[np.ndarray, np.ndarray]:
    """
    real_imgs/fake_imgs: (N,C,H,W) in [-1,1]
    returns: (N,D) real_feats, (N,D) fake_feats
    """
    real_gray = _to_gray01_from_tensor(real_imgs)
    fake_gray = _to_gray01_from_tensor(fake_imgs)

    real_list = []
    fake_list = []
    N = real_gray.shape[0]
    for i in range(N):
        real_list.append(_geo_features_single(real_gray[i], sgs_params, agc_params, det_params, edge_type=edge_type))
        fake_list.append(_geo_features_single(fake_gray[i], sgs_params, agc_params, det_params, edge_type=edge_type))

    return np.stack(real_list, axis=0), np.stack(fake_list, axis=0)

def geo_features_only(
    imgs: torch.Tensor,
    sgs_params,
    agc_params,
    det_params,
    edge_type="glog",
):
    gray = _to_gray01_from_tensor(imgs)

    out = []

    for i in range(gray.shape[0]):
        out.append(
            _geo_features_single(
                gray[i],
                sgs_params,
                agc_params,
                det_params,
                edge_type=edge_type,
            )
        )

    return np.stack(out, axis=0)

def estimate_mmd_gamma_median(Z: np.ndarray, max_points: int = 400, seed: int = 0) -> float:
    Z = Z.astype(np.float64)
    n = Z.shape[0]
    m = min(int(max_points), n)
    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)

    Zs = Z[idx]
    # pairwise squared distances (m x m)
    d2 = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=2)
    d2 = d2[np.isfinite(d2)]
    d2 = d2[d2 > 0]
    if d2.size == 0:
        return 1.0
    med = np.median(d2)
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(1.0 / (2.0 * med))

def estimate_group_gammas(real_feat: np.ndarray, fake_feat: np.ndarray, seed: int = 0) -> Dict[str, float]:
    gammas = {}
    for name, sl in GEO_FEAT_LAYOUT.items():
        Z = np.vstack([real_feat[:, sl], fake_feat[:, sl]])
        gammas[name] = estimate_mmd_gamma_median(Z, max_points=400, seed=seed + hash(name) % 10000)
    # Total
    Zt = np.vstack([real_feat, fake_feat])
    gammas["Total"] = estimate_mmd_gamma_median(Zt, max_points=400, seed=seed + 999)
    return gammas

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = np.vstack([X, Y])
    # median heuristic
    if gamma is None:
        m = min(200, Z.shape[0])
        idx = np.random.choice(Z.shape[0], size=m, replace=False)
        Zs = Z[idx]
        d2 = np.sum((Zs[:, None, :] - Zs[None, :, :])**2, axis=2)
        med = np.median(d2[d2 > 0])
        if not np.isfinite(med) or med <= 0:
            gamma = 1.0
        else:
            gamma = 1.0 / (2.0 * med)

    def _k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
        return np.exp(-gamma * d2)

    Kxx = _k(X, X)
    Kyy = _k(Y, Y)
    Kxy = _k(X, Y)
    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(mmd2)

def wasserstein_mean_per_dim(X: np.ndarray, Y: np.ndarray) -> float:
    D = X.shape[1]
    ds = []
    for d in range(D):
        ds.append(wasserstein_distance(X[:, d], Y[:, d]))
    return float(np.mean(ds))
# =========================
# Feature index layout (0-based)
# =========================
GEO_FEAT_LAYOUT = {
    "Structure": slice(0, 15),   # 0..14
    "Texture":   slice(15, 21),  # 15..20
    "Spectrum":  slice(21, 25),  # 21..24
    "Energy":    slice(25, 28),  # 25..27
}

def split_geo_features(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    X: (N,D) geo feature vectors
    return: dict of (N, D_sub)
    """
    out = {}
    for k, sl in GEO_FEAT_LAYOUT.items():
        out[k] = X[:, sl]
    return out

def geo_subscores(real_feat: np.ndarray, fake_feat: np.ndarray,
                  mmd_gamma: float = None) -> Dict[str, Dict[str, float]]:
    """
    returns:
    {
      "Structure": {"mmd":..., "w":...},
      "Texture":   {"mmd":..., "w":...},
      "Spectrum":  {"mmd":..., "w":...},
      "Energy":    {"mmd":..., "w":...},
      "Total":     {"mmd":..., "w":...},
    }
    """
    R = split_geo_features(real_feat)
    F = split_geo_features(fake_feat)

    scores = {}
    for name in ("Structure", "Texture", "Spectrum", "Energy"):
        gamma_use = mmd_gamma[name] if isinstance(mmd_gamma, dict) else mmd_gamma
        mmd_val = mmd_rbf(R[name], F[name], gamma=gamma_use)
        w_val   = wasserstein_mean_per_dim(R[name], F[name])
        scores[name] = {"mmd": float(mmd_val), "w": float(w_val)}

    # Total (full vector)
    gamma_total = mmd_gamma["Total"] if isinstance(mmd_gamma, dict) else mmd_gamma
    scores["Total"] = {
        "mmd": float(mmd_rbf(real_feat, fake_feat, gamma=gamma_total)),
        "w":   float(wasserstein_mean_per_dim(real_feat, fake_feat))
    }
    return scores

PHI_LAYOUT = {
    "StructTex": slice(0, 32),   # 32 structural/texture scalars
    "Spectrum":  slice(32, 96),  # 64 spectral bins
    "Energy":    slice(96, 160), # 64 amplitude-histogram bins
}
PHI_DIM = max(sl.stop for sl in PHI_LAYOUT.values())


def standardize_features_by_real(
    real_feat: np.ndarray,
    fake_feat: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features using statistics fitted only on the real reference set.

    Dimensions whose real-set standard deviation is <= ``eps`` are marked
    invalid and set to zero in both standardized arrays.  This avoids
    dividing by a near-zero scale while preserving the original column layout
    needed by ``PHI_LAYOUT``.  Callers may remove invalid columns for metrics
    that operate only on the full embedding (for example KID and PRDC).
    """
    R = np.asarray(real_feat, dtype=np.float64)
    F_ = np.asarray(fake_feat, dtype=np.float64)

    if R.ndim != 2 or F_.ndim != 2:
        raise ValueError(f"Expected 2-D feature arrays, got {R.shape} and {F_.shape}")
    if R.shape[1] != F_.shape[1]:
        raise ValueError(f"Feature dimensions differ: real={R.shape[1]}, fake={F_.shape[1]}")
    if R.shape[0] < 2 or F_.shape[0] < 2:
        raise ValueError("At least two real and two fake samples are required")
    if not np.isfinite(R).all() or not np.isfinite(F_).all():
        raise ValueError("Feature arrays contain NaN or Inf values")

    mu = R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True)
    valid = np.isfinite(sd[0]) & (sd[0] > float(eps))
    if not np.any(valid):
        raise ValueError("No non-constant real-reference feature dimensions remain")

    scale = np.where(valid[None, :], sd + float(eps), 1.0)
    Rn = (R - mu) / scale
    Fn = (F_ - mu) / scale
    Rn[:, ~valid] = 0.0
    Fn[:, ~valid] = 0.0

    return Rn, Fn, mu, sd, valid

def phi_features_single(gray01: np.ndarray,
                        sgs_params: Dict[str,Any],
                        agc_params: Dict[str,Any],
                        det_params: Dict[str,Any],
                        edge_type: str = "glog",
                        n_spec: int = 64,
                        n_hist: int = 64) -> np.ndarray:

    pp = preprocess_image_once(gray01, sgs_params, agc_params)
    coh = pp["coherence"].astype(np.float32)
    ori = pp["orientation"].astype(np.float32)
    agc = np.clip(pp["agc"].astype(np.float32), 0.0, 1.0)

    edge = compute_edge_map_from_preprocessed(pp, det_params, edge_type=edge_type)

    # ---- Structure scalars (example 16 dims) ----
    coh_mean = float(coh.mean()); coh_std = float(coh.std())
    coh_q = np.percentile(coh, [10,50,90]).astype(np.float32)

    # orientation: use sin/cos mean to avoid wrap issues (still scalar)
    ori_clip = np.clip(ori, -np.pi/2, np.pi/2)
    sin_m = float(np.sin(ori_clip).mean())
    cos_m = float(np.cos(ori_clip).mean())

    edge_thr = float(det_params.get("edge_thr_eval", 0.05))
    edge_density = float((edge > edge_thr).mean())
    edge_mean = float(edge.mean())
    edge_q = np.percentile(edge, [50, 90]).astype(np.float32)

    # a few texture scalars
    gx = np.gradient(agc, axis=1)
    gy = np.gradient(agc, axis=0)
    gmag = np.sqrt(gx*gx + gy*gy + 1e-12).astype(np.float32)
    g_mean = float(gmag.mean()); g_std = float(gmag.std())
    lap = gaussian_laplace(agc, sigma=1.0).astype(np.float32)
    lap_energy = float(np.mean(np.abs(lap)))

    base_16 = np.array([
        coh_mean, coh_std, coh_q[0], coh_q[1], coh_q[2],
        sin_m, cos_m,
        edge_density, edge_mean, edge_q[0], edge_q[1],
        g_mean, g_std, lap_energy,
        float(np.percentile(gmag, 90)), float(np.percentile(agc, 90)),
    ], dtype=np.float32)  # 16 dims

    extra_16 = np.array([
        float(np.percentile(coh, 25)), float(np.percentile(coh, 75)),
        float(np.mean(np.abs(ori_clip))), float(np.std(ori_clip)),
        float(np.percentile(edge, 10)), float(np.percentile(edge, 95)),
        float(np.percentile(gmag, 50)), float(np.percentile(gmag, 95)),
        float(np.mean(gx*gx)), float(np.mean(gy*gy)),
        float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy))),
        float(np.percentile(agc, 10)), float(np.percentile(agc, 50)),
        float(np.mean(np.abs(lap))), float(np.std(lap)),
    ], dtype=np.float32)

    struct_tex = np.concatenate([base_16, extra_16], axis=0).astype(np.float32)

    # ---- Spectrum vector (64 dims) ----
    H, W = agc.shape
    x = agc - agc.mean()
    spec = np.fft.rfft(x, axis=0)
    amp = (np.abs(spec).mean(axis=1) + 1e-12).astype(np.float32)  # (Hf,)
    # take first n_spec bins (pad if needed)
    if amp.shape[0] >= n_spec:
        spec64 = amp[:n_spec]
    else:
        spec64 = np.pad(amp, (0, n_spec-amp.shape[0]), mode='constant')
    spec64 = spec64 / (spec64.sum() + 1e-12)

    # ---- Amplitude histogram of agc (64 dims) ----
    hist, _ = np.histogram(agc, bins=n_hist, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)

    phi = np.concatenate([struct_tex, spec64, hist], axis=0).astype(np.float32)
    if phi.shape[0] != PHI_DIM:
        raise ValueError(
            f"Unexpected phi dimension {phi.shape[0]}; expected {PHI_DIM}. "
            f"Keep n_spec=64 and n_hist=64 when using the fixed PHI_LAYOUT."
        )
    return phi

def phi_features_batch(real_imgs: torch.Tensor,
                       fake_imgs: torch.Tensor,
                       sgs_params: Dict[str,Any],
                       agc_params: Dict[str,Any],
                       det_params: Dict[str,Any],
                       edge_type: str = "glog") -> Tuple[np.ndarray, np.ndarray]:
    real_gray = _to_gray01_from_tensor(real_imgs)
    fake_gray = _to_gray01_from_tensor(fake_imgs)

    R, F = [], []
    N = real_gray.shape[0]
    for i in range(N):
        R.append(phi_features_single(real_gray[i], sgs_params, agc_params, det_params, edge_type=edge_type))
        F.append(phi_features_single(fake_gray[i], sgs_params, agc_params, det_params, edge_type=edge_type))
    return np.stack(R, 0), np.stack(F, 0)

def phi_features_only(
    imgs: torch.Tensor,
    sgs_params,
    agc_params,
    det_params,
    edge_type="glog",
):
    gray = _to_gray01_from_tensor(imgs)

    out = []

    for i in range(gray.shape[0]):
        out.append(
            phi_features_single(
                gray[i],
                sgs_params,
                agc_params,
                det_params,
                edge_type=edge_type,
            )
        )

    return np.stack(out, axis=0)

def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    cov1 = np.atleast_2d(cov1).astype(np.float64)
    cov2 = np.atleast_2d(cov2).astype(np.float64)

    # numerical stability
    cov1 = cov1 + np.eye(cov1.shape[0]) * eps
    cov2 = cov2 + np.eye(cov2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(cov1 + cov2 - 2.0 * covmean))

def frechet_phi_scores(
    phi_real: np.ndarray,
    phi_fake: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute total and grouped FD from a common phi coordinate system.

    The caller is responsible for standardization.  ``valid_mask`` is applied
    consistently to the total embedding and to each PHI_LAYOUT group.
    """
    R_all = np.asarray(phi_real, dtype=np.float64)
    F_all = np.asarray(phi_fake, dtype=np.float64)
    if R_all.ndim != 2 or F_all.ndim != 2 or R_all.shape[1] != F_all.shape[1]:
        raise ValueError(f"Invalid phi shapes: real={R_all.shape}, fake={F_all.shape}")
    if R_all.shape[1] != PHI_DIM:
        raise ValueError(f"Expected {PHI_DIM}-D phi features, got {R_all.shape[1]}")

    if valid_mask is None:
        valid = np.ones(R_all.shape[1], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != R_all.shape[1]:
            raise ValueError(f"valid_mask has {valid.size} entries; expected {R_all.shape[1]}")

    def _fd(R: np.ndarray, F_: np.ndarray) -> float:
        if R.shape[1] == 0:
            return float("nan")
        return frechet_distance(
            R.mean(axis=0), np.cov(R, rowvar=False),
            F_.mean(axis=0), np.cov(F_, rowvar=False),
        )

    scores = {"Total": _fd(R_all[:, valid], F_all[:, valid])}
    for name, sl in PHI_LAYOUT.items():
        group_valid = valid[sl]
        scores[name] = _fd(R_all[:, sl][:, group_valid], F_all[:, sl][:, group_valid])
    return scores

def rr_frechet_thresholds(
    phi_real: np.ndarray,
    trials: int = 20,
    percentile: float = 95.0,
    min_per_split: int = 64,
    seed: int = 0,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:

    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    phi_real = np.asarray(phi_real, dtype=np.float64)
    if phi_real.ndim != 2 or phi_real.shape[1] != PHI_DIM:
        raise ValueError(f"Expected phi_real with shape (N,{PHI_DIM}), got {phi_real.shape}")
    N = phi_real.shape[0]

    if valid_mask is None:
        valid = np.ones(phi_real.shape[1], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != phi_real.shape[1]:
            raise ValueError(f"valid_mask has {valid.size} entries; expected {phi_real.shape[1]}")
    if not np.any(valid):
        return {}

    if N < 2 * int(min_per_split):
        return {}

    thr = {}

    def _fd(A, B):
        mu_a = A.mean(axis=0)
        mu_b = B.mean(axis=0)
        cov_a = np.cov(A, rowvar=False)
        cov_b = np.cov(B, rowvar=False)
        return frechet_distance(mu_a, cov_a, mu_b, cov_b)

    rr = {k: [] for k in ["Total"] + list(PHI_LAYOUT.keys())}

    for _ in range(int(trials)):
        idx = rng.permutation(N)

        nA = max(int(min_per_split), N // 2)
        nB = N - nA
        if nB < int(min_per_split):
            nB = int(min_per_split)
            nA = N - nB
        if nA < int(min_per_split) or nB < int(min_per_split):
            continue

        A = phi_real[idx[:nA]]
        B = phi_real[idx[nA:nA + nB]]

        rr["Total"].append(_fd(A[:, valid], B[:, valid]))
        for name, sl in PHI_LAYOUT.items():
            group_valid = valid[sl]
            if np.any(group_valid):
                rr[name].append(_fd(A[:, sl][:, group_valid], B[:, sl][:, group_valid]))

    if len(rr["Total"]) < max(3, int(trials) // 3):
        return {}

    q = float(percentile)
    for k, vals in rr.items():
        if vals:
            thr[k] = float(np.percentile(np.asarray(vals, dtype=np.float64), q))

    return thr

def rr_kid_thresholds(
    real_feat: np.ndarray,
    trials: int = 20,
    percentile: float = 95.0,
    subset_size: int = 100,
    n_subsets: int = 20,
    degree: int = 3,
    gamma: float = None,
    coef0: float = 1.0,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Real-Real KID baseline by repeatedly splitting real features into two halves.
    Returns percentile threshold of KID(MMD2) across trials.
    """
    X = np.asarray(real_feat, dtype=np.float64)
    N = X.shape[0]
    if N < 4:
        return {}

    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    vals = []
    for t in range(int(trials)):
        perm = rng.permutation(N)
        half = N // 2
        A = X[perm[:half]]
        B = X[perm[half:2 * half]]
        if A.shape[0] < 2 or B.shape[0] < 2:
            continue

        # match eval protocol: z-score by "real" stats (use A as "real")
        mu = A.mean(axis=0, keepdims=True)
        sd = A.std(axis=0, keepdims=True) + 1e-8
        A_n = (A - mu) / sd
        B_n = (B - mu) / sd

        m, _ = kid_poly_mmd2(
            A_n, B_n,
            subset_size=subset_size,
            n_subsets=n_subsets,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            seed=int(seed + t),
        )
        if np.isfinite(m):
            vals.append(float(m))

    if not vals:
        return {}
    return {
        "thr": float(np.percentile(vals, percentile)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def rr_swd_thresholds(
    real_imgs: torch.Tensor,
    trials: int = 10,
    percentile: float = 95.0,
    scales=(1, 2, 4),
    patch_size: int = 7,
    n_patches: int = 128,
    n_projections: int = 128,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Real-Real SWD baseline by repeatedly splitting real images into two halves.
    Returns percentile thresholds for each scale + Total.
    """
    if (real_imgs is None) or (not torch.is_tensor(real_imgs)):
        return {}
    N = int(real_imgs.shape[0])
    if N < 4:
        return {}

    rng = np.random.RandomState(int(seed) & 0x7fffffff)

    vals = {f"s{int(sc)}": [] for sc in scales}
    vals["Total"] = []

    for t in range(int(trials)):
        perm = rng.permutation(N)
        half = N // 2
        A = real_imgs[perm[:half]]
        B = real_imgs[perm[half:2 * half]]
        if A.shape[0] < 2 or B.shape[0] < 2:
            continue

        s = multiscale_swd(
            A, B,
            scales=scales,
            patch_size=patch_size,
            n_patches=n_patches,
            n_projections=n_projections,
            seed=int(seed + t),
        )
        for k, v in s.items():
            if np.isfinite(v):
                vals[k].append(float(v))

    if len(vals["Total"]) == 0:
        return {}

    thr = {}
    for k, arr in vals.items():
        if len(arr) > 0:
            thr[k] = float(np.percentile(arr, percentile))
    return thr


def rr_w1ks_thresholds(
    phi_real: np.ndarray,
    trials: int = 20,
    percentile: float = 95.0,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Real-Real W1/KS baseline on phi features by repeated split.
    Returns percentile thresholds for each group and Total.
    """
    X = np.asarray(phi_real, dtype=np.float64)
    N = X.shape[0]
    if N < 4:
        return {}

    rng = np.random.RandomState(int(seed) & 0x7fffffff)

    w1_vals = {name: [] for name in PHI_LAYOUT.keys()}
    ks_vals = {name: [] for name in PHI_LAYOUT.keys()}

    for t in range(int(trials)):
        perm = rng.permutation(N)
        half = N // 2
        A = X[perm[:half]]
        B = X[perm[half:2 * half]]
        if A.shape[0] < 2 or B.shape[0] < 2:
            continue

        s = group_w1_ks(A, B)
        for name, d in s.items():
            w1_vals[name].append(float(d["w1_mean"]))
            ks_vals[name].append(float(d["ks_mean"]))

    out = {}
    for name in PHI_LAYOUT.keys():
        if len(w1_vals[name]) == 0:
            continue
        out[name] = {
            "w1_thr": float(np.percentile(w1_vals[name], percentile)),
            "ks_thr": float(np.percentile(ks_vals[name], percentile)),
        }
    return out

def group_w1_ks(phi_real: np.ndarray, phi_fake: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, sl in PHI_LAYOUT.items():
        R = phi_real[:, sl]
        F = phi_fake[:, sl]
        # per-dimension
        w1s, kss = [], []
        for d in range(R.shape[1]):
            w1s.append(wasserstein_distance(R[:, d], F[:, d]))
            kss.append(ks_2samp(R[:, d], F[:, d]).statistic)
        out[name] = {"w1_mean": float(np.mean(w1s)), "ks_mean": float(np.mean(kss))}
    return out

def _pairwise_dist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # (Na,D),(Nb,D) -> (Na,Nb)
    A = A.astype(np.float64); B = B.astype(np.float64)
    AA = np.sum(A*A, axis=1, keepdims=True)
    BB = np.sum(B*B, axis=1, keepdims=True).T
    return AA + BB - 2.0 * A.dot(B.T)

# =========================
# KID / (unbiased) polynomial MMD^2  +  PRDC  +  bootstrap CI  +  multi-scale SWD
# Notes:
# - KID: unbiased MMD^2 with polynomial kernel, estimated by averaging over random subsets (like common KID practice).
# - PRDC: precision/recall/density/coverage (ICML 2020) computed on an embedding space (here: phi / geo z-scored).
# - SWD: multi-scale sliced Wasserstein distance over random patches (Progressive GAN-style texture statistic).
# =========================
def compute_prdc(phi_real: np.ndarray, phi_fake: np.ndarray, k: int = 5) -> Dict[str, float]:

    R = phi_real.astype(np.float64)
    F = phi_fake.astype(np.float64)
    k = int(k)
    if k < 1:
        k = 1
    if R.shape[0] <= k or F.shape[0] <= k:
        return {"precision": np.nan, "recall": np.nan, "density": np.nan, "coverage": np.nan}

    # distances
    dRR = np.sqrt(np.maximum(_pairwise_dist2(R, R), 0.0) + 1e-12)
    np.fill_diagonal(dRR, np.inf)
    r_kth = np.partition(dRR, kth=k-1, axis=1)[:, k-1]  # (Nr,)

    dFF = np.sqrt(np.maximum(_pairwise_dist2(F, F), 0.0) + 1e-12)
    np.fill_diagonal(dFF, np.inf)
    f_kth = np.partition(dFF, kth=k-1, axis=1)[:, k-1]  # (Nf,)

    dFR = np.sqrt(np.maximum(_pairwise_dist2(F, R), 0.0) + 1e-12)  # (Nf, Nr)

    # precision: Y_j in manifold(X)
    precision = float((dFR <= r_kth[None, :]).any(axis=1).mean())

    # recall: X_i in manifold(Y)
    recall = float((dFR.T <= f_kth[None, :]).any(axis=1).mean())

    # density: average number of real-balls covering each fake, normalized by k
    density = float((dFR <= r_kth[None, :]).sum(axis=1).mean() / float(k))

    # coverage: fraction of real points covered by at least one fake in its real-ball
    coverage = float((dFR <= r_kth[None, :]).any(axis=0).mean())

    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


def prdc_bootstrap_ci(phi_real: np.ndarray,
                      phi_fake: np.ndarray,
                      k: int = 5,
                      n_boot: int = 200,
                      ci: float = 95.0,
                      seed: int = 0) -> Dict[str, Dict[str, float]]:

    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    Nr = phi_real.shape[0]
    Nf = phi_fake.shape[0]

    base = compute_prdc(phi_real, phi_fake, k=k)
    if n_boot <= 0:
        return {m: {"mean": float(base[m]), "lo": float("nan"), "hi": float("nan")} for m in base.keys()}

    stats = {m: [] for m in base.keys()}
    for _ in range(int(n_boot)):
        ir = rng.randint(0, Nr, size=Nr)
        jf = rng.randint(0, Nf, size=Nf)
        out = compute_prdc(phi_real[ir], phi_fake[jf], k=k)
        for m in stats.keys():
            stats[m].append(float(out[m]))

    alpha = (100.0 - float(ci)) / 2.0
    res = {}
    for m, arr in stats.items():
        a = np.asarray(arr, dtype=np.float64)
        res[m] = {
            "mean": float(np.nanmean(a)),
            "lo": float(np.nanpercentile(a, alpha)),
            "hi": float(np.nanpercentile(a, 100.0 - alpha)),
        }
    return res


def prdc_curve_with_ci(phi_real: np.ndarray,
                       phi_fake: np.ndarray,
                       k_list: List[int],
                       n_boot: int = 200,
                       ci: float = 95.0,
                       seed: int = 0) -> Dict[int, Dict[str, Dict[str, float]]]:

    out = {}
    for i, k in enumerate(list(k_list)):
        out[int(k)] = prdc_bootstrap_ci(phi_real, phi_fake, k=int(k),
                                        n_boot=int(n_boot), ci=float(ci), seed=int(seed + 1337 * i))
    return out

def _unbiased_mmd2_from_kernel(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """
    Unbiased MMD^2 estimate given kernel matrices.
    Kxx: (m,m), Kyy: (n,n), Kxy: (m,n)
    """
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    # remove diagonal terms for unbiased estimate
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = np.sum(Kxy) / (m * n)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)

def polynomial_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: float = None, coef0: float = 1.0) -> np.ndarray:
    """
    k(x,y) = (gamma * <x,y> + coef0) ^ degree
    If gamma is None: gamma = 1 / dim.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = X.shape[1]
    g = (1.0 / float(d)) if (gamma is None) else float(gamma)
    return (g * (X @ Y.T) + float(coef0)) ** int(degree)

def kid_poly_mmd2(
    X: np.ndarray,
    Y: np.ndarray,
    subset_size: int = 100,
    n_subsets: int = 20,
    degree: int = 3,
    gamma: float = None,
    coef0: float = 1.0,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Kernel Inception Distance style estimator (mean, std) using polynomial-kernel unbiased MMD^2.
    Works on ANY embedding (not necessarily Inception).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.RandomState(int(seed) & 0x7fffffff)

    m = X.shape[0]; n = Y.shape[0]
    ss = int(min(subset_size, m, n))
    if ss < 2:
        return float("nan"), float("nan")

    vals = []
    for _ in range(int(n_subsets)):
        ix = rng.choice(m, size=ss, replace=False) if ss < m else np.arange(m)
        iy = rng.choice(n, size=ss, replace=False) if ss < n else np.arange(n)
        Xs = X[ix]
        Ys = Y[iy]
        Kxx = polynomial_kernel(Xs, Xs, degree=degree, gamma=gamma, coef0=coef0)
        Kyy = polynomial_kernel(Ys, Ys, degree=degree, gamma=gamma, coef0=coef0)
        Kxy = polynomial_kernel(Xs, Ys, degree=degree, gamma=gamma, coef0=coef0)
        vals.append(_unbiased_mmd2_from_kernel(Kxx, Kyy, Kxy))

    vals = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(vals)), float(np.nanstd(vals))

def _extract_random_patches_2d(imgs01: np.ndarray, patch_size: int, n_patches: int, rng: np.random.RandomState) -> np.ndarray:
    """
    imgs01: (N,H,W) float32 in [0,1]
    returns: (n_patches, patch_size*patch_size) float32 (per-patch standardized)
    """
    imgs01 = np.asarray(imgs01, dtype=np.float32)
    N, H, W = imgs01.shape
    ps = int(patch_size)
    if H < ps or W < ps:
        raise ValueError(f"patch_size {ps} is larger than image ({H},{W})")
    out = np.empty((int(n_patches), ps * ps), dtype=np.float32)
    for i in range(int(n_patches)):
        idx = int(rng.randint(0, N))
        y = int(rng.randint(0, H - ps + 1))
        x = int(rng.randint(0, W - ps + 1))
        p = imgs01[idx, y:y+ps, x:x+ps].reshape(-1)
        # per-patch standardization (texture-focused)
        p = p - p.mean()
        p = p / (p.std() + 1e-8)
        out[i] = p.astype(np.float32)
    return out

def sliced_wasserstein_distance_patches(
    patches_a: np.ndarray,
    patches_b: np.ndarray,
    n_projections: int = 128,
    rng: np.random.RandomState = None,
) -> float:
    """
    SWD between two patch sets using random 1D projections.
    """
    A = np.asarray(patches_a, dtype=np.float32)
    B = np.asarray(patches_b, dtype=np.float32)
    if rng is None:
        rng = np.random.RandomState(0)
    assert A.shape[1] == B.shape[1]
    d = A.shape[1]
    P = int(n_projections)

    dirs = rng.normal(size=(P, d)).astype(np.float32)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

    proj_a = A @ dirs.T  # (Na,P)
    proj_b = B @ dirs.T  # (Nb,P)
    proj_a.sort(axis=0)
    proj_b.sort(axis=0)

    # match counts by min length (usually equal)
    m = min(proj_a.shape[0], proj_b.shape[0])
    return float(np.mean(np.abs(proj_a[:m] - proj_b[:m])))

def multiscale_swd(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    scales: Tuple[int, ...] = (1, 2, 4),
    patch_size: int = 7,
    n_patches: int = 256,
    n_projections: int = 128,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Multi-scale SWD on raw images (texture statistic).
    real_imgs/fake_imgs: (N,C,H,W) in [-1,1]
    Returns dict with per-scale SWD and 'Total' (mean across scales).
    """
    rng = np.random.RandomState(int(seed) & 0x7fffffff)

    real01 = _to_gray01_from_tensor(real_imgs)  # (N,H,W) float32 [0,1]
    fake01 = _to_gray01_from_tensor(fake_imgs)

    out = {}
    vals = []
    for s in scales:
        s = int(s)
        if s <= 0:
            continue
        if s == 1:
            r_s = real01
            f_s = fake01
        else:
            # downsample using torch for consistency
            tr = torch.from_numpy(real01).unsqueeze(1)  # (N,1,H,W)
            tf = torch.from_numpy(fake01).unsqueeze(1)
            tr_s = F.interpolate(tr, scale_factor=1.0/float(s), mode="bilinear", align_corners=False)
            tf_s = F.interpolate(tf, scale_factor=1.0/float(s), mode="bilinear", align_corners=False)
            r_s = tr_s.squeeze(1).numpy()
            f_s = tf_s.squeeze(1).numpy()

        pa = _extract_random_patches_2d(r_s, patch_size=patch_size, n_patches=n_patches, rng=rng)
        pb = _extract_random_patches_2d(f_s, patch_size=patch_size, n_patches=n_patches, rng=rng)
        swd = sliced_wasserstein_distance_patches(pa, pb, n_projections=n_projections, rng=rng)
        out[f"s{s}"] = float(swd)
        vals.append(float(swd))

    out["Total"] = float(np.mean(vals)) if len(vals) else float("nan")
    return out

def scores_to_radar_values(scores: Dict[str, Dict[str, float]],
                           metric: str = "mmd",
                           invert: bool = False,
                           eps: float = 1e-12) -> Tuple[List[str], List[float]]:

    labels = ["Structure", "Texture", "Spectrum", "Energy"]
    vals = []
    for k in labels:
        v = float(scores[k][metric])
        if invert:
            v = 1.0 / (v + eps)
        vals.append(v)
    return labels, vals

def _rand_int(a, b, device=None):
    return int(torch.randint(low=a, high=b+1, size=(1,), device=device).item())

def shift_pad_crop(x: torch.Tensor, dy: int, dx: int, fill: float = -1.0) -> torch.Tensor:

    B, C, H, W = x.shape
    out = x.new_full((B, C, H, W), float(fill))

    y0_dst = max(0, dy)
    y1_dst = min(H, H + dy)   
    x0_dst = max(0, dx)
    x1_dst = min(W, W + dx)

    y0_src = max(0, -dy)
    y1_src = min(H, H - dy)
    x0_src = max(0, -dx)
    x1_src = min(W, W - dx)

    if (y1_dst > y0_dst) and (x1_dst > x0_dst):
        out[:, :, y0_dst:y1_dst, x0_dst:x1_dst] = x[:, :, y0_src:y1_src, x0_src:x1_src]
    return out

def paired_geom_aug_real(real_raw: torch.Tensor,
                         p: float = 0.2,
                         max_px: int = 1,
                         hflip_prob: float = 0.5,
                         fill: float = -1.0) -> torch.Tensor:

    if p <= 0.0:
        return real_raw

    x = real_raw
    B, C, H, W = x.shape
    out = x.clone()

    for b in range(B):
        if random.random() < p:
            # hflip
            if random.random() < hflip_prob:
                out[b] = torch.flip(out[b], dims=[2])  

            # jitter shift
            if max_px and int(max_px) > 0:
                dx = random.randint(-int(max_px), int(max_px))
                dy = random.randint(-int(max_px), int(max_px))
                out[b:b+1] = shift_pad_crop(out[b:b+1], dy=dy, dx=dx, fill=fill)

    return out

def _density(edge, thr=0.0):
    # edge: [B,1,H,W]
    return (edge > thr).float().mean(dim=(2,3), keepdim=True)  # [B,1,1,1]

def cond_augment_after_thr(cond_tex, args, allow_jitter: bool = True):
    """
    Apply AFTER detector thresholding. Only modify EDGE channel.
    cond_tex: [B,C,H,W] in [-1,1]
    """
    if random.random() >= float(getattr(args, "cond_aug_prob", 0.0)):
        return cond_tex

    x = cond_tex
    B, C, H, W = x.shape

    # locate edge channel index from tex_layout
    layout = str(getattr(args, "tex_layout", "")).strip().lower()
    keys = [k.strip() for k in layout.split(",") if k.strip()]
    edge_idx = keys.index("edge") if "edge" in keys else (C - 1)

    # --- struct channel indices (fault / coherence) ---
    struct_names = ("fault", "coherence")
    struct_idx = [keys.index(n) for n in struct_names if n in keys]

    # extract edge
    edge = x[:, edge_idx:edge_idx+1, :, :]          # [-1,1]
    edge01 = (edge + 1.0) * 0.5                     # [0,1]

    # -------- density-aware scaling --------
    d = edge01.mean(dim=(2,3), keepdim=True)
    d_low  = float(getattr(args, "edge_dense_low", 0.35))
    d_high = float(getattr(args, "edge_dense_high", 0.60))
    dense_boost = float(getattr(args, "edge_dense_boost", 1.8))
    t = ((d - d_low) / max(1e-6, (d_high - d_low))).clamp(0.0, 1.0)
    s = 1.0 + t * (dense_boost - 1.0)

    # -------- 1) pixel dropout (edge only) --------
    base  = float(getattr(args, "edge_drop_base", 0.0))
    boost = float(getattr(args, "edge_drop_boost", 0.0))
    gamma = float(getattr(args, "edge_drop_gamma", 1.0))

    strength = edge01.clamp(0, 1).pow(gamma)
    drop_prob = (base + boost * strength) * s
    drop_prob = drop_prob.clamp(0.0, 0.95)
    keep = (torch.rand_like(edge01) > drop_prob).float()
    edge01 = edge01 * keep

    # write back edge into [-1,1]
    edge = edge01 * 2.0 - 1.0                       # [-1,1]

    # -------- 2) block cutout (edge only) --------
    if random.random() < float(getattr(args, "edge_block_prob", 0.0)):
        bs = int(getattr(args, "edge_block_size", 16))
        bn = int(getattr(args, "edge_block_num", 1))
        bs = max(1, min(bs, H, W))
        for b in range(B):
            for _ in range(bn):
                y0 = _rand_int(0, max(0, H - bs), device=edge.device)
                x0 = _rand_int(0, max(0, W - bs), device=edge.device)
                edge[b:b+1, :, y0:y0+bs, x0:x0+bs] = -1.0  # 对应 edge01=0

    # -------- 3) downsample->upsample (edge only) --------
    if random.random() < float(getattr(args, "edge_ds_prob", 0.0)):
        fmin = int(getattr(args, "edge_ds_min", 2))
        fmax = int(getattr(args, "edge_ds_max", 8))
        f = _rand_int(fmin, fmax, device=edge.device)
        if f > 1:
            e_small = F.interpolate(edge, scale_factor=1.0/f, mode='bilinear', align_corners=False)
            edge = F.interpolate(e_small, size=(H, W), mode='bilinear', align_corners=False)

    # -------- 4) jitter (edge only) --------
    if allow_jitter and (random.random() < float(getattr(args, "cond_jitter_prob", 0.0))):
        max_px = int(getattr(args, "cond_jitter_px", 0))
        if max_px > 0:
            dy = random.randint(-max_px, max_px)
            dx = random.randint(-max_px, max_px)
            edge = shift_pad_crop(edge, dy=dy, dx=dx, fill=-1.0)

    # -------- write edge back without changing other channels --------
    x_out = x.clone()
    x_out[:, edge_idx:edge_idx+1, :, :] = edge

    # =========================
    # Extra: struct channels augment (fault/coherence), mild (不改几何)
    # =========================
    if struct_idx:
        if random.random() < float(getattr(args, "cond_channel_drop_prob", 0.0)):
            for si in struct_idx:
                x_out[:, si:si+1, :, :] = 0.0

        p_pix = float(getattr(args, "cond_struct_pixel_dropout_rate", 0.0))
        if p_pix > 0:
            for si in struct_idx:
                m = (torch.rand_like(x_out[:, si:si+1]) < p_pix)
                x_out[:, si:si+1][m] = 0.0

        if random.random() < float(getattr(args, "cond_struct_block_prob", 0.0)):
            bs = int(getattr(args, "cond_struct_block_size", 32))
            bn = int(getattr(args, "cond_struct_block_num", 1))
            bs = max(1, min(bs, H, W))
            for b in range(B):
                for _ in range(bn):
                    y0 = _rand_int(0, max(0, H - bs), device=x_out.device)
                    x0 = _rand_int(0, max(0, W - bs), device=x_out.device)
                    for si in struct_idx:
                        x_out[b, si:si+1, y0:y0+bs, x0:x0+bs] = 0.0

    return x_out


def shift_pad_crop_chw(x_chw: torch.Tensor, dy: int, dx: int, pad_mode = 'reflect') -> torch.Tensor:
    # x_chw: (C,H,W)
    C, H, W = x_chw.shape

    top = max(0, dy);
    bottom = max(0, -dy)
    left = max(0, dx);
    right = max(0, -dx)
    x_pad = F.pad(x_chw, (left, right, top, bottom), mode = pad_mode)
    y0 = bottom
    x0 = right
    return x_pad[:, y0:y0 + H, x0:x0 + W]

def seismic_physics_aug(real: torch.Tensor, p: float,
                        gain_range=(0.7, 1.3),
                        gamma_range=(0.85, 1.15),
                        noise_std_range=(0.0, 0.04),
                        blur_sigma_range=(0.0, 1.2),
                        device=None,
                        stable_gamma: bool = False) -> torch.Tensor:

    if p <= 0.0:
        return real
    x = real
    N = x.shape[0]
    if device is None:
        device = x.device
        
    orig_dtype = x.dtype
    x01 = (x + 1.0) * 0.5

    out = []
    eps32 = 1e-12
    for i in range(N):
        xi = x01[i:i+1]  # (1,1,H,W)
        if random.random() < p:
            # 1) random gain
            g = random.uniform(*gain_range)
            xi = xi * g

        if random.random() < p:
            # 2) random gamma (amplitude nonlinearity).  The F3 profile uses
            # the stabilized endpoint-preserving implementation from the
            # field-domain runs; Australian training retains the original map.
            gm = random.uniform(*gamma_range)
            xi = torch.clamp(xi, 0.0, 1.0)
            if stable_gamma:
                eps_gamma = 1e-6
                y0 = eps_gamma ** gm
                y1 = (1.0 + eps_gamma) ** gm
                xi = ((xi + eps_gamma).pow(gm) - y0) / (y1 - y0)
                xi = torch.clamp(xi, 0.0, 1.0)
            else:
                xi = xi ** gm

        if random.random() < p:
            # 3) noise
            ns = random.uniform(*noise_std_range)
            if ns > 0:
                xi = xi + torch.randn_like(xi) * ns

        if random.random() < p:
            # 4) bandwidth proxy: slight blur
            bs = random.uniform(*blur_sigma_range)
            if bs > 0:
                # separable gaussian blur via conv (cheap)
                k = int(max(3, round(bs * 6))) | 1
                xi32 = xi.float()
                grid = torch.arange(k, device=device, dtype=torch.float32) - (k//2)
                denom = 2.0 * (bs * bs) + eps32
                gauss = torch.exp(-(grid * grid) / denom)
                gauss_sum = torch.clamp(gauss.sum(), min=eps32)
                gauss = gauss / gauss_sum
                
                g1 = gauss.view(1,1,k,1)
                g2 = gauss.view(1,1,1,k)
                
                xi32 = F.pad(xi32, (0, 0, k // 2, k // 2), mode='reflect')
                xi32 = F.conv2d(xi32, g1)
                xi32 = F.pad(xi32, (k // 2, k // 2, 0, 0), mode='reflect')
                xi32 = F.conv2d(xi32, g2)

                xi = xi32.to(dtype=orig_dtype)

        xi = torch.clamp(xi, 0.0, 1.0)
        out.append(xi)

    x01_aug = torch.cat(out, dim=0)
    return x01_aug * 2.0 - 1.0

@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))
    # buffers (e.g. running stats)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)

def cuda_mem_mb(device):
    if device.type != "cuda":
        return {}
    return {
        "mem_alloc_mb": float(torch.cuda.memory_allocated(device) / (1024**2)),
        "mem_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024**2)),
    }

@torch.no_grad()
def cond_channel_stats(cond_tex: torch.Tensor, keys: list):
    # cond_tex: (N,C,H,W) in [-1,1]
    x01 = (cond_tex + 1.0) * 0.5
    stats = {}
    for i, k in enumerate(keys):
        ch = x01[:, i]
        stats[f"{k}_mean"] = float(ch.mean().item())
        stats[f"{k}_std"]  = float(ch.std().item())
        if k == "edge":
            stats["edge_density@0.5"] = float((ch > 0.5).float().mean().item())
        if k == "fault":
            stats["fault_density@0.5"] = float((ch > 0.5).float().mean().item())
    return stats

class AdaptiveStructureController:
    def __init__(self,
                 mul_grad=1.0, mul_dir=1.0, mul_ori=1.0,
                 mul_min=0.3, mul_max=6.0,
                 up_strength=0.25, down_strength=0.08,
                 ema=0.8):
        self.mul_grad = float(mul_grad)
        self.mul_dir  = float(mul_dir)
        self.mul_ori  = float(mul_ori)
        self.mul_min  = float(mul_min)
        self.mul_max  = float(mul_max)
        self.up_s     = float(up_strength)
        self.down_s   = float(down_strength)
        self.ema      = float(ema)
        self.r_ema    = 1.0

    def update(self, struct_ratio: float):
        r = float(struct_ratio)
        self.r_ema = self.ema * self.r_ema + (1.0 - self.ema) * r

        e = self.r_ema - 1.0

        if e > 0:
            k = 1.0 + self.up_s * e
        else:
            k = 1.0 + self.down_s * e  

        def _upd(x):
            x = x * k
            return float(max(self.mul_min, min(self.mul_max, x)))

        self.mul_grad = _upd(self.mul_grad)
        self.mul_dir  = _upd(self.mul_dir)
        self.mul_ori  = _upd(self.mul_ori)

        return {"r_ema": float(self.r_ema),
                "mul_grad": float(self.mul_grad),
                "mul_dir": float(self.mul_dir),
                "mul_ori": float(self.mul_ori)}
        
# -------------------------
# Experiment profiles
# -------------------------
def _experiment_profile(args) -> str:
    profile = str(getattr(args, "profile", "australian")).strip().lower()
    if profile not in ("australian", "f3"):
        raise ValueError(f"Unknown experiment profile: {profile}")
    return profile

def _profile_config(args) -> Dict[str, Any]:
    profile = _experiment_profile(args)
    if profile == "f3":
        return {
            "img_size": 128,
            "epochs": 253,
            "sgs_sigma": 2.5,
            "sgs_anisotropy": 2.5,
            "sgs_gamma": 0.04,
            "agc_window": 11,
            "agc_p1": 3,
            "fault_w_coh": 0.6470588,
            "fault_w_ori": 0.3529412,
            "fault_w_edge": 0.0,
            "ada_pad": 5,
            "ada_translation_px": 1,
            "patch_scales": (1.0, 0.5),
            "cond_struct_block_size": 6,
            "edge_block_size": 4,
            "stable_gamma": True,
        }
    return {
        "img_size": 512,
        "epochs": 1000,
        "sgs_sigma": 1.5,
        "sgs_anisotropy": 2.2,
        "sgs_gamma": 0.06,
        "agc_window": 31,
        "agc_p1": 1,
        "fault_w_coh": 0.55,
        "fault_w_ori": 0.30,
        "fault_w_edge": 0.15,
        "ada_pad": 20,
        "ada_translation_px": 2,
        "patch_scales": (1.0, 0.5, 0.25),
        "cond_struct_block_size": 24,
        "edge_block_size": 16,
        "stable_gamma": False,
    }

def _apply_profile_cli_defaults(args):
    cfg = _profile_config(args)
    for name in ("img_size", "epochs", "cond_struct_block_size", "edge_block_size"):
        if getattr(args, name, None) is None:
            setattr(args, name, cfg[name])
    return args

# -------------------------
# Training loop
# -------------------------
def train(args):
    # === ADA setup ===
    ADA_TARGET   = float(getattr(args, "ada_target", 0.6))
    ADA_INTERVAL = int(getattr(args, "ada_interval", 4))
    ADA_SPEED    = float(getattr(args, "ada_speed", 0.0002))
    augment_p    = float(getattr(args, "augment_p_init", 0.2))
    sign_stat_ema = 0.0
    
    edge_type = getattr(args, "edge_type", "glog").strip().lower()
    if edge_type not in ("slog", "glog", "log", "canny"):
        raise ValueError(f"edge_type must be one of ['slog','glog','log','canny'], got '{edge_type}'")
    evaluation_edge_type = str(getattr(args, "eval_edge_type", "log")).strip().lower()
    if evaluation_edge_type not in ("slog", "glog", "log", "canny"):
        raise ValueError(
            f"eval_edge_type must be one of ['slog','glog','log','canny'], "
            f"got '{evaluation_edge_type}'"
        )
    if getattr(args, "device", ""):
        device = torch.device(str(args.device))
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        torch.cuda.set_device(device)
    print('Device:', device)
    amp = (args.amp and device.type == 'cuda')
    profile_cfg = _profile_config(args)

    logger = setup_logger(args.out_dir, "train")
    jlog   = JsonlWriter(os.path.join(args.out_dir, "metrics_train.jsonl"))
    elog   = JsonlWriter(os.path.join(args.out_dir, "metrics_eval.jsonl"))

    logger.info(
        f"Profile={args.profile} | Device={device} | AMP={amp} | img_size={args.img_size} | "
        f"batch={args.batch} | epochs={args.epochs} | condition_edge={edge_type} | "
        f"evaluation_edge={evaluation_edge_type} | tex_layout={args.tex_layout}"
    )
    ctrl = AdaptiveStructureController(
        mul_grad=1.0, mul_dir=1.0, mul_ori=1.0,
        mul_min=float(getattr(args, "rrfd_mul_min", 0.3)),
        mul_max=float(getattr(args, "rrfd_mul_max", 6.0)),
        up_strength=float(getattr(args, "rrfd_up", 0.25)),
        down_strength=float(getattr(args, "rrfd_down", 0.08)),
        ema=float(getattr(args, "rrfd_ema", 0.8)),
    )

    best_struct_ratio = float("inf")
    best_total_ratio  = float("inf")
    r1_gamma = float(getattr(args, "r1_gamma", 10.0))

    # --- spectrum loss config ---
    lambda_spec = float(getattr(args, "lambda_spec", 0.0))
    spec_scales = tuple(int(x) for x in str(getattr(args, "spec_scales", "1,2,4")).split(",") if x.strip())
    grad_scales = tuple(int(x) for x in str(args.grad_scales).split(",") if x.strip())
    # --- orientation loss config ---
    dir_scales_str = str(getattr(args, "grad_dir_scales", "")).strip()
    if dir_scales_str:
        grad_dir_scales = tuple(int(x) for x in dir_scales_str.split(",") if x.strip())
    else:
        grad_dir_scales = grad_scales

    ori_scales_str = str(getattr(args, "st_ori_scales", "")).strip()
    if ori_scales_str:
        st_ori_scales = tuple(int(x) for x in ori_scales_str.split(",") if x.strip())
    else:
        st_ori_scales = grad_scales

    st_sigmas = tuple(float(x) for x in str(getattr(args, "st_sigmas", "1.0")).split(",") if x.strip())
    if len(st_sigmas) == 0:
        st_sigmas = (1.0,)

    if len(spec_scales) == 0:
        spec_scales = (1, 2, 4)
    spec_use_log = bool(int(getattr(args, "spec_log", 1)))

    # --- eval metric config (PRDC / KID / SWD) ---
    prdc_ks = tuple(int(x) for x in str(getattr(args, "prdc_ks", "3,5,10,20")).split(",") if x.strip())
    if len(prdc_ks) == 0:
        prdc_ks = (5,)
    prdc_boot = int(getattr(args, "prdc_boot", 200))
    prdc_ci = float(getattr(args, "prdc_ci", 95.0))

    kid_subsets = int(getattr(args, "kid_subsets", 20))
    kid_subset_size = int(getattr(args, "kid_subset_size", 100))
    kid_degree = int(getattr(args, "kid_degree", 3))
    kid_coef0 = float(getattr(args, "kid_coef0", 1.0))
    kid_gamma = float(getattr(args, "kid_gamma", -1.0))
    kid_gamma = None if (kid_gamma < 0) else float(kid_gamma)

    swd_scales = tuple(int(x) for x in str(getattr(args, "swd_scales", "1,2,4")).split(",") if x.strip())
    if len(swd_scales) == 0:
        swd_scales = (1, 2, 4)
    swd_patch = int(getattr(args, "swd_patch", 7))
    swd_patches = int(getattr(args, "swd_patches", 256))
    swd_projections = int(getattr(args, "swd_projections", 128))
    
    edge_type, sgs_params, agc_params, det_params, cond_channels = _build_preproc_params(
        args, device, amp
    )
    layout = str(det_params.get("tex_layout", "sgs,agc,edge")).strip().lower()
    keys = [k.strip() for k in layout.split(",") if k.strip()]

    print(f"[cond] tex_layout={layout} -> keys={keys} -> cond_channels={cond_channels}")

    # ====== moved logger blocks (now all variables exist) ======
    logger.info(f"[cond] tex_layout={layout} | keys={keys} | cond_channels={cond_channels}")
    logger.info(
        "LossSchedule | "
        f"lambda_spec={lambda_spec} warmup={args.spec_warmup_steps} ramp={args.spec_ramp_steps} | "
        f"lambda_grad={args.lambda_grad} warmup={args.grad_warmup_steps} ramp={args.grad_ramp_steps} | "
        f"lambda_dir={args.lambda_grad_dir} warmup={args.grad_dir_warmup_steps} ramp={args.grad_dir_ramp_steps} | "
        f"lambda_ori={args.lambda_st_ori} warmup={args.st_ori_warmup_steps} ramp={args.st_ori_ramp_steps}"
    )
    logger.info(
        "ADA | "
        f"target={ADA_TARGET} interval={ADA_INTERVAL} speed={ADA_SPEED} augment_p_init={augment_p} | "
        f"r1_interval={args.r1_interval} r1_gamma={r1_gamma}"
    )

    class ADAAug(nn.Module):
        def __init__(self, max_rotate_deg=5, pad=20, brightness=0.1, contrast=0.1):
            super().__init__()
            self.max_rotate = float(max_rotate_deg)
            self.pad = int(pad)
            self.brightness = float(brightness)
            self.contrast = float(contrast)

        def _rotate_tensor(self, img, angle_deg):
            angle = -angle_deg * math.pi / 180.0
            C, H, W = img.shape
            theta = torch.tensor([
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle),  math.cos(angle), 0.0]
            ], dtype=img.dtype, device=img.device).unsqueeze(0)
            grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)
            img_b = img.unsqueeze(0)
            out = F.grid_sample(img_b, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            return out[0]

        def _brightness_contrast(self, img, b_strength, c_strength):
            if b_strength != 0:
                b = 1.0 + random.uniform(-b_strength, b_strength)
                img = img * b
            if c_strength != 0:
                c = 1.0 + random.uniform(-c_strength, c_strength)
                mean = img.mean(dim=[1,2], keepdim=True)
                img = (img - mean) * c + mean
            return img.clamp(0.0, 1.0)

        def forward(self, x, p):
            if p <= 0.0:
                return x
            amp = min(max(p, 0.0), 0.95)
            x01 = (x + 1.0) / 2.0
            out = []
            N, C, H, W = x01.shape
            pad = self.pad

            for i in range(N):
                xi = x01[i]
                if random.random() < amp:
                    xi = torch.flip(xi, dims=[2])
                if pad > 0 and random.random() < amp:
                    xi = F.pad(xi, (pad, pad, pad, pad), mode='reflect')
                    if self.max_rotate > 0:
                        ang = random.uniform(-self.max_rotate, self.max_rotate)
                        xi = self._rotate_tensor(xi, ang)
                    cH, cW = xi.shape[1], xi.shape[2]
                    if cH > H or cW > W:
                        sy = random.randint(0, max(0, cH - H))
                        sx = random.randint(0, max(0, cW - W))
                        xi = xi[:, sy:sy + H, sx:sx + W]

                if (self.brightness != 0 or self.contrast != 0) and random.random() < amp:
                    xi = self._brightness_contrast(xi, self.brightness, self.contrast)

                if random.random() < amp:
                    shift_px = int(profile_cfg["ada_translation_px"])
                    tx = random.randint(-shift_px, shift_px)
                    ty = random.randint(-shift_px, shift_px)
                    xi = shift_pad_crop_chw(xi, dy=ty, dx=tx, pad_mode='reflect')

                cH, cW = xi.shape[1], xi.shape[2]
                if cH != H or cW != W:
                    sy = (cH - H) // 2
                    sx = (cW - W) // 2
                    xi = xi[:, sy:sy + H, sx:sx + W]
                out.append(xi)
            x01_aug = torch.stack(out, dim=0).to(x.device)
            return x01_aug * 2.0 - 1.0

    ada_aug_geometric = ADAAug(
        max_rotate_deg=5,
        pad=int(profile_cfg["ada_pad"]),
        brightness=0.0,
        contrast=0.0,
    )

    ds = ImageFolderDataset(args.data_dir, crop_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    sample = ds[0]
    in_ch = int(sample.shape[0])

    if hasattr(T, 'InterpolationMode'):
        interp = T.InterpolationMode.BILINEAR
    else:
        interp = Image.BILINEAR

    G = GeneratorWithTexture(
        z_dim=args.z_dim,
        w_dim=args.w_dim,
        base_channels=512,
        img_size=args.img_size,
        out_channels=in_ch,
        cond_channels=cond_channels,
        enc_ch=64,
        cond_mode='film'
    ).to(device)

    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    apply_g_bounds_(G, args)
    apply_g_bounds_(G_ema, args)
    ema_decay = float(getattr(args, "ema_decay", 0.999))

    D_base = PatchDiscriminator(in_channels=in_ch + cond_channels,
                           base_channels=128, n_layers=5, use_mbstd=True).to(device)
    D_patch = MultiScaleDiscriminator(
        D_base, scales=tuple(profile_cfg["patch_scales"])
    ).to(device)
    D_global = GlobalDiscriminator(in_channels=in_ch + cond_channels,
                               base_channels=128, n_layers=6, use_mbstd=True).to(device)
    D = CombinedDiscriminator(D_patch, D_global).to(device)

    d_lr0 = float(getattr(args, "d_lr", 5e-5))
    g_lr0 = float(getattr(args, "g_lr", 1e-4))

    d_opt = torch.optim.Adam(D.parameters(), lr=d_lr0, betas=(0.0, 0.99))
    g_opt = torch.optim.Adam(G.parameters(), lr=g_lr0, betas=(0.0, 0.99))

    d_decay_start = int(getattr(args, "d_lr_decay_start", 14000))
    d_decay_every = int(getattr(args, "d_lr_decay_every", 1000))   
    d_decay_gamma = float(getattr(args, "d_lr_decay_gamma", 0.5))   
    d_lr_min      = float(getattr(args, "d_lr_min", 1e-6))          

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    fixed_z = torch.randn(8, args.z_dim, device=device)

    eval_cond_num = int(getattr(args, "eval_cond_num", 256))
    train_eval_seed = int(getattr(args, "train_eval_seed", 42))

    py_state = random.getstate()
    np_state = np.random.get_state()

    try:
        random.seed(train_eval_seed)
        np.random.seed(train_eval_seed)

        rng = np.random.RandomState(train_eval_seed + 123)

        fixed_cond_imgs = []
        for _ in range(eval_cond_num):
            idx = int(rng.randint(0, len(ds)))
            fixed_cond_imgs.append(ds[idx].unsqueeze(0))

        fixed_cond_imgs = torch.cat(
            fixed_cond_imgs,
            dim=0
        ).to(device)

        with torch.no_grad():
            fixed_cond_tex = compute_texture_maps_batch(
                fixed_cond_imgs,
                sgs_params,
                agc_params,
                det_params,
                device=device,
                edge_type=edge_type
            )

    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    
    step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(loader)
        for real in pbar:
            t_step0 = time.time()
            real = real.to(device)
            batch_size = real.shape[0]

            if step >= d_decay_start:

                k = (step - d_decay_start) // d_decay_every   
                new_lr = max(d_lr0 * (d_decay_gamma ** k), d_lr_min)

                for pg in d_opt.param_groups:
                    pg["lr"] = new_lr

            real_raw = real  # (N,1,H,W) in [-1,1]
            phys_p = float(getattr(args, "phys_aug_prob", 0.6))

            pair_p   = float(getattr(args, "pair_geom_prob", 0.20))
            pair_px  = int(getattr(args, "pair_geom_px", 1))
            pair_hfp = float(getattr(args, "pair_geom_hflip", 0.50))

            real_pair = paired_geom_aug_real(
                real_raw, p=pair_p, max_px=pair_px, hflip_prob=pair_hfp, fill=-1.0
            )

            with torch.no_grad():
                cond_tex = compute_texture_maps_batch(
                    real_pair, sgs_params, agc_params, det_params, device=device, edge_type=edge_type,
                )  # (N,cond_channels,H,W) in [-1,1]
                assert_finite_torch(cond_tex, "cond_tex_raw", step)
            cond_tex = cond_augment_after_thr(cond_tex, args, allow_jitter=False)
            assert_finite_torch(cond_tex, "cond_tex", step)

            real_phys = seismic_physics_aug(real_pair, p=phys_p, stable_gamma=bool(profile_cfg["stable_gamma"]))

            with torch.cuda.amp.autocast(enabled=amp):
                z = torch.randn(batch_size, args.z_dim, device=device)
                z2 = torch.randn(batch_size, args.z_dim, device=device)
                fake = G(z, cond_tex=cond_tex,
                         mixing_prob=args.style_mixing_prob,
                         z2=z2).detach()
                
                assert_finite_torch(fake, "fake_raw", step)
                fake_phys = seismic_physics_aug(fake, p=phys_p, stable_gamma=bool(profile_cfg["stable_gamma"]))
                assert_finite_torch(fake_phys, "fake_phys", step)

                real_plus = torch.cat([real_phys, cond_tex], dim=1)
                fake_plus = torch.cat([fake_phys, cond_tex], dim=1)

                real_plus_aug = ada_aug_geometric(real_plus, augment_p)
                fake_plus_aug = ada_aug_geometric(fake_plus, augment_p)
                assert_finite_torch(real_plus_aug, "real_plus_aug", step)
                assert_finite_torch(fake_plus_aug, "fake_plus_aug", step)

                real_preds = D(real_plus_aug)
                fake_preds = D(fake_plus_aug)
                for i, rp in enumerate(real_preds if isinstance(real_preds, (list, tuple)) else [real_preds]):
                    assert_finite_torch(rp, f"real_pred_{i}", step)
                for i, fp in enumerate(fake_preds if isinstance(fake_preds, (list, tuple)) else [fake_preds]):
                    assert_finite_torch(fp, f"fake_pred_{i}", step)
                d_loss = 0.0
                for rp, fp in zip(real_preds, fake_preds):
                    d_loss = d_loss + d_logistic_loss(rp, fp)

            # --- D backward ---
            if not torch.isfinite(d_loss):
                logger.error(f"NaN/Inf in d_loss at step={step}, abort to protect weights.")
                raise FloatingPointError("d_loss is NaN/Inf")
            d_opt.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            # grad clip (works with/without AMP)
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                if amp:
                    scaler.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
            scaler.step(d_opt)
            scaler.update()

            # R1
            if step % args.r1_interval == 0:
                with torch.cuda.amp.autocast(enabled=False):
                    real_plus_r1 = torch.cat([real_phys, cond_tex], dim=1)
                    real_plus_r1 = ada_aug_geometric(real_plus_r1, augment_p)
                    real_plus_r1 = real_plus_r1.detach().requires_grad_(True)
                    real_r1_preds = D(real_plus_r1)
                    real_r1_pred = sum([p.mean() for p in real_r1_preds])

                    grad = torch.autograd.grad(
                        outputs=real_r1_pred,
                        inputs=real_plus_r1,
                        create_graph=True
                    )[0]
                    in_ch = real_phys.shape[1]     
                    grad_img = grad[:, :in_ch]
                    r1 = grad_img.pow(2).reshape(grad_img.shape[0], -1).sum(1).mean()
                    r1_term = 0.5 * r1_gamma * r1
                    logger.info(f"R1 | step={step} r1={float(r1.item()):.6f} r1_term={float(r1_term.item()):.6f}")
                    jlog.write({"step": int(step), "r1": float(r1.item()), "r1_term": float(r1_term.item())})

                if not torch.isfinite(r1_term):
                    logger.error(f"NaN/Inf in r1_term at step={step}, abort.")
                    raise FloatingPointError("r1_term is NaN/Inf")
                d_opt.zero_grad(set_to_none=True)
                scaler.scale(r1_term).backward()
                if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                    if amp:
                        scaler.unscale_(d_opt)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                scaler.step(d_opt)
                scaler.update()

            with torch.cuda.amp.autocast(enabled=amp):
                z = torch.randn(batch_size, args.z_dim, device=device)
                z2 = torch.randn(batch_size, args.z_dim, device=device)
                fake = G(z, cond_tex=cond_tex,
                         mixing_prob=args.style_mixing_prob,
                         z2=z2)
                
                fake_phys = seismic_physics_aug(fake, p=phys_p, stable_gamma=bool(profile_cfg["stable_gamma"])) 

                fake_plus_for_g = torch.cat([fake_phys, cond_tex], dim=1)
                fake_aug_for_g = ada_aug_geometric(fake_plus_for_g, augment_p)

                fake_preds_for_g = D(fake_aug_for_g)
                g_adv = 0.0
                for fp in fake_preds_for_g:
                    g_adv = g_adv + g_nonsaturating_loss(fp)

            lambda_spec_t = _ramp(step, lambda_spec, args.spec_warmup_steps, args.spec_ramp_steps)

            lambda_grad_t = _ramp(step, args.lambda_grad, args.grad_warmup_steps, args.grad_ramp_steps) * ctrl.mul_grad
            lambda_dir_t  = _ramp(step, args.lambda_grad_dir, args.grad_dir_warmup_steps, args.grad_dir_ramp_steps) * ctrl.mul_dir
            lambda_ori_t  = _ramp(step, args.lambda_st_ori,  args.st_ori_warmup_steps,  args.st_ori_ramp_steps) * ctrl.mul_ori

            with torch.cuda.amp.autocast(enabled=False):
                if lambda_spec_t > 0:
                    spec_loss = multiscale_fft_l1_loss(
                        fake.float(), real_pair.float(),
                        scales=spec_scales,
                        use_log=spec_use_log
                    )
                else:
                    spec_loss = fake.new_tensor(0.0)

                hq = float(getattr(args, "struct_hard_q", 0.85))
                hmin = float(getattr(args, "struct_hard_min", 0.0))
                
                if lambda_grad_t > 0:
                    grad_loss = multiscale_sobel_grad_l1_masked(
                        fake.float(), real_pair.float(),
                        scales=grad_scales,
                        hard_q=hq, hard_min=hmin
                    )
                else:
                    grad_loss = fake.new_tensor(0.0)
                    
                if lambda_dir_t > 0:
                    dir_loss = grad_dir_loss(
                        fake.float(), real_pair.float(),
                        scales=grad_dir_scales,
                        mag_power=float(getattr(args, "grad_dir_mag_power", 1.0)),
                        mag_thresh=float(getattr(args, "grad_dir_mag_thresh", 0.0)),
                        use_hard_mask=True,
                        hard_q=hq, hard_min=hmin
                    )
                else:
                    dir_loss = fake.new_tensor(0.0)

                if lambda_ori_t > 0:
                    ori_loss = structure_tensor_orientation_loss(
                        fake.float(), real_pair.float(),
                        scales=st_ori_scales,
                        sigmas=st_sigmas,
                        coh_power=float(getattr(args, "st_coh_power", 1.0)),
                        use_hard_mask=True,
                        hard_q=hq, hard_min=hmin
                    )
                else:
                    ori_loss = fake.new_tensor(0.0)

            g_loss = g_adv + lambda_spec_t * spec_loss + lambda_grad_t * grad_loss + lambda_dir_t * dir_loss + lambda_ori_t * ori_loss

            if not torch.isfinite(g_loss):
                logger.error(f"NaN/Inf in g_loss at step={step}, abort.")
                raise FloatingPointError("g_loss is NaN/Inf")
            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                if amp:
                    scaler.unscale_(g_opt)
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            scaler.step(g_opt)
            scaler.update()
            clamp_g_params_(G, args)
            ema_update(G_ema, G, decay=ema_decay)

            # --- ADA p update ---
            if step % ADA_INTERVAL == 0:
                with torch.no_grad():
                    real_plus_for_stat = torch.cat([real_phys, cond_tex], dim=1)
                    real_plus_for_stat = ada_aug_geometric(real_plus_for_stat, augment_p)

                    preds = D(real_plus_for_stat)  # list
                    stat = 0.0
                    for p in preds:
                        stat += (p > 0).float().mean().item()
                        
                    stat /= max(1, len(preds))
                sign_stat_ema = 0.99 * sign_stat_ema + 0.01 * stat
                augment_p += math.copysign(ADA_SPEED, sign_stat_ema - ADA_TARGET)
                augment_p = float(max(0.0, min(augment_p, 0.95)))

            if step % args.log_interval == 0:
                pbar.set_description(
                    f"e{epoch} s{step} "
                    f"D{d_loss.item():.3f} G{g_loss.item():.3f} "
                    f"adv{g_adv.item():.3f} sp{spec_loss.item():.3f} gr{grad_loss.item():.3f} "
                    f"dir{dir_loss.item():.3f} ori{ori_loss.item():.3f} "
                    f"p{augment_p:.3f}"
                )
                dt = time.time() - t_step0
                ips = (batch_size / dt) if dt > 0 else 0.0

                lr_g = float(g_opt.param_groups[0]["lr"])
                lr_d = float(d_opt.param_groups[0]["lr"])
                scale_val = float(scaler.get_scale()) if amp else 1.0

                log_obj = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "d_loss": float(d_loss.item()),
                    "g_loss": float(g_loss.item()),
                    "g_adv": float(g_adv.item()),
                    "spec": float(spec_loss.item()),
                    "grad": float(grad_loss.item()),
                    "dir": float(dir_loss.item()),
                    "ori": float(ori_loss.item()),
                    "lambda_spec_t": float(lambda_spec_t),
                    "lambda_grad_t": float(lambda_grad_t),
                    "lambda_dir_t": float(lambda_dir_t),
                    "lambda_ori_t": float(lambda_ori_t),
                    "augment_p": float(augment_p),
                    "sign_stat_ema": float(sign_stat_ema),
                    "ada_target": float(ADA_TARGET),
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "amp": bool(amp),
                    "grad_scaler": scale_val,
                    "sec_per_step": float(dt),
                    "img_per_sec": float(ips),
                }
                log_obj.update(cuda_mem_mb(device))

                try:
                    log_obj.update(cond_channel_stats(cond_tex, keys))
                except Exception as _:
                    pass

                logger.info(
                    " | ".join([f"{k}={v}" for k, v in log_obj.items()
                                if k in ("epoch","step","d_loss","g_loss","g_adv","spec","grad","dir","ori",
                                         "lambda_spec_t","lambda_grad_t","lambda_dir_t","lambda_ori_t",
                                         "augment_p","sign_stat_ema","lr_g","lr_d","img_per_sec",
                                         "edge_mean","edge_std","edge_density@0.5",
                                         "fault_mean","fault_std","fault_density@0.5")])
                )
                jlog.write(log_obj)


            if step>0 and step % args.save_interval == 0:
                G.eval()
                with torch.no_grad():
                    cond_tex_fixed = cond_tex[:fixed_z.shape[0]]
                    samp = G_ema(fixed_z, cond_tex=cond_tex_fixed, mixing_prob=0.0).cpu()
                    vutils.save_image((samp+1)/2, os.path.join(args.out_dir, f'sample_{step}.png'), nrow=4)
                torch.save({
                    'G': G.state_dict(),
                    "G_ema": G_ema.state_dict(),
                    'D': D.state_dict(),
                    'g_opt': g_opt.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'scaler': scaler.state_dict() if hasattr(scaler, 'state_dict') else None,
                    'step': step,
                    'epoch': epoch,
                    'augment_p': augment_p,
                    'sign_stat_ema': sign_stat_ema
                }, os.path.join(args.out_dir, f'checkpoint_{step}.pth'))
                ckpt_path = os.path.join(args.out_dir, f'checkpoint_{step}.pth')
                png_path  = os.path.join(args.out_dir, f'sample_{step}.png')
                logger.info(
                    f"SAVE | step={step} ckpt={ckpt_path} sample={png_path} "
                    f"augment_p={augment_p:.3f} sign_stat_ema={sign_stat_ema:.3f}"
                )
                jlog.write({"step": int(step), "event": "save", "ckpt": ckpt_path, "sample": png_path,
                            "augment_p": float(augment_p), "sign_stat_ema": float(sign_stat_ema)})

                G.train()

            EVAL_START_STEP = 1500
            
            if (step > EVAL_START_STEP) and (step % args.eval_interval == 0):
                G.eval()
                with torch.no_grad():
                    n_need = int(args.eval_n)
                    bs = min(16, n_need)
                    got = 0
                    
                    acc = {
                        "Structure": {"mmd": [], "w": []},
                        "Texture":   {"mmd": [], "w": []},
                        "Spectrum":  {"mmd": [], "w": []},
                        "Energy":    {"mmd": [], "w": []},
                        "Total":     {"mmd": [], "w": []},
                    }
                    
                    all_real = []
                    all_fake = []

                    all_phi_real = []
                    all_phi_fake = []

                    # raw images for SWD
                    all_real_imgs = []
                    all_fake_imgs = []
                    
                    while got < n_need:
                        cur = min(bs, n_need - got)
                        z_eval = torch.randn(cur, args.z_dim, device=device)
                        
                        idx = torch.randint(0, eval_cond_num, (cur,), device=device)
                        cond_tex_eval = fixed_cond_tex[idx]        
                        real_ref = fixed_cond_imgs[idx]            

                        fake_batch = G_ema(z_eval, cond_tex=cond_tex_eval, mixing_prob=0.0)
                        all_real_imgs.append(real_ref.detach().cpu())
                        all_fake_imgs.append(fake_batch.detach().cpu())

                        real_feat, fake_feat = geo_features_batch(
                            real_ref, fake_batch,
                            sgs_params=sgs_params,
                            agc_params=agc_params,
                            det_params=det_params,
                            edge_type=evaluation_edge_type
                        )
                        
                        all_real.append(real_feat)
                        all_fake.append(fake_feat)

                        phi_r, phi_f = phi_features_batch(
                            real_ref, fake_batch,
                            sgs_params=sgs_params,
                            agc_params=agc_params,
                            det_params=det_params,
                            edge_type=evaluation_edge_type
                        )
                        
                        all_phi_real.append(phi_r)
                        all_phi_fake.append(phi_f)

                        got += cur
                        
                    real_feat_all = np.concatenate(all_real, axis=0)
                    fake_feat_all = np.concatenate(all_fake, axis=0)

                    # --- Multi-scale SWD on raw images (texture statistic) ---
                    real_imgs_all = torch.cat(all_real_imgs, dim=0)[:n_need]
                    fake_imgs_all = torch.cat(all_fake_imgs, dim=0)[:n_need]

                    swd_scores = multiscale_swd(
                        real_imgs_all, fake_imgs_all,
                        scales=swd_scales,
                        patch_size=swd_patch,
                        n_patches=swd_patches,
                        n_projections=swd_projections,
                        seed=int(args.seed + step + 30000),
                    )

                    # print per-scale + total
                    msg = " ".join([f"{k}:{v:.6f}" for k, v in swd_scores.items()])
                    print(f"[eval-swd] step {step} | {msg}")

                    # ===== RR-SWD (Real-Real) thresholds =====
                    rr_swd = rr_swd_thresholds(
                        real_imgs_all,
                        trials=int(getattr(args, "rr_swd_trials", 10)),
                        percentile=float(getattr(args, "rr_swd_percentile", 95.0)),
                        scales=swd_scales,
                        patch_size=swd_patch,
                        n_patches=swd_patches,
                        n_projections=swd_projections,
                        seed=int(args.seed + step + 31000),
                    )
                    if rr_swd and ("Total" in rr_swd):
                        ratio = float(swd_scores.get("Total", np.nan)) / (rr_swd["Total"] + 1e-12)
                        rr_msg = " ".join([f"{k}:{rr_swd[k]:.6f}" for k in rr_swd.keys()])
                        print(f"[eval-rrswd] step {step} | {rr_msg} | GR/RR_Total {ratio:.3f}")

                    # --- PR in geo_features space (recommended: standardize by real stats) ---
                    geo_real = real_feat_all.astype(np.float64)
                    geo_fake = fake_feat_all.astype(np.float64)

                    mu = geo_real.mean(axis=0, keepdims=True)
                    sd = geo_real.std(axis=0, keepdims=True) + 1e-8
                    geo_real_n = (geo_real - mu) / sd
                    geo_fake_n = (geo_fake - mu) / sd

                    k_list = list(prdc_ks) 
                    n_boot = int(prdc_boot)
                    ci_pct = float(prdc_ci)

                    geo_prdc_curve = prdc_curve_with_ci(
                        geo_real_n, geo_fake_n,
                        k_list=k_list,
                        n_boot=n_boot,
                        ci=ci_pct,
                        seed=int(args.seed + step + 10000),
                    )

                    # --- KID (poly-kernel unbiased MMD^2) on geo embedding ---
                    geo_kid_mean, geo_kid_std = kid_poly_mmd2(
                        geo_real_n, geo_fake_n,
                        subset_size=kid_subset_size,
                        n_subsets=kid_subsets,
                        degree=kid_degree,
                        gamma=kid_gamma,
                        coef0=kid_coef0,
                        seed=int(args.seed + step + 11000),
                    )
                    print(f"[eval-geo-kid] step {step} | KID(MMD2) {geo_kid_mean:.6f} ± {geo_kid_std:.6f}")
                    # ===== RR-KID in GEO feature space =====
                    rr_geo_kid = rr_kid_thresholds(
                        geo_real_n,
                        trials=int(getattr(args, "rr_kid_trials", 20)),
                        percentile=float(getattr(args, "rr_kid_percentile", 95.0)),
                        subset_size=int(getattr(args, "kid_subset_size", 100)),
                        n_subsets=int(getattr(args, "kid_subsets", 20)),
                        degree=int(kid_degree),
                        gamma=kid_gamma,
                        coef0=float(kid_coef0),
                        seed=int(args.seed + step + 11100),
                    )
                    if rr_geo_kid:
                        ratio = float(geo_kid_mean) / (rr_geo_kid["thr"] + 1e-12)
                        p = float(getattr(args, "rr_kid_percentile", 95.0))
                        print(f"[eval-geo-rrkid] step {step} | KID_RR@p{p:.1f} {rr_geo_kid['thr']:.6f} | GR/RR {ratio:.3f}")

                    for k_pr in k_list:
                        m = geo_prdc_curve[int(k_pr)]
                        print(
                            f"[eval-geo-prdc] step {step} | k {k_pr} | "
                            f"P {m['precision']['mean']:.3f} [{m['precision']['lo']:.3f},{m['precision']['hi']:.3f}] "
                            f"R {m['recall']['mean']:.3f} [{m['recall']['lo']:.3f},{m['recall']['hi']:.3f}] "
                            f"Dens {m['density']['mean']:.3f} [{m['density']['lo']:.3f},{m['density']['hi']:.3f}] "
                            f"Cov {m['coverage']['mean']:.3f} [{m['coverage']['lo']:.3f},{m['coverage']['hi']:.3f}]"
                        )

                    
                    gammas = estimate_group_gammas(geo_real_n, geo_fake_n, seed=args.seed + step)
                    mean_scores = geo_subscores(geo_real_n, geo_fake_n, mmd_gamma=gammas)

                    # --- PRDC by GEO groups (Structure/Texture/Spectrum/Energy) ---
                    for gname, sl in GEO_FEAT_LAYOUT.items():
                        if gname == "Total":
                            continue
                        Rg = geo_real[:, sl].astype(np.float64)
                        Fg = geo_fake[:, sl].astype(np.float64)

                        mu_g = Rg.mean(axis=0, keepdims=True)
                        sd_g = Rg.std(axis=0, keepdims=True) + 1e-8
                        Rg_n = (Rg - mu_g) / sd_g
                        Fg_n = (Fg - mu_g) / sd_g

                        g_curve = prdc_curve_with_ci(
                            Rg_n, Fg_n,
                            k_list=list(prdc_ks),
                            n_boot=prdc_boot,
                            ci=prdc_ci,
                            seed=int(args.seed + step + 50000 + (hash(gname) & 0xffff)),
                        )
                        k0 = int(list(prdc_ks)[0])
                        m = g_curve[k0]
                        print(
                            f"[eval-geo-prdc-{gname}] step {step} | k {k0} | "
                            f"P {m['precision']['mean']:.3f} "
                            f"R {m['recall']['mean']:.3f} "
                            f"Dens {m['density']['mean']:.3f} "
                            f"Cov {m['coverage']['mean']:.3f}"
                        )

                    print(
                        f"[eval-geo] step {step} | "
                        f"MMD  S {mean_scores['Structure']['mmd']:.6f} "
                        f"T {mean_scores['Texture']['mmd']:.6f} "
                        f"Sp {mean_scores['Spectrum']['mmd']:.6f} "
                        f"E {mean_scores['Energy']['mmd']:.6f} | "
                        f"Total {mean_scores['Total']['mmd']:.6f}"
                    )
                    print(
                        f"[eval-geo] step {step} | "
                        f"WDist S {mean_scores['Structure']['w']:.6f} "
                        f"T {mean_scores['Texture']['w']:.6f} "
                        f"Sp {mean_scores['Spectrum']['w']:.6f} "
                        f"E {mean_scores['Energy']['w']:.6f} | "
                        f"Total {mean_scores['Total']['w']:.6f}"
                    )

                    phi_real_all = np.concatenate(all_phi_real, axis=0)
                    phi_fake_all = np.concatenate(all_phi_fake, axis=0)

                    # Fit z-score statistics only on the real reference set, then
                    # use the same standardized coordinates for FD, RR-FD, KID and PRDC.
                    phi_real_n_full, phi_fake_n_full, _, _, phi_valid = standardize_features_by_real(
                        phi_real_all, phi_fake_all, eps=1e-8
                    )
                    phi_real_n = phi_real_n_full[:, phi_valid]
                    phi_fake_n = phi_fake_n_full[:, phi_valid]

                    # Fréchet(phi): total + groups on standardized features.
                    fd_scores = frechet_phi_scores(
                        phi_real_n_full, phi_fake_n_full, valid_mask=phi_valid
                    )
                    # ===== RR-FD (Real-Real self-consistency) thresholds =====
                    rr_fd = rr_frechet_thresholds(
                        phi_real_n_full,
                        trials=int(getattr(args, "rr_fd_trials", 20)),
                        percentile=float(getattr(args, "rr_fd_percentile", 95.0)),
                        min_per_split=int(getattr(args, "rr_fd_min_n", 64)),
                        seed=int(args.seed + step),
                        valid_mask=phi_valid,
                    )

                    if rr_fd:
                        ratio_total = fd_scores["Total"] / (rr_fd["Total"] + 1e-12)
                        print(f"[eval-rrfd] step {step} | RR-FD@p{getattr(args,'rr_fd_percentile',95.0)} "
                              f"Total_thr {rr_fd['Total']:.6f} | FD/RR {ratio_total:.3f}")

                        for g in ["StructTex", "Spectrum", "Energy"]:
                            if g in rr_fd and g in fd_scores:
                                r = fd_scores[g] / (rr_fd[g] + 1e-12)
                                ok = "OK" if r <= 1.0 else "HIGH"
                                print(f"[eval-rrfd] step {step} | {g} thr {rr_fd[g]:.6f} | FD {fd_scores[g]:.6f} | FD/RR {r:.3f} ({ok})")
                    else:
                        print(f"[eval-rrfd] step {step} | skip (need >= {2*int(getattr(args,'rr_fd_min_n',64))} real samples)")
                        
                    # --- RRFD -> auto reweight + auto ckpt select ---
                    if rr_fd:
                        struct_ratio = float(fd_scores["StructTex"] / (rr_fd["StructTex"] + 1e-12))
                        total_ratio  = float(fd_scores["Total"]     / (rr_fd["Total"]     + 1e-12))

                        upd = ctrl.update(struct_ratio)

                        logger.info(
                            f"RRFD-AUTO | step={step} "
                            f"struct_ratio={struct_ratio:.3f} total_ratio={total_ratio:.3f} "
                            f"r_ema={upd['r_ema']:.3f} "
                            f"mul_grad={upd['mul_grad']:.3f} mul_dir={upd['mul_dir']:.3f} mul_ori={upd['mul_ori']:.3f}"
                        )
                        elog.write({
                            "step": int(step),
                            "rrfd_struct_ratio": struct_ratio,
                            "rrfd_total_ratio": total_ratio,
                            "rrfd_r_ema": upd["r_ema"],
                            "rrfd_mul_grad": upd["mul_grad"],
                            "rrfd_mul_dir": upd["mul_dir"],
                            "rrfd_mul_ori": upd["mul_ori"],
                        })

                        def _save_best(tag: str):
                            ckpt_path = os.path.join(args.out_dir, f"best_{tag}.pth")
                            torch.save({
                                'G': G.state_dict(),
                                'G_ema': G_ema.state_dict(),
                                'D': D.state_dict(),
                                'g_opt': g_opt.state_dict(),
                                'd_opt': d_opt.state_dict(),
                                'scaler': scaler.state_dict() if hasattr(scaler, 'state_dict') else None,
                                'step': step,
                                'epoch': epoch,
                                'augment_p': augment_p,
                                'sign_stat_ema': sign_stat_ema,
                                'rrfd_mul': {"grad": ctrl.mul_grad, "dir": ctrl.mul_dir, "ori": ctrl.mul_ori},
                                'rrfd_ratio': {"structure": struct_ratio, "total": total_ratio},
                            }, ckpt_path)
                            logger.info(f"SAVE_BEST | {tag} | step={step} path={ckpt_path}")

                        if struct_ratio < best_struct_ratio:
                            best_struct_ratio = struct_ratio
                            _save_best("structure")

                        if total_ratio < best_total_ratio:
                            best_total_ratio = total_ratio
                            _save_best("total")


                    # W1/KS by groups
                    dist_scores = group_w1_ks(phi_real_all, phi_fake_all)

                    # --- PRDC/KID in the same standardized phi embedding ---
                    k_list_phi = list(prdc_ks)
                    n_boot = int(prdc_boot)
                    ci_pct = float(prdc_ci)

                    phi_prdc_curve = prdc_curve_with_ci(
                        phi_real_n, phi_fake_n,
                        k_list=k_list_phi,
                        n_boot=n_boot,
                        ci=ci_pct,
                        seed=int(args.seed + step + 20000),
                    )

                    print(f"[eval-phi] step {step} | FD Total {fd_scores['Total']:.6f} "
                          f"S {fd_scores['StructTex']:.6f} "
                          f"Sp {fd_scores['Spectrum']:.6f} E {fd_scores['Energy']:.6f}")

                    # --- KID (poly-kernel unbiased MMD^2) on φ embedding ---
                    phi_kid_mean, phi_kid_std = kid_poly_mmd2(
                        phi_real_n, phi_fake_n,
                        subset_size=kid_subset_size,
                        n_subsets=kid_subsets,
                        degree=kid_degree,
                        gamma=kid_gamma,
                        coef0=kid_coef0,
                        seed=int(args.seed + step + 21000),
                    )
                    print(f"[eval-phi-kid] step {step} | KID(MMD2) {phi_kid_mean:.6f} ± {phi_kid_std:.6f}")
                    
                    for k_pr in k_list_phi:
                        m = phi_prdc_curve[int(k_pr)]
                        print(
                            f"[eval-phi-prdc] step {step} | k {k_pr} | "
                            f"P {m['precision']['mean']:.3f} [{m['precision']['lo']:.3f},{m['precision']['hi']:.3f}] "
                            f"R {m['recall']['mean']:.3f} [{m['recall']['lo']:.3f},{m['recall']['hi']:.3f}] "
                            f"Dens {m['density']['mean']:.3f} [{m['density']['lo']:.3f},{m['density']['hi']:.3f}] "
                            f"Cov {m['coverage']['mean']:.3f} [{m['coverage']['lo']:.3f},{m['coverage']['hi']:.3f}]"
                        )
                        

                    print(f"[eval-phi-dist] step {step} | "
                          f"W1(mean) S {dist_scores['StructTex']['w1_mean']:.6f} "
                          f"Sp {dist_scores['Spectrum']['w1_mean']:.6f} "
                          f"E {dist_scores['Energy']['w1_mean']:.6f} | "
                          f"KS(mean) S {dist_scores['StructTex']['ks_mean']:.6f} "
                          f"Sp {dist_scores['Spectrum']['ks_mean']:.6f} "
                          f"E {dist_scores['Energy']['ks_mean']:.6f}")

                    # ===== RR-W1/KS in PHI space =====
                    rr_dist = rr_w1ks_thresholds(
                        phi_real_all,
                        trials=int(getattr(args, "rr_dist_trials", 20)),
                        percentile=float(getattr(args, "rr_dist_percentile", 95.0)),
                        seed=int(args.seed + step + 12100),
                    )
                    if rr_dist:
                        p = float(getattr(args, "rr_dist_percentile", 95.0))
                        print(
                            f"[eval-rrdist] step {step} | "
                            f"W1_RR@p{p:.1f} S {rr_dist['StructTex']['w1_thr']:.6f} "
                            f"Sp {rr_dist['Spectrum']['w1_thr']:.6f} "
                            f"E {rr_dist['Energy']['w1_thr']:.6f} "
                            f"| KS_RR@p{p:.1f} S {rr_dist['StructTex']['ks_thr']:.6f} "
                            f"Sp {rr_dist['Spectrum']['ks_thr']:.6f} "
                            f"E {rr_dist['Energy']['ks_thr']:.6f}"
                        )

                        rS = dist_scores["StructTex"]["w1_mean"] / (rr_dist["StructTex"]["w1_thr"] + 1e-12)
                        rSp = dist_scores["Spectrum"]["w1_mean"] / (rr_dist["Spectrum"]["w1_thr"] + 1e-12)
                        rE = dist_scores["Energy"]["w1_mean"] / (rr_dist["Energy"]["w1_thr"] + 1e-12)
                        print(f"[eval-rrdist-ratio] step {step} | W1_GR/RR S {rS:.3f} Sp {rSp:.3f} E {rE:.3f}")

                    
                    labels, radar_mmd = scores_to_radar_values(mean_scores, metric="mmd", invert=False)
                    _,      radar_w   = scores_to_radar_values(mean_scores, metric="w",   invert=False)


                    radar_path = os.path.join(args.out_dir, f"eval_geo_radar_step{step}.npz")
                    np.savez(
                        radar_path,
                        labels=np.array(labels),
                        radar_mmd=np.array(radar_mmd, dtype=np.float32),
                        radar_w=np.array(radar_w, dtype=np.float32),
                        mean_scores=mean_scores  
                    )
                    eval_obj = {
                        "step": int(step),
                        "eval_n": int(args.eval_n),
                        "augment_p": float(augment_p),
                        "swd": swd_scores,                  # dict
                        "geo_mmd": {k: mean_scores[k]["mmd"] for k in mean_scores},
                        "geo_w":   {k: mean_scores[k]["w"]   for k in mean_scores},
                        "geo_kid_mean": float(geo_kid_mean),
                        "geo_kid_std":  float(geo_kid_std),
                        "phi_fd": fd_scores,                # standardized phi FD
                        "phi_valid_dimensions": int(np.sum(phi_valid)),
                        "phi_kid_mean": float(phi_kid_mean),
                        "phi_kid_std":  float(phi_kid_std),
                    }

                    if rr_fd:
                        eval_obj["rr_fd_thr"] = rr_fd
                        eval_obj["rr_fd_ratio_total"] = float(fd_scores["Total"] / (rr_fd["Total"] + 1e-12))

                    elog.write(eval_obj)

                    logger.info(
                        f"EVAL | step={step} "
                        f"SWD={swd_scores.get('Total', float('nan')):.6f} "
                        f"GEO_MMD_Tot={mean_scores['Total']['mmd']:.6f} GEO_W_Tot={mean_scores['Total']['w']:.6f} "
                        f"PHI_FD_Tot={fd_scores['Total']:.6f} "
                        f"geoKID={geo_kid_mean:.6f}±{geo_kid_std:.6f} "
                        f"phiKID={phi_kid_mean:.6f}±{phi_kid_std:.6f} "
                        + (f"RRFD_ratio={eval_obj['rr_fd_ratio_total']:.3f}" if rr_fd else "")
                    )

                    z_vis = torch.randn(32, args.z_dim, device=device)
                    idx_vis = torch.randint(0, eval_cond_num, (32,), device=device)
                    fake_vis = G_ema(z_vis, cond_tex=fixed_cond_tex[idx_vis], mixing_prob=0.0).detach().cpu()
                    vutils.save_image((fake_vis + 1) / 2, os.path.join(args.out_dir, f"eval_geo_fake_step{step}.png"), nrow=8)

                G.train()
            step += 1
    print("Training finished.")


# -------------------------
# Test / inference
# -------------------------
def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _build_preproc_params(args, device, amp: bool):
    edge_type = getattr(args, "edge_type", "slog").strip().lower()
    cfg = _profile_config(args)

    def _value(name, default):
        value = getattr(args, name, None)
        return default if value is None else value

    sgs_params = {
        "sigma": float(_value("sgs_sigma", cfg["sgs_sigma"])),
        "anisotropy": float(_value("sgs_anisotropy", cfg["sgs_anisotropy"])),
        "iterations": 1,
        "base_kappa": 10.0,
        "gamma": float(_value("sgs_gamma", cfg["sgs_gamma"])),
        "device": device,
        "prefer_fp16": amp,
        "nl_h": 0.8,
    }
    agc_params = {
        "window": int(_value("agc_window", cfg["agc_window"])),
        "eps": 1e-8,
        "p1": float(_value("agc_p1", cfg["agc_p1"])),
    }
    det_params = {
        "tex_layout": args.tex_layout,
        "glog_sigma": 0.4,
        "glog_anisotropy": 2.1,
        "glog_nangles": 8,
        "glog_sharpness": 14.0,
        "glog_alpha": 0.92,
        "glog_aniso_clip": 4.0,
        "glog_gamma_scale": 1.0,
        "glog_dist_mode": "relative",
        "log_sigma": 1.1,
        "canny_sigma": 1.7,
        "canny_low": 0.1,
        "canny_high": 0.9,
        "leak_down": int(getattr(args, "leak_down", 4)),
        "leak_noise": float(getattr(args, "leak_noise", 0.03)),
        "sgs_leak_down": int(getattr(args, "sgs_leak_down", 0)),
        "sgs_leak_noise": float(getattr(args, "sgs_leak_noise", -1.0)),
        "sgs_leak_blur": float(getattr(args, "sgs_leak_blur", 0.0)),
        "sgs_leak_qbits": int(getattr(args, "sgs_leak_qbits", 0)),
        "fault_leak_down": int(getattr(args, "fault_leak_down", 1)),
        "fault_leak_noise": float(getattr(args, "fault_leak_noise", 0.01)),
        "fault_w_coh": float(_value("fault_w_coh", cfg["fault_w_coh"])),
        "fault_w_ori": float(_value("fault_w_ori", cfg["fault_w_ori"])),
        "fault_w_edge": float(_value("fault_w_edge", cfg["fault_w_edge"])),
        "fault_thr": float(getattr(args, "fault_thr", 0.45)),
        "fault_k": float(getattr(args, "fault_k", 12.0)),
        "fault_seed_thr": float(getattr(args, "fault_seed_thr", 0.65)),
        "fault_band_sigma": float(getattr(args, "fault_band_sigma", 3.5)),
        "fault_smooth_sigma": float(getattr(args, "fault_smooth_sigma", 0.8)),
        "edge_soft_k": 12.0,
        "ori_coh_tau": float(getattr(args, "ori_coh_tau", 0.35)),
        "device": device,
    }

    layout = str(det_params.get("tex_layout", "fault,agc,edge")).strip().lower()
    keys = [k.strip() for k in layout.split(",") if k.strip()]
    cond_channels = len(keys)
    return edge_type, sgs_params, agc_params, det_params, cond_channels

def _load_generator_from_ckpt(args, device, cond_channels, in_ch: int):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = None
    print("[ckpt] using", "G_ema" if ("G_ema" in ckpt and ckpt["G_ema"] is not None) else ("G" if "G" in ckpt else "state_dict"))
    if isinstance(ckpt, dict):
        if "G_ema" in ckpt and ckpt["G_ema"] is not None:
            state = ckpt["G_ema"]
        elif "G" in ckpt:
            state = ckpt["G"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
    if state is None:
        raise ValueError(f"Unrecognized checkpoint format: keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    G = GeneratorWithTexture(
        z_dim=args.z_dim,
        w_dim=args.w_dim,
        base_channels=512,
        img_size=args.img_size,
        out_channels=in_ch,
        cond_channels=cond_channels,
        enc_ch=64,
        cond_mode="film",
    ).to(device)

    missing, unexpected = G.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print("[warn] missing keys in G:", missing[:20], ("..." if len(missing) > 20 else ""))
    if len(unexpected) > 0:
        print("[warn] unexpected keys in G:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))
    strict_generate = bool(
        getattr(args, "strict_generate_ckpt", False)
    )

    if strict_generate and (
        len(missing) > 0
        or len(unexpected) > 0
    ):
        raise RuntimeError(
            "Checkpoint does not exactly match the current "
            "generator architecture.\n"
            f"Missing keys: {missing[:20]}\n"
            f"Unexpected keys: {unexpected[:20]}"
        )
    apply_g_bounds_(G, args)
    G.eval()
    return G

@torch.no_grad()
def prepare_test_condition_pool(
    args,
    device,
    sgs_params,
    agc_params,
    det_params,
    condition_edge_type,
):
    ds = ImageFolderDataset(
        args.data_dir,
        crop_size=args.img_size
    )

    eval_cond_num = int(
        getattr(args, "eval_cond_num", 256)
    )

    base_seed = int(
        getattr(args, "seed", 42)
    )

    rng = np.random.RandomState(base_seed + 123)

    # 保持原 evaluate_once 中生成 condition pool 时完全相同的随机状态
    set_seed_all(base_seed)

    eval_pool_path = str(
        getattr(args, "eval_pool_path", "")
    ).strip()

    if eval_pool_path and os.path.isfile(eval_pool_path):

        pool_obj = torch.load(
            eval_pool_path,
            map_location="cpu"
        )

        if isinstance(pool_obj, dict):
            fixed_cond_imgs = pool_obj["real_crops"]
        else:
            fixed_cond_imgs = pool_obj

        if fixed_cond_imgs.shape[0] != eval_cond_num:
            raise ValueError(
                f"Saved evaluation pool contains "
                f"{fixed_cond_imgs.shape[0]} crops, "
                f"but eval_cond_num={eval_cond_num}."
            )

    else:
        fixed_cond_imgs = []
        selected_indices = []

        for _ in range(eval_cond_num):

            idx = int(
                rng.randint(0, len(ds))
            )

            selected_indices.append(idx)

            fixed_cond_imgs.append(
                ds[idx].unsqueeze(0)
            )

        fixed_cond_imgs = torch.cat(
            fixed_cond_imgs,
            dim=0
        )

        if eval_pool_path:

            os.makedirs(
                os.path.dirname(eval_pool_path) or ".",
                exist_ok=True
            )

            torch.save(
                {
                    "real_crops": fixed_cond_imgs.cpu(),
                    "dataset_indices": selected_indices,
                    "base_seed": base_seed,
                    "eval_cond_num": eval_cond_num,
                },
                eval_pool_path,
            )

    fixed_cond_imgs = fixed_cond_imgs.to(
        device,
        non_blocking=True
    )

    fixed_cond_tex = compute_texture_maps_batch(
        fixed_cond_imgs,
        sgs_params=sgs_params,
        agc_params=agc_params,
        det_params=det_params,
        device=device,
        edge_type=condition_edge_type,
    )

    return fixed_cond_imgs, fixed_cond_tex

@torch.no_grad()
def evaluate_once(
    args,
    device,
    G: nn.Module,
    sgs_params,
    agc_params,
    det_params,
    condition_edge_type: str,
    evaluation_edge_type: str,
    seed: int,
    fixed_cond_imgs: torch.Tensor,
    fixed_cond_tex: torch.Tensor,
    fixed_real_geo: np.ndarray,
    fixed_real_phi: np.ndarray,
):

    set_seed_all(seed)

    eval_cond_num = int(fixed_cond_imgs.shape[0])

    base_seed = int(
        getattr(args, "seed", seed)
    )

    n_need = int(
        getattr(args, "eval_n", 256)
    )

    z_dim = int(
        getattr(args, "z_dim", 512)
    )

    bs_eval = min(int(getattr(args, "batch", 8)), 16)

    all_real_imgs = []
    all_fake_imgs = []
    all_real_geo  = []
    all_fake_geo  = []
    all_phi_real  = []
    all_phi_fake  = []

    got = 0
    while got < n_need:
        cur = min(bs_eval, n_need - got)

        idx = torch.randint(0, eval_cond_num, (cur,), device=device)
        real = fixed_cond_imgs[idx]      # (cur,1,H,W)
        cond_tex = fixed_cond_tex[idx]   # (cur,C,H,W)
        idx_np = idx.detach().cpu().numpy()

        z = torch.randn(cur, z_dim, device=device)
        fake = G(
            z,
            cond_tex=cond_tex,
            mixing_prob=float(getattr(args, "test_mixing_prob", 0.0)),
        ).detach()

        all_real_imgs.append(real.detach().cpu())
        all_fake_imgs.append(fake.detach().cpu())

        # real feature 直接从固定池缓存中索引
        geo_r = fixed_real_geo[idx_np]
        phi_r = fixed_real_phi[idx_np]

        # 只有 fake 需要真正计算
        geo_f = geo_features_only(
            fake,
            sgs_params,
            agc_params,
            det_params,
            edge_type=evaluation_edge_type,
        )

        phi_f = phi_features_only(
            fake,
            sgs_params,
            agc_params,
            det_params,
            edge_type=evaluation_edge_type,
        )

        all_real_geo.append(geo_r)
        all_fake_geo.append(geo_f)

        all_phi_real.append(phi_r)
        all_phi_fake.append(phi_f)

        got += cur

    real_imgs_all = torch.cat(all_real_imgs, dim=0)[:n_need]
    fake_imgs_all = torch.cat(all_fake_imgs, dim=0)[:n_need]
    geo_real = np.concatenate(all_real_geo, axis=0)[:n_need]
    geo_fake = np.concatenate(all_fake_geo, axis=0)[:n_need]
    phi_real = np.concatenate(all_phi_real, axis=0)[:n_need]
    phi_fake = np.concatenate(all_phi_fake, axis=0)[:n_need]

    swd_scales = tuple(int(x) for x in str(getattr(args, "swd_scales", "1,2,4")).split(",") if str(x).strip())
    swd_patch = int(getattr(args, "swd_patch", 7))
    swd_patches = int(getattr(args, "swd_patches", 256))
    swd_projs = int(getattr(args, "swd_projections", 128))
    swd_scores = multiscale_swd(
        real_imgs_all, fake_imgs_all,
        scales=swd_scales, patch_size=swd_patch, n_patches=swd_patches, n_projections=swd_projs,
        seed=seed + 12345,
    )

    mu = geo_real.mean(axis=0, keepdims=True)
    sd = geo_real.std(axis=0, keepdims=True) + 1e-8
    geo_real_n = (geo_real - mu) / sd
    geo_fake_n = (geo_fake - mu) / sd

    gammas = estimate_group_gammas(geo_real_n, geo_fake_n, seed=seed + 22345)
    mean_scores = geo_subscores(geo_real_n, geo_fake_n, mmd_gamma=gammas)

    # KID params (shared)
    kid_subset_size = int(getattr(args, "kid_subset_size", 100))
    kid_subsets     = int(getattr(args, "kid_subsets", 20))
    kid_degree      = int(getattr(args, "kid_degree", 3))
    kid_coef0       = float(getattr(args, "kid_coef0", 1.0))
    kid_gamma       = _resolve_kid_gamma(args)
    geo_kid_mean, geo_kid_std = kid_poly_mmd2(
        geo_real_n, geo_fake_n,
        subset_size=kid_subset_size,
        n_subsets=kid_subsets,
        degree=kid_degree,
        gamma=kid_gamma,
        coef0=kid_coef0,
        seed=seed + 32345,
    )

    prdc_ks_str = str(getattr(args, "prdc_ks", "3,5")).strip()
    prdc_ks_all = [int(x) for x in prdc_ks_str.split(",") if str(x).strip().isdigit()]
    prdc_ks = [k for k in prdc_ks_all if k in (3, 5)]
    if 3 not in prdc_ks:
        prdc_ks.append(3)
    if 5 not in prdc_ks:
        prdc_ks.append(5)
    prdc_ks = sorted(set(prdc_ks))
    prdc_boot = int(getattr(args, "prdc_boot", 200))   
    prdc_ci   = float(getattr(args, "prdc_ci", 95.0))  
    geo_prdc_curve = prdc_curve_with_ci(
        geo_real_n, geo_fake_n,
        k_list=prdc_ks,
        n_boot=prdc_boot,
        ci=prdc_ci,
        seed=seed + 22300,
    )

    # Standardize phi once from real-reference statistics and reuse the same
    # coordinates for FD, RR-FD, KID and PRDC.
    phi_real_n_full, phi_fake_n_full, _, _, phi_valid = standardize_features_by_real(
        phi_real, phi_fake, eps=1e-8
    )
    phi_real_n = phi_real_n_full[:, phi_valid]
    phi_fake_n = phi_fake_n_full[:, phi_valid]

    fd_scores = frechet_phi_scores(
        phi_real_n_full, phi_fake_n_full, valid_mask=phi_valid
    )
    rr_fd = rr_frechet_thresholds(
        phi_real_n_full,
        trials=int(getattr(args, "rr_fd_trials", 20)),
        percentile=float(getattr(args, "rr_fd_percentile", 95.0)),
        min_per_split=int(getattr(args, "rr_fd_min_n", 64)),
        seed=seed + 42345,
        valid_mask=phi_valid,
    )
    rr_ratio_total = float("nan")
    rr_ratio_structtex = float("nan")
    rr_ratio_spectrum = float("nan")
    rr_ratio_energy = float("nan")
    if rr_fd and ("Total" in rr_fd) and ("Total" in fd_scores):
        rr_ratio_total = float(fd_scores["Total"] / (rr_fd["Total"] + 1e-12))
    if rr_fd and ("StructTex" in rr_fd) and ("StructTex" in fd_scores):
        rr_ratio_structtex = float(fd_scores["StructTex"] / (rr_fd["StructTex"] + 1e-12))
    if rr_fd and ("Spectrum" in rr_fd) and ("Spectrum" in fd_scores):
        rr_ratio_spectrum = float(fd_scores["Spectrum"] / (rr_fd["Spectrum"] + 1e-12))
    if rr_fd and ("Energy" in rr_fd) and ("Energy" in fd_scores):
        rr_ratio_energy = float(fd_scores["Energy"] / (rr_fd["Energy"] + 1e-12))

    phi_kid_mean, phi_kid_std = kid_poly_mmd2(
        phi_real_n, phi_fake_n,
        subset_size=kid_subset_size,
        n_subsets=kid_subsets,
        degree=kid_degree,
        gamma=kid_gamma,
        coef0=kid_coef0,
        seed=seed + 52300,
    )

    phi_prdc_curve = prdc_curve_with_ci(
        phi_real_n, phi_fake_n,
        k_list=prdc_ks,
        n_boot=prdc_boot,
        ci=prdc_ci,
        seed=seed + 52400,
    )

    dist_scores = group_w1_ks(phi_real, phi_fake)

    out = {
        "seed": int(seed),
        "n": int(n_need),

        "condition_edge_type": str(condition_edge_type),
        "evaluation_edge_type": str(evaluation_edge_type),
        "evaluation_pool_seed": int(base_seed),
        "evaluation_condition_pool_size": int(eval_cond_num),
        "phi_valid_dimensions": int(np.sum(phi_valid)),

        "swd_total": float(swd_scores.get("Total", float("nan"))),

        "geo_mmd_total": float(mean_scores["Total"]["mmd"]),
        "geo_w_total": float(mean_scores["Total"]["w"]),
        "geo_mmd_structure": float(mean_scores["Structure"]["mmd"]),
        "geo_mmd_texture": float(mean_scores["Texture"]["mmd"]),
        "geo_mmd_spectrum": float(mean_scores["Spectrum"]["mmd"]),
        "geo_mmd_energy": float(mean_scores["Energy"]["mmd"]),

        "geo_kid_mean": float(geo_kid_mean),
        "geo_kid_std": float(geo_kid_std),

        "geo_prdc_k3_p_mean": float(geo_prdc_curve[3]["precision"]["mean"]),
        "geo_prdc_k3_p_lo": float(geo_prdc_curve[3]["precision"]["lo"]),
        "geo_prdc_k3_p_hi": float(geo_prdc_curve[3]["precision"]["hi"]),
        "geo_prdc_k3_r_mean": float(geo_prdc_curve[3]["recall"]["mean"]),
        "geo_prdc_k3_r_lo": float(geo_prdc_curve[3]["recall"]["lo"]),
        "geo_prdc_k3_r_hi": float(geo_prdc_curve[3]["recall"]["hi"]),
        "geo_prdc_k3_d_mean": float(geo_prdc_curve[3]["density"]["mean"]),
        "geo_prdc_k3_d_lo": float(geo_prdc_curve[3]["density"]["lo"]),
        "geo_prdc_k3_d_hi": float(geo_prdc_curve[3]["density"]["hi"]),
        "geo_prdc_k3_c_mean": float(geo_prdc_curve[3]["coverage"]["mean"]),
        "geo_prdc_k3_c_lo": float(geo_prdc_curve[3]["coverage"]["lo"]),
        "geo_prdc_k3_c_hi": float(geo_prdc_curve[3]["coverage"]["hi"]),
        "geo_prdc_k5_p_mean": float(geo_prdc_curve[5]["precision"]["mean"]),
        "geo_prdc_k5_p_lo": float(geo_prdc_curve[5]["precision"]["lo"]),
        "geo_prdc_k5_p_hi": float(geo_prdc_curve[5]["precision"]["hi"]),
        "geo_prdc_k5_r_mean": float(geo_prdc_curve[5]["recall"]["mean"]),
        "geo_prdc_k5_r_lo": float(geo_prdc_curve[5]["recall"]["lo"]),
        "geo_prdc_k5_r_hi": float(geo_prdc_curve[5]["recall"]["hi"]),
        "geo_prdc_k5_d_mean": float(geo_prdc_curve[5]["density"]["mean"]),
        "geo_prdc_k5_d_lo": float(geo_prdc_curve[5]["density"]["lo"]),
        "geo_prdc_k5_d_hi": float(geo_prdc_curve[5]["density"]["hi"]),
        "geo_prdc_k5_c_mean": float(geo_prdc_curve[5]["coverage"]["mean"]),
        "geo_prdc_k5_c_lo": float(geo_prdc_curve[5]["coverage"]["lo"]),
        "geo_prdc_k5_c_hi": float(geo_prdc_curve[5]["coverage"]["hi"]),

        "phi_fd_total": float(fd_scores.get("Total", float("nan"))),
        "phi_fd_structtex": float(fd_scores.get("StructTex", float("nan"))),
        "phi_fd_spectrum": float(fd_scores.get("Spectrum", float("nan"))),
        "phi_fd_energy": float(fd_scores.get("Energy", float("nan"))),

        "phi_kid_mean": float(phi_kid_mean),
        "phi_kid_std": float(phi_kid_std),

        "rr_fd_total_thr": float(rr_fd["Total"]) if rr_fd and ("Total" in rr_fd) else float("nan"),
        "rr_fd_ratio_total": float(rr_ratio_total),
        "rr_fd_ratio_structtex": float(rr_ratio_structtex),
        "rr_fd_ratio_spectrum": float(rr_ratio_spectrum),
        "rr_fd_ratio_energy": float(rr_ratio_energy),

        "phi_prdc_k3_p_mean": float(phi_prdc_curve[3]["precision"]["mean"]),
        "phi_prdc_k3_p_lo": float(phi_prdc_curve[3]["precision"]["lo"]),
        "phi_prdc_k3_p_hi": float(phi_prdc_curve[3]["precision"]["hi"]),
        "phi_prdc_k3_r_mean": float(phi_prdc_curve[3]["recall"]["mean"]),
        "phi_prdc_k3_r_lo": float(phi_prdc_curve[3]["recall"]["lo"]),
        "phi_prdc_k3_r_hi": float(phi_prdc_curve[3]["recall"]["hi"]),
        "phi_prdc_k3_d_mean": float(phi_prdc_curve[3]["density"]["mean"]),
        "phi_prdc_k3_d_lo": float(phi_prdc_curve[3]["density"]["lo"]),
        "phi_prdc_k3_d_hi": float(phi_prdc_curve[3]["density"]["hi"]),
        "phi_prdc_k3_c_mean": float(phi_prdc_curve[3]["coverage"]["mean"]),
        "phi_prdc_k3_c_lo": float(phi_prdc_curve[3]["coverage"]["lo"]),
        "phi_prdc_k3_c_hi": float(phi_prdc_curve[3]["coverage"]["hi"]),
        "phi_prdc_k5_p_mean": float(phi_prdc_curve[5]["precision"]["mean"]),
        "phi_prdc_k5_p_lo": float(phi_prdc_curve[5]["precision"]["lo"]),
        "phi_prdc_k5_p_hi": float(phi_prdc_curve[5]["precision"]["hi"]),
        "phi_prdc_k5_r_mean": float(phi_prdc_curve[5]["recall"]["mean"]),
        "phi_prdc_k5_r_lo": float(phi_prdc_curve[5]["recall"]["lo"]),
        "phi_prdc_k5_r_hi": float(phi_prdc_curve[5]["recall"]["hi"]),
        "phi_prdc_k5_d_mean": float(phi_prdc_curve[5]["density"]["mean"]),
        "phi_prdc_k5_d_lo": float(phi_prdc_curve[5]["density"]["lo"]),
        "phi_prdc_k5_d_hi": float(phi_prdc_curve[5]["density"]["hi"]),
        "phi_prdc_k5_c_mean": float(phi_prdc_curve[5]["coverage"]["mean"]),
        "phi_prdc_k5_c_lo": float(phi_prdc_curve[5]["coverage"]["lo"]),
        "phi_prdc_k5_c_hi": float(phi_prdc_curve[5]["coverage"]["hi"]),
    }


    try:
        out["phi_w1_total"] = float(dist_scores["Total"]["w1"])
        out["phi_ks_total"] = float(dist_scores["Total"]["ks"])
    except Exception:
        out["phi_w1_total"] = float("nan")
        out["phi_ks_total"] = float("nan")

    if bool(getattr(args, "save_test_images", True)):
        os.makedirs(args.out_dir, exist_ok=True)
        vutils.save_image((fake_imgs_all[:32] + 1) / 2, os.path.join(args.out_dir, f"test_fake_seed{seed}.png"), nrow=8)
        vutils.save_image((real_imgs_all[:32] + 1) / 2, os.path.join(args.out_dir, f"test_real_seed{seed}.png"), nrow=8)

        pair_dir = os.path.join(args.out_dir, "single_pairs")
        os.makedirs(pair_dir, exist_ok=True)

        n_save = min(3, int(real_imgs_all.shape[0]))
        for i in range(n_save):
            vutils.save_image(
                (fake_imgs_all[i:i+1] + 1) / 2,
                os.path.join(pair_dir, f"seed{seed}_idx{i:02d}_fake.png"),
                nrow=1
            )

            vutils.save_image(
                (real_imgs_all[i:i+1] + 1) / 2,
                os.path.join(pair_dir, f"seed{seed}_idx{i:02d}_real.png"),
                nrow=1
            )
    return out

def _aggregate_runs(run_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({k for d in run_dicts for k in d.keys() if k not in ("seed",)})
    agg = {"n_runs": len(run_dicts)}
    for k in keys:
        vals = []
        for d in run_dicts:
            v = d.get(k, None)
            if isinstance(v, (int, float)) and (v is not None) and np.isfinite(v):
                vals.append(float(v))
        if len(vals) == 0:
            continue
        if len(vals) == 1:
            agg[k] = {"mean": vals[0], "std": 0.0}
        else:
            agg[k] = {"mean": float(statistics.mean(vals)), "std": float(statistics.pstdev(vals))}
    return agg

def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


@torch.no_grad()
def pairwise_lpips_diversity(
    images: torch.Tensor,
    lpips_model: nn.Module,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    images: [M,1,H,W] in [-1,1]
    Computes LPIPS for all M(M-1)/2 pairs.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected [M,C,H,W], got {images.shape}")

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    pair_indices = [
        (i, j)
        for i in range(images.shape[0])
        for j in range(i + 1, images.shape[0])
    ]

    values = []

    for start in range(0, len(pair_indices), batch_size):
        current_pairs = pair_indices[start:start + batch_size]

        x1 = torch.stack(
            [images[i] for i, _ in current_pairs],
            dim=0,
        )
        x2 = torch.stack(
            [images[j] for _, j in current_pairs],
            dim=0,
        )

        distance = lpips_model(x1, x2)
        values.extend(
            distance.reshape(-1).detach().cpu().numpy().tolist()
        )

    result = _summary_stats(np.asarray(values))
    result["n_pairs"] = int(len(values))
    return result

FAULT_GEOMETRY_NAMES = [
    "soft_density",
    "area_fraction",
    "centroid_x",
    "centroid_y",
    "major_spread",
    "minor_spread",
    "orientation_cos2",
    "orientation_sin2",
    "component_count",
]


def fault_geometry_features_single(
    image01: np.ndarray,
    sgs_params: Dict[str, Any],
    agc_params: Dict[str, Any],
    det_params: Dict[str, Any],
    evaluation_edge_type: str = "log",
    min_component_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a fault-likelihood geometry proxy from one image.

    Parameters
    ----------
    image01:
        Grayscale image with shape [H, W] and values in [0, 1].
    min_component_size:
        Minimum size, in pixels, retained when counting connected components.

    Returns
    -------
    feature:
        Nine geometry-proxy features in ``FAULT_GEOMETRY_NAMES`` order.
    fault_map:
        Continuous fault-likelihood map with shape [H, W].
    """
    pp = preprocess_image_once(
        image01,
        sgs_params=sgs_params,
        agc_params=agc_params,
    )

    edge_map = compute_edge_map_from_preprocessed(
        pp,
        det_params=det_params,
        edge_type=evaluation_edge_type,
    )

    fault_map = compute_fault_map_from_preprocessed(
        pp,
        edge_map=edge_map,
        det_params=det_params,
    )
    fault_map = np.clip(
        np.nan_to_num(fault_map, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float64)

    height, width = fault_map.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx / max(width - 1, 1)
    yy = yy / max(height - 1, 1)

    total_weight = float(fault_map.sum())
    if total_weight <= 1e-12:
        centroid_x = 0.5
        centroid_y = 0.5
        major_spread = 0.0
        minor_spread = 0.0
        orientation_cos2 = 1.0
        orientation_sin2 = 0.0
    else:
        weights = fault_map / total_weight
        centroid_x = float(np.sum(weights * xx))
        centroid_y = float(np.sum(weights * yy))

        dx = xx - centroid_x
        dy = yy - centroid_y
        covariance = np.array(
            [
                [
                    np.sum(weights * dx * dx),
                    np.sum(weights * dx * dy),
                ],
                [
                    np.sum(weights * dx * dy),
                    np.sum(weights * dy * dy),
                ],
            ],
            dtype=np.float64,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]

        major_spread = float(np.sqrt(eigenvalues[0]))
        minor_spread = float(np.sqrt(eigenvalues[1]))

        principal_vector = eigenvectors[:, 0]
        orientation = float(
            np.arctan2(principal_vector[1], principal_vector[0])
        )
        orientation_cos2 = float(np.cos(2.0 * orientation))
        orientation_sin2 = float(np.sin(2.0 * orientation))

    threshold = float(det_params.get("fault_seed_thr", 0.65))
    binary_map = fault_map >= threshold

    connectivity = np.ones((3, 3), dtype=np.uint8)
    labels, _ = ndimage.label(binary_map, structure=connectivity)
    component_sizes = np.bincount(labels.reshape(-1))

    minimum_size = max(1, int(min_component_size))
    keep_labels = np.flatnonzero(component_sizes >= minimum_size)
    keep_labels = keep_labels[keep_labels != 0]
    clean_binary_map = np.isin(labels, keep_labels)

    _, component_count = ndimage.label(
        clean_binary_map,
        structure=connectivity,
    )

    feature = np.array(
        [
            float(fault_map.mean()),
            float(clean_binary_map.mean()),
            centroid_x,
            centroid_y,
            major_spread,
            minor_spread,
            orientation_cos2,
            orientation_sin2,
            float(component_count),
        ],
        dtype=np.float32,
    )

    return feature, fault_map.astype(np.float32)

@torch.no_grad()
def generate_fixed_condition_latents(
    G: nn.Module,
    condition: torch.Tensor,
    latents: torch.Tensor,
    fixed_noise_seed: int,
) -> torch.Tensor:
    """
    condition: [1,C,H,W]
    latents: [M,z_dim]

    Each forward call uses the same synthesis-noise seed.
    Therefore, variation primarily comes from z.
    """
    outputs = []

    for latent_index in range(latents.shape[0]):
        # z has already been generated, so resetting the random seed here
        # fixes only the random sequence used inside the synthesis network.
        set_seed_all(int(fixed_noise_seed))

        fake = G(
            latents[latent_index:latent_index + 1],
            cond_tex=condition,
            mixing_prob=0.0,
        )

        fake = torch.nan_to_num(
            fake,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

        outputs.append(fake.detach().cpu())

    return torch.cat(outputs, dim=0)

@torch.no_grad()
def fixed_condition_diversity(args):
    """Evaluate multi-latent diversity under a fixed condition pool.

    The same condition indices, latent vectors, and synthesis-noise seed are
    reused across conditions and across SLoG/LoG/Canny runs. GEO, phi, and
    fault-likelihood geometry features are standardized using statistics fitted
    only on the complete real condition pool.
    """
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "device", ""):
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    requested_conditions = int(args.diversity_n_conditions)
    number_of_latents = int(args.diversity_n_latents)
    lpips_batch_size = int(args.diversity_lpips_batch)
    standardizer_eps = float(args.diversity_std_eps)
    minimum_component_size = int(args.fault_component_min_size)

    if requested_conditions < 2:
        raise ValueError("diversity_n_conditions must be at least 2.")
    if number_of_latents < 2:
        raise ValueError("diversity_n_latents must be at least 2.")
    if lpips_batch_size < 1:
        raise ValueError("diversity_lpips_batch must be positive.")
    if standardizer_eps <= 0.0:
        raise ValueError("diversity_std_eps must be positive.")
    if minimum_component_size < 1:
        raise ValueError("fault_component_min_size must be at least 1.")

    amp = bool(getattr(args, "amp", False)) and device.type == "cuda"

    (
        condition_edge_type,
        sgs_params,
        agc_params,
        det_params,
        cond_channels,
    ) = _build_preproc_params(args, device, amp)

    evaluation_edge_type = str(
        getattr(args, "eval_edge_type", "log")
    ).strip().lower()
    if evaluation_edge_type not in ("slog", "glog", "log", "canny"):
        raise ValueError(
            "eval_edge_type must be one of slog, glog, log, or canny; "
            f"got {evaluation_edge_type!r}."
        )

    # --------------------------------------------------
    # 1. Load and validate the existing fixed condition pool
    # --------------------------------------------------
    eval_pool_path = Path(str(getattr(args, "eval_pool_path", "")).strip())
    if not str(eval_pool_path):
        raise ValueError("--eval_pool_path is required for diversity analysis.")
    if not eval_pool_path.is_file():
        raise FileNotFoundError(
            f"Fixed evaluation pool does not exist: {eval_pool_path}"
        )

    pool_object = torch.load(eval_pool_path, map_location="cpu")
    if isinstance(pool_object, dict):
        if "real_crops" not in pool_object:
            raise KeyError(
                "The fixed evaluation pool must contain the key 'real_crops'."
            )
        fixed_real_images = pool_object["real_crops"]
        source_indices = pool_object.get(
            "dataset_indices",
            list(range(int(fixed_real_images.shape[0]))),
        )
    elif torch.is_tensor(pool_object):
        fixed_real_images = pool_object
        source_indices = list(range(int(fixed_real_images.shape[0])))
    else:
        raise TypeError(
            "The fixed evaluation pool must be a tensor or a dictionary "
            "containing 'real_crops'."
        )

    if not torch.is_tensor(fixed_real_images):
        fixed_real_images = torch.as_tensor(fixed_real_images)
    fixed_real_images = fixed_real_images.detach().to(dtype=torch.float32)

    if fixed_real_images.ndim != 4:
        raise ValueError(
            "real_crops must have shape [N,C,H,W], got "
            f"{tuple(fixed_real_images.shape)}."
        )
    if fixed_real_images.shape[0] < requested_conditions:
        raise ValueError(
            f"The pool contains {fixed_real_images.shape[0]} conditions, "
            f"but {requested_conditions} were requested."
        )
    if fixed_real_images.shape[2] != int(args.img_size) or fixed_real_images.shape[3] != int(args.img_size):
        raise ValueError(
            "The fixed pool crop size does not match img_size: "
            f"pool={tuple(fixed_real_images.shape[2:])}, img_size={args.img_size}."
        )

    pool_min = float(fixed_real_images.min())
    pool_max = float(fixed_real_images.max())
    if pool_min < -1.01 or pool_max > 1.01:
        raise ValueError(
            "real_crops are expected in [-1,1], got range "
            f"[{pool_min:.6f}, {pool_max:.6f}]."
        )

    source_indices = list(source_indices)
    if len(source_indices) != int(fixed_real_images.shape[0]):
        raise ValueError(
            "dataset_indices length does not match real_crops: "
            f"{len(source_indices)} versus {fixed_real_images.shape[0]}."
        )

    eval_cond_num = int(getattr(args, "eval_cond_num", fixed_real_images.shape[0]))
    if eval_cond_num != int(fixed_real_images.shape[0]):
        raise ValueError(
            f"eval_cond_num={eval_cond_num}, but the fixed pool contains "
            f"{fixed_real_images.shape[0]} crops."
        )

    in_channels = int(fixed_real_images.shape[1])
    fixed_real_images = fixed_real_images.to(device)

    # --------------------------------------------------
    # 2. Load the EMA generator after inferring channels from the pool
    # --------------------------------------------------
    G = _load_generator_from_ckpt(
        args,
        device,
        cond_channels=cond_channels,
        in_ch=in_channels,
    )
    G.eval()

    # --------------------------------------------------
    # 3. Construct each condition exactly once
    # --------------------------------------------------
    set_seed_all(int(args.diversity_seed))
    fixed_conditions = compute_texture_maps_batch(
        fixed_real_images,
        sgs_params=sgs_params,
        agc_params=agc_params,
        det_params=det_params,
        device=device,
        edge_type=condition_edge_type,
    )

    number_of_conditions = min(
        requested_conditions,
        int(fixed_real_images.shape[0]),
    )
    condition_indices = np.linspace(
        0,
        fixed_real_images.shape[0] - 1,
        number_of_conditions,
        dtype=int,
    ).tolist()

    # --------------------------------------------------
    # 4. Create one shared latent set for every condition and model
    # --------------------------------------------------
    latent_generator = torch.Generator(device="cpu")
    latent_generator.manual_seed(int(args.diversity_seed))
    shared_latents = torch.randn(
        number_of_latents,
        int(args.z_dim),
        generator=latent_generator,
        dtype=torch.float32,
    ).to(device)

    torch.save(
        {
            "latents": shared_latents.detach().cpu(),
            "seed": int(args.diversity_seed),
            "n_latents": number_of_latents,
            "z_dim": int(args.z_dim),
        },
        output_dir / "shared_latents.pt",
    )

    # --------------------------------------------------
    # 5. Initialize LPIPS
    # --------------------------------------------------
    if not _have_lpips:
        raise ImportError("LPIPS is required. Install it with: pip install lpips")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    # --------------------------------------------------
    # 6. Fit real-reference standardizers on the complete pool
    # --------------------------------------------------
    real_geo, _ = geo_features_batch(
        fixed_real_images,
        fixed_real_images,
        sgs_params,
        agc_params,
        det_params,
        edge_type=evaluation_edge_type,
    )
    real_phi, _ = phi_features_batch(
        fixed_real_images,
        fixed_real_images,
        sgs_params,
        agc_params,
        det_params,
        edge_type=evaluation_edge_type,
    )

    geo_mean, geo_std, geo_valid = fit_valid_standardizer(
        real_geo,
        eps=standardizer_eps,
    )
    phi_mean, phi_std, phi_valid = fit_valid_standardizer(
        real_phi,
        eps=standardizer_eps,
    )

    real_gray01 = _to_gray01_from_tensor(fixed_real_images)
    real_fault_geometry = []
    for image in real_gray01:
        geometry, _ = fault_geometry_features_single(
            image,
            sgs_params,
            agc_params,
            det_params,
            evaluation_edge_type=evaluation_edge_type,
            min_component_size=minimum_component_size,
        )
        real_fault_geometry.append(geometry)
    real_fault_geometry = np.stack(real_fault_geometry, axis=0)

    (
        fault_geometry_mean,
        fault_geometry_std,
        fault_geometry_valid,
    ) = fit_valid_standardizer(
        real_fault_geometry,
        eps=standardizer_eps,
    )

    def select_valid_group(
        standardized_features: np.ndarray,
        valid_mask: np.ndarray,
        feature_slice: slice,
        group_name: str,
    ) -> np.ndarray:
        group_valid = np.asarray(valid_mask[feature_slice], dtype=bool)
        if not np.any(group_valid):
            raise ValueError(
                f"No valid real-reference dimensions remain for {group_name}."
            )
        return standardized_features[:, feature_slice][:, group_valid]

    # --------------------------------------------------
    # 7. Fixed-condition multi-z evaluation
    # --------------------------------------------------
    result_rows: List[Dict[str, Any]] = []
    geo_structure_sets: List[np.ndarray] = []
    geo_texture_sets: List[np.ndarray] = []
    phi_structtex_sets: List[np.ndarray] = []
    fault_geometry_sets: List[np.ndarray] = []

    for condition_number, pool_index in enumerate(condition_indices):
        condition_dir = output_dir / f"condition_{condition_number:02d}"
        condition_dir.mkdir(parents=True, exist_ok=True)

        condition = fixed_conditions[pool_index:pool_index + 1]
        source_real = fixed_real_images[pool_index:pool_index + 1]

        generated = generate_fixed_condition_latents(
            G=G,
            condition=condition,
            latents=shared_latents,
            fixed_noise_seed=int(args.diversity_noise_seed),
        )

        generated01 = ((generated + 1.0) * 0.5).clamp(0.0, 1.0)
        pixel_variance_map = generated01.var(dim=0, unbiased=True)[0]
        pixel_var_mean = float(pixel_variance_map.mean())
        pixel_var_p95 = float(
            torch.quantile(pixel_variance_map.reshape(-1), 0.95)
        )
        active_variance_fraction = float(
            (pixel_variance_map > 1e-4).float().mean()
        )

        lpips_result = pairwise_lpips_diversity(
            generated.to(device),
            lpips_model=lpips_model,
            batch_size=lpips_batch_size,
        )

        fake_device = generated.to(device)
        _, fake_geo = geo_features_batch(
            fake_device,
            fake_device,
            sgs_params,
            agc_params,
            det_params,
            edge_type=evaluation_edge_type,
        )
        _, fake_phi = phi_features_batch(
            fake_device,
            fake_device,
            sgs_params,
            agc_params,
            det_params,
            edge_type=evaluation_edge_type,
        )

        fake_geo_z = (fake_geo - geo_mean) / geo_std
        fake_phi_z = (fake_phi - phi_mean) / phi_std

        geo_structure = select_valid_group(
            fake_geo_z,
            geo_valid,
            GEO_FEAT_LAYOUT["Structure"],
            "GEO Structure",
        )
        geo_texture = select_valid_group(
            fake_geo_z,
            geo_valid,
            GEO_FEAT_LAYOUT["Texture"],
            "GEO Texture",
        )
        phi_structtex = select_valid_group(
            fake_phi_z,
            phi_valid,
            PHI_LAYOUT["StructTex"],
            "phi StructTex",
        )

        geo_structure_variance = float(
            np.var(geo_structure, axis=0, ddof=1).mean()
        )
        geo_texture_variance = float(
            np.var(geo_texture, axis=0, ddof=1).mean()
        )
        phi_structtex_variance = float(
            np.var(phi_structtex, axis=0, ddof=1).mean()
        )

        generated_gray01 = _to_gray01_from_tensor(generated)
        geometry_features = []
        for image in generated_gray01:
            geometry, _ = fault_geometry_features_single(
                image,
                sgs_params,
                agc_params,
                det_params,
                evaluation_edge_type=evaluation_edge_type,
                min_component_size=minimum_component_size,
            )
            geometry_features.append(geometry)
        geometry_features = np.stack(geometry_features, axis=0)

        geometry_z_full = (
            geometry_features - fault_geometry_mean
        ) / fault_geometry_std
        geometry_z = geometry_z_full[:, fault_geometry_valid]

        fault_geometry_variance = float(
            np.var(geometry_z, axis=0, ddof=1).mean()
        )
        centroid_sd = float(
            np.sqrt(
                np.var(geometry_features[:, 2], ddof=1)
                + np.var(geometry_features[:, 3], ddof=1)
            )
        )
        orientation_resultant = float(
            np.sqrt(
                np.mean(geometry_features[:, 6]) ** 2
                + np.mean(geometry_features[:, 7]) ** 2
            )
        )
        orientation_dispersion = float(
            np.clip(1.0 - orientation_resultant, 0.0, 1.0)
        )

        # Save an auditable visual record for every selected condition.
        vutils.save_image(
            ((source_real.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0),
            condition_dir / "source_real.png",
            nrow=1,
        )
        vutils.save_image(
            ((condition.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0),
            condition_dir / "condition_channels.png",
            nrow=max(1, int(condition.shape[1])),
        )
        vutils.save_image(
            generated01,
            condition_dir / "multi_z_grid.png",
            nrow=10 if number_of_latents >= 10 else number_of_latents,
        )
        vutils.save_image(
            generated01.mean(dim=0, keepdim=True),
            condition_dir / "mean_image.png",
            nrow=1,
        )

        normalized_variance = pixel_variance_map / (
            pixel_variance_map.max() + 1e-12
        )
        vutils.save_image(
            normalized_variance.unsqueeze(0),
            condition_dir / "pixel_variance_map.png",
            nrow=1,
        )
        np.save(
            condition_dir / "pixel_variance.npy",
            pixel_variance_map.numpy(),
        )
        np.save(
            condition_dir / "fault_likelihood_geometry.npy",
            geometry_features,
        )

        source_index_value = source_indices[pool_index]
        if torch.is_tensor(source_index_value):
            source_index_value = source_index_value.item()

        result_rows.append(
            {
                "condition_number": int(condition_number),
                "pool_index": int(pool_index),
                "source_index": int(source_index_value),
                "n_latents": number_of_latents,
                "pixel_variance_mean": pixel_var_mean,
                "pixel_variance_p95": pixel_var_p95,
                "active_variance_fraction": active_variance_fraction,
                "lpips_mean": lpips_result["mean"],
                "lpips_std": lpips_result["std"],
                "lpips_median": lpips_result["median"],
                "lpips_p95": lpips_result["p95"],
                "lpips_n_pairs": lpips_result["n_pairs"],
                "geo_structure_variance_z": geo_structure_variance,
                "geo_texture_variance_z": geo_texture_variance,
                "phi_structtex_variance_z": phi_structtex_variance,
                "fault_likelihood_geometry_variance_z": fault_geometry_variance,
                "fault_likelihood_centroid_sd": centroid_sd,
                "fault_likelihood_orientation_dispersion": orientation_dispersion,
            }
        )

        geo_structure_sets.append(geo_structure)
        geo_texture_sets.append(geo_texture)
        phi_structtex_sets.append(phi_structtex)
        fault_geometry_sets.append(geometry_z)

    # --------------------------------------------------
    # 8. Within-condition / between-condition decomposition
    # --------------------------------------------------
    def within_between_ratio(
        feature_sets: List[np.ndarray],
    ) -> Dict[str, float]:
        if len(feature_sets) < 2:
            raise ValueError("At least two conditions are required.")
        if any(features.shape[0] < 2 for features in feature_sets):
            raise ValueError("At least two latent samples per condition are required.")

        within = float(
            np.mean(
                [
                    np.var(features, axis=0, ddof=1).mean()
                    for features in feature_sets
                ]
            )
        )
        condition_means = np.stack(
            [features.mean(axis=0) for features in feature_sets],
            axis=0,
        )
        between = float(
            np.var(condition_means, axis=0, ddof=1).mean()
        )
        return {
            "within": within,
            "between": between,
            "within_between_ratio": float(within / (between + 1e-12)),
        }

    summary_metric_names = [
        "pixel_variance_mean",
        "pixel_variance_p95",
        "active_variance_fraction",
        "lpips_mean",
        "geo_structure_variance_z",
        "geo_texture_variance_z",
        "phi_structtex_variance_z",
        "fault_likelihood_geometry_variance_z",
        "fault_likelihood_centroid_sd",
        "fault_likelihood_orientation_dispersion",
    ]
    condition_level_summary = {
        metric_name: _summary_stats(
            np.asarray(
                [row[metric_name] for row in result_rows],
                dtype=np.float64,
            )
        )
        for metric_name in summary_metric_names
    }

    aggregate = {
        "checkpoint": str(Path(args.ckpt_path).resolve()),
        "evaluation_pool": str(eval_pool_path.resolve()),
        "pool_size": int(fixed_real_images.shape[0]),
        "n_conditions": number_of_conditions,
        "n_latents_per_condition": number_of_latents,
        "condition_indices": condition_indices,
        "source_indices": [
            int(source_indices[index].item())
            if torch.is_tensor(source_indices[index])
            else int(source_indices[index])
            for index in condition_indices
        ],
        "latent_seed": int(args.diversity_seed),
        "fixed_synthesis_noise_seed": int(args.diversity_noise_seed),
        "condition_edge_type": condition_edge_type,
        "evaluation_edge_type": evaluation_edge_type,
        "tex_layout": str(args.tex_layout),
        "standardizer_eps": standardizer_eps,
        "fault_component_min_size": minimum_component_size,
        "geo_valid_dimensions": int(np.sum(geo_valid)),
        "phi_valid_dimensions": int(np.sum(phi_valid)),
        "fault_geometry_valid_dimensions": int(np.sum(fault_geometry_valid)),
        "fault_geometry_feature_names": FAULT_GEOMETRY_NAMES,
        "condition_level_summary": condition_level_summary,
        "geo_structure": within_between_ratio(geo_structure_sets),
        "geo_texture": within_between_ratio(geo_texture_sets),
        "phi_structtex": within_between_ratio(phi_structtex_sets),
        "fault_likelihood_geometry": within_between_ratio(fault_geometry_sets),
    }

    csv_path = output_dir / "fixed_condition_diversity.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary_csv_path = output_dir / "fixed_condition_diversity_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["metric", "mean", "std", "median", "p95"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name, statistics_dict in condition_level_summary.items():
            writer.writerow({"metric": metric_name, **statistics_dict})

    with (
        output_dir / "fixed_condition_diversity_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2, ensure_ascii=False)

    print(f"Diversity analysis completed: {output_dir}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

def test(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, "test")
    jlog = JsonlWriter(os.path.join(args.out_dir, "metrics_test_runs.jsonl"))

    # device / amp
    if getattr(args, "device", ""):
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and int(getattr(args, "gpus", 1)) > 0 else "cpu")
    amp = (bool(getattr(args, "amp", False)) and device.type == "cuda")

    condition_edge_type, sgs_params, agc_params, det_params, cond_channels = (
        _build_preproc_params(args, device, amp)
    )

    evaluation_edge_type = str(
        getattr(args, "eval_edge_type", "log")
    ).strip().lower()

    # infer in_ch from one sample
    ds_tmp = ImageFolderDataset(args.data_dir, crop_size=args.img_size)
    in_ch = int(ds_tmp[0].shape[0])

    # load ckpt
    G = _load_generator_from_ckpt(args, device, cond_channels=cond_channels, in_ch=in_ch)
    apply_g_bounds_(G, args)
    G.eval()

    print("[test] Preparing fixed condition pool once...")

    fixed_cond_imgs, fixed_cond_tex = prepare_test_condition_pool(
        args=args,
        device=device,
        sgs_params=sgs_params,
        agc_params=agc_params,
        det_params=det_params,
        condition_edge_type=condition_edge_type,
    )
    print("[test] Precomputing real GEO features...")

    fixed_real_geo = geo_features_only(
        fixed_cond_imgs,
        sgs_params,
        agc_params,
        det_params,
        edge_type=evaluation_edge_type,
    )

    print("[test] Precomputing real PHI features...")

    fixed_real_phi = phi_features_only(
        fixed_cond_imgs,
        sgs_params,
        agc_params,
        det_params,
        edge_type=evaluation_edge_type,
    )

    print(
        f"[test] real GEO cache: {fixed_real_geo.shape}, "
        f"real PHI cache: {fixed_real_phi.shape}"
    )

    print(
        f"[test] Fixed condition pool ready: "
        f"{tuple(fixed_cond_imgs.shape)}"
    )
    # seeds
    if getattr(args, "seeds", ""):
        seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    else:
        base = int(getattr(args, "seed", 42))
        seeds = [base + i for i in range(5)]

    repeats = int(getattr(args, "repeats", 1))
    runs = []
    logger.info(f"TEST | ckpt={args.ckpt_path} | data_dir={args.data_dir} | seeds={seeds} | repeats={repeats}")
    for s in seeds:
        for r in range(repeats):
            run_seed = int(s + r * 100000)
            out = evaluate_once(
                args=args,
                device=device,
                G=G,
                sgs_params=sgs_params,
                agc_params=agc_params,
                det_params=det_params,
                condition_edge_type=condition_edge_type,
                evaluation_edge_type=evaluation_edge_type,
                seed=run_seed,
                fixed_cond_imgs=fixed_cond_imgs,
                fixed_cond_tex=fixed_cond_tex,
                fixed_real_geo=fixed_real_geo,
                fixed_real_phi=fixed_real_phi,
            )
            runs.append(out)
            jlog.write(out)
            logger.info(
                f"RUN seed={run_seed} | "
                f"SWD={out.get('swd_total', float('nan')):.6f} "
                f"GEO_MMD={out.get('geo_mmd_total', float('nan')):.6f} "
                f"GEO_KID={out.get('geo_kid_mean', float('nan')):.6f}±{out.get('geo_kid_std', float('nan')):.6f} | "
                f"GEO_PRDC@k3 P={out.get('geo_prdc_k3_p_mean', float('nan')):.3f}[{out.get('geo_prdc_k3_p_lo', float('nan')):.3f},{out.get('geo_prdc_k3_p_hi', float('nan')):.3f}] "
                f"R={out.get('geo_prdc_k3_r_mean', float('nan')):.3f}[{out.get('geo_prdc_k3_r_lo', float('nan')):.3f},{out.get('geo_prdc_k3_r_hi', float('nan')):.3f}] "
                f"D={out.get('geo_prdc_k3_d_mean', float('nan')):.3f}[{out.get('geo_prdc_k3_d_lo', float('nan')):.3f},{out.get('geo_prdc_k3_d_hi', float('nan')):.3f}] "
                f"C={out.get('geo_prdc_k3_c_mean', float('nan')):.3f}[{out.get('geo_prdc_k3_c_lo', float('nan')):.3f},{out.get('geo_prdc_k3_c_hi', float('nan')):.3f}] | "
                f"GEO_PRDC@k5 P={out.get('geo_prdc_k5_p_mean', float('nan')):.3f}[{out.get('geo_prdc_k5_p_lo', float('nan')):.3f},{out.get('geo_prdc_k5_p_hi', float('nan')):.3f}] "
                f"R={out.get('geo_prdc_k5_r_mean', float('nan')):.3f}[{out.get('geo_prdc_k5_r_lo', float('nan')):.3f},{out.get('geo_prdc_k5_r_hi', float('nan')):.3f}] "
                f"D={out.get('geo_prdc_k5_d_mean', float('nan')):.3f}[{out.get('geo_prdc_k5_d_lo', float('nan')):.3f},{out.get('geo_prdc_k5_d_hi', float('nan')):.3f}] "
                f"C={out.get('geo_prdc_k5_c_mean', float('nan')):.3f}[{out.get('geo_prdc_k5_c_lo', float('nan')):.3f},{out.get('geo_prdc_k5_c_hi', float('nan')):.3f}] | "
                f"PHI_FD T={out.get('phi_fd_total', float('nan')):.6f} "
                f"ST={out.get('phi_fd_structtex', float('nan')):.6f} "
                f"SP={out.get('phi_fd_spectrum', float('nan')):.6f} "
                f"EN={out.get('phi_fd_energy', float('nan')):.6f} | "
                f"RRFD_ratio T={out.get('rr_fd_ratio_total', float('nan')):.3f} "
                f"ST={out.get('rr_fd_ratio_structtex', float('nan')):.3f} "
                f"SP={out.get('rr_fd_ratio_spectrum', float('nan')):.3f} "
                f"EN={out.get('rr_fd_ratio_energy', float('nan')):.3f} | "
                f"PHI_KID={out.get('phi_kid_mean', float('nan')):.6f}±{out.get('phi_kid_std', float('nan')):.6f} | "
                f"PHI_PRDC@k3 P={out.get('phi_prdc_k3_p_mean', float('nan')):.3f}[{out.get('phi_prdc_k3_p_lo', float('nan')):.3f},{out.get('phi_prdc_k3_p_hi', float('nan')):.3f}] "
                f"R={out.get('phi_prdc_k3_r_mean', float('nan')):.3f}[{out.get('phi_prdc_k3_r_lo', float('nan')):.3f},{out.get('phi_prdc_k3_r_hi', float('nan')):.3f}] "
                f"D={out.get('phi_prdc_k3_d_mean', float('nan')):.3f}[{out.get('phi_prdc_k3_d_lo', float('nan')):.3f},{out.get('phi_prdc_k3_d_hi', float('nan')):.3f}] "
                f"C={out.get('phi_prdc_k3_c_mean', float('nan')):.3f}[{out.get('phi_prdc_k3_c_lo', float('nan')):.3f},{out.get('phi_prdc_k3_c_hi', float('nan')):.3f}] | "
                f"PHI_PRDC@k5 P={out.get('phi_prdc_k5_p_mean', float('nan')):.3f}[{out.get('phi_prdc_k5_p_lo', float('nan')):.3f},{out.get('phi_prdc_k5_p_hi', float('nan')):.3f}] "
                f"R={out.get('phi_prdc_k5_r_mean', float('nan')):.3f}[{out.get('phi_prdc_k5_r_lo', float('nan')):.3f},{out.get('phi_prdc_k5_r_hi', float('nan')):.3f}] "
                f"D={out.get('phi_prdc_k5_d_mean', float('nan')):.3f}[{out.get('phi_prdc_k5_d_lo', float('nan')):.3f},{out.get('phi_prdc_k5_d_hi', float('nan')):.3f}] "
                f"C={out.get('phi_prdc_k5_c_mean', float('nan')):.3f}[{out.get('phi_prdc_k5_c_lo', float('nan')):.3f},{out.get('phi_prdc_k5_c_hi', float('nan')):.3f}] | "
                f"TEST ckpt={args.ckpt_path} | "
                f"condition_edge={condition_edge_type} | "
                f"evaluation_edge={evaluation_edge_type} | "
                f"data_dir={args.data_dir} | "
                f"seeds={seeds} | repeats={repeats}"
            )

    agg = _aggregate_runs(runs)
    with open(os.path.join(args.out_dir, "metrics_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "summary": agg}, f, ensure_ascii=False, indent=2)

    def _p(k):
        if k in agg:
            return f"{agg[k]['mean']:.6f} ± {agg[k]['std']:.6f}"
        return "n/a"
    logger.info("SUMMARY | "
                f"SWD_total={_p('swd_total')} | "
                f"GEO_MMD_total={_p('geo_mmd_total')} | "
                f"GEO_W_total={_p('geo_w_total')} | "
                f"GEO_KID_mean={_p('geo_kid_mean')} | "
                f"PHI_FD_total={_p('phi_fd_total')} | "
                f"RRFD_ratio_total={_p('rr_fd_ratio_total')}|"
                f"PHI_KID_mean={_p('phi_kid_mean')}")
    print("Test finished.")

@torch.no_grad()
def generate_downstream_dataset(args):
    """
    按输入图像顺序，每个真实条件生成一张合成图，
    并按照原始数字文件名保存。
    """

    # --------------------------------------------------------
    # 1. 设备和随机种子
    # --------------------------------------------------------
    if getattr(args, "device", ""):
        device = torch.device(str(args.device))
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            and int(getattr(args, "gpus", 1)) > 0
            else "cpu"
        )

    amp_enabled = (
        bool(getattr(args, "amp", False))
        and device.type == "cuda"
    )

    generate_seed = int(
        getattr(args, "generate_seed", 43000)
    )

    set_seed_all(generate_seed)

    # --------------------------------------------------------
    # 2. 输出目录
    # --------------------------------------------------------
    output_root = Path(args.out_dir)
    image_output_dir = output_root / "images"
    label_output_dir = output_root / "labels"

    overwrite = bool(
        getattr(args, "overwrite_generate", False)
    )

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}\n"
                "Use --overwrite_generate to replace it."
            )

        shutil.rmtree(output_root)

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    label_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 3. 读取350对图像和标签
    # --------------------------------------------------------
    dataset = PairedGenerationDataset(
        image_dir=args.data_dir,
        label_dir=args.label_dir,
        img_size=args.img_size,
        expected_num=args.generate_num,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.generate_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"[generate] paired samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # 4. 建立条件预处理参数
    # --------------------------------------------------------
    (
        condition_edge_type,
        sgs_params,
        agc_params,
        det_params,
        cond_channels,
    ) = _build_preproc_params(
        args,
        device,
        amp_enabled,
    )

    # 输入固定为单通道灰度
    in_ch = 1

    # --------------------------------------------------------
    # 5. 加载生成器
    # --------------------------------------------------------
    G = _load_generator_from_ckpt(
        args,
        device,
        cond_channels=cond_channels,
        in_ch=in_ch,
    )

    apply_g_bounds_(G, args)
    G.eval()

    # --------------------------------------------------------
    # 6. CSV记录
    # --------------------------------------------------------
    manifest_path = (
        output_root / "generation_manifest.csv"
    )

    manifest_fields = [
        "image_id",
        "source_image",
        "source_label",
        "synthetic_image",
        "copied_label",
        "condition_edge_type",
        "tex_layout",
        "checkpoint",
        "latent_seed",
        "generate_seed",
    ]

    preview_real = []
    preview_fake = []

    generated_count = 0

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=manifest_fields,
        )
        writer.writeheader()

        for batch in tqdm(
            loader,
            desc="Generating downstream dataset",
        ):
            real = batch["image"].to(
                device,
                non_blocking=True,
            )

            image_ids = [
                int(x)
                for x in batch["image_id"]
            ]

            source_image_paths = list(
                batch["image_path"]
            )
            source_label_paths = list(
                batch["label_path"]
            )

            # ----------------------------------------------
            # 条件通道
            # ----------------------------------------------
            cond_tex = compute_texture_maps_batch(
                real,
                sgs_params=sgs_params,
                agc_params=agc_params,
                det_params=det_params,
                device=device,
                edge_type=condition_edge_type,
            )

            # ----------------------------------------------
            # 每个编号使用固定潜变量
            # ----------------------------------------------
            z, latent_seeds = make_latent_batch(
                image_ids=image_ids,
                z_dim=args.z_dim,
                base_seed=generate_seed,
                device=device,
            )

            # ----------------------------------------------
            # 生成图像
            # ----------------------------------------------
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                fake = G(
                    z,
                    cond_tex=cond_tex,
                    mixing_prob=0.0,
                )

            fake = torch.nan_to_num(
                fake,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            fake = fake.clamp(-1.0, 1.0)

            # ----------------------------------------------
            # 按原始文件编号保存
            # ----------------------------------------------
            for local_index, image_id in enumerate(image_ids):
                output_name = f"{image_id}.png"

                synthetic_path = (
                    image_output_dir / output_name
                )
                copied_label_path = (
                    label_output_dir / output_name
                )

                vutils.save_image(
                    (fake[local_index:local_index + 1] + 1.0)
                    / 2.0,
                    synthetic_path,
                    nrow=1,
                    normalize=False,
                )

                # 标签统一保存成PNG，不进行缩放或插值
                with Image.open(
                    source_label_paths[local_index]
                ) as label_image:
                    label_image = label_image.convert("L")
                    label_image.save(
                        copied_label_path,
                        format="PNG",
                    )

                writer.writerow({
                    "image_id": image_id,
                    "source_image":
                        source_image_paths[local_index],
                    "source_label":
                        source_label_paths[local_index],
                    "synthetic_image":
                        str(synthetic_path.resolve()),
                    "copied_label":
                        str(copied_label_path.resolve()),
                    "condition_edge_type":
                        condition_edge_type,
                    "tex_layout":
                        args.tex_layout,
                    "checkpoint":
                        str(Path(args.ckpt_path).resolve()),
                    "latent_seed":
                        latent_seeds[local_index],
                    "generate_seed":
                        generate_seed,
                })

                generated_count += 1

            if len(preview_real) < 16:
                remain = 16 - len(preview_real)
                cur = min(remain, real.shape[0])

                preview_real.extend(
                    real[:cur].detach().cpu()
                )
                preview_fake.extend(
                    fake[:cur].detach().cpu()
                )

    # --------------------------------------------------------
    # 7. 数量检查
    # --------------------------------------------------------
    expected_num = int(args.generate_num)

    if generated_count != expected_num:
        raise RuntimeError(
            f"Expected {expected_num} generated images, "
            f"but saved {generated_count}."
        )

    saved_images = list(
        image_output_dir.glob("*.png")
    )
    saved_labels = list(
        label_output_dir.glob("*.png")
    )

    if len(saved_images) != expected_num:
        raise RuntimeError(
            f"Synthetic image count mismatch: "
            f"{len(saved_images)}"
        )

    if len(saved_labels) != expected_num:
        raise RuntimeError(
            f"Copied label count mismatch: "
            f"{len(saved_labels)}"
        )

    # --------------------------------------------------------
    # 8. 预览图
    # --------------------------------------------------------
    if preview_real:
        preview_real_tensor = torch.stack(
            preview_real,
            dim=0,
        )
        preview_fake_tensor = torch.stack(
            preview_fake,
            dim=0,
        )

        vutils.save_image(
            (preview_real_tensor + 1.0) / 2.0,
            output_root / "preview_real.png",
            nrow=4,
        )

        vutils.save_image(
            (preview_fake_tensor + 1.0) / 2.0,
            output_root / "preview_fake.png",
            nrow=4,
        )

    # --------------------------------------------------------
    # 9. 保存本次配置
    # --------------------------------------------------------
    config = {
        "mode": "generate",
        "generated_count": generated_count,
        "data_dir": str(Path(args.data_dir).resolve()),
        "label_dir": str(Path(args.label_dir).resolve()),
        "out_dir": str(output_root.resolve()),
        "checkpoint": str(Path(args.ckpt_path).resolve()),
        "condition_edge_type": condition_edge_type,
        "tex_layout": args.tex_layout,
        "img_size": int(args.img_size),
        "z_dim": int(args.z_dim),
        "generate_seed": generate_seed,
        "generate_batch": int(args.generate_batch),
        "test_mixing_prob": 0.0,
        "amp": amp_enabled,
    }

    with (
        output_root / "generation_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nGeneration finished.")
    print(f"Algorithm: {condition_edge_type}")
    print(f"Generated images: {generated_count}")
    print(f"Images: {image_output_dir}")
    print(f"Labels: {label_output_dir}")
    print(f"Manifest: {manifest_path}")

# -------------------------
# CLI / main
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="australian", choices=["australian", "f3"],
                        help="survey-specific protocol defaults: australian or f3")
    parser.add_argument("--mode",type=str,required=True,choices=["train", "test", "generate", "diversity"],help="train, test, generate downstream dataset, or fixed-condition diversity analysis",)
    parser.add_argument("--diversity_n_conditions",type=int,default=20,help="number of fixed conditions used for diversity analysis",)
    parser.add_argument("--diversity_n_latents", type=int, default=50, help="number of latent samples generated for each fixed condition")
    parser.add_argument("--diversity_seed", type=int, default=46000, help="base seed used to generate the shared latent set")
    parser.add_argument("--diversity_noise_seed", type=int, default=2026, help="fixed synthesis-noise seed used to isolate latent variation")
    parser.add_argument("--diversity_lpips_batch", type=int, default=8, help="batch size used for pairwise LPIPS calculation")
    parser.add_argument("--diversity_std_eps", type=float, default=1e-6, help="minimum real-reference standard deviation retained for diversity features")
    parser.add_argument("--fault_component_min_size", type=int, default=32, help="minimum connected-component size for the fault-likelihood geometry proxy")

    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path (.pth) for test/generate/diversity')
    parser.add_argument('--seeds', type=str, default='43,44,45,46,47', help='comma-separated seeds for testing, e.g. 42,43,44,45,46')
    parser.add_argument('--repeats', type=int, default=30, help='how many repeats per seed (seed offseted)')
    parser.add_argument('--test_mixing_prob', type=float, default=0.0, help='style mixing prob during test (usually 0)')
    parser.add_argument('--save_test_images', default=True, help='save some real/fake grids per seed')
    parser.add_argument('--device', type=str, default='', help='torch device override, e.g. cuda:0 or cpu; empty selects CUDA automatically')
    parser.add_argument("--label_dir",type=str,default="",help="paired fault-label directory for generation mode",)
    parser.add_argument("--generate_num",type=int,default=350,help="expected number of image-label pairs",)
    parser.add_argument("--generate_batch",type=int,default=8,help="batch size used only in generation mode",)
    parser.add_argument("--generate_seed",type=int,default=43000,help="base seed for deterministic per-image latent codes",)
    parser.add_argument("--overwrite_generate",action="store_true",help="delete and recreate a non-empty generation output directory",)
    parser.add_argument("--strict_generate_ckpt",action="store_true",help="stop if checkpoint keys do not exactly match generator",)
    parser.add_argument('--data_dir', type=str, required=True, help="path to training images")
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--d_lr', type=float, default=5e-5)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr_decay_start', type=int, default=10000)
    parser.add_argument('--d_lr_decay_every', type=int, default=3000)
    parser.add_argument('--d_lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--d_lr_min', type=float, default=1e-6)
    parser.add_argument('--img_size', type=int, default=None, help='override profile image size')
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--w_dim', type=int, default=512)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=None, help='override profile epoch count')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--amp', action='store_true', help='use mixed precision')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_n', type=int, default=256)
    parser.add_argument('--eval_cond_num',type=int,default=256,help='number of fixed real crops in the shared evaluation condition pool',)
    parser.add_argument('--eval_edge_type',type=str,default='log',choices=['slog', 'glog', 'log', 'canny'],help='common edge operator used only for GEO and phi feature extraction',)
    parser.add_argument('--eval_pool_path',type=str,default='',help='path used to save or load the shared real-crop evaluation pool',)
    parser.add_argument('--seed', type=int, default=35)
    parser.add_argument('--train_eval_seed',type=int,default=42,help='fixed seed used only for the model-development evaluation pool')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--style_mixing_prob', type=float, default=0.5,
                        help='probability of applying style-mixing per-sample during training (0..1). 0 disables mixing.')
    parser.add_argument('--edge_type',type=str,default='slog',choices=['slog','glog', 'log', 'canny'],
                        help='edge / texture detector after nlmeans+SGS+AGC. glog applies cv2.normalize to suppress background.')
    parser.add_argument('--tex_layout', type=str, default='fault,agc,edge',
                        help="Comma-separated cond channels: edge,coherence,ori_sin,ori_cos,sgs,agc,hp,gray,fault. ")
    # Optional preprocessing/conditioning overrides.  Leave unset to use the
    # selected experiment profile exactly.
    parser.add_argument('--sgs_sigma', type=float, default=None)
    parser.add_argument('--sgs_anisotropy', type=float, default=None)
    parser.add_argument('--sgs_gamma', type=float, default=None)
    parser.add_argument('--agc_window', type=int, default=None)
    parser.add_argument('--agc_p1', type=float, default=None)
    parser.add_argument('--fault_w_coh', type=float, default=None)
    parser.add_argument('--fault_w_ori', type=float, default=None)
    parser.add_argument('--fault_w_edge', type=float, default=None)


    parser.add_argument('--leak_down', type=int, default=4,help='Downsample factor for leaky channels (sgs/agc/hp). 1 disables.')
    parser.add_argument('--leak_noise', type=float, default=0.03,help='Gaussian noise std added to leaky channels after down/up. 0 disables.')

    parser.add_argument("--cond_scale_min", type=float, default=0.0)
    parser.add_argument("--cond_scale_max", type=float, default=0.5)

    parser.add_argument("--noise_strength_max", type=float, default=1.5)

    parser.add_argument("--film_gamma_max", type=float, default=2.0)
    parser.add_argument("--film_beta_max",  type=float, default=2.0)

    parser.add_argument("--adapter_w_clip", type=float, default=0.0)
    parser.add_argument("--adapter_b_clip", type=float, default=0.0)

    # --- anti-copy cond augmentation (applied AFTER glog thr) ---
    parser.add_argument('--cond_aug_prob', type=float, default=0.25)
    # --- paired geometry augmentation BEFORE cond (on real_raw) ---
    parser.add_argument('--pair_geom_prob', type=float, default=0.20,
                        help='probability of paired geometric aug on real BEFORE computing cond_tex')
    parser.add_argument('--pair_geom_px', type=int, default=1,
                        help='max pixel shift for paired geom aug (translation jitter)')
    parser.add_argument('--pair_geom_hflip', type=float, default=0.50,
                        help='hflip probability inside paired geom aug')

    # --- struct channels (fault/coherence) augment ---
    parser.add_argument('--cond_channel_drop_prob', type=float, default=0.10) 
    parser.add_argument('--cond_struct_pixel_dropout_rate', type=float, default=0.04)
    parser.add_argument('--cond_struct_block_prob', type=float, default=0.15)
    parser.add_argument('--cond_struct_block_size', type=int, default=None)

    # density-aware pixel dropout on edge channel
    parser.add_argument('--edge_drop_base', type=float, default=0.01)   
    parser.add_argument('--edge_drop_boost', type=float, default=0.15)  
    parser.add_argument('--edge_drop_gamma', type=float, default=1.2)   

    # density-aware scaling (if edge is too dense, drop stronger)
    parser.add_argument('--edge_dense_low', type=float, default=0.55)   
    parser.add_argument('--edge_dense_high', type=float, default=0.85)  
    parser.add_argument('--edge_dense_boost', type=float, default=1.8) 

    # block cutout (edge only)
    parser.add_argument('--edge_block_prob', type=float, default=0.3)
    parser.add_argument('--edge_block_size', type=int, default=None)
    parser.add_argument('--edge_block_num', type=int, default=2)

    # random downsample->upsample on edge (edge only)
    parser.add_argument('--edge_ds_prob', type=float, default=0.2)
    parser.add_argument('--edge_ds_min', type=int, default=2)
    parser.add_argument('--edge_ds_max', type=int, default=8)

    # optional small jitter (use carefully)
    parser.add_argument('--cond_jitter_prob', type=float, default=0.10)
    parser.add_argument('--cond_jitter_px', type=int, default=1)

    parser.add_argument('--phys_aug_prob', type=float, default=0.6)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    parser.add_argument('--lambda_spec', type=float, default=0.15,
                    help='weight of multiscale FFT magnitude L1 loss')
    parser.add_argument('--spec_scales', type=str, default="1,2,4,8",
                        help='comma-separated downsample scales for FFT loss, e.g. "1,2,4"')
    parser.add_argument('--spec_log', type=int, default=1,
                        help='1: use log1p(|FFT|) in spec loss, 0: use raw magnitude')

    parser.add_argument('--r1_interval', type=int, default=16)  
    parser.add_argument('--r1_gamma', type=float, default=5.0) 

    parser.add_argument('--ada_target', type=float, default=0.6)
    parser.add_argument('--ada_interval', type=int, default=4)
    parser.add_argument('--ada_speed', type=float, default=0.0002)   
    parser.add_argument('--augment_p_init', type=float, default=0.2)

    # --- RR-FD thresholds (Real-Real self-consistency) ---
    parser.add_argument('--rr_fd_trials', type=int, default=20,
                        help='number of RR splits to estimate RR-FD threshold')
    parser.add_argument('--rr_fd_percentile', type=float, default=95.0,
                        help='percentile for RR-FD threshold (e.g., 90/95/97.5)')
    parser.add_argument('--rr_fd_min_n', type=int, default=64,
                        help='min samples per split for RR-FD; need >= 2*rr_fd_min_n real samples')
    # --- Extra evaluation metrics (KID / PRDC / SWD) ---
    parser.add_argument('--kid_subsets', type=int, default=20,
                        help='number of random subsets for KID estimation (mean±std across subsets)')
    parser.add_argument('--kid_subset_size', type=int, default=100,
                        help='subset size for KID (clipped by eval_n)')
    parser.add_argument('--kid_degree', type=int, default=3,
                        help='polynomial degree for KID kernel')
    parser.add_argument('--kid_coef0', type=float, default=1.0,
                        help='coef0 for polynomial kernel in KID')
    parser.add_argument('--kid_gamma', type=float, default=-1.0,
                        help='gamma for polynomial kernel in KID; <0 means auto (1/dim)')

    parser.add_argument('--prdc_ks', type=str, default="3,5,10,20",
                        help='comma-separated k list for PRDC curve')
    parser.add_argument('--prdc_boot', type=int, default=200,
                        help='bootstrap iterations for PRDC confidence intervals; 0 disables CI')
    parser.add_argument('--prdc_ci', type=float, default=95.0,
                        help='CI level (percent) for PRDC bootstrap intervals, e.g. 95')

    parser.add_argument('--swd_scales', type=str, default="1,2,4",
                        help='comma-separated scales for multi-scale SWD (downsample factors)')
    parser.add_argument('--swd_patch', type=int, default=7,
                        help='patch size for SWD')
    parser.add_argument('--swd_patches', type=int, default=256,
                        help='number of random patches per scale for SWD')
    parser.add_argument('--swd_projections', type=int, default=128,
                        help='number of random projections for SWD')

    # --- spec loss schedule ---
    parser.add_argument('--spec_warmup_steps', type=int, default=800)
    parser.add_argument('--spec_ramp_steps', type=int, default=1200)

    # --- gradient(structure) loss ---
    parser.add_argument('--lambda_grad', type=float, default=0.20)
    parser.add_argument('--grad_scales', type=str, default="1,2,4")
    parser.add_argument('--grad_warmup_steps', type=int, default=0)
    parser.add_argument('--grad_ramp_steps', type=int, default=400)
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip grad-norm for both G and D; 0 disables')
    # --- direction / orientation losses (optional) ---
    parser.add_argument('--lambda_grad_dir', type=float, default=0.03,
                        help='weight of gradient direction consistency loss (polarity-invariant)')
    parser.add_argument('--grad_dir_scales', type=str, default="",
                        help='comma-separated scales for grad_dir_loss; empty -> use grad_scales')
    parser.add_argument('--grad_dir_warmup_steps', type=int, default=0)
    parser.add_argument('--grad_dir_ramp_steps', type=int, default=400)
    parser.add_argument('--grad_dir_mag_power', type=float, default=1.0,
                        help='weight exponent for real grad magnitude in grad_dir_loss (detach)')
    parser.add_argument('--grad_dir_mag_thresh', type=float, default=0.0,
                        help='optional threshold on real grad magnitude (after Sobel)')

    parser.add_argument('--lambda_st_ori', type=float, default=0.02,
                        help='weight of structure-tensor orientation loss (2theta, polarity-invariant)')
    parser.add_argument('--st_ori_scales', type=str, default="",
                        help='comma-separated scales for structure_tensor_orientation_loss; empty -> use grad_scales')
    parser.add_argument('--st_sigmas', type=str, default="0.8",
                        help='comma-separated Gaussian sigmas for structure tensor smoothing, e.g. "0.8,1.2"')
    parser.add_argument('--st_ori_warmup_steps', type=int, default=100)
    parser.add_argument('--st_ori_ramp_steps', type=int, default=600)
    parser.add_argument('--st_coh_power', type=float, default=1.5,
                        help='weight exponent for real coherence in structure tensor orientation loss (detach)')

    # --- SGS-only stronger degradation (keep other channels unchanged) ---
    parser.add_argument('--sgs_leak_down', type=int, default=0,
                        help='SGS-only override for leak_down. <=0 uses default max(leak_down, leak_down*2).')
    parser.add_argument('--sgs_leak_noise', type=float, default=-1.0,
                        help='SGS-only override for leak_noise. <0 uses default leak_noise*2.')
    parser.add_argument('--sgs_leak_blur', type=float, default=0.0,
                        help='Extra blur sigma applied only to SGS after down/up. 0 disables.')
    parser.add_argument('--sgs_leak_qbits', type=int, default=0,
                        help='Uniform quantization bits applied only to SGS. 0 disables.')
    # --- RRFD auto reweight ---
    parser.add_argument('--rrfd_mul_min', type=float, default=0.3)
    parser.add_argument('--rrfd_mul_max', type=float, default=6.0)
    parser.add_argument('--rrfd_up', type=float, default=0.25)
    parser.add_argument('--rrfd_down', type=float, default=0.08)
    parser.add_argument('--rrfd_ema', type=float, default=0.8)

    # --- hard mask (high-gradient only) ---
    parser.add_argument('--struct_hard_q', type=float, default=0.85,
                        help='keep pixels >= q-quantile of |grad(real)| per-sample')
    parser.add_argument('--struct_hard_min', type=float, default=0.0,
                        help='absolute minimum threshold for |grad(real)| hard mask')

    # --- optional: structure-first phase ---
    parser.add_argument('--struct_focus_steps', type=int, default=0,
                        help='if >0, first N steps boost structure losses and downscale spec')
    parser.add_argument('--struct_focus_boost', type=float, default=1.5)
    parser.add_argument('--struct_focus_spec_scale', type=float, default=0.3)

    
    args = parser.parse_args()
    args = _apply_profile_cli_defaults(args)

    cfg = _profile_config(args)
    print(
        f"[profile] {args.profile} | img_size={args.img_size} | epochs={args.epochs} | "
        f"edge_type={args.edge_type} | eval_edge_type={args.eval_edge_type}"
    )
    if args.profile == "f3":
        print(
            "[profile] F3 edge-independent defaults: "
            f"fault_w_coh={cfg['fault_w_coh']}, "
            f"fault_w_ori={cfg['fault_w_ori']}, fault_w_edge={cfg['fault_w_edge']}"
        )

    if args.mode == "train":
        os.makedirs(args.out_dir, exist_ok=True)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        train(args)

    elif args.mode == "test":
        if not args.ckpt_path:
            raise ValueError(
                "--ckpt_path is required for --mode test"
            )

        os.makedirs(args.out_dir, exist_ok=True)
        test(args)

    elif args.mode == "generate":
        if not args.ckpt_path:
            raise ValueError(
                "--ckpt_path is required for --mode generate"
            )

        if not args.label_dir:
            raise ValueError(
                "--label_dir is required for --mode generate"
            )

        generate_downstream_dataset(args)
        
    elif args.mode == "diversity":
        if not args.ckpt_path:
            raise ValueError(
                "--ckpt_path is required for --mode diversity"
            )

        os.makedirs(args.out_dir, exist_ok=True)
        fixed_condition_diversity(args)

    else:
        raise ValueError(
            f"Unknown mode: {args.mode}"
        )
