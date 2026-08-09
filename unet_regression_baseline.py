"""
Deterministic U-Net regression baselines for source-associated seismic synthesis.

Purpose
-------
This script evaluates how much of the target seismic profile can be recovered
directly from the source-derived conditional tensor without a latent code,
adversarial discriminator, or StyleGAN2 synthesis blocks.

The experiments reported in the paper use the same three-channel
fault + AGC + SLoG condition as the corresponding StyleGAN2-SLoG model.

Two deterministic baselines are evaluated:
  * U-Net-L1: reconstruction using the L1 loss only;
  * U-Net-Struct: L1 reconstruction supplemented by spectral, gradient,
    gradient-direction, and structure-tensor orientation losses.

The script dynamically imports run_conditional_stylegan_experiment.py to reuse
the same data cropping, condition construction, channel degradation,
structural losses, and GEO/phi/SWD distributional evaluation code.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Reproducibility and logging
# -----------------------------------------------------------------------------

def set_seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(out_dir: str, name: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(f"{name}_{os.path.abspath(out_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(out_dir, f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(obj)
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_module(source_code: str):
    source_path = Path(source_code).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Original model code was not found: {source_path}")

    spec = importlib.util.spec_from_file_location("seismic_stylegan_source", str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source code: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "ImageFolderDataset",
        "compute_texture_maps_batch",
        "paired_geom_aug_real",
        "cond_augment_after_thr",
        "multiscale_fft_l1_loss",
        "multiscale_sobel_grad_l1_masked",
        "grad_dir_loss",
        "structure_tensor_orientation_loss",
        "_build_preproc_params",
        "evaluate_once",
        "_aggregate_runs",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "The supplied source code is incompatible with this baseline. "
            f"Missing functions/classes: {missing}"
        )
    return module, source_path


# -----------------------------------------------------------------------------
# Deterministic U-Net regression model
# -----------------------------------------------------------------------------

class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        norm_groups = min(groups, out_ch)
        while out_ch % norm_groups != 0 and norm_groups > 1:
            norm_groups -= 1
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_ch),
            nn.SiLU(inplace=True),
        ])
        self.block = nn.Sequential(*layers)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = ConvNormAct(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv = ConvNormAct(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetRegression(nn.Module):
    """Deterministic condition-to-seismic U-Net with a tanh output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        b = int(base_channels)
        self.stem = ConvNormAct(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.down4 = DownBlock(b * 8, b * 16, dropout=dropout)

        self.bottleneck = ConvNormAct(b * 16, b * 16, dropout=dropout)

        self.up4 = UpBlock(b * 16, b * 8, b * 8, dropout=dropout)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.out = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, out_channels, 1),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(cond)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.bottleneck(x)
        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        return self.out(x)


class UNetEvaluationAdapter(nn.Module):
    """
    Adapts the deterministic U-Net to the original evaluate_once() generator API.
    The latent input is intentionally ignored.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        cond_tex: Optional[torch.Tensor] = None,
        mixing_prob: float = 0.0,
        z2: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        if cond_tex is None:
            raise ValueError("cond_tex is required for the U-Net regression baseline")
        return self.model(cond_tex)


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def load_pool_tensor(path: str) -> Tuple[torch.Tensor, Optional[List[int]]]:
    if not path:
        raise ValueError("A pool path is required")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        real = obj.get("real_crops", None)
        indices = obj.get("dataset_indices", None)
    else:
        real = obj
        indices = None
    if real is None or not torch.is_tensor(real):
        raise ValueError(f"Pool file does not contain a real_crops tensor: {path}")
    if real.dim() != 4:
        raise ValueError(f"Expected NCHW pool tensor, got {tuple(real.shape)}")
    return real.float(), indices


class CroppedImageDirectory(Dataset):
    """Reads already-cropped images while preserving numeric filenames."""

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root: str, img_size: int):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
        paths = [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in self.EXTENSIONS]
        if not paths:
            raise RuntimeError(f"No images found in {root}")
        self.paths = sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        image = Image.open(path).convert("L")
        if image.size != (self.img_size, self.img_size):
            raise ValueError(
                f"{path.name}: expected {self.img_size}x{self.img_size}, got {image.size}"
            )
        tensor = T.ToTensor()(image)
        tensor = T.Normalize([0.5], [0.5])(tensor)
        return {"image": tensor, "name": path.name, "stem": path.stem}


# -----------------------------------------------------------------------------
# Loss and paired metrics
# -----------------------------------------------------------------------------

def ramp_weight(step: int, final_value: float, warmup: int, ramp: int) -> float:
    if final_value <= 0 or step < warmup:
        return 0.0
    if ramp <= 0:
        return float(final_value)
    t = min(1.0, max(0.0, (step - warmup) / float(ramp)))
    return float(final_value) * t


def parse_int_tuple(value: str, fallback: Tuple[int, ...]) -> Tuple[int, ...]:
    items = tuple(int(x.strip()) for x in str(value).split(",") if x.strip())
    return items or fallback


def parse_float_tuple(value: str, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
    items = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    return items or fallback


def regression_losses(
    src,
    pred: torch.Tensor,
    target: torch.Tensor,
    args,
    step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    spec_scales = parse_int_tuple(args.spec_scales, (1, 2, 4, 8))
    grad_scales = parse_int_tuple(args.grad_scales, (1, 2, 4))
    dir_scales = parse_int_tuple(args.grad_dir_scales, grad_scales)
    ori_scales = parse_int_tuple(args.st_ori_scales, grad_scales)
    st_sigmas = parse_float_tuple(args.st_sigmas, (0.8,))

    w_l1 = float(args.lambda_l1)
    w_spec = ramp_weight(step, args.lambda_spec, args.spec_warmup_steps, args.spec_ramp_steps)
    w_grad = ramp_weight(step, args.lambda_grad, args.grad_warmup_steps, args.grad_ramp_steps)
    w_dir = ramp_weight(step, args.lambda_grad_dir, args.grad_dir_warmup_steps, args.grad_dir_ramp_steps)
    w_ori = ramp_weight(step, args.lambda_st_ori, args.st_ori_warmup_steps, args.st_ori_ramp_steps)

    with torch.cuda.amp.autocast(enabled=False):
        pred32 = pred.float()
        target32 = target.float()
        l1 = F.l1_loss(pred32, target32)
        spec = (
            src.multiscale_fft_l1_loss(
                pred32, target32, scales=spec_scales, use_log=bool(args.spec_log)
            )
            if w_spec > 0
            else pred32.new_tensor(0.0)
        )
        grad = (
            src.multiscale_sobel_grad_l1_masked(
                pred32,
                target32,
                scales=grad_scales,
                hard_q=float(args.struct_hard_q),
                hard_min=float(args.struct_hard_min),
            )
            if w_grad > 0
            else pred32.new_tensor(0.0)
        )
        direction = (
            src.grad_dir_loss(
                pred32,
                target32,
                scales=dir_scales,
                mag_power=float(args.grad_dir_mag_power),
                mag_thresh=float(args.grad_dir_mag_thresh),
                use_hard_mask=True,
                hard_q=float(args.struct_hard_q),
                hard_min=float(args.struct_hard_min),
            )
            if w_dir > 0
            else pred32.new_tensor(0.0)
        )
        orientation = (
            src.structure_tensor_orientation_loss(
                pred32,
                target32,
                scales=ori_scales,
                sigmas=st_sigmas,
                coh_power=float(args.st_coh_power),
                use_hard_mask=True,
                hard_q=float(args.struct_hard_q),
                hard_min=float(args.struct_hard_min),
            )
            if w_ori > 0
            else pred32.new_tensor(0.0)
        )

        total = w_l1 * l1 + w_spec * spec + w_grad * grad + w_dir * direction + w_ori * orientation

    values = {
        "loss_total": float(total.detach().item()),
        "loss_l1": float(l1.detach().item()),
        "loss_spec": float(spec.detach().item()),
        "loss_grad": float(grad.detach().item()),
        "loss_dir": float(direction.detach().item()),
        "loss_ori": float(orientation.detach().item()),
        "weight_l1": w_l1,
        "weight_spec": w_spec,
        "weight_grad": w_grad,
        "weight_dir": w_dir,
        "weight_ori": w_ori,
    }
    return total, values


def tensor_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Input is in [-1,1]; convert to [0,1].
    p = ((pred.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    t = ((target.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    mse = F.mse_loss(p, t).item()
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


@torch.no_grad()
def evaluate_paired_pool(
    src,
    model: nn.Module,
    real_pool: torch.Tensor,
    args,
    device: torch.device,
    sgs_params,
    agc_params,
    det_params,
    condition_edge_type: str,
    save_dir: Optional[str] = None,
    prefix: str = "paired",
) -> Dict[str, float]:
    model.eval()
    set_seed_all(int(args.eval_condition_seed))

    real_pool = real_pool.float()
    batch_size = max(1, int(args.eval_batch))
    totals: Dict[str, List[float]] = {
        "l1": [], "mse_01": [], "psnr": [], "spec": [], "grad": [], "dir": [], "ori": []
    }
    saved_real: List[torch.Tensor] = []
    saved_pred: List[torch.Tensor] = []

    for start in tqdm(range(0, len(real_pool), batch_size), desc=f"{prefix} evaluation", leave=False):
        real = real_pool[start:start + batch_size].to(device, non_blocking=True)
        cond = src.compute_texture_maps_batch(
            real,
            sgs_params=sgs_params,
            agc_params=agc_params,
            det_params=det_params,
            device=device,
            edge_type=condition_edge_type,
        )
        pred = model(cond)

        pred01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
        real01 = ((real + 1.0) * 0.5).clamp(0.0, 1.0)
        totals["l1"].append(float(F.l1_loss(pred, real).item()))
        totals["mse_01"].append(float(F.mse_loss(pred01, real01).item()))
        totals["psnr"].append(tensor_psnr(pred, real))

        with torch.cuda.amp.autocast(enabled=False):
            totals["spec"].append(float(src.multiscale_fft_l1_loss(
                pred.float(), real.float(), scales=parse_int_tuple(args.spec_scales, (1,2,4,8)),
                use_log=bool(args.spec_log)
            ).item()))
            totals["grad"].append(float(src.multiscale_sobel_grad_l1_masked(
                pred.float(), real.float(), scales=parse_int_tuple(args.grad_scales, (1,2,4)),
                hard_q=float(args.struct_hard_q), hard_min=float(args.struct_hard_min)
            ).item()))
            totals["dir"].append(float(src.grad_dir_loss(
                pred.float(), real.float(), scales=parse_int_tuple(args.grad_dir_scales, parse_int_tuple(args.grad_scales,(1,2,4))),
                mag_power=float(args.grad_dir_mag_power), mag_thresh=float(args.grad_dir_mag_thresh),
                use_hard_mask=True, hard_q=float(args.struct_hard_q), hard_min=float(args.struct_hard_min)
            ).item()))
            totals["ori"].append(float(src.structure_tensor_orientation_loss(
                pred.float(), real.float(), scales=parse_int_tuple(args.st_ori_scales, parse_int_tuple(args.grad_scales,(1,2,4))),
                sigmas=parse_float_tuple(args.st_sigmas, (0.8,)), coh_power=float(args.st_coh_power),
                use_hard_mask=True, hard_q=float(args.struct_hard_q), hard_min=float(args.struct_hard_min)
            ).item()))

        if len(saved_real) * batch_size < int(args.save_pair_num):
            remaining = int(args.save_pair_num) - sum(x.shape[0] for x in saved_real)
            if remaining > 0:
                saved_real.append(real[:remaining].cpu())
                saved_pred.append(pred[:remaining].cpu())

    result = {f"paired_{k}": float(np.mean(v)) for k, v in totals.items() if v}
    result["paired_n"] = int(len(real_pool))

    if save_dir and saved_real:
        os.makedirs(save_dir, exist_ok=True)
        real = torch.cat(saved_real, dim=0)[: int(args.save_pair_num)]
        pred = torch.cat(saved_pred, dim=0)[: int(args.save_pair_num)]
        residual = (pred - real).abs() * 0.5  # [0,1]-scale absolute difference
        grid = torch.cat([
            (real + 1.0) * 0.5,
            (pred + 1.0) * 0.5,
            residual.clamp(0.0, 1.0),
        ], dim=0)
        vutils.save_image(
            grid,
            os.path.join(save_dir, f"{prefix}_real_pred_absdiff.png"),
            nrow=real.shape[0],
        )
    return result


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    epoch: int,
    args,
    source_path: Path,
    best_score: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "args": vars(args),
        "source_code": str(source_path),
        "model_name": "UNetRegression",
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    prefer_ema: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
    if prefer_ema and "model_ema" in ckpt:
        state = ckpt["model_ema"]
        state_name = "model_ema"
    elif "model" in ckpt:
        state = ckpt["model"]
        state_name = "model"
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
        state_name = "state_dict"
    else:
        raise ValueError(f"No model weights found in checkpoint. Keys: {list(ckpt.keys())}")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[checkpoint] missing={missing[:10]} unexpected={unexpected[:10]}")
    print(f"[checkpoint] loaded {state_name} from {path}")
    model.to(device).eval()
    return ckpt


# -----------------------------------------------------------------------------
# Train / test / generate
# -----------------------------------------------------------------------------

def train_baseline(args) -> None:
    src, source_path = load_source_module(args.source_code)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, "unet_train")
    metrics_path = os.path.join(args.out_dir, "metrics_train.jsonl")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp and device.type == "cuda")
    set_seed_all(args.seed, deterministic=not args.allow_nondeterministic)

    condition_edge_type, sgs_params, agc_params, det_params, cond_channels = src._build_preproc_params(
        args, device, amp
    )
    ds = src.ImageFolderDataset(
        args.data_dir,
        crop_size=args.img_size,
        start_max_right=args.start_max_right,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        generator=loader_generator,
        persistent_workers=(args.num_workers > 0),
    )

    sample = ds[0]
    out_channels = int(sample.shape[0])
    model = UNetRegression(
        in_channels=cond_channels,
        out_channels=out_channels,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    model_ema = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_step = 0
    start_epoch = 0
    best_score = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model_ema.load_state_dict(ckpt.get("model_ema", ckpt["model"]), strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0)) + 1
        start_epoch = int(ckpt.get("epoch", 0))
        best_score = float(ckpt.get("best_score", best_score))
        logger.info(f"Resumed from {args.resume} at step={start_step}")

    val_pool = None
    if args.val_pool_path:
        val_pool, _ = load_pool_tensor(args.val_pool_path)
        logger.info(
            f"Validation pool: {args.val_pool_path} | shape={tuple(val_pool.shape)}. "
            "This pool must not be the final test pool."
        )

    logger.info(
        f"device={device} amp={amp} data={args.data_dir} n_files={len(ds)} "
        f"img={args.img_size} batch={args.batch} max_steps={args.max_steps} "
        f"edge={condition_edge_type} layout={args.tex_layout} cond_channels={cond_channels} "
        f"parameters={count_parameters(model):,} source={source_path}"
    )

    running_score = None
    step = start_step
    epoch = start_epoch
    stop = False
    while not stop:
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for real in pbar:
            if step >= args.max_steps:
                stop = True
                break
            t0 = time.time()
            model.train()
            real = real.to(device, non_blocking=True)

            real_pair = src.paired_geom_aug_real(
                real,
                p=float(args.pair_geom_prob),
                max_px=int(args.pair_geom_px),
                hflip_prob=float(args.pair_geom_hflip),
                fill=-1.0,
            )

            with torch.no_grad():
                cond = src.compute_texture_maps_batch(
                    real_pair,
                    sgs_params=sgs_params,
                    agc_params=agc_params,
                    det_params=det_params,
                    device=device,
                    edge_type=condition_edge_type,
                )
                if not args.no_cond_aug:
                    cond = src.cond_augment_after_thr(cond, args, allow_jitter=False)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(cond)
            total, loss_values = regression_losses(src, pred, real_pair, args, step)

            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite U-Net regression loss at step {step}")
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema_update(model_ema, model, decay=args.ema_decay)

            # Exponential moving score only for fallback checkpoint selection.
            score = loss_values["loss_total"]
            running_score = score if running_score is None else 0.98 * running_score + 0.02 * score

            if step % args.log_interval == 0:
                dt = time.time() - t0
                row = {
                    "epoch": epoch,
                    "step": step,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "img_per_sec": float(real.shape[0] / max(dt, 1e-8)),
                    "running_score": float(running_score),
                    **loss_values,
                }
                append_jsonl(metrics_path, row)
                pbar.set_postfix(
                    step=step,
                    total=f"{loss_values['loss_total']:.4f}",
                    l1=f"{loss_values['loss_l1']:.4f}",
                    spec=f"{loss_values['loss_spec']:.4f}",
                )
                logger.info(" | ".join(f"{k}={v}" for k, v in row.items()))

            if step > 0 and step % args.preview_interval == 0:
                model_ema.eval()
                with torch.no_grad():
                    n = min(args.preview_num, real_pair.shape[0])
                    pred_preview = model_ema(cond[:n]).cpu()
                    real_preview = real_pair[:n].cpu()
                    cond_preview = cond[:n].cpu()
                    # First three rows: condition channels; then target, prediction, residual.
                    panels = []
                    for ch in range(min(cond_preview.shape[1], 3)):
                        panels.append((cond_preview[:, ch:ch+1] + 1.0) * 0.5)
                    panels.extend([
                        (real_preview + 1.0) * 0.5,
                        (pred_preview + 1.0) * 0.5,
                        ((pred_preview - real_preview).abs() * 0.5).clamp(0.0, 1.0),
                    ])
                    vutils.save_image(
                        torch.cat(panels, dim=0),
                        os.path.join(args.out_dir, f"preview_{step:06d}.png"),
                        nrow=n,
                    )

            should_validate = val_pool is not None and step > 0 and step % args.val_interval == 0
            if should_validate:
                val_metrics = evaluate_paired_pool(
                    src,
                    model_ema,
                    val_pool,
                    args,
                    device,
                    sgs_params,
                    agc_params,
                    det_params,
                    condition_edge_type,
                    save_dir=args.out_dir,
                    prefix=f"val_step{step:06d}",
                )
                val_metrics.update({"event": "validation", "step": step, "epoch": epoch})
                append_jsonl(os.path.join(args.out_dir, "metrics_validation.jsonl"), val_metrics)
                current = float(val_metrics["paired_l1"])
                logger.info(f"VALIDATION step={step} | {json.dumps(val_metrics, ensure_ascii=False)}")
                if current < best_score:
                    best_score = current
                    save_checkpoint(
                        os.path.join(args.out_dir, "best_unet_regression.pth"),
                        model,
                        model_ema,
                        optimizer,
                        scaler,
                        step,
                        epoch,
                        args,
                        source_path,
                        best_score,
                    )
                    logger.info(f"New best validation L1={best_score:.6f}")
            elif val_pool is None and running_score is not None and running_score < best_score:
                best_score = float(running_score)
                save_checkpoint(
                    os.path.join(args.out_dir, "best_unet_regression.pth"),
                    model,
                    model_ema,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    args,
                    source_path,
                    best_score,
                )

            if step > 0 and step % args.save_interval == 0:
                save_checkpoint(
                    os.path.join(args.out_dir, f"checkpoint_{step:06d}.pth"),
                    model,
                    model_ema,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    args,
                    source_path,
                    best_score,
                )
                save_checkpoint(
                    os.path.join(args.out_dir, "latest_unet_regression.pth"),
                    model,
                    model_ema,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    args,
                    source_path,
                    best_score,
                )
            step += 1
        epoch += 1

    save_checkpoint(
        os.path.join(args.out_dir, "final_unet_regression.pth"),
        model,
        model_ema,
        optimizer,
        scaler,
        max(0, step - 1),
        epoch,
        args,
        source_path,
        best_score,
    )
    logger.info("Training finished.")


def build_loaded_model(args, src, device: torch.device) -> Tuple[nn.Module, Any, Any, Any, str, int]:
    amp = bool(args.amp and device.type == "cuda")
    condition_edge_type, sgs_params, agc_params, det_params, cond_channels = src._build_preproc_params(
        args, device, amp
    )
    model = UNetRegression(
        in_channels=cond_channels,
        out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    load_checkpoint(args.ckpt_path, model, device, prefer_ema=not args.use_raw_weights, strict=True)
    return model, sgs_params, agc_params, det_params, condition_edge_type, cond_channels


def test_baseline(args) -> None:
    src, source_path = load_source_module(args.source_code)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, "unet_test")
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    set_seed_all(args.seed)

    model, sgs_params, agc_params, det_params, condition_edge_type, cond_channels = build_loaded_model(
        args, src, device
    )
    model.eval()
    adapter = UNetEvaluationAdapter(model).to(device).eval()

    # 1) Direct paired reconstruction on every crop in the fixed pool.
    if args.eval_pool_path:
        real_pool, pool_indices = load_pool_tensor(args.eval_pool_path)
        paired = evaluate_paired_pool(
            src,
            model,
            real_pool,
            args,
            device,
            sgs_params,
            agc_params,
            det_params,
            condition_edge_type,
            save_dir=args.out_dir,
            prefix="test_paired",
        )
        paired.update({
            "checkpoint": args.ckpt_path,
            "source_code": str(source_path),
            "condition_edge_type": condition_edge_type,
            "tex_layout": args.tex_layout,
            "pool_path": args.eval_pool_path,
            "pool_has_indices": pool_indices is not None,
        })
        with open(os.path.join(args.out_dir, "paired_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(paired, f, ensure_ascii=False, indent=2)
        logger.info(f"PAIRED | {json.dumps(paired, ensure_ascii=False)}")

    # 2) Reuse the exact original distributional evaluation.
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        seeds = [args.seed]
    runs: List[Dict[str, Any]] = []
    for base_seed in seeds:
        for repeat in range(args.repeats):
            run_seed = int(base_seed + repeat * 100000)
            out = src.evaluate_once(
                args=args,
                device=device,
                G=adapter,
                sgs_params=sgs_params,
                agc_params=agc_params,
                det_params=det_params,
                condition_edge_type=condition_edge_type,
                evaluation_edge_type=args.eval_edge_type,
                seed=run_seed,
            )
            out["model"] = "UNetRegression"
            out["checkpoint"] = args.ckpt_path
            runs.append(out)
            append_jsonl(os.path.join(args.out_dir, "metrics_test_runs.jsonl"), out)
            logger.info(
                f"RUN seed={run_seed} | SWD={out.get('swd_total', float('nan')):.6f} | "
                f"GEO_KID={out.get('geo_kid_mean', float('nan')):.6f} | "
                f"PHI_FD={out.get('phi_fd_total', float('nan')):.6f} | "
                f"RRFD={out.get('rr_fd_ratio_total', float('nan')):.6f}"
            )

    summary = src._aggregate_runs(runs)
    with open(os.path.join(args.out_dir, "metrics_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "summary": summary}, f, ensure_ascii=False, indent=2)
    logger.info(f"SUMMARY | {json.dumps(summary, ensure_ascii=False)}")


def generate_baseline(args) -> None:
    src, _ = load_source_module(args.source_code)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    set_seed_all(args.generate_seed)
    model, sgs_params, agc_params, det_params, condition_edge_type, _ = build_loaded_model(args, src, device)
    model.eval()

    image_out = Path(args.out_dir) / "images"
    label_out = Path(args.out_dir) / "labels"
    if args.overwrite_generate and Path(args.out_dir).exists():
        shutil.rmtree(args.out_dir)
    image_out.mkdir(parents=True, exist_ok=True)
    if args.label_dir:
        label_out.mkdir(parents=True, exist_ok=True)

    ds = CroppedImageDirectory(args.input_dir, args.img_size)
    loader = DataLoader(ds, batch_size=args.generate_batch, shuffle=False, num_workers=args.num_workers)
    manifest_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating U-Net regression dataset"):
            real = batch["image"].to(device)
            cond = src.compute_texture_maps_batch(
                real,
                sgs_params=sgs_params,
                agc_params=agc_params,
                det_params=det_params,
                device=device,
                edge_type=condition_edge_type,
            )
            pred = model(cond).cpu()
            names = list(batch["name"])
            for i, name in enumerate(names):
                out_name = Path(name).with_suffix(".png").name
                vutils.save_image((pred[i:i+1] + 1.0) * 0.5, image_out / out_name, nrow=1)
                label_copied = False
                if args.label_dir:
                    stem = Path(name).stem
                    candidates = [p for p in Path(args.label_dir).iterdir() if p.is_file() and p.stem == stem]
                    if len(candidates) != 1:
                        raise RuntimeError(f"Expected one label for image ID {stem}, found {len(candidates)}")
                    shutil.copy2(candidates[0], label_out / candidates[0].name)
                    label_copied = True
                manifest_rows.append({
                    "source_name": name,
                    "generated_name": out_name,
                    "condition_edge_type": condition_edge_type,
                    "tex_layout": args.tex_layout,
                    "label_copied": label_copied,
                })

    manifest_path = Path(args.out_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Generated {len(manifest_rows)} images in {image_out}")
    print(f"Manifest: {manifest_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deterministic U-Net regression baseline for seismic conditional synthesis",
    )
    p.add_argument("--mode", choices=["train", "test", "generate"], default="train")
    p.add_argument("--source_code", type=str,default="", help="Path to the original StyleGAN2 Python file")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_nondeterministic", action="store_true")
    p.add_argument("--amp", action="store_true")

    # Paths and basic data settings.
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--val_pool_path", type=str, default="", help="Validation pool; do not use the final test pool")
    p.add_argument("--eval_pool_path", type=str, default="")
    p.add_argument("--input_dir", type=str, default="", help="Already-cropped input images for generate mode")
    p.add_argument("--label_dir", type=str, default="", help="Optional paired labels copied in generate mode")
    p.add_argument("--overwrite_generate", action="store_true")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--start_max_right", type=int, default=900)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    # U-Net and optimization.
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=25000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--use_raw_weights", action="store_true", help="Test raw weights rather than EMA weights")

    # Condition definition: same defaults as the uploaded source.
    p.add_argument("--edge_type", choices=["slog", "glog", "log", "canny"], default="slog")
    p.add_argument("--eval_edge_type", choices=["slog", "glog", "log", "canny"], default="log")
    p.add_argument("--tex_layout", type=str, default="fault,agc,edge")
    p.add_argument("--leak_down", type=int, default=4)
    p.add_argument("--leak_noise", type=float, default=0.03)
    p.add_argument("--sgs_leak_down", type=int, default=0)
    p.add_argument("--sgs_leak_noise", type=float, default=-1.0)
    p.add_argument("--sgs_leak_blur", type=float, default=0.0)
    p.add_argument("--sgs_leak_qbits", type=int, default=0)
    p.add_argument("--fault_leak_down", type=int, default=1)
    p.add_argument("--fault_leak_noise", type=float, default=0.01)
    p.add_argument("--fault_w_coh", type=float, default=0.55)
    p.add_argument("--fault_w_ori", type=float, default=0.30)
    p.add_argument("--fault_w_edge", type=float, default=0.15)
    p.add_argument("--fault_thr", type=float, default=0.45)
    p.add_argument("--fault_k", type=float, default=12.0)
    p.add_argument("--fault_seed_thr", type=float, default=0.65)
    p.add_argument("--fault_band_sigma", type=float, default=3.5)
    p.add_argument("--fault_smooth_sigma", type=float, default=0.8)
    p.add_argument("--ori_coh_tau", type=float, default=0.35)

    # Paired geometric and condition perturbations copied from the source code.
    p.add_argument("--pair_geom_prob", type=float, default=0.20)
    p.add_argument("--pair_geom_px", type=int, default=1)
    p.add_argument("--pair_geom_hflip", type=float, default=0.50)
    p.add_argument("--no_cond_aug", action="store_true", help="Disable online post-threshold condition perturbations")
    p.add_argument("--cond_aug_prob", type=float, default=0.25)
    p.add_argument("--cond_channel_drop_prob", type=float, default=0.10)
    p.add_argument("--cond_struct_pixel_dropout_rate", type=float, default=0.04)
    p.add_argument("--cond_struct_block_prob", type=float, default=0.15)
    p.add_argument("--cond_struct_block_size", type=int, default=24)
    p.add_argument("--cond_struct_block_num", type=int, default=1)
    p.add_argument("--edge_drop_base", type=float, default=0.01)
    p.add_argument("--edge_drop_boost", type=float, default=0.15)
    p.add_argument("--edge_drop_gamma", type=float, default=1.2)
    p.add_argument("--edge_dense_low", type=float, default=0.55)
    p.add_argument("--edge_dense_high", type=float, default=0.85)
    p.add_argument("--edge_dense_boost", type=float, default=1.8)
    p.add_argument("--edge_block_prob", type=float, default=0.30)
    p.add_argument("--edge_block_size", type=int, default=16)
    p.add_argument("--edge_block_num", type=int, default=2)
    p.add_argument("--edge_ds_prob", type=float, default=0.20)
    p.add_argument("--edge_ds_min", type=int, default=2)
    p.add_argument("--edge_ds_max", type=int, default=8)
    p.add_argument("--cond_jitter_prob", type=float, default=0.10)
    p.add_argument("--cond_jitter_px", type=int, default=1)

    # Regression loss. Defaults intentionally mirror the source auxiliary losses.
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_spec", type=float, default=0.15)
    p.add_argument("--lambda_grad", type=float, default=0.2)
    p.add_argument("--lambda_grad_dir", type=float, default=0.03)
    p.add_argument("--lambda_st_ori", type=float, default=0.02)
    p.add_argument("--spec_scales", type=str, default="1,2,4,8")
    p.add_argument("--spec_log", type=int, default=1)
    p.add_argument("--grad_scales", type=str, default="1,2,4")
    p.add_argument("--grad_dir_scales", type=str, default="")
    p.add_argument("--st_ori_scales", type=str, default="")
    p.add_argument("--st_sigmas", type=str, default="0.8")
    p.add_argument("--grad_dir_mag_power", type=float, default=1.0)
    p.add_argument("--grad_dir_mag_thresh", type=float, default=0.0)
    p.add_argument("--st_coh_power", type=float, default=1.5)
    p.add_argument("--struct_hard_q", type=float, default=0.85)
    p.add_argument("--struct_hard_min", type=float, default=0.0)
    p.add_argument("--spec_warmup_steps", type=int, default=800)
    p.add_argument("--spec_ramp_steps", type=int, default=1200)
    p.add_argument("--grad_warmup_steps", type=int, default=0)
    p.add_argument("--grad_ramp_steps", type=int, default=400)
    p.add_argument("--grad_dir_warmup_steps", type=int, default=0)
    p.add_argument("--grad_dir_ramp_steps", type=int, default=400)
    p.add_argument("--st_ori_warmup_steps", type=int, default=100)
    p.add_argument("--st_ori_ramp_steps", type=int, default=600)

    # Logging/checkpointing.
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--preview_interval", type=int, default=500)
    p.add_argument("--preview_num", type=int, default=4)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--val_interval", type=int, default=500)

    # Original distributional evaluation options.
    p.add_argument("--z_dim", type=int, default=512, help="Ignored by U-Net but required by evaluate_once")
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--eval_cond_num", type=int, default=256)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--eval_condition_seed", type=int, default=42001)
    p.add_argument("--seeds", type=str, default="43,44,45,46,47")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--test_mixing_prob", type=float, default=0.0)
    p.add_argument("--save_test_images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_pair_num", type=int, default=8)
    p.add_argument("--rr_fd_trials", type=int, default=20)
    p.add_argument("--rr_fd_percentile", type=float, default=95.0)
    p.add_argument("--rr_fd_min_n", type=int, default=64)
    p.add_argument("--kid_subsets", type=int, default=20)
    p.add_argument("--kid_subset_size", type=int, default=100)
    p.add_argument("--kid_degree", type=int, default=3)
    p.add_argument("--kid_coef0", type=float, default=1.0)
    p.add_argument("--kid_gamma", type=float, default=-1.0)
    p.add_argument("--prdc_ks", type=str, default="3,5,10,20")
    p.add_argument("--prdc_boot", type=int, default=200)
    p.add_argument("--prdc_ci", type=float, default=95.0)
    p.add_argument("--swd_scales", type=str, default="1,2,4")
    p.add_argument("--swd_patch", type=int, default=7)
    p.add_argument("--swd_patches", type=int, default=256)
    p.add_argument("--swd_projections", type=int, default=128)

    # Generate mode.
    p.add_argument("--generate_batch", type=int, default=4)
    p.add_argument("--generate_seed", type=int, default=43000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train_baseline(args)
    elif args.mode == "test":
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required for test mode")
        test_baseline(args)
    elif args.mode == "generate":
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required for generate mode")
        if not args.input_dir:
            raise ValueError("--input_dir is required for generate mode")
        generate_baseline(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
