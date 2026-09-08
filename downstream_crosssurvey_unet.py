"""
下游断层检测实验脚本

Cross-survey 下游断层分割实验：
使用同一个 2D U-Net、公用初始权重，分别以 real、slog、log、canny 四组
Thebe 训练数据训练，在 Thebe validation 上选模，并在 CRACKS/F3 expert 标注上
进行零微调（zero-shot）测试。F3 测试图保持原生尺寸，不拉伸到 512x512。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
from PIL import Image, ImageEnhance
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(text: str) -> List[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(device_text: str) -> torch.device:
    device = torch.device(device_text or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了CUDA，但当前环境无法使用CUDA。")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"指定了 cuda:{index}，但当前仅检测到 {torch.cuda.device_count()} 张GPU。"
            )
    return device


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class PairRecord:
    stem: str
    image_path: str
    mask_path: str
    source_name: str


def collect_files(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    mapping: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in mapping:
            raise RuntimeError(
                f"发现重复文件名主体：{mapping[path.stem].name} 与 {path.name}"
            )
        mapping[path.stem] = path
    if not mapping:
        raise RuntimeError(f"目录中没有可读取图像：{directory}")
    return mapping


def build_pair_records(
    image_dir: str,
    mask_dir: str,
    source_name: str,
    expected_count: int = 0,
    allow_images_without_masks: bool = False,
) -> List[PairRecord]:
    image_root = Path(image_dir).expanduser().resolve()
    mask_root = Path(mask_dir).expanduser().resolve()
    image_map = collect_files(image_root)
    mask_map = collect_files(mask_root)

    missing_masks = sorted(set(image_map) - set(mask_map), key=natural_key)
    missing_images = sorted(set(mask_map) - set(image_map), key=natural_key)
    if missing_masks and not allow_images_without_masks:
        raise RuntimeError(f"{source_name}中有图像缺少标签：{missing_masks[:20]}")
    if missing_masks and allow_images_without_masks:
        print(
            f"[pairing] {source_name}: {len(missing_masks)} images have no expert mask and "
            "are intentionally excluded; only the image/mask intersection is evaluated."
        )
    if missing_images:
        raise RuntimeError(f"{source_name}中有标签缺少图像：{missing_images[:20]}")

    stems = sorted(set(image_map) & set(mask_map), key=natural_key)
    if expected_count > 0 and len(stems) != expected_count:
        raise RuntimeError(
            f"{source_name}期望{expected_count}对，实际找到{len(stems)}对。"
        )

    return [
        PairRecord(
            stem=stem,
            image_path=str(image_map[stem]),
            mask_path=str(mask_map[stem]),
            source_name=source_name,
        )
        for stem in stems
    ]


def binarize_fault_mask(mask: Image.Image, threshold_u8: int) -> torch.Tensor:
    """
    将灰度断层标签转换为严格二值掩码。

    当前标签集的统计结果：
    - 背景灰度为 9；
    - 单次断层描线主体灰度为 38；
    - 10~37 主要为抗锯齿过渡；
    - >38 为重叠描线或更亮的断层区域。

    因此默认使用 >=38，将断层主体及更亮区域统一置为1，
    同时排除背景与抗锯齿灰边。
    """
    threshold_u8 = int(threshold_u8)
    if not 0 <= threshold_u8 <= 255:
        raise ValueError(f"mask_threshold_u8必须位于0~255，当前为{threshold_u8}。")

    array = np.asarray(mask, dtype=np.uint8)
    binary = (array >= threshold_u8).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)




def _seismic_palette_u8() -> np.ndarray:
    """Return the 256 RGB colors of Matplotlib's standard seismic colormap."""
    try:
        cmap = matplotlib.colormaps["seismic"].resampled(256)
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("seismic", 256)
    return np.rint(cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.int16)


CRACKS_SEISMIC_PALETTE = _seismic_palette_u8()


def decode_cracks_seismic_image(image: Image.Image, max_rgb_error: float = 6.0) -> torch.Tensor:
    """Decode CRACKS rendered RGB seismic images back to a signed scalar image.

    The uploaded CRACKS PNGs are RGBA renderings of Matplotlib's ``seismic``
    colormap (blue -> negative, white -> near zero, red -> positive).  A direct
    RGB-to-grayscale conversion would collapse the positive/negative polarity
    information.  We therefore map each unique RGB color to its nearest
    colormap index and convert index 0..255 to amplitude -1..1.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)

    u = unique.astype(np.int16)
    p = CRACKS_SEISMIC_PALETTE.astype(np.int16)
    diff = u[:, None, :] - p[None, :, :]
    dist2 = np.sum(diff.astype(np.int32) ** 2, axis=2)
    nearest = np.argmin(dist2, axis=1).astype(np.int64)
    max_err = float(np.sqrt(np.max(np.min(dist2, axis=1))))
    if max_err > float(max_rgb_error):
        raise ValueError(
            f"CRACKS image colors do not match Matplotlib seismic colormap: "
            f"max nearest RGB error={max_err:.3f} > {max_rgb_error:.3f}"
        )

    idx = nearest[inverse].reshape(h, w).astype(np.float32)
    amplitude = idx / 127.5 - 1.0
    return torch.from_numpy(amplitude).unsqueeze(0)


def audit_cracks_image_records(records: Sequence[PairRecord], source_name: str) -> Dict[str, object]:
    """Check CRACKS image geometry/mode and seismic-colormap compatibility."""
    sizes = set()
    modes = set()
    max_errors: List[float] = []
    palette = CRACKS_SEISMIC_PALETTE.astype(np.int16)
    for record in records:
        image = Image.open(record.image_path)
        sizes.add(tuple(image.size))
        modes.add(str(image.mode))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        unique = np.unique(rgb.reshape(-1, 3), axis=0).astype(np.int16)
        diff = unique[:, None, :] - palette[None, :, :]
        dist2 = np.sum(diff.astype(np.int32) ** 2, axis=2)
        max_errors.append(float(np.sqrt(np.max(np.min(dist2, axis=1)))))
    summary = {
        "source": source_name,
        "count": len(records),
        "sizes_wh": [list(x) for x in sorted(sizes)],
        "modes": sorted(modes),
        "max_seismic_palette_rgb_error": float(max(max_errors) if max_errors else 0.0),
    }
    print(
        f"[CRACKS image audit] {source_name}: n={len(records)}, "
        f"sizes={summary['sizes_wh']}, modes={summary['modes']}, "
        f"max_palette_error={summary['max_seismic_palette_rgb_error']:.3f}"
    )
    return summary


def decode_cracks_expert_mask(mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode the CRACKS expert palette mask.

    The uploaded CRACKS labels use palette indices/colors:
      0 = white   -> unlabeled / not used for scoring
      1 = orange  -> confident non-fault (valid negative)
      2 = green   -> uncertain fault (ignored in the primary evaluation)
      3 = blue    -> confident fault (valid positive)

    Primary evaluation therefore uses only confident annotations:
      target=1 for class 3; target=0 for class 1; classes 0 and 2 are ignored.
    """
    if mask.mode == "P":
        arr = np.asarray(mask, dtype=np.uint8)
    else:
        rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        arr = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        color_to_index = {
            (255, 255, 255): 0,
            (255, 127, 14): 1,
            (44, 160, 44): 2,
            (31, 119, 180): 3,
        }
        for color, idx in color_to_index.items():
            arr[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = idx
        unknown = arr == 255
        if unknown.any():
            colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)[:10].tolist()
            raise ValueError(f"CRACKS expert mask contains unknown RGB colors: {colors}")

    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected CRACKS expert class indices: {sorted(unique)}")

    target = (arr == 3).astype(np.float32)
    valid = np.logical_or(arr == 1, arr == 3).astype(np.float32)
    return torch.from_numpy(target).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def audit_cracks_expert_records(records: Sequence[PairRecord], source_name: str) -> Dict[str, object]:
    """Audit CRACKS expert masks without converting their palette to grayscale."""
    counts = np.zeros(4, dtype=np.int64)
    sizes = set()
    for record in records:
        mask = Image.open(record.mask_path)
        sizes.add(tuple(mask.size))
        if mask.mode == "P":
            arr = np.asarray(mask, dtype=np.uint8)
        else:
            target, valid = decode_cracks_expert_mask(mask)
            # Reconstruct only the counts needed for a concise audit when RGB is supplied.
            counts[3] += int(target.sum().item())
            counts[1] += int((valid - target).clamp(min=0).sum().item())
            continue
        binc = np.bincount(arr.ravel(), minlength=4)[:4]
        counts += binc.astype(np.int64)

    total = int(counts.sum())
    valid_total = int(counts[1] + counts[3])
    summary = {
        "source": source_name,
        "count": len(records),
        "sizes_wh": [list(x) for x in sorted(sizes)],
        "class_counts": {str(i): int(counts[i]) for i in range(4)},
        "valid_confident_pixels": valid_total,
        "confident_fault_pixels": int(counts[3]),
        "confident_fault_fraction_within_valid": float(counts[3] / max(1, valid_total)),
    }
    print(
        f"[CRACKS audit] {source_name}: n={len(records)}, sizes={summary['sizes_wh']}, "
        f"class0/1/2/3={counts.tolist()}, valid={valid_total}, "
        f"fault(valid)={summary['confident_fault_fraction_within_valid']:.4f}"
    )
    return summary


def audit_mask_records(
    records: Sequence[PairRecord],
    threshold_u8: int,
    source_name: str,
) -> Dict[str, object]:
    """在训练前检查标签灰度、空标签数量和二值化后的正类比例。"""
    modes: List[int] = []
    minimums: List[int] = []
    maximums: List[int] = []
    positive_ratios: List[float] = []
    empty_stems: List[str] = []

    for record in records:
        array = np.asarray(
            Image.open(record.mask_path).convert("L"),
            dtype=np.uint8,
        )
        histogram = np.bincount(array.ravel(), minlength=256)
        modes.append(int(histogram.argmax()))
        minimums.append(int(array.min()))
        maximums.append(int(array.max()))

        binary = array >= int(threshold_u8)
        ratio = float(binary.mean())
        positive_ratios.append(ratio)
        if not binary.any():
            empty_stems.append(record.stem)

    if not positive_ratios:
        raise RuntimeError(f"{source_name}没有可审计标签。")

    summary = {
        "source": source_name,
        "count": len(records),
        "threshold_u8": int(threshold_u8),
        "background_modes": sorted(set(modes)),
        "minimum_values": sorted(set(minimums)),
        "maximum_values": sorted(set(maximums)),
        "positive_ratio_mean": float(np.mean(positive_ratios)),
        "positive_ratio_median": float(np.median(positive_ratios)),
        "positive_ratio_min": float(np.min(positive_ratios)),
        "positive_ratio_max": float(np.max(positive_ratios)),
        "empty_count": len(empty_stems),
        "empty_stems": empty_stems,
    }

    print(
        f"[mask audit] {source_name}: n={len(records)}, "
        f"threshold>={int(threshold_u8)}, "
        f"bg_modes={summary['background_modes']}, "
        f"positive mean/median="
        f"{summary['positive_ratio_mean']:.4f}/"
        f"{summary['positive_ratio_median']:.4f}, "
        f"range={summary['positive_ratio_min']:.4f}~"
        f"{summary['positive_ratio_max']:.4f}, "
        f"empty={summary['empty_count']}"
    )

    if summary["positive_ratio_max"] > 0.20:
        print(
            f"[warning] {source_name}存在正类比例超过20%的标签，"
            "请确认标签是否被错误二值化。"
        )

    return summary


class PairedFaultDataset(Dataset):
    """Paired grayscale seismic image and fault mask.

    Training/validation use the original Thebe binary-label protocol.
    CRACKS/F3 test data can keep native image size and return an additional
    valid_mask so uncertain/unlabeled expert pixels do not enter metrics.
    """

    def __init__(
        self,
        records: Sequence[PairRecord],
        img_size: int = 512,
        augment: bool = False,
        hflip_prob: float = 0.5,
        intensity_aug_prob: float = 0.25,
        strict_size: bool = True,
        mask_threshold_u8: int = 38,
        native_size: bool = False,
        mask_mode: str = "thebe",
        image_mode: str = "thebe_grayscale",
    ) -> None:
        if not records:
            raise ValueError("No records found.")
        if mask_mode not in {"thebe", "cracks_expert"}:
            raise ValueError(f"Unsupported mask_mode: {mask_mode}")
        if image_mode not in {"thebe_grayscale", "cracks_seismic"}:
            raise ValueError(f"Unsupported image_mode: {image_mode}")
        self.records = list(records)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.intensity_aug_prob = float(intensity_aug_prob)
        self.strict_size = bool(strict_size)
        self.mask_threshold_u8 = int(mask_threshold_u8)
        self.native_size = bool(native_size)
        self.mask_mode = str(mask_mode)
        self.image_mode = str(image_mode)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        raw_image = Image.open(record.image_path)
        image = raw_image.convert("L") if self.image_mode == "thebe_grayscale" else raw_image
        raw_mask = Image.open(record.mask_path)
        expected = (self.img_size, self.img_size)

        if self.native_size:
            if image.size != raw_mask.size:
                raise ValueError(
                    f"Native-size test pair mismatch: image={image.size}, mask={raw_mask.size}, "
                    f"stem={record.stem}"
                )
        elif self.strict_size:
            if image.size != expected:
                raise ValueError(f"{record.image_path}尺寸{image.size}，期望{expected}。")
            if raw_mask.size != expected:
                raise ValueError(f"{record.mask_path}尺寸{raw_mask.size}，期望{expected}。")
        else:
            if image.size != expected:
                image = image.resize(expected, Image.Resampling.BILINEAR)
            if raw_mask.size != expected:
                raw_mask = raw_mask.resize(expected, Image.Resampling.NEAREST)

        if self.augment:
            if self.image_mode != "thebe_grayscale":
                raise ValueError("Augmentation is only defined for Thebe grayscale training images.")
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                raw_mask = TF.hflip(raw_mask)
            if random.random() < self.intensity_aug_prob:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.10))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))

        if self.image_mode == "cracks_seismic":
            image_tensor = decode_cracks_seismic_image(image)
        else:
            image_tensor = TF.to_tensor(image.convert("L")).float() * 2.0 - 1.0
        if self.mask_mode == "cracks_expert":
            mask_tensor, valid_mask = decode_cracks_expert_mask(raw_mask)
        else:
            mask_tensor = binarize_fault_mask(raw_mask.convert("L"), self.mask_threshold_u8)
            valid_mask = torch.ones_like(mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "valid_mask": valid_mask,
            "stem": record.stem,
            "source": record.source_name,
        }


def dataset_records(dataset: Dataset) -> List[PairRecord]:
    if isinstance(dataset, PairedFaultDataset):
        return list(dataset.records)
    if isinstance(dataset, ConcatDataset):
        output: List[PairRecord] = []
        for part in dataset.datasets:
            output.extend(dataset_records(part))
        return output
    raise TypeError(f"不支持的数据集类型：{type(dataset)}")


def validate_no_leakage(
    train_records: Sequence[PairRecord],
    eval_records: Sequence[PairRecord],
    allow_same_stem: bool,
    eval_name: str,
) -> None:
    if allow_same_stem:
        return
    overlap = sorted(
        {r.stem for r in train_records} & {r.stem for r in eval_records},
        key=natural_key,
    )
    if overlap:
        raise RuntimeError(
            f"训练集与{eval_name}存在同名编号，可能数据泄漏：{overlap[:20]}\n"
            "请重新划分；只有确认它们不是同一切片时才使用"
            " --allow_same_stem_across_splits。"
        )


# -------------------- U-Net --------------------

def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class FaultUNet(nn.Module):
    def __init__(self, base_channels: int = 32, dropout: float = 0.10):
        super().__init__()
        b = int(base_channels)
        self.enc1 = ConvBlock(1, b)
        self.enc2 = DownBlock(b, b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8, dropout)
        self.bottleneck = DownBlock(b * 8, b * 16, dropout)
        self.up4 = UpBlock(b * 16, b * 8, b * 8, dropout)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        x = self.bottleneck(e4)
        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.head(x)


# -------------------- Loss --------------------

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float, bce_weight: float, dice_weight: float):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        dice = soft_dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def estimate_pos_weight(dataset: Dataset, max_weight: float, mask_threshold_u8: int) -> float:
    positive = 0.0
    total = 0.0
    for record in tqdm(dataset_records(dataset), desc="Estimating pos_weight", leave=False):
        arr = np.asarray(Image.open(record.mask_path).convert("L"), dtype=np.uint8)
        binary = arr >= int(mask_threshold_u8)
        positive += float(binary.sum())
        total += float(binary.size)
    if positive <= 0:
        raise RuntimeError("训练标签中没有正类断层像素。")
    return float(np.clip((total - positive) / positive, 1.0, max_weight))


# -------------------- Metrics --------------------

@dataclass
class MetricResult:
    loss: float
    precision: float
    recall: float
    f1_dice: float
    iou: float
    accuracy: float
    specificity: float
    tolerant_precision: float
    tolerant_recall: float
    tolerant_f1: float
    per_image_dice_mean: float
    per_image_dice_std: float
    tp: int
    fp: int
    fn: int
    tn: int


class MetricAccumulator:
    def __init__(self, tolerance_radius: int):
        self.radius = int(tolerance_radius)
        self.tp = self.fp = self.fn = self.tn = 0
        self.tol_pred_match = self.tol_pred_total = 0
        self.tol_gt_match = self.tol_gt_total = 0
        self.per_image_dice: List[float] = []
        self.loss_sum = 0.0
        self.batch_count = 0

    @torch.no_grad()
    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, loss: float,
        threshold: float, valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        preds = torch.sigmoid(logits) >= threshold
        gts = targets >= 0.5
        valid = torch.ones_like(gts, dtype=torch.bool) if valid_mask is None else (valid_mask >= 0.5)

        self.tp += int((preds & gts & valid).sum().item())
        self.fp += int((preds & ~gts & valid).sum().item())
        self.fn += int((~preds & gts & valid).sum().item())
        self.tn += int((~preds & ~gts & valid).sum().item())

        for i in range(preds.shape[0]):
            vi = valid[i]
            if not bool(vi.any()):
                continue
            tp = float((preds[i] & gts[i] & vi).sum().item())
            fp = float((preds[i] & ~gts[i] & vi).sum().item())
            fn = float((~preds[i] & gts[i] & vi).sum().item())
            self.per_image_dice.append((2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8))

        preds_valid = preds & valid
        gts_valid = gts & valid
        if self.radius > 0:
            k = 2 * self.radius + 1
            dilated_gt = F.max_pool2d(gts_valid.float(), k, 1, self.radius) >= 0.5
            dilated_pred = F.max_pool2d(preds_valid.float(), k, 1, self.radius) >= 0.5
            self.tol_pred_match += int((preds_valid & dilated_gt).sum().item())
            self.tol_pred_total += int(preds_valid.sum().item())
            self.tol_gt_match += int((gts_valid & dilated_pred).sum().item())
            self.tol_gt_total += int(gts_valid.sum().item())
        else:
            self.tol_pred_match += int((preds_valid & gts_valid).sum().item())
            self.tol_pred_total += int(preds_valid.sum().item())
            self.tol_gt_match += int((preds_valid & gts_valid).sum().item())
            self.tol_gt_total += int(gts_valid.sum().item())

        self.loss_sum += float(loss)
        self.batch_count += 1

    def compute(self) -> MetricResult:
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        tol_p = self.tol_pred_match / (self.tol_pred_total + eps)
        tol_r = self.tol_gt_match / (self.tol_gt_total + eps)
        tol_f1 = 2 * tol_p * tol_r / (tol_p + tol_r + eps)
        mean_dice = statistics.mean(self.per_image_dice) if self.per_image_dice else 0.0
        std_dice = statistics.pstdev(self.per_image_dice) if len(self.per_image_dice) > 1 else 0.0
        return MetricResult(
            loss=self.loss_sum / max(1, self.batch_count),
            precision=float(precision), recall=float(recall), f1_dice=float(f1),
            iou=float(iou), accuracy=float(accuracy), specificity=float(specificity),
            tolerant_precision=float(tol_p), tolerant_recall=float(tol_r),
            tolerant_f1=float(tol_f1), per_image_dice_mean=float(mean_dice),
            per_image_dice_std=float(std_dice), tp=self.tp, fp=self.fp, fn=self.fn, tn=self.tn,
        )


# -------------------- Train / Eval --------------------

def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int, pin: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=pin, drop_last=False, persistent_workers=(workers > 0),
        worker_init_fn=seed_worker, generator=generator,
    )


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: BCEDiceLoss, device: torch.device, scaler, amp: bool,
    threshold: float, tolerance_radius: int, grad_clip: float,
) -> MetricResult:
    model.train()
    meter = MetricAccumulator(tolerance_radius)
    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"损失出现NaN/Inf：{loss.item()}")
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        meter.update(logits.detach(), masks, float(loss.item()), threshold)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return meter.compute()


def masked_bce_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor, criterion: BCEDiceLoss
) -> torch.Tensor:
    """Apply the same BCE+Dice form only on valid expert-labeled pixels."""
    valid = valid_mask.float()
    denom_valid = valid.sum().clamp_min(1.0)
    bce_map = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=criterion.pos_weight, reduction="none"
    )
    bce = (bce_map * valid).sum() / denom_valid

    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = torch.sum(probs * targets * valid, dim=dims)
    denom = torch.sum(probs * valid, dim=dims) + torch.sum(targets * valid, dim=dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    dice_loss = 1.0 - dice.mean()
    return criterion.bce_weight * bce + criterion.dice_weight * dice_loss


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: BCEDiceLoss,
    device: torch.device, amp: bool, threshold: float, tolerance_radius: int,
    preview_dir: Optional[Path] = None, preview_count: int = 0,
) -> MetricResult:
    model.eval()
    meter = MetricAccumulator(tolerance_radius)
    saved = 0
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)
        with autocast_context(amp):
            logits = model(images)
            if valid_mask is None:
                loss = criterion(logits, masks)
            else:
                loss = masked_bce_dice_loss(logits, masks, valid_mask, criterion)
        meter.update(logits, masks, float(loss.item()), threshold, valid_mask=valid_mask)

        if preview_dir and saved < preview_count:
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            for i in range(images.shape[0]):
                if saved >= preview_count:
                    break
                if valid_mask is None:
                    valid_vis = torch.ones_like(masks[i:i+1])
                else:
                    valid_vis = valid_mask[i:i+1].float()
                panel = torch.cat([
                    (images[i:i+1] + 1) / 2,
                    masks[i:i+1], valid_vis, probs[i:i+1], preds[i:i+1]
                ], dim=0)
                save_image(
                    panel,
                    preview_dir / f"{batch['stem'][i]}_image_gt_valid_prob_pred.png",
                    nrow=5,
                )
                saved += 1
    return meter.compute()


def append_csv(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def metric_columns(prefix: str, m: MetricResult) -> Dict[str, object]:
    return {
        f"{prefix}_loss": m.loss,
        f"{prefix}_precision": m.precision,
        f"{prefix}_recall": m.recall,
        f"{prefix}_f1_dice": m.f1_dice,
        f"{prefix}_iou": m.iou,
        f"{prefix}_accuracy": m.accuracy,
        f"{prefix}_specificity": m.specificity,
        f"{prefix}_tolerant_f1": m.tolerant_f1,
        f"{prefix}_per_image_dice_mean": m.per_image_dice_mean,
        f"{prefix}_per_image_dice_std": m.per_image_dice_std,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, score: float, args, experiment: str, pos_weight: float) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "best_score": score,
        "experiment": experiment, "pos_weight": pos_weight, "args": vars(args),
    }, path)


# -------------------- Experiment setup --------------------

def source_paths(args) -> Dict[str, Tuple[str, str]]:
    return {
        "real": (args.real_train_images, args.real_train_labels),
        "slog": (args.slog_train_images, args.slog_train_labels),
        "log": (args.log_train_images, args.log_train_labels),
        "canny": (args.canny_train_images, args.canny_train_labels),
    }


def experiment_sources(experiment: str) -> List[str]:
    parts = [x.strip().lower() for x in experiment.split("+") if x.strip()]
    valid = {"real", "slog", "log", "canny"}
    unknown = [x for x in parts if x not in valid]
    if not parts or unknown:
        raise ValueError(f"无效实验名：{experiment}；未知部分：{unknown}")
    return parts


def build_train_dataset(experiment: str, args) -> Dataset:
    datasets: List[Dataset] = []
    paths = source_paths(args)
    for source in experiment_sources(experiment):
        images, labels = paths[source]
        records = build_pair_records(images, labels, source, args.expected_train_count)
        if not args.disable_mask_audit:
            audit_mask_records(records, args.mask_threshold_u8, source)
        datasets.append(PairedFaultDataset(
            records, img_size=args.img_size, augment=not args.disable_augmentation,
            hflip_prob=args.hflip_prob, intensity_aug_prob=args.intensity_aug_prob,
            strict_size=not args.allow_resize, mask_threshold_u8=args.mask_threshold_u8,
        ))
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_eval_dataset(
    images: str, labels: str, name: str, expected: int, args,
    *, native_size: bool = False, mask_mode: str = "thebe",
    image_mode: str = "thebe_grayscale",
) -> PairedFaultDataset:
    records = build_pair_records(
        images, labels, name, expected,
        allow_images_without_masks=(mask_mode == "cracks_expert"),
    )
    if not args.disable_mask_audit:
        if mask_mode == "cracks_expert":
            audit_cracks_expert_records(records, name)
            if image_mode == "cracks_seismic":
                audit_cracks_image_records(records, name)
        else:
            audit_mask_records(records, args.mask_threshold_u8, name)
    return PairedFaultDataset(
        records, img_size=args.img_size, augment=False,
        strict_size=(not args.allow_resize) and (not native_size),
        mask_threshold_u8=args.mask_threshold_u8,
        native_size=native_size, mask_mode=mask_mode, image_mode=image_mode,
    )


def create_common_initial_state(args, output_root: Path) -> Path:
    path = output_root / "common_initial_state.pth"
    if path.exists() and not args.recreate_initial_state:
        return path
    set_global_seed(args.seed, args.deterministic)
    model = FaultUNet(args.base_channels, args.dropout)
    torch.save(model.state_dict(), path)
    return path


def train_experiment(experiment: str, args, device: torch.device, initial_state: Path, test_dataset: PairedFaultDataset, val_dataset: Optional[PairedFaultDataset]) -> Dict[str, object]:
    exp_dir = Path(args.output_root).expanduser().resolve() / experiment.replace("+", "_")
    if exp_dir.exists() and args.overwrite_experiment:
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    result_path = exp_dir / "test_metrics.json"
    if result_path.exists() and not args.overwrite_experiment:
        print(f"[skip] 已存在结果：{result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    set_global_seed(args.seed, args.deterministic)
    train_dataset = build_train_dataset(experiment, args)
    train_records = dataset_records(train_dataset)
    # The CRACKS/F3 test set comes from an independent survey. Identical filename
    # stems across surveys do not imply leakage, so stem-based leakage checking is
    # applied to the within-Thebe validation split but skipped for CRACKS test.
    if args.test_mask_mode != "cracks_expert":
        validate_no_leakage(
            train_records, dataset_records(test_dataset),
            args.allow_same_stem_across_splits, "真实测试集"
        )
    if val_dataset is not None:
        validate_no_leakage(
            train_records, dataset_records(val_dataset),
            args.allow_same_stem_across_splits, "真实验证集"
        )

    train_loader = build_loader(train_dataset, args.batch_size, True, args.num_workers, args.seed, device.type == "cuda")
    val_loader = build_loader(val_dataset, args.eval_batch_size, False, args.num_workers, args.seed, device.type == "cuda") if val_dataset else None
    test_loader = build_loader(test_dataset, args.test_batch_size, False, args.num_workers, args.seed, device.type == "cuda")

    model = FaultUNet(args.base_channels, args.dropout).to(device)
    model.load_state_dict(torch.load(initial_state, map_location="cpu"), strict=True)

    pos_weight = args.pos_weight if args.disable_auto_pos_weight else estimate_pos_weight(train_dataset, args.max_pos_weight, args.mask_threshold_u8)
    criterion = BCEDiceLoss(pos_weight, args.bce_weight, args.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_scaler(amp_enabled)

    best_path = exp_dir / "best_model.pth"
    last_path = exp_dir / "last_model.pth"
    history_path = exp_dir / "history.csv"
    best_score = -math.inf
    best_epoch = 0
    no_improve = 0

    save_json({
        "experiment": experiment, "train_count": len(train_dataset),
        "validation_count": len(val_dataset) if val_dataset else 0,
        "test_count": len(test_dataset), "pos_weight": pos_weight,
        "device": str(device), "args": vars(args),
    }, exp_dir / "experiment_config.json")

    print("\n" + "=" * 72)
    print(f"Experiment: {experiment} | train={len(train_dataset)} | val={len(val_dataset) if val_dataset else 0} | cross-survey test={len(test_dataset)}")
    print(f"Device={device} | AMP={amp_enabled} | pos_weight={pos_weight:.4f}")
    print("=" * 72)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            amp_enabled, args.threshold, args.tolerance_radius, args.grad_clip,
        )
        if val_loader:
            val_m = evaluate(model, val_loader, criterion, device, amp_enabled, args.threshold, args.tolerance_radius)
            monitor = val_m.f1_dice
        else:
            val_m = None
            monitor = train_m.f1_dice

        scheduler.step(monitor)
        row: Dict[str, object] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "monitor": monitor}
        row.update(metric_columns("train", train_m))
        if val_m:
            row.update(metric_columns("val", val_m))
        append_csv(history_path, row)

        val_text = f" | val Dice={val_m.f1_dice:.4f} IoU={val_m.iou:.4f} TolF1={val_m.tolerant_f1:.4f}" if val_m else " | no validation"
        print(f"[{experiment}] {epoch:03d}/{args.epochs} | loss={train_m.loss:.4f} | train Dice={train_m.f1_dice:.4f}{val_text} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if monitor > best_score + args.min_delta:
            best_score = monitor
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, args, experiment, pos_weight)
        else:
            no_improve += 1
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_score, args, experiment, pos_weight)

        if val_loader and args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            print(f"[{experiment}] early stopping；best epoch={best_epoch}")
            break

    selected = best_path if val_loader else last_path
    selection_rule = "best_validation_dice" if val_loader else "final_epoch_no_validation"
    checkpoint = torch.load(selected, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    test_m = evaluate(
        model, test_loader, criterion, device, amp_enabled,
        args.threshold, args.tolerance_radius,
        preview_dir=exp_dir / "test_previews", preview_count=args.preview_count,
    )
    result = {
        "experiment": experiment, "selection_rule": selection_rule,
        "selected_epoch": int(checkpoint["epoch"]),
        "train_count": len(train_dataset),
        "validation_count": len(val_dataset) if val_dataset else 0,
        "test_count": len(test_dataset), "pos_weight": float(pos_weight),
        "elapsed_minutes": (time.time() - start) / 60.0,
        "test": asdict(test_m),
    }
    save_json(result, result_path)
    print(f"[{experiment}] F3 CRACKS TEST | Dice={test_m.f1_dice:.4f} | IoU={test_m.iou:.4f} | Precision={test_m.precision:.4f} | Recall={test_m.recall:.4f} | TolF1={test_m.tolerant_f1:.4f}")
    return result


def write_summary(results: Sequence[Dict[str, object]], output_root: Path) -> None:
    rows: List[Dict[str, object]] = []
    for result in results:
        test = result["test"]
        rows.append({
            "experiment": result["experiment"],
            "selection_rule": result["selection_rule"],
            "selected_epoch": result["selected_epoch"],
            "train_count": result["train_count"],
            "validation_count": result["validation_count"],
            "test_count": result["test_count"],
            "dice_f1": test["f1_dice"], "iou": test["iou"],
            "precision": test["precision"], "recall": test["recall"],
            "accuracy": test["accuracy"], "specificity": test["specificity"],
            "tolerant_f1": test["tolerant_f1"],
            "per_image_dice_mean": test["per_image_dice_mean"],
            "per_image_dice_std": test["per_image_dice_std"],
            "elapsed_minutes": result["elapsed_minutes"],
        })
    csv_path = output_root / "all_experiments_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    save_json(rows, output_root / "all_experiments_summary.json")

    print("\n" + "=" * 78)
    print("Cross-survey F3/CRACKS expert test results")
    print(f"{'Experiment':<15}{'Dice/F1':>11}{'IoU':>10}{'Precision':>12}{'Recall':>10}{'TolF1':>10}")
    for row in rows:
        print(f"{row['experiment']:<15}{row['dice_f1']:>11.4f}{row['iou']:>10.4f}{row['precision']:>12.4f}{row['recall']:>10.4f}{row['tolerant_f1']:>10.4f}")
    print("=" * 78)
    print(f"汇总文件：{csv_path}")


def run(args) -> None:
    device = resolve_device(args.device)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed, args.deterministic)

    test_dataset = build_eval_dataset(
        args.real_test_images, args.real_test_labels,
        "f3_cracks_expert", args.expected_test_count, args,
        native_size=args.test_native_size, mask_mode=args.test_mask_mode,
        image_mode=args.test_image_mode,
    )

    val_images = str(args.real_val_images).strip()
    val_labels = str(args.real_val_labels).strip()
    if bool(val_images) != bool(val_labels):
        raise ValueError("real_val_images与real_val_labels必须同时提供或同时留空。")
    if val_images:
        val_dataset = build_eval_dataset(
            val_images, val_labels, "real_validation",
            args.expected_val_count, args,
        )
    else:
        val_dataset = None
        print("[warning] No independent validation set provided: train for a fixed number of epochs and use the model from the last epoch.")

    initial_state = create_common_initial_state(args, output_root)
    experiments = [x.strip().lower() for x in args.experiments.split(",") if x.strip()]
    results = []
    for experiment in experiments:
        results.append(train_experiment(
            experiment, args, device, initial_state, test_dataset, val_dataset
        ))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_summary(results, output_root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="真实/SLoG/LoG/Canny下游断层检测公平比较",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiments", default="real,slog,log,canny")
    p.add_argument("--output_root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=35)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite_experiment", action="store_true")
    p.add_argument("--recreate_initial_state", action="store_true")

    p.add_argument("--real_train_images",required=True)
    p.add_argument("--real_train_labels",required=True)
    p.add_argument("--real_val_images",required=True)
    p.add_argument("--real_val_labels",required=True)
    p.add_argument("--real_test_images",required=True)
    p.add_argument("--real_test_labels",required=True)

    p.add_argument("--slog_train_images",required=True)
    p.add_argument("--slog_train_labels",required=True)
    p.add_argument("--log_train_images",required=True)
    p.add_argument("--log_train_labels",required=True)
    p.add_argument("--canny_train_images",required=True)
    p.add_argument("--canny_train_labels",required=True)

    p.add_argument("--expected_train_count", type=int, default=350)
    p.add_argument("--expected_val_count", type=int, default=50)
    p.add_argument("--expected_test_count", type=int, default=40)
    p.add_argument("--img_size", type=int, default=512, help="Thebe train/validation size; F3 test can stay native")
    p.add_argument("--test_native_size", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep F3/CRACKS test images at native size instead of resizing to img_size")
    p.add_argument("--test_image_mode", choices=["thebe_grayscale", "cracks_seismic"], default="cracks_seismic",
                   help="CRACKS mode inverses the rendered Matplotlib seismic colormap instead of collapsing RGB to luminance")
    p.add_argument("--test_mask_mode", choices=["thebe", "cracks_expert"], default="cracks_expert",
                   help="CRACKS expert mode: blue=confident fault, orange=confident non-fault; green/white ignored")
    p.add_argument("--mask_threshold_u8", type=int, default=38, help="Thebe train/validation mask binarization threshold >=38; not used for CRACKS expert labels")
    p.add_argument("--disable_mask_audit", action="store_true", help="关闭训练前标签灰度与正类比例检查。")
    p.add_argument("--allow_resize", action="store_true")
    p.add_argument("--allow_same_stem_across_splits", action="store_true")

    p.add_argument("--disable_augmentation", action="store_true")
    p.add_argument("--hflip_prob", type=float, default=0.5)
    p.add_argument("--intensity_aug_prob", type=float, default=0.25)

    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4, help="Thebe validation batch size")
    p.add_argument("--test_batch_size", type=int, default=1, help="F3 native-size test batch size; 1 is safest")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--early_stopping_patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--disable_auto_pos_weight", action="store_true")
    p.add_argument("--pos_weight", type=float, default=5.0)
    p.add_argument("--max_pos_weight", type=float, default=20.0)

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tolerance_radius", type=int, default=3)
    p.add_argument("--preview_count", type=int, default=16)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
