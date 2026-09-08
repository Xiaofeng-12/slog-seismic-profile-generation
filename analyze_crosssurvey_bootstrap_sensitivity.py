"""
Post-hoc analysis for the Thebe -> F3/CRACKS cross-survey fault segmentation experiment.

This script DOES NOT retrain any network.

It:
1) reloads Real / SLoG / LoG / Canny best_model.pth checkpoints;
2) runs the same 40 CRACKS expert sections at native size;
3) evaluates two expert-label definitions:
   primary:
       class 3 (blue)   = fault
       class 1 (orange) = non-fault
       classes 0,2      = ignore
   inclusive_uncertain:
       classes 2,3      = fault
       class 1          = non-fault
       class 0          = ignore
4) saves per-section TP/FP/FN/TN and tolerant matching counts;
5) performs paired section-level bootstrap for SLoG - LoG;
6) if multiple independent U-Net run directories are supplied, also performs
   a hierarchical paired bootstrap over training runs and test sections.

Expected run directory layout:
    crosssurvey_seed42/
        real/best_model.pth
        slog/best_model.pth
        log/best_model.pth
        canny/best_model.pth
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


METHODS_DEFAULT = ("real", "slog", "log", "canny")
SCHEMES = ("primary", "inclusive_uncertain")
EPS = 1e-8


def natural_key(text: str):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def load_base_module(path: str):
    path_obj = Path(path).expanduser().resolve()

    if not path_obj.is_file():
        raise FileNotFoundError(f"base_code not found: {path_obj}")

    spec = importlib.util.spec_from_file_location(
        "crosssurvey_base",
        str(path_obj)
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path_obj}")

    module = importlib.util.module_from_spec(spec)

    import sys
    sys.modules["crosssurvey_base"] = module

    spec.loader.exec_module(module)

    return module


def collect_pngs(directory: str) -> Dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = {}
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            out[p.stem] = p
    return out


def paired_f3_records(image_dir: str, label_dir: str) -> List[Tuple[str, Path, Path]]:
    images = collect_pngs(image_dir)
    labels = collect_pngs(label_dir)
    stems = sorted(set(images) & set(labels), key=natural_key)
    if not stems:
        raise RuntimeError("No F3 image/expert-mask pairs found.")
    missing = sorted(set(labels) - set(images), key=natural_key)
    if missing:
        raise RuntimeError(f"Expert masks without image: {missing[:20]}")
    print(f"[F3 pairing] images={len(images)}, expert_masks={len(labels)}, paired={len(stems)}")
    return [(s, images[s], labels[s]) for s in stems]


def read_cracks_class_indices(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.mode == "P":
        arr = np.asarray(mask, dtype=np.uint8)
    else:
        rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        arr = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        mapping = {
            (255, 255, 255): 0,
            (255, 127, 14): 1,
            (44, 160, 44): 2,
            (31, 119, 180): 3,
        }
        for color, idx in mapping.items():
            arr[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = idx
        if np.any(arr == 255):
            unknown = np.unique(rgb[arr == 255].reshape(-1, 3), axis=0)[:20]
            raise ValueError(f"Unknown CRACKS RGB colors in {mask_path}: {unknown.tolist()}")
    uniq = set(np.unique(arr).tolist())
    if not uniq.issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected class indices in {mask_path}: {sorted(uniq)}")
    return arr


def build_target_valid(arr: np.ndarray, scheme: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if scheme == "primary":
        # blue=confident fault; orange=confident non-fault; white/green ignored
        target = arr == 3
        valid = (arr == 1) | (arr == 3)
    elif scheme == "inclusive_uncertain":
        # blue + green are treated as fault; orange is non-fault; white ignored
        target = (arr == 2) | (arr == 3)
        valid = (arr == 1) | (arr == 2) | (arr == 3)
    else:
        raise ValueError(scheme)
    return (
        torch.from_numpy(target.astype(np.bool_)).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(valid.astype(np.bool_)).unsqueeze(0).unsqueeze(0),
    )


def section_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    tolerance_radius: int,
) -> Dict[str, int]:
    pred = pred.bool()
    target = target.bool()
    valid = valid.bool()

    tp = int((pred & target & valid).sum().item())
    fp = int((pred & ~target & valid).sum().item())
    fn = int((~pred & target & valid).sum().item())
    tn = int((~pred & ~target & valid).sum().item())

    pred_v = pred & valid
    gt_v = target & valid

    if tolerance_radius > 0:
        k = 2 * int(tolerance_radius) + 1
        dil_gt = F.max_pool2d(gt_v.float(), k, 1, int(tolerance_radius)) >= 0.5
        dil_pred = F.max_pool2d(pred_v.float(), k, 1, int(tolerance_radius)) >= 0.5
        tol_pred_match = int((pred_v & dil_gt).sum().item())
        tol_pred_total = int(pred_v.sum().item())
        tol_gt_match = int((gt_v & dil_pred).sum().item())
        tol_gt_total = int(gt_v.sum().item())
    else:
        hit = int((pred_v & gt_v).sum().item())
        tol_pred_match = hit
        tol_pred_total = int(pred_v.sum().item())
        tol_gt_match = hit
        tol_gt_total = int(gt_v.sum().item())

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tol_pred_match": tol_pred_match,
        "tol_pred_total": tol_pred_total,
        "tol_gt_match": tol_gt_match,
        "tol_gt_total": tol_gt_total,
        "valid_pixels": int(valid.sum().item()),
        "fault_pixels": int((target & valid).sum().item()),
    }


def metrics_from_counts(c: Dict[str, float]) -> Dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    specificity = tn / (tn + fp + EPS)

    tol_p = c["tol_pred_match"] / (c["tol_pred_total"] + EPS)
    tol_r = c["tol_gt_match"] / (c["tol_gt_total"] + EPS)
    tol_f1 = 2.0 * tol_p * tol_r / (tol_p + tol_r + EPS)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tolerant_precision": float(tol_p),
        "tolerant_recall": float(tol_r),
        "tolerant_f1": float(tol_f1),
    }


COUNT_KEYS = [
    "tp", "fp", "fn", "tn",
    "tol_pred_match", "tol_pred_total", "tol_gt_match", "tol_gt_total",
    "valid_pixels", "fault_pixels",
]


def sum_counts(rows: List[Dict[str, int]], indices: np.ndarray | None = None) -> Dict[str, float]:
    if indices is None:
        selected = rows
    else:
        selected = [rows[int(i)] for i in indices]
    return {k: float(sum(r[k] for r in selected)) for k in COUNT_KEYS}


def load_model(base, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_saved = ckpt.get("args", {})
    base_channels = int(args_saved.get("base_channels", 32))
    dropout = float(args_saved.get("dropout", 0.10))
    model = base.FaultUNet(base_channels=base_channels, dropout=dropout).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_one_model(
    base,
    model,
    records,
    device: torch.device,
    threshold: float,
    tolerance_radius: int,
) -> Dict[str, List[Dict[str, int]]]:
    output = {scheme: [] for scheme in SCHEMES}

    for stem, image_path, mask_path in records:
        image = Image.open(image_path)
        x = base.decode_cracks_seismic_image(image).unsqueeze(0).to(device)
        logits = model(x)
        pred = torch.sigmoid(logits) >= float(threshold)
        pred_cpu = pred.cpu()

        arr = read_cracks_class_indices(mask_path)
        for scheme in SCHEMES:
            target, valid = build_target_valid(arr, scheme)
            c = section_counts(pred_cpu, target, valid, tolerance_radius)
            c["stem"] = stem
            output[scheme].append(c)
    return output


def percentile_ci(x: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(x, alpha / 2.0))
    hi = float(np.quantile(x, 1.0 - alpha / 2.0))
    return lo, hi


def paired_bootstrap_single_run(
    method_rows: Dict[str, List[Dict[str, int]]],
    compare_a: str,
    compare_b: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    n = len(method_rows[compare_a])
    rng = np.random.default_rng(seed)
    metric_names = ["dice", "iou", "precision", "recall", "tolerant_f1"]
    deltas = {m: np.empty(n_boot, dtype=np.float64) for m in metric_names}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma = metrics_from_counts(sum_counts(method_rows[compare_a], idx))
        mb = metrics_from_counts(sum_counts(method_rows[compare_b], idx))
        for m in metric_names:
            deltas[m][b] = ma[m] - mb[m]

    out = {}
    for m, arr in deltas.items():
        lo, hi = percentile_ci(arr)
        out[m] = {
            "delta_mean": float(arr.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "prob_delta_gt_0": float(np.mean(arr > 0)),
        }
    return out


def hierarchical_paired_bootstrap(
    runs_rows: List[Dict[str, List[Dict[str, int]]]],
    compare_a: str,
    compare_b: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """
    Paired hierarchical bootstrap:
      - resample U-Net training runs with replacement;
      - for each selected run, resample the SAME F3 section indices for A and B;
      - calculate each run's global metric;
      - average metrics across selected runs;
      - store A-B.
    """
    rng = np.random.default_rng(seed)
    n_runs = len(runs_rows)
    n_sections = len(runs_rows[0][compare_a])
    metric_names = ["dice", "iou", "precision", "recall", "tolerant_f1"]
    deltas = {m: np.empty(n_boot, dtype=np.float64) for m in metric_names}

    for b in range(n_boot):
        run_idx = rng.integers(0, n_runs, size=n_runs)
        vals_a = {m: [] for m in metric_names}
        vals_b = {m: [] for m in metric_names}

        for r in run_idx:
            sec_idx = rng.integers(0, n_sections, size=n_sections)
            ma = metrics_from_counts(sum_counts(runs_rows[int(r)][compare_a], sec_idx))
            mb = metrics_from_counts(sum_counts(runs_rows[int(r)][compare_b], sec_idx))
            for m in metric_names:
                vals_a[m].append(ma[m])
                vals_b[m].append(mb[m])

        for m in metric_names:
            deltas[m][b] = float(np.mean(vals_a[m]) - np.mean(vals_b[m]))

    out = {}
    for m, arr in deltas.items():
        lo, hi = percentile_ci(arr)
        out[m] = {
            "delta_mean": float(arr.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "prob_delta_gt_0": float(np.mean(arr > 0)),
        }
    return out


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="Paired bootstrap + CRACKS expert-label sensitivity analysis without retraining."
    )
    p.add_argument("--base_code",required=True,help="Path to downstream_crosssurvey_unet.py")
    p.add_argument("--run_dirs",required=True,help="Comma-separated cross-survey output roots, e.g. seed35,seed39,seed42")
    p.add_argument("--f3_images",required=True)
    p.add_argument("--f3_labels",required=True)
    p.add_argument("--methods", default="real,slog,log,canny")
    p.add_argument("--compare", default="slog,log",help="Paired comparison A,B; default SLoG-LoG")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tolerance_radius", type=int, default=3)
    p.add_argument("--bootstrap_n", type=int, default=10000)
    p.add_argument("--bootstrap_seed", type=int, default=20260821)
    p.add_argument("--output_dir",required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = load_base_module(args.base_code)
    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    compare = [x.strip().lower() for x in args.compare.split(",") if x.strip()]
    if len(compare) != 2:
        raise ValueError("--compare must be A,B")
    for x in compare:
        if x not in methods:
            raise ValueError(f"{x} not in --methods")

    run_dirs = [Path(x.strip()).expanduser().resolve() for x in args.run_dirs.split(",") if x.strip()]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = paired_f3_records(args.f3_images, args.f3_labels)
    print(f"[setup] test sections={len(records)}, threshold={args.threshold}, tolerance={args.tolerance_radius}")

    all_runs = {scheme: [] for scheme in SCHEMES}
    aggregate_rows = []
    per_section_rows = []

    for run_dir in run_dirs:
        print(f"\n=== Run: {run_dir} ===")
        run_scheme_rows = {scheme: {} for scheme in SCHEMES}

        for method in methods:
            ckpt_path = run_dir / method / "best_model.pth"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            model, ckpt = load_model(base, ckpt_path, device)
            selected_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
            print(f"[{method}] checkpoint={ckpt_path} epoch={selected_epoch}")

            result = run_one_model(
                base, model, records, device,
                threshold=args.threshold,
                tolerance_radius=args.tolerance_radius,
            )

            for scheme in SCHEMES:
                run_scheme_rows[scheme][method] = result[scheme]
                agg = metrics_from_counts(sum_counts(result[scheme]))
                aggregate_rows.append({
                    "run_dir": str(run_dir),
                    "method": method,
                    "scheme": scheme,
                    "selected_epoch": selected_epoch,
                    **agg,
                })
                for c in result[scheme]:
                    per_section_rows.append({
                        "run_dir": str(run_dir),
                        "method": method,
                        "scheme": scheme,
                        **c,
                        **metrics_from_counts({k: float(c[k]) for k in COUNT_KEYS}),
                    })

        for scheme in SCHEMES:
            all_runs[scheme].append(run_scheme_rows[scheme])

    save_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    save_csv(out_dir / "per_section_metrics_and_counts.csv", per_section_rows)

    # Mean ± SD across independent U-Net training runs.
    run_summary = []
    for scheme in SCHEMES:
        for method in methods:
            rows = [r for r in aggregate_rows if r["scheme"] == scheme and r["method"] == method]
            for metric in ["dice", "iou", "precision", "recall", "tolerant_f1"]:
                vals = np.asarray([float(r[metric]) for r in rows], dtype=float)
                run_summary.append({
                    "scheme": scheme,
                    "method": method,
                    "metric": metric,
                    "n_training_runs": len(vals),
                    "mean": float(vals.mean()),
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                })
    save_csv(out_dir / "training_seed_summary.csv", run_summary)

    bootstrap_output = {
        "compare": f"{compare[0]}-{compare[1]}",
        "bootstrap_n": args.bootstrap_n,
        "bootstrap_seed": args.bootstrap_seed,
        "n_test_sections": len(records),
        "n_training_runs": len(run_dirs),
        "schemes": {},
    }

    for scheme in SCHEMES:
        scheme_out = {
            "per_run_section_bootstrap": {},
        }
        for run_dir, run_rows in zip(run_dirs, all_runs[scheme]):
            scheme_out["per_run_section_bootstrap"][str(run_dir)] = paired_bootstrap_single_run(
                run_rows, compare[0], compare[1],
                n_boot=args.bootstrap_n,
                seed=args.bootstrap_seed,
            )

        if len(run_dirs) > 1:
            scheme_out["hierarchical_bootstrap"] = hierarchical_paired_bootstrap(
                all_runs[scheme], compare[0], compare[1],
                n_boot=args.bootstrap_n,
                seed=args.bootstrap_seed + 1,
            )
        bootstrap_output["schemes"][scheme] = scheme_out

    save_json(out_dir / "paired_bootstrap_results.json", bootstrap_output)

    print("\n=== Aggregate metrics ===")
    for r in aggregate_rows:
        if r["scheme"] == "primary":
            print(
                f"{Path(r['run_dir']).name:>24s}  {r['method']:<6s} "
                f"Dice={r['dice']:.4f} IoU={r['iou']:.4f} "
                f"P={r['precision']:.4f} R={r['recall']:.4f} "
                f"TolF1={r['tolerant_f1']:.4f}"
            )

    print("\n=== Sensitivity: inclusive uncertain vs primary ===")
    for run_dir in run_dirs:
        for method in methods:
            a = next(r for r in aggregate_rows if r["run_dir"] == str(run_dir)
                     and r["method"] == method and r["scheme"] == "primary")
            b = next(r for r in aggregate_rows if r["run_dir"] == str(run_dir)
                     and r["method"] == method and r["scheme"] == "inclusive_uncertain")
            print(
                f"{Path(run_dir).name:>24s}  {method:<6s} "
                f"Dice primary={a['dice']:.4f} -> inclusive={b['dice']:.4f}; "
                f"TolF1 {a['tolerant_f1']:.4f} -> {b['tolerant_f1']:.4f}"
            )

    print(f"\nSaved to: {out_dir}")
    print("Key files:")
    print("  aggregate_metrics.csv")
    print("  per_section_metrics_and_counts.csv")
    print("  training_seed_summary.csv")
    print("  paired_bootstrap_results.json")


if __name__ == "__main__":
    main()
