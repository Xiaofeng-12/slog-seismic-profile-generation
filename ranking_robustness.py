"""
Ranking robustness analysis for SLoG / LoG / Canny checkpoint comparisons.

Purpose
-------
This script tests whether the cross-prior conclusion depends on the internally
chosen checkpoint criterion (e.g., total phi FD/RR), a particular late checkpoint,
or a particular weighting of phi subspaces.

It supports two levels of evidence:
1) Development-log robustness (works immediately with checkpoint-wise logs).
2) Held-out final-pool robustness (recommended): provide a second table containing
   metrics for the candidate checkpoints evaluated on the SAME fixed final pool.

"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


CANONICAL_METRICS = ["phi_fdrr", "geo_kid", "swd", "structtex", "spectrum", "energy"]
PRIMARY_SELECTION_METRICS = ["phi_fdrr", "geo_kid", "swd"]
PHI_GROUPS = ["structtex", "spectrum", "energy"]

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "prior": ["prior", "edge", "edge_type", "method", "condition", "operator", "algorithm"],
    "seed": ["seed", "train_seed", "training_seed", "model_seed", "run_seed"],
    "step": ["step", "steps", "global_step", "iteration", "iter", "checkpoint_step", "ckpt_step"],
    "phi_fdrr": [
        "phi_fdrr", "phi_fd_rr", "total_phi_fdrr", "total_phi_fd_rr", "total_fdrr",
        "total_fd_rr", "phi_total_fdrr", "phi_total_fd_rr", "fd_rr", "fdrr", "rrfd",
    ],
    "geo_kid": ["geo_kid", "geokid", "kid_geo", "geo_kid_mean", "geo_kid_m2", "geo_mmd2"],
    "swd": ["swd", "sliced_wasserstein", "sliced_wasserstein_distance", "swd_mean"],
    "structtex": [
        "structtex", "struct_tex", "structtex_fdrr", "structtex_fd_rr", "phi_structtex",
        "phi_structtex_fdrr", "phi_structtex_fd_rr", "structure_texture_fdrr",
    ],
    "spectrum": [
        "spectrum", "spectrum_fdrr", "spectrum_fd_rr", "phi_spectrum", "phi_spectrum_fdrr",
        "phi_spectrum_fd_rr", "spectral_fdrr",
    ],
    "energy": [
        "energy", "energy_fdrr", "energy_fd_rr", "phi_energy", "phi_energy_fdrr",
        "phi_energy_fd_rr", "amplitude_energy_fdrr",
    ],
}

PRIOR_CANON = {
    "slog": "SLoG",
    "s-log": "SLoG",
    "s_log": "SLoG",
    "log": "LoG",
    "canny": "Canny",
}


def normalize_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def canonical_prior(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if "slog" in s or "s-log" in s or "s_log" in s:
        return "SLoG"
    if "canny" in s:
        return "Canny"
    # Match LoG only after SLoG to avoid substring ambiguity.
    if re.search(r"(^|[^a-z])log([^a-z]|$)", s) or s == "log":
        return "LoG"
    return str(x).strip()


def infer_prior_seed_from_name(path: Path) -> Tuple[Optional[str], Optional[int]]:
    name = str(path).lower().replace("\\", "/")
    m = re.search(r"(?:^|/|_)(slog|log|canny)[_\-]?(?:seed|s)?[_\-]?([0-9]{1,5})(?:/|_|$)", name)
    if not m:
        m = re.search(r"train[_\-]?(slog|log|canny)[_\-]?([0-9]{1,5})", name)
    if m:
        prior = canonical_prior(m.group(1))
        return prior, int(m.group(2))
    prior = canonical_prior(name)
    if prior not in {"SLoG", "LoG", "Canny"}:
        prior = None
    m2 = re.search(r"(?:seed|s)[_\-]?([0-9]{1,5})", name)
    seed = int(m2.group(1)) if m2 else None
    return prior, seed


def _safe_ratio(num, den):
    try:
        num = float(num)
        den = float(den)
        if np.isfinite(num) and np.isfinite(den) and abs(den) > 0:
            return num / den
    except Exception:
        pass
    return np.nan


def parse_training_metrics_jsonl_text(text: str, source: Path) -> pd.DataFrame:
    """Parse the user's checkpoint-wise metrics_eval.jsonl format.

    Important behavior:
    - only full evaluator records (those containing SWD / phi_fd) are used;
    - nested SWD Total is flattened;
    - StructTex/Spectrum/Energy FD/RR are reconstructed as phi_fd / rr_fd_thr;
    - when one checkpoint was re-evaluated multiple times,
      the LATEST timestamp is retained rather than averaging old/new evaluator runs.
    """
    prior, seed = infer_prior_seed_from_name(source)
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "step" not in obj:
            continue
        # The file contains a short RR-control row and then the full evaluator row.
        # Keep only the latter.
        if not any(k in obj for k in ["swd", "geo_kid_mean", "phi_fd", "rr_fd_ratio_total"]):
            continue
        swd_obj = obj.get("swd") or {}
        phi = obj.get("phi_fd") or {}
        rr = obj.get("rr_fd_thr") or {}
        row = {
            "prior": prior,
            "seed": seed,
            "step": obj.get("step"),
            "phi_fdrr": obj.get("rr_fd_ratio_total", obj.get("rrfd_total_ratio")),
            "geo_kid": obj.get("geo_kid_mean"),
            "swd": swd_obj.get("Total") if isinstance(swd_obj, dict) else swd_obj,
            "structtex": _safe_ratio(phi.get("StructTex"), rr.get("StructTex")),
            "spectrum": _safe_ratio(phi.get("Spectrum"), rr.get("Spectrum")),
            "energy": _safe_ratio(phi.get("Energy"), rr.get("Energy")),
            "ts": obj.get("ts", ""),
            "source_file": str(source),
            "line_no": line_no,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ["seed", "step"] + CANONICAL_METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Prefer the latest re-evaluation for duplicate checkpoint rows. ISO timestamps
    # sort lexicographically; line_no provides a deterministic fallback.
    df = df.sort_values(["prior", "seed", "step", "ts", "line_no"])
    df = df.drop_duplicates(["prior", "seed", "step"], keep="last")
    return df


def parse_training_metrics_jsonl(path: Path) -> pd.DataFrame:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pd.DataFrame()
    return parse_training_metrics_jsonl_text(text, path)


def find_alias_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    norm_to_orig = {normalize_name(c): c for c in df.columns}
    for alias in COLUMN_ALIASES[canonical]:
        n = normalize_name(alias)
        if n in norm_to_orig:
            return norm_to_orig[n]
    # Fuzzy containment, useful for columns like eval_total_phi_fdrr_mean.
    for n, orig in norm_to_orig.items():
        for alias in COLUMN_ALIASES[canonical]:
            a = normalize_name(alias)
            if a and (n.endswith(a) or n.startswith(a) or f"_{a}_" in f"_{n}_"):
                return orig
    return None


def standardize_dataframe(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canonical in ["prior", "seed", "step"] + CANONICAL_METRICS:
        col = find_alias_column(df, canonical)
        if col is not None:
            out[canonical] = df[col]

    inferred_prior, inferred_seed = infer_prior_seed_from_name(source)
    if "prior" not in out.columns and inferred_prior is not None:
        out["prior"] = inferred_prior
    if "seed" not in out.columns and inferred_seed is not None:
        out["seed"] = inferred_seed

    if "prior" in out.columns:
        out["prior"] = out["prior"].map(canonical_prior)
    if "seed" in out.columns:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    if "step" in out.columns:
        out["step"] = pd.to_numeric(out["step"], errors="coerce")
    for m in CANONICAL_METRICS:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")

    out["source_file"] = str(source)
    return out


def parse_key_value_log(path: Path) -> pd.DataFrame:
    """Best-effort parser for text logs containing step + key/value metrics."""
    rows: List[dict] = []
    inferred_prior, inferred_seed = infer_prior_seed_from_name(path)
    step_re = re.compile(r"(?:global[_ ]?step|step|iter(?:ation)?)\s*[:= ]\s*([0-9]+)", re.I)

    alias_patterns: Dict[str, re.Pattern] = {}
    for metric in CANONICAL_METRICS:
        aliases = sorted(COLUMN_ALIASES[metric], key=len, reverse=True)
        token = "|".join(re.escape(a).replace("_", r"[_\s\-/]*") for a in aliases)
        alias_patterns[metric] = re.compile(
            rf"(?:{token})\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I
        )

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sm = step_re.search(line)
            if not sm:
                continue
            row = {"step": int(sm.group(1)), "prior": inferred_prior, "seed": inferred_seed}
            found = 0
            for metric, pat in alias_patterns.items():
                mm = pat.search(line)
                if mm:
                    row[metric] = float(mm.group(1))
                    found += 1
            if found:
                rows.append(row)

    return pd.DataFrame(rows)


def read_one_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return standardize_dataframe(pd.read_csv(path), path)
        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(path, sheet_name=None)
            frames = [standardize_dataframe(v, path) for v in sheets.values()]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if suffix == ".jsonl":
            # Special handling for the real training evaluator logs.
            if path.name.lower() == "metrics_eval.jsonl":
                special = parse_training_metrics_jsonl(path)
                if not special.empty:
                    return special
            return standardize_dataframe(pd.read_json(path, lines=True), path)
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return standardize_dataframe(pd.DataFrame(obj), path)
            if isinstance(obj, dict):
                # Common patterns: {"records": [...]}, or a single record.
                for key in ["records", "rows", "metrics", "history", "data"]:
                    if key in obj and isinstance(obj[key], list):
                        return standardize_dataframe(pd.DataFrame(obj[key]), path)
                return standardize_dataframe(pd.DataFrame([obj]), path)
        if suffix in {".txt", ".log", ".out"}:
            df = parse_key_value_log(path)
            if not df.empty:
                df["source_file"] = str(path)
            return df
    except Exception as exc:
        print(f"[WARN] Failed to parse {path}: {exc}", file=sys.stderr)
    return pd.DataFrame()


def collect_input(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    # Direct ZIP support for the user's exported train/ tree. This means the user
    # does not need to extract the archive or manually build a CSV.
    if path.is_file() and path.suffix.lower() == ".zip":
        frames = []
        with zipfile.ZipFile(path, "r") as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith("metrics_eval.jsonl"))
            for name in names:
                try:
                    text = zf.read(name).decode("utf-8", errors="replace")
                except Exception as exc:
                    print(f"[WARN] Failed reading {name} from ZIP: {exc}", file=sys.stderr)
                    continue
                pseudo = Path(name)
                d = parse_training_metrics_jsonl_text(text, pseudo)
                if not d.empty:
                    frames.append(d)
        if not frames:
            raise ValueError(f"No parseable metrics_eval.jsonl rows found in ZIP {path}")
        return clean_metrics(pd.concat(frames, ignore_index=True, sort=False))

    if path.is_file():
        files = [path]
    else:
        allowed = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt", ".log", ".out"}
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
        # If an extracted training tree contains metrics_eval.jsonl files, prefer
        # those and do not mix in metrics_train/train.log duplicates.
        eval_files = [p for p in files if p.name.lower() == "metrics_eval.jsonl"]
        if eval_files:
            files = eval_files
    frames = []
    for f in files:
        d = read_one_file(f)
        if not d.empty:
            frames.append(d)
    if not frames:
        raise ValueError(f"No parseable metric rows found under {path}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    return clean_metrics(df)


def clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = ["prior", "seed", "step"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Preferred tidy schema: prior, seed, step, "
            "phi_fdrr, geo_kid, swd, structtex, spectrum, energy."
        )
    df = df.copy()
    df["prior"] = df["prior"].map(canonical_prior)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for m in CANONICAL_METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=["prior", "seed", "step"])
    df["seed"] = df["seed"].astype(int)
    df["step"] = df["step"].astype(int)
    df = df[df["prior"].isin(["SLoG", "LoG", "Canny"])]

    metric_cols = [m for m in CANONICAL_METRICS if m in df.columns]
    if not metric_cols:
        raise ValueError("No recognized metric columns were found.")

    # Average repeated measurements of the same checkpoint. This is intentional for
    # logs that contain repeated metric-resampling rows.
    agg = {m: "mean" for m in metric_cols}
    df = (
        df.groupby(["prior", "seed", "step"], as_index=False)
        .agg(agg)
        .sort_values(["prior", "seed", "step"])
    )
    return df


def availability_report(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    metrics = [m for m in CANONICAL_METRICS if m in df.columns]
    rows = []
    for (prior, seed), g in df.groupby(["prior", "seed"]):
        row = {
            "prior": prior,
            "seed": seed,
            "n_checkpoints": len(g),
            "min_step": int(g.step.min()),
            "max_step": int(g.step.max()),
        }
        for m in metrics:
            row[f"n_{m}"] = int(g[m].notna().sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"{label}_availability.csv", index=False)


def choose_best_step(g: pd.DataFrame, metric: str) -> Optional[int]:
    if metric not in g.columns or g[metric].notna().sum() == 0:
        return None
    # Lower is better for all supported metrics.
    idx = g[metric].idxmin()
    return int(g.loc[idx, "step"])


def choose_mean_rank_step(g: pd.DataFrame, metrics: Sequence[str]) -> Optional[int]:
    use = [m for m in metrics if m in g.columns and g[m].notna().sum() > 0]
    if len(use) < 2:
        return None
    x = g[["step"] + use].copy()
    for m in use:
        x[f"rank_{m}"] = x[m].rank(method="average", ascending=True)
    rank_cols = [f"rank_{m}" for m in use]
    x["mean_rank"] = x[rank_cols].mean(axis=1, skipna=True)
    x = x.dropna(subset=["mean_rank"])
    if x.empty:
        return None
    return int(x.loc[x["mean_rank"].idxmin(), "step"])


def build_selection_table(dev: pd.DataFrame, fixed_steps: Sequence[int]) -> pd.DataFrame:
    rows = []
    for (prior, seed), g in dev.groupby(["prior", "seed"]):
        for metric in PRIMARY_SELECTION_METRICS:
            s = choose_best_step(g, metric)
            if s is not None:
                rows.append({"prior": prior, "seed": seed, "selection_rule": f"min_{metric}", "step": s})
        s = choose_mean_rank_step(g, PRIMARY_SELECTION_METRICS)
        if s is not None:
            rows.append({"prior": prior, "seed": seed, "selection_rule": "min_mean_rank", "step": s})
        latest = int(g.step.max())
        rows.append({"prior": prior, "seed": seed, "selection_rule": "latest_available", "step": latest})
        available_steps = set(int(v) for v in g.step.unique())
        for fs in fixed_steps:
            if fs in available_steps:
                rows.append({"prior": prior, "seed": seed, "selection_rule": f"fixed_{fs}", "step": fs})
    return pd.DataFrame(rows).sort_values(["selection_rule", "prior", "seed"])


def complete_rules(selection: pd.DataFrame, expected_groups: pd.DataFrame) -> List[str]:
    expected = set(map(tuple, expected_groups[["prior", "seed"]].drop_duplicates().to_records(index=False)))
    good = []
    for rule, g in selection.groupby("selection_rule"):
        got = set(map(tuple, g[["prior", "seed"]].drop_duplicates().to_records(index=False)))
        if got == expected:
            good.append(rule)
    return sorted(good)


def join_selected_metrics(selection: pd.DataFrame, metrics_df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    merged = selection.merge(metrics_df, on=["prior", "seed", "step"], how="left", validate="many_to_one")
    merged["metric_source"] = source_label
    return merged


def summarize_ranking(selected_metrics: pd.DataFrame, out_dir: Path, prefix: str) -> pd.DataFrame:
    metric_cols = [m for m in CANONICAL_METRICS if m in selected_metrics.columns]
    rows = []
    for rule, rg in selected_metrics.groupby("selection_rule"):
        for metric in metric_cols:
            summary = rg.groupby("prior")[metric].agg(["mean", "std", "count"]).reset_index()
            summary = summary.dropna(subset=["mean"])
            if summary.empty:
                continue
            summary["rank"] = summary["mean"].rank(method="min", ascending=True).astype(int)
            for _, r in summary.iterrows():
                rows.append({
                    "selection_rule": rule,
                    "outcome_metric": metric,
                    "prior": r["prior"],
                    "mean": r["mean"],
                    "std": r["std"],
                    "n_seeds": int(r["count"]),
                    "rank": int(r["rank"]),
                })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{prefix}_ranking_by_selection_rule.csv", index=False)
    if not out.empty:
        winners = out[out["rank"] == 1].groupby(["outcome_metric", "prior"]).size().rename("n_rule_wins").reset_index()
        totals = out[["selection_rule", "outcome_metric"]].drop_duplicates().groupby("outcome_metric").size().rename("n_rules").reset_index()
        winners = winners.merge(totals, on="outcome_metric", how="left")
        winners["win_fraction"] = winners["n_rule_wins"] / winners["n_rules"]
        winners.to_csv(out_dir / f"{prefix}_winner_frequency.csv", index=False)
    return out


def plot_metric_trajectories(dev: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot trajectories only at checkpoints shared by ALL prior × training-seed runs.

    This avoids extending a prior trajectory into late steps where fewer
    independently trained models remain available.
    """

    # ---------------------------------------------------------
    # Find checkpoint steps shared by every prior × seed run
    # ---------------------------------------------------------
    all_step_sets = []

    for (prior, seed), g in dev.groupby(["prior", "seed"]):
        steps = set(g["step"].dropna().astype(int).tolist())
        if steps:
            all_step_sets.append(steps)

    if not all_step_sets:
        print("[WARN] No prior/seed groups available for trajectory plotting.")
        return

    common_steps = sorted(set.intersection(*all_step_sets))

    if not common_steps:
        print("[WARN] No checkpoint steps are shared by all prior/seed runs.")
        return

    print(
        f"[Trajectory] Using {len(common_steps)} common checkpoints: "
        f"{common_steps[0]} - {common_steps[-1]}"
    )

    # Restrict trajectory plots to strictly comparable checkpoints
    plot_dev = dev[dev["step"].isin(common_steps)].copy()

    # ---------------------------------------------------------
    # Plot each metric
    # ---------------------------------------------------------
    metrics = [m for m in CANONICAL_METRICS if m in plot_dev.columns]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        plotted = False

        for prior, g in plot_dev.groupby("prior"):

            s = (
                g.groupby("step")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("step")
            )

            s = s.dropna(subset=["mean"])

            if s.empty:
                continue

            line, = ax.plot(
                s["step"],
                s["mean"],
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                label=prior,
            )

            if s["std"].notna().any():
                lo = s["mean"] - s["std"].fillna(0)
                hi = s["mean"] + s["std"].fillna(0)

                ax.fill_between(
                    s["step"],
                    lo,
                    hi,
                    alpha=0.15,
                    color=line.get_color(),
                )

            plotted = True

        if plotted:
            ax.set_xlabel("Training step")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"Late-window trajectory: {metric}")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)

            # Force x-axis to exactly the shared comparison window
            ax.set_xlim(common_steps[0], common_steps[-1])

            fig.tight_layout()

            fig.savefig(
                out_dir / f"trajectory_{metric}.png",
                dpi=300,
            )
            fig.savefig(
                out_dir / f"trajectory_{metric}.pdf",
            )

        plt.close(fig)


def plot_ranking_by_rule(ranking: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if ranking.empty:
        return
    for metric, g in ranking.groupby("outcome_metric"):
        rules = list(dict.fromkeys(g["selection_rule"].tolist()))
        x = np.arange(len(rules))
        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(rules)), 4.8))
        plotted = False
        for prior, pg in g.groupby("prior"):
            pg = pg.set_index("selection_rule").reindex(rules)
            y = pg["mean"].to_numpy(dtype=float)
            yerr = pg["std"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5, label=prior)
            plotted = True
        if plotted:
            ax.set_xticks(x)
            ax.set_xticklabels(rules, rotation=35, ha="right")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"Ranking robustness across checkpoint rules: {metric}")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_ranking_robustness_{metric}.png", dpi=300)
            fig.savefig(out_dir / f"{prefix}_ranking_robustness_{metric}.pdf")
        plt.close(fig)


def selection_agreement(selection: pd.DataFrame, out_dir: Path) -> None:
    pivot = selection.pivot_table(index=["prior", "seed"], columns="selection_rule", values="step", aggfunc="first")
    rules = list(pivot.columns)
    rows = []
    for a in rules:
        for b in rules:
            if a == b:
                valid = pivot[[a]].dropna()
                exact = 1.0 if not valid.empty else np.nan
                mean_abs = 0.0 if not valid.empty else np.nan
            else:
                valid = pivot[[a, b]].dropna()
                if valid.empty:
                    exact = np.nan
                    mean_abs = np.nan
                else:
                    exact = float((valid[a] == valid[b]).mean())
                    mean_abs = float((valid[a] - valid[b]).abs().mean())
            rows.append({"rule_a": a, "rule_b": b, "exact_agreement": exact, "mean_abs_step_difference": mean_abs, "n": len(valid)})
    pd.DataFrame(rows).to_csv(out_dir / "selection_rule_agreement.csv", index=False)
    pivot.reset_index().to_csv(out_dir / "selected_steps_wide.csv", index=False)


def metric_correlations(dev: pd.DataFrame, out_dir: Path) -> None:
    metrics = [m for m in CANONICAL_METRICS if m in dev.columns]
    rows = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            d = dev[[a, b]].dropna()
            if len(d) < 3:
                continue
            if spearmanr is not None:
                rho, p = spearmanr(d[a], d[b])
            else:
                rho = d[a].rank().corr(d[b].rank())
                p = np.nan
            rows.append({"metric_a": a, "metric_b": b, "spearman_rho": rho, "p_value": p, "n": len(d)})
    pd.DataFrame(rows).to_csv(out_dir / "metric_spearman_correlations.csv", index=False)


def common_step_ranking(dev: pd.DataFrame, out_dir: Path) -> None:
    groups = dev[["prior", "seed"]].drop_duplicates()
    all_sets = []
    for _, r in groups.iterrows():
        steps = set(dev[(dev.prior == r.prior) & (dev.seed == r.seed)].step.tolist())
        all_sets.append(steps)
    if not all_sets:
        return
    common = sorted(set.intersection(*all_sets))
    if not common:
        return
    metrics = [m for m in CANONICAL_METRICS if m in dev.columns]
    rows = []
    for step in common:
        d = dev[dev.step == step]
        for metric in metrics:
            s = d.groupby("prior")[metric].agg(["mean", "std", "count"]).reset_index().dropna(subset=["mean"])
            if s.empty:
                continue
            s["rank"] = s["mean"].rank(method="min", ascending=True).astype(int)
            for _, r in s.iterrows():
                rows.append({"step": step, "metric": metric, "prior": r.prior, "mean": r["mean"], "std": r["std"], "n_seeds": int(r["count"]), "rank": int(r["rank"])})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "common_step_rankings.csv", index=False)
    if not out.empty:
        win = out[out["rank"] == 1].groupby(["metric", "prior"]).size().rename("n_step_wins").reset_index()
        totals = out[["metric", "step"]].drop_duplicates().groupby("metric").size().rename("n_common_steps").reset_index()
        win = win.merge(totals, on="metric", how="left")
        win["win_fraction"] = win["n_step_wins"] / win["n_common_steps"]
        win.to_csv(out_dir / "common_step_winner_frequency.csv", index=False)


def subgroup_preference_sensitivity(selected: pd.DataFrame, out_dir: Path, label: str) -> None:
    if not all(m in selected.columns for m in PHI_GROUPS):
        return
    base = selected.groupby("prior")[PHI_GROUPS].mean().dropna()
    if len(base) < 2:
        return
    base.to_csv(out_dir / f"{label}_subgroup_means.csv")

    # 1D slice: Energy weight varies from 0 to 1; StructTex and Spectrum split the remainder equally.
    ews = np.linspace(0, 1, 201)
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for prior, r in base.iterrows():
        scores = []
        for we in ews:
            ws = (1 - we) / 2
            wf = (1 - we) / 2
            score = ws * r["structtex"] + wf * r["spectrum"] + we * r["energy"]
            scores.append(score)
            rows.append({"prior": prior, "w_structtex": ws, "w_spectrum": wf, "w_energy": we, "weighted_score": score})
        ax.plot(ews, scores, linewidth=1.7, label=prior)
    ax.set_xlabel("Energy weight")
    ax.set_ylabel("Exploratory weighted subgroup FD/RR")
    ax.set_title("Preference sensitivity: increasing emphasis on Energy")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_energy_weight_sensitivity.png", dpi=300)
    fig.savefig(out_dir / f"{label}_energy_weight_sensitivity.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / f"{label}_energy_weight_sensitivity.csv", index=False)

    # Full simplex winner map. This is exploratory and does NOT redefine total phi FD/RR.
    simplex_rows = []
    grid = np.linspace(0, 1, 51)
    prior_names = list(base.index)
    for wst in grid:
        for wsp in grid:
            we = 1.0 - wst - wsp
            if we < -1e-12:
                continue
            we = max(0.0, we)
            scores = {
                p: wst * base.loc[p, "structtex"] + wsp * base.loc[p, "spectrum"] + we * base.loc[p, "energy"]
                for p in prior_names
            }
            winner = min(scores, key=scores.get)
            # Ternary-to-Cartesian transform: vertices ST=(0,0), SP=(1,0), E=(0.5,sqrt(3)/2)
            x = wsp + 0.5 * we
            y = (math.sqrt(3) / 2) * we
            simplex_rows.append({"w_structtex": wst, "w_spectrum": wsp, "w_energy": we, "x": x, "y": y, "winner": winner, **{f"score_{p}": v for p, v in scores.items()}})
    simp = pd.DataFrame(simplex_rows)
    simp.to_csv(out_dir / f"{label}_phi_preference_simplex.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    # Use Matplotlib default color cycle; no hard-coded journal colors.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    cmap = {p: cycle[i % len(cycle)] if cycle else None for i, p in enumerate(prior_names)}
    for p in prior_names:
        q = simp[simp.winner == p]
        ax.scatter(q.x, q.y, s=10, alpha=0.75, label=p, color=cmap[p])
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, math.sqrt(3)/2, 0]
    ax.plot(tri_x, tri_y, linewidth=1)
    ax.text(-0.02, -0.03, "StructTex", ha="right", va="top")
    ax.text(1.02, -0.03, "Spectrum", ha="left", va="top")
    ax.text(0.5, math.sqrt(3)/2 + 0.03, "Energy", ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Exploratory phi-subspace preference sensitivity")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_phi_preference_simplex.png", dpi=300)
    fig.savefig(out_dir / f"{label}_phi_preference_simplex.pdf")
    plt.close(fig)

    freq = simp.groupby("winner").size().rename("grid_points").reset_index()
    freq["fraction"] = freq["grid_points"] / len(simp)
    freq.to_csv(out_dir / f"{label}_phi_preference_winner_fraction.csv", index=False)


def write_summary(out_dir: Path, dev: pd.DataFrame, selection: pd.DataFrame, final: Optional[pd.DataFrame]) -> None:
    metrics = [m for m in CANONICAL_METRICS if m in dev.columns]
    with (out_dir / "README_RESULTS.txt").open("w", encoding="utf-8") as f:
        f.write("RANKING ROBUSTNESS OUTPUT\n")
        f.write("=========================\n\n")
        f.write("Interpretation rule: lower is better for all analyzed discrepancy metrics.\n")
        f.write("The phi subgroup weighting analysis is exploratory and must not be presented as a replacement for total phi FD/RR.\n\n")
        f.write(f"Development rows after cleaning: {len(dev)}\n")
        f.write(f"Development metrics available: {', '.join(metrics)}\n")
        f.write(f"Selection rows: {len(selection)}\n")
        f.write(f"Held-out final metrics supplied: {'yes' if final is not None else 'no'}\n\n")
        f.write("Most important files for the paper:\n")
        f.write("- trajectory_*.png/pdf: late-window metric trajectories across priors.\n")
        f.write("- selected_steps_wide.csv: checkpoints selected by each rule.\n")
        f.write("- selection_rule_agreement.csv: how much the checkpoint rules disagree.\n")
        f.write("- common_step_rankings.csv: prior ordering at identical late checkpoints.\n")
        if final is not None:
            f.write("- final_ranking_by_selection_rule.csv: PRIMARY robustness result; alternate dev selection rules evaluated on the held-out final pool.\n")
            f.write("- final_ranking_robustness_*.png/pdf: paper-ready selection-rule sensitivity curves.\n")
        else:
            f.write("- development_ranking_by_selection_rule.csv: exploratory only; it reuses development metrics and should not be the final anti-bias evidence.\n")
        f.write("- *_energy_weight_sensitivity.* and *_phi_preference_simplex.*: preference-sensitivity diagnostics.\n")


def parse_fixed_steps(text: str) -> List[int]:
    if not text.strip():
        return []
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return sorted(set(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpoint/evaluator ranking robustness for SLoG, LoG, and Canny.")
    ap.add_argument("--input", required=True, help="Development metric file or directory.")
    ap.add_argument("--final-metrics", default=None, help="Optional held-out final-pool per-checkpoint metric file or directory.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--min-step", type=int, default=10000, help="Analyze checkpoints at or after this step.")
    ap.add_argument(
        "--fixed-steps",
        default="20000",
        help="Comma-separated fixed checkpoint steps to include when present. Default is 20000 because it is available for all nine supplied training runs; late-window same-step curves still use all logged checkpoints.",
    )
    ap.add_argument(
        "--preference-selection-rule",
        default="min_phi_fdrr",
        help="Selection rule used as the base checkpoint set for phi-subspace preference sensitivity.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = collect_input(args.input)
    dev = dev[dev.step >= args.min_step].copy()
    if dev.empty:
        raise ValueError(f"No development rows remain after --min-step {args.min_step}.")
    dev.to_csv(out_dir / "development_metrics_clean.csv", index=False)
    availability_report(dev, out_dir, "development")

    fixed_steps = parse_fixed_steps(args.fixed_steps)
    selection = build_selection_table(dev, fixed_steps)
    expected_groups = dev[["prior", "seed"]].drop_duplicates()
    good_rules = complete_rules(selection, expected_groups)
    selection = selection[selection.selection_rule.isin(good_rules)].copy()
    selection.to_csv(out_dir / "selected_checkpoints.csv", index=False)
    candidates = selection[["prior", "seed", "step"]].drop_duplicates().sort_values(["prior", "seed", "step"])
    candidates.to_csv(out_dir / "candidate_checkpoints_to_evaluate.csv", index=False)

    selection_agreement(selection, out_dir)
    metric_correlations(dev, out_dir)
    common_step_ranking(dev, out_dir)
    plot_metric_trajectories(dev, out_dir)

    dev_selected = join_selected_metrics(selection, dev, "development")
    dev_selected.to_csv(out_dir / "development_selected_metrics.csv", index=False)
    dev_ranking = summarize_ranking(dev_selected, out_dir, "development")
    plot_ranking_by_rule(dev_ranking, out_dir, "development")

    pref_rule = args.preference_selection_rule
    pref_dev = dev_selected[dev_selected.selection_rule == pref_rule]
    if not pref_dev.empty:
        subgroup_preference_sensitivity(pref_dev, out_dir, "development")

    final_df = None
    if args.final_metrics:
        final_df = collect_input(args.final_metrics)
        final_df = final_df[final_df.step >= args.min_step].copy()
        final_df.to_csv(out_dir / "final_metrics_clean.csv", index=False)
        availability_report(final_df, out_dir, "final")
        final_selected = join_selected_metrics(selection, final_df, "heldout_final")
        # Report missing re-evaluations explicitly.
        metric_cols = [m for m in CANONICAL_METRICS if m in final_selected.columns]
        final_selected["has_any_final_metric"] = final_selected[metric_cols].notna().any(axis=1) if metric_cols else False
        final_selected.to_csv(out_dir / "final_selected_metrics.csv", index=False)
        missing = final_selected[~final_selected["has_any_final_metric"]][["prior", "seed", "selection_rule", "step"]]
        missing.to_csv(out_dir / "MISSING_FINAL_EVALUATIONS.csv", index=False)

        available = final_selected[final_selected["has_any_final_metric"]].copy()
        final_ranking = summarize_ranking(available, out_dir, "final")
        plot_ranking_by_rule(final_ranking, out_dir, "final")

        pref_final = available[available.selection_rule == pref_rule]
        if not pref_final.empty:
            subgroup_preference_sensitivity(pref_final, out_dir, "final")

    write_summary(out_dir, dev, selection, final_df)
    print(f"Done. Results written to: {out_dir.resolve()}")
    if final_df is None:
        print("NOTE: No --final-metrics were supplied. Development-only outputs are useful diagnostics, but the strongest paper claim requires held-out re-evaluation of the checkpoints selected by alternate rules.")


if __name__ == "__main__":
    main()
