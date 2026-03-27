from __future__ import annotations

import copy
import csv
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariantRun:
    label: str
    seed: int
    config_payload: dict[str, Any]


@dataclass(frozen=True)
class RunMetric:
    label: str
    seed: int
    run_dir: Path
    avg_e2e_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float
    cold_start_share: float


def repo_root() -> Path:
    return REPO_ROOT


def resolve_config_path(token: str, *, root: Path | None = None) -> Path:
    base = root or repo_root()
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = base / token
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"config file not found: {token}")
    return candidate


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid yaml root in {path}")
    return raw


def build_variant_configs(
    *,
    base_config: dict[str, Any],
    base_experiment_name: str,
    variant_overrides: list[dict[str, Any]],
    seeds: list[int],
) -> list[VariantRun]:
    variants: list[VariantRun] = []
    for item in variant_overrides:
        label = str(item["label"])
        overrides = dict(item.get("overrides", {}))
        for seed in seeds:
            payload = copy.deepcopy(base_config)
            _deep_merge(payload, overrides)
            experiment = dict(payload.get("experiment", {}) or {})
            experiment["random_seed"] = int(seed)
            experiment["name"] = f"{base_experiment_name}_{label}_seed{seed}"
            payload["experiment"] = experiment
            variants.append(
                VariantRun(
                    label=label,
                    seed=int(seed),
                    config_payload=payload,
                )
            )
    return variants


def run_variant(
    *,
    python_bin: str,
    root: Path,
    variant: VariantRun,
    temp_dir: Path,
    idx: int,
) -> Path:
    config_path = temp_dir / f"variant_{idx:04d}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(variant.config_payload, f, sort_keys=False)

    result = subprocess.run(
        [python_bin, "-m", "simulator.main", "--config", str(config_path)],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    merged_output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode != 0:
        raise RuntimeError(
            f"simulation failed (exit={result.returncode}) for {variant.label}/seed={variant.seed}:\n{merged_output}"
        )
    return _extract_run_dir(merged_output, root)


def run_with_temp_configs(
    *,
    python_bin: str,
    root: Path,
    variants: list[VariantRun],
) -> tuple[list[RunMetric], list[dict[str, Any]]]:
    metrics: list[RunMetric] = []
    failures: list[dict[str, Any]] = []
    # Use repo-local temp workspace to avoid ACL issues from OS temp dirs.
    temp_root = root / ".tmp" / "sim_suite"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"run_{uuid4().hex[:10]}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        for idx, variant in enumerate(variants, start=1):
            print(f"[run {idx}/{len(variants)}] {variant.label} seed={variant.seed}")
            try:
                run_dir = run_variant(
                    python_bin=python_bin,
                    root=root,
                    variant=variant,
                    temp_dir=temp_dir,
                    idx=idx,
                )
                print(f"completed: {run_dir}")
                metrics.append(collect_metrics(run_dir=run_dir, label=variant.label, seed=variant.seed))
            except Exception as exc:  # noqa: BLE001
                print(f"failed: {variant.label} seed={variant.seed}: {exc}")
                failures.append(
                    {
                        "label": variant.label,
                        "seed": variant.seed,
                        "error": str(exc),
                    }
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return metrics, failures


def collect_metrics(*, run_dir: Path, label: str, seed: int) -> RunMetric:
    summary_path = run_dir / "summary.json"
    request_path = run_dir / "request_paths.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json in {run_dir}")
    if not request_path.exists():
        raise FileNotFoundError(f"missing request_paths.csv in {run_dir}")

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    success_rate = float(summary.get("success_rate", 0.0))

    completed_total: list[float] = []
    completed_cold: list[float] = []
    with request_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("completed") != "1":
                continue
            raw_total = row.get("total_latency_ms")
            if raw_total in (None, ""):
                continue
            total_val = float(raw_total)
            completed_total.append(total_val)
            completed_cold.append(float(row.get("cold_start_latency_ms", "0") or 0))
    if not completed_total:
        raise RuntimeError(f"no completed request latency in {run_dir}")

    avg_e2e = statistics.fmean(completed_total)
    p50 = float(summary.get("p50_latency_ms") or _percentile(completed_total, 50))
    p95 = float(summary.get("p95_latency_ms") or _percentile(completed_total, 95))
    p99 = float(summary.get("p99_latency_ms") or _percentile(completed_total, 99))
    cold_start_share = (
        (sum(completed_cold) / sum(completed_total)) if sum(completed_total) > 0 else 0.0
    )

    return RunMetric(
        label=label,
        seed=seed,
        run_dir=run_dir,
        avg_e2e_ms=avg_e2e,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        success_rate=success_rate,
        cold_start_share=cold_start_share,
    )


def aggregate_metrics(metrics: list[RunMetric]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunMetric]] = {}
    for item in metrics:
        grouped.setdefault(item.label, []).append(item)

    rows: list[dict[str, Any]] = []
    for label in sorted(grouped.keys()):
        group = grouped[label]
        avg_mean, avg_std = _mean_std([m.avg_e2e_ms for m in group])
        p50_mean, p50_std = _mean_std([m.p50_ms for m in group])
        p95_mean, p95_std = _mean_std([m.p95_ms for m in group])
        p99_mean, p99_std = _mean_std([m.p99_ms for m in group])
        succ_mean, succ_std = _mean_std([m.success_rate for m in group])
        cold_mean, cold_std = _mean_std([m.cold_start_share for m in group])
        rows.append(
            {
                "label": label,
                "runs": len(group),
                "avg_e2e_ms_mean": avg_mean,
                "avg_e2e_ms_std": avg_std,
                "p50_ms_mean": p50_mean,
                "p50_ms_std": p50_std,
                "p95_ms_mean": p95_mean,
                "p95_ms_std": p95_std,
                "p99_ms_mean": p99_mean,
                "p99_ms_std": p99_std,
                "success_rate_mean": succ_mean,
                "success_rate_std": succ_std,
                "cold_start_share_mean": cold_mean,
                "cold_start_share_std": cold_std,
            }
        )
    return rows


def ensure_results_dir(*, root: Path | None = None, relative: str = "results") -> Path:
    base = root or repo_root()
    out = Path(relative)
    if not out.is_absolute():
        out = base / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = ["label"]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_grouped_e2e(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
    metric_specs: list[tuple[str, str, str]] | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")
    if metric_specs is None:
        metric_specs = [
            ("avg_e2e_ms_mean", "Avg E2E", "#4E79A7"),
            ("p50_ms_mean", "P50", "#59A14F"),
            ("p99_ms_mean", "P99", "#E15759"),
        ]
    if not metric_specs:
        raise RuntimeError("metric_specs must not be empty")

    x = list(range(len(rows)))
    width = min(0.32, 0.82 / len(metric_specs))
    labels = [str(row["label"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.4), 6))
    total_metrics = len(metric_specs)
    all_bars = []
    for idx, (metric_key, display_name, color) in enumerate(metric_specs):
        offset = (idx - (total_metrics - 1) / 2.0) * width
        vals = [float(row[metric_key]) for row in rows]
        bars = ax.bar(
            [v + offset for v in x],
            vals,
            width=width,
            color=color,
            label=display_name,
        )
        all_bars.append(bars)

    for bars in all_bars:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 2.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_progressive_e2e(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")

    ordered = sorted(rows, key=lambda r: int(r.get("stage_order", 999)))
    labels = [str(r.get("stage_id") or r.get("label", "")) for r in ordered]
    x = list(range(len(ordered)))

    avg = [float(r["avg_e2e_ms_mean"]) for r in ordered]
    avg_std = [float(r.get("avg_e2e_ms_std", 0.0) or 0.0) for r in ordered]
    p95 = [float(r["p95_ms_mean"]) for r in ordered]
    p95_std = [float(r.get("p95_ms_std", 0.0) or 0.0) for r in ordered]
    p99 = [float(r["p99_ms_mean"]) for r in ordered]
    p99_std = [float(r.get("p99_ms_std", 0.0) or 0.0) for r in ordered]

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.errorbar(x, avg, yerr=avg_std, marker="o", linewidth=2.0, capsize=3, color="#2D6A4F", label="Avg")
    ax.errorbar(x, p95, yerr=p95_std, marker="s", linewidth=2.0, capsize=3, color="#E76F51", label="P95")
    ax.errorbar(x, p99, yerr=p99_std, marker="^", linewidth=2.0, capsize=3, color="#264653", label="P99")

    for i, row in enumerate(ordered):
        if i == 0:
            continue
        raw = row.get("delta_avg_vs_prev_pct", "")
        if raw in ("", None):
            continue
        delta = float(raw)
        marker = "▼" if delta < 0 else "▲"
        ax.text(
            x[i],
            avg[i] + max(8.0, avg_std[i] + 6.0),
            f"{marker}{abs(delta):.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
        )

    ax.text(
        0.01,
        0.98,
        "Lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#4B5563",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_progressive_e2e_by_scenario(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
    scenario_order: list[str],
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")

    fig, axes = plt.subplots(1, len(scenario_order), figsize=(6.0 * len(scenario_order), 5.6), sharey=True)
    if len(scenario_order) == 1:
        axes = [axes]

    palette = {
        "avg": "#2D6A4F",
        "p95": "#E76F51",
        "p99": "#264653",
    }

    for idx, scenario in enumerate(scenario_order):
        ax = axes[idx]
        srows = [r for r in rows if str(r.get("scenario")) == scenario]
        srows.sort(key=lambda r: int(r.get("stage_order", 999)))
        if not srows:
            ax.set_visible(False)
            continue

        labels = [str(r.get("stage_id") or r.get("stage", "")) for r in srows]
        x = list(range(len(srows)))

        avg = [float(r["avg_e2e_ms_mean"]) for r in srows]
        p95 = [float(r["p95_ms_mean"]) for r in srows]
        p99 = [float(r["p99_ms_mean"]) for r in srows]

        ax.plot(x, avg, marker="o", linewidth=2.0, color=palette["avg"], label="Avg")
        ax.plot(x, p95, marker="s", linewidth=2.0, color=palette["p95"], label="P95")
        ax.plot(x, p99, marker="^", linewidth=2.0, color=palette["p99"], label="P99")

        for i, row in enumerate(srows):
            if i == 0:
                continue
            deltas = [
                (row.get("delta_avg_vs_prev_pct"), avg[i], "#2D6A4F"),
                (row.get("delta_p95_vs_prev_pct"), p95[i], "#E76F51"),
                (row.get("delta_p99_vs_prev_pct"), p99[i], "#264653"),
            ]
            for raw, yv, color in deltas:
                if raw in ("", None):
                    continue
                delta = float(raw)
                ax.text(
                    x[i],
                    yv + 14.0,
                    f"{delta:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    fontweight="bold",
                    color=color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{scenario} load")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Latency (ms)")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title, y=0.975)
    fig.text(
        0.01,
        0.07,
        "Lower is better; labels show signed delta vs previous stage",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ablation_gain(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
    scenario_order: list[str],
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")

    fig, axes = plt.subplots(1, len(scenario_order), figsize=(6.0 * len(scenario_order), 5.6), sharey=True)
    if len(scenario_order) == 1:
        axes = [axes]

    for idx, scenario in enumerate(scenario_order):
        ax = axes[idx]
        srows = [r for r in rows if str(r.get("scenario")) == scenario]
        srows.sort(key=lambda r: int(r.get("stage_order", 999)))
        srows = [r for r in srows if int(r.get("stage_order", 999)) > 0]
        if not srows:
            ax.set_visible(False)
            continue

        labels = [str(r.get("stage_id") or r.get("stage", "")) for r in srows]
        x = list(range(len(srows)))
        width = 0.34

        avg_vals = [float(r.get("delta_avg_vs_g0_pct", 0.0) or 0.0) for r in srows]
        p95_vals = [float(r.get("delta_p95_vs_g0_pct", 0.0) or 0.0) for r in srows]

        avg_colors = ["#2D6A4F" if v <= 0 else "#C44536" for v in avg_vals]
        p95_colors = ["#4C78A8" if v <= 0 else "#F58518" for v in p95_vals]

        bars_avg = ax.bar([v - width / 2 for v in x], avg_vals, width=width, color=avg_colors, label="ΔAvg vs G0")
        bars_p95 = ax.bar([v + width / 2 for v in x], p95_vals, width=width, color=p95_colors, label="ΔP95 vs G0")

        for bars in (bars_avg, bars_p95):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (0.45 if h >= 0 else -0.85),
                    f"{h:.2f}%",
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=8,
                )

        ax.axhline(0, linestyle="--", linewidth=1.0, color="#6B7280")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{scenario} load")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Delta vs G0 (%)")

    legend_handles = [
        Patch(facecolor="#2D6A4F", label="Avg improved (<=0)"),
        Patch(facecolor="#C44536", label="Avg regressed (>0)"),
        Patch(facecolor="#4C78A8", label="P95 improved (<=0)"),
        Patch(facecolor="#F58518", label="P95 regressed (>0)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2)
    fig.suptitle(title, y=0.98)
    fig.text(0.01, 0.02, "Negative values mean better latency", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.91))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_single_metric_bar(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    metric_key: str,
    metric_label: str,
    title: str,
    color: str = "#4E79A7",
    y_label: str = "Value",
    value_scale: float = 1.0,
    value_suffix: str = "",
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")
    if value_scale <= 0:
        raise ValueError("value_scale must be > 0")

    labels = [str(row["label"]) for row in rows]
    values = [float(row[metric_key]) * value_scale for row in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.3), 5.6))
    bars = ax.bar(x, values, color=color, width=0.62)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + (0.8 if y_label.endswith("(%)") else 2.0),
            f"{h:.2f}{value_suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tradeoff(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
    base_point: dict[str, float] | None = None,
    historical_point: dict[str, float] | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")

    metric_rows: list[dict[str, float | str | tuple[float, float]]] = []
    for idx, row in enumerate(rows, start=1):
        metric_rows.append(
            {
                "id": f"T{idx:02d}",
                "label": str(row["label"]),
                "x": float(row["avg_e2e_ms_mean"]),
                "y": float(row["p95_ms_mean"]),
                "xerr": _safe_float(row.get("avg_e2e_ms_std", 0.0)),
                "yerr": _safe_float(row.get("p95_ms_std", 0.0)),
                "css": _safe_float(row.get("cold_start_share_mean", 0.0)),
                "as": _safe_float(row.get("hpwp_alpha_stable", row.get("alpha_stable", 0.0))),
                "rho": _safe_float(row.get("hpwp_rho_mass", row.get("rho_mass", float("nan")))),
                "tau": _safe_float(row.get("hpwp_tau_p", row.get("tau_p", float("nan")))),
                "e": _safe_float(row.get("hpwp_sched_eta_exec", row.get("sched_eta_exec", float("nan")))),
                "h": _safe_float(row.get("hpwp_horizon_alpha", row.get("horizon_alpha", float("nan")))),
                "bh": _safe_float(row.get("hpwp_beta_hi", row.get("beta_hi", float("nan")))),
                "bl": _safe_float(row.get("hpwp_beta_lo", row.get("beta_lo", float("nan")))),
                "beta_key": (
                    _safe_float(row.get("hpwp_beta_hi", row.get("beta_hi", float("nan")))),
                    _safe_float(row.get("hpwp_beta_lo", row.get("beta_lo", float("nan")))),
                ),
            }
        )

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    has_rho = any(not math.isnan(float(item["rho"])) for item in metric_rows)
    has_tau = any(not math.isnan(float(item["tau"])) for item in metric_rows)
    color_values = [
        float(item["rho"]) if has_rho else float(item["as"])
        for item in metric_rows
    ]
    css_values = [float(item["css"]) for item in metric_rows]
    color_min = min(color_values)
    color_max = max(color_values)
    norm = Normalize(vmin=color_min, vmax=(color_max if color_max > color_min else color_min + 1e-6))
    cmap = plt.cm.viridis

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p"]
    marker_keys: list[Any] = []
    for item in metric_rows:
        if has_tau:
            marker_keys.append(("tau", float(item["tau"])))
        else:
            marker_keys.append(("beta", item["beta_key"]))
    unique_marker_keys = sorted(set(marker_keys), key=lambda k: str(k))
    marker_map = {
        key: marker_cycle[idx % len(marker_cycle)]
        for idx, key in enumerate(unique_marker_keys)
    }

    css_min = min(css_values)
    css_max = max(css_values)

    def _size_from_css(value: float) -> float:
        if css_max - css_min < 1e-9:
            return 130.0
        ratio = (value - css_min) / (css_max - css_min)
        return 80.0 + ratio * 180.0

    for idx, item in enumerate(metric_rows):
        color = cmap(norm(color_values[idx]))
        marker_key = marker_keys[idx]
        ax.scatter(
            float(item["x"]),
            float(item["y"]),
            c=[color],
            marker=marker_map[marker_key],
            s=_size_from_css(float(item["css"])),
            edgecolors="#1F2937",
            linewidths=0.35,
            zorder=3,
        )
        ax.annotate(
            str(item["id"]),
            (float(item["x"]), float(item["y"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#222222",
        )

    best = min(metric_rows, key=lambda item: (float(item["x"]), float(item["y"])))
    ax.scatter(
        float(best["x"]),
        float(best["y"]),
        marker="*",
        s=320,
        c="#D62728",
        edgecolors="#111111",
        linewidths=0.9,
        zorder=6,
    )
    ax.annotate(
        "BEST",
        (float(best["x"]), float(best["y"])),
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=9,
        fontweight="bold",
        color="#B22222",
    )

    del base_point, historical_point

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("RHO" if has_rho else "AS")

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[key],
            linestyle="",
            color="none",
            markerfacecolor="#A9A9A9",
            markeredgecolor="#222222",
            markersize=7,
            label=(
                f"T{_fmt_num(float(key[1]))}"
                if key[0] == "tau"
                else f"B{_fmt_num(key[1][0])}/{_fmt_num(key[1][1])}"
            ),
        )
        for key in unique_marker_keys
    ]
    extra_handles = [
        Line2D([0], [0], marker="*", linestyle="", color="#D62728", markersize=10, label="BEST"),
    ]
    handles = marker_handles + extra_handles
    ax.legend(handles=handles, title="B / Marks", fontsize=8, title_fontsize=9, loc="upper right")

    ax.set_xlabel("AE")
    ax.set_ylabel("P95")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="-", linewidth=0.7)
    fig.text(
        0.01,
        0.01,
        (
            "AE: AvgE2E | P95: P95 | E: sched_eta_exec | H: horizon_alpha | "
            + ("RHO: rho_mass | T: tau_p | " if has_rho or has_tau else "B: beta_hi/beta_lo | AS: alpha_stable | ")
            + "CSS: cold_start_share"
        ),
        fontsize=8,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_hparam_rank_strip(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no rows to plot")

    ranked = sorted(rows, key=lambda r: (float(r["avg_e2e_ms_mean"]), float(r["p95_ms_mean"])))
    has_rho = any(not math.isnan(_safe_float(r.get("hpwp_rho_mass", float("nan")))) for r in ranked)
    has_tau = any(not math.isnan(_safe_float(r.get("hpwp_tau_p", float("nan")))) for r in ranked)
    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked, start=1):
        entries.append(
            {
                "id": f"T{idx:02d}",
                "rank": idx,
                "label": str(row["label"]),
                "avg": float(row["avg_e2e_ms_mean"]),
                "p95": float(row["p95_ms_mean"]),
                "eta": _safe_float(row.get("hpwp_sched_eta_exec")),
                "horizon": _safe_float(row.get("hpwp_horizon_alpha")),
                "rho": _safe_float(row.get("hpwp_rho_mass", row.get("rho_mass", float("nan")))),
                "tau": _safe_float(row.get("hpwp_tau_p", row.get("tau_p", float("nan")))),
                "beta": f"{_fmt_num(_safe_float(row.get('hpwp_beta_hi')))}"
                f"/{_fmt_num(_safe_float(row.get('hpwp_beta_lo')))}",
                "alpha": _safe_float(row.get("hpwp_alpha_stable")),
            }
        )

    def _value_to_color(values: list[Any], cmap_name: str) -> dict[Any, Any]:
        unique = sorted(set(values))
        cmap = plt.get_cmap(cmap_name)
        if len(unique) == 1:
            return {unique[0]: cmap(0.6)}
        return {val: cmap(i / (len(unique) - 1)) for i, val in enumerate(unique)}

    eta_map = _value_to_color([e["eta"] for e in entries], "Blues")
    horizon_map = _value_to_color([e["horizon"] for e in entries], "Greens")
    beta_map = _value_to_color([e["beta"] for e in entries], "tab20")
    alpha_map = _value_to_color([e["alpha"] for e in entries], "Purples")
    rho_map = _value_to_color([e["rho"] for e in entries], "Oranges")
    tau_map = _value_to_color([e["tau"] for e in entries], "Reds")
    if has_rho or has_tau:
        row_defs = [
            ("E", "eta", eta_map),
            ("H", "horizon", horizon_map),
            ("R", "rho", rho_map),
            ("T", "tau", tau_map),
        ]
    else:
        row_defs = [
            ("E", "eta", eta_map),
            ("H", "horizon", horizon_map),
            ("B", "beta", beta_map),
            ("AS", "alpha", alpha_map),
        ]

    fig = plt.figure(figsize=(max(12.0, len(entries) * 0.45), 7.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.7], hspace=0.16)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_strip = fig.add_subplot(grid[1, 0], sharex=ax_top)

    x_vals = [e["rank"] for e in entries]
    y_vals = [e["avg"] for e in entries]
    ax_top.plot(x_vals, y_vals, color="#4E79A7", linewidth=1.2, alpha=0.8, zorder=1)
    ax_top.scatter(x_vals, y_vals, s=36, color="#4E79A7", zorder=2)
    for e in entries:
        ax_top.annotate(
            e["id"],
            (e["rank"], e["avg"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color="#1F2937",
        )

    best = min(entries, key=lambda e: (e["avg"], e["p95"]))
    ax_top.scatter(
        best["rank"],
        best["avg"],
        marker="*",
        s=260,
        color="#D62728",
        edgecolors="#111111",
        linewidths=0.9,
        zorder=3,
    )
    ax_top.annotate(
        "BEST",
        (best["rank"], best["avg"]),
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=8.5,
        color="#B22222",
        fontweight="bold",
    )
    ax_top.set_ylabel("AE (ms)")
    ax_top.set_title(title)
    ax_top.grid(axis="y", alpha=0.25)

    y_positions = list(reversed(range(len(row_defs))))
    for y_pos, (row_label, key, color_map) in zip(y_positions, row_defs):
        for e in entries:
            color = color_map[e[key]]
            rect = Rectangle(
                (e["rank"] - 0.45, y_pos - 0.35),
                0.9,
                0.7,
                facecolor=color,
                edgecolor="#FFFFFF",
                linewidth=0.4,
            )
            ax_strip.add_patch(rect)
        ax_strip.text(
            0.2,
            y_pos,
            row_label,
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="left",
            color="#111111",
        )

    ax_strip.set_xlim(0.5, len(entries) + 0.5)
    ax_strip.set_ylim(-0.8, len(row_defs) - 0.2)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Rank (best to worst)")
    ax_strip.grid(False)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    legend_handles: list[Patch] = []
    for _, _, cmap in row_defs:
        for val, color in cmap.items():
            legend_handles.append(Patch(facecolor=color, edgecolor="none", label=str(val)))
    dedup_handles: list[Patch] = []
    seen_labels: set[str] = set()
    for handle in legend_handles:
        label = str(handle.get_label())
        if label in seen_labels:
            continue
        seen_labels.add(label)
        dedup_handles.append(handle)
    fig.legend(
        handles=dedup_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(8, max(4, len(dedup_handles) // 2)),
        fontsize=7,
        frameon=False,
    )
    fig.text(
        0.01,
        0.07,
        (
            "Top: ranking by AvgE2E (tie-breaker P95). Bottom strips: "
            + ("E/H/R/T parameter values per rank." if has_rho or has_tau else "E/H/B/AS parameter values per rank.")
        ),
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _extract_run_dir(output: str, root: Path) -> Path:
    for line in reversed(output.splitlines()):
        prefix = "simulation completed:"
        if line.strip().startswith(prefix):
            raw = line.split(":", 1)[1].strip()
            run_dir = Path(raw)
            if not run_dir.is_absolute():
                run_dir = (root / run_dir).resolve()
            return run_dir
    raise RuntimeError(f"failed to parse run directory from output:\n{output}")


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)


def _percentile(samples: list[float], pct: int) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    ratio = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * ratio


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def _safe_float(raw: Any) -> float:
    if raw is None:
        return float("nan")
    if raw == "":
        return float("nan")
    return float(raw)


def _fmt_num(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


