from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments.compare_experiments import (  # noqa: E402
    ALL_METRICS,
    METRIC_COLD_STEP_RATE,
    METRIC_CTRE95,
    METRIC_E2E_BUNDLE,
    METRIC_MEMORY_TIME_COST,
    METRIC_PRED_PREWARM,
    METRIC_PREWARM_COST,
    _render_metric_plot,
    normalize_metric_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render compare figures directly from results/metrics CSV files (no simulation rerun)."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/compare/metrics",
        help="Metrics directory. Default: results/compare/metrics",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/compare/figures",
        help="Figures directory. Default: results/compare/figures",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric IDs to render. If omitted, render all.",
    )
    return parser.parse_args()


def _safe_float(raw: str | None) -> float:
    if raw is None or str(raw).strip() == "":
        return float("nan")
    return float(raw)


def _load_metric_rows(metric_id: str, csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing metric csv: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    rows: list[dict[str, Any]] = []
    if metric_id == METRIC_E2E_BUNDLE:
        for row in raw_rows:
            rows.append(
                {
                    "scenario": str(row.get("scenario", "")),
                    "strategy": str(row.get("strategy", "")),
                    "avg_e2e_ms_mean": _safe_float(row.get("avg_e2e_ms_mean")),
                    "avg_e2e_ms_std": _safe_float(row.get("avg_e2e_ms_std")),
                    "p95_ms_mean": _safe_float(row.get("p95_ms_mean")),
                    "p95_ms_std": _safe_float(row.get("p95_ms_std")),
                    "p99_ms_mean": _safe_float(row.get("p99_ms_mean")),
                    "p99_ms_std": _safe_float(row.get("p99_ms_std")),
                }
            )
        return rows

    key_map = {
        METRIC_COLD_STEP_RATE: ("cold_start_step_rate_mean", "cold_start_step_rate_std"),
        METRIC_PRED_PREWARM: ("prewarm_utilization_mean", "prewarm_utilization_std"),
        METRIC_PREWARM_COST: ("prewarm_cost_mean", "prewarm_cost_std"),
        METRIC_MEMORY_TIME_COST: ("memory_time_cost_mean", "memory_time_cost_std"),
        METRIC_CTRE95: ("ctre95_mean", "ctre95_std"),
    }
    mean_key, std_key = key_map[metric_id]
    for row in raw_rows:
        rows.append(
            {
                "scenario": str(row.get("scenario", "")),
                "strategy": str(row.get("strategy", "")),
                mean_key: _safe_float(row.get("value_mean")),
                std_key: _safe_float(row.get("value_std")),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    metric_ids = normalize_metric_ids(args.metrics)
    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    requested = metric_ids if metric_ids else list(ALL_METRICS)
    for metric_id in requested:
        csv_path = metrics_dir / f"{metric_id}.csv"
        rows = _load_metric_rows(metric_id, csv_path)
        fig_path = figures_dir / f"{metric_id}.png"
        _render_metric_plot(metric_id, rows, fig_path)
        print(f"saved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
