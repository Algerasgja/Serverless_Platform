from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiment_common import (  # noqa: E402
    build_variant_configs,
    ensure_results_dir,
    load_base_config,
    repo_root,
    resolve_config_path,
    run_with_temp_configs,
    write_csv,
)

COMPARE_OUTPUT_SUBDIR = "compare"
DEFAULT_AUTOSCALERS = [
    "hpwp_v1",
    "kpa_v1",
    "lass_v1",
    "rl_q_v1",
    "hptd_v1",
    "xanadu_opt_v1",
    "oracle_future_v1",
    "depth_breadth_v1",
    "kraken_vomm_v1",
]
ALIAS = {
    "hpwp_v1": "hpwp",
    "kpa_v1": "kpa",
    "hpa_v1": "kpa",
    "lass_v1": "lass",
    "rl_q_v1": "rl_q",
    "hptd_v1": "hptd",
    # Keep legacy xanadu impl available but out of default compare set.
    "xanadu_v1": "xanadu_legacy",
    # Current compare "Xunadu" refers to optimized branch.
    "xanadu_opt_v1": "xanadu",
    "oracle_future_v1": "oracle",
    "depth_breadth_v1": "depth_breadth",
    "kraken_vomm_v1": "kraken_vomm",
    "hist_keepalive_prewarm_v1": "hist",
    "no_autoscale_v1": "no_as",
}
STRATEGY_ORDER = [
    "hpwp",
    "xanadu",
    "kraken_vomm",
    "lass",
    "rl_q",
    "hptd",
    "hist",
    "kpa",
    "depth_breadth",
    "oracle",
    "xanadu_legacy",
    "no_as",
]
DISPLAY_LABEL = {
    "hpwp": "ConScale",
    "kpa": "KPA",
    "lass": "LaSS",
    "rl_q": "QLAS",
    "hptd": "HPTD",
    "xanadu": "Xanadu",
    "xanadu_legacy": "Xunadu(v1)",
    "oracle": "Oracle",
    "depth_breadth": "DBW",
    "kraken_vomm": "Kraken",
    "hist": "Hist",
    "no_as": "\uff2e\uff21",
}
SCENARIO_FACTORS: list[tuple[str, float]] = [("low", 0.5), ("mid", 1.0), ("high", 2.0)]
SCENARIO_ORDER = [item[0] for item in SCENARIO_FACTORS]

METRIC_E2E_BUNDLE = "e2e_bundle"
METRIC_COLD_STEP_RATE = "cold_start_step_rate"
METRIC_PRED_PREWARM = "predictor_prewarm_utilization"
METRIC_PREWARM_COST = "prewarm_cost"
METRIC_MEMORY_TIME_COST = "memory_time_cost"
METRIC_CTRE95 = "ctre95"
ALL_METRICS = [
    METRIC_E2E_BUNDLE,
    METRIC_COLD_STEP_RATE,
    METRIC_PRED_PREWARM,
    METRIC_PREWARM_COST,
    METRIC_MEMORY_TIME_COST,
    METRIC_CTRE95,
]

SCENARIO_COLORS = {
    "low": "#4C78A8",
    "mid": "#F58518",
    "high": "#E45756",
}

# Strategy color palette (9-method compare)
STRATEGY_COLORS = {
    "hpwp": "#264653",  # 深海蓝绿
    "xanadu": "#287271",  # 墨青绿
    "kraken_vomm": "#2A9D8C",  # 湖水青
    "lass": "#5FA49A",  # 灰调青绿
    "rl_q": "#8AB07D",  # 鼠尾草绿
    "hptd": "#E9C46B",  # 暖芥末黄
    "kpa": "#F3A261",  # 杏橙色
    "depth_breadth": "#D98573",  # 柔和珊瑚粉橘
    "oracle": "#E66F51",  # 陶土橘红
}
DEFAULT_STRATEGY_COLOR = "#5B6E78"

COMPARE_SUPTITLE_FS = 38
COMPARE_SUBTITLE_FS = 32
COMPARE_AXIS_LABEL_FS = 30
COMPARE_TICK_FS = 26
COMPARE_ANNOT_FS = 24
COMPARE_LEGEND_FS = 26
COMPARE_LEGEND_TITLE_FS = 28


def _setup_plot_font(plt: Any) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _strategy_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(str(strategy), DEFAULT_STRATEGY_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoscaler comparison with metric-specific outputs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/default.yaml"],
        help="Config files used as base profiles.",
    )
    parser.add_argument(
        "--autoscalers",
        nargs="*",
        default=None,
        help="Optional autoscaler override list.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=["low", "mid", "high"],
        default=None,
        help="Optional scenario subset. Omitted means low/mid/high all.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Metric IDs to run: "
            "e2e_bundle cold_start_step_rate predictor_prewarm_utilization prewarm_cost "
            "memory_time_cost ctre95 (or all). "
            "If omitted, run all."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Repeated-run seeds.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for simulation subprocess runs.",
    )
    return parser.parse_args()


def autoscaler_label(raw: str) -> str:
    return ALIAS.get(str(raw).lower(), str(raw).lower())


def normalize_metric_ids(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(ALL_METRICS)
    lowered = [str(item).strip().lower() for item in raw if str(item).strip()]
    if not lowered:
        return list(ALL_METRICS)
    if "all" in lowered:
        return list(ALL_METRICS)
    unknown = sorted(set(lowered) - set(ALL_METRICS))
    if unknown:
        raise ValueError(f"unsupported metric ids: {', '.join(unknown)}")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in lowered:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def normalize_scenarios(raw: list[str] | None) -> list[str]:
    if not raw:
        return [item[0] for item in SCENARIO_FACTORS]
    ordered: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = str(item).strip().lower()
        if key not in {"low", "mid", "high"}:
            raise ValueError(f"unsupported scenario: {item}")
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def default_compare_autoscalers() -> list[str]:
    return list(DEFAULT_AUTOSCALERS)


def required_autoscalers_for_metrics(metric_ids: list[str], autoscalers: list[str]) -> list[str]:
    ordered = [str(item) for item in autoscalers]
    selected: list[str] = []

    def _append_if_present(pred: set[str]) -> None:
        for raw in ordered:
            if autoscaler_label(raw) in pred and raw not in selected:
                selected.append(raw)

    need_main = any(
        metric in {
            METRIC_E2E_BUNDLE,
            METRIC_COLD_STEP_RATE,
            METRIC_PRED_PREWARM,
            METRIC_PREWARM_COST,
            METRIC_MEMORY_TIME_COST,
            METRIC_CTRE95,
        }
        for metric in metric_ids
    )
    if need_main:
        for raw in ordered:
            if raw not in selected:
                selected.append(raw)
    if not selected:
        raise ValueError("no autoscalers selected for requested metrics")
    return selected


def build_compare_variants(
    *,
    autoscalers: list[str],
    scenario_rate_multipliers: list[tuple[str, float]] | None = None,
    base_rate_multiplier: float | None = None,
) -> list[dict[str, Any]]:
    if scenario_rate_multipliers is None or base_rate_multiplier is None:
        variants: list[dict[str, Any]] = []
        for item in autoscalers:
            variants.append(
                {
                    "label": autoscaler_label(item),
                    "overrides": {
                        "autoscaler": {
                            "type": str(item),
                        }
                    },
                }
            )
        return variants

    variants = []
    for scenario, factor in scenario_rate_multipliers:
        effective_rate = float(base_rate_multiplier) * float(factor)
        for item in autoscalers:
            strategy = autoscaler_label(item)
            variants.append(
                {
                    "label": f"{scenario}::{strategy}",
                    "overrides": {
                        "autoscaler": {
                            "type": str(item),
                        },
                        "workload": {
                            "rate_multiplier": effective_rate,
                        },
                    },
                    "scenario": scenario,
                    "strategy": strategy,
                    "rate_multiplier": effective_rate,
                }
            )
    return variants


def _split_compound_label(raw: str) -> tuple[str, str]:
    parts = str(raw).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid run label, expected 'scenario::strategy', got: {raw}")
    return parts[0], parts[1]


def _safe_mean_std(values: list[float]) -> tuple[float, float]:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    if not cleaned:
        return math.nan, math.nan
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    return float(statistics.fmean(cleaned)), float(statistics.stdev(cleaned))


def _integrate_memory_time_cost_mb_sec(node_metrics_path: Path) -> float:
    if not node_metrics_path.exists():
        raise FileNotFoundError(f"missing node_metrics.csv in {node_metrics_path.parent}")
    per_sec_mem_mb: dict[int, float] = defaultdict(float)
    with node_metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_sec = row.get("timestamp_sec")
            raw_mem = row.get("mem_reserved_mb")
            if raw_sec in (None, "") or raw_mem in (None, ""):
                continue
            sec = int(float(raw_sec))
            mem_mb = max(0.0, float(raw_mem))
            per_sec_mem_mb[sec] += mem_mb
    if not per_sec_mem_mb:
        return 0.0
    secs = sorted(per_sec_mem_mb.keys())
    mem_time_mb_sec = 0.0
    for idx, sec in enumerate(secs):
        next_sec = secs[idx + 1] if (idx + 1) < len(secs) else sec + 1
        dt_sec = max(1, int(next_sec - sec))
        mem_time_mb_sec += per_sec_mem_mb[sec] * float(dt_sec)
    return mem_time_mb_sec


def _read_run_side_metrics(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "summary.json"
    req_path = run_dir / "request_paths.csv"
    sch_path = run_dir / "scheduler_decisions.csv"
    node_path = run_dir / "node_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json in {run_dir}")
    if not req_path.exists():
        raise FileNotFoundError(f"missing request_paths.csv in {run_dir}")
    if not sch_path.exists():
        raise FileNotFoundError(f"missing scheduler_decisions.csv in {run_dir}")
    if not node_path.exists():
        raise FileNotFoundError(f"missing node_metrics.csv in {run_dir}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) or {}
    autoscaler_runtime = payload.get("autoscaler_runtime", {}) or {}
    cold_steps = 0
    warm_steps = 0
    with sch_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decision = str(row.get("decision_type", ""))
            if decision == "cold_start":
                cold_steps += 1
            elif decision == "warm_reuse":
                warm_steps += 1
    denom = cold_steps + warm_steps
    cold_start_step_rate = (float(cold_steps) / float(denom)) if denom > 0 else math.nan

    completed_count = 0
    with req_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("completed") != "1":
                continue
            raw_total = row.get("total_latency_ms")
            if raw_total in (None, ""):
                continue
            completed_count += 1

    prewarm_created = float(autoscaler_runtime.get("prewarm_created", 0.0) or 0.0)
    prewarm_consumed = float(autoscaler_runtime.get("prewarm_consumed", 0.0) or 0.0)
    prewarm_utilization = (prewarm_consumed / prewarm_created) if prewarm_created > 0 else 0.0

    # Prewarm cost is defined as scale-out container count in this run.
    prewarm_cost = prewarm_created
    memory_time_cost = _integrate_memory_time_cost_mb_sec(node_path)

    return {
        "cold_start_step_rate": cold_start_step_rate,
        "prewarm_utilization": prewarm_utilization,
        "prewarm_cost": prewarm_cost,
        "memory_time_cost": memory_time_cost,
    }


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[(str(item["scenario"]), str(item["strategy"]))].append(item)

    rows: list[dict[str, Any]] = []
    for (scenario, strategy), group in grouped.items():
        avg_e2e_mean, avg_e2e_std = _safe_mean_std([float(r["avg_e2e_ms"]) for r in group])
        p95_mean, p95_std = _safe_mean_std([float(r["p95_ms"]) for r in group])
        p99_mean, p99_std = _safe_mean_std([float(r["p99_ms"]) for r in group])
        success_mean, success_std = _safe_mean_std([float(r["success_rate"]) for r in group])
        cold_step_mean, cold_step_std = _safe_mean_std([float(r["cold_start_step_rate"]) for r in group])
        prewarm_mean, prewarm_std = _safe_mean_std([float(r["prewarm_utilization"]) for r in group])
        prewarm_cost_mean, prewarm_cost_std = _safe_mean_std([float(r["prewarm_cost"]) for r in group])
        memory_time_cost_mean, memory_time_cost_std = _safe_mean_std([float(r["memory_time_cost"]) for r in group])

        rows.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "runs": len(group),
                "rate_multiplier": float(group[0]["rate_multiplier"]),
                "avg_e2e_ms_mean": avg_e2e_mean,
                "avg_e2e_ms_std": avg_e2e_std,
                "p95_ms_mean": p95_mean,
                "p95_ms_std": p95_std,
                "p99_ms_mean": p99_mean,
                "p99_ms_std": p99_std,
                "success_rate_mean": success_mean,
                "success_rate_std": success_std,
                "cold_start_step_rate_mean": cold_step_mean,
                "cold_start_step_rate_std": cold_step_std,
                "prewarm_utilization_mean": prewarm_mean,
                "prewarm_utilization_std": prewarm_std,
                "prewarm_cost_mean": prewarm_cost_mean,
                "prewarm_cost_std": prewarm_cost_std,
                "memory_time_cost_mean": memory_time_cost_mean,
                "memory_time_cost_std": memory_time_cost_std,
                "ctre95_mean": math.nan,
                "ctre95_std": math.nan,
            }
        )

    rows.sort(
        key=lambda x: (
            SCENARIO_ORDER.index(str(x["scenario"])) if str(x["scenario"]) in SCENARIO_ORDER else 999,
            STRATEGY_ORDER.index(str(x["strategy"])) if str(x["strategy"]) in STRATEGY_ORDER else 999,
        )
    )
    return rows


def _parse_float(raw: Any, *, default: float = math.nan) -> float:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _load_aggregate_snapshot(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    has_legacy_pair = any(str(r.get("strategy", "")) == "xanadu_opt" for r in raw_rows)
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        strategy = str(row.get("strategy", ""))
        # Backward compatibility:
        # old compare snapshots had xanadu(v1)=xanadu and xanadu_opt=optimized.
        # New compare keeps optimized as "xanadu" and drops legacy by default.
        if has_legacy_pair and strategy == "xanadu":
            continue
        if strategy == "xanadu_opt":
            strategy = "xanadu"
        out.append(
            {
                "scenario": str(row.get("scenario", "")),
                "strategy": strategy,
                "runs": int(_parse_float(row.get("runs"), default=0.0)),
                "rate_multiplier": _parse_float(row.get("rate_multiplier"), default=math.nan),
                "avg_e2e_ms_mean": _parse_float(row.get("avg_e2e_ms_mean")),
                "avg_e2e_ms_std": _parse_float(row.get("avg_e2e_ms_std")),
                "p95_ms_mean": _parse_float(row.get("p95_ms_mean")),
                "p95_ms_std": _parse_float(row.get("p95_ms_std")),
                "p99_ms_mean": _parse_float(row.get("p99_ms_mean")),
                "p99_ms_std": _parse_float(row.get("p99_ms_std")),
                "success_rate_mean": _parse_float(row.get("success_rate_mean")),
                "success_rate_std": _parse_float(row.get("success_rate_std")),
                "cold_start_step_rate_mean": _parse_float(row.get("cold_start_step_rate_mean")),
                "cold_start_step_rate_std": _parse_float(row.get("cold_start_step_rate_std")),
                "prewarm_utilization_mean": _parse_float(row.get("prewarm_utilization_mean")),
                "prewarm_utilization_std": _parse_float(row.get("prewarm_utilization_std")),
                "prewarm_cost_mean": _parse_float(row.get("prewarm_cost_mean")),
                "prewarm_cost_std": _parse_float(row.get("prewarm_cost_std")),
                "memory_time_cost_mean": _parse_float(row.get("memory_time_cost_mean")),
                "memory_time_cost_std": _parse_float(row.get("memory_time_cost_std")),
                "ctre95_mean": _parse_float(row.get("ctre95_mean")),
                "ctre95_std": _parse_float(row.get("ctre95_std")),
            }
        )
    return out


def _merge_aggregate_rows(
    *,
    existing_rows: list[dict[str, Any]],
    updated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing_rows:
        key = (str(row.get("scenario", "")), str(row.get("strategy", "")))
        merged[key] = dict(row)
    for row in updated_rows:
        key = (str(row.get("scenario", "")), str(row.get("strategy", "")))
        merged[key] = dict(row)

    out = list(merged.values())
    out.sort(
        key=lambda x: (
            SCENARIO_ORDER.index(str(x["scenario"])) if str(x["scenario"]) in SCENARIO_ORDER else 999,
            STRATEGY_ORDER.index(str(x["strategy"])) if str(x["strategy"]) in STRATEGY_ORDER else 999,
            str(x["strategy"]),
        )
    )
    return out


def _filter_rows_by_strategy(rows: list[dict[str, Any]], allowed_strategies: set[str]) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("strategy", "")) in allowed_strategies]
    filtered.sort(
        key=lambda x: (
            SCENARIO_ORDER.index(str(x["scenario"])) if str(x["scenario"]) in SCENARIO_ORDER else 999,
            STRATEGY_ORDER.index(str(x["strategy"])) if str(x["strategy"]) in STRATEGY_ORDER else 999,
            str(x["strategy"]),
        )
    )
    return filtered


def _attach_ctre95(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oracle_by_scenario: dict[str, tuple[float, float]] = {}

    for row in rows:
        scenario = str(row.get("scenario", ""))
        strategy = str(row.get("strategy", ""))
        if strategy != "oracle":
            continue
        oracle_by_scenario[scenario] = (
            float(row.get("p95_ms_mean", math.nan)),
            float(row.get("memory_time_cost_mean", math.nan)),
        )

    out: list[dict[str, Any]] = []
    eps = 1e-9
    for row in rows:
        scenario = str(row.get("scenario", ""))
        strategy = str(row.get("strategy", ""))
        p95_m = float(row.get("p95_ms_mean", math.nan))
        cost_m = float(row.get("memory_time_cost_mean", math.nan))
        oracle = oracle_by_scenario.get(scenario)
        if oracle is None:
            ctre95 = math.nan
        else:
            p95_oracle, cost_oracle = oracle
            if (
                math.isnan(p95_oracle)
                or math.isnan(cost_oracle)
                or math.isnan(p95_m)
                or math.isnan(cost_m)
                or cost_m <= 0.0
                or cost_oracle <= 0.0
                or p95_m <= 0.0
                or p95_oracle <= 0.0
            ):
                ctre95 = math.nan
            elif strategy == "oracle":
                ctre95 = 0.0
            else:
                # EP95(m) = (P95_oracle / P95_m) * (Cost_oracle / Cost_m)
                ep95 = (p95_oracle / (p95_m + eps)) * (cost_oracle / (cost_m + eps))
                if ep95 <= 0.0 or math.isnan(ep95):
                    ctre95 = math.nan
                else:
                    # LogEP95(m) = -log(EP95(m))
                    ctre95 = -math.log(ep95)

        payload = dict(row)
        payload["ctre95_mean"] = ctre95
        payload["ctre95_std"] = 0.0
        out.append(payload)
    return out


def _format_num(raw: float, *, digits: int = 6) -> str:
    if math.isnan(float(raw)):
        return ""
    return f"{float(raw):.{digits}f}"


def _write_metric_csv(metric_id: str, rows: list[dict[str, Any]], path: Path) -> None:
    if metric_id == METRIC_E2E_BUNDLE:
        payload = [
            {
                "scenario": row["scenario"],
                "strategy": row["strategy"],
                "runs": row["runs"],
                "rate_multiplier": _format_num(float(row["rate_multiplier"]), digits=8),
                "avg_e2e_ms_mean": _format_num(float(row["avg_e2e_ms_mean"])),
                "avg_e2e_ms_std": _format_num(float(row["avg_e2e_ms_std"])),
                "p95_ms_mean": _format_num(float(row["p95_ms_mean"])),
                "p95_ms_std": _format_num(float(row["p95_ms_std"])),
                "p99_ms_mean": _format_num(float(row["p99_ms_mean"])),
                "p99_ms_std": _format_num(float(row["p99_ms_std"])),
            }
            for row in rows
        ]
        write_csv(path, payload)
        return

    mapping = {
        METRIC_COLD_STEP_RATE: ("cold_start_step_rate_mean", "cold_start_step_rate_std"),
        METRIC_PRED_PREWARM: ("prewarm_utilization_mean", "prewarm_utilization_std"),
        METRIC_PREWARM_COST: ("prewarm_cost_mean", "prewarm_cost_std"),
        METRIC_MEMORY_TIME_COST: ("memory_time_cost_mean", "memory_time_cost_std"),
        METRIC_CTRE95: ("ctre95_mean", "ctre95_std"),
    }
    mean_key, std_key = mapping[metric_id]
    payload = []
    for row in rows:
        value = float(row[mean_key])
        std = float(row[std_key])
        if math.isnan(value):
            continue
        payload.append(
            {
                "scenario": row["scenario"],
                "strategy": row["strategy"],
                "runs": row["runs"],
                "rate_multiplier": _format_num(float(row["rate_multiplier"]), digits=8),
                "value_mean": _format_num(value),
                "value_std": _format_num(std),
            }
        )
    write_csv(path, payload)


def _plot_e2e_bundle(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _setup_plot_font(plt)

    row_map = {
        (str(row["scenario"]), str(row["strategy"])): row
        for row in rows
    }
    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if any((scenario, strategy) in row_map for scenario in SCENARIO_ORDER)
    ]
    if not strategies:
        raise RuntimeError("no rows available for e2e_bundle plot")

    metric_specs = [
        ("avg_e2e_ms_mean", "Avg"),
        ("p95_ms_mean", "P95"),
        ("p99_ms_mean", "P99"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(25, 14.5), sharex=False, sharey="row")
    x = list(range(len(strategies)))

    for metric_idx, (mean_key, metric_label) in enumerate(metric_specs):
        for scenario_idx, scenario in enumerate(SCENARIO_ORDER):
            ax = axes[metric_idx][scenario_idx]
            vals: list[float] = []
            for strategy in strategies:
                row = row_map.get((scenario, strategy))
                vals.append(math.nan if row is None else float(row[mean_key]))
            bar_colors = [_strategy_color(strategy) for strategy in strategies]

            ax.bar(
                x,
                vals,
                width=0.72,
                color=bar_colors,
                edgecolor="#2F3E46",
                linewidth=0.35,
            )
            ax.set_title(f"{metric_label} - {scenario.upper()}", fontsize=COMPARE_SUBTITLE_FS)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
            ax.tick_params(axis="y", labelsize=COMPARE_TICK_FS)
            if scenario_idx == 0:
                ax.set_ylabel("Latency (ms)", fontsize=COMPARE_AXIS_LABEL_FS)

    legend_handles = [
        Patch(
            facecolor=_strategy_color(strategy),
            edgecolor="#2F3E46",
            linewidth=0.35,
            label=DISPLAY_LABEL.get(str(strategy), str(strategy)),
        )
        for strategy in strategies
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=COMPARE_LEGEND_FS + 1,
    )
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scalar_metric(
    rows: list[dict[str, Any]],
    *,
    mean_key: str,
    std_key: str,
    out_path: Path,
    title: str,
    y_label: str,
    percent: bool = False,
    value_scale: float = 1.0,
    annotation_digits: int = 2,
    draw_zero_line: bool = False,
    y_tick_decimals: int | None = None,
    show_legend: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter
    _setup_plot_font(plt)

    scenario_strategy_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        strategy = str(row["strategy"])
        scenario_strategy_map[(scenario, strategy)] = row

    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if any((scenario, strategy) in scenario_strategy_map for scenario in SCENARIO_ORDER)
    ]
    if not strategies:
        raise RuntimeError(f"no rows available for {mean_key} plot")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), sharey=True)
    x = list(range(len(strategies)))

    for idx, scenario in enumerate(SCENARIO_ORDER):
        ax = axes[idx]
        vals: list[float] = []
        for strategy in strategies:
            row = scenario_strategy_map.get((scenario, strategy))
            if row is None:
                vals.append(math.nan)
                continue
            val = float(row[mean_key])
            if percent:
                val *= 100.0
            val *= float(value_scale)
            vals.append(val)

        finite_vals = [v for v in vals if not math.isnan(v)]
        if not finite_vals:
            continue

        ax.bar(
            x,
            vals,
            width=0.72,
            color=[_strategy_color(strategy) for strategy in strategies],
            edgecolor="#2F3E46",
            linewidth=0.35,
        )
        if draw_zero_line:
            ax.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.75, zorder=0)

        # Hide bar-value annotations for cleaner compare figures.

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        ax.tick_params(axis="y", labelsize=COMPARE_TICK_FS)
        if y_tick_decimals is not None:
            ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{int(y_tick_decimals)}f"))
        ax.set_title(f"{scenario.upper()}", fontsize=COMPARE_SUBTITLE_FS)
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=COMPARE_AXIS_LABEL_FS)

    legend_handles = [
        Patch(
            facecolor=_strategy_color(strategy),
            edgecolor="#2F3E46",
            linewidth=0.35,
            label=DISPLAY_LABEL.get(str(strategy), str(strategy)),
        )
        for strategy in strategies
    ]
    if show_legend:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=max(1, len(legend_handles)),
            frameon=False,
            fontsize=COMPARE_LEGEND_FS,
        )
    fig.tight_layout(rect=(0.0, 0.19, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_metric_plot(metric_id: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    if metric_id == METRIC_E2E_BUNDLE:
        _plot_e2e_bundle(rows, out_path)
        return
    if metric_id == METRIC_COLD_STEP_RATE:
        _plot_scalar_metric(
            rows,
            mean_key="cold_start_step_rate_mean",
            std_key="cold_start_step_rate_std",
            out_path=out_path,
            title="Cold Start Step Rate",
            y_label="Rate (%)",
            percent=True,
        )
        return
    if metric_id == METRIC_PRED_PREWARM:
        _plot_scalar_metric(
            rows,
            mean_key="prewarm_utilization_mean",
            std_key="prewarm_utilization_std",
            out_path=out_path,
            title="Scaling Utilization",
            y_label="Utilization (%)",
            percent=True,
        )
        return
    if metric_id == METRIC_PREWARM_COST:
        _plot_scalar_metric(
            rows,
            mean_key="prewarm_cost_mean",
            std_key="prewarm_cost_std",
            out_path=out_path,
            title="Scale-out Container Count",
            y_label="Scale-out Container Count",
            percent=False,
        )
        return
    if metric_id == METRIC_MEMORY_TIME_COST:
        _plot_scalar_metric(
            rows,
            mean_key="memory_time_cost_mean",
            std_key="memory_time_cost_std",
            out_path=out_path,
            title="Memory-Time Cost",
            y_label="Memory-Time Cost (×1e8 MB*sec)",
            percent=False,
            value_scale=1e-8,
            annotation_digits=2,
            y_tick_decimals=2,
        )
        return
    if metric_id == METRIC_CTRE95:
        _plot_scalar_metric(
            rows,
            mean_key="ctre95_mean",
            std_key="ctre95_std",
            out_path=out_path,
            title="LogEP95",
            y_label="LogEP95",
            percent=False,
            value_scale=1.0,
            annotation_digits=2,
            draw_zero_line=True,
        )
        return
    raise ValueError(f"unsupported metric id: {metric_id}")


def _metric_rows_for_id(metric_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if metric_id in {METRIC_PREWARM_COST, METRIC_MEMORY_TIME_COST}:
        return list(rows)
    if metric_id == METRIC_CTRE95:
        # Hide Oracle for EP95/LogEP95 visualization.
        return [row for row in rows if str(row.get("strategy", "")) != "oracle"]
    # Oracle is only meaningful for scale-out cost upper-bound comparison.
    return [row for row in rows if str(row.get("strategy", "")) != "oracle"]


def _build_records(
    *,
    all_run_metrics: list[Any],
    scenario_rate_multipliers: dict[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in all_run_metrics:
        scenario, strategy = _split_compound_label(str(item.label))
        side = _read_run_side_metrics(item.run_dir)
        records.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "seed": int(item.seed),
                "rate_multiplier": float(scenario_rate_multipliers[scenario]),
                "run_dir": str(item.run_dir),
                "avg_e2e_ms": float(item.avg_e2e_ms),
                "p95_ms": float(item.p95_ms),
                "p99_ms": float(item.p99_ms),
                "success_rate": float(item.success_rate),
                **side,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    metric_ids = normalize_metric_ids(args.metrics)
    selected_scenarios = normalize_scenarios(args.scenarios)
    root = repo_root()
    root_results = ensure_results_dir(root=root, relative=args.results_dir)

    autoscalers = (
        [str(x) for x in args.autoscalers]
        if args.autoscalers
        else default_compare_autoscalers()
    )
    selected_autoscalers = required_autoscalers_for_metrics(
        metric_ids=metric_ids,
        autoscalers=autoscalers,
    )
    blocked_autoscalers = {"xanadu_v1", "no_autoscale_v1"}
    filtered_autoscalers: list[str] = []
    skipped: list[str] = []
    for item in selected_autoscalers:
        if str(item).lower() in blocked_autoscalers:
            skipped.append(str(item))
            continue
        filtered_autoscalers.append(str(item))
    selected_autoscalers = filtered_autoscalers
    if skipped:
        print(f"[compare] skipped unsupported baselines in unified compare: {', '.join(skipped)}")
    if not selected_autoscalers:
        raise ValueError("no autoscalers left after filtering unsupported baselines")

    selected_factors = [(name, factor) for name, factor in SCENARIO_FACTORS if name in selected_scenarios]
    selected_factor_map = {name: factor for name, factor in selected_factors}

    all_run_metrics: list[Any] = []
    failures: list[dict[str, Any]] = []
    for cfg_token in args.configs:
        cfg_path = resolve_config_path(cfg_token, root=root)
        base = load_base_config(cfg_path)
        exp_name = str((base.get("experiment") or {}).get("name", cfg_path.stem))
        base_rate = float((base.get("workload") or {}).get("rate_multiplier", 1.0))

        variant_defs = build_compare_variants(
            autoscalers=selected_autoscalers,
            scenario_rate_multipliers=selected_factors,
            base_rate_multiplier=base_rate,
        )
        variants = build_variant_configs(
            base_config=base,
            base_experiment_name=exp_name,
            variant_overrides=variant_defs,
            seeds=[int(s) for s in args.seeds],
        )
        metrics, errs = run_with_temp_configs(
            python_bin=args.python,
            root=root,
            variants=variants,
        )
        all_run_metrics.extend(metrics)
        failures.extend(errs)

    records = _build_records(
        all_run_metrics=all_run_metrics,
        scenario_rate_multipliers=selected_factor_map,
    )
    updated_rows = _aggregate_records(records)

    compare_dir = root_results / COMPARE_OUTPUT_SUBDIR
    metrics_dir = compare_dir / "metrics"
    figures_dir = compare_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    snapshot_csv = compare_dir / "compare_metrics.csv"
    existing_rows = _load_aggregate_snapshot(snapshot_csv)
    merged_rows = _merge_aggregate_rows(
        existing_rows=existing_rows,
        updated_rows=updated_rows,
    )
    allowed_strategies = {autoscaler_label(item) for item in default_compare_autoscalers()}
    allowed_strategies.update(autoscaler_label(item) for item in selected_autoscalers)
    merged_rows = _filter_rows_by_strategy(merged_rows, allowed_strategies=allowed_strategies)
    merged_rows = _attach_ctre95(merged_rows)
    write_csv(snapshot_csv, merged_rows)
    print(f"[compare] output dir: {compare_dir}")
    print(f"[compare] updated snapshot csv: {snapshot_csv}")

    for metric_id in metric_ids:
        rows_for_metric = _metric_rows_for_id(metric_id, merged_rows)
        csv_path = metrics_dir / f"{metric_id}.csv"
        fig_path = figures_dir / f"{metric_id}.png"
        _write_metric_csv(metric_id, rows_for_metric, csv_path)
        _render_metric_plot(metric_id, rows_for_metric, fig_path)
        print(f"[compare] updated metric csv: {csv_path}")
        print(f"[compare] updated figure: {fig_path}")

    if failures:
        print(f"[compare] failures: {len(failures)}")
        for err in failures:
            print(f"[compare] failed: {err['label']} seed={err['seed']}: {err['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
