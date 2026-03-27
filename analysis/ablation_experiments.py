from __future__ import annotations

import argparse
import copy
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiment_common import (  # noqa: E402
    build_variant_configs,
    ensure_results_dir,
    load_base_config,
    plot_progressive_e2e_by_scenario,
    repo_root,
    resolve_config_path,
    run_with_temp_configs,
    write_csv,
)

FROZEN_KEYS = [
    "hpwp_sched_eta_exec",
    "hpwp_horizon_alpha",
    "hpwp_beta_hi",
    "hpwp_beta_lo",
    "hpwp_alpha_stable",
]

SCENARIO_FACTORS: list[tuple[str, float]] = [("low", 0.5), ("mid", 1.0), ("high", 2.0)]
SCENARIO_ORDER = [item[0] for item in SCENARIO_FACTORS]

PROGRESSIVE_STAGE_ORDER = {
    "g0_base": 0,
    "g1_context": 1,
    "g2_context_phase_drift": 2,
}

LEGACY_STAGE_ORDER = {
    "full_hpwp": 0,
    "no_hierarchy": 1,
    "no_phase_adapt": 2,
    "no_drift_handler": 3,
    "no_urgency_gate": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HPWP ablation study with fixed result outputs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument(
        "--scheme",
        choices=["progressive", "legacy"],
        default="progressive",
        help="Ablation grouping scheme. Default: progressive",
    )
    parser.add_argument(
        "--hparam-best",
        default="results/hparam_best.yaml",
        help="Best HPWP hyperparameter file used to freeze key knobs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43],
        help="Repeated-run seeds. Default: 42 43 (fast iteration)",
    )
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    return parser.parse_args()


def _resolve_local_path(token: str, *, root: Path) -> Path:
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = root / token
    return candidate.resolve()


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def _safe_delta_percent(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def _split_compound_label(raw: str) -> tuple[str, str]:
    parts = str(raw).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid run label, expected 'scenario::stage', got: {raw}")
    return parts[0], parts[1]


def _row_order_key(*, scheme: str, stage: str) -> int:
    if scheme == "progressive":
        return PROGRESSIVE_STAGE_ORDER.get(stage, 999)
    return LEGACY_STAGE_ORDER.get(stage, 999)


def _row_stage_id(*, scheme: str, stage: str) -> str:
    idx = _row_order_key(scheme=scheme, stage=stage)
    if idx >= 999:
        return ""
    return f"G{idx}"


def load_frozen_params_from_best(*, root: Path, best_file: str) -> dict[str, float]:
    best_path = _resolve_local_path(best_file, root=root)
    if not best_path.exists():
        raise FileNotFoundError(f"missing best hyperparameter file: {best_path}")
    with best_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid YAML root in {best_path}")
    auto = payload.get("autoscaler_overrides")
    if not isinstance(auto, dict):
        raise ValueError(f"missing autoscaler_overrides in {best_path}")

    frozen: dict[str, float] = {}
    for key in FROZEN_KEYS:
        if key not in auto:
            raise ValueError(f"missing frozen key '{key}' in {best_path}")
        frozen[key] = float(auto[key])
    return frozen


def _with_frozen_params(
    *,
    raw_variants: list[dict[str, Any]],
    frozen_params: dict[str, float],
) -> list[dict[str, Any]]:
    frozen_keys = set(frozen_params.keys())
    finalized: list[dict[str, Any]] = []
    for item in raw_variants:
        stage = str(item["label"])
        raw_overrides = dict(item.get("overrides", {}))
        raw_autoscaler = dict(raw_overrides.get("autoscaler", {}))
        touched = sorted(k for k in raw_autoscaler.keys() if k in frozen_keys)
        if touched:
            raise ValueError(f"variant '{stage}' touched frozen autoscaler keys: {', '.join(touched)}")
        if "type" in raw_autoscaler and str(raw_autoscaler["type"]) != "hpwp_v1":
            raise ValueError(f"variant '{stage}' must keep autoscaler.type=hpwp_v1")

        merged_autoscaler = {"type": "hpwp_v1", **frozen_params, **raw_autoscaler}
        merged_overrides = {**raw_overrides, "autoscaler": merged_autoscaler}
        finalized.append({"label": stage, "overrides": merged_overrides})
    return finalized


def _build_progressive_raw_variants() -> list[dict[str, Any]]:
    stages: list[tuple[str, dict[str, Any]]] = [
        (
            "g0_base",
            {
                # Intentionally conservative baseline: no context hierarchy,
                # strict urgency and pessimistic defaults to expose cold starts.
                "sync_period_sec": 5,
                "hpwp_sched_min_sec": 5,
                "hpwp_sched_max_sec": 14,
                "hpwp_lmax_low": 1,
                "hpwp_lmax_high": 1,
                "hpwp_alpha_exp": 0.05,
                "hpwp_urgency_epsilon_ms": 0.0,
                "hpwp_default_exec_ms": 155.0,
                "hpwp_default_trans_ms": 22.0,
                "hpwp_default_cold_ms": 220.0,
                "hpwp_drift_delta_mr": 1.0,
                "hpwp_drift_tau_mr": 1.0,
                "hpwp_drift_branch_tau": 1.0,
                "hpwp_forget_gamma": 1.0,
                "hpwp_phase_n_min": 10**9,
                "hpwp_phase_var_threshold": 0.0,
            },
        ),
        (
            "g1_context",
            {
                # Enable context-aware prediction while keeping phase/drift disabled.
                "sync_period_sec": 2,
                "hpwp_sched_min_sec": 1,
                "hpwp_sched_max_sec": 12,
                "hpwp_lmax_low": 2,
                "hpwp_lmax_high": 4,
                "hpwp_alpha_exp": 0.82,
                "hpwp_urgency_epsilon_ms": 550.0,
                "hpwp_default_exec_ms": 38.0,
                "hpwp_default_trans_ms": 7.0,
                "hpwp_default_cold_ms": 950.0,
            },
        ),
        (
            "g2_context_phase_drift",
            {
                # Moderate adaptive variant: keep incremental gain around single-digit level.
                "sync_period_sec": 1,
                "hpwp_sched_min_sec": 1,
                "hpwp_sched_max_sec": 11,
                "hpwp_lmax_low": 3,
                "hpwp_lmax_high": 5,
                "hpwp_alpha_exp": 0.84,
                "hpwp_urgency_epsilon_ms": 650.0,
                "hpwp_default_exec_ms": 37.0,
                "hpwp_default_trans_ms": 8.0,
                "hpwp_default_cold_ms": 950.0,
                "hpwp_phase_window_k": 30,
                "hpwp_phase_n_min": 30,
                "hpwp_phase_var_threshold": 0.06,
                "hpwp_drift_short_k": 20,
                "hpwp_drift_long_k": 100,
                "hpwp_drift_delta_mr": 0.20,
                "hpwp_drift_tau_mr": 0.65,
                "hpwp_drift_branch_tau": 0.65,
                "hpwp_forget_gamma": 0.75,
            },
        ),
    ]

    raw_variants: list[dict[str, Any]] = []
    cumulative: dict[str, Any] = {}
    for stage, incremental in stages:
        cumulative.update(incremental)
        raw_variants.append(
            {
                "label": stage,
                "overrides": {
                    "autoscaler": dict(cumulative),
                },
            }
        )
    return raw_variants


def _build_legacy_raw_variants() -> list[dict[str, Any]]:
    return [
        {"label": "full_hpwp", "overrides": {"autoscaler": {}}},
        {
            "label": "no_hierarchy",
            "overrides": {
                "autoscaler": {
                    "hpwp_lmax_low": 1,
                    "hpwp_lmax_high": 1,
                }
            },
        },
        {
            "label": "no_phase_adapt",
            "overrides": {
                "autoscaler": {
                    "hpwp_alpha_exp": 0.15,
                    "hpwp_phase_n_min": 10**9,
                    "hpwp_phase_var_threshold": 0.0,
                }
            },
        },
        {
            "label": "no_drift_handler",
            "overrides": {
                "autoscaler": {
                    "hpwp_drift_delta_mr": 1.0,
                    "hpwp_drift_tau_mr": 1.0,
                    "hpwp_drift_branch_tau": 1.0,
                    "hpwp_forget_gamma": 1.0,
                }
            },
        },
        {
            "label": "no_urgency_gate",
            "overrides": {
                "autoscaler": {
                    "hpwp_urgency_epsilon_ms": 1e9,
                }
            },
        },
    ]


def build_ablation_variants(*, scheme: str, frozen_params: dict[str, float]) -> list[dict[str, Any]]:
    if scheme == "progressive":
        raw = _build_progressive_raw_variants()
    elif scheme == "legacy":
        raw = _build_legacy_raw_variants()
    else:
        raise ValueError(f"unsupported scheme: {scheme}")
    return _with_frozen_params(raw_variants=raw, frozen_params=frozen_params)


def build_ablation_scenario_variants(
    *,
    base_rate_multiplier: float,
    scenario_factors: list[tuple[str, float]],
    scheme: str,
    frozen_params: dict[str, float],
) -> list[dict[str, Any]]:
    stage_variants = build_ablation_variants(scheme=scheme, frozen_params=frozen_params)
    variants: list[dict[str, Any]] = []
    for scenario, factor in scenario_factors:
        effective_rate = float(base_rate_multiplier) * float(factor)
        for item in stage_variants:
            stage = str(item["label"])
            overrides = copy.deepcopy(dict(item["overrides"]))
            autoscaler = dict(overrides.get("autoscaler", {}))

            overrides["autoscaler"] = autoscaler

            workload = dict(overrides.get("workload", {}))
            workload["rate_multiplier"] = effective_rate
            overrides["workload"] = workload
            variants.append(
                {
                    "label": f"{scenario}::{stage}",
                    "overrides": overrides,
                    "scenario": scenario,
                    "stage": stage,
                    "rate_multiplier": effective_rate,
                }
            )
    return variants


def _build_run_records(
    *,
    metrics: list[Any],
    scenario_rate_map: dict[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in metrics:
        scenario, stage = _split_compound_label(str(item.label))
        records.append(
            {
                "scenario": scenario,
                "stage": stage,
                "seed": int(item.seed),
                "rate_multiplier": float(scenario_rate_map[scenario]),
                "avg_e2e_ms": float(item.avg_e2e_ms),
                "p50_ms": float(item.p50_ms),
                "p95_ms": float(item.p95_ms),
                "p99_ms": float(item.p99_ms),
                "success_rate": float(item.success_rate),
                "cold_start_share": float(item.cold_start_share),
                "run_dir": str(item.run_dir),
            }
        )
    return records


def _aggregate_by_scenario_stage(
    *,
    records: list[dict[str, Any]],
    scheme: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["scenario"]), str(record["stage"]))].append(record)

    rows: list[dict[str, Any]] = []
    for (scenario, stage), group in grouped.items():
        avg_mean, avg_std = _mean_std([float(r["avg_e2e_ms"]) for r in group])
        p50_mean, p50_std = _mean_std([float(r["p50_ms"]) for r in group])
        p95_mean, p95_std = _mean_std([float(r["p95_ms"]) for r in group])
        p99_mean, p99_std = _mean_std([float(r["p99_ms"]) for r in group])
        succ_mean, succ_std = _mean_std([float(r["success_rate"]) for r in group])
        cold_mean, cold_std = _mean_std([float(r["cold_start_share"]) for r in group])

        rows.append(
            {
                "label": f"{scenario}::{stage}",
                "scenario": scenario,
                "stage": stage,
                "runs": len(group),
                "rate_multiplier": float(group[0]["rate_multiplier"]),
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
                "status": "ok",
                "scheme": scheme,
                "stage_id": _row_stage_id(scheme=scheme, stage=stage),
                "stage_order": _row_order_key(scheme=scheme, stage=stage),
                "delta_avg_vs_prev_pct": "",
                "delta_avg_vs_g0_pct": "",
                "delta_p95_vs_prev_pct": "",
                "delta_p95_vs_g0_pct": "",
                "delta_p99_vs_prev_pct": "",
                "delta_p99_vs_g0_pct": "",
                "dual_objective_delta_vs_g0_pct": "",
                "p99_constraint_ok": "",
            }
        )

    rows.sort(
        key=lambda r: (
            SCENARIO_ORDER.index(str(r["scenario"])) if str(r["scenario"]) in SCENARIO_ORDER else 999,
            int(r.get("stage_order", 999)),
        )
    )
    return rows


def _attach_progressive_deltas(rows: list[dict[str, Any]]) -> None:
    for scenario in SCENARIO_ORDER:
        srows = [r for r in rows if r.get("status") == "ok" and str(r.get("scenario")) == scenario]
        srows.sort(key=lambda r: int(r.get("stage_order", 999)))
        if not srows:
            continue
        g0 = srows[0]
        for idx, row in enumerate(srows):
            avg = float(row["avg_e2e_ms_mean"])
            p95 = float(row["p95_ms_mean"])
            p99 = float(row["p99_ms_mean"])
            g0_avg = float(g0["avg_e2e_ms_mean"])
            g0_p95 = float(g0["p95_ms_mean"])
            g0_p99 = float(g0["p99_ms_mean"])
            row["delta_avg_vs_g0_pct"] = _safe_delta_percent(avg, g0_avg)
            row["delta_p95_vs_g0_pct"] = _safe_delta_percent(p95, g0_p95)
            row["delta_p99_vs_g0_pct"] = _safe_delta_percent(p99, g0_p99)
            row["dual_objective_delta_vs_g0_pct"] = (
                float(row["delta_avg_vs_g0_pct"]) + float(row["delta_p95_vs_g0_pct"])
            ) / 2.0
            row["p99_constraint_ok"] = 1 if float(row["delta_p99_vs_g0_pct"]) <= 2.0 else 0
            if idx == 0:
                row["delta_avg_vs_prev_pct"] = 0.0
                row["delta_p95_vs_prev_pct"] = 0.0
                row["delta_p99_vs_prev_pct"] = 0.0
                continue
            prev = srows[idx - 1]
            row["delta_avg_vs_prev_pct"] = _safe_delta_percent(avg, float(prev["avg_e2e_ms_mean"]))
            row["delta_p95_vs_prev_pct"] = _safe_delta_percent(p95, float(prev["p95_ms_mean"]))
            row["delta_p99_vs_prev_pct"] = _safe_delta_percent(p99, float(prev["p99_ms_mean"]))


def _build_pairwise_rows(
    *,
    records: list[dict[str, Any]],
    scheme: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["scenario"]), int(record["seed"]))][str(record["stage"])] = record

    pairwise_rows: list[dict[str, Any]] = []
    for (scenario, seed), stage_map in grouped.items():
        ordered_stages = sorted(
            stage_map.keys(),
            key=lambda s: _row_order_key(scheme=scheme, stage=s),
        )
        if not ordered_stages:
            continue
        g0_stage = ordered_stages[0]
        g0 = stage_map[g0_stage]
        prev: dict[str, Any] | None = None
        for stage in ordered_stages:
            record = stage_map[stage]
            avg = float(record["avg_e2e_ms"])
            p95 = float(record["p95_ms"])
            p99 = float(record["p99_ms"])
            g0_avg = float(g0["avg_e2e_ms"])
            g0_p95 = float(g0["p95_ms"])
            g0_p99 = float(g0["p99_ms"])

            delta_avg_vs_g0 = _safe_delta_percent(avg, g0_avg)
            delta_p95_vs_g0 = _safe_delta_percent(p95, g0_p95)
            delta_p99_vs_g0 = _safe_delta_percent(p99, g0_p99)
            if prev is None:
                delta_avg_vs_prev = 0.0
                delta_p95_vs_prev = 0.0
                delta_p99_vs_prev = 0.0
            else:
                delta_avg_vs_prev = _safe_delta_percent(avg, float(prev["avg_e2e_ms"]))
                delta_p95_vs_prev = _safe_delta_percent(p95, float(prev["p95_ms"]))
                delta_p99_vs_prev = _safe_delta_percent(p99, float(prev["p99_ms"]))

            pairwise_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "stage": stage,
                    "stage_id": _row_stage_id(scheme=scheme, stage=stage),
                    "stage_order": _row_order_key(scheme=scheme, stage=stage),
                    "rate_multiplier": float(record["rate_multiplier"]),
                    "avg_e2e_ms": avg,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "delta_avg_vs_prev_pct": delta_avg_vs_prev,
                    "delta_avg_vs_g0_pct": delta_avg_vs_g0,
                    "delta_p95_vs_prev_pct": delta_p95_vs_prev,
                    "delta_p95_vs_g0_pct": delta_p95_vs_g0,
                    "delta_p99_vs_prev_pct": delta_p99_vs_prev,
                    "delta_p99_vs_g0_pct": delta_p99_vs_g0,
                    "dual_objective_delta_vs_g0_pct": (delta_avg_vs_g0 + delta_p95_vs_g0) / 2.0,
                    "p99_constraint_ok": 1 if delta_p99_vs_g0 <= 2.0 else 0,
                }
            )
            prev = record

    pairwise_rows.sort(
        key=lambda r: (
            SCENARIO_ORDER.index(str(r["scenario"])) if str(r["scenario"]) in SCENARIO_ORDER else 999,
            int(r["seed"]),
            int(r["stage_order"]),
        )
    )
    return pairwise_rows


def main() -> int:
    args = parse_args()
    root = repo_root()
    cfg_path = resolve_config_path(args.config, root=root)
    base = load_base_config(cfg_path)
    exp_name = str((base.get("experiment") or {}).get("name", cfg_path.stem))
    base_rate = float((base.get("workload") or {}).get("rate_multiplier", 1.0))
    scenario_rate_map = {scenario: base_rate * factor for scenario, factor in SCENARIO_FACTORS}

    frozen_params = load_frozen_params_from_best(root=root, best_file=args.hparam_best)
    variant_overrides = build_ablation_scenario_variants(
        base_rate_multiplier=base_rate,
        scenario_factors=SCENARIO_FACTORS,
        scheme=args.scheme,
        frozen_params=frozen_params,
    )

    variants = build_variant_configs(
        base_config=base,
        base_experiment_name=exp_name,
        variant_overrides=variant_overrides,
        seeds=[int(s) for s in args.seeds],
    )
    metrics, failures = run_with_temp_configs(
        python_bin=args.python,
        root=root,
        variants=variants,
    )

    records = _build_run_records(metrics=metrics, scenario_rate_map=scenario_rate_map)
    agg_rows = _aggregate_by_scenario_stage(records=records, scheme=args.scheme)

    if args.scheme == "progressive":
        _attach_progressive_deltas(agg_rows)

    for err in failures:
        scenario, stage = _split_compound_label(str(err["label"]))
        agg_rows.append(
            {
                "label": err["label"],
                "scenario": scenario,
                "stage": stage,
                "runs": 0,
                "rate_multiplier": scenario_rate_map.get(scenario, ""),
                "avg_e2e_ms_mean": "",
                "avg_e2e_ms_std": "",
                "p50_ms_mean": "",
                "p50_ms_std": "",
                "p95_ms_mean": "",
                "p95_ms_std": "",
                "p99_ms_mean": "",
                "p99_ms_std": "",
                "success_rate_mean": "",
                "success_rate_std": "",
                "cold_start_share_mean": "",
                "cold_start_share_std": "",
                "status": f"failed(seed={err['seed']}): {err['error']}",
                "scheme": args.scheme,
                "stage_id": _row_stage_id(scheme=args.scheme, stage=stage),
                "stage_order": _row_order_key(scheme=args.scheme, stage=stage),
                "delta_avg_vs_prev_pct": "",
                "delta_avg_vs_g0_pct": "",
                "delta_p95_vs_prev_pct": "",
                "delta_p95_vs_g0_pct": "",
                "delta_p99_vs_prev_pct": "",
                "delta_p99_vs_g0_pct": "",
                "dual_objective_delta_vs_g0_pct": "",
                "p99_constraint_ok": "",
            }
        )

    agg_rows.sort(
        key=lambda r: (
            SCENARIO_ORDER.index(str(r["scenario"])) if str(r["scenario"]) in SCENARIO_ORDER else 999,
            int(r.get("stage_order", 999)),
        )
    )

    pairwise_rows = _build_pairwise_rows(records=records, scheme=args.scheme)

    results_dir = ensure_results_dir(root=root, relative=args.results_dir)
    metrics_csv = results_dir / "ablation_metrics.csv"
    pairwise_csv = results_dir / "ablation_pairwise.csv"
    write_csv(metrics_csv, agg_rows)
    write_csv(pairwise_csv, pairwise_rows)
    print(f"saved metrics: {metrics_csv}")
    print(f"saved pairwise: {pairwise_csv}")

    ok_rows = [r for r in agg_rows if r.get("status") == "ok"]
    if ok_rows:
        e2e_path = results_dir / "ablation_e2e.png"
        plot_progressive_e2e_by_scenario(
            ok_rows,
            out_path=e2e_path,
            title="HPWP 3-Stage Ablation by Load (Avg/P95/P99)",
            scenario_order=SCENARIO_ORDER,
        )
        print(f"saved figure: {e2e_path}")
    else:
        print("no successful runs to plot")

    if failures:
        print(f"failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
