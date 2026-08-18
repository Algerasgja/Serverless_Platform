from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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
from analysis.experiments.compare_experiments import SCENARIO_FACTORS, SCENARIO_ORDER, normalize_scenarios  # noqa: E402


PARAM_LMAX = "lmax"
PARAM_RHO_MASS = "rho_mass"
PARAM_TAU_P = "tau_p"
PARAM_FORGET_GAMMA = "forget_gamma"

PARAM_ORDER = [PARAM_LMAX, PARAM_RHO_MASS, PARAM_TAU_P, PARAM_FORGET_GAMMA]

PARAM_LABELS = {
    PARAM_LMAX: r"$L_{max}$",
    PARAM_RHO_MASS: r"$\rho_{mass}$",
    PARAM_TAU_P: r"$\tau_p$",
    PARAM_FORGET_GAMMA: r"$\gamma$",
}

PARAM_KEYS = {
    PARAM_LMAX: "hpwp_lmax_high",
    PARAM_RHO_MASS: "hpwp_rho_mass",
    PARAM_TAU_P: "hpwp_tau_p",
    PARAM_FORGET_GAMMA: "hpwp_forget_gamma",
}

DEFAULT_PARAM_VALUES = {
    PARAM_LMAX: [1, 2, 3, 4, 5],
    PARAM_RHO_MASS: [0.3, 0.4, 0.5, 0.6, 0.7],
    PARAM_TAU_P: [0.008, 0.012, 0.016, 0.02, 0.032],
    PARAM_FORGET_GAMMA: [0.75, 0.82, 0.87, 0.9, 0.95],
}

SCENARIO_LABELS = {
    "low": "Low",
    "mid": "Medium",
    "high": "High",
}

LATENCY_METRICS = [
    ("avg_e2e_ms", "Avg", "avg_e2e_ms_mean", "avg_e2e_ms_std", "-"),
    ("p95_ms", "P95", "p95_ms_mean", "p95_ms_std", "--"),
    ("p99_ms", "P99", "p99_ms_mean", "p99_ms_std", ":"),
]

SCENARIO_COLORS = {
    "low": "#4C78A8",
    "mid": "#F58518",
    "high": "#E45756",
}

OLD_HPARAM_ARTIFACTS = [
    "hparam_metrics.csv",
    "hparam_metrics_curated.csv",
    "hparam_best.yaml",
    "hparam_curated_labels.yaml",
    "hparam_tradeoff.png",
    "hparam_tradeoff_curated.png",
]


@dataclass(frozen=True)
class HParamVariant:
    label: str
    parameter: str
    parameter_key: str
    parameter_value: float
    scenario: str
    rate_multiplier: float
    overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ConScale one-parameter hyperparameter latency sweeps under low/mid/high load."
    )
    parser.add_argument("--config", default="configs/hyperparameter.yaml", help="Base config path.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Repeated-run seeds.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=["low", "mid", "high"],
        default=None,
        help="Scenario subset. Omitted means low/mid/high all.",
    )
    parser.add_argument(
        "--lmax-values",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_VALUES[PARAM_LMAX],
        help="Values for hpwp_lmax_high. hpwp_lmax_low is clipped to not exceed this value.",
    )
    parser.add_argument(
        "--rho-mass-values",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_VALUES[PARAM_RHO_MASS],
        help="Values for hpwp_rho_mass.",
    )
    parser.add_argument(
        "--tau-p-values",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_VALUES[PARAM_TAU_P],
        help="Values for hpwp_tau_p.",
    )
    parser.add_argument(
        "--forget-gamma-values",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_VALUES[PARAM_FORGET_GAMMA],
        help="Values for hpwp_forget_gamma.",
    )
    parser.add_argument("--results-dir", default="results/hparam/latest", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV files only.")
    return parser.parse_args()


def build_hparam_variants(
    *,
    base_config: dict[str, Any],
    scenarios: list[str],
    param_values: dict[str, list[float]],
) -> list[HParamVariant]:
    base_workload = dict(base_config.get("workload") or {})
    base_autoscaler = dict(base_config.get("autoscaler") or {})
    base_rate = float(base_workload.get("rate_multiplier", 1.0))
    scenario_factors = dict(SCENARIO_FACTORS)
    variants: list[HParamVariant] = []

    for parameter in PARAM_ORDER:
        values = _unique_numeric_values(param_values[parameter])
        for value in values:
            for scenario in scenarios:
                factor = float(scenario_factors[scenario])
                rate_multiplier = base_rate * factor
                autoscaler_overrides: dict[str, Any] = {
                    "type": "hpwp_v1",
                    PARAM_KEYS[parameter]: _coerce_value(parameter, value),
                }
                if parameter == PARAM_LMAX:
                    lmax_high = max(1, int(round(value)))
                    current_low = int(base_autoscaler.get("hpwp_lmax_low", 1))
                    autoscaler_overrides["hpwp_lmax_low"] = max(1, min(current_low, lmax_high))
                    autoscaler_overrides["hpwp_lmax_high"] = lmax_high

                label = f"{parameter}_{_value_token(value)}_{scenario}"
                variants.append(
                    HParamVariant(
                        label=label,
                        parameter=parameter,
                        parameter_key=PARAM_KEYS[parameter],
                        parameter_value=float(value),
                        scenario=scenario,
                        rate_multiplier=rate_multiplier,
                        overrides={
                            "autoscaler": autoscaler_overrides,
                            "workload": {"rate_multiplier": rate_multiplier},
                        },
                    )
                )
    return variants


def aggregate_hparam_rows(*, by_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in by_seed_rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        key = (str(row["parameter"]), float(row["parameter_value"]), str(row["scenario"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (parameter, parameter_value, scenario), group in grouped.items():
        row: dict[str, Any] = {
            "parameter": parameter,
            "parameter_label": PARAM_LABELS.get(parameter, parameter),
            "parameter_key": PARAM_KEYS.get(parameter, parameter),
            "parameter_value": parameter_value,
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "rate_multiplier": float(group[0]["rate_multiplier"]),
            "runs": len(group),
            "status": "ok",
        }
        for _, _, mean_key, std_key, _ in LATENCY_METRICS:
            raw_name = mean_key.removesuffix("_mean")
            mean, std = _mean_std([float(item[raw_name]) for item in group])
            row[mean_key] = mean
            row[std_key] = std
        success_mean, success_std = _mean_std([float(item["success_rate"]) for item in group])
        cold_mean, cold_std = _mean_std([float(item["cold_start_share"]) for item in group])
        row["success_rate_mean"] = success_mean
        row["success_rate_std"] = success_std
        row["cold_start_share_mean"] = cold_mean
        row["cold_start_share_std"] = cold_std
        rows.append(row)

    rows.sort(key=_summary_sort_key)
    return rows


def build_plot_series_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        for metric_id, metric_label, mean_key, std_key, _ in LATENCY_METRICS:
            rows.append(
                {
                    "parameter": row["parameter"],
                    "parameter_label": row["parameter_label"],
                    "parameter_value": row["parameter_value"],
                    "scenario": row["scenario"],
                    "scenario_label": row["scenario_label"],
                    "latency_metric": metric_id,
                    "latency_metric_label": metric_label,
                    "latency_ms_mean": row[mean_key],
                    "latency_ms_std": row[std_key],
                    "runs": row["runs"],
                }
            )
    rows.sort(key=_plot_series_sort_key)
    return rows


def plot_latency_trends(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not summary_rows:
        raise RuntimeError("no summary rows available for plotting")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, parameter in zip(axes_flat, PARAM_ORDER):
        parameter_rows = [row for row in summary_rows if str(row["parameter"]) == parameter]
        values = sorted({float(row["parameter_value"]) for row in parameter_rows})
        for scenario in SCENARIO_ORDER:
            scenario_rows = [row for row in parameter_rows if str(row["scenario"]) == scenario]
            if not scenario_rows:
                continue
            row_by_value = {float(row["parameter_value"]): row for row in scenario_rows}
            for _, _metric_label, mean_key, _, linestyle in LATENCY_METRICS:
                xs = [value for value in values if value in row_by_value]
                ys = [float(row_by_value[value][mean_key]) for value in xs]
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    linestyle=linestyle,
                    marker="o",
                    markersize=3.8,
                    linewidth=1.6,
                    color=SCENARIO_COLORS.get(scenario, "#4B5563"),
                    alpha=0.9,
                )

        ax.set_title(PARAM_LABELS.get(parameter, parameter), fontsize=15)
        ax.set_xlabel(PARAM_LABELS.get(parameter, parameter), fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)
        if parameter == PARAM_LMAX:
            ax.set_xticks(values)
            ax.set_xticklabels([str(int(v)) for v in values])

    scenario_handles = [
        Line2D([0], [0], color=SCENARIO_COLORS[scenario], lw=2.0, label=SCENARIO_LABELS.get(scenario, scenario))
        for scenario in SCENARIO_ORDER
    ]
    metric_handles = [
        Line2D([0], [0], color="#222222", lw=2.0, linestyle=linestyle, label=metric_label)
        for _, metric_label, _, _, linestyle in LATENCY_METRICS
    ]
    legend1 = fig.legend(
        handles=scenario_handles,
        loc="lower center",
        bbox_to_anchor=(0.34, 0.01),
        ncol=len(scenario_handles),
        frameon=False,
        fontsize=11,
        title="Load",
        title_fontsize=12,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.68, 0.01),
        ncol=len(metric_handles),
        frameon=False,
        fontsize=11,
        title="Latency Metric",
        title_fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def cleanup_old_outputs(results_dir: Path) -> None:
    for name in OLD_HPARAM_ARTIFACTS:
        path = results_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def _build_by_seed_rows(*, metrics: list[Any], variants: list[HParamVariant]) -> list[dict[str, Any]]:
    meta = {variant.label: variant for variant in variants}
    rows: list[dict[str, Any]] = []
    for item in metrics:
        variant = meta[str(item.label)]
        rows.append(
            {
                "label": str(item.label),
                "parameter": variant.parameter,
                "parameter_label": PARAM_LABELS.get(variant.parameter, variant.parameter),
                "parameter_key": variant.parameter_key,
                "parameter_value": variant.parameter_value,
                "scenario": variant.scenario,
                "scenario_label": SCENARIO_LABELS.get(variant.scenario, variant.scenario),
                "rate_multiplier": variant.rate_multiplier,
                "seed": int(item.seed),
                "run_dir": str(item.run_dir),
                "avg_e2e_ms": float(item.avg_e2e_ms),
                "p50_ms": float(item.p50_ms),
                "p95_ms": float(item.p95_ms),
                "p99_ms": float(item.p99_ms),
                "success_rate": float(item.success_rate),
                "cold_start_share": float(item.cold_start_share),
                "status": "ok",
            }
        )
    rows.sort(key=_by_seed_sort_key)
    return rows


def _failure_rows(*, failures: list[dict[str, Any]], variants: list[HParamVariant]) -> list[dict[str, Any]]:
    meta = {variant.label: variant for variant in variants}
    rows: list[dict[str, Any]] = []
    for failure in failures:
        variant = meta.get(str(failure["label"]))
        if variant is None:
            continue
        rows.append(
            {
                "label": str(failure["label"]),
                "parameter": variant.parameter,
                "parameter_label": PARAM_LABELS.get(variant.parameter, variant.parameter),
                "parameter_key": variant.parameter_key,
                "parameter_value": variant.parameter_value,
                "scenario": variant.scenario,
                "scenario_label": SCENARIO_LABELS.get(variant.scenario, variant.scenario),
                "rate_multiplier": variant.rate_multiplier,
                "seed": int(failure["seed"]),
                "run_dir": "",
                "avg_e2e_ms": "",
                "p50_ms": "",
                "p95_ms": "",
                "p99_ms": "",
                "success_rate": "",
                "cold_start_share": "",
                "status": f"failed: {failure['error']}",
            }
        )
    return rows


def _unique_numeric_values(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _coerce_value(parameter: str, value: float) -> int | float:
    if parameter == PARAM_LMAX:
        return max(1, int(round(value)))
    return float(value)


def _value_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _mean_std(values: list[float]) -> tuple[float, float]:
    cleaned = [float(value) for value in values if not math.isnan(float(value))]
    if not cleaned:
        return math.nan, math.nan
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / (len(cleaned) - 1)
    return mean, math.sqrt(variance)


def _scenario_index(scenario: str) -> int:
    return SCENARIO_ORDER.index(scenario) if scenario in SCENARIO_ORDER else 999


def _parameter_index(parameter: str) -> int:
    return PARAM_ORDER.index(parameter) if parameter in PARAM_ORDER else 999


def _summary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _parameter_index(str(row["parameter"])),
        float(row["parameter_value"]),
        _scenario_index(str(row["scenario"])),
    )


def _plot_series_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metric_order = {metric_id: idx for idx, (metric_id, *_rest) in enumerate(LATENCY_METRICS)}
    return (
        _parameter_index(str(row["parameter"])),
        float(row["parameter_value"]),
        _scenario_index(str(row["scenario"])),
        metric_order.get(str(row["latency_metric"]), 999),
    )


def _by_seed_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _parameter_index(str(row["parameter"])),
        float(row["parameter_value"]),
        _scenario_index(str(row["scenario"])),
        int(row["seed"]),
    )


def main() -> int:
    args = parse_args()
    root = repo_root()
    cfg_path = resolve_config_path(args.config, root=root)
    base_config = load_base_config(cfg_path)
    exp_name = str((base_config.get("experiment") or {}).get("name", cfg_path.stem))
    scenarios = normalize_scenarios(args.scenarios)
    selected_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(scenarios)]
    param_values = {
        PARAM_LMAX: [float(value) for value in args.lmax_values],
        PARAM_RHO_MASS: [float(value) for value in args.rho_mass_values],
        PARAM_TAU_P: [float(value) for value in args.tau_p_values],
        PARAM_FORGET_GAMMA: [float(value) for value in args.forget_gamma_values],
    }

    variants = build_hparam_variants(
        base_config=base_config,
        scenarios=selected_scenarios,
        param_values=param_values,
    )
    variant_defs = [{"label": variant.label, "overrides": variant.overrides} for variant in variants]
    sim_variants = build_variant_configs(
        base_config=base_config,
        base_experiment_name=exp_name,
        variant_overrides=variant_defs,
        seeds=[int(seed) for seed in args.seeds],
    )

    results_dir = ensure_results_dir(root=root, relative=args.results_dir)
    cleanup_old_outputs(results_dir)

    metrics, failures = run_with_temp_configs(
        python_bin=args.python,
        root=root,
        variants=sim_variants,
    )
    by_seed_rows = _build_by_seed_rows(metrics=metrics, variants=variants)
    by_seed_rows.extend(_failure_rows(failures=failures, variants=variants))
    summary_rows = aggregate_hparam_rows(by_seed_rows=by_seed_rows)
    plot_series_rows = build_plot_series_rows(summary_rows)

    by_seed_path = results_dir / "hparam_latency_by_seed.csv"
    summary_path = results_dir / "hparam_latency_summary.csv"
    plot_series_path = results_dir / "hparam_plot_series.csv"
    write_csv(by_seed_path, by_seed_rows)
    write_csv(summary_path, summary_rows)
    write_csv(plot_series_path, plot_series_rows)
    print(f"saved by-seed metrics: {by_seed_path}")
    print(f"saved summary metrics: {summary_path}")
    print(f"saved plot series: {plot_series_path}")

    if not args.no_plot:
        fig_path = results_dir / "hparam_latency_trends.png"
        plot_latency_trends(summary_rows, fig_path)
        print(f"saved figure: {fig_path}")

    if failures:
        print(f"failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
