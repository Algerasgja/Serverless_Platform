from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiment_common import (  # noqa: E402
    aggregate_metrics,
    build_variant_configs,
    ensure_results_dir,
    load_base_config,
    plot_hparam_rank_strip,
    plot_tradeoff,
    repo_root,
    resolve_config_path,
    run_with_temp_configs,
    write_csv,
)


# ConScale (autoscaler.type=hpwp_v1) baseline in current platform config.
HPWP_DEFAULT_POINT = {
    "hpwp_sched_eta_exec": 0.4,
    "hpwp_horizon_alpha": 1.2,
    "hpwp_rho_mass": 0.5,
    "hpwp_tau_p": 0.02,
    "hpwp_beta_hi": 60.0,
    "hpwp_beta_lo": 10.0,
    "hpwp_alpha_stable": 0.12,
}

# Previous best discovered point, treated as an anchor for local refinement.
HPWP_PREV_BEST_POINT = {
    "hpwp_sched_eta_exec": 0.5,
    "hpwp_horizon_alpha": 3.5,
    "hpwp_rho_mass": 0.5,
    "hpwp_tau_p": 0.02,
    "hpwp_beta_hi": 60.0,
    "hpwp_beta_lo": 10.0,
    "hpwp_alpha_stable": 0.08,
}

TUNED_KEYS = [
    "hpwp_sched_eta_exec",
    "hpwp_horizon_alpha",
    "hpwp_rho_mass",
    "hpwp_tau_p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HPWP key-parameter study with fixed outputs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Repeated-run seeds.",
    )
    parser.add_argument(
        "--trial-count",
        type=int,
        default=36,
        help="Number of parameter sets to run (default sampled subset is 36).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260313,
        help="Sampling seed for deterministic subset selection.",
    )
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    return parser.parse_args()


def _round_unique(values: list[float], *, lo: float, hi: float, ndigits: int = 4) -> list[float]:
    clipped = [min(hi, max(lo, float(v))) for v in values]
    uniq = sorted({round(v, ndigits) for v in clipped})
    return [float(v) for v in uniq]


def _load_current_best_overrides(*, root: Path, results_dir: str) -> dict[str, float]:
    best_path = (root / results_dir / "hparam_best.yaml").resolve()
    if not best_path.exists():
        return {}
    with best_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    overrides = payload.get("autoscaler_overrides") or {}
    out: dict[str, float] = {}
    for key, val in overrides.items():
        if key == "type":
            continue
        try:
            out[str(key)] = float(val)
        except Exception:
            continue
    return out


def build_hparam_parameter_sets(
    *,
    trial_count: int,
    random_seed: int,
    anchor: dict[str, float],
) -> list[dict[str, float]]:
    eta0 = float(anchor.get("hpwp_sched_eta_exec", HPWP_PREV_BEST_POINT["hpwp_sched_eta_exec"]))
    h0 = float(anchor.get("hpwp_horizon_alpha", HPWP_PREV_BEST_POINT["hpwp_horizon_alpha"]))
    rho0 = float(anchor.get("hpwp_rho_mass", HPWP_PREV_BEST_POINT["hpwp_rho_mass"]))
    tau0 = float(anchor.get("hpwp_tau_p", HPWP_PREV_BEST_POINT["hpwp_tau_p"]))

    eta_candidates = _round_unique(
        [eta0 * 0.7, eta0 * 0.85, eta0, eta0 * 1.15, eta0 * 1.35, eta0 * 1.7],
        lo=0.1,
        hi=20.0,
        ndigits=3,
    )
    horizon_candidates = _round_unique(
        [h0 * 0.55, h0 * 0.75, h0, h0 * 1.2, h0 * 1.4, h0 * 1.7],
        lo=1.0,
        hi=10.0,
        ndigits=3,
    )
    rho_candidates = _round_unique(
        [rho0 - 0.20, rho0 - 0.10, rho0 - 0.05, rho0, rho0 + 0.05, rho0 + 0.10, rho0 + 0.20],
        lo=0.2,
        hi=0.95,
        ndigits=3,
    )
    tau_candidates = _round_unique(
        [tau0 * 0.4, tau0 * 0.6, tau0 * 0.8, tau0, tau0 * 1.2, tau0 * 1.6, tau0 * 2.0],
        lo=0.001,
        hi=0.20,
        ndigits=4,
    )

    all_points: list[dict[str, float]] = []
    for eta, horizon, rho_mass, tau_p in itertools.product(
        eta_candidates,
        horizon_candidates,
        rho_candidates,
        tau_candidates,
    ):
        all_points.append(
            {
                "hpwp_sched_eta_exec": float(eta),
                "hpwp_horizon_alpha": float(horizon),
                "hpwp_rho_mass": float(rho_mass),
                "hpwp_tau_p": float(tau_p),
            }
        )

    trial_count = max(1, min(int(trial_count), len(all_points)))
    key_order = list(TUNED_KEYS)
    all_keys = {tuple(point[k] for k in key_order): point for point in all_points}
    anchor_point = {
        "hpwp_sched_eta_exec": eta0,
        "hpwp_horizon_alpha": h0,
        "hpwp_rho_mass": rho0,
        "hpwp_tau_p": tau0,
    }
    anchor_key = tuple(anchor_point[k] for k in key_order)
    if anchor_key not in all_keys:
        anchor_key = min(
            all_keys.keys(),
            key=lambda k: sum((float(k[i]) - float(anchor_key[i])) ** 2 for i in range(len(k))),
        )
        anchor_point = dict(all_keys[anchor_key])

    if trial_count == 1:
        return [anchor_point]

    rng = random.Random(int(random_seed))
    selected: list[dict[str, float]] = [anchor_point]
    selected_keys = {anchor_key}

    coverage_specs = [
        (min(eta_candidates), min(horizon_candidates), min(rho_candidates), min(tau_candidates)),
        (max(eta_candidates), max(horizon_candidates), max(rho_candidates), max(tau_candidates)),
        (eta_candidates[len(eta_candidates) // 2], horizon_candidates[len(horizon_candidates) // 2], rho0, tau0),
    ]
    for eta, horizon, rho_mass, tau_p in coverage_specs:
        if len(selected) >= trial_count:
            break
        key = (
            float(eta),
            float(horizon),
            float(rho_mass),
            float(tau_p),
        )
        point = all_keys.get(key)
        if point is None or key in selected_keys:
            continue
        selected.append(point)
        selected_keys.add(key)

    if len(selected) < trial_count:
        pool = [point for key, point in all_keys.items() if key not in selected_keys]
        rng.shuffle(pool)
        selected.extend(pool[: (trial_count - len(selected))])
    return selected[:trial_count]


def build_hparam_variants(
    *,
    trial_count: int,
    random_seed: int,
    fixed_autoscaler: dict[str, Any],
) -> list[dict[str, Any]]:
    points = build_hparam_parameter_sets(
        trial_count=trial_count,
        random_seed=random_seed,
        anchor={k: float(v) for k, v in fixed_autoscaler.items() if k in TUNED_KEYS},
    )
    variants: list[dict[str, Any]] = []
    for idx, point in enumerate(points, start=1):
        label = f"hpwp_{idx:02d}"
        overrides = {"autoscaler": dict(fixed_autoscaler)}
        overrides["autoscaler"]["type"] = "hpwp_v1"
        overrides["autoscaler"].update(point)
        variants.append(
            {
                "label": label,
                "overrides": overrides,
                "params": {
                    k: float(v)
                    for k, v in {
                        **{kk: vv for kk, vv in fixed_autoscaler.items() if kk.startswith("hpwp_")},
                        **point,
                    }.items()
                    if k.startswith("hpwp_")
                },
            }
        )
    return variants


def _sortable_row_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row.get("avg_e2e_ms_mean", 1e18)), float(row.get("p95_ms_mean", 1e18)))


def main() -> int:
    args = parse_args()
    root = repo_root()
    cfg_path = resolve_config_path(args.config, root=root)
    base = load_base_config(cfg_path)
    exp_name = str((base.get("experiment") or {}).get("name", cfg_path.stem))
    base_autoscaler = dict(base.get("autoscaler") or {})
    current_best = _load_current_best_overrides(root=root, results_dir=str(args.results_dir))
    fixed_autoscaler = {
        k: v
        for k, v in base_autoscaler.items()
        if str(k).startswith("hpwp_")
    }
    fixed_autoscaler.update({k: v for k, v in current_best.items() if str(k).startswith("hpwp_")})
    fixed_autoscaler["type"] = "hpwp_v1"

    variant_defs = build_hparam_variants(
        trial_count=args.trial_count,
        random_seed=args.sample_seed,
        fixed_autoscaler=fixed_autoscaler,
    )
    variants = build_variant_configs(
        base_config=base,
        base_experiment_name=exp_name,
        variant_overrides=variant_defs,
        seeds=[int(s) for s in args.seeds],
    )
    metrics, failures = run_with_temp_configs(
        python_bin=args.python,
        root=root,
        variants=variants,
    )

    rows = aggregate_metrics(metrics)
    param_map = {item["label"]: item["params"] for item in variant_defs}
    for row in rows:
        params = param_map.get(str(row["label"]), {})
        row.update(params)
        row["status"] = "ok"
    rows.sort(key=_sortable_row_key)

    for err in failures:
        rows.append(
            {
                "label": err["label"],
                "runs": 0,
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
                "hpwp_sched_eta_exec": "",
                "hpwp_horizon_alpha": "",
                "hpwp_rho_mass": "",
                "hpwp_tau_p": "",
                "hpwp_beta_hi": "",
                "hpwp_beta_lo": "",
                "hpwp_alpha_stable": "",
                "status": f"failed(seed={err['seed']}): {err['error']}",
            }
        )

    results_dir = ensure_results_dir(root=root, relative=args.results_dir)
    csv_path = results_dir / "hparam_metrics.csv"
    write_csv(csv_path, rows)
    print(f"saved metrics: {csv_path}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        fig_path = results_dir / "hparam_tradeoff.png"
        plot_tradeoff(
            ok_rows,
            out_path=fig_path,
            title="HParam Main Plot (AE vs P95, Eta/Horizon/Rho/Tau)",
            base_point=None,
            historical_point=None,
        )
        print(f"saved figure: {fig_path}")
        rank_fig_path = results_dir / "hparam_rank_strip.png"
        plot_hparam_rank_strip(
            ok_rows,
            out_path=rank_fig_path,
            title="HParam Rank + Parameter Heat Strips",
        )
        print(f"saved figure: {rank_fig_path}")

        best = sorted(ok_rows, key=_sortable_row_key)[0]
        best_overrides = dict(fixed_autoscaler)
        for key in TUNED_KEYS:
            best_overrides[key] = float(best[key])
        best_overrides["type"] = "hpwp_v1"
        best_payload = {
            "label": best["label"],
            "metrics": {
                "avg_e2e_ms_mean": float(best["avg_e2e_ms_mean"]),
                "p50_ms_mean": float(best["p50_ms_mean"]),
                "p95_ms_mean": float(best["p95_ms_mean"]),
                "p99_ms_mean": float(best["p99_ms_mean"]),
                "success_rate_mean": float(best["success_rate_mean"]),
                "cold_start_share_mean": float(best["cold_start_share_mean"]),
            },
            "autoscaler_overrides": best_overrides,
        }
        best_path = results_dir / "hparam_best.yaml"
        with best_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(best_payload, f, sort_keys=False)
        print(f"saved best config: {best_path}")
    else:
        print("no successful runs to plot or rank")

    if failures:
        print(f"failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
