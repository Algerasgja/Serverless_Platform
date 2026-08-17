from __future__ import annotations

import argparse
import csv
import itertools
import math
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
    "hpwp_alpha_exp": 0.8,
    "hpwp_forget_gamma": 0.9,
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
    "hpwp_alpha_exp": 0.8,
    "hpwp_forget_gamma": 0.9,
    "hpwp_alpha_stable": 0.08,
}

CORE_TUNED_KEYS = [
    "hpwp_sched_eta_exec",
    "hpwp_horizon_alpha",
    "hpwp_rho_mass",
    "hpwp_tau_p",
]

DRIFT_TUNED_KEYS = [
    "hpwp_forget_gamma",
    "hpwp_beta_hi",
    "hpwp_beta_lo",
    "hpwp_alpha_exp",
]


def _tuned_keys_for(tune_set: str) -> list[str]:
    if str(tune_set).lower() == "drift":
        return list(DRIFT_TUNED_KEYS)
    return list(CORE_TUNED_KEYS)


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
        "--tune-set",
        choices=["core", "drift"],
        default="core",
        help="Parameter family to tune: core=(eta,horizon,rho,tau), drift=(forget,beta_hi,beta_lo,alpha_exp).",
    )
    parser.add_argument(
        "--anchor-source",
        choices=["best", "default"],
        default="best",
        help="Anchor source for candidate generation. Use default to ignore results/hparam_best.yaml.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260313,
        help="Sampling seed for deterministic subset selection.",
    )
    parser.add_argument(
        "--display-points",
        type=int,
        default=24,
        help="Number of curated tactical display points (including best).",
    )
    parser.add_argument(
        "--curate-primary",
        choices=["p95", "avg"],
        default="p95",
        help="Primary ranking metric for curated display selection.",
    )
    parser.add_argument(
        "--curate-secondary",
        choices=["avg", "p95"],
        default="avg",
        help="Secondary ranking metric for curated display selection.",
    )
    parser.add_argument(
        "--curate-only",
        action="store_true",
        help="Skip simulation reruns and curate directly from existing results/hparam_metrics.csv.",
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
    tune_set: str,
) -> list[dict[str, float]]:
    tune_set = str(tune_set).lower()
    key_order = _tuned_keys_for(tune_set)
    all_points: list[dict[str, float]] = []

    if tune_set == "drift":
        gamma0 = float(anchor.get("hpwp_forget_gamma", HPWP_DEFAULT_POINT["hpwp_forget_gamma"]))
        beta_hi0 = float(anchor.get("hpwp_beta_hi", HPWP_DEFAULT_POINT["hpwp_beta_hi"]))
        beta_lo0 = float(anchor.get("hpwp_beta_lo", HPWP_DEFAULT_POINT["hpwp_beta_lo"]))
        alpha_exp0 = float(anchor.get("hpwp_alpha_exp", HPWP_DEFAULT_POINT["hpwp_alpha_exp"]))

        gamma_candidates = _round_unique(
            [gamma0 - 0.15, gamma0 - 0.08, gamma0 - 0.03, gamma0, gamma0 + 0.03, gamma0 + 0.06],
            lo=0.65,
            hi=0.99,
            ndigits=3,
        )
        beta_hi_candidates = _round_unique(
            [beta_hi0 * 0.6, beta_hi0 * 0.8, beta_hi0, beta_hi0 * 1.25, beta_hi0 * 1.5, beta_hi0 * 1.8],
            lo=20.0,
            hi=140.0,
            ndigits=3,
        )
        beta_lo_candidates = _round_unique(
            [beta_lo0 * 0.6, beta_lo0 * 0.8, beta_lo0, beta_lo0 * 1.2, beta_lo0 * 1.4, beta_lo0 * 1.6],
            lo=4.0,
            hi=30.0,
            ndigits=3,
        )
        alpha_exp_candidates = _round_unique(
            [alpha_exp0 * 0.75, alpha_exp0 * 0.875, alpha_exp0, alpha_exp0 * 1.06, alpha_exp0 * 1.15],
            lo=0.45,
            hi=0.98,
            ndigits=3,
        )

        for gamma, beta_hi, beta_lo, alpha_exp in itertools.product(
            gamma_candidates,
            beta_hi_candidates,
            beta_lo_candidates,
            alpha_exp_candidates,
        ):
            if beta_hi <= beta_lo:
                continue
            ratio = float(beta_hi) / max(1e-9, float(beta_lo))
            if ratio < 3.0 or ratio > 10.0:
                continue
            all_points.append(
                {
                    "hpwp_forget_gamma": float(gamma),
                    "hpwp_beta_hi": float(beta_hi),
                    "hpwp_beta_lo": float(beta_lo),
                    "hpwp_alpha_exp": float(alpha_exp),
                }
            )
        anchor_point = {
            "hpwp_forget_gamma": gamma0,
            "hpwp_beta_hi": beta_hi0,
            "hpwp_beta_lo": beta_lo0,
            "hpwp_alpha_exp": alpha_exp0,
        }
    else:
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
        anchor_point = {
            "hpwp_sched_eta_exec": eta0,
            "hpwp_horizon_alpha": h0,
            "hpwp_rho_mass": rho0,
            "hpwp_tau_p": tau0,
        }

    if not all_points:
        return [anchor_point]

    trial_count = max(1, min(int(trial_count), len(all_points)))
    all_keys = {tuple(point[k] for k in key_order): point for point in all_points}
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
    if tune_set == "drift":
        coverage_specs = [
            (
                min(gamma_candidates),
                min(beta_hi_candidates),
                min(beta_lo_candidates),
                min(alpha_exp_candidates),
            ),
            (
                max(gamma_candidates),
                max(beta_hi_candidates),
                max(beta_lo_candidates),
                max(alpha_exp_candidates),
            ),
            (
                gamma_candidates[len(gamma_candidates) // 2],
                beta_hi0,
                beta_lo0,
                alpha_exp0,
            ),
        ]
    else:
        coverage_specs = [
            (min(eta_candidates), min(horizon_candidates), min(rho_candidates), min(tau_candidates)),
            (max(eta_candidates), max(horizon_candidates), max(rho_candidates), max(tau_candidates)),
            (eta_candidates[len(eta_candidates) // 2], horizon_candidates[len(horizon_candidates) // 2], rho0, tau0),
        ]

    for dims in coverage_specs:
        if len(selected) >= trial_count:
            break
        key = tuple(float(v) for v in dims)
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
    tune_set: str,
    tuned_keys: list[str],
) -> list[dict[str, Any]]:
    points = build_hparam_parameter_sets(
        trial_count=trial_count,
        random_seed=random_seed,
        anchor={k: float(v) for k, v in fixed_autoscaler.items() if k in tuned_keys},
        tune_set=tune_set,
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


def _safe_float(raw: Any, *, default: float = math.nan) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _rank_key(
    row: dict[str, Any],
    *,
    primary: str = "p95",
    secondary: str = "avg",
) -> tuple[float, float, str]:
    field_map = {
        "avg": "avg_e2e_ms_mean",
        "p95": "p95_ms_mean",
    }
    p = field_map[primary]
    s = field_map[secondary]
    return (
        _safe_float(row.get(p), default=1e18),
        _safe_float(row.get(s), default=1e18),
        str(row.get("label", "")),
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    ratio = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * ratio


def _fps_pick(
    rows: list[dict[str, Any]],
    *,
    count: int,
    x_key: str = "avg_e2e_ms_mean",
    y_key: str = "p95_ms_mean",
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)

    x_vals = [_safe_float(r.get(x_key), default=0.0) for r in rows]
    y_vals = [_safe_float(r.get(y_key), default=0.0) for r in rows]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def _coord(r: dict[str, Any]) -> tuple[float, float]:
        x = (_safe_float(r.get(x_key), default=min_x) - min_x) / span_x
        y = (_safe_float(r.get(y_key), default=min_y) - min_y) / span_y
        return (x, y)

    coords = {str(r.get("label", id(r))): _coord(r) for r in rows}

    # Deterministic seed point: best by (P95, Avg) within this bucket.
    start = min(rows, key=lambda r: (_safe_float(r.get("p95_ms_mean"), default=1e18), _safe_float(r.get("avg_e2e_ms_mean"), default=1e18), str(r.get("label", ""))))
    selected: list[dict[str, Any]] = [start]
    selected_keys = {str(start.get("label", id(start)))}

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    while len(selected) < count:
        best_candidate: dict[str, Any] | None = None
        best_min_dist = -1.0
        for candidate in rows:
            key = str(candidate.get("label", id(candidate)))
            if key in selected_keys:
                continue
            c = coords[key]
            nearest = min(
                _dist(c, coords[str(s.get("label", id(s)))])
                for s in selected
            )
            if nearest > best_min_dist + 1e-12:
                best_min_dist = nearest
                best_candidate = candidate
            elif abs(nearest - best_min_dist) <= 1e-12 and best_candidate is not None:
                cand_key = (
                    _safe_float(candidate.get("p95_ms_mean"), default=1e18),
                    _safe_float(candidate.get("avg_e2e_ms_mean"), default=1e18),
                    str(candidate.get("label", "")),
                )
                prev_key = (
                    _safe_float(best_candidate.get("p95_ms_mean"), default=1e18),
                    _safe_float(best_candidate.get("avg_e2e_ms_mean"), default=1e18),
                    str(best_candidate.get("label", "")),
                )
                if cand_key < prev_key:
                    best_candidate = candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        selected_keys.add(str(best_candidate.get("label", id(best_candidate))))
    return selected


def _compute_bucket_targets(non_best_count: int) -> dict[str, int]:
    if non_best_count <= 0:
        return {"near": 0, "mid": 0, "far": 0}
    if non_best_count == 11:
        return {"near": 4, "mid": 4, "far": 3}

    weights = {"near": 4.0, "mid": 4.0, "far": 3.0}
    total_weight = sum(weights.values())
    raw = {
        k: (weights[k] / total_weight) * non_best_count
        for k in ("near", "mid", "far")
    }
    targets = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = non_best_count - sum(targets.values())
    remainders = sorted(
        ((k, raw[k] - targets[k]) for k in targets),
        key=lambda x: (-x[1], x[0]),
    )
    idx = 0
    while remain > 0 and idx < len(remainders):
        targets[remainders[idx][0]] += 1
        remain -= 1
        idx += 1
        if idx >= len(remainders):
            idx = 0
    return targets


def curate_hparam_rows(
    rows: list[dict[str, Any]],
    *,
    display_points: int = 12,
    primary: str = "p95",
    secondary: str = "avg",
    preferred_best_label: str | None = None,
) -> list[dict[str, Any]]:
    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if not ok_rows:
        return []

    ordered = sorted(ok_rows, key=lambda r: _rank_key(r, primary=primary, secondary=secondary))
    best_index = 0
    if preferred_best_label:
        matched_idx = next(
            (idx for idx, r in enumerate(ordered) if str(r.get("label", "")) == str(preferred_best_label)),
            None,
        )
        if matched_idx is not None:
            best_index = int(matched_idx)
    best = dict(ordered[best_index])
    best_label = str(best.get("label", ""))
    worse_than_best = [dict(r) for r in ordered[(best_index + 1) :]]
    better_than_best = [dict(r) for r in ordered[:best_index]]
    non_best = worse_than_best + better_than_best
    if not non_best:
        non_best = [dict(r) for r in ordered if str(r.get("label", "")) != best_label]
    if display_points <= 1 or not non_best:
        best["curation_tier"] = "best"
        best["curation_slot"] = "best"
        best["curation_rank"] = 1
        return [best]

    target_non_best = min(len(non_best), max(0, int(display_points) - 1))
    best_p95 = _safe_float(best.get("p95_ms_mean"), default=0.0)
    for row in non_best:
        row["_delta_p95"] = max(0.0, _safe_float(row.get("p95_ms_mean"), default=best_p95) - best_p95)

    deltas = [float(r["_delta_p95"]) for r in non_best]
    q1 = _quantile(deltas, 1 / 3)
    q2 = _quantile(deltas, 2 / 3)
    near = [r for r in non_best if float(r["_delta_p95"]) <= q1]
    mid = [r for r in non_best if q1 < float(r["_delta_p95"]) <= q2]
    far = [r for r in non_best if float(r["_delta_p95"]) > q2]

    # Degenerate-quantile fallback: split by ordered deltas to keep 3 layers non-empty.
    if not near or not mid or not far:
        by_delta = sorted(
            non_best,
            key=lambda r: (float(r["_delta_p95"]), _safe_float(r.get("avg_e2e_ms_mean"), default=1e18), str(r.get("label", ""))),
        )
        n = len(by_delta)
        cut1 = max(1, n // 3)
        cut2 = max(cut1 + 1, (2 * n) // 3)
        near = by_delta[:cut1]
        mid = by_delta[cut1:cut2]
        far = by_delta[cut2:]

    buckets = {"near": near, "mid": mid, "far": far}
    targets = _compute_bucket_targets(target_non_best)

    picked: list[dict[str, Any]] = []
    picked_keys: set[str] = set()
    for tier in ("near", "mid", "far"):
        want = min(targets[tier], len(buckets[tier]))
        chosen = _fps_pick(buckets[tier], count=want)
        for row in chosen:
            key = str(row.get("label", ""))
            if key in picked_keys or key == best_label:
                continue
            item = dict(row)
            item["curation_tier"] = tier
            item["curation_slot"] = tier
            picked.append(item)
            picked_keys.add(key)

    # Fill deficits using adjacent tiers first.
    tier_adj = {
        "near": ["mid", "far"],
        "mid": ["near", "far"],
        "far": ["mid", "near"],
    }
    slot_counts = {
        "near": sum(1 for r in picked if r.get("curation_slot") == "near"),
        "mid": sum(1 for r in picked if r.get("curation_slot") == "mid"),
        "far": sum(1 for r in picked if r.get("curation_slot") == "far"),
    }
    for slot in ("near", "mid", "far"):
        while slot_counts[slot] < targets[slot] and len(picked) < target_non_best:
            donor: dict[str, Any] | None = None
            for source_tier in tier_adj[slot]:
                candidates = [
                    r
                    for r in buckets[source_tier]
                    if str(r.get("label", "")) not in picked_keys and str(r.get("label", "")) != best_label
                ]
                if not candidates:
                    continue
                donor = min(
                    candidates,
                    key=lambda r: _rank_key(r, primary=primary, secondary=secondary),
                )
                break
            if donor is None:
                break
            item = dict(donor)
            item["curation_tier"] = str(donor.get("curation_tier", "mid"))
            item["curation_slot"] = slot
            key = str(item.get("label", ""))
            picked.append(item)
            picked_keys.add(key)
            slot_counts[slot] += 1

    # Final deterministic fill from remaining worst-space points if still short.
    if len(picked) < target_non_best:
        remaining = [
            r
            for r in non_best
            if str(r.get("label", "")) not in picked_keys and str(r.get("label", "")) != best_label
        ]
        extra = _fps_pick(remaining, count=(target_non_best - len(picked)))
        for row in extra:
            key = str(row.get("label", ""))
            if key in picked_keys:
                continue
            item = dict(row)
            item["curation_tier"] = "far"
            item["curation_slot"] = "far"
            picked.append(item)
            picked_keys.add(key)

    curated = [dict(best)] + picked[:target_non_best]
    curated[0]["curation_tier"] = "best"
    curated[0]["curation_slot"] = "best"

    ranked = sorted(curated, key=lambda r: _rank_key(r, primary=primary, secondary=secondary))
    for idx, row in enumerate(ranked, start=1):
        row["curation_rank"] = idx
        row.pop("_delta_p95", None)
    return ranked


def _value_to_color(values: list[Any], cmap_name: str, *, plt: Any) -> dict[Any, Any]:
    unique = sorted(set(values))
    cmap = plt.get_cmap(cmap_name)
    if len(unique) <= 1:
        only = unique[0] if unique else ""
        return {only: cmap(0.6)}
    return {val: cmap(i / (len(unique) - 1)) for i, val in enumerate(unique)}


def plot_tradeoff_curated(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    if not rows:
        raise RuntimeError("no curated rows to plot")

    rho_values = sorted(
        {
            _safe_float(r.get("hpwp_rho_mass"), default=math.nan)
            for r in rows
            if not math.isnan(_safe_float(r.get("hpwp_rho_mass"), default=math.nan))
        }
    )
    tau_values = sorted(
        {
            _safe_float(r.get("hpwp_tau_p"), default=math.nan)
            for r in rows
            if not math.isnan(_safe_float(r.get("hpwp_tau_p"), default=math.nan))
        }
    )
    if not rho_values:
        rho_values = [0.0]
    if not tau_values:
        tau_values = [0.0]

    rho_color_map = _value_to_color(rho_values, "viridis", plt=plt)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p"]
    tau_marker_map = {
        tau: marker_cycle[idx % len(marker_cycle)]
        for idx, tau in enumerate(tau_values)
    }

    fig, ax = plt.subplots(figsize=(12.6, 7.8))
    for row in rows:
        x = _safe_float(row.get("avg_e2e_ms_mean"), default=0.0)
        y = _safe_float(row.get("p95_ms_mean"), default=0.0)
        label = str(row.get("label", ""))
        is_best = str(row.get("curation_tier", "")) == "best"
        rho = _safe_float(row.get("hpwp_rho_mass"), default=rho_values[0])
        tau = _safe_float(row.get("hpwp_tau_p"), default=tau_values[0])
        color = rho_color_map.get(rho, "#7F7F7F")
        marker = tau_marker_map.get(tau, "o")
        size = 260 if is_best else 95
        edge = "#111111" if is_best else "#222222"
        ax.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=size,
            edgecolors=edge,
            linewidths=1.0 if is_best else 0.7,
            zorder=4 if is_best else 3,
        )
        if is_best:
            ax.scatter(
                x,
                y,
                c="none",
                marker="*",
                s=340,
                edgecolors="#B22222",
                linewidths=1.2,
                zorder=5,
            )
            ax.annotate(
                "BEST",
                (x, y),
                textcoords="offset points",
                xytext=(8, -14),
                fontsize=8.5,
                color="#B22222",
                fontweight="bold",
            )
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#111111")

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="none",
            markerfacecolor=rho_color_map[rho],
            markeredgecolor="#222222",
            markersize=7,
            label=f"R={rho:g}",
        )
        for rho in rho_values
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=tau_marker_map[tau],
            linestyle="",
            color="none",
            markerfacecolor="#B0B0B0",
            markeredgecolor="#222222",
            markersize=7,
            label=f"T={tau:g}",
        )
        for tau in tau_values
    ]
    legend_color = ax.legend(
        handles=color_handles,
        title="Color: rho_mass",
        loc="upper left",
        fontsize=7.5,
        title_fontsize=8.5,
        frameon=True,
    )
    ax.add_artist(legend_color)
    ax.legend(
        handles=shape_handles,
        title="Marker: tau_p",
        loc="upper right",
        fontsize=7.5,
        title_fontsize=8.5,
        frameon=True,
    )

    ax.set_xlabel("AvgE2E (ms)")
    ax.set_ylabel("P95 (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "Legend encoding for tactical display: color=hpwp_rho_mass, marker=hpwp_tau_p",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _load_rows_from_metrics_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing hparam metrics csv: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _load_best_label_from_yaml(results_dir: Path) -> str | None:
    best_path = results_dir / "hparam_best.yaml"
    if not best_path.exists():
        return None
    with best_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    label = payload.get("label")
    if label is None:
        return None
    text = str(label).strip()
    return text or None


def _write_curated_artifacts(
    *,
    curated_rows: list[dict[str, Any]],
    results_dir: Path,
    display_points: int,
    primary: str,
    secondary: str,
) -> None:
    curated_csv_path = results_dir / "hparam_metrics_curated.csv"
    write_csv(curated_csv_path, curated_rows)
    print(f"saved curated metrics: {curated_csv_path}")

    labels_payload = {
        "display_points": int(display_points),
        "primary": str(primary),
        "secondary": str(secondary),
        "best_label": next((str(r.get("label", "")) for r in curated_rows if str(r.get("curation_tier", "")) == "best"), ""),
        "labels": [str(r.get("label", "")) for r in curated_rows],
        "items": [
            {
                "label": str(r.get("label", "")),
                "tier": str(r.get("curation_tier", "")),
                "rank": int(_safe_float(r.get("curation_rank"), default=0)),
            }
            for r in curated_rows
        ],
    }
    labels_path = results_dir / "hparam_curated_labels.yaml"
    with labels_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(labels_payload, f, sort_keys=False, allow_unicode=True)
    print(f"saved curated labels: {labels_path}")

    tradeoff_path = results_dir / "hparam_tradeoff_curated.png"
    plot_tradeoff_curated(
        curated_rows,
        out_path=tradeoff_path,
        title="HParam Tactical Tradeoff (Curated, P95-first)",
    )
    print(f"saved figure: {tradeoff_path}")


def main() -> int:
    args = parse_args()
    root = repo_root()
    cfg_path = resolve_config_path(args.config, root=root)
    base = load_base_config(cfg_path)
    exp_name = str((base.get("experiment") or {}).get("name", cfg_path.stem))
    base_autoscaler = dict(base.get("autoscaler") or {})
    tuned_keys = _tuned_keys_for(str(args.tune_set))
    fixed_autoscaler = {
        k: v
        for k, v in base_autoscaler.items()
        if str(k).startswith("hpwp_")
    }
    if str(args.anchor_source).lower() == "best":
        current_best = _load_current_best_overrides(root=root, results_dir=str(args.results_dir))
        fixed_autoscaler.update({k: v for k, v in current_best.items() if str(k).startswith("hpwp_")})
    fixed_autoscaler["type"] = "hpwp_v1"

    results_dir = ensure_results_dir(root=root, relative=args.results_dir)
    if args.curate_only:
        csv_path = results_dir / "hparam_metrics.csv"
        rows = _load_rows_from_metrics_csv(csv_path)
        ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
        if not ok_rows:
            raise RuntimeError("no successful rows found in existing hparam_metrics.csv")
        print(f"loaded existing metrics: {csv_path}")
    else:
        variant_defs = build_hparam_variants(
            trial_count=args.trial_count,
            random_seed=args.sample_seed,
            fixed_autoscaler=fixed_autoscaler,
            tune_set=str(args.tune_set),
            tuned_keys=tuned_keys,
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
                    "hpwp_alpha_exp": "",
                    "hpwp_forget_gamma": "",
                    "hpwp_alpha_stable": "",
                    "status": f"failed(seed={err['seed']}): {err['error']}",
                }
            )

        csv_path = results_dir / "hparam_metrics.csv"
        write_csv(csv_path, rows)
        print(f"saved metrics: {csv_path}")

        ok_rows = [r for r in rows if r.get("status") == "ok"]
        if ok_rows:
            fig_path = results_dir / "hparam_tradeoff.png"
            plot_tradeoff(
                ok_rows,
                out_path=fig_path,
                title=f"HParam Main Plot (AE vs P95, tune_set={args.tune_set})",
                base_point=None,
                historical_point=None,
            )
            print(f"saved figure: {fig_path}")

            best = sorted(ok_rows, key=_sortable_row_key)[0]
            best_overrides = dict(fixed_autoscaler)
            for key in tuned_keys:
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

    best_label_anchor = _load_best_label_from_yaml(results_dir)
    curated_rows = curate_hparam_rows(
        ok_rows,
        display_points=int(args.display_points),
        primary=str(args.curate_primary),
        secondary=str(args.curate_secondary),
        preferred_best_label=best_label_anchor,
    )
    if not curated_rows:
        raise RuntimeError("curation failed: no curated rows generated")
    _write_curated_artifacts(
        curated_rows=curated_rows,
        results_dir=results_dir,
        display_points=int(args.display_points),
        primary=str(args.curate_primary),
        secondary=str(args.curate_secondary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
