from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiment_common import (  # noqa: E402
    aggregate_metrics,
    build_variant_configs,
    ensure_results_dir,
    load_base_config,
    repo_root,
    resolve_config_path,
    run_with_temp_configs,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-parameter sensitivity study on top of default config. "
            "Only the target parameter changes; all others stay identical to base config."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path (default config).")
    parser.add_argument(
        "--param",
        required=True,
        help=(
            "Parameter name. "
            "Use 'hpwp_tau_p' (auto-resolved to autoscaler.hpwp_tau_p) or explicit dotted path like "
            "'autoscaler.hpwp_tau_p'."
        ),
    )
    parser.add_argument(
        "--values",
        required=True,
        help=(
            "Parameter values. "
            "Formats: '0.1,0.2,0.3' or range 'start:end:step' (inclusive if hits end)."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Repeated-run seeds.",
    )
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    return parser.parse_args()


def _parse_values(spec: str) -> list[float]:
    text = str(spec).strip()
    if not text:
        raise ValueError("empty --values")
    if "," in text:
        out = [float(x.strip()) for x in text.split(",") if x.strip()]
        if not out:
            raise ValueError("invalid --values list")
        return out
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError("range format must be start:end:step")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2])
        if step == 0:
            raise ValueError("range step cannot be 0")
        values: list[float] = []
        cur = start
        eps = abs(step) * 1e-9
        if step > 0:
            while cur <= end + eps:
                values.append(float(cur))
                cur += step
        else:
            while cur >= end - eps:
                values.append(float(cur))
                cur += step
        if not values:
            raise ValueError("range produced no values")
        return values
    return [float(text)]


def _slugify(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("_")


def _resolve_param_path(base: dict[str, Any], raw_param: str) -> list[str]:
    token = str(raw_param).strip()
    if "." in token:
        path = token.split(".")
        _ = _get_nested(base, path)  # validate existence
        return path
    autoscaler = base.get("autoscaler")
    if isinstance(autoscaler, dict) and token in autoscaler:
        return ["autoscaler", token]
    if token in base:
        return [token]
    raise KeyError(f"parameter not found in base config: {raw_param}")


def _get_nested(obj: dict[str, Any], path: list[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(path))
        cur = cur[key]
    return cur


def _build_nested_override(path: list[str], value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {path[-1]: value}
    for key in reversed(path[:-1]):
        out = {key: out}
    return out


def _cast_like(value: float, sample: Any) -> Any:
    if isinstance(sample, bool):
        # For bool params, treat >=0.5 as True, else False.
        return bool(value >= 0.5)
    if isinstance(sample, int) and not isinstance(sample, bool):
        return int(round(value))
    if isinstance(sample, float):
        return float(value)
    # Fallback: keep numeric float.
    return float(value)


def _param_value_from_label(label: str, mapping: dict[str, float]) -> float:
    if label not in mapping:
        raise KeyError(f"missing label mapping for {label}")
    return float(mapping[label])


def _plot_sensitivity(rows: list[dict[str, Any]], *, out_path: Path, param_key: str) -> None:
    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if not ok_rows:
        raise RuntimeError("no successful rows to plot")

    x = [float(r["param_value"]) for r in ok_rows]
    avg = [float(r["avg_e2e_ms_mean"]) for r in ok_rows]
    p95 = [float(r["p95_ms_mean"]) for r in ok_rows]
    p99 = [float(r["p99_ms_mean"]) for r in ok_rows]

    avg_std = [float(r["avg_e2e_ms_std"]) for r in ok_rows]
    p95_std = [float(r["p95_ms_std"]) for r in ok_rows]
    p99_std = [float(r["p99_ms_std"]) for r in ok_rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.errorbar(x, avg, yerr=avg_std, marker="o", linewidth=2.0, capsize=3, label="AvgE2E")
    ax.errorbar(x, p95, yerr=p95_std, marker="s", linewidth=2.0, capsize=3, label="P95")
    ax.errorbar(x, p99, yerr=p99_std, marker="^", linewidth=2.0, capsize=3, label="P99")

    ax.set_xlabel(param_key)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Single-Parameter Sensitivity ({param_key})")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = repo_root()
    cfg_path = resolve_config_path(args.config, root=root)
    base = load_base_config(cfg_path)
    exp_name = str((base.get("experiment") or {}).get("name", cfg_path.stem))

    path = _resolve_param_path(base, args.param)
    sample_val = _get_nested(base, path)
    raw_values = _parse_values(args.values)
    cast_values = [_cast_like(v, sample_val) for v in raw_values]

    # Remove duplicates while keeping order.
    seen: set[str] = set()
    unique_values: list[Any] = []
    for v in cast_values:
        key = repr(v)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(v)

    slug = _slugify(".".join(path))
    variant_defs: list[dict[str, Any]] = []
    label_to_value: dict[str, float] = {}
    for idx, val in enumerate(unique_values, start=1):
        label = f"{slug}_{idx:02d}"
        override = _build_nested_override(path, val)
        variant_defs.append({"label": label, "overrides": override})
        label_to_value[label] = float(val)

    variants = build_variant_configs(
        base_config=base,
        base_experiment_name=f"{exp_name}_sens_{slug}",
        variant_overrides=variant_defs,
        seeds=[int(s) for s in args.seeds],
    )
    metrics, failures = run_with_temp_configs(
        python_bin=args.python,
        root=root,
        variants=variants,
    )

    rows = aggregate_metrics(metrics)
    for row in rows:
        label = str(row["label"])
        row["param_name"] = ".".join(path)
        row["param_value"] = _param_value_from_label(label, label_to_value)
        row["status"] = "ok"

    for err in failures:
        label = str(err["label"])
        rows.append(
            {
                "label": label,
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
                "param_name": ".".join(path),
                "param_value": label_to_value.get(label, math.nan),
                "status": f"failed(seed={err['seed']}): {err['error']}",
            }
        )

    rows.sort(
        key=lambda r: (
            float(r.get("param_value", math.inf))
            if str(r.get("status", "")).lower() == "ok"
            else math.inf,
            str(r.get("label", "")),
        )
    )

    results_dir = ensure_results_dir(root=root, relative=args.results_dir)
    csv_path = results_dir / f"sensitivity_{slug}.csv"
    write_csv(csv_path, rows)
    print(f"saved metrics: {csv_path}")

    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if not ok_rows:
        raise RuntimeError("all runs failed, no figure generated")
    fig_path = results_dir / f"sensitivity_{slug}.png"
    _plot_sensitivity(ok_rows, out_path=fig_path, param_key=".".join(path))
    print(f"saved figure: {fig_path}")

    if failures:
        print(f"failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

