from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_COMPARE = REPO_ROOT / "results" / "compare" / "compare_metrics.csv"
DEFAULT_STORE_DIR = REPO_ROOT / "results" / "compare" / "tuning_memory"
INDEX_CSV = "index.csv"

SHARED_AUTOSCALER_KEYS = {
    "type",
    "sync_period_sec",
    "min_replicas",
    "max_replicas_per_node",
}

STRATEGY_PARAM_PREFIX = {
    "hpwp": ("hpwp_",),
    "kpa": ("kpa_",),
    "lass": ("lass_",),
    "rl_q": ("rl_",),
    "hptd": ("hptd_",),
    "xanadu": ("xanadu_",),
    "kraken_vomm": ("kraken_",),
    "depth_breadth": ("dbw_", "hpwp_"),
    "oracle": ("oracle_",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist and query autoscaler tuning memory.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    record = sub.add_parser("record", help="Record one tuning round snapshot.")
    record.add_argument("--strategy", required=True, help="Strategy label in compare_metrics.csv, e.g. kraken_vomm")
    record.add_argument("--round-id", default="", help="Round identifier. Empty means auto timestamp id.")
    record.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config yaml path.")
    record.add_argument("--compare-csv", default=str(DEFAULT_COMPARE), help="compare_metrics.csv path.")
    record.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Tuning memory output directory.")
    record.add_argument("--mechanism-change", default="", help="Short mechanism change description.")
    record.add_argument("--notes", default="", help="Short notes.")

    suggest = sub.add_parser("suggest", help="Suggest top historical rounds.")
    suggest.add_argument("--strategy", required=True, help="Strategy label, e.g. kraken_vomm")
    suggest.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Tuning memory output directory.")
    suggest.add_argument(
        "--objective",
        default="near_gap_10",
        choices=("near_gap_10", "min_avg_e2e", "min_cold_rate", "max_prewarm_util"),
        help="Ranking objective.",
    )
    suggest.add_argument("--top", type=int, default=5, help="Top N rows to show.")
    suggest.add_argument("--target-gap", type=float, default=10.0, help="Target avg gap to hpwp for near_gap_10.")

    guard = sub.add_parser("guard", help="Check platform hash consistency before tuning.")
    guard.add_argument("--strategy", required=True, help="Strategy label, e.g. kraken_vomm")
    guard.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config yaml path.")
    guard.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Tuning memory output directory.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_compare_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except Exception:
        return math.nan


def safe_mean(values: list[float]) -> float:
    clean = [x for x in values if not math.isnan(x)]
    if not clean:
        return math.nan
    return float(fmean(clean))


def platform_hash(cfg: dict[str, Any]) -> str:
    platform_view = {
        "workload": cfg.get("workload", {}),
        "runtime": cfg.get("runtime", {}),
        "physical_nodes": cfg.get("physical_nodes", {}),
        "capacity": cfg.get("capacity", {}),
    }
    raw = json.dumps(platform_view, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def pick_strategy_params(strategy: str, cfg: dict[str, Any]) -> dict[str, Any]:
    autoscaler = dict(cfg.get("autoscaler", {}))
    out: dict[str, Any] = {}
    for key in SHARED_AUTOSCALER_KEYS:
        if key in autoscaler:
            out[key] = autoscaler[key]
    prefixes = STRATEGY_PARAM_PREFIX.get(strategy, tuple())
    for key, value in autoscaler.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            out[key] = value
    return dict(sorted(out.items(), key=lambda x: x[0]))


def collect_metrics(rows: list[dict[str, Any]], strategy: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    selected = [r for r in rows if str(r.get("strategy", "")) == strategy]
    by_scenario: dict[str, dict[str, float]] = {}
    for row in selected:
        scenario = str(row.get("scenario", ""))
        by_scenario[scenario] = {
            "avg_e2e_ms_mean": as_float(row, "avg_e2e_ms_mean"),
            "p95_ms_mean": as_float(row, "p95_ms_mean"),
            "p99_ms_mean": as_float(row, "p99_ms_mean"),
            "cold_start_step_rate_mean": as_float(row, "cold_start_step_rate_mean"),
            "prewarm_utilization_mean": as_float(row, "prewarm_utilization_mean"),
            "prewarm_cost_mean": as_float(row, "prewarm_cost_mean"),
            "success_rate_mean": as_float(row, "success_rate_mean"),
        }
    aggregate = {
        "avg_e2e_ms_mean": safe_mean([v["avg_e2e_ms_mean"] for v in by_scenario.values()]),
        "p95_ms_mean": safe_mean([v["p95_ms_mean"] for v in by_scenario.values()]),
        "p99_ms_mean": safe_mean([v["p99_ms_mean"] for v in by_scenario.values()]),
        "cold_start_step_rate_mean": safe_mean([v["cold_start_step_rate_mean"] for v in by_scenario.values()]),
        "prewarm_utilization_mean": safe_mean([v["prewarm_utilization_mean"] for v in by_scenario.values()]),
        "prewarm_cost_mean": safe_mean([v["prewarm_cost_mean"] for v in by_scenario.values()]),
        "success_rate_mean": safe_mean([v["success_rate_mean"] for v in by_scenario.values()]),
    }
    return by_scenario, aggregate


def collect_gap_to_hpwp(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    hpwp = {str(r.get("scenario", "")): r for r in rows if str(r.get("strategy", "")) == "hpwp"}
    target = {str(r.get("scenario", "")): r for r in rows if str(r.get("strategy", "")) == strategy}
    scenario_gap: dict[str, dict[str, float]] = {}
    avg_gap_values: list[float] = []
    p95_gap_values: list[float] = []
    p99_gap_values: list[float] = []
    cold_gap_values: list[float] = []
    for scenario, tr in target.items():
        hr = hpwp.get(scenario)
        if not hr:
            continue
        avg_gap = ((as_float(tr, "avg_e2e_ms_mean") / as_float(hr, "avg_e2e_ms_mean")) - 1.0) * 100.0
        p95_gap = ((as_float(tr, "p95_ms_mean") / as_float(hr, "p95_ms_mean")) - 1.0) * 100.0
        p99_gap = ((as_float(tr, "p99_ms_mean") / as_float(hr, "p99_ms_mean")) - 1.0) * 100.0
        cold_gap = (
            (as_float(tr, "cold_start_step_rate_mean") / as_float(hr, "cold_start_step_rate_mean")) - 1.0
        ) * 100.0
        scenario_gap[scenario] = {
            "avg_gap_pct": avg_gap,
            "p95_gap_pct": p95_gap,
            "p99_gap_pct": p99_gap,
            "cold_gap_pct": cold_gap,
        }
        avg_gap_values.append(avg_gap)
        p95_gap_values.append(p95_gap)
        p99_gap_values.append(p99_gap)
        cold_gap_values.append(cold_gap)
    return {
        "scenario_gap_pct": scenario_gap,
        "avg_gap_pct_mean": safe_mean(avg_gap_values),
        "p95_gap_pct_mean": safe_mean(p95_gap_values),
        "p99_gap_pct_mean": safe_mean(p99_gap_values),
        "cold_gap_pct_mean": safe_mean(cold_gap_values),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def upsert_index(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    # Keep all rows, but replace exact same strategy+round_id if exists.
    filtered = [r for r in existing if not (r.get("strategy") == row["strategy"] and r.get("round_id") == row["round_id"])]
    filtered.append(row)
    fieldnames = [
        "timestamp",
        "strategy",
        "round_id",
        "avg_e2e_ms_mean",
        "p95_ms_mean",
        "p99_ms_mean",
        "cold_start_step_rate_mean",
        "prewarm_utilization_mean",
        "prewarm_cost_mean",
        "avg_gap_pct_mean",
        "p95_gap_pct_mean",
        "p99_gap_pct_mean",
        "cold_gap_pct_mean",
        "platform_hash",
        "mechanism_change",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in filtered:
            writer.writerow({k: item.get(k, "") for k in fieldnames})


def run_record(args: argparse.Namespace) -> None:
    strategy = str(args.strategy).strip()
    config_path = Path(args.config).resolve()
    compare_path = Path(args.compare_csv).resolve()
    store_dir = Path(args.store_dir).resolve()
    now = dt.datetime.now().replace(microsecond=0).isoformat()
    round_id = str(args.round_id).strip() or dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    cfg = read_yaml(config_path)
    rows = read_compare_rows(compare_path)

    by_scenario, aggregate = collect_metrics(rows, strategy)
    if not by_scenario:
        raise ValueError(f"strategy '{strategy}' not found in {compare_path}")
    gap = collect_gap_to_hpwp(rows, strategy)
    params = pick_strategy_params(strategy, cfg)
    phash = platform_hash(cfg)

    payload = {
        "timestamp": now,
        "round_id": round_id,
        "strategy": strategy,
        "config_path": str(config_path),
        "compare_csv_path": str(compare_path),
        "platform_hash": phash,
        "mechanism_change": str(args.mechanism_change),
        "notes": str(args.notes),
        "strategy_params": params,
        "metrics_by_scenario": by_scenario,
        "aggregate_metrics": aggregate,
        "gap_to_hpwp": gap,
    }

    strategy_file = store_dir / f"{strategy}.jsonl"
    append_jsonl(strategy_file, payload)

    index_row = {
        "timestamp": now,
        "strategy": strategy,
        "round_id": round_id,
        "avg_e2e_ms_mean": aggregate["avg_e2e_ms_mean"],
        "p95_ms_mean": aggregate["p95_ms_mean"],
        "p99_ms_mean": aggregate["p99_ms_mean"],
        "cold_start_step_rate_mean": aggregate["cold_start_step_rate_mean"],
        "prewarm_utilization_mean": aggregate["prewarm_utilization_mean"],
        "prewarm_cost_mean": aggregate["prewarm_cost_mean"],
        "avg_gap_pct_mean": gap["avg_gap_pct_mean"],
        "p95_gap_pct_mean": gap["p95_gap_pct_mean"],
        "p99_gap_pct_mean": gap["p99_gap_pct_mean"],
        "cold_gap_pct_mean": gap["cold_gap_pct_mean"],
        "platform_hash": phash,
        "mechanism_change": str(args.mechanism_change),
        "notes": str(args.notes),
    }
    upsert_index(store_dir / INDEX_CSV, index_row)
    print(f"[tuning-memory] recorded {strategy} round={round_id}")
    print(f"[tuning-memory] strategy log: {strategy_file}")
    print(f"[tuning-memory] index: {store_dir / INDEX_CSV}")


def run_suggest(args: argparse.Namespace) -> None:
    strategy = str(args.strategy).strip()
    store_dir = Path(args.store_dir).resolve()
    path = store_dir / f"{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no tuning memory for strategy '{strategy}': {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty tuning memory: {path}")

    objective = str(args.objective)
    target_gap = float(args.target_gap)

    def score(item: dict[str, Any]) -> float:
        agg = item.get("aggregate_metrics", {}) or {}
        gap = item.get("gap_to_hpwp", {}) or {}
        if objective == "min_avg_e2e":
            return float(agg.get("avg_e2e_ms_mean", math.inf))
        if objective == "min_cold_rate":
            return float(agg.get("cold_start_step_rate_mean", math.inf))
        if objective == "max_prewarm_util":
            return -float(agg.get("prewarm_utilization_mean", -math.inf))
        # near_gap_10
        gap_val = float(gap.get("avg_gap_pct_mean", math.inf))
        return abs(gap_val - target_gap)

    ranked = sorted(rows, key=score)[: max(1, int(args.top))]
    print(f"[tuning-memory] strategy={strategy} objective={objective} top={len(ranked)}")
    for idx, item in enumerate(ranked, start=1):
        agg = item.get("aggregate_metrics", {}) or {}
        gap = item.get("gap_to_hpwp", {}) or {}
        print(
            f"{idx:02d}. round={item.get('round_id')} "
            f"avg={float(agg.get('avg_e2e_ms_mean', math.nan)):.2f} "
            f"cold={float(agg.get('cold_start_step_rate_mean', math.nan)):.4f} "
            f"util={float(agg.get('prewarm_utilization_mean', math.nan)):.4f} "
            f"cost={float(agg.get('prewarm_cost_mean', math.nan)):.2f} "
            f"gap={float(gap.get('avg_gap_pct_mean', math.nan)):.2f}% "
            f"change={item.get('mechanism_change', '')}"
        )


def run_guard(args: argparse.Namespace) -> None:
    strategy = str(args.strategy).strip()
    store_dir = Path(args.store_dir).resolve()
    config_path = Path(args.config).resolve()
    path = store_dir / f"{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no tuning memory for strategy '{strategy}': {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty tuning memory: {path}")
    last = json.loads(lines[-1])
    last_hash = str(last.get("platform_hash", ""))
    cfg = read_yaml(config_path)
    cur_hash = platform_hash(cfg)
    print(f"[tuning-memory] strategy={strategy}")
    print(f"[tuning-memory] last_round={last.get('round_id')} last_platform_hash={last_hash}")
    print(f"[tuning-memory] current_platform_hash={cur_hash}")
    if cur_hash == last_hash:
        print("[tuning-memory] guard=PASS (platform unchanged)")
        return
    print("[tuning-memory] guard=FAIL (platform changed)")
    raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.cmd == "record":
        run_record(args)
        return
    if args.cmd == "suggest":
        run_suggest(args)
        return
    if args.cmd == "guard":
        run_guard(args)
        return
    raise ValueError(f"unsupported cmd: {args.cmd}")


if __name__ == "__main__":
    main()
