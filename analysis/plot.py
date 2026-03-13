from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation run metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_timeseries = subparsers.add_parser("timeseries", help="Plot node metrics timeseries.")
    p_timeseries.add_argument("--run-dir", help="Run directory path or folder name under runs/")
    p_timeseries.add_argument("--out", default="figures.png", help="Output image name")

    p_breakdown = subparsers.add_parser(
        "latency-breakdown",
        help="Plot E2E latency component percentage bars (single or multiple runs).",
    )
    p_breakdown.add_argument(
        "--run-dirs",
        nargs="*",
        help="Run directory paths or names under runs/. If omitted, use latest run.",
    )
    p_breakdown.add_argument(
        "--out",
        default="latency_breakdown.png",
        help="Output image name",
    )

    return parser.parse_args()


def load_node_metrics(path: Path) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"t": [], "replicas": [], "queue": [], "util": []}
    )
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row["node"]
            series[node]["t"].append(float(row["timestamp_sec"]))
            series[node]["replicas"].append(float(row["replicas"]))
            series[node]["queue"].append(float(row["queue_len"]))
            series[node]["util"].append(float(row["utilization"]))
    return series


def resolve_run_dir(token: str | None, *, runs_root: Path = Path("runs")) -> Path:
    if token is None:
        return latest_run_dir(runs_root)
    explicit = Path(token)
    if explicit.exists():
        return explicit
    relative = runs_root / token
    if relative.exists():
        return relative
    raise FileNotFoundError(f"run directory not found: {token}")


def latest_run_dir(runs_root: Path = Path("runs")) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError("runs directory does not exist")
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError("no run directories found under runs/")
    return sorted(dirs, key=lambda p: p.name)[-1]


def load_run_label(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return run_dir.name
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    scheduler = summary.get("scheduler", "unknown_scheduler")
    autoscaler = summary.get("autoscaler", "unknown_autoscaler")
    return f"{scheduler}/{autoscaler}"


def plot_timeseries(run_dir: Path, out_name: str) -> Path:
    metrics_path = run_dir / "node_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing node_metrics.csv in {run_dir}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    series = load_node_metrics(metrics_path)
    if not series:
        raise RuntimeError("node_metrics.csv has no data")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for node, values in series.items():
        axes[0].plot(values["t"], values["replicas"], label=node)
        axes[1].plot(values["t"], values["queue"], label=node)
        axes[2].plot(values["t"], values["util"], label=node)

    axes[0].set_ylabel("Replicas")
    axes[1].set_ylabel("Queue Length")
    axes[2].set_ylabel("Utilization")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()

    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def load_latency_breakdown(run_dir: Path) -> dict[str, Any]:
    request_path = run_dir / "request_paths.csv"
    if not request_path.exists():
        raise FileNotFoundError(f"missing request_paths.csv in {run_dir}")

    with request_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "total_latency_ms",
            "cold_start_latency_ms",
            "data_transfer_latency_ms",
            "execution_latency_ms",
        }
        if not required.issubset(fieldnames):
            raise RuntimeError(
                f"{run_dir} does not contain latency breakdown columns. "
                "Rerun simulation with updated simulator."
            )

        total = 0.0
        cold = 0.0
        transfer = 0.0
        execution = 0.0
        queue_wait = 0.0
        count = 0
        completed_count = 0

        for row in reader:
            raw_total = row.get("total_latency_ms", "")
            if not raw_total:
                continue
            if row.get("completed") == "1":
                completed_count += 1
            total += float(raw_total)
            cold += float(row.get("cold_start_latency_ms", "0") or 0)
            transfer += float(row.get("data_transfer_latency_ms", "0") or 0)
            execution += float(row.get("execution_latency_ms", "0") or 0)
            queue_wait += float(row.get("queue_wait_latency_ms", "0") or 0)
            count += 1

    if count == 0 or total <= 0:
        raise RuntimeError(f"{run_dir} has no completed requests with valid latency data")

    avg_total = total / count
    avg_cold = cold / count
    avg_transfer = transfer / count
    avg_execution = execution / count
    avg_queue = queue_wait / count
    residual = max(0.0, avg_total - (avg_cold + avg_transfer + avg_execution))
    label = load_run_label(run_dir)

    return {
        "run_dir": run_dir,
        "label": label,
        "avg_total": avg_total,
        "avg_cold": avg_cold,
        "avg_transfer": avg_transfer,
        "avg_execution": avg_execution,
        "avg_queue": avg_queue,
        "avg_residual": residual,
        "count": count,
        "completed_count": completed_count,
    }


def plot_latency_breakdown(run_dirs: list[Path], out_name: str) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    stats: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_dir in run_dirs:
        try:
            stats.append(load_latency_breakdown(run_dir))
        except Exception as exc:  # noqa: BLE001 - keep plotting resilient across mixed run formats.
            warnings.append(f"skip {run_dir}: {exc}")

    if not stats:
        raise RuntimeError("no compatible run directories contain latency breakdown data")
    categories = ["total", "cold_start", "data_transfer", "execution"]
    x = [0.0, 1.0, 2.0, 3.0]
    width = 0.75 / max(1, len(stats))

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, item in enumerate(stats):
        offset = (idx - (len(stats) - 1) / 2.0) * width
        positions = [v + offset for v in x]
        total = item["avg_total"]
        values_pct = [
            100.0,
            (item["avg_cold"] / total) * 100.0,
            (item["avg_transfer"] / total) * 100.0,
            (item["avg_execution"] / total) * 100.0,
        ]
        values_ms = [
            total,
            item["avg_cold"],
            item["avg_transfer"],
            item["avg_execution"],
        ]
        bars = ax.bar(positions, values_pct, width=width, label=item["label"])
        for bar, pct, ms in zip(bars, values_pct, values_ms):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.5,
                f"{pct:.1f}%\n{ms:.1f}ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Latency Share (%)")
    ax.set_title("E2E Latency Component Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    notes = [
        f"{item['label']}: queue_wait={item['avg_queue']:.1f}ms, residual={item['avg_residual']:.1f}ms"
        for item in stats
    ]
    fig.text(0.01, 0.01, " | ".join(notes), fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_paths: list[Path] = []
    for item in stats:
        out_path = item["run_dir"] / out_name
        fig.savefig(out_path, dpi=150)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths, stats, warnings


def main() -> int:
    args = parse_args()
    if args.command == "timeseries":
        run_dir = resolve_run_dir(args.run_dir)
        out_path = plot_timeseries(run_dir, args.out)
        print(f"saved: {out_path}")
        return 0

    if args.command == "latency-breakdown":
        tokens = args.run_dirs if args.run_dirs else [None]
        run_dirs = [resolve_run_dir(token) for token in tokens]
        out_paths, stats, warnings = plot_latency_breakdown(run_dirs, args.out)
        for path in out_paths:
            print(f"saved: {path}")
        for warning in warnings:
            print(f"warning: {warning}")
        for item in stats:
            print(
                (
                    f"{item['label']}: avg_total={item['avg_total']:.2f}ms, "
                    f"cold={item['avg_cold']:.2f}ms, transfer={item['avg_transfer']:.2f}ms, "
                    f"execution={item['avg_execution']:.2f}ms, queue_wait={item['avg_queue']:.2f}ms, "
                    f"requests={item['count']}, completed={item['completed_count']}"
                )
            )
        return 0

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
