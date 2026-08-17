from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import (
        STRATEGY_COLORS,
        STRATEGY_LABELS,
        load_metric_rows,
        parse_float,
        setup_plot_font,
        strategy_sort_key,
        write_table,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    from common import (
        STRATEGY_COLORS,
        STRATEGY_LABELS,
        load_metric_rows,
        parse_float,
        setup_plot_font,
        strategy_sort_key,
        write_table,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot memory_time_cost using average load (low/mid/high).")
    parser.add_argument("--metrics-dir", default="results/compare/metrics")
    parser.add_argument("--out-dir", default="results/compare/derived")
    return parser.parse_args()


def aggregate_avg_cost(rows: list[dict[str, str]]) -> dict[str, float]:
    costs: dict[str, list[float]] = {}
    for row in rows:
        strategy = str(row.get("strategy", "")).strip()
        value = parse_float(row.get("value_mean"))
        if value != value:  # NaN
            continue
        costs.setdefault(strategy, []).append(value)

    avg_cost: dict[str, float] = {}
    for strategy, vals in costs.items():
        if vals:
            avg_cost[strategy] = sum(vals) / float(len(vals))
    return avg_cost


def plot(metrics_dir: Path, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_font(plt)
    rows = load_metric_rows(metrics_dir / "memory_time_cost.csv")
    avg_cost = aggregate_avg_cost(rows)
    if not avg_cost:
        raise RuntimeError("no memory_time_cost rows available")

    # Keep only strategies present in compare palette/order, and sort by avg raw cost.
    items = sorted(avg_cost.items(), key=lambda kv: (kv[1],) + strategy_sort_key(kv[0]))
    strategies = [item[0] for item in items]
    costs_raw = [item[1] for item in items]
    costs_scaled = [v / 1e8 for v in costs_raw]
    y_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(11.5, max(6.8, 0.6 * len(strategies) + 3.0)))

    # Lollipop style: baseline line + endpoint marker.
    for idx, (strategy, xval) in enumerate(zip(strategies, costs_scaled)):
        color = STRATEGY_COLORS.get(strategy, "#5B6E78")
        ax.hlines(y=idx, xmin=0.0, xmax=xval, color=color, linewidth=2.1, alpha=0.95)
        ax.plot(xval, idx, "o", color=color, markersize=8)
        ax.text(xval + max(0.02, xval * 0.02), idx, f"{xval:.2f}", va="center", ha="left", fontsize=16)

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(y_labels, fontsize=18, color="#111111")
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlabel("Memory-Time Cost (×1e8 MB*sec)", fontsize=18, color="#111111")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(left=0.0)
    fig.tight_layout()

    fig_path = out_dir / "figures" / "memory_time_cost_avg_load.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[list[object]] = []
    for strategy, raw, scaled in zip(strategies, costs_raw, costs_scaled):
        table_rows.append([strategy, raw, scaled])
    write_table(
        out_dir / "tables" / "memory_time_cost_avg_load.csv",
        ["strategy", "avg_memory_time_cost_mb_sec", "avg_memory_time_cost_x1e8"],
        table_rows,
    )
    return fig_path


def main() -> int:
    args = parse_args()
    fig_path = plot(Path(args.metrics_dir), Path(args.out_dir))
    print(f"saved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
