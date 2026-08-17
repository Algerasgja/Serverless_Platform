from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Plot memory_time_cost diverging bars vs Oracle.")
    parser.add_argument("--metrics-dir", default="results/compare/metrics")
    parser.add_argument("--out-dir", default="results/compare/new_plots")
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
    if "oracle" not in avg_cost:
        raise RuntimeError("oracle memory_time_cost missing; cannot build diverging chart")

    oracle_avg = avg_cost["oracle"]
    if oracle_avg <= 0:
        raise RuntimeError("oracle memory_time_cost must be positive")

    deltas: list[tuple[str, float, float]] = []
    for strategy, value in avg_cost.items():
        if strategy == "oracle":
            continue
        delta_pct = ((value - oracle_avg) / oracle_avg) * 100.0
        deltas.append((strategy, value, delta_pct))
    deltas.sort(key=lambda x: x[2])

    strategies = [d[0] for d in deltas]
    delta_vals = [d[2] for d in deltas]
    y_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(11.5, max(6.8, 0.6 * len(strategies) + 3.0)))
    colors = [STRATEGY_COLORS.get(s, "#5B6E78") for s in strategies]
    bars = ax.barh(range(len(strategies)), delta_vals, color=colors, edgecolor="#2F3E46", linewidth=0.35)
    ax.axvline(0.0, color="#444444", linewidth=1.2)

    for idx, (bar, value) in enumerate(zip(bars, delta_vals)):
        x = value + (0.7 if value >= 0 else -0.7)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", ha=ha, fontsize=14)

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(y_labels, fontsize=16)
    for tick in ax.get_yticklabels():
        tick.set_color("#111111")

    ax.set_xlabel("", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    fig_path = out_dir / "figures" / "memory_time_cost_diverging_vs_oracle_avg.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for strategy, avg_value, delta_pct in deltas:
        table_rows.append([strategy, avg_value, oracle_avg, delta_pct])
    write_table(
        out_dir / "tables" / "memory_time_cost_vs_oracle_avg.csv",
        ["strategy", "avg_memory_time_cost", "oracle_avg_memory_time_cost", "delta_pct_vs_oracle"],
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
