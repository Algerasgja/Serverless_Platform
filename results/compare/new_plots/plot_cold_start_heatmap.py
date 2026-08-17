from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    STRATEGY_LABELS,
    load_metric_rows,
    parse_float,
    setup_plot_font,
    strategy_sort_key,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cold_start_step_rate heatmap.")
    parser.add_argument("--metrics-dir", default="results/compare/metrics")
    parser.add_argument("--out-dir", default="results/compare/new_plots")
    return parser.parse_args()


def build_matrix(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[list[float]]]:
    value_map: dict[tuple[str, str], float] = {}
    strategies: set[str] = set()
    for row in rows:
        scenario = str(row.get("scenario", "")).strip().lower()
        strategy = str(row.get("strategy", "")).strip()
        value = parse_float(row.get("value_mean"))
        if scenario not in SCENARIO_ORDER:
            continue
        value_map[(strategy, scenario)] = value * 100.0
        strategies.add(strategy)

    ordered_strategies = sorted(strategies, key=strategy_sort_key)
    matrix: list[list[float]] = []
    for strategy in ordered_strategies:
        matrix.append([value_map.get((strategy, s), float("nan")) for s in SCENARIO_ORDER])
    return ordered_strategies, SCENARIO_ORDER, matrix


def plot(metrics_dir: Path, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    setup_plot_font(plt)
    rows = load_metric_rows(metrics_dir / "cold_start_step_rate.csv")
    strategies, scenarios, matrix = build_matrix(rows)
    if not strategies:
        raise RuntimeError("no cold_start_step_rate rows available")

    fig, ax = plt.subplots(figsize=(10.5, max(6.8, 0.55 * len(strategies) + 3.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.set_ylabel("Cold Start Step Rate (%)", rotation=90, fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}%"))

    x_labels = [SCENARIO_LABELS.get(s, s.upper()) for s in scenarios]
    y_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(y_labels, fontsize=18)
    for tick in ax.get_yticklabels():
        tick.set_color("#111111")

    for i, strategy in enumerate(strategies):
        for j, scenario in enumerate(scenarios):
            value = matrix[i][j]
            if value == value:  # not NaN
                ax.text(j, i, f"{value:.2f}%", ha="center", va="center", fontsize=16, color="#111111")

    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    fig_path = out_dir / "figures" / "cold_start_step_rate_heatmap.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[list[object]] = []
    for i, strategy in enumerate(strategies):
        table_rows.append([strategy] + [matrix[i][j] for j in range(len(scenarios))])
    write_table(out_dir / "tables" / "cold_start_step_rate_pivot.csv", ["strategy"] + scenarios, table_rows)
    return fig_path


def main() -> int:
    args = parse_args()
    fig_path = plot(Path(args.metrics_dir), Path(args.out_dir))
    print(f"saved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
