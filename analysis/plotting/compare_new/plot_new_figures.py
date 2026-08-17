from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from .plot_cold_start_heatmap import plot as plot_cold_start_heatmap  # noqa: E402
    from .plot_ctre95_matrix import plot as plot_ctre95_matrix  # noqa: E402
    from .plot_memory_time_avg_load import plot as plot_memory_time_avg_load  # noqa: E402
    from .plot_memory_time_diverging import plot as plot_memory_time_diverging  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution fallback.
    from plot_cold_start_heatmap import plot as plot_cold_start_heatmap  # noqa: E402
    from plot_ctre95_matrix import plot as plot_ctre95_matrix  # noqa: E402
    from plot_memory_time_avg_load import plot as plot_memory_time_avg_load  # noqa: E402
    from plot_memory_time_diverging import plot as plot_memory_time_diverging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot new compare figures from existing metrics.")
    parser.add_argument("--metrics-dir", default="results/compare/metrics")
    parser.add_argument("--out-dir", default="results/compare/derived")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    fig1 = plot_cold_start_heatmap(metrics_dir, out_dir)
    fig2 = plot_memory_time_avg_load(metrics_dir, out_dir)
    fig3 = plot_ctre95_matrix(metrics_dir, out_dir)
    fig4 = plot_memory_time_diverging(metrics_dir, out_dir)

    print(f"saved figure: {fig1}")
    print(f"saved figure: {fig2}")
    print(f"saved figure: {fig3}")
    print(f"saved figure: {fig4}")
    print(f"saved tables in: {out_dir / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
