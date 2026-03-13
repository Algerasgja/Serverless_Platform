from __future__ import annotations

import argparse
import sys

from simulator.config import load_config
from simulator.dag.dataset import AlibabaDatasetAdapter
from simulator.simulation import SimulationRunner


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional DAG serverless simulator")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = load_config(args.config)

    adapter = AlibabaDatasetAdapter(config.dataset, config.workload)
    corpus = adapter.load_corpus()
    runner = SimulationRunner(config, corpus)
    run_dir = runner.run()
    print(f"simulation completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
