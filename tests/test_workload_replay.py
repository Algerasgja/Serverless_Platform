import tempfile
import unittest
from pathlib import Path
import random

from simulator.config import WorkloadConfig
from simulator.runtime.workload import generate_arrivals
from simulator.types import DagCorpus, DagTemplate


def _corpus() -> DagCorpus:
    return DagCorpus(
        templates={
            "dag_0001": DagTemplate(
                um="dag_0001",
                transitions={"__start__": {"dag_0001_fn_001": 1.0}},
                node_latency_ms={"dag_0001_fn_001": 10},
            ),
            "dag_0002": DagTemplate(
                um="dag_0002",
                transitions={"__start__": {"dag_0002_fn_001": 1.0}},
                node_latency_ms={"dag_0002_fn_001": 10},
            ),
        },
        um_weights={"dag_0001": 0.5, "dag_0002": 0.5},
        replay_total_qps_per_minute=[],
        metadata={},
    )


def _write_cdfs(base: Path) -> tuple[Path, Path]:
    invokes = base / "invokesCDF.csv"
    cvs = base / "CVs.csv"
    invokes.write_text("1,0.5\n2,1.0\n", encoding="utf-8")
    cvs.write_text("0.1,0.5\n0.5,1.0\n", encoding="utf-8")
    return invokes, cvs


def _replay_cfg(invokes_path: Path, cvs_path: Path, rate_multiplier: float) -> WorkloadConfig:
    return WorkloadConfig(
        mode="replay",
        baseline_rps=1.0,
        burst_rps=1.0,
        burst_start_sec=0,
        burst_duration_sec=0,
        rate_multiplier=rate_multiplier,
        invokes_cdf_file=str(invokes_path),
        cvs_cdf_file=str(cvs_path),
        realworld_seed_offset=303,
        min_iat_ms=1.0,
    )


class ReplayWorkloadTests(unittest.TestCase):
    def test_replay_arrivals_are_reproducible_with_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            invokes, cvs = _write_cdfs(base)
            cfg = _replay_cfg(invokes, cvs, rate_multiplier=0.001)
            corpus = _corpus()

            profile_a: dict[str, object] = {}
            arrivals_a = generate_arrivals(
                workload_cfg=cfg,
                corpus=corpus,
                duration_seconds=60,
                rng=random.Random(1),
                base_seed=42,
                profile_sink=profile_a,
            )
            profile_b: dict[str, object] = {}
            arrivals_b = generate_arrivals(
                workload_cfg=cfg,
                corpus=corpus,
                duration_seconds=60,
                rng=random.Random(999),
                base_seed=42,
                profile_sink=profile_b,
            )

            self.assertEqual(arrivals_a, arrivals_b)
            self.assertEqual(profile_a["generated_requests"], len(arrivals_a))
            self.assertEqual(profile_a["generated_requests"], profile_b["generated_requests"])

    def test_rate_multiplier_monotonicity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            invokes, cvs = _write_cdfs(base)
            corpus = _corpus()

            low = generate_arrivals(
                workload_cfg=_replay_cfg(invokes, cvs, rate_multiplier=0.0001),
                corpus=corpus,
                duration_seconds=600,
                rng=random.Random(1),
                base_seed=42,
            )
            high = generate_arrivals(
                workload_cfg=_replay_cfg(invokes, cvs, rate_multiplier=0.001),
                corpus=corpus,
                duration_seconds=600,
                rng=random.Random(1),
                base_seed=42,
            )
            self.assertGreaterEqual(len(high), len(low))

    def test_replay_raises_when_cdf_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            missing_invokes = base / "missing_invokes.csv"
            missing_cvs = base / "missing_cvs.csv"
            cfg = _replay_cfg(missing_invokes, missing_cvs, rate_multiplier=0.001)
            with self.assertRaises(FileNotFoundError):
                generate_arrivals(
                    workload_cfg=cfg,
                    corpus=_corpus(),
                    duration_seconds=10,
                    rng=random.Random(1),
                    base_seed=42,
                )


if __name__ == "__main__":
    unittest.main()
