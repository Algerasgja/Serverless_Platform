import csv
import tempfile
import unittest
from pathlib import Path

from simulator.config import DatasetConfig, WorkloadConfig
from simulator.dag.dataset import AlibabaDatasetAdapter


def _workload(mode: str = "replay") -> WorkloadConfig:
    return WorkloadConfig(
        mode=mode,
        baseline_rps=1.0,
        burst_rps=1.0,
        burst_start_sec=0,
        burst_duration_sec=0,
        rate_multiplier=1.0,
        invokes_cdf_file="unused.csv",
        cvs_cdf_file="unused.csv",
        realworld_seed_offset=303,
        min_iat_ms=1.0,
    )


class DagDatasetTopKTests(unittest.TestCase):
    def test_unique_structure_topk_and_dirty_task_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            tasks_path = base / "filtered_tasks.csv"
            with tasks_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["task_name", "job_name"])
                # structure A (support=2)
                w.writerow(["M1", "job_a1"])
                w.writerow(["R2_1", "job_a1"])
                w.writerow(["M3_1", "job_a1"])
                w.writerow(["J4_2_3", "job_a1"])
                w.writerow(["M10", "job_a2"])
                w.writerow(["R11_10", "job_a2"])
                w.writerow(["J12_10", "job_a2"])
                w.writerow(["M13_11_12", "job_a2"])
                # structure B (node count larger, should rank first)
                w.writerow(["M1", "job_b1"])
                w.writerow(["R2_1_Stg9", "job_b1"])
                w.writerow(["M3_1", "job_b1"])
                w.writerow(["J4_2_3_", "job_b1"])
                w.writerow(["M5_4", "job_b1"])
                # structure C (same nodes as A but fewer edges)
                w.writerow(["M1", "job_c1"])
                w.writerow(["R2_1", "job_c1"])
                w.writerow(["M3_2", "job_c1"])
                w.writerow(["J4_3", "job_c1"])
                # invalid row
                w.writerow(["INVALID_TASK", "job_bad"])

            cfg = DatasetConfig(
                source="unit_test",
                callgraph_file=str(base / "unused_callgraph.csv"),
                qps_file=str(base / "unused_qps.csv"),
                processed_dir=str(base / "processed"),
                time_window_minutes=10,
                sample_um_count=8,
                checksum_required=False,
                dag_tasks_file=str(tasks_path),
                dag_top_k=2,
                dag_selection_mode="topk_unique",
            )
            corpus = AlibabaDatasetAdapter(cfg, _workload()).load_corpus()

            self.assertEqual(2, len(corpus.templates))
            self.assertEqual(["dag_0001", "dag_0002"], sorted(corpus.templates.keys()))
            self.assertAlmostEqual(1.0, sum(corpus.um_weights.values()), places=6)

            dag_selection = corpus.metadata["dag_selection"]
            self.assertEqual("unique_structure_topk", dag_selection["policy"])
            self.assertEqual(2, dag_selection["top_k"])
            self.assertEqual(2, len(dag_selection["selected"]))
            self.assertGreaterEqual(
                dag_selection["selected"][0]["node_count"],
                dag_selection["selected"][1]["node_count"],
            )
            self.assertTrue(any(item["support_count"] == 2 for item in dag_selection["selected"]))
            self.assertGreaterEqual(corpus.metadata["invalid_rows"], 1)

    def test_replay_requires_dag_tasks_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = DatasetConfig(
                source="unit_test",
                callgraph_file=str(base / "unused_callgraph.csv"),
                qps_file=str(base / "unused_qps.csv"),
                processed_dir=str(base / "processed"),
                time_window_minutes=10,
                sample_um_count=8,
                checksum_required=False,
                dag_tasks_file=str(base / "missing_filtered_tasks.csv"),
                dag_top_k=20,
                dag_selection_mode="random_unique",
            )
            with self.assertRaises(FileNotFoundError):
                AlibabaDatasetAdapter(cfg, _workload(mode="replay")).load_corpus()


if __name__ == "__main__":
    unittest.main()
