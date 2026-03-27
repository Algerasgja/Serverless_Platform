import json
import csv
import tempfile
import unittest
from pathlib import Path

from simulator.config import (
    AutoscalerConfig,
    CapacityConfig,
    DagPolicyConfig,
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PhysicalNodesConfig,
    RuntimeConfig,
    SchedulerConfig,
    SimulationConfig,
    WorkloadConfig,
)
from simulator.dag.dataset import AlibabaDatasetAdapter
from simulator.simulation import SimulationRunner


class SimulationOutputsTests(unittest.TestCase):
    def test_run_persists_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = SimulationConfig(
                experiment=ExperimentConfig(name="test", duration_seconds=20, random_seed=123),
                dataset=DatasetConfig(
                    source="alibaba_cluster_trace_microservices_v2021",
                    callgraph_file=str(base / "missing_callgraph.csv"),
                    qps_file=str(base / "missing_qps.csv"),
                    processed_dir=str(base / "processed"),
                    time_window_minutes=60,
                    sample_um_count=3,
                    checksum_required=False,
                    dag_tasks_file=str(base / "missing_filtered_tasks.csv"),
                    dag_top_k=20,
                ),
                dag_policy=DagPolicyConfig(
                    granularity="um",
                    path_rule="critical_path",
                    markov_order=1,
                    session_gap_sec=30,
                    session_continue_prob=0.7,
                    context_alpha=0.35,
                ),
                workload=WorkloadConfig(
                    mode="generative",
                    baseline_rps=2.0,
                    burst_rps=6.0,
                    burst_start_sec=5,
                    burst_duration_sec=5,
                    rate_multiplier=1.0,
                ),
                runtime=RuntimeConfig(
                    node_base_latency_ms=20,
                    node_latency_jitter_ms=5,
                    cold_start_ms_min=5,
                    cold_start_ms_max=10,
                    max_concurrency_per_instance=1,
                    request_timeout_ms=2000,
                    frame_tick_ms=1,
                    function_profile_mode="cpu_intensive_random",
                    compute_mb_per_sec_per_1000mcpu=500.0,
                    function_cpu_request_mcpu_min=1000,
                    function_cpu_request_mcpu_max=1000,
                    function_memory_mb_min=128,
                    function_memory_mb_max=256,
                    function_output_data_mb_min=0.01,
                    function_output_data_mb_max=0.05,
                    function_compute_data_mb_min=1.0,
                    function_compute_data_mb_max=3.0,
                    function_cold_start_ms_min=5,
                    function_cold_start_ms_max=10,
                    cold_start_ratio_floor=0.0,
                ),
                physical_nodes=PhysicalNodesConfig(
                    count=4,
                    max_containers_per_node=4,
                    idle_ttl_sec=10,
                    same_node_transfer_ms_min=1,
                    same_node_transfer_ms_max=2,
                    cross_node_transfer_ms_min=3,
                    cross_node_transfer_ms_max=5,
                    cpu_total_mcpu_per_node=2000,
                    mem_total_mb_per_node=4096,
                    bandwidth_mode="random_uniform",
                    bandwidth_mbps_min=1000,
                    bandwidth_mbps_max=1000,
                ),
                scheduler=SchedulerConfig(type="least_load"),
                autoscaler=AutoscalerConfig(
                    type="kpa_v1",
                    target_utilization=0.7,
                    sync_period_sec=5,
                    scale_down_stabilization_sec=30,
                    min_replicas=0,
                    max_replicas_per_node=8,
                ),
                capacity=CapacityConfig(max_total_instances=20),
                output=OutputConfig(runs_dir=str(base / "runs")),
            )

            corpus = AlibabaDatasetAdapter(cfg.dataset, cfg.workload).load_corpus()
            run_dir = Path(SimulationRunner(cfg, corpus).run())

            expected_files = {
                "config.snapshot.yaml",
                "scheduler_decisions.csv",
                "autoscaler_decisions.csv",
                "node_metrics.csv",
                "request_paths.csv",
                "summary.json",
            }
            self.assertTrue(run_dir.exists())
            self.assertTrue(expected_files.issubset({p.name for p in run_dir.iterdir()}))

            with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual("least_load", summary["scheduler"])
            self.assertEqual("kpa_v1", summary["autoscaler"])
            self.assertIn("dataset_metadata", summary)

            with (run_dir / "request_paths.csv").open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames or [])
                rows = list(reader)
            self.assertTrue(
                {
                    "total_latency_ms",
                    "cold_start_latency_ms",
                    "data_transfer_latency_ms",
                    "execution_latency_ms",
                    "failed_reason",
                }.issubset(cols)
            )
            for row in rows:
                self.assertEqual("0", row.get("queue_wait_latency_ms", "0"))


if __name__ == "__main__":
    unittest.main()
