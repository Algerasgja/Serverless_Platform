import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from simulator.simulation import SimulationRunner
from simulator.types import DagCorpus, DagTemplate


def _base_config(base: Path, *, node_latency_ms: int, cold_start_ms: int) -> SimulationConfig:
    return SimulationConfig(
        experiment=ExperimentConfig(name="test_no_queue", duration_seconds=2, random_seed=7),
        dataset=DatasetConfig(
            source="unit_test",
            callgraph_file=str(base / "missing_callgraph.csv"),
            qps_file=str(base / "missing_qps.csv"),
            processed_dir=str(base / "processed"),
            time_window_minutes=1,
            sample_um_count=1,
            checksum_required=False,
            dag_tasks_file=str(base / "missing_filtered_tasks.csv"),
            dag_top_k=20,
        ),
        dag_policy=DagPolicyConfig(
            granularity="um",
            path_rule="critical_path",
            markov_order=1,
            session_gap_sec=60,
            session_continue_prob=1.0,
            context_alpha=0.0,
        ),
        workload=WorkloadConfig(
            mode="generative",
            baseline_rps=1.0,
            burst_rps=1.0,
            burst_start_sec=0,
            burst_duration_sec=0,
            rate_multiplier=1.0,
        ),
        runtime=RuntimeConfig(
            node_base_latency_ms=node_latency_ms,
            node_latency_jitter_ms=0,
            cold_start_ms_min=cold_start_ms,
            cold_start_ms_max=cold_start_ms,
            request_timeout_ms=10_000,
            max_concurrency_per_instance=1,
            frame_tick_ms=1,
            function_profile_mode="cpu_intensive_random",
            function_profile_seed_offset=202,
            compute_mb_per_sec_per_1000mcpu=500.0,
            function_cpu_request_mcpu_min=1000,
            function_cpu_request_mcpu_max=1000,
            function_memory_mb_min=256,
            function_memory_mb_max=256,
            function_output_data_mb_min=0.01,
            function_output_data_mb_max=0.01,
            function_compute_data_mb_min=2.0,
            function_compute_data_mb_max=2.0,
            function_cold_start_ms_min=cold_start_ms,
            function_cold_start_ms_max=cold_start_ms,
            cold_start_ratio_floor=0.0,
        ),
        physical_nodes=PhysicalNodesConfig(
            count=1,
            max_containers_per_node=1,
            idle_ttl_sec=60,
            same_node_transfer_ms_min=0,
            same_node_transfer_ms_max=0,
            cross_node_transfer_ms_min=0,
            cross_node_transfer_ms_max=0,
            cpu_total_mcpu_per_node=1000,
            mem_total_mb_per_node=1024,
            bandwidth_mode="random_uniform",
            bandwidth_mbps_min=1000,
            bandwidth_mbps_max=1000,
        ),
        scheduler=SchedulerConfig(type="least_load"),
        autoscaler=AutoscalerConfig(
            type="hpa_v1",
            target_utilization=0.7,
            sync_period_sec=15,
            scale_down_stabilization_sec=60,
            min_replicas=0,
            max_replicas_per_node=10,
        ),
        capacity=CapacityConfig(max_total_instances=1),
        output=OutputConfig(runs_dir=str(base / "runs")),
    )


def _single_fn_corpus(latency_ms: int) -> DagCorpus:
    template = DagTemplate(
        um="u1",
        transitions={"__start__": {"fn": 1.0}},
        node_latency_ms={"fn": latency_ms},
    )
    return DagCorpus(
        templates={"u1": template},
        um_weights={"u1": 1.0},
        replay_total_qps_per_minute=[60.0],
        metadata={"source": "unit_test"},
    )


class NoQueueRuntimeTests(unittest.TestCase):
    def test_warm_reuse_is_preferred_when_idle_container_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=10, cold_start_ms=5)
            corpus = _single_fn_corpus(latency_ms=10)

            with patch("simulator.simulation.generate_arrivals", return_value=[(0, "u1"), (100, "u1")]):
                run_dir = Path(SimulationRunner(cfg, corpus).run())

            with (run_dir / "scheduler_decisions.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            decision_types = [row["decision_type"] for row in rows]
            self.assertIn("cold_start", decision_types)
            self.assertIn("warm_reuse", decision_types)
            self.assertNotIn("capacity_exhausted", decision_types)

    def test_capacity_exhausted_fails_immediately_without_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=1000, cold_start_ms=0)
            corpus = _single_fn_corpus(latency_ms=1000)

            with patch("simulator.simulation.generate_arrivals", return_value=[(0, "u1"), (1, "u1")]):
                run_dir = Path(SimulationRunner(cfg, corpus).run())

            with (run_dir / "request_paths.csv").open("r", encoding="utf-8", newline="") as f:
                req_rows = list(csv.DictReader(f))
            self.assertEqual(2, len(req_rows))
            self.assertTrue(any(row["failed_reason"] == "capacity_exhausted" for row in req_rows))
            self.assertTrue(all(row["queue_wait_latency_ms"] == "0" for row in req_rows))

            with (run_dir / "scheduler_decisions.csv").open("r", encoding="utf-8", newline="") as f:
                sch_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["decision_type"] == "capacity_exhausted" for row in sch_rows))

    def test_cpu_memory_hard_constraints_trigger_capacity_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=50, cold_start_ms=20)
            cfg.capacity.max_total_instances = 2
            cfg.physical_nodes.max_containers_per_node = 2
            cfg.physical_nodes.cpu_total_mcpu_per_node = 1500
            cfg.physical_nodes.mem_total_mb_per_node = 512
            cfg.runtime.function_cpu_request_mcpu_min = 1000
            cfg.runtime.function_cpu_request_mcpu_max = 1000
            cfg.runtime.function_memory_mb_min = 400
            cfg.runtime.function_memory_mb_max = 400
            cfg.runtime.function_compute_data_mb_min = 10.0
            cfg.runtime.function_compute_data_mb_max = 10.0
            cfg.runtime.function_cold_start_ms_min = 100
            cfg.runtime.function_cold_start_ms_max = 100
            cfg.runtime.cold_start_ratio_floor = 0.0

            corpus = _single_fn_corpus(latency_ms=50)
            with patch("simulator.simulation.generate_arrivals", return_value=[(0, "u1"), (1, "u1")]):
                run_dir = Path(SimulationRunner(cfg, corpus).run())

            with (run_dir / "request_paths.csv").open("r", encoding="utf-8", newline="") as f:
                req_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["failed_reason"] == "capacity_exhausted" for row in req_rows))


if __name__ == "__main__":
    unittest.main()
