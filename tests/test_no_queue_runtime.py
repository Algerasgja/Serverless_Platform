import csv
import json
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
            type="kpa_v1",
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


def _two_step_corpus(latency_ms: int) -> DagCorpus:
    template = DagTemplate(
        um="u1",
        transitions={"__start__": {"A": 1.0}, "A": {"B": 1.0}},
        node_latency_ms={"A": latency_ms, "B": latency_ms},
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

    def test_cpu_memory_capacity_ignores_legacy_slot_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=50, cold_start_ms=20)
            cfg.capacity.max_total_instances = 2
            cfg.physical_nodes.max_containers_per_node = 1
            cfg.physical_nodes.cpu_total_mcpu_per_node = 4000
            cfg.physical_nodes.mem_total_mb_per_node = 2048
            cfg.runtime.function_cpu_request_mcpu_min = 1000
            cfg.runtime.function_cpu_request_mcpu_max = 1000
            cfg.runtime.function_memory_mb_min = 256
            cfg.runtime.function_memory_mb_max = 256
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
            self.assertTrue(all(row["failed_reason"] != "capacity_exhausted" for row in req_rows))

            with (run_dir / "scheduler_decisions.csv").open("r", encoding="utf-8", newline="") as f:
                sch_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(sum(1 for row in sch_rows if row["decision_type"] == "cold_start"), 2)

    def test_kpa_scale_down_reclaims_idle_before_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=10, cold_start_ms=5)
            cfg.capacity.max_total_instances = 4
            cfg.physical_nodes.max_containers_per_node = 1
            cfg.physical_nodes.cpu_total_mcpu_per_node = 4000
            cfg.physical_nodes.mem_total_mb_per_node = 4096
            cfg.physical_nodes.idle_ttl_sec = 300
            cfg.autoscaler.type = "kpa_v1"
            cfg.autoscaler.sync_period_sec = 1
            cfg.autoscaler.target_utilization = 1.0
            cfg.autoscaler.kpa_target_concurrency = 1.0
            cfg.autoscaler.kpa_stable_window_sec = 1
            cfg.autoscaler.kpa_panic_window_sec = 1
            cfg.autoscaler.kpa_panic_threshold = 10.0
            cfg.autoscaler.min_replicas = 0
            cfg.autoscaler.max_replicas_per_node = 10

            corpus = _single_fn_corpus(latency_ms=10)
            with patch("simulator.simulation.generate_arrivals", return_value=[(0, "u1")]):
                run_dir = Path(SimulationRunner(cfg, corpus).run())

            with (run_dir / "node_metrics.csv").open("r", encoding="utf-8", newline="") as f:
                node_rows = list(csv.DictReader(f))
            sec1_rows = [row for row in node_rows if row["timestamp_sec"] == "1"]
            self.assertTrue(sec1_rows)
            self.assertTrue(all(row["active_containers"] == "0" for row in sec1_rows))

            with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreater(summary.get("kpa", {}).get("scale_down_requested", 0), 0)

    def test_xanadu_prewarm_requires_cold_start_then_can_be_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=20, cold_start_ms=1500)
            cfg.capacity.max_total_instances = 10
            cfg.physical_nodes.count = 2
            cfg.physical_nodes.max_containers_per_node = 5
            cfg.physical_nodes.cpu_total_mcpu_per_node = 4000
            cfg.physical_nodes.mem_total_mb_per_node = 4096
            cfg.runtime.function_cpu_request_mcpu_min = 1000
            cfg.runtime.function_cpu_request_mcpu_max = 1000
            cfg.runtime.function_memory_mb_min = 256
            cfg.runtime.function_memory_mb_max = 256
            cfg.runtime.function_compute_data_mb_min = 1.0
            cfg.runtime.function_compute_data_mb_max = 1.0
            cfg.runtime.function_cold_start_ms_min = 1500
            cfg.runtime.function_cold_start_ms_max = 1500
            cfg.runtime.cold_start_ratio_floor = 0.0
            cfg.autoscaler.type = "xanadu_v1"
            cfg.autoscaler.sync_period_sec = 1
            cfg.autoscaler.xanadu_depth = 1
            cfg.autoscaler.xanadu_ewma_alpha = 0.2

            corpus = _two_step_corpus(latency_ms=20)
            with patch(
                "simulator.simulation.generate_arrivals",
                return_value=[(0, "u1"), (3000, "u1")],
            ):
                run_dir = Path(SimulationRunner(cfg, corpus).run())

            with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            xanadu = summary.get("xanadu", {})
            self.assertEqual("xanadu_v1", xanadu.get("type"))
            self.assertGreater(xanadu.get("prewarm_created", 0), 0)
            self.assertGreater(xanadu.get("prewarm_ready", 0), 0)
            self.assertGreater(xanadu.get("prewarm_consumed", 0), 0)

    def test_hpwp_reconcile_prewarm_behaves_like_xanadu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _base_config(base, node_latency_ms=20, cold_start_ms=800)
            cfg.capacity.max_total_instances = 20
            cfg.physical_nodes.count = 2
            cfg.physical_nodes.max_containers_per_node = 10
            cfg.autoscaler.type = "hpwp_v1"
            cfg.autoscaler.sync_period_sec = 1
            cfg.runtime.function_cold_start_ms_min = 800
            cfg.runtime.function_cold_start_ms_max = 800
            cfg.runtime.function_compute_data_mb_min = 2.0
            cfg.runtime.function_compute_data_mb_max = 2.0
            cfg.runtime.cold_start_ratio_floor = 0.0

            corpus = _two_step_corpus(latency_ms=20)
            with patch(
                "simulator.simulation.generate_arrivals",
                return_value=[(0, "u1"), (200, "u1"), (400, "u1")],
            ):
                run_dir = Path(SimulationRunner(cfg, corpus).run())

            with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            hpwp = summary.get("hpwp", {})
            self.assertEqual("hpwp_v1", hpwp.get("type"))
            self.assertEqual("active_reconcile", hpwp.get("mode"))
            self.assertGreater(hpwp.get("prewarm_create_attempted", 0), 0)
            self.assertGreaterEqual(
                hpwp.get("prewarm_created", 0) + hpwp.get("capacity_blocked_creations", 0),
                hpwp.get("prewarm_create_attempted", 0),
            )
            self.assertIn("final_desired_by_function", hpwp)


if __name__ == "__main__":
    unittest.main()
