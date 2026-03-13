import unittest

from simulator.config import PhysicalNodesConfig, RuntimeConfig
from simulator.runtime.resource_model import build_bandwidth_matrix, build_function_profiles


class ResourceModelTests(unittest.TestCase):
    def test_random_function_profiles_are_reproducible(self) -> None:
        runtime = RuntimeConfig(
            node_base_latency_ms=10,
            node_latency_jitter_ms=0,
            cold_start_ms_min=10,
            cold_start_ms_max=20,
            request_timeout_ms=2000,
            function_profile_mode="cpu_intensive_random",
            function_profile_seed_offset=202,
            compute_mb_per_sec_per_1000mcpu=200.0,
            function_cpu_request_mcpu_min=800,
            function_cpu_request_mcpu_max=1200,
            function_memory_mb_min=256,
            function_memory_mb_max=512,
            function_output_data_mb_min=0.1,
            function_output_data_mb_max=0.2,
            function_compute_data_mb_min=10.0,
            function_compute_data_mb_max=20.0,
            function_cold_start_ms_min=100,
            function_cold_start_ms_max=200,
            cold_start_ratio_floor=0.0,
        )
        physical = PhysicalNodesConfig(
            count=2,
            max_containers_per_node=2,
            cpu_total_mcpu_per_node=2000,
            mem_total_mb_per_node=2048,
            bandwidth_mode="random_uniform",
            bandwidth_mbps_min=1000,
            bandwidth_mbps_max=2000,
        )
        function_ids = {"a", "b", "c"}
        p1, _ = build_function_profiles(
            function_ids=function_ids,
            runtime_cfg=runtime,
            physical_cfg=physical,
            base_seed=42,
        )
        p2, _ = build_function_profiles(
            function_ids=function_ids,
            runtime_cfg=runtime,
            physical_cfg=physical,
            base_seed=42,
        )
        self.assertEqual(p1, p2)

    def test_random_bandwidth_matrix_is_reproducible(self) -> None:
        physical = PhysicalNodesConfig(
            count=3,
            max_containers_per_node=2,
            cpu_total_mcpu_per_node=2000,
            mem_total_mb_per_node=2048,
            bandwidth_mode="random_uniform",
            bandwidth_mbps_min=1000,
            bandwidth_mbps_max=2000,
            bandwidth_seed_offset=101,
        )
        node_ids = ["host-1", "host-2", "host-3"]
        m1, _ = build_bandwidth_matrix(node_ids=node_ids, physical_cfg=physical, base_seed=42)
        m2, _ = build_bandwidth_matrix(node_ids=node_ids, physical_cfg=physical, base_seed=42)
        self.assertEqual(m1, m2)
        self.assertNotIn("host-1", m1["host-1"])


if __name__ == "__main__":
    unittest.main()
