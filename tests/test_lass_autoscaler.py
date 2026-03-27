import unittest

from simulator.strategies.autoscaler.lass import LassAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={"__start__": {"A": 1.0}, "A": {"B": 1.0}},
            node_latency_ms={"A": 1, "B": 1},
        )
    }


class LassAutoscalerTests(unittest.TestCase):
    def test_formula_generates_expected_desired(self) -> None:
        scaler = LassAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            latency_target_ms=1000.0,
            load_window_sec=10,
            speed_ewma_alpha=0.2,
            min_speed_req_per_sec=0.05,
            min_samples=1,
            default_exec_ms=40.0,
            min_replicas=0,
            max_replicas=20,
        )

        for i in range(30):
            scaler.on_step_start(request_id=f"r{i}", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_step_observed(
            request_id="r0",
            um="u1",
            function_node="A",
            timestamp_ms=10,
            execution_ms=100,
            cold_start_ms=0,
            transfer_ms=0,
            prefix=("A",),
        )
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        # Queueing model finds c=1 sufficient under this light load.
        self.assertEqual(1, plan_map.get("A"))

    def test_insufficient_samples_uses_default_exec_speed(self) -> None:
        scaler = LassAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            latency_target_ms=1000.0,
            load_window_sec=10,
            speed_ewma_alpha=0.2,
            min_speed_req_per_sec=0.05,
            min_samples=5,
            default_exec_ms=50.0,
            min_replicas=0,
            max_replicas=20,
        )

        for i in range(10):
            scaler.on_step_start(request_id=f"r{i}", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        # With default service speed and light load, one container is sufficient.
        self.assertEqual(1, plan_map.get("A"))

    def test_ready_pool_suppresses_create(self) -> None:
        scaler = LassAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            latency_target_ms=1000.0,
            load_window_sec=10,
            speed_ewma_alpha=0.2,
            min_speed_req_per_sec=0.05,
            min_samples=1,
            default_exec_ms=40.0,
            min_replicas=0,
            max_replicas=20,
        )

        for i in range(30):
            scaler.on_step_start(request_id=f"r{i}", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_step_observed(
            request_id="r0",
            um="u1",
            function_node="A",
            timestamp_ms=10,
            execution_ms=100,
            cold_start_ms=0,
            transfer_ms=0,
            prefix=("A",),
        )
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={"A": 3})
        self.assertFalse(any(getattr(plan, "function_node", None) == "A" and plan.__class__.__name__ == "PrewarmPlan" for plan in plans))

    def test_tight_slo_requires_more_containers(self) -> None:
        scaler = LassAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            latency_target_ms=120.0,
            load_window_sec=10,
            speed_ewma_alpha=0.2,
            min_speed_req_per_sec=0.05,
            min_samples=1,
            default_exec_ms=40.0,
            min_replicas=0,
            max_replicas=20,
        )
        for i in range(30):
            scaler.on_step_start(request_id=f"r{i}", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_step_observed(
            request_id="r0",
            um="u1",
            function_node="A",
            timestamp_ms=10,
            execution_ms=100,
            cold_start_ms=0,
            transfer_ms=0,
            prefix=("A",),
        )
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertGreaterEqual(plan_map.get("A", 0), 2)


if __name__ == "__main__":
    unittest.main()
