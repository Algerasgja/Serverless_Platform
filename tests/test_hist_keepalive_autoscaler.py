import unittest

from simulator.strategies.autoscaler.hist_keepalive import HistogramKeepalivePrewarmAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={"__start__": {"A": 1.0}, "A": {"B": 1.0}},
            node_latency_ms={"A": 1, "B": 1},
        )
    }


class HistogramKeepalivePrewarmAutoscalerTests(unittest.TestCase):
    def test_recent_traffic_drives_keepalive_and_prewarm(self) -> None:
        scaler = HistogramKeepalivePrewarmAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            hist_window_sec=60,
            hist_quantile=0.9,
            keepalive_idle_sec=30,
            keepalive_min_replicas=1,
            prewarm_buffer=1,
            min_replicas=0,
            max_replicas=20,
        )

        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertGreaterEqual(plan_map.get("A", 0), 2)

    def test_ready_pool_suppresses_create_and_idle_can_decay(self) -> None:
        scaler = HistogramKeepalivePrewarmAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            hist_window_sec=20,
            hist_quantile=0.9,
            keepalive_idle_sec=5,
            keepalive_min_replicas=1,
            prewarm_buffer=1,
            min_replicas=0,
            max_replicas=20,
        )

        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={"A": 2})
        scaler.on_step_observed(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=200,
            execution_ms=50,
            cold_start_ms=0,
            transfer_ms=0,
            prefix=("A",),
        )

        for sec in range(1, 40):
            plans = scaler.on_tick(timestamp_sec=sec, timestamp_ms=sec * 1000, ready_pool_by_function={"A": 2})
            self.assertEqual([], plans)
        self.assertEqual({}, scaler.summary().get("final_desired_by_function", {}))


if __name__ == "__main__":
    unittest.main()
