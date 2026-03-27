import unittest

from simulator.strategies.autoscaler.base import ScaleDownPlan
from simulator.strategies.autoscaler.oracle import OracleFutureAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={
                "__start__": {"A": 1.0},
                "A": {"B": 1.0},
                "B": {"C": 1.0},
                "C": {"D": 1.0},
            },
            node_latency_ms={"A": 1, "B": 1, "C": 1, "D": 1},
        )
    }


class OracleAutoscalerTests(unittest.TestCase):
    def test_window_steps_limits_desired_pairs(self) -> None:
        scaler = OracleFutureAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            window_steps=1,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
            true_future_path=("B", "C", "D"),
        )
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([("B", 1)], [(p.function_node, p.count) for p in plans])

    def test_window_refresh_tracks_step_progress(self) -> None:
        scaler = OracleFutureAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            window_steps=2,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
            true_future_path=("B", "C", "D"),
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="B",
            timestamp_ms=10,
            true_future_path=("C", "D"),
        )
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertNotIn("B", plan_map)
        self.assertEqual(1, plan_map.get("C"))
        self.assertEqual(1, plan_map.get("D"))

    def test_request_finish_clears_plan_pairs(self) -> None:
        scaler = OracleFutureAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            window_steps=2,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
            true_future_path=("B", "C"),
        )
        scaler.on_request_finish(request_id="r1", status="completed", timestamp_ms=100)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([], plans)

    def test_scale_down_plan_is_emitted_when_ready_exceeds_desired(self) -> None:
        scaler = OracleFutureAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            window_steps=1,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
            true_future_path=("B",),
        )
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={"B": 2})
        self.assertEqual(1, len(plans))
        self.assertIsInstance(plans[0], ScaleDownPlan)
        self.assertEqual("B", plans[0].function_node)
        self.assertEqual(1, plans[0].count)

    def test_scale_down_is_capped_by_idle_pool(self) -> None:
        scaler = OracleFutureAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            window_steps=1,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
            true_future_path=("B",),
        )
        plans = scaler.on_tick(
            timestamp_sec=1,
            timestamp_ms=1000,
            ready_pool_by_function={"B": 3},
            idle_pool_by_function={"B": 1},
        )
        self.assertEqual(1, len(plans))
        self.assertIsInstance(plans[0], ScaleDownPlan)
        self.assertEqual(1, plans[0].count)


if __name__ == "__main__":
    unittest.main()
