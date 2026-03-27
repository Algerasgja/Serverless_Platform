import unittest

from simulator.strategies.autoscaler.xanadu import XanaduAutoscaler, XanaduOptimizedAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={
                "__start__": {"A": 1.0},
                "A": {"B": 0.5, "C": 0.5},
                "B": {"D": 1.0},
                "C": {"D": 1.0},
            },
            node_latency_ms={"A": 1, "B": 1, "C": 1, "D": 1},
        )
    }


class XanaduAutoscalerTests(unittest.TestCase):
    def test_ewma_transition_updates_prediction(self) -> None:
        scaler = XanaduAutoscaler(
            templates=_templates(),
            depth=1,
            ewma_alpha=0.2,
            sync_period_sec=1,
        )

        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([("B", 1)], [(p.function_node, p.count) for p in plans])

        scaler.on_transition(
            um="u1",
            src_node="A",
            dst_node="C",
            timestamp_ms=1010,
        )
        scaler.on_request_finish(request_id="r1", status="completed", timestamp_ms=1020)

        scaler.on_step_start(
            request_id="r2",
            um="u1",
            function_node="A",
            timestamp_ms=2000,
        )
        plans = scaler.on_tick(timestamp_sec=2, timestamp_ms=2000, ready_pool_by_function={})
        self.assertEqual([("C", 1)], [(p.function_node, p.count) for p in plans])

    def test_offpath_disables_future_planning(self) -> None:
        scaler = XanaduAutoscaler(
            templates=_templates(),
            depth=1,
            ewma_alpha=0.2,
            sync_period_sec=1,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="C",
            timestamp_ms=10,
        )

        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([], plans)
        self.assertEqual(1, scaler.summary()["offpath_disabled_requests"])

    def test_depth_counts_stack_across_requests(self) -> None:
        scaler = XanaduAutoscaler(
            templates=_templates(),
            depth=2,
            ewma_alpha=0.2,
            sync_period_sec=1,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )
        scaler.on_step_start(
            request_id="r2",
            um="u1",
            function_node="A",
            timestamp_ms=1,
        )
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertEqual(2, plan_map.get("B"))
        self.assertEqual(2, plan_map.get("D"))

    def test_optimized_xanadu_reenters_after_offpath(self) -> None:
        scaler = XanaduOptimizedAutoscaler(
            templates=_templates(),
            depth=1,
            ewma_alpha=0.2,
            sync_period_sec=1,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="C",
            timestamp_ms=10,
        )
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        # Off-path does not permanently disable request in optimized mode.
        self.assertEqual([("D", 1)], [(p.function_node, p.count) for p in plans])


if __name__ == "__main__":
    unittest.main()
