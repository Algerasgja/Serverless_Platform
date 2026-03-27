import unittest

from simulator.strategies.autoscaler.base import ScaleDownPlan
from simulator.strategies.autoscaler.depth_breadth import DepthBreadthAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={
                "__start__": {"A": 1.0},
                "A": {"B": 0.5, "C": 0.5},
                "B": {"D": 1.0},
                "C": {"E": 1.0},
            },
            node_latency_ms={"A": 1, "B": 1, "C": 1, "D": 1, "E": 1},
        )
    }


class DepthBreadthAutoscalerTests(unittest.TestCase):
    def test_reachable_nodes_exclude_current_node(self) -> None:
        scaler = DepthBreadthAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
            guard_buffer_min=0,
            down_margin=0,
            min_idle_age_sec=0,
            down_cooldown_sec=0,
            max_down_ratio=1.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertNotIn("A", plan_map)
        self.assertEqual(1, plan_map.get("B"))
        self.assertEqual(1, plan_map.get("C"))
        self.assertEqual(1, plan_map.get("D"))
        self.assertEqual(1, plan_map.get("E"))

    def test_horizon_gate_filters_unreachable_nodes(self) -> None:
        scaler = DepthBreadthAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=800.0,
            default_trans_ms=300.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([], plans)

    def test_sync_period_and_pair_cleanup(self) -> None:
        scaler = DepthBreadthAutoscaler(
            templates=_templates(),
            sync_period_sec=5,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        self.assertEqual([], scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={}))
        plans = scaler.on_tick(timestamp_sec=5, timestamp_ms=5000, ready_pool_by_function={})
        self.assertTrue(len(plans) > 0)
        scaler.on_request_finish(request_id="r1", status="completed", timestamp_ms=6000)
        plans_after = scaler.on_tick(timestamp_sec=10, timestamp_ms=10000, ready_pool_by_function={})
        self.assertEqual([], plans_after)

    def test_multi_request_accumulates_and_ready_pool_deducts(self) -> None:
        scaler = DepthBreadthAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
            guard_buffer_min=0,
            down_margin=0,
            min_idle_age_sec=0,
            down_cooldown_sec=0,
            max_down_ratio=1.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_step_start(request_id="r2", um="u1", function_node="A", timestamp_ms=1)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={"B": 1})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertEqual(1, plan_map.get("B"))
        self.assertEqual(2, plan_map.get("C"))

    def test_no_active_scale_down_plan_when_ready_exceeds_desired(self) -> None:
        scaler = DepthBreadthAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
            guard_buffer_min=0,
            down_margin=0,
            min_idle_age_sec=0,
            down_cooldown_sec=0,
            max_down_ratio=1.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(
            timestamp_sec=1,
            timestamp_ms=1000,
            ready_pool_by_function={"B": 3},
            idle_pool_by_function={"B": 1},
        )
        self.assertEqual([], [p for p in plans if isinstance(p, ScaleDownPlan)])
        plan_map = {p.function_node: p.count for p in plans if not isinstance(p, ScaleDownPlan)}
        self.assertEqual(1, plan_map.get("C"))
        self.assertEqual(1, plan_map.get("D"))
        self.assertEqual(1, plan_map.get("E"))

    def test_no_active_scale_down_plan_even_with_idle_pool(self) -> None:
        scaler = DepthBreadthAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
            guard_buffer_min=0,
            down_margin=0,
            min_idle_age_sec=0,
            down_cooldown_sec=0,
            max_down_ratio=1.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(
            timestamp_sec=1,
            timestamp_ms=1000,
            ready_pool_by_function={"B": 3},
            idle_pool_by_function={"B": 0},
        )
        self.assertEqual([], [p for p in plans if isinstance(p, ScaleDownPlan)])
        plan_map = {p.function_node: p.count for p in plans if not isinstance(p, ScaleDownPlan)}
        self.assertNotIn("B", plan_map)
        self.assertEqual(1, plan_map.get("C"))
        self.assertEqual(1, plan_map.get("D"))
        self.assertEqual(1, plan_map.get("E"))

if __name__ == "__main__":
    unittest.main()
