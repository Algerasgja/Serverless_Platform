import unittest
from collections import deque

from simulator.strategies.autoscaler.kpa import KpaAutoscaler
from simulator.strategies.autoscaler.base import ScaleDownPlan
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={"__start__": {"A": 1.0}, "A": {"B": 1.0}},
            node_latency_ms={"A": 1, "B": 1},
        )
    }


class KpaAutoscalerTests(unittest.TestCase):
    def test_effective_target_uses_target_utilization(self) -> None:
        scaler = KpaAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            target_concurrency=2.0,
            target_utilization=0.5,
            use_target_utilization=True,
            stable_window_sec=2,
            panic_window_sec=1,
            panic_threshold=2.0,
            panic_min_hold_sec=0,
            panic_exit_streak_sec=1,
            max_scale_up_step=0,
            min_replicas=0,
            max_replicas=20,
        )
        scaler._history_by_function["A"] = deque([(0, 2), (1, 2)])  # noqa: SLF001
        desired = scaler._compute_desired_by_function(timestamp_sec=1)  # noqa: SLF001
        self.assertEqual(2, desired.get("A"))

    def test_panic_mode_holds_then_exits(self) -> None:
        scaler = KpaAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            target_concurrency=1.0,
            target_utilization=1.0,
            use_target_utilization=True,
            stable_window_sec=10,
            panic_window_sec=2,
            panic_threshold=1.2,
            panic_min_hold_sec=3,
            panic_exit_streak_sec=2,
            max_scale_up_step=0,
            min_replicas=0,
            max_replicas=20,
        )

        # Trigger panic with burst.
        for rid in ("r1", "r2", "r3"):
            scaler.on_step_start(request_id=rid, um="u1", function_node="A", timestamp_ms=0)
        scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        self.assertEqual("panic", scaler._mode_by_function.get("A"))  # noqa: SLF001

        # End burst.
        for rid in ("r1", "r2", "r3"):
            scaler.on_step_observed(
                request_id=rid,
                um="u1",
                function_node="A",
                timestamp_ms=500,
                execution_ms=100,
                cold_start_ms=0,
                transfer_ms=0,
                prefix=("A",),
            )

        scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        scaler.on_tick(timestamp_sec=2, timestamp_ms=2000, ready_pool_by_function={})
        # Hold period not satisfied yet.
        self.assertEqual("panic", scaler._mode_by_function.get("A"))  # noqa: SLF001

        scaler.on_tick(timestamp_sec=3, timestamp_ms=3000, ready_pool_by_function={})
        self.assertEqual("stable", scaler._mode_by_function.get("A"))  # noqa: SLF001

    def test_scale_plan_uses_ready_plus_inflight_capacity(self) -> None:
        scaler = KpaAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            target_concurrency=1.0,
            target_utilization=1.0,
            use_target_utilization=True,
            stable_window_sec=2,
            panic_window_sec=1,
            panic_threshold=2.0,
            panic_min_hold_sec=0,
            panic_exit_streak_sec=1,
            max_scale_up_step=0,
            min_replicas=0,
            max_replicas=20,
        )

        scaler._history_by_function["A"] = deque([(0, 2), (1, 2)])  # noqa: SLF001
        scaler.on_step_start(request_id="r-busy-1", um="u1", function_node="A", timestamp_ms=1000)
        scaler.on_step_start(request_id="r-busy-2", um="u1", function_node="A", timestamp_ms=1000)
        # desired ~=2, current capacity = ready(0)+inflight(2), so no extra create.
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([], plans)

    def test_scale_down_plan_is_emitted_when_ready_exceeds_desired(self) -> None:
        scaler = KpaAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            target_concurrency=1.0,
            target_utilization=1.0,
            use_target_utilization=True,
            stable_window_sec=2,
            panic_window_sec=1,
            panic_threshold=2.0,
            panic_min_hold_sec=0,
            panic_exit_streak_sec=1,
            max_scale_up_step=0,
            min_replicas=0,
            max_replicas=20,
        )

        scaler._history_by_function["A"] = deque([(0, 0), (1, 0)])  # noqa: SLF001
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={"A": 2})
        self.assertEqual(1, len(plans))
        self.assertIsInstance(plans[0], ScaleDownPlan)
        self.assertEqual("A", plans[0].function_node)
        self.assertEqual(2, plans[0].count)

    def test_scale_down_uses_idle_pool_not_ready_pool(self) -> None:
        scaler = KpaAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            target_concurrency=1.0,
            target_utilization=1.0,
            use_target_utilization=True,
            stable_window_sec=2,
            panic_window_sec=1,
            panic_threshold=2.0,
            panic_min_hold_sec=0,
            panic_exit_streak_sec=1,
            max_scale_up_step=0,
            min_replicas=0,
            max_replicas=20,
        )

        scaler._history_by_function["A"] = deque([(0, 0), (1, 0)])  # noqa: SLF001
        plans = scaler.on_tick(
            timestamp_sec=1,
            timestamp_ms=1000,
            ready_pool_by_function={"A": 3},
            idle_pool_by_function={"A": 1},
        )
        self.assertEqual(1, len(plans))
        self.assertIsInstance(plans[0], ScaleDownPlan)
        self.assertEqual(1, plans[0].count)


if __name__ == "__main__":
    unittest.main()
