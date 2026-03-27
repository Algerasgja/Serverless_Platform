import unittest

from simulator.strategies.autoscaler.base import ScaleDownPlan
from simulator.strategies.autoscaler.hptd import HptdAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={"__start__": {"A": 1.0}, "A": {"B": 1.0}},
            node_latency_ms={"A": 1, "B": 1},
        )
    }


class HptdAutoscalerTests(unittest.TestCase):
    def _build(self) -> HptdAutoscaler:
        return HptdAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            time_granularity_sec=1,
            wcall_t=20,
            whistory_t=20,
            wchange_t=4,
            alpha=0.4,
            beta=1.2,
            mu_floor=0.01,
            std_floor=0.0,
            temp_floor=1e-6,
            scale_max_step=8,
            min_replicas=0,
            max_replicas=20,
        )

    def test_no_traffic_no_scale_plan(self) -> None:
        scaler = self._build()
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        self.assertEqual([], plans)

    def test_recent_burst_triggers_scale_out(self) -> None:
        scaler = self._build()
        for sec in range(10):
            scaler.on_step_start(request_id=f"b{sec}", um="u1", function_node="A", timestamp_ms=sec * 1000)
            scaler.on_step_observed(
                request_id=f"b{sec}",
                um="u1",
                function_node="A",
                timestamp_ms=sec * 1000 + 1,
                execution_ms=20,
                cold_start_ms=0,
                transfer_ms=0,
                prefix=("A",),
            )
            scaler.on_tick(timestamp_sec=sec, timestamp_ms=sec * 1000, ready_pool_by_function={})

        # burst on sec=10
        for idx in range(8):
            rid = f"burst-{idx}"
            scaler.on_step_start(request_id=rid, um="u1", function_node="A", timestamp_ms=10_000)
            scaler.on_step_observed(
                request_id=rid,
                um="u1",
                function_node="A",
                timestamp_ms=10_001,
                execution_ms=20,
                cold_start_ms=0,
                transfer_ms=0,
                prefix=("A",),
            )
        plans = scaler.on_tick(timestamp_sec=10, timestamp_ms=10_000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertGreater(plan_map.get("A", 0), 0)

    def test_scale_out_when_current_capacity_is_zero(self) -> None:
        scaler = self._build()
        for sec in range(6):
            for idx in range(6):
                rid = f"r{sec}-{idx}"
                scaler.on_step_start(request_id=rid, um="u1", function_node="A", timestamp_ms=sec * 1000)
                scaler.on_step_observed(
                    request_id=rid,
                    um="u1",
                    function_node="A",
                    timestamp_ms=sec * 1000 + 1,
                    execution_ms=20,
                    cold_start_ms=0,
                    transfer_ms=0,
                    prefix=("A",),
                )
            plans = scaler.on_tick(timestamp_sec=sec, timestamp_ms=sec * 1000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertGreaterEqual(plan_map.get("A", 0), 1)

    def test_temperature_decays_over_time(self) -> None:
        scaler = self._build()
        scaler.on_step_start(request_id="x1", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={})
        temp_now = scaler._current_temp_by_function.get("A", 0.0)  # noqa: SLF001

        for sec in range(1, 15):
            scaler.on_tick(timestamp_sec=sec, timestamp_ms=sec * 1000, ready_pool_by_function={})
        temp_later = scaler._current_temp_by_function.get("A", 0.0)  # noqa: SLF001
        self.assertLessEqual(temp_later, temp_now + 1e-6)

    def test_scale_down_is_capped_by_idle_pool(self) -> None:
        scaler = self._build()
        for sec in range(6):
            for idx in range(5):
                rid = f"seed-{sec}-{idx}"
                scaler.on_step_start(request_id=rid, um="u1", function_node="A", timestamp_ms=sec * 1000)
                scaler.on_step_observed(
                    request_id=rid,
                    um="u1",
                    function_node="A",
                    timestamp_ms=sec * 1000 + 1,
                    execution_ms=15,
                    cold_start_ms=0,
                    transfer_ms=0,
                    prefix=("A",),
                )
            scaler.on_tick(timestamp_sec=sec, timestamp_ms=sec * 1000, ready_pool_by_function={})

        plans = scaler.on_tick(
            timestamp_sec=7,
            timestamp_ms=7000,
            ready_pool_by_function={"A": 100},
            idle_pool_by_function={"A": 20},
        )
        self.assertEqual(1, len(plans))
        self.assertIsInstance(plans[0], ScaleDownPlan)
        self.assertEqual(20, plans[0].count)


if __name__ == "__main__":
    unittest.main()
