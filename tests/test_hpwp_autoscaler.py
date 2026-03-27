import unittest

from simulator.strategies.autoscaler.hpwp import HpwpAutoscaler
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


def _build_scaler(**overrides) -> HpwpAutoscaler:
    params = {
        "templates": _templates(),
        "sync_period_sec": 1,
        "lmax_low": 2,
        "lmax_high": 4,
        "beta_hi": 80.0,
        "beta_lo": 12.0,
        "alpha_exp": 0.8,
        "alpha_stable": 0.15,
        "sched_eta_exec": 0.5,
        "sched_min_sec": 1,
        "sched_max_sec": 15,
        "horizon_alpha": 2.0,
        "urgency_epsilon_ms": 5.0,
        "phase_window_k": 10,
        "phase_n_min": 2,
        "phase_var_threshold": 0.05,
        "drift_short_k": 2,
        "drift_long_k": 6,
        "drift_delta_mr": 0.2,
        "drift_tau_mr": 0.4,
        "drift_branch_tau": 0.4,
        "forget_gamma": 0.35,
        "default_exec_ms": 40.0,
        "default_cold_ms": 1500.0,
        "default_trans_ms": 8.0,
        "seed_offset": 2027,
    }
    params.update(overrides)
    return HpwpAutoscaler(**params)


class HpwpAutoscalerTests(unittest.TestCase):
    def test_tick_outputs_scale_up_plans_by_desired_gap(self) -> None:
        scaler = _build_scaler()
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )

        plans = scaler.on_tick(
            timestamp_sec=0,
            timestamp_ms=0,
            ready_pool_by_function={},
        )
        plan_map = {p.function_node: p.count for p in plans}
        self.assertGreaterEqual(plan_map.get("B", 0), 1)
        self.assertGreaterEqual(plan_map.get("D", 0), 1)
        summary = scaler.summary()
        desired = summary["final_desired_by_function"]
        self.assertGreaterEqual(desired.get("B", 0), 1)
        self.assertGreaterEqual(desired.get("D", 0), 1)
        self.assertGreaterEqual(summary.get("prewarm_create_attempted", 0), 2)

    def test_transition_updates_preferred_next_hop(self) -> None:
        scaler = _build_scaler()
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )
        scaler.on_transition(
            request_id="r1",
            um="u1",
            src_node="A",
            dst_node="C",
            timestamp_ms=10,
            transfer_ms=12,
            prefix=("A",),
        )
        scaler.on_tick(
            timestamp_sec=1,
            timestamp_ms=1000,
            ready_pool_by_function={},
        )

        desired = scaler.summary()["final_desired_by_function"]
        self.assertGreaterEqual(desired.get("C", 0), 1)
        self.assertEqual(0, desired.get("B", 0))

    def test_drift_detection_switches_to_exploration(self) -> None:
        scaler = _build_scaler(
            phase_window_k=4,
            phase_n_min=1,
            phase_var_threshold=1.0,
            drift_short_k=2,
            drift_long_k=4,
            drift_delta_mr=0.2,
            drift_tau_mr=0.45,
        )
        scaler.on_step_start(
            request_id="r1",
            um="u1",
            function_node="A",
            timestamp_ms=0,
        )

        # Warm up with mostly correct transitions.
        for sec in range(4):
            scaler.on_transition(
                request_id="r1",
                um="u1",
                src_node="A",
                dst_node="B",
                timestamp_ms=sec * 1000,
                transfer_ms=8,
                prefix=("A",),
            )
            scaler.on_tick(
                timestamp_sec=sec,
                timestamp_ms=sec * 1000,
                ready_pool_by_function={},
            )

        # Introduce misses to trigger drift.
        for sec in range(4, 7):
            scaler.on_transition(
                request_id="r1",
                um="u1",
                src_node="A",
                dst_node="C",
                timestamp_ms=sec * 1000,
                transfer_ms=8,
                prefix=("A",),
            )
            scaler.on_tick(
                timestamp_sec=sec,
                timestamp_ms=sec * 1000,
                ready_pool_by_function={},
            )

        summary = scaler.summary()
        self.assertGreaterEqual(summary["drift_events"], 1)
        self.assertEqual("exploration", summary["phase"])


if __name__ == "__main__":
    unittest.main()
