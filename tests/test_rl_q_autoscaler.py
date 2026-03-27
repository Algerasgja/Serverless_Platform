import unittest

from simulator.strategies.autoscaler.rl_q import RlQAutoscaler
from simulator.types import DagTemplate


def _templates() -> dict[str, DagTemplate]:
    return {
        "u1": DagTemplate(
            um="u1",
            transitions={"__start__": {"A": 1.0}, "A": {"B": 1.0}},
            node_latency_ms={"A": 1, "B": 1},
        )
    }


class RlQAutoscalerTests(unittest.TestCase):
    def _build(
        self,
        *,
        epsilon_init: float = 0.8,
        epsilon_decay: float = 0.9,
        epsilon_min: float = 0.1,
        learning_rate: float = 0.5,
        discount_factor: float = 0.9,
    ) -> RlQAutoscaler:
        return RlQAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            time_granularity_sec=1,
            wcall_t=20,
            whistory_t=50,
            wchange_t=10,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_init=epsilon_init,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            step_size=1,
            util_threshold=0.8,
            reward_tolerance=0.05,
            scalability_alpha=0.15,
            inhibit_token_max=3,
            min_replicas=0,
            max_replicas=20,
        )

    def test_unseen_state_uses_util_heuristic(self) -> None:
        scaler = self._build(epsilon_init=0.0, epsilon_decay=1.0, epsilon_min=0.0)
        for idx in range(9):
            scaler.on_step_start(
                request_id=f"r{idx}",
                um="u1",
                function_node="A",
                timestamp_ms=0,
            )
        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={"A": 1})
        self.assertGreaterEqual(sum(p.count for p in plans if p.function_node == "A"), 1)
        self.assertEqual(1, scaler.summary()["action_counts"]["+1"])

    def test_epsilon_decay_has_floor(self) -> None:
        scaler = self._build(epsilon_init=0.5, epsilon_decay=0.1, epsilon_min=0.2)
        for sec in range(6):
            scaler.on_tick(timestamp_sec=sec, timestamp_ms=sec * 1000, ready_pool_by_function={})
        self.assertAlmostEqual(0.2, scaler.summary()["epsilon_final"])

    def test_reward_tolerance_band(self) -> None:
        scaler = self._build(epsilon_init=0.0, epsilon_decay=1.0, epsilon_min=0.0)
        reward_flat = scaler._compute_reward(  # noqa: SLF001
            throughput_i=10.0,
            throughput_ref=10.0,
            resource_i=2.0,
            resource_ref=2.0,
        )
        reward_gain = scaler._compute_reward(  # noqa: SLF001
            throughput_i=12.0,
            throughput_ref=10.0,
            resource_i=2.0,
            resource_ref=2.0,
        )
        self.assertEqual(1.0, reward_flat)
        self.assertAlmostEqual(1.2, reward_gain)

    def test_q_update_formula(self) -> None:
        scaler = self._build(
            epsilon_init=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
            learning_rate=0.5,
            discount_factor=0.9,
        )
        s_prev = (1, 1, 1, 1)
        s_next = (2, 2, 2, 2)
        scaler._q[s_prev] = {-1: 0.0, 0: 0.0, 1: 0.0}  # noqa: SLF001
        scaler._q[s_next] = {-1: 0.0, 0: 0.0, 1: 1.0}  # noqa: SLF001
        scaler._update_q(prev_state=s_prev, action=1, reward=2.0, next_state=s_next)  # noqa: SLF001
        self.assertAlmostEqual(1.45, scaler._q[s_prev][1], places=6)  # noqa: SLF001

    def test_negative_action_no_scale_in_and_inhibit_following_scale_out(self) -> None:
        scaler = self._build(epsilon_init=0.0, epsilon_decay=1.0, epsilon_min=0.0)

        plans = scaler.on_tick(timestamp_sec=0, timestamp_ms=0, ready_pool_by_function={"A": 5})
        self.assertEqual([], plans)

        # Build enough throughput baseline to make vertical correction non-positive.
        for idx in range(200):
            scaler.on_step_observed(
                request_id=f"obs-{idx}",
                um="u1",
                function_node="A",
                timestamp_ms=1000,
                execution_ms=10,
                cold_start_ms=0,
                transfer_ms=0,
                prefix=("A",),
            )

        scaler.on_step_start(request_id="r-hot", um="u1", function_node="A", timestamp_ms=2000)
        plans = scaler.on_tick(timestamp_sec=2, timestamp_ms=2000, ready_pool_by_function={})
        self.assertEqual([], plans)

        summary = scaler.summary()
        self.assertEqual(1, summary["action_counts"]["-1"])
        self.assertEqual(1, summary["action_counts"]["+1"])
        self.assertEqual(1, summary["inhibit_events"])


if __name__ == "__main__":
    unittest.main()

