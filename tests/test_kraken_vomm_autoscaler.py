import unittest

from simulator.strategies.autoscaler.kraken_vomm import KrakenVomMAutoscaler
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


class KrakenVomMAutoscalerTests(unittest.TestCase):
    def test_markov_probability_and_graph_factors(self) -> None:
        scaler = KrakenVomMAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
        )
        # Two active requests currently at A.
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_step_start(request_id="r2", um="u1", function_node="A", timestamp_ms=1)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}

        # From A:
        # P(B)=0.5, P(C)=0.5, P(D)=1.0
        # Conn(B)=Conn(C)=1/4, Conn(D)=0
        # Comm(B)=Comm(C)=1/2, Comm(D)=1
        # factor(B)=factor(C)=0.75, factor(D)=1
        # With default batch_size=2 and load split per active context:
        # batches=1, base(B)=base(C)=1, base(D)=1
        # desired(B)=base+extra=2, desired(C)=2, desired(D)=2
        self.assertEqual(2, plan_map.get("B"))
        self.assertEqual(2, plan_map.get("C"))
        self.assertEqual(2, plan_map.get("D"))
        self.assertNotIn("A", plan_map)

    def test_horizon_filters_far_nodes(self) -> None:
        scaler = KrakenVomMAutoscaler(
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

    def test_sync_period_and_ready_pool(self) -> None:
        scaler = KrakenVomMAutoscaler(
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
        plans = scaler.on_tick(timestamp_sec=5, timestamp_ms=5000, ready_pool_by_function={"D": 1})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertEqual(1, plan_map.get("D", 0))

    def test_request_finish_cleans_active_state(self) -> None:
        scaler = KrakenVomMAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
        )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        scaler.on_request_finish(request_id="r1", status="completed", timestamp_ms=100)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        self.assertEqual([], plans)

    def test_vomm_counts_shift_next_hop_distribution(self) -> None:
        scaler = KrakenVomMAutoscaler(
            templates=_templates(),
            sync_period_sec=1,
            sched_eta_exec=1.0,
            sched_min_sec=1,
            sched_max_sec=1,
            horizon_alpha=1.0,
            default_exec_ms=40.0,
            default_trans_ms=8.0,
            vomm_order_max=1,
            vomm_context_min_count=1,
            uniform_mix=0.0,
        )
        # Feed transitions that strongly bias A->C.
        for i in range(8):
            scaler.on_transition(
                um="u1",
                src_node="A",
                dst_node="C",
                timestamp_ms=10 + i,
                transfer_ms=8,
                prefix=("A",),
            )
        scaler.on_step_start(request_id="r1", um="u1", function_node="A", timestamp_ms=0)
        plans = scaler.on_tick(timestamp_sec=1, timestamp_ms=1000, ready_pool_by_function={})
        plan_map = {p.function_node: p.count for p in plans}
        self.assertGreaterEqual(plan_map.get("C", 0), plan_map.get("B", 0))


if __name__ == "__main__":
    unittest.main()
