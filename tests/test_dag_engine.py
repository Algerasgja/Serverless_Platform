import random
import unittest

from simulator.dag.engine import ConditionalDagEngine
from simulator.types import DagTemplate


class DagEngineTests(unittest.TestCase):
    def test_generates_single_path(self) -> None:
        template = DagTemplate(
            um="search",
            transitions={
                "__start__": {"gateway": 1.0},
                "gateway": {"auth": 1.0},
                "auth": {"response": 1.0},
            },
            node_latency_ms={"gateway": 10, "auth": 10, "response": 10},
        )
        engine = ConditionalDagEngine(
            templates={"search": template},
            session_gap_sec=30,
            session_continue_prob=1.0,
            context_alpha=0.5,
            rng=random.Random(11),
        )
        sid = engine.assign_session("search", now_sec=0)
        path = engine.generate_path("search", sid)
        self.assertEqual(["gateway", "auth", "response"], path)

    def test_session_context_biases_future_branches(self) -> None:
        template = DagTemplate(
            um="search",
            transitions={
                "__start__": {"gateway": 1.0},
                "gateway": {"auth": 0.5, "cache": 0.5},
                "auth": {"response": 1.0},
                "cache": {"response": 1.0},
            },
            node_latency_ms={},
        )
        engine = ConditionalDagEngine(
            templates={"search": template},
            session_gap_sec=300,
            session_continue_prob=1.0,
            context_alpha=2.0,
            rng=random.Random(7),
        )
        sid = engine.assign_session("search", now_sec=0)
        first = engine.generate_path("search", sid)
        branch = first[1]
        same_branch = 0
        for _ in range(20):
            path = engine.generate_path("search", sid)
            if path[1] == branch:
                same_branch += 1
        self.assertGreaterEqual(same_branch, 12)

    def test_fixed_mode_prefix_is_reproducible(self) -> None:
        template = _two_stage_branch_template("dag_0001")
        engine_a = ConditionalDagEngine(
            templates={"dag_0001": template},
            session_gap_sec=60,
            session_continue_prob=1.0,
            context_alpha=0.0,
            rng=random.Random(13),
            path_rule="mode_prefix_coupled_v1",
            context_regime="fixed",
            mode_count=3,
            prefix_window=3,
            base_seed=42,
            coupling_seed_offset=707,
        )
        engine_b = ConditionalDagEngine(
            templates={"dag_0001": template},
            session_gap_sec=60,
            session_continue_prob=1.0,
            context_alpha=0.0,
            rng=random.Random(13),
            path_rule="mode_prefix_coupled_v1",
            context_regime="fixed",
            mode_count=3,
            prefix_window=3,
            base_seed=42,
            coupling_seed_offset=707,
        )

        seq_a: list[list[str]] = []
        seq_b: list[list[str]] = []
        for sec in range(80):
            sid_a = engine_a.assign_session("dag_0001", now_sec=sec)
            sid_b = engine_b.assign_session("dag_0001", now_sec=sec)
            seq_a.append(engine_a.generate_path("dag_0001", sid_a, now_sec=sec))
            seq_b.append(engine_b.generate_path("dag_0001", sid_b, now_sec=sec))
        self.assertEqual(seq_a, seq_b)

        summary = engine_a.path_model_summary()
        item = summary["per_um"]["dag_0001"]
        self.assertEqual(item["pi_fixed"], item["pi_fixed"])

    def test_prefix_influences_later_branch_in_mode_prefix(self) -> None:
        template = _two_stage_branch_template("dag_0001")
        engine = ConditionalDagEngine(
            templates={"dag_0001": template},
            session_gap_sec=60,
            session_continue_prob=1.0,
            context_alpha=0.0,
            rng=random.Random(17),
            path_rule="mode_prefix_coupled_v1",
            context_regime="fixed",
            mode_count=1,
            mode_strength=0.0,
            prefix_strength=4.0,
            prefix_decay=1.0,
            prefix_window=3,
            base_seed=99,
            coupling_seed_offset=707,
        )

        x_after_a = 0
        total_a = 0
        x_after_b = 0
        total_b = 0
        for sec in range(5000):
            sid = engine.assign_session("dag_0001", now_sec=sec)
            path = engine.generate_path("dag_0001", sid, now_sec=sec)
            # path shape: entry -> (a|b) -> (x|y) -> end
            first_branch = path[1]
            second_branch = path[2]
            if first_branch == "a":
                total_a += 1
                if second_branch == "x":
                    x_after_a += 1
            else:
                total_b += 1
                if second_branch == "x":
                    x_after_b += 1

        p_x_a = x_after_a / max(1, total_a)
        p_x_b = x_after_b / max(1, total_b)
        self.assertGreater(total_a, 500)
        self.assertGreater(total_b, 500)
        self.assertGreater(abs(p_x_a - p_x_b), 0.03)

    def test_drifting_updates_mode_distribution_and_strength_zero_degenerates(self) -> None:
        template = _two_stage_branch_template("dag_0001")
        drifting_engine = ConditionalDagEngine(
            templates={"dag_0001": template},
            session_gap_sec=60,
            session_continue_prob=1.0,
            context_alpha=0.0,
            rng=random.Random(19),
            path_rule="mode_prefix_coupled_v1",
            context_regime="drifting",
            mode_count=3,
            drifting_interval_sec=5,
            drifting_strength=0.2,
            drifting_concentration=80.0,
            base_seed=123,
            coupling_seed_offset=707,
        )
        for sec in range(0, 120, 2):
            sid = drifting_engine.assign_session("dag_0001", now_sec=sec)
            drifting_engine.generate_path("dag_0001", sid, now_sec=sec)
        drift_item = drifting_engine.path_model_summary()["per_um"]["dag_0001"]
        pi_initial = drift_item["pi_initial"]
        pi_current = drift_item["pi_current"]
        l1 = sum(abs(a - b) for a, b in zip(pi_initial, pi_current))
        self.assertGreater(l1, 1e-3)

        fixed_like_engine = ConditionalDagEngine(
            templates={"dag_0001": template},
            session_gap_sec=60,
            session_continue_prob=1.0,
            context_alpha=0.0,
            rng=random.Random(19),
            path_rule="mode_prefix_coupled_v1",
            context_regime="drifting",
            mode_count=3,
            drifting_interval_sec=5,
            drifting_strength=0.0,
            drifting_concentration=80.0,
            base_seed=123,
            coupling_seed_offset=707,
        )
        for sec in range(0, 120, 2):
            sid = fixed_like_engine.assign_session("dag_0001", now_sec=sec)
            fixed_like_engine.generate_path("dag_0001", sid, now_sec=sec)
        fixed_like_item = fixed_like_engine.path_model_summary()["per_um"]["dag_0001"]
        self.assertEqual(fixed_like_item["pi_initial"], fixed_like_item["pi_current"])


def _two_stage_branch_template(um: str) -> DagTemplate:
    return DagTemplate(
        um=um,
        transitions={
            "__start__": {"entry": 1.0},
            "entry": {"a": 0.5, "b": 0.5},
            "a": {"x": 0.5, "y": 0.5},
            "b": {"x": 0.5, "y": 0.5},
            "x": {"end": 1.0},
            "y": {"end": 1.0},
        },
        node_latency_ms={"entry": 10, "a": 10, "b": 10, "x": 10, "y": 10, "end": 10},
    )


if __name__ == "__main__":
    unittest.main()
