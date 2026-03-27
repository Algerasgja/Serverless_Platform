import unittest
from pathlib import Path

from analysis.ablation_experiments import build_ablation_scenario_variants, build_ablation_variants
from analysis.compare_experiments import (
    METRIC_COLD_STEP_RATE,
    METRIC_E2E_BUNDLE,
    METRIC_PRED_PREWARM,
    METRIC_PREWARM_COST,
    autoscaler_label,
    build_compare_variants,
    default_compare_autoscalers,
    normalize_metric_ids,
    normalize_scenarios,
    required_autoscalers_for_metrics,
)
from analysis.experiment_common import RunMetric, aggregate_metrics, build_variant_configs
from analysis.hparam_experiments import HPWP_DEFAULT_POINT, build_hparam_parameter_sets


class ExperimentSuiteTests(unittest.TestCase):
    def test_compare_variant_labels_are_mapped(self) -> None:
        variants = build_compare_variants(autoscalers=["hpwp_v1", "kpa_v1", "xanadu_opt_v1"])
        labels = [item["label"] for item in variants]
        self.assertEqual(["hpwp", "kpa", "xanadu"], labels)
        self.assertEqual("hpwp", autoscaler_label("hpwp_v1"))
        self.assertEqual("hist", autoscaler_label("hist_keepalive_prewarm_v1"))
        self.assertEqual("lass", autoscaler_label("lass_v1"))
        self.assertEqual("rl_q", autoscaler_label("rl_q_v1"))
        self.assertEqual("hptd", autoscaler_label("hptd_v1"))
        self.assertEqual("no_as", autoscaler_label("no_autoscale_v1"))
        self.assertEqual("xanadu", autoscaler_label("xanadu_opt_v1"))
        self.assertEqual("xanadu_legacy", autoscaler_label("xanadu_v1"))
        self.assertEqual("oracle", autoscaler_label("oracle_future_v1"))
        self.assertEqual("depth_breadth", autoscaler_label("depth_breadth_v1"))
        self.assertEqual("kraken_vomm", autoscaler_label("kraken_vomm_v1"))

    def test_compare_defaults(self) -> None:
        self.assertEqual(
            ["low", "high"],
            normalize_scenarios(["low", "high"]),
        )
        self.assertEqual(
            ["hpwp_v1", "kpa_v1", "lass_v1", "rl_q_v1", "hptd_v1", "xanadu_opt_v1", "oracle_future_v1", "depth_breadth_v1", "kraken_vomm_v1"],
            default_compare_autoscalers(),
        )

    def test_compare_variants_expand_with_scenarios(self) -> None:
        variants = build_compare_variants(
            autoscalers=["hpwp_v1", "xanadu_opt_v1"],
            scenario_rate_multipliers=[("low", 0.5), ("high", 2.0)],
            base_rate_multiplier=0.1,
        )
        labels = [item["label"] for item in variants]
        self.assertEqual(
            ["low::hpwp", "low::xanadu", "high::hpwp", "high::xanadu"],
            labels,
        )
        rates = [item["overrides"]["workload"]["rate_multiplier"] for item in variants]
        self.assertEqual([0.05, 0.05, 0.2, 0.2], rates)

    def test_compare_metric_selector(self) -> None:
        self.assertEqual(
            [METRIC_E2E_BUNDLE, METRIC_COLD_STEP_RATE],
            normalize_metric_ids([METRIC_E2E_BUNDLE, METRIC_COLD_STEP_RATE]),
        )
        self.assertEqual(
            ["hpwp_v1", "xanadu_v1", "no_autoscale_v1"],
            required_autoscalers_for_metrics(
                metric_ids=[METRIC_PRED_PREWARM],
                autoscalers=["hpwp_v1", "xanadu_v1", "no_autoscale_v1"],
            ),
        )
        self.assertEqual(
            ["hpwp_v1", "kpa_v1"],
            required_autoscalers_for_metrics(
                metric_ids=[METRIC_PREWARM_COST],
                autoscalers=["hpwp_v1", "kpa_v1"],
            ),
        )

    def test_ablation_progressive_and_legacy_groups(self) -> None:
        frozen = {
            "hpwp_sched_eta_exec": 0.5,
            "hpwp_horizon_alpha": 1.5,
            "hpwp_beta_hi": 60.0,
            "hpwp_beta_lo": 10.0,
            "hpwp_alpha_stable": 0.08,
        }
        progressive = build_ablation_variants(scheme="progressive", frozen_params=frozen)
        labels = [item["label"] for item in progressive]
        self.assertEqual(
            ["g0_core_min", "g1_plus_hierarchy", "g2_plus_urgency", "g3_plus_phase", "g4_full"],
            labels,
        )
        for item in progressive:
            autoscaler = item["overrides"]["autoscaler"]
            self.assertEqual("hpwp_v1", autoscaler["type"])
            for key, val in frozen.items():
                self.assertEqual(val, autoscaler[key])

        legacy = build_ablation_variants(scheme="legacy", frozen_params=frozen)
        legacy_labels = [item["label"] for item in legacy]
        self.assertEqual(
            ["full_hpwp", "no_hierarchy", "no_phase_adapt", "no_drift_handler", "no_urgency_gate"],
            legacy_labels,
        )

    def test_ablation_scenario_variants_expand(self) -> None:
        frozen = {
            "hpwp_sched_eta_exec": 0.5,
            "hpwp_horizon_alpha": 1.5,
            "hpwp_beta_hi": 60.0,
            "hpwp_beta_lo": 10.0,
            "hpwp_alpha_stable": 0.08,
        }
        variants = build_ablation_scenario_variants(
            base_rate_multiplier=0.1,
            scenario_factors=[("low", 0.5), ("mid", 1.0), ("high", 2.0)],
            scheme="progressive",
            frozen_params=frozen,
        )
        self.assertEqual(15, len(variants))
        labels = [item["label"] for item in variants[:5]]
        self.assertEqual(
            ["low::g0_core_min", "low::g1_plus_hierarchy", "low::g2_plus_urgency", "low::g3_plus_phase", "low::g4_full"],
            labels,
        )
        rates = [item["overrides"]["workload"]["rate_multiplier"] for item in variants]
        self.assertEqual(5, rates.count(0.05))
        self.assertEqual(5, rates.count(0.1))
        self.assertEqual(5, rates.count(0.2))

    def test_hparam_sampling_size_and_default_point(self) -> None:
        points = build_hparam_parameter_sets(trial_count=24, random_seed=20260313)
        self.assertEqual(24, len(points))
        self.assertEqual(HPWP_DEFAULT_POINT, points[0])
        keys = {
            (
                p["hpwp_sched_eta_exec"],
                p["hpwp_horizon_alpha"],
                p["hpwp_beta_hi"],
                p["hpwp_beta_lo"],
                p["hpwp_alpha_stable"],
            )
            for p in points
        }
        self.assertEqual(24, len(keys))

    def test_build_variant_configs_with_seeds(self) -> None:
        base = {"experiment": {"name": "demo", "random_seed": 1}, "autoscaler": {"type": "kpa_v1"}}
        variants = build_variant_configs(
            base_config=base,
            base_experiment_name="demo",
            variant_overrides=[
                {"label": "hpwp", "overrides": {"autoscaler": {"type": "hpwp_v1"}}},
            ],
            seeds=[42, 43, 44],
        )
        self.assertEqual(3, len(variants))
        self.assertEqual(["hpwp", "hpwp", "hpwp"], [item.label for item in variants])
        self.assertEqual([42, 43, 44], [item.seed for item in variants])
        self.assertTrue(all(item.config_payload["autoscaler"]["type"] == "hpwp_v1" for item in variants))

    def test_aggregate_metrics_mean_std(self) -> None:
        rows = aggregate_metrics(
            [
                RunMetric(
                    label="hpwp",
                    seed=42,
                    run_dir=Path("runs/a"),
                    avg_e2e_ms=100.0,
                    p50_ms=90.0,
                    p95_ms=120.0,
                    p99_ms=150.0,
                    success_rate=1.0,
                    cold_start_share=0.5,
                ),
                RunMetric(
                    label="hpwp",
                    seed=43,
                    run_dir=Path("runs/b"),
                    avg_e2e_ms=120.0,
                    p50_ms=95.0,
                    p95_ms=130.0,
                    p99_ms=170.0,
                    success_rate=0.9,
                    cold_start_share=0.6,
                ),
            ]
        )
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("hpwp", row["label"])
        self.assertAlmostEqual(110.0, row["avg_e2e_ms_mean"])
        self.assertGreater(row["avg_e2e_ms_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
