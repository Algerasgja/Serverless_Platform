from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings

import yaml


@dataclass
class ExperimentConfig:
    name: str
    duration_seconds: int
    random_seed: int


@dataclass
class DatasetConfig:
    source: str
    processed_dir: str
    time_window_minutes: int
    callgraph_file: str = "data/raw/unused_callgraph.csv"
    qps_file: str = "data/raw/unused_qps.csv"
    sample_um_count: int = 8
    checksum_required: bool = False
    dag_tasks_file: str = "data/raw/filtered_tasks.csv"
    dag_top_k: int = 20
    dag_selection_mode: str = "random_unique"
    dag_selection_seed: int = 42
    dag_max_nodes: int | None = None
    dag_max_edges: int | None = None
    dag_max_longest_path: int | None = None
    dag_max_splits: int | None = None


@dataclass
class DagPolicyConfig:
    granularity: str = "um"
    path_rule: str = "mode_prefix_coupled_v1"
    markov_order: int = 1
    session_gap_sec: int = 30
    session_continue_prob: float = 0.7
    context_alpha: float = 0.35
    context_regime: str = "drifting"
    context_drifting_enabled: bool = True
    mode_count: int = 3
    mode_prior_concentration: float = 2.0
    mode_strength: float = 1.0
    prefix_strength: float = 1.2
    prefix_decay: float = 0.85
    prefix_window: int = 3
    temperature: float = 1.0
    coupling_seed_offset: int = 707
    drifting_interval_sec: int = 30
    drifting_strength: float = 0.08
    drifting_concentration: float = 200.0
    drifting_floor: float = 1e-3
    mode_preference_scale: float = 1.8
    coupling_preference_scale: float = 2.4
    path_noise_std: float = 0.15
    eps: float = 1e-9


@dataclass
class WorkloadConfig:
    mode: str
    baseline_rps: float = 6.0
    burst_rps: float = 35.0
    burst_start_sec: int = 90
    burst_duration_sec: int = 90
    rate_multiplier: float = 0.0001
    invokes_cdf_file: str = "data/real-world-emulation/CDFs/invokesCDF.csv"
    cvs_cdf_file: str = "data/real-world-emulation/CDFs/CVs.csv"
    realworld_seed_offset: int = 303
    min_iat_ms: float = 1.0


@dataclass
class RuntimeConfig:
    node_base_latency_ms: int = 50
    node_latency_jitter_ms: int = 20
    cold_start_ms_min: int = 1200
    cold_start_ms_max: int = 3200
    request_timeout_ms: int = 5000
    max_concurrency_per_instance: int = 1
    data_transfer_ms_min: int = 5
    data_transfer_ms_max: int = 15
    frame_tick_ms: int = 1
    function_profile_mode: str = "cpu_intensive_random"
    function_profiles_file: str = "data/raw/function_profiles/function_profiles.csv"
    function_profile_seed_offset: int = 202
    compute_mb_per_sec_per_1000mcpu: float = 180.0
    function_cpu_request_mcpu_min: int = 700
    function_cpu_request_mcpu_max: int = 2600
    function_memory_mb_min: int = 256
    function_memory_mb_max: int = 3072
    function_output_data_mb_min: float = 5.0
    function_output_data_mb_max: float = 35.0
    function_compute_data_mb_min: float = 40.0
    function_compute_data_mb_max: float = 320.0
    function_cold_start_ms_min: int = 400
    function_cold_start_ms_max: int = 1600
    cold_start_ratio_floor: float = 1.2


@dataclass
class PhysicalNodesConfig:
    count: int = 12
    max_containers_per_node: int = 8
    idle_ttl_sec: int = 20
    same_node_transfer_ms_min: int = 1
    same_node_transfer_ms_max: int = 3
    cross_node_transfer_ms_min: int = 8
    cross_node_transfer_ms_max: int = 20
    cpu_total_mcpu_per_node: int = 32000
    mem_total_mb_per_node: int = 131072
    bandwidth_mode: str = "random_uniform"
    bandwidth_mbps_min: float = 1000.0
    bandwidth_mbps_max: float = 25000.0
    bandwidth_seed_offset: int = 101
    bandwidth_matrix_file: str = "data/raw/network/node_bandwidth_matrix.csv"


@dataclass
class SchedulerConfig:
    type: str


@dataclass
class AutoscalerConfig:
    type: str = "kpa_v1"
    target_utilization: float = 0.7
    sync_period_sec: int = 15
    scale_down_stabilization_sec: int = 60
    min_replicas: int = 0
    max_replicas_per_node: int = 20
    kpa_target_concurrency: float = 1.0
    kpa_stable_window_sec: int = 60
    kpa_panic_window_sec: int = 6
    kpa_panic_threshold: float = 2.0
    kpa_use_target_utilization: bool = True
    kpa_panic_min_hold_sec: int = 6
    kpa_panic_exit_streak_sec: int = 60
    kpa_max_scale_up_step: int = 0
    lass_latency_target_ms: float = 7000.0
    lass_load_window_sec: int = 10
    lass_speed_ewma_alpha: float = 0.2
    lass_min_speed_req_per_sec: float = 0.05
    lass_min_samples: int = 20
    lass_desired_scale: float = 1.0
    lass_min_desired_when_active: int = 0
    lass_topk_ratio: float = 1.0
    lass_low_load_boost: float = 1.0
    lass_high_load_dampen: float = 1.0
    lass_low_avg_load_threshold: float = 0.0
    lass_high_avg_load_threshold: float = 1e9
    lass_inflight_credit: float = 0.0
    lass_scale_cooldown_sec: int = 0
    lass_max_desired_cap: int = 0
    lass_max_create_per_tick: int = 0
    lass_low_load_max_create_per_tick: int = 0
    lass_low_load_max_create_threshold: float = 0.0
    hptd_time_granularity_sec: int = 1
    hptd_wcall_t: int = 20
    hptd_whistory_t: int = 50
    hptd_wchange_t: int = 10
    hptd_alpha: float = 0.134206
    hptd_beta: float = 1.911787
    hptd_mu_floor: float = 0.01
    hptd_std_floor: float = 0.02
    hptd_temp_floor: float = 1e-6
    hptd_scale_max_step: int = 8
    rl_time_granularity_sec: int = 1
    rl_wcall_t: int = 20
    rl_whistory_t: int = 50
    rl_wchange_t: int = 10
    rl_learning_rate: float = 0.5
    rl_discount_factor: float = 0.9
    rl_epsilon_init: float = 0.8
    rl_epsilon_decay: float = 0.9
    rl_epsilon_min: float = 0.1
    rl_step_size: int = 1
    rl_util_threshold: float = 0.8
    rl_reward_tolerance: float = 0.05
    rl_scalability_alpha: float = 0.15
    rl_inhibit_token_max: int = 3
    rl_slo_p95_ms: float = 4000.0
    rl_failure_window_sec: int = 300
    rl_failure_rate_threshold: float = 0.05
    rl_sla_penalty_weight: float = 0.35
    rl_profile_warmup_sec: int = 60
    rl_profile_ewma_alpha: float = 0.2
    rl_alpha_scalability_min: float = 0.0
    rl_alpha_scalability_max: float = 0.5
    rl_tp_floor: float = 0.05
    hist_sync_period_sec: int = 15
    hist_window_sec: int = 120
    hist_quantile: float = 0.9
    hist_keepalive_idle_sec: int = 30
    hist_keepalive_min_replicas: int = 1
    hist_prewarm_buffer: int = 1
    hist_bin_minutes: int = 1
    hist_range_minutes: int = 240
    hist_head_percentile: float = 0.05
    hist_tail_percentile: float = 0.99
    hist_margin_ratio: float = 0.10
    hist_min_samples: int = 20
    hist_cv_threshold: float = 0.25
    hist_oob_ratio_threshold: float = 0.30
    hist_history_retention_sec: int = 21600
    hist_forecast_margin_ratio: float = 0.15
    hist_forecast_alpha: float = 0.35
    hist_forecast_min_samples: int = 8
    xanadu_depth: int = 1
    xanadu_ewma_alpha: float = 0.2
    xanadu_seed_offset: int = 1313
    oracle_window_steps: int = 2
    hpwp_lmax_low: int = 2
    hpwp_lmax_high: int = 4
    hpwp_beta_hi: float = 80.0
    hpwp_beta_lo: float = 12.0
    hpwp_alpha_exp: float = 0.8
    hpwp_alpha_stable: float = 0.15
    hpwp_sched_eta_exec: float = 0.5
    hpwp_sched_min_sec: int = 1
    hpwp_sched_max_sec: int = 15
    hpwp_horizon_alpha: float = 2.0
    hpwp_rho_mass: float = 0.5
    hpwp_tau_p: float = 0.02
    hpwp_urgency_epsilon_ms: float = 5.0
    hpwp_phase_window_k: int = 30
    hpwp_phase_n_min: int = 50
    hpwp_phase_var_threshold: float = 0.015
    hpwp_drift_short_k: int = 10
    hpwp_drift_long_k: int = 60
    hpwp_drift_delta_mr: float = 0.08
    hpwp_drift_tau_mr: float = 0.35
    hpwp_drift_branch_tau: float = 0.4
    hpwp_forget_gamma: float = 0.35
    hpwp_default_exec_ms: float = 40.0
    hpwp_default_cold_ms: float = 1500.0
    hpwp_default_trans_ms: float = 8.0
    hpwp_seed_offset: int = 2027
    dbw_horizon_boost: float = 1.0
    dbw_desired_scale: float = 1.0
    dbw_desired_ewma_alpha: float = 0.35
    dbw_guard_buffer_min: int = 1
    dbw_down_margin: int = 1
    dbw_min_idle_age_sec: int = 5
    dbw_down_cooldown_sec: int = 6
    dbw_max_down_ratio: float = 0.15
    dbw_osc_window_sec: int = 30
    dbw_osc_trigger_count: int = 3
    kraken_misallocation_ratio: float = 0.0
    kraken_desired_scale: float = 1.0
    kraken_max_prewarm_per_tick: int = 0
    kraken_min_desired_units: float = 0.0
    kraken_uniform_mix: float = 0.0
    kraken_mid_pressure_threshold: float = 55.0
    kraken_high_pressure_threshold: float = 75.0
    kraken_mid_pressure_scale: float = 0.85
    kraken_high_pressure_scale: float = 0.7
    kraken_reconcile_stride: int = 1
    kraken_active_mid_threshold: int = 2
    kraken_active_high_threshold: int = 3
    kraken_active_mid_scale: float = 0.75
    kraken_active_high_scale: float = 0.55


@dataclass
class CapacityConfig:
    max_total_instances: int


@dataclass
class OutputConfig:
    runs_dir: str


@dataclass
class SimulationConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    dag_policy: DagPolicyConfig
    workload: WorkloadConfig
    runtime: RuntimeConfig
    physical_nodes: PhysicalNodesConfig
    scheduler: SchedulerConfig
    autoscaler: AutoscalerConfig
    capacity: CapacityConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_config(raw: dict[str, Any]) -> SimulationConfig:
    autoscaler_raw = raw.get(
        "autoscaler",
        {
            "type": "kpa_v1",
            "target_utilization": 0.7,
            "sync_period_sec": 15,
            "scale_down_stabilization_sec": 60,
            "min_replicas": 0,
            "max_replicas_per_node": 20,
            "kpa_target_concurrency": 1.0,
            "kpa_stable_window_sec": 60,
            "kpa_panic_window_sec": 6,
            "kpa_panic_threshold": 2.0,
            "kpa_use_target_utilization": True,
            "kpa_panic_min_hold_sec": 6,
            "kpa_panic_exit_streak_sec": 60,
            "kpa_max_scale_up_step": 0,
            "lass_latency_target_ms": 7000.0,
            "lass_load_window_sec": 10,
            "lass_speed_ewma_alpha": 0.2,
            "lass_min_speed_req_per_sec": 0.05,
            "lass_min_samples": 20,
            "lass_desired_scale": 1.0,
            "lass_min_desired_when_active": 0,
            "lass_topk_ratio": 1.0,
            "lass_low_load_boost": 1.0,
            "lass_high_load_dampen": 1.0,
            "lass_low_avg_load_threshold": 0.0,
            "lass_high_avg_load_threshold": 1e9,
            "lass_inflight_credit": 0.0,
            "lass_scale_cooldown_sec": 0,
            "lass_max_desired_cap": 0,
            "lass_max_create_per_tick": 0,
            "lass_low_load_max_create_per_tick": 0,
            "lass_low_load_max_create_threshold": 0.0,
            "hptd_time_granularity_sec": 1,
            "hptd_wcall_t": 20,
            "hptd_whistory_t": 50,
            "hptd_wchange_t": 10,
            "hptd_alpha": 0.134206,
            "hptd_beta": 1.911787,
            "hptd_mu_floor": 0.01,
            "hptd_std_floor": 0.02,
            "hptd_temp_floor": 1e-6,
            "hptd_scale_max_step": 8,
            "rl_time_granularity_sec": 1,
            "rl_wcall_t": 20,
            "rl_whistory_t": 50,
            "rl_wchange_t": 10,
            "rl_learning_rate": 0.5,
            "rl_discount_factor": 0.9,
            "rl_epsilon_init": 0.8,
            "rl_epsilon_decay": 0.9,
            "rl_epsilon_min": 0.1,
            "rl_step_size": 1,
            "rl_util_threshold": 0.8,
            "rl_reward_tolerance": 0.05,
            "rl_scalability_alpha": 0.15,
            "rl_inhibit_token_max": 3,
            "rl_slo_p95_ms": 4000.0,
            "rl_failure_window_sec": 300,
            "rl_failure_rate_threshold": 0.05,
            "rl_sla_penalty_weight": 0.35,
            "rl_profile_warmup_sec": 60,
            "rl_profile_ewma_alpha": 0.2,
            "rl_alpha_scalability_min": 0.0,
            "rl_alpha_scalability_max": 0.5,
            "rl_tp_floor": 0.05,
            "hist_sync_period_sec": 15,
            "hist_window_sec": 120,
            "hist_quantile": 0.9,
            "hist_keepalive_idle_sec": 30,
            "hist_keepalive_min_replicas": 1,
            "hist_prewarm_buffer": 1,
            "hist_bin_minutes": 1,
            "hist_range_minutes": 240,
            "hist_head_percentile": 0.05,
            "hist_tail_percentile": 0.99,
            "hist_margin_ratio": 0.10,
            "hist_min_samples": 20,
            "hist_cv_threshold": 0.25,
            "hist_oob_ratio_threshold": 0.30,
            "hist_history_retention_sec": 21600,
            "hist_forecast_margin_ratio": 0.15,
            "hist_forecast_alpha": 0.35,
            "hist_forecast_min_samples": 8,
            "xanadu_depth": 1,
            "xanadu_ewma_alpha": 0.2,
            "xanadu_seed_offset": 1313,
            "oracle_window_steps": 2,
            "hpwp_lmax_low": 2,
            "hpwp_lmax_high": 4,
            "hpwp_beta_hi": 80.0,
            "hpwp_beta_lo": 12.0,
            "hpwp_alpha_exp": 0.8,
            "hpwp_alpha_stable": 0.15,
            "hpwp_sched_eta_exec": 0.5,
            "hpwp_sched_min_sec": 1,
            "hpwp_sched_max_sec": 15,
            "hpwp_horizon_alpha": 2.0,
            "hpwp_rho_mass": 0.5,
            "hpwp_tau_p": 0.02,
            "hpwp_urgency_epsilon_ms": 5.0,
            "hpwp_phase_window_k": 30,
            "hpwp_phase_n_min": 50,
            "hpwp_phase_var_threshold": 0.015,
            "hpwp_drift_short_k": 10,
            "hpwp_drift_long_k": 60,
            "hpwp_drift_delta_mr": 0.08,
            "hpwp_drift_tau_mr": 0.35,
            "hpwp_drift_branch_tau": 0.4,
            "hpwp_forget_gamma": 0.35,
            "hpwp_default_exec_ms": 40.0,
            "hpwp_default_cold_ms": 1500.0,
            "hpwp_default_trans_ms": 8.0,
            "hpwp_seed_offset": 2027,
            "dbw_horizon_boost": 1.0,
            "dbw_desired_scale": 1.0,
            "dbw_desired_ewma_alpha": 0.35,
            "dbw_guard_buffer_min": 1,
            "dbw_down_margin": 1,
            "dbw_min_idle_age_sec": 5,
            "dbw_down_cooldown_sec": 6,
            "dbw_max_down_ratio": 0.15,
            "dbw_osc_window_sec": 30,
            "dbw_osc_trigger_count": 3,
            "kraken_misallocation_ratio": 0.0,
            "kraken_desired_scale": 1.0,
            "kraken_max_prewarm_per_tick": 0,
            "kraken_min_desired_units": 0.0,
            "kraken_uniform_mix": 0.0,
            "kraken_mid_pressure_threshold": 55.0,
            "kraken_high_pressure_threshold": 75.0,
            "kraken_mid_pressure_scale": 0.85,
            "kraken_high_pressure_scale": 0.7,
            "kraken_reconcile_stride": 1,
            "kraken_active_mid_threshold": 2,
            "kraken_active_high_threshold": 3,
            "kraken_active_mid_scale": 0.75,
            "kraken_active_high_scale": 0.55,
        },
    )
    autoscaler_raw.setdefault("xanadu_depth", 1)
    autoscaler_raw.setdefault("xanadu_ewma_alpha", 0.2)
    autoscaler_raw.setdefault("xanadu_seed_offset", 1313)
    autoscaler_raw.setdefault("oracle_window_steps", 2)
    autoscaler_raw.setdefault("kpa_target_concurrency", 1.0)
    autoscaler_raw.setdefault("kpa_stable_window_sec", 60)
    autoscaler_raw.setdefault("kpa_panic_window_sec", 6)
    autoscaler_raw.setdefault("kpa_panic_threshold", 2.0)
    autoscaler_raw.setdefault("kpa_use_target_utilization", True)
    autoscaler_raw.setdefault("kpa_panic_min_hold_sec", 6)
    autoscaler_raw.setdefault("kpa_panic_exit_streak_sec", 60)
    autoscaler_raw.setdefault("kpa_max_scale_up_step", 0)
    autoscaler_raw.setdefault("lass_latency_target_ms", 7000.0)
    autoscaler_raw.setdefault("lass_load_window_sec", 10)
    autoscaler_raw.setdefault("lass_speed_ewma_alpha", 0.2)
    autoscaler_raw.setdefault("lass_min_speed_req_per_sec", 0.05)
    autoscaler_raw.setdefault("lass_min_samples", 20)
    autoscaler_raw.setdefault("lass_desired_scale", 1.0)
    autoscaler_raw.setdefault("lass_min_desired_when_active", 0)
    autoscaler_raw.setdefault("lass_topk_ratio", 1.0)
    autoscaler_raw.setdefault("lass_low_load_boost", 1.0)
    autoscaler_raw.setdefault("lass_high_load_dampen", 1.0)
    autoscaler_raw.setdefault("lass_low_avg_load_threshold", 0.0)
    autoscaler_raw.setdefault("lass_high_avg_load_threshold", 1e9)
    autoscaler_raw.setdefault("lass_inflight_credit", 0.0)
    autoscaler_raw.setdefault("lass_scale_cooldown_sec", 0)
    autoscaler_raw.setdefault("lass_max_desired_cap", 0)
    autoscaler_raw.setdefault("lass_max_create_per_tick", 0)
    autoscaler_raw.setdefault("lass_low_load_max_create_per_tick", 0)
    autoscaler_raw.setdefault("lass_low_load_max_create_threshold", 0.0)
    autoscaler_raw.setdefault("hptd_time_granularity_sec", 1)
    autoscaler_raw.setdefault("hptd_wcall_t", 20)
    autoscaler_raw.setdefault("hptd_whistory_t", 50)
    autoscaler_raw.setdefault("hptd_wchange_t", 10)
    autoscaler_raw.setdefault("hptd_alpha", 0.134206)
    autoscaler_raw.setdefault("hptd_beta", 1.911787)
    autoscaler_raw.setdefault("hptd_mu_floor", 0.01)
    autoscaler_raw.setdefault("hptd_std_floor", 0.02)
    autoscaler_raw.setdefault("hptd_temp_floor", 1e-6)
    autoscaler_raw.setdefault("hptd_scale_max_step", 8)
    # Backward compatibility: ignore removed HPTD enhancement knobs that were never wired.
    autoscaler_raw.pop("hptd_trigger_z", None)
    autoscaler_raw.pop("hptd_confirm_ticks", None)
    autoscaler_raw.pop("hptd_reuse_low", None)
    autoscaler_raw.pop("hptd_reuse_dampen", None)
    autoscaler_raw.pop("hptd_topk_ratio", None)
    autoscaler_raw.pop("hptd_active_boost_scale", None)
    autoscaler_raw.pop("hptd_active_boost_min", None)
    autoscaler_raw.pop("hptd_successor_boost_ratio", None)
    autoscaler_raw.pop("hptd_request_assist_depth", None)
    autoscaler_raw.pop("hptd_request_assist_scale", None)
    autoscaler_raw.setdefault("rl_time_granularity_sec", 1)
    autoscaler_raw.setdefault("rl_wcall_t", 20)
    autoscaler_raw.setdefault("rl_whistory_t", 50)
    autoscaler_raw.setdefault("rl_wchange_t", 10)
    autoscaler_raw.setdefault("rl_learning_rate", 0.5)
    autoscaler_raw.setdefault("rl_discount_factor", 0.9)
    autoscaler_raw.setdefault("rl_epsilon_init", 0.8)
    autoscaler_raw.setdefault("rl_epsilon_decay", 0.9)
    autoscaler_raw.setdefault("rl_epsilon_min", 0.1)
    autoscaler_raw.setdefault("rl_step_size", 1)
    autoscaler_raw.setdefault("rl_util_threshold", 0.8)
    autoscaler_raw.setdefault("rl_reward_tolerance", 0.05)
    autoscaler_raw.setdefault("rl_scalability_alpha", 0.15)
    autoscaler_raw.setdefault("rl_inhibit_token_max", 3)
    autoscaler_raw.setdefault("rl_slo_p95_ms", 4000.0)
    autoscaler_raw.setdefault("rl_failure_window_sec", 300)
    autoscaler_raw.setdefault("rl_failure_rate_threshold", 0.05)
    autoscaler_raw.setdefault("rl_sla_penalty_weight", 0.35)
    autoscaler_raw.setdefault("rl_profile_warmup_sec", 60)
    autoscaler_raw.setdefault("rl_profile_ewma_alpha", 0.2)
    autoscaler_raw.setdefault("rl_alpha_scalability_min", 0.0)
    autoscaler_raw.setdefault("rl_alpha_scalability_max", 0.5)
    autoscaler_raw.setdefault("rl_tp_floor", 0.05)
    autoscaler_raw.setdefault("hist_sync_period_sec", autoscaler_raw.get("sync_period_sec", 15))
    autoscaler_raw.setdefault("hist_window_sec", 120)
    autoscaler_raw.setdefault("hist_quantile", 0.9)
    autoscaler_raw.setdefault("hist_keepalive_idle_sec", 30)
    autoscaler_raw.setdefault("hist_keepalive_min_replicas", 1)
    autoscaler_raw.setdefault("hist_prewarm_buffer", 1)
    autoscaler_raw.setdefault("hist_bin_minutes", 1)
    autoscaler_raw.setdefault("hist_range_minutes", 240)
    autoscaler_raw.setdefault("hist_head_percentile", 0.05)
    autoscaler_raw.setdefault("hist_tail_percentile", autoscaler_raw.get("hist_quantile", 0.99))
    autoscaler_raw.setdefault("hist_margin_ratio", 0.10)
    autoscaler_raw.setdefault("hist_min_samples", 20)
    autoscaler_raw.setdefault("hist_cv_threshold", 0.25)
    autoscaler_raw.setdefault("hist_oob_ratio_threshold", 0.30)
    autoscaler_raw.setdefault("hist_history_retention_sec", 21600)
    autoscaler_raw.setdefault("hist_forecast_margin_ratio", 0.15)
    autoscaler_raw.setdefault("hist_forecast_alpha", 0.35)
    autoscaler_raw.setdefault("hist_forecast_min_samples", 8)
    autoscaler_raw.setdefault("hpwp_lmax_low", 2)
    autoscaler_raw.setdefault("hpwp_lmax_high", 4)
    autoscaler_raw.setdefault("hpwp_beta_hi", 80.0)
    autoscaler_raw.setdefault("hpwp_beta_lo", 12.0)
    autoscaler_raw.setdefault("hpwp_alpha_exp", 0.8)
    autoscaler_raw.setdefault("hpwp_alpha_stable", 0.15)
    autoscaler_raw.setdefault("hpwp_sched_eta_exec", 0.5)
    autoscaler_raw.setdefault("hpwp_sched_min_sec", 1)
    autoscaler_raw.setdefault("hpwp_sched_max_sec", 15)
    autoscaler_raw.setdefault("hpwp_horizon_alpha", 2.0)
    autoscaler_raw.setdefault("hpwp_rho_mass", 0.5)
    autoscaler_raw.setdefault("hpwp_tau_p", 0.02)
    autoscaler_raw.setdefault("hpwp_urgency_epsilon_ms", 5.0)
    autoscaler_raw.setdefault("hpwp_phase_window_k", 30)
    autoscaler_raw.setdefault("hpwp_phase_n_min", 50)
    autoscaler_raw.setdefault("hpwp_phase_var_threshold", 0.015)
    autoscaler_raw.setdefault("hpwp_drift_short_k", 10)
    autoscaler_raw.setdefault("hpwp_drift_long_k", 60)
    autoscaler_raw.setdefault("hpwp_drift_delta_mr", 0.08)
    autoscaler_raw.setdefault("hpwp_drift_tau_mr", 0.35)
    autoscaler_raw.setdefault("hpwp_drift_branch_tau", 0.4)
    autoscaler_raw.setdefault("hpwp_forget_gamma", 0.35)
    autoscaler_raw.setdefault("hpwp_default_exec_ms", 40.0)
    autoscaler_raw.setdefault("hpwp_default_cold_ms", 1500.0)
    autoscaler_raw.setdefault("hpwp_default_trans_ms", 8.0)
    autoscaler_raw.setdefault("hpwp_seed_offset", 2027)
    # Backward compatibility: ignore removed DBW misallocation knob if present.
    autoscaler_raw.pop("dbw_misallocation_ratio", None)
    autoscaler_raw.setdefault("dbw_horizon_boost", 1.0)
    autoscaler_raw.setdefault("dbw_desired_scale", 1.0)
    autoscaler_raw.setdefault("dbw_desired_ewma_alpha", 0.35)
    autoscaler_raw.setdefault("dbw_guard_buffer_min", 1)
    autoscaler_raw.setdefault("dbw_down_margin", 1)
    autoscaler_raw.setdefault("dbw_min_idle_age_sec", 5)
    autoscaler_raw.setdefault("dbw_down_cooldown_sec", 6)
    autoscaler_raw.setdefault("dbw_max_down_ratio", 0.15)
    autoscaler_raw.setdefault("dbw_osc_window_sec", 30)
    autoscaler_raw.setdefault("dbw_osc_trigger_count", 3)
    autoscaler_raw.setdefault("kraken_misallocation_ratio", 0.0)
    autoscaler_raw.setdefault("kraken_desired_scale", 1.0)
    autoscaler_raw.setdefault("kraken_max_prewarm_per_tick", 0)
    autoscaler_raw.setdefault("kraken_min_desired_units", 0.0)
    autoscaler_raw.setdefault("kraken_uniform_mix", 0.0)
    autoscaler_raw.setdefault("kraken_mid_pressure_threshold", 55.0)
    autoscaler_raw.setdefault("kraken_high_pressure_threshold", 75.0)
    autoscaler_raw.setdefault("kraken_mid_pressure_scale", 0.85)
    autoscaler_raw.setdefault("kraken_high_pressure_scale", 0.7)
    autoscaler_raw.setdefault("kraken_reconcile_stride", 1)
    autoscaler_raw.setdefault("kraken_active_mid_threshold", 2)
    autoscaler_raw.setdefault("kraken_active_high_threshold", 3)
    autoscaler_raw.setdefault("kraken_active_mid_scale", 0.75)
    autoscaler_raw.setdefault("kraken_active_high_scale", 0.55)
    autoscaler_type = str(autoscaler_raw.get("type", "")).lower()
    if autoscaler_type == "hpa_v1":
        warnings.warn(
            "autoscaler.type=hpa_v1 is deprecated and mapped to kpa_v1",
            RuntimeWarning,
            stacklevel=2,
        )
        autoscaler_raw["type"] = "kpa_v1"
        autoscaler_type = "kpa_v1"
    if ("autoscaler" in raw) and str(autoscaler_raw.get("type", "")).lower() not in {
        "xanadu_v1",
        "xanadu_opt_v1",
        "oracle_future_v1",
        "depth_breadth_v1",
        "kraken_vomm_v1",
        "hpwp_v1",
        "kpa_v1",
        "lass_v1",
        "hptd_v1",
        "rl_q_v1",
        "hist_keepalive_prewarm_v1",
        "no_autoscale_v1",
    }:
        warnings.warn(
            (
                "autoscaler type is running in compatibility mode; set autoscaler.type to "
                "hpwp_v1/xanadu_v1/xanadu_opt_v1/oracle_future_v1/depth_breadth_v1/kraken_vomm_v1/kpa_v1/lass_v1/hptd_v1/rl_q_v1/hist_keepalive_prewarm_v1/no_autoscale_v1"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    physical_nodes_raw = raw.get(
        "physical_nodes",
        {
            "count": 12,
            "max_containers_per_node": 8,
            "idle_ttl_sec": 20,
            "same_node_transfer_ms_min": 1,
            "same_node_transfer_ms_max": 3,
            "cross_node_transfer_ms_min": 8,
            "cross_node_transfer_ms_max": 20,
        },
    )
    runtime_raw = dict(raw["runtime"])
    if runtime_raw.get("max_concurrency_per_instance", 1) != 1:
        warnings.warn(
            "max_concurrency_per_instance is ignored in no-queue physical-node mode; effective behavior is single-task containers",
            RuntimeWarning,
            stacklevel=2,
        )
    if "function_cold_start_ms_min" not in runtime_raw:
        runtime_raw["function_cold_start_ms_min"] = runtime_raw.get("cold_start_ms_min", 1200)
    if "function_cold_start_ms_max" not in runtime_raw:
        runtime_raw["function_cold_start_ms_max"] = runtime_raw.get("cold_start_ms_max", 3200)
    if "frame_tick_ms" in runtime_raw and int(runtime_raw["frame_tick_ms"]) <= 0:
        warnings.warn(
            "frame_tick_ms must be > 0; forcing to 1",
            RuntimeWarning,
            stacklevel=2,
        )
        runtime_raw["frame_tick_ms"] = 1
    dag_policy_input = dict(raw["dag_policy"])
    dag_policy_raw = dict(dag_policy_input)
    if "context_regime" not in dag_policy_raw:
        dag_policy_raw["context_regime"] = "drifting"
    if "context_drifting_enabled" in dag_policy_raw:
        drifting_enabled = bool(dag_policy_raw.get("context_drifting_enabled"))
        dag_policy_raw["context_drifting_enabled"] = drifting_enabled
        dag_policy_raw["context_regime"] = "drifting" if drifting_enabled else "fixed"
    else:
        dag_policy_raw["context_drifting_enabled"] = (
            str(dag_policy_raw.get("context_regime", "fixed")).lower() == "drifting"
        )
    if (
        str(dag_policy_raw.get("path_rule", "")).lower() == "mode_prefix_coupled_v1"
        and (
            ("session_continue_prob" in dag_policy_input)
            or ("context_alpha" in dag_policy_input)
        )
    ):
        warnings.warn(
            "dag_policy.session_continue_prob/context_alpha are ignored when path_rule=mode_prefix_coupled_v1",
            RuntimeWarning,
            stacklevel=2,
        )
    regime = str(dag_policy_raw.get("context_regime", "fixed")).lower()
    if regime not in {"fixed", "drifting"}:
        warnings.warn(
            f"unsupported dag_policy.context_regime={dag_policy_raw.get('context_regime')}, forcing to 'fixed'",
            RuntimeWarning,
            stacklevel=2,
        )
        regime = "fixed"
    dag_policy_raw["context_regime"] = regime
    dag_policy_raw["context_drifting_enabled"] = (regime == "drifting")
    dataset_input = dict(raw["dataset"])
    dataset_raw = dict(dataset_input)
    dataset_raw.setdefault("callgraph_file", "data/raw/unused_callgraph.csv")
    dataset_raw.setdefault("qps_file", "data/raw/unused_qps.csv")
    dataset_raw.setdefault("sample_um_count", 8)
    dataset_raw.setdefault("checksum_required", False)
    dataset_raw.setdefault("dag_tasks_file", "data/raw/filtered_tasks.csv")
    dataset_raw.setdefault("dag_top_k", 20)
    dataset_raw.setdefault("dag_selection_mode", "random_unique")
    dataset_raw.setdefault("dag_selection_seed", 42)
    dataset_raw.setdefault("dag_max_nodes", None)
    dataset_raw.setdefault("dag_max_edges", None)
    dataset_raw.setdefault("dag_max_longest_path", None)
    dataset_raw.setdefault("dag_max_splits", None)
    workload_raw = dict(raw["workload"])
    workload_raw.setdefault("invokes_cdf_file", "data/real-world-emulation/CDFs/invokesCDF.csv")
    workload_raw.setdefault("cvs_cdf_file", "data/real-world-emulation/CDFs/CVs.csv")
    workload_raw.setdefault("realworld_seed_offset", 303)
    workload_raw.setdefault("min_iat_ms", 1.0)
    if (
        str(workload_raw.get("mode", "")).lower() == "replay"
        and (("callgraph_file" in dataset_input) or ("qps_file" in dataset_input))
    ):
        warnings.warn(
            "workload.mode=replay now uses filtered_tasks + Real-World CDFs; callgraph_file/qps_file are ignored for replay generation",
            RuntimeWarning,
            stacklevel=2,
        )
    return SimulationConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        dataset=DatasetConfig(**dataset_raw),
        dag_policy=DagPolicyConfig(**dag_policy_raw),
        workload=WorkloadConfig(**workload_raw),
        runtime=RuntimeConfig(**runtime_raw),
        physical_nodes=PhysicalNodesConfig(**physical_nodes_raw),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        autoscaler=AutoscalerConfig(**autoscaler_raw),
        capacity=CapacityConfig(**raw["capacity"]),
        output=OutputConfig(**raw["output"]),
    )


def load_config(path: str | Path) -> SimulationConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _build_config(raw)


def dump_config(config: SimulationConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
