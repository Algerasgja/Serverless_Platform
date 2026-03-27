from __future__ import annotations

import random
import warnings

from simulator.config import AutoscalerConfig, SchedulerConfig
from simulator.strategies.autoscaler import (
    DepthBreadthAutoscaler,
    HistogramKeepalivePrewarmAutoscaler,
    HptdAutoscaler,
    HpwpAutoscaler,
    KrakenVomMAutoscaler,
    KpaAutoscaler,
    LassAutoscaler,
    NoOpAutoscaler,
    OracleFutureAutoscaler,
    RlQAutoscaler,
    XanaduAutoscaler,
    XanaduOptimizedAutoscaler,
)
from simulator.strategies.autoscaler.base import AutoscalerStrategy
from simulator.strategies.scheduler import LeastLoadScheduler
from simulator.strategies.scheduler.base import SchedulerStrategy
from simulator.types import DagTemplate


def build_scheduler(config: SchedulerConfig, rng: random.Random) -> SchedulerStrategy:
    if config.type == "least_load":
        return LeastLoadScheduler(rng)
    raise ValueError(f"unsupported scheduler type: {config.type}")


def build_autoscaler(
    config: AutoscalerConfig,
    *,
    templates: dict[str, DagTemplate],
) -> AutoscalerStrategy:
    resolved_type = str(config.type).lower()
    if resolved_type == "hpa_v1":
        warnings.warn(
            "autoscaler.type=hpa_v1 is deprecated and mapped to kpa_v1",
            RuntimeWarning,
            stacklevel=2,
        )
        resolved_type = "kpa_v1"

    if resolved_type == "hpwp_v1":
        return HpwpAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            lmax_low=config.hpwp_lmax_low,
            lmax_high=config.hpwp_lmax_high,
            beta_hi=config.hpwp_beta_hi,
            beta_lo=config.hpwp_beta_lo,
            alpha_exp=config.hpwp_alpha_exp,
            alpha_stable=config.hpwp_alpha_stable,
            sched_eta_exec=config.hpwp_sched_eta_exec,
            sched_min_sec=config.hpwp_sched_min_sec,
            sched_max_sec=config.hpwp_sched_max_sec,
            horizon_alpha=config.hpwp_horizon_alpha,
            rho_mass=config.hpwp_rho_mass,
            tau_p=config.hpwp_tau_p,
            urgency_epsilon_ms=config.hpwp_urgency_epsilon_ms,
            phase_window_k=config.hpwp_phase_window_k,
            phase_n_min=config.hpwp_phase_n_min,
            phase_var_threshold=config.hpwp_phase_var_threshold,
            drift_short_k=config.hpwp_drift_short_k,
            drift_long_k=config.hpwp_drift_long_k,
            drift_delta_mr=config.hpwp_drift_delta_mr,
            drift_tau_mr=config.hpwp_drift_tau_mr,
            drift_branch_tau=config.hpwp_drift_branch_tau,
            forget_gamma=config.hpwp_forget_gamma,
            default_exec_ms=config.hpwp_default_exec_ms,
            default_cold_ms=config.hpwp_default_cold_ms,
            default_trans_ms=config.hpwp_default_trans_ms,
            seed_offset=config.hpwp_seed_offset,
        )
    if resolved_type == "xanadu_v1":
        return XanaduAutoscaler(
            templates=templates,
            depth=config.xanadu_depth,
            ewma_alpha=config.xanadu_ewma_alpha,
            sync_period_sec=config.sync_period_sec,
        )
    if resolved_type == "xanadu_opt_v1":
        return XanaduOptimizedAutoscaler(
            templates=templates,
            depth=config.xanadu_depth,
            ewma_alpha=config.xanadu_ewma_alpha,
            sync_period_sec=config.sync_period_sec,
        )
    if resolved_type == "oracle_future_v1":
        return OracleFutureAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            window_steps=config.oracle_window_steps,
        )
    if resolved_type == "depth_breadth_v1":
        return DepthBreadthAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            sched_eta_exec=config.hpwp_sched_eta_exec,
            sched_min_sec=config.hpwp_sched_min_sec,
            sched_max_sec=config.hpwp_sched_max_sec,
            horizon_alpha=config.hpwp_horizon_alpha,
            default_exec_ms=config.hpwp_default_exec_ms,
            default_trans_ms=config.hpwp_default_trans_ms,
            horizon_boost=config.dbw_horizon_boost,
            desired_scale=config.dbw_desired_scale,
            desired_ewma_alpha=config.dbw_desired_ewma_alpha,
            guard_buffer_min=config.dbw_guard_buffer_min,
            down_margin=config.dbw_down_margin,
            min_idle_age_sec=config.dbw_min_idle_age_sec,
            down_cooldown_sec=config.dbw_down_cooldown_sec,
            max_down_ratio=config.dbw_max_down_ratio,
            osc_window_sec=config.dbw_osc_window_sec,
            osc_trigger_count=config.dbw_osc_trigger_count,
        )
    if resolved_type == "kraken_vomm_v1":
        return KrakenVomMAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            sched_eta_exec=config.hpwp_sched_eta_exec,
            sched_min_sec=config.hpwp_sched_min_sec,
            sched_max_sec=config.hpwp_sched_max_sec,
            horizon_alpha=config.hpwp_horizon_alpha,
            default_exec_ms=config.hpwp_default_exec_ms,
            default_trans_ms=config.hpwp_default_trans_ms,
            misallocation_ratio=config.kraken_misallocation_ratio,
            desired_scale=config.kraken_desired_scale,
            max_prewarm_per_tick=config.kraken_max_prewarm_per_tick,
            min_desired_units=config.kraken_min_desired_units,
            uniform_mix=config.kraken_uniform_mix,
            mid_pressure_threshold=config.kraken_mid_pressure_threshold,
            high_pressure_threshold=config.kraken_high_pressure_threshold,
            mid_pressure_scale=config.kraken_mid_pressure_scale,
            high_pressure_scale=config.kraken_high_pressure_scale,
            reconcile_stride=config.kraken_reconcile_stride,
            active_mid_threshold=config.kraken_active_mid_threshold,
            active_high_threshold=config.kraken_active_high_threshold,
            active_mid_scale=config.kraken_active_mid_scale,
            active_high_scale=config.kraken_active_high_scale,
        )
    if resolved_type == "kpa_v1":
        return KpaAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            target_concurrency=config.kpa_target_concurrency,
            target_utilization=config.target_utilization,
            use_target_utilization=config.kpa_use_target_utilization,
            stable_window_sec=config.kpa_stable_window_sec,
            panic_window_sec=config.kpa_panic_window_sec,
            panic_threshold=config.kpa_panic_threshold,
            panic_min_hold_sec=config.kpa_panic_min_hold_sec,
            panic_exit_streak_sec=config.kpa_panic_exit_streak_sec,
            max_scale_up_step=config.kpa_max_scale_up_step,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas_per_node,
        )
    if resolved_type == "lass_v1":
        return LassAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            latency_target_ms=config.lass_latency_target_ms,
            load_window_sec=config.lass_load_window_sec,
            speed_ewma_alpha=config.lass_speed_ewma_alpha,
            min_speed_req_per_sec=config.lass_min_speed_req_per_sec,
            min_samples=config.lass_min_samples,
            default_exec_ms=config.hpwp_default_exec_ms,
            desired_scale=config.lass_desired_scale,
            min_desired_when_active=config.lass_min_desired_when_active,
            topk_ratio=config.lass_topk_ratio,
            low_load_boost=config.lass_low_load_boost,
            high_load_dampen=config.lass_high_load_dampen,
            low_avg_load_threshold=config.lass_low_avg_load_threshold,
            high_avg_load_threshold=config.lass_high_avg_load_threshold,
            inflight_credit=config.lass_inflight_credit,
            scale_cooldown_sec=config.lass_scale_cooldown_sec,
            max_desired_cap=config.lass_max_desired_cap,
            max_create_per_tick=config.lass_max_create_per_tick,
            low_load_max_create_per_tick=config.lass_low_load_max_create_per_tick,
            low_load_max_create_threshold=config.lass_low_load_max_create_threshold,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas_per_node,
        )
    if resolved_type == "hptd_v1":
        return HptdAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            time_granularity_sec=config.hptd_time_granularity_sec,
            wcall_t=config.hptd_wcall_t,
            whistory_t=config.hptd_whistory_t,
            wchange_t=config.hptd_wchange_t,
            alpha=config.hptd_alpha,
            beta=config.hptd_beta,
            mu_floor=config.hptd_mu_floor,
            std_floor=config.hptd_std_floor,
            temp_floor=config.hptd_temp_floor,
            scale_max_step=config.hptd_scale_max_step,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas_per_node,
        )
    if resolved_type == "rl_q_v1":
        return RlQAutoscaler(
            templates=templates,
            sync_period_sec=config.sync_period_sec,
            time_granularity_sec=config.rl_time_granularity_sec,
            wcall_t=config.rl_wcall_t,
            whistory_t=config.rl_whistory_t,
            wchange_t=config.rl_wchange_t,
            learning_rate=config.rl_learning_rate,
            discount_factor=config.rl_discount_factor,
            epsilon_init=config.rl_epsilon_init,
            epsilon_decay=config.rl_epsilon_decay,
            epsilon_min=config.rl_epsilon_min,
            step_size=config.rl_step_size,
            util_threshold=config.rl_util_threshold,
            reward_tolerance=config.rl_reward_tolerance,
            scalability_alpha=config.rl_scalability_alpha,
            inhibit_token_max=config.rl_inhibit_token_max,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas_per_node,
            slo_p95_ms=config.rl_slo_p95_ms,
            failure_window_sec=config.rl_failure_window_sec,
            failure_rate_threshold=config.rl_failure_rate_threshold,
            sla_penalty_weight=config.rl_sla_penalty_weight,
            profile_warmup_sec=config.rl_profile_warmup_sec,
            profile_ewma_alpha=config.rl_profile_ewma_alpha,
            alpha_scalability_min=config.rl_alpha_scalability_min,
            alpha_scalability_max=config.rl_alpha_scalability_max,
            tp_floor=config.rl_tp_floor,
        )
    if resolved_type == "hist_keepalive_prewarm_v1":
        return HistogramKeepalivePrewarmAutoscaler(
            templates=templates,
            sync_period_sec=(
                config.hist_sync_period_sec
                if int(config.hist_sync_period_sec) > 0
                else config.sync_period_sec
            ),
            keepalive_idle_sec=config.hist_keepalive_idle_sec,
            keepalive_min_replicas=config.hist_keepalive_min_replicas,
            prewarm_buffer=config.hist_prewarm_buffer,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas_per_node,
            bin_minutes=config.hist_bin_minutes,
            range_minutes=config.hist_range_minutes,
            head_percentile=config.hist_head_percentile,
            tail_percentile=config.hist_tail_percentile,
            margin_ratio=config.hist_margin_ratio,
            min_samples=config.hist_min_samples,
            cv_threshold=config.hist_cv_threshold,
            oob_ratio_threshold=config.hist_oob_ratio_threshold,
            history_retention_sec=config.hist_history_retention_sec,
            forecast_margin_ratio=config.hist_forecast_margin_ratio,
            forecast_alpha=config.hist_forecast_alpha,
            forecast_min_samples=config.hist_forecast_min_samples,
        )
    if resolved_type == "no_autoscale_v1":
        return NoOpAutoscaler(name="no_autoscale_v1")
    raise ValueError(f"unsupported autoscaler type: {config.type}")
