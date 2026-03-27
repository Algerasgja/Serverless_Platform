from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from simulator.config import SimulationConfig, dump_config


class RunArtifactsWriter:
    def __init__(
        self,
        *,
        output_root: str | Path,
        config: SimulationConfig,
        scheduler_name: str,
        autoscaler_name: str,
        dataset_metadata: dict[str, Any],
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(output_root) / f"{timestamp}_{scheduler_name}_{autoscaler_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_metadata = dataset_metadata
        self._config = config

        dump_config(config, self.run_dir / "config.snapshot.yaml")

        self._scheduler_f = (self.run_dir / "scheduler_decisions.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._scheduler_writer = csv.DictWriter(
            self._scheduler_f,
            fieldnames=[
                "timestamp_ms",
                "request_id",
                "um",
                "function_node",
                "decision_type",
                "physical_node",
                "container_id",
                "cold_start_ms",
                "transfer_ms",
                "execution_ms",
                "container_state_before",
                "container_state_after",
                "cpu_request_mcpu",
                "memory_mb",
                "transfer_data_mb",
                "bandwidth_mbps",
                "allocated_cpu_mcpu",
                "reason",
            ],
        )
        self._scheduler_writer.writeheader()

        self._autoscaler_f = (self.run_dir / "autoscaler_decisions.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._autoscaler_writer = csv.DictWriter(
            self._autoscaler_f,
            fieldnames=[
                "timestamp_sec",
                "node",
                "current_replicas",
                "current_utilization",
                "target_utilization",
                "desired_replicas",
                "applied_replicas",
                "reason",
            ],
        )
        self._autoscaler_writer.writeheader()

        self._node_metrics_f = (self.run_dir / "node_metrics.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._node_metrics_writer = csv.DictWriter(
            self._node_metrics_f,
            fieldnames=[
                "timestamp_sec",
                "node",
                "active_containers",
                "busy_containers",
                "max_containers",
                "replicas",
                "inflight",
                "queue_len",
                "utilization",
                "draining_replicas",
                "cpu_total_mcpu",
                "cpu_reserved_mcpu",
                "cpu_utilization",
                "mem_total_mb",
                "mem_reserved_mb",
                "mem_utilization",
                "cold_starting_containers",
                "running_containers",
                "idle_containers",
            ],
        )
        self._node_metrics_writer.writeheader()

        self._request_paths_f = (self.run_dir / "request_paths.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._request_paths_writer = csv.DictWriter(
            self._request_paths_f,
            fieldnames=[
                "request_id",
                "session_id",
                "um",
                "arrival_ms",
                "path",
                "completed",
                "timed_out",
                "failed_reason",
                "total_latency_ms",
                "cold_start_latency_ms",
                "data_transfer_latency_ms",
                "execution_latency_ms",
                "queue_wait_latency_ms",
            ],
        )
        self._request_paths_writer.writeheader()

    def log_scheduler_decision(
        self,
        *,
        timestamp_ms: int,
        request_id: str,
        um: str,
        function_node: str,
        decision_type: str,
        physical_node: str | None,
        container_id: str | None,
        cold_start_ms: int,
        transfer_ms: int,
        execution_ms: int,
        container_state_before: str | None,
        container_state_after: str | None,
        cpu_request_mcpu: int | None,
        memory_mb: int | None,
        transfer_data_mb: float,
        bandwidth_mbps: float | None,
        allocated_cpu_mcpu: float | None,
        reason: str,
    ) -> None:
        self._scheduler_writer.writerow(
            {
                "timestamp_ms": timestamp_ms,
                "request_id": request_id,
                "um": um,
                "function_node": function_node,
                "decision_type": decision_type,
                "physical_node": physical_node or "",
                "container_id": container_id or "",
                "cold_start_ms": cold_start_ms,
                "transfer_ms": transfer_ms,
                "execution_ms": execution_ms,
                "container_state_before": container_state_before or "",
                "container_state_after": container_state_after or "",
                "cpu_request_mcpu": "" if cpu_request_mcpu is None else cpu_request_mcpu,
                "memory_mb": "" if memory_mb is None else memory_mb,
                "transfer_data_mb": f"{transfer_data_mb:.6f}",
                "bandwidth_mbps": "" if bandwidth_mbps is None else f"{bandwidth_mbps:.6f}",
                "allocated_cpu_mcpu": "" if allocated_cpu_mcpu is None else f"{allocated_cpu_mcpu:.6f}",
                "reason": reason,
            }
        )

    def log_autoscaler_decision(
        self,
        *,
        timestamp_sec: int,
        node: str,
        current_replicas: int,
        current_utilization: float,
        target_utilization: float,
        desired_replicas: int,
        applied_replicas: int,
        reason: str,
    ) -> None:
        self._autoscaler_writer.writerow(
            {
                "timestamp_sec": timestamp_sec,
                "node": node,
                "current_replicas": current_replicas,
                "current_utilization": f"{current_utilization:.6f}",
                "target_utilization": f"{target_utilization:.6f}",
                "desired_replicas": desired_replicas,
                "applied_replicas": applied_replicas,
                "reason": reason,
            }
        )

    def log_node_metric(
        self,
        *,
        timestamp_sec: int,
        node: str,
        active_containers: int,
        busy_containers: int,
        max_containers: int,
        replicas: int,
        inflight: int,
        queue_len: int,
        utilization: float,
        draining_replicas: int,
        cpu_total_mcpu: int,
        cpu_reserved_mcpu: int,
        cpu_utilization: float,
        mem_total_mb: int,
        mem_reserved_mb: int,
        mem_utilization: float,
        cold_starting_containers: int,
        running_containers: int,
        idle_containers: int,
    ) -> None:
        self._node_metrics_writer.writerow(
            {
                "timestamp_sec": timestamp_sec,
                "node": node,
                "active_containers": active_containers,
                "busy_containers": busy_containers,
                "max_containers": max_containers,
                "replicas": replicas,
                "inflight": inflight,
                "queue_len": queue_len,
                "utilization": f"{utilization:.6f}",
                "draining_replicas": draining_replicas,
                "cpu_total_mcpu": cpu_total_mcpu,
                "cpu_reserved_mcpu": cpu_reserved_mcpu,
                "cpu_utilization": f"{cpu_utilization:.6f}",
                "mem_total_mb": mem_total_mb,
                "mem_reserved_mb": mem_reserved_mb,
                "mem_utilization": f"{mem_utilization:.6f}",
                "cold_starting_containers": cold_starting_containers,
                "running_containers": running_containers,
                "idle_containers": idle_containers,
            }
        )

    def log_request_path(
        self,
        *,
        request_id: str,
        session_id: str,
        um: str,
        arrival_ms: int,
        path: list[str],
        completed: bool,
        timed_out: bool,
        failed_reason: str | None,
        total_latency_ms: int | None,
        cold_start_latency_ms: int,
        data_transfer_latency_ms: int,
        execution_latency_ms: int,
        queue_wait_latency_ms: int,
    ) -> None:
        self._request_paths_writer.writerow(
            {
                "request_id": request_id,
                "session_id": session_id,
                "um": um,
                "arrival_ms": arrival_ms,
                "path": "->".join(path),
                "completed": int(completed),
                "timed_out": int(timed_out),
                "failed_reason": failed_reason or "",
                "total_latency_ms": "" if total_latency_ms is None else total_latency_ms,
                "cold_start_latency_ms": cold_start_latency_ms,
                "data_transfer_latency_ms": data_transfer_latency_ms,
                "execution_latency_ms": execution_latency_ms,
                "queue_wait_latency_ms": queue_wait_latency_ms,
            }
        )

    def finalize(
        self,
        summary: dict[str, Any],
        resource_metadata: dict[str, Any] | None = None,
        dag_selection: dict[str, Any] | None = None,
        workload_replay_profile: dict[str, Any] | None = None,
        path_model: dict[str, Any] | None = None,
        autoscaler_summary: dict[str, Any] | None = None,
    ) -> None:
        autoscaler_payload = autoscaler_summary or {}
        xanadu_payload = autoscaler_payload if self._config.autoscaler.type == "xanadu_v1" else {}
        xanadu_opt_payload = autoscaler_payload if self._config.autoscaler.type == "xanadu_opt_v1" else {}
        oracle_payload = autoscaler_payload if self._config.autoscaler.type == "oracle_future_v1" else {}
        depth_breadth_payload = autoscaler_payload if self._config.autoscaler.type == "depth_breadth_v1" else {}
        kraken_vomm_payload = autoscaler_payload if self._config.autoscaler.type == "kraken_vomm_v1" else {}
        hpwp_payload = autoscaler_payload if self._config.autoscaler.type == "hpwp_v1" else {}
        kpa_payload = autoscaler_payload if self._config.autoscaler.type == "kpa_v1" else {}
        lass_payload = autoscaler_payload if self._config.autoscaler.type == "lass_v1" else {}
        hptd_payload = autoscaler_payload if self._config.autoscaler.type == "hptd_v1" else {}
        rl_q_payload = autoscaler_payload if self._config.autoscaler.type == "rl_q_v1" else {}
        hist_payload = autoscaler_payload if self._config.autoscaler.type == "hist_keepalive_prewarm_v1" else {}
        payload = {
            "experiment": self._config.experiment.name,
            "execution_model": "physical_nodes_no_queue",
            "scheduler": self._config.scheduler.type,
            "autoscaler": self._config.autoscaler.type,
            "autoscaler_params": {
                "target_utilization": self._config.autoscaler.target_utilization,
                "sync_period_sec": self._config.autoscaler.sync_period_sec,
                "scale_down_stabilization_sec": self._config.autoscaler.scale_down_stabilization_sec,
                "min_replicas": self._config.autoscaler.min_replicas,
                "max_replicas_per_node": self._config.autoscaler.max_replicas_per_node,
                "kpa_target_concurrency": self._config.autoscaler.kpa_target_concurrency,
                "kpa_stable_window_sec": self._config.autoscaler.kpa_stable_window_sec,
                "kpa_panic_window_sec": self._config.autoscaler.kpa_panic_window_sec,
                "kpa_panic_threshold": self._config.autoscaler.kpa_panic_threshold,
                "kpa_use_target_utilization": self._config.autoscaler.kpa_use_target_utilization,
                "kpa_panic_min_hold_sec": self._config.autoscaler.kpa_panic_min_hold_sec,
                "kpa_panic_exit_streak_sec": self._config.autoscaler.kpa_panic_exit_streak_sec,
                "kpa_max_scale_up_step": self._config.autoscaler.kpa_max_scale_up_step,
                "lass_latency_target_ms": self._config.autoscaler.lass_latency_target_ms,
                "lass_load_window_sec": self._config.autoscaler.lass_load_window_sec,
                "lass_speed_ewma_alpha": self._config.autoscaler.lass_speed_ewma_alpha,
                "lass_min_speed_req_per_sec": self._config.autoscaler.lass_min_speed_req_per_sec,
                "lass_min_samples": self._config.autoscaler.lass_min_samples,
                "hptd_time_granularity_sec": self._config.autoscaler.hptd_time_granularity_sec,
                "hptd_wcall_t": self._config.autoscaler.hptd_wcall_t,
                "hptd_whistory_t": self._config.autoscaler.hptd_whistory_t,
                "hptd_wchange_t": self._config.autoscaler.hptd_wchange_t,
                "hptd_alpha": self._config.autoscaler.hptd_alpha,
                "hptd_beta": self._config.autoscaler.hptd_beta,
                "hptd_mu_floor": self._config.autoscaler.hptd_mu_floor,
                "hptd_std_floor": self._config.autoscaler.hptd_std_floor,
                "hptd_temp_floor": self._config.autoscaler.hptd_temp_floor,
                "hptd_scale_max_step": self._config.autoscaler.hptd_scale_max_step,
                "rl_time_granularity_sec": self._config.autoscaler.rl_time_granularity_sec,
                "rl_wcall_t": self._config.autoscaler.rl_wcall_t,
                "rl_whistory_t": self._config.autoscaler.rl_whistory_t,
                "rl_wchange_t": self._config.autoscaler.rl_wchange_t,
                "rl_learning_rate": self._config.autoscaler.rl_learning_rate,
                "rl_discount_factor": self._config.autoscaler.rl_discount_factor,
                "rl_epsilon_init": self._config.autoscaler.rl_epsilon_init,
                "rl_epsilon_decay": self._config.autoscaler.rl_epsilon_decay,
                "rl_epsilon_min": self._config.autoscaler.rl_epsilon_min,
                "rl_step_size": self._config.autoscaler.rl_step_size,
                "rl_util_threshold": self._config.autoscaler.rl_util_threshold,
                "rl_reward_tolerance": self._config.autoscaler.rl_reward_tolerance,
                "rl_scalability_alpha": self._config.autoscaler.rl_scalability_alpha,
                "rl_inhibit_token_max": self._config.autoscaler.rl_inhibit_token_max,
                "hist_sync_period_sec": self._config.autoscaler.hist_sync_period_sec,
                "hist_window_sec": self._config.autoscaler.hist_window_sec,
                "hist_quantile": self._config.autoscaler.hist_quantile,
                "hist_keepalive_idle_sec": self._config.autoscaler.hist_keepalive_idle_sec,
                "hist_keepalive_min_replicas": self._config.autoscaler.hist_keepalive_min_replicas,
                "hist_prewarm_buffer": self._config.autoscaler.hist_prewarm_buffer,
                "xanadu_depth": self._config.autoscaler.xanadu_depth,
                "xanadu_ewma_alpha": self._config.autoscaler.xanadu_ewma_alpha,
                "xanadu_seed_offset": self._config.autoscaler.xanadu_seed_offset,
                "oracle_window_steps": self._config.autoscaler.oracle_window_steps,
                "hpwp_lmax_low": self._config.autoscaler.hpwp_lmax_low,
                "hpwp_lmax_high": self._config.autoscaler.hpwp_lmax_high,
                "hpwp_beta_hi": self._config.autoscaler.hpwp_beta_hi,
                "hpwp_beta_lo": self._config.autoscaler.hpwp_beta_lo,
                "hpwp_alpha_exp": self._config.autoscaler.hpwp_alpha_exp,
                "hpwp_alpha_stable": self._config.autoscaler.hpwp_alpha_stable,
                "hpwp_sched_eta_exec": self._config.autoscaler.hpwp_sched_eta_exec,
                "hpwp_sched_min_sec": self._config.autoscaler.hpwp_sched_min_sec,
                "hpwp_sched_max_sec": self._config.autoscaler.hpwp_sched_max_sec,
                "hpwp_horizon_alpha": self._config.autoscaler.hpwp_horizon_alpha,
                "hpwp_urgency_epsilon_ms": self._config.autoscaler.hpwp_urgency_epsilon_ms,
                "hpwp_phase_window_k": self._config.autoscaler.hpwp_phase_window_k,
                "hpwp_phase_n_min": self._config.autoscaler.hpwp_phase_n_min,
                "hpwp_phase_var_threshold": self._config.autoscaler.hpwp_phase_var_threshold,
                "hpwp_drift_short_k": self._config.autoscaler.hpwp_drift_short_k,
                "hpwp_drift_long_k": self._config.autoscaler.hpwp_drift_long_k,
                "hpwp_drift_delta_mr": self._config.autoscaler.hpwp_drift_delta_mr,
                "hpwp_drift_tau_mr": self._config.autoscaler.hpwp_drift_tau_mr,
                "hpwp_drift_branch_tau": self._config.autoscaler.hpwp_drift_branch_tau,
                "hpwp_forget_gamma": self._config.autoscaler.hpwp_forget_gamma,
                "hpwp_default_exec_ms": self._config.autoscaler.hpwp_default_exec_ms,
                "hpwp_default_cold_ms": self._config.autoscaler.hpwp_default_cold_ms,
                "hpwp_default_trans_ms": self._config.autoscaler.hpwp_default_trans_ms,
                "hpwp_seed_offset": self._config.autoscaler.hpwp_seed_offset,
                "dbw_horizon_boost": self._config.autoscaler.dbw_horizon_boost,
                "dbw_desired_scale": self._config.autoscaler.dbw_desired_scale,
                "dbw_desired_ewma_alpha": self._config.autoscaler.dbw_desired_ewma_alpha,
                "dbw_guard_buffer_min": self._config.autoscaler.dbw_guard_buffer_min,
                "dbw_down_margin": self._config.autoscaler.dbw_down_margin,
                "dbw_min_idle_age_sec": self._config.autoscaler.dbw_min_idle_age_sec,
                "dbw_down_cooldown_sec": self._config.autoscaler.dbw_down_cooldown_sec,
                "dbw_max_down_ratio": self._config.autoscaler.dbw_max_down_ratio,
                "dbw_osc_window_sec": self._config.autoscaler.dbw_osc_window_sec,
                "dbw_osc_trigger_count": self._config.autoscaler.dbw_osc_trigger_count,
                "kraken_misallocation_ratio": self._config.autoscaler.kraken_misallocation_ratio,
            },
            "capacity": {
                "max_total_instances": self._config.capacity.max_total_instances,
            },
            "physical_nodes": {
                "count": self._config.physical_nodes.count,
                "max_containers_per_node": self._config.physical_nodes.max_containers_per_node,
                "idle_ttl_sec": self._config.physical_nodes.idle_ttl_sec,
                "same_node_transfer_ms_min": self._config.physical_nodes.same_node_transfer_ms_min,
                "same_node_transfer_ms_max": self._config.physical_nodes.same_node_transfer_ms_max,
                "cross_node_transfer_ms_min": self._config.physical_nodes.cross_node_transfer_ms_min,
                "cross_node_transfer_ms_max": self._config.physical_nodes.cross_node_transfer_ms_max,
                "cpu_total_mcpu_per_node": self._config.physical_nodes.cpu_total_mcpu_per_node,
                "mem_total_mb_per_node": self._config.physical_nodes.mem_total_mb_per_node,
                "bandwidth_mode": self._config.physical_nodes.bandwidth_mode,
                "bandwidth_mbps_min": self._config.physical_nodes.bandwidth_mbps_min,
                "bandwidth_mbps_max": self._config.physical_nodes.bandwidth_mbps_max,
                "bandwidth_matrix_file": self._config.physical_nodes.bandwidth_matrix_file,
            },
            "resource_model": resource_metadata or {},
            "seed_info": {
                "experiment_random_seed": self._config.experiment.random_seed,
                "function_profile_seed_offset": self._config.runtime.function_profile_seed_offset,
                "bandwidth_seed_offset": self._config.physical_nodes.bandwidth_seed_offset,
            },
            "dag_selection": dag_selection or {},
            "workload_replay_profile": workload_replay_profile or {},
            "path_model": path_model or {},
            "autoscaler_runtime": autoscaler_payload,
            "kpa": kpa_payload,
            "hist_keepalive_prewarm": hist_payload,
            "xanadu": xanadu_payload,
            "xanadu_opt": xanadu_opt_payload,
            "oracle": oracle_payload,
            "depth_breadth": depth_breadth_payload,
            "kraken_vomm": kraken_vomm_payload,
            "hpwp": hpwp_payload,
            "lass": lass_payload,
            "hptd": hptd_payload,
            "rl_q": rl_q_payload,
            "dataset_metadata": self._dataset_metadata,
            "summary": summary,
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        self.close()

    def close(self) -> None:
        self._scheduler_f.close()
        self._autoscaler_f.close()
        self._node_metrics_f.close()
        self._request_paths_f.close()
