from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Any

from simulator.strategies.autoscaler.base import AutoscalerStrategy, PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


class RlQAutoscaler(AutoscalerStrategy):
    """Q-learning autoscaler with SLA-aware reward and profile-guided scale-out."""

    name = "rl_q_v1"
    _ACTIONS = (-1, 0, 1)

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        time_granularity_sec: int,
        wcall_t: int,
        whistory_t: int,
        wchange_t: int,
        learning_rate: float,
        discount_factor: float,
        epsilon_init: float,
        epsilon_decay: float,
        epsilon_min: float,
        step_size: int,
        util_threshold: float,
        reward_tolerance: float,
        scalability_alpha: float,
        inhibit_token_max: int,
        min_replicas: int,
        max_replicas: int,
        # Paper-lite extensions for profile + SLA constraints.
        slo_p95_ms: float = 4000.0,
        failure_window_sec: int = 300,
        failure_rate_threshold: float = 0.05,
        sla_penalty_weight: float = 0.35,
        profile_warmup_sec: int = 60,
        profile_ewma_alpha: float = 0.2,
        alpha_scalability_min: float = 0.0,
        alpha_scalability_max: float = 0.5,
        tp_floor: float = 0.05,
    ) -> None:
        del templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._t_sec = max(1, int(time_granularity_sec))
        self._wcall_sec = max(1, int(wcall_t)) * self._t_sec
        self._whistory_sec = max(2, int(whistory_t)) * self._t_sec
        self._wchange_sec = max(1, int(wchange_t)) * self._t_sec
        if self._wchange_sec >= self._whistory_sec:
            self._wchange_sec = max(1, self._whistory_sec // 2)

        self._lr = min(1.0, max(0.0, float(learning_rate)))
        self._gamma = min(1.0, max(0.0, float(discount_factor)))
        self._epsilon_init = min(1.0, max(0.0, float(epsilon_init)))
        self._epsilon = self._epsilon_init
        self._epsilon_decay = min(1.0, max(0.0, float(epsilon_decay)))
        self._epsilon_min = min(1.0, max(0.0, float(epsilon_min)))
        self._step_size = max(1, int(step_size))
        self._util_threshold = min(1.0, max(0.0, float(util_threshold)))
        self._reward_tolerance = max(0.0, float(reward_tolerance))
        self._scalability_alpha = max(0.0, float(scalability_alpha))
        self._inhibit_token_max = max(0, int(inhibit_token_max))
        self._min_replicas = max(0, int(min_replicas))
        self._max_replicas = max(0, int(max_replicas))
        if self._max_replicas > 0 and self._max_replicas < self._min_replicas:
            self._max_replicas = self._min_replicas

        self._slo_p95_ms = max(1.0, float(slo_p95_ms))
        self._failure_window_sec = max(10, int(failure_window_sec))
        self._failure_rate_threshold = min(1.0, max(0.0, float(failure_rate_threshold)))
        self._sla_penalty_weight = min(1.0, max(0.0, float(sla_penalty_weight)))
        self._profile_warmup_sec = max(1, int(profile_warmup_sec))
        self._profile_ewma_alpha = min(1.0, max(0.0, float(profile_ewma_alpha)))
        self._alpha_scalability_min = float(alpha_scalability_min)
        self._alpha_scalability_max = max(self._alpha_scalability_min, float(alpha_scalability_max))
        self._tp_floor = max(1e-4, float(tp_floor))

        self._rng = random.Random(20260318)

        self._q: dict[tuple[int, int, int, int], dict[int, float]] = {}
        self._prev_state_action_by_function: dict[str, tuple[tuple[int, int, int, int], int]] = {}

        self._inflight_by_function: dict[str, int] = defaultdict(int)
        self._request_active_function: dict[str, str] = {}
        self._inhibit_tokens_by_function: dict[str, int] = defaultdict(int)

        self._arrivals_by_function: dict[str, deque[int]] = defaultdict(deque)
        self._step_history_by_function: dict[str, deque[tuple[int, int, int]]] = defaultdict(deque)
        self._latency_samples_by_function: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._capacity_history_by_function: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        self._desired_by_function: dict[str, int] = {}

        # Service profile estimates (paper-lite approximation).
        self._tp_by_function: dict[str, float] = {}
        self._tpm_by_function: dict[str, float] = {}
        self._alpha_by_function: dict[str, float] = {}
        self._profile_ticks_by_function: dict[str, int] = defaultdict(int)

        self._action_counts = {"-1": 0, "0": 0, "+1": 0}
        self._inhibit_events = 0
        self._q_update_count = 0

        self._reconcile_rounds = 0
        self._desired_peak_total = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._capacity_blocked_creations = 0
        self._scale_down_requested = 0

    def on_step_start(
        self,
        *,
        request_id: str,
        um: str,
        function_node: str,
        timestamp_ms: int,
        true_future_path: tuple[str, ...] | None = None,
    ) -> None:
        del um, true_future_path
        previous = self._request_active_function.get(request_id)
        if previous and previous != function_node:
            self._decrease_inflight(previous)
        self._request_active_function[request_id] = function_node
        self._inflight_by_function[function_node] += 1

        arrivals = self._arrivals_by_function[function_node]
        arrivals.append(int(timestamp_ms))
        self._trim_arrivals(function_node=function_node, now_ms=int(timestamp_ms))

    def on_transition(
        self,
        *,
        request_id: str | None = None,
        um: str,
        src_node: str,
        dst_node: str,
        timestamp_ms: int,
        transfer_ms: int | None = None,
        prefix: tuple[str, ...] | None = None,
    ) -> None:
        del request_id, um, src_node, dst_node, timestamp_ms, transfer_ms, prefix
        return None

    def on_step_observed(
        self,
        *,
        request_id: str,
        um: str,
        function_node: str,
        timestamp_ms: int,
        execution_ms: int,
        cold_start_ms: int,
        transfer_ms: int,
        prefix: tuple[str, ...],
    ) -> None:
        del um, prefix
        sec = max(0, int(timestamp_ms) // 1000)
        cold = 1 if int(cold_start_ms) > 0 else 0
        self._append_step_bucket(function_node=function_node, timestamp_sec=sec, total_inc=1, cold_inc=cold)
        total_step_latency = float(max(0, int(execution_ms) + int(cold_start_ms) + int(transfer_ms)))
        self._append_latency_sample(
            function_node=function_node,
            timestamp_sec=sec,
            latency_ms=total_step_latency,
        )

        self._decrease_inflight(function_node)
        active_fn = self._request_active_function.get(request_id)
        if active_fn == function_node:
            self._request_active_function.pop(request_id, None)

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del status, timestamp_ms
        active_fn = self._request_active_function.pop(request_id, None)
        if active_fn:
            self._decrease_inflight(active_fn)

    def on_tick(
        self,
        *,
        timestamp_sec: int,
        timestamp_ms: int,
        ready_pool_by_function: dict[str, int],
        idle_pool_by_function: dict[str, int] | None = None,
    ) -> list[PrewarmPlan | ScaleDownPlan]:
        if idle_pool_by_function is None:
            idle_pool_by_function = ready_pool_by_function
        now_sec = int(timestamp_sec)
        now_ms = int(timestamp_ms)
        self._trim_all_windows(now_sec=now_sec, now_ms=now_ms)

        if now_sec % self._sync_period_sec != 0:
            return []

        self._reconcile_rounds += 1
        desired: dict[str, int] = {}
        plans: list[PrewarmPlan | ScaleDownPlan] = []

        tracked_functions = (
            set(self._inflight_by_function.keys())
            | set(self._arrivals_by_function.keys())
            | set(self._step_history_by_function.keys())
            | set(self._latency_samples_by_function.keys())
            | set(self._capacity_history_by_function.keys())
            | set(self._prev_state_action_by_function.keys())
            | set(ready_pool_by_function.keys())
        )

        for function_node in sorted(tracked_functions):
            inflight = max(0, int(self._inflight_by_function.get(function_node, 0)))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))
            current_capacity = inflight + ready
            self._append_capacity_bucket(
                function_node=function_node,
                timestamp_sec=now_sec,
                capacity=current_capacity,
            )

            throughput_i = self._throughput_recent(function_node=function_node, now_sec=now_sec)
            throughput_ref = self._throughput_reference(function_node=function_node, now_sec=now_sec)
            resource_i = max(1.0, float(current_capacity))
            resource_ref = self._resource_reference(function_node=function_node, now_sec=now_sec)
            cold_ratio = self._cold_ratio_recent(function_node=function_node, now_sec=now_sec)
            p95_recent = self._p95_latency_recent(function_node=function_node, now_sec=now_sec)
            fail_rate_recent = self._failure_rate_recent(function_node=function_node, now_sec=now_sec)
            rps_recent = self._arrival_rate_recent(function_node=function_node, now_ms=now_ms)
            rps_ref = self._arrival_rate_reference(function_node=function_node, now_ms=now_ms)

            latency_slack = p95_recent / max(1e-6, self._slo_p95_ms)
            rps_ratio = rps_recent / max(1e-6, rps_ref)
            util = inflight / max(1.0, float(current_capacity))
            current_state = self._build_state(
                util=util,
                latency_slack=latency_slack,
                rps_ratio=rps_ratio,
                cold_ratio=cold_ratio,
            )

            reward = self._compute_reward(
                throughput_i=throughput_i,
                throughput_ref=throughput_ref,
                resource_i=resource_i,
                resource_ref=resource_ref,
            )
            reward = self._apply_sla_penalty(
                reward=reward,
                p95_latency_ms=p95_recent,
                failure_rate=fail_rate_recent,
            )

            previous = self._prev_state_action_by_function.get(function_node)
            if previous is not None:
                prev_state, prev_action = previous
                self._update_q(prev_state=prev_state, action=prev_action, reward=reward, next_state=current_state)

            selected_action = self._select_action(state=current_state, util=util)
            self._action_counts[self._action_key(selected_action)] += 1

            effective_action = selected_action
            if selected_action < 0:
                if self._inhibit_token_max > 0:
                    self._inhibit_tokens_by_function[function_node] = min(
                        self._inhibit_token_max,
                        self._inhibit_tokens_by_function.get(function_node, 0) + 1,
                    )
            elif selected_action > 0:
                tokens = self._inhibit_tokens_by_function.get(function_node, 0)
                if tokens > 0:
                    self._inhibit_tokens_by_function[function_node] = tokens - 1
                    self._inhibit_events += 1
                    effective_action = 0

            self._update_profile_estimate(
                function_node=function_node,
                now_sec=now_sec,
                throughput_recent=throughput_i,
                capacity=current_capacity,
            )
            desired_count = self._desired_from_policy(
                function_node=function_node,
                inflight=inflight,
                ready=ready,
                action=effective_action,
                rps_recent=rps_recent,
                throughput_ref=throughput_ref,
            )
            if desired_count > 0:
                desired[function_node] = desired_count

            to_create = max(0, desired_count - current_capacity)
            to_remove = min(idle, max(0, current_capacity - desired_count))
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
            elif to_remove > 0:
                plans.append(ScaleDownPlan(function_node=function_node, count=to_remove))
                self._scale_down_requested += to_remove

            self._prev_state_action_by_function[function_node] = (current_state, selected_action)

        self._desired_by_function = desired
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        return plans

    def on_prewarm_create_result(
        self,
        *,
        function_node: str,
        success: bool,
        timestamp_ms: int,
        reason: str,
    ) -> None:
        del function_node, timestamp_ms, reason
        if success:
            self._prewarm_created += 1
        else:
            self._capacity_blocked_creations += 1

    def on_prewarm_ready(
        self,
        *,
        function_node: str,
        container_id: str,
        timestamp_ms: int,
    ) -> None:
        del function_node, container_id, timestamp_ms
        self._prewarm_ready += 1

    def on_prewarm_consumed(
        self,
        *,
        function_node: str,
        request_id: str,
        container_id: str,
        timestamp_ms: int,
    ) -> None:
        del function_node, request_id, container_id, timestamp_ms
        self._prewarm_consumed += 1

    def summary(self) -> dict[str, Any]:
        avg_alpha = None
        if self._alpha_by_function:
            avg_alpha = float(sum(self._alpha_by_function.values()) / len(self._alpha_by_function))
        return {
            "type": self.name,
            "mode": "q_learning_profile_sla_reconcile",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "time_granularity_sec": self._t_sec,
                "wcall_t": int(self._wcall_sec / self._t_sec),
                "whistory_t": int(self._whistory_sec / self._t_sec),
                "wchange_t": int(self._wchange_sec / self._t_sec),
                "learning_rate": self._lr,
                "discount_factor": self._gamma,
                "epsilon_init": self._epsilon_init,
                "epsilon_decay": self._epsilon_decay,
                "epsilon_min": self._epsilon_min,
                "step_size": self._step_size,
                "util_threshold": self._util_threshold,
                "reward_tolerance": self._reward_tolerance,
                "scalability_alpha": self._scalability_alpha,
                "inhibit_token_max": self._inhibit_token_max,
                "slo_p95_ms": self._slo_p95_ms,
                "failure_window_sec": self._failure_window_sec,
                "failure_rate_threshold": self._failure_rate_threshold,
                "sla_penalty_weight": self._sla_penalty_weight,
                "profile_warmup_sec": self._profile_warmup_sec,
                "profile_ewma_alpha": self._profile_ewma_alpha,
                "alpha_scalability_min": self._alpha_scalability_min,
                "alpha_scalability_max": self._alpha_scalability_max,
                "tp_floor": self._tp_floor,
                "min_replicas": self._min_replicas,
                "max_replicas": self._max_replicas,
            },
            "q_state_count": len(self._q),
            "q_update_count": self._q_update_count,
            "epsilon_final": self._epsilon,
            "action_counts": dict(self._action_counts),
            "inhibit_events": self._inhibit_events,
            "profile_function_count": len(self._tp_by_function),
            "profile_avg_alpha": avg_alpha,
            "reconcile_rounds": self._reconcile_rounds,
            "desired_peak_total": self._desired_peak_total,
            "final_desired_by_function": dict(sorted(self._desired_by_function.items())),
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "capacity_blocked_creations": self._capacity_blocked_creations,
            "scale_down_requested": self._scale_down_requested,
        }

    def _select_action(self, *, state: tuple[int, int, int, int], util: float) -> int:
        table = self._q.get(state)
        if table is None:
            self._q[state] = self._new_action_row()
            return 1 if util > self._util_threshold else -1

        if self._rng.random() < self._epsilon:
            return self._rng.choice(list(self._ACTIONS))
        return max(self._ACTIONS, key=lambda act: (table.get(act, 0.0), act))

    def _update_q(
        self,
        *,
        prev_state: tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int],
    ) -> None:
        prev_row = self._q.setdefault(prev_state, self._new_action_row())
        next_row = self._q.setdefault(next_state, self._new_action_row())
        max_next = max(next_row.values())
        old_q = prev_row.get(action, 0.0)
        new_q = (1.0 - self._lr) * old_q + self._lr * (reward + self._gamma * max_next)
        prev_row[action] = new_q
        self._q_update_count += 1

    def _compute_reward(
        self,
        *,
        throughput_i: float,
        throughput_ref: float,
        resource_i: float,
        resource_ref: float,
    ) -> float:
        throughput_ref_safe = max(1e-6, float(throughput_ref))
        resource_i_safe = max(1.0, float(resource_i))
        resource_ref_safe = max(1.0, float(resource_ref))
        delta = (float(throughput_i) * resource_ref_safe) / (throughput_ref_safe * resource_i_safe)
        if abs(1.0 - delta) <= self._reward_tolerance:
            return 1.0
        return float(delta)

    def _apply_sla_penalty(self, *, reward: float, p95_latency_ms: float, failure_rate: float) -> float:
        penalty = 1.0
        if float(p95_latency_ms) > self._slo_p95_ms:
            penalty *= max(0.1, 1.0 - self._sla_penalty_weight)
        if float(failure_rate) > self._failure_rate_threshold:
            penalty *= max(0.1, 1.0 - 0.5 * self._sla_penalty_weight)
        return float(reward) * penalty

    def _update_profile_estimate(
        self,
        *,
        function_node: str,
        now_sec: int,
        throughput_recent: float,
        capacity: int,
    ) -> None:
        # Throughput per "single logical instance" proxy.
        inst_tp = float(throughput_recent) / max(1.0, float(capacity))
        if inst_tp <= 0.0:
            return

        prev_tp = self._tp_by_function.get(function_node, inst_tp)
        tp = (1.0 - self._profile_ewma_alpha) * prev_tp + self._profile_ewma_alpha * inst_tp
        self._tp_by_function[function_node] = max(self._tp_floor, tp)

        prev_tpm = self._tpm_by_function.get(function_node, inst_tp)
        # Slight decay to keep profile adaptive.
        decayed_tpm = prev_tpm * 0.995
        tpm = max(decayed_tpm, inst_tp, self._tp_by_function[function_node])
        self._tpm_by_function[function_node] = tpm

        tp_safe = max(self._tp_floor, self._tp_by_function[function_node])
        alpha_est = (tpm - tp_safe) / tp_safe
        alpha_est = min(self._alpha_scalability_max, max(self._alpha_scalability_min, alpha_est))
        self._alpha_by_function[function_node] = alpha_est
        self._profile_ticks_by_function[function_node] += 1
        # Keep counter bounded over long experiments.
        if self._profile_ticks_by_function[function_node] > max(1, 10 * now_sec):
            self._profile_ticks_by_function[function_node] = max(1, 10 * now_sec)

    def _desired_from_policy(
        self,
        *,
        function_node: str,
        inflight: int,
        ready: int,
        action: int,
        rps_recent: float,
        throughput_ref: float,
    ) -> int:
        n_current = max(0, int(inflight + ready))
        if action < 0:
            return self._bound_desired(n_current)

        tp = self._tp_effective(function_node=function_node, throughput_ref=throughput_ref, n_current=n_current)
        alpha = self._alpha_effective(function_node=function_node)
        n_prime = max(0, int(math.ceil(float(rps_recent) / max(self._tp_floor, tp))))
        n_warm = int(math.ceil((float(rps_recent) / max(self._tp_floor, tp)) - ((1.0 + alpha) * n_current / 2.0)))
        n_warm = max(0, n_warm)
        desired_model = max(n_prime, n_current + n_warm)

        if action > 0:
            desired = max(n_current + self._step_size, desired_model)
        else:
            desired = n_current
        return self._bound_desired(desired)

    def _tp_effective(self, *, function_node: str, throughput_ref: float, n_current: int) -> float:
        ticks = int(self._profile_ticks_by_function.get(function_node, 0))
        if ticks >= self._profile_warmup_sec and function_node in self._tp_by_function:
            return max(self._tp_floor, float(self._tp_by_function[function_node]))
        fallback = float(throughput_ref) / max(1.0, float(n_current))
        return max(self._tp_floor, fallback)

    def _alpha_effective(self, *, function_node: str) -> float:
        ticks = int(self._profile_ticks_by_function.get(function_node, 0))
        if ticks >= self._profile_warmup_sec and function_node in self._alpha_by_function:
            return float(self._alpha_by_function[function_node])
        return min(self._alpha_scalability_max, max(self._alpha_scalability_min, self._scalability_alpha))

    def _build_state(
        self,
        *,
        util: float,
        latency_slack: float,
        rps_ratio: float,
        cold_ratio: float,
    ) -> tuple[int, int, int, int]:
        util_bin = _bin_float(util, thresholds=(0.40, 0.70, 0.90))
        latency_slack_bin = _bin_float(latency_slack, thresholds=(0.80, 1.00, 1.20))
        rps_ratio_bin = _bin_float(rps_ratio, thresholds=(0.80, 1.00, 1.20))
        cold_ratio_bin = _bin_float(cold_ratio, thresholds=(0.10, 0.30, 0.60))
        return util_bin, latency_slack_bin, rps_ratio_bin, cold_ratio_bin

    def _throughput_recent(self, *, function_node: str, now_sec: int) -> float:
        hist = self._step_history_by_function.get(function_node)
        if not hist:
            return 0.0
        start = now_sec - self._wchange_sec + 1
        total = 0
        for sec, count, _ in hist:
            if sec >= start:
                total += count
        return float(total) / float(self._wchange_sec)

    def _throughput_reference(self, *, function_node: str, now_sec: int) -> float:
        hist = self._step_history_by_function.get(function_node)
        if not hist:
            return 1e-6
        start = now_sec - self._whistory_sec + 1
        total = 0
        for sec, count, _ in hist:
            if sec >= start:
                total += count
        return max(1e-6, float(total) / float(self._whistory_sec))

    def _arrival_rate_recent(self, *, function_node: str, now_ms: int) -> float:
        arrivals = self._arrivals_by_function.get(function_node)
        if not arrivals:
            return 0.0
        threshold = now_ms - self._wchange_sec * 1000
        count = 0
        for ts in arrivals:
            if ts >= threshold:
                count += 1
        return float(count) / float(self._wchange_sec)

    def _arrival_rate_reference(self, *, function_node: str, now_ms: int) -> float:
        arrivals = self._arrivals_by_function.get(function_node)
        if not arrivals:
            return 1e-6
        threshold = now_ms - self._whistory_sec * 1000
        count = 0
        for ts in arrivals:
            if ts >= threshold:
                count += 1
        return max(1e-6, float(count) / float(self._whistory_sec))

    def _cold_ratio_recent(self, *, function_node: str, now_sec: int) -> float:
        hist = self._step_history_by_function.get(function_node)
        if not hist:
            return 0.0
        start = now_sec - self._wchange_sec + 1
        total = 0
        cold = 0
        for sec, count, cold_count in hist:
            if sec >= start:
                total += count
                cold += cold_count
        if total <= 0:
            return 0.0
        return float(cold) / float(total)

    def _p95_latency_recent(self, *, function_node: str, now_sec: int) -> float:
        samples = self._latency_samples_by_function.get(function_node)
        if not samples:
            return 0.0
        start = now_sec - self._wchange_sec + 1
        vals = [lat for sec, lat in samples if sec >= start]
        if not vals:
            return 0.0
        return _percentile(vals, 95.0)

    def _failure_rate_recent(self, *, function_node: str, now_sec: int) -> float:
        samples = self._latency_samples_by_function.get(function_node)
        if not samples:
            return 0.0
        start = now_sec - self._failure_window_sec + 1
        vals = [lat for sec, lat in samples if sec >= start]
        if not vals:
            return 0.0
        failed = sum(1 for lat in vals if lat > self._slo_p95_ms)
        return float(failed) / float(len(vals))

    def _resource_reference(self, *, function_node: str, now_sec: int) -> float:
        hist = self._capacity_history_by_function.get(function_node)
        if not hist:
            return 1.0
        start = now_sec - self._whistory_sec + 1
        values = [cap for sec, cap in hist if sec >= start]
        if not values:
            return 1.0
        return max(1.0, float(sum(values)) / float(len(values)))

    def _append_step_bucket(
        self,
        *,
        function_node: str,
        timestamp_sec: int,
        total_inc: int,
        cold_inc: int,
    ) -> None:
        hist = self._step_history_by_function[function_node]
        if hist and hist[-1][0] == timestamp_sec:
            sec, total, cold = hist[-1]
            hist[-1] = (sec, total + int(total_inc), cold + int(cold_inc))
        else:
            hist.append((timestamp_sec, int(total_inc), int(cold_inc)))
        self._trim_step_history(function_node=function_node, now_sec=timestamp_sec)

    def _append_latency_sample(
        self,
        *,
        function_node: str,
        timestamp_sec: int,
        latency_ms: float,
    ) -> None:
        hist = self._latency_samples_by_function[function_node]
        hist.append((timestamp_sec, float(latency_ms)))
        self._trim_latency_history(function_node=function_node, now_sec=timestamp_sec)

    def _append_capacity_bucket(
        self,
        *,
        function_node: str,
        timestamp_sec: int,
        capacity: int,
    ) -> None:
        hist = self._capacity_history_by_function[function_node]
        if hist and hist[-1][0] == timestamp_sec:
            hist[-1] = (timestamp_sec, int(capacity))
        else:
            hist.append((timestamp_sec, int(capacity)))
        self._trim_capacity_history(function_node=function_node, now_sec=timestamp_sec)

    def _trim_all_windows(self, *, now_sec: int, now_ms: int) -> None:
        for fn in list(self._arrivals_by_function.keys()):
            self._trim_arrivals(function_node=fn, now_ms=now_ms)
        for fn in list(self._step_history_by_function.keys()):
            self._trim_step_history(function_node=fn, now_sec=now_sec)
        for fn in list(self._latency_samples_by_function.keys()):
            self._trim_latency_history(function_node=fn, now_sec=now_sec)
        for fn in list(self._capacity_history_by_function.keys()):
            self._trim_capacity_history(function_node=fn, now_sec=now_sec)

    def _trim_arrivals(self, *, function_node: str, now_ms: int) -> None:
        arrivals = self._arrivals_by_function.get(function_node)
        if arrivals is None:
            return
        keep_sec = max(self._wcall_sec, self._whistory_sec, self._failure_window_sec)
        threshold = now_ms - (keep_sec * 1000)
        while arrivals and arrivals[0] < threshold:
            arrivals.popleft()
        if not arrivals:
            self._arrivals_by_function.pop(function_node, None)

    def _trim_step_history(self, *, function_node: str, now_sec: int) -> None:
        hist = self._step_history_by_function.get(function_node)
        if hist is None:
            return
        threshold = now_sec - self._whistory_sec
        while hist and hist[0][0] < threshold:
            hist.popleft()
        if not hist:
            self._step_history_by_function.pop(function_node, None)

    def _trim_latency_history(self, *, function_node: str, now_sec: int) -> None:
        hist = self._latency_samples_by_function.get(function_node)
        if hist is None:
            return
        keep_sec = max(self._whistory_sec, self._failure_window_sec)
        threshold = now_sec - keep_sec
        while hist and hist[0][0] < threshold:
            hist.popleft()
        if not hist:
            self._latency_samples_by_function.pop(function_node, None)

    def _trim_capacity_history(self, *, function_node: str, now_sec: int) -> None:
        hist = self._capacity_history_by_function.get(function_node)
        if hist is None:
            return
        threshold = now_sec - self._whistory_sec
        while hist and hist[0][0] < threshold:
            hist.popleft()
        if not hist:
            self._capacity_history_by_function.pop(function_node, None)

    def _decrease_inflight(self, function_node: str) -> None:
        current = int(self._inflight_by_function.get(function_node, 0))
        if current <= 1:
            self._inflight_by_function.pop(function_node, None)
        else:
            self._inflight_by_function[function_node] = current - 1

    def _bound_desired(self, desired: int) -> int:
        out = max(self._min_replicas, int(desired))
        if self._max_replicas > 0:
            out = min(out, self._max_replicas)
        return out

    def _new_action_row(self) -> dict[int, float]:
        return {-1: 0.0, 0: 0.0, 1: 0.0}

    @staticmethod
    def _action_key(action: int) -> str:
        if action > 0:
            return "+1"
        if action < 0:
            return "-1"
        return "0"


def _bin_float(value: float, *, thresholds: tuple[float, float, float]) -> int:
    if value < thresholds[0]:
        return 0
    if value < thresholds[1]:
        return 1
    if value < thresholds[2]:
        return 2
    return 3


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ranked = sorted(float(v) for v in values)
    # Linear interpolation percentile.
    k = (len(ranked) - 1) * (float(p) / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return ranked[f]
    d0 = ranked[f] * (c - k)
    d1 = ranked[c] * (k - f)
    return d0 + d1
