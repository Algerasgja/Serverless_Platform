from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


class LassAutoscaler:
    """Model-driven LaSS-style autoscaler (queueing-theoretic, scale-up only).

    The autoscaler estimates per-function capacity using an M/M/c queueing model:
    it finds the minimal server count c such that P(Q <= t_wait_target) >= 0.99.
    """

    name = "lass_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        latency_target_ms: float,
        load_window_sec: int,
        speed_ewma_alpha: float,
        min_speed_req_per_sec: float,
        min_samples: int,
        default_exec_ms: float,
        desired_scale: float = 1.0,
        min_desired_when_active: int = 0,
        topk_ratio: float = 1.0,
        low_load_boost: float = 1.0,
        high_load_dampen: float = 1.0,
        low_avg_load_threshold: float = 0.0,
        high_avg_load_threshold: float = 1e9,
        inflight_credit: float = 0.0,
        scale_cooldown_sec: int = 0,
        max_desired_cap: int = 0,
        max_create_per_tick: int = 0,
        low_load_max_create_per_tick: int = 0,
        low_load_max_create_threshold: float = 0.0,
        min_replicas: int,
        max_replicas: int,
    ) -> None:
        del templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._latency_target_ms = max(1.0, float(latency_target_ms))
        self._load_window_sec = max(1, int(load_window_sec))
        self._load_window_ms = self._load_window_sec * 1000
        self._speed_ewma_alpha = min(1.0, max(0.0, float(speed_ewma_alpha)))
        self._min_speed_req_per_sec = max(1e-4, float(min_speed_req_per_sec))
        self._min_samples = max(1, int(min_samples))
        self._default_exec_ms = max(1.0, float(default_exec_ms))
        self._default_speed_req_per_sec = max(self._min_speed_req_per_sec, 1000.0 / self._default_exec_ms)
        self._desired_scale = max(0.0, float(desired_scale))
        self._min_desired_when_active = max(0, int(min_desired_when_active))
        self._inflight_credit = float(max(0.0, min(1.0, inflight_credit)))
        self._scale_cooldown_sec = max(0, int(scale_cooldown_sec))
        self._max_desired_cap = max(0, int(max_desired_cap))
        self._max_create_per_tick = max(0, int(max_create_per_tick))
        self._low_load_max_create_per_tick = max(0, int(low_load_max_create_per_tick))
        self._low_load_max_create_threshold = float(max(0.0, low_load_max_create_threshold))
        self._min_replicas = max(0, int(min_replicas))
        self._max_replicas = max(0, int(max_replicas))
        if self._max_replicas > 0 and self._max_replicas < self._min_replicas:
            self._max_replicas = self._min_replicas

        # Backward-compatible knobs retained but demoted to light shaping.
        self._topk_ratio = float(max(0.0, min(1.0, topk_ratio)))
        self._low_load_boost = float(max(0.1, low_load_boost))
        self._high_load_dampen = float(max(0.1, high_load_dampen))
        self._low_avg_load_threshold = float(max(0.0, low_avg_load_threshold))
        self._high_avg_load_threshold = float(max(self._low_avg_load_threshold + 1e-6, high_avg_load_threshold))

        self._arrivals_by_function: dict[str, deque[int]] = defaultdict(deque)
        self._inflight_by_function: dict[str, int] = defaultdict(int)
        self._request_active_function: dict[str, str] = {}
        self._speed_by_function: dict[str, float] = {}
        self._samples_by_function: dict[str, int] = defaultdict(int)
        self._exec_samples_ms_by_function: dict[str, deque[int]] = defaultdict(deque)
        self._desired_by_function: dict[str, int] = {}
        self._last_scale_sec_by_function: dict[str, int] = {}

        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._capacity_blocked_creations = 0
        self._scale_down_requested = 0
        self._reconcile_rounds = 0
        self._desired_peak_total = 0

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
        del um, timestamp_ms, cold_start_ms, transfer_ms, prefix
        self._decrease_inflight(function_node)
        active_fn = self._request_active_function.get(request_id)
        if active_fn == function_node:
            self._request_active_function.pop(request_id, None)

        duration_ms = max(1, int(execution_ms))
        observed_speed = 1000.0 / float(duration_ms)
        prev = self._speed_by_function.get(function_node)
        if prev is None:
            next_speed = observed_speed
        else:
            alpha = self._speed_ewma_alpha
            next_speed = (1.0 - alpha) * prev + alpha * observed_speed
        self._speed_by_function[function_node] = max(self._min_speed_req_per_sec, next_speed)
        self._samples_by_function[function_node] += 1

        history = self._exec_samples_ms_by_function[function_node]
        history.append(duration_ms)
        max_hist = max(64, self._load_window_sec * 8)
        while len(history) > max_hist:
            history.popleft()

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
        return None

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
        if timestamp_sec % self._sync_period_sec != 0:
            return []

        self._reconcile_rounds += 1
        now_ms = int(timestamp_ms)

        tracked = (
            set(self._arrivals_by_function.keys())
            | set(self._speed_by_function.keys())
            | set(ready_pool_by_function.keys())
        )

        total_recent_load = 0
        active_fn_count = 0
        load_recent_by_fn: dict[str, int] = {}
        for fn in tracked:
            self._trim_arrivals(function_node=fn, now_ms=now_ms)
            load_recent = len(self._arrivals_by_function.get(fn, ()))
            load_recent_by_fn[fn] = load_recent
            if load_recent > 0:
                total_recent_load += load_recent
                active_fn_count += 1

        avg_load = float(total_recent_load) / float(max(1, active_fn_count))
        adaptive_scale = 1.0
        if avg_load <= self._low_avg_load_threshold:
            adaptive_scale = self._low_load_boost
        elif avg_load >= self._high_avg_load_threshold:
            adaptive_scale = self._high_load_dampen

        dynamic_max_create = self._max_create_per_tick
        if (
            self._low_load_max_create_per_tick > 0
            and avg_load <= self._low_load_max_create_threshold
        ):
            if dynamic_max_create <= 0:
                dynamic_max_create = self._low_load_max_create_per_tick
            else:
                dynamic_max_create = min(dynamic_max_create, self._low_load_max_create_per_tick)

        desired: dict[str, int] = {}
        for function_node in sorted(tracked):
            load_recent = int(load_recent_by_fn.get(function_node, 0))
            lambda_hat = float(load_recent) / float(self._load_window_sec)
            mu_hat = max(self._min_speed_req_per_sec, self._effective_speed(function_node=function_node))
            service_p99_ms = self._effective_service_p99_ms(function_node=function_node)
            wait_target_ms = max(1.0, self._latency_target_ms - service_p99_ms)
            hetero_penalty = max(1.0, service_p99_ms / max(1.0, self._effective_service_mean_ms(function_node)))
            mu_effective = max(self._min_speed_req_per_sec, mu_hat / hetero_penalty)

            desired_model = self._solve_required_containers(
                lambda_rate=lambda_hat,
                mu_rate=mu_effective,
                wait_target_ms=wait_target_ms,
            )
            desired_scaled = (
                int(math.ceil(float(desired_model) * self._desired_scale * adaptive_scale))
                if desired_model > 0
                else 0
            )
            if load_recent > 0 and self._min_desired_when_active > 0:
                desired_scaled = max(desired_scaled, self._min_desired_when_active)
            desired_count = max(self._min_replicas, desired_scaled)
            if self._max_replicas > 0:
                desired_count = min(desired_count, self._max_replicas)
            if self._max_desired_cap > 0:
                desired_count = min(desired_count, self._max_desired_cap)
            if desired_count > 0:
                desired[function_node] = desired_count

        # Optional top-k shaping retained for compatibility.
        if self._topk_ratio < 1.0 and desired:
            candidates = [
                fn for fn in desired if int(desired.get(fn, 0)) > int(ready_pool_by_function.get(fn, 0))
            ]
            if candidates:
                keep_k = max(1, int(math.ceil(len(candidates) * self._topk_ratio)))
                ranked = sorted(
                    candidates,
                    key=lambda fn: (
                        int(load_recent_by_fn.get(fn, 0)),
                        int(desired.get(fn, 0)) - int(ready_pool_by_function.get(fn, 0)),
                        str(fn),
                    ),
                    reverse=True,
                )
                keep_set = set(ranked[:keep_k])
                for fn in candidates:
                    if fn in keep_set:
                        continue
                    desired[fn] = min(int(desired.get(fn, 0)), int(ready_pool_by_function.get(fn, 0)))
                desired = {k: v for k, v in desired.items() if int(v) > 0}

        self._desired_by_function = desired
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))

        plans: list[PrewarmPlan | ScaleDownPlan] = []
        plan_functions = set(desired.keys()) | set(ready_pool_by_function.keys()) | set(self._inflight_by_function.keys())
        for function_node in sorted(plan_functions):
            desired_count = int(desired.get(function_node, 0))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))
            inflight = max(0, int(self._inflight_by_function.get(function_node, 0)))
            effective_pool = int(ready) + int(math.floor(float(inflight) * self._inflight_credit))
            to_create = max(0, int(desired_count) - effective_pool)
            to_remove = min(idle, max(0, ready + inflight - int(desired_count)))
            if to_create > 0 and self._scale_cooldown_sec > 0:
                last_scale = int(self._last_scale_sec_by_function.get(function_node, -10**9))
                if int(timestamp_sec) - last_scale < self._scale_cooldown_sec:
                    to_create = 0
            if to_create > 0 and dynamic_max_create > 0:
                to_create = min(to_create, dynamic_max_create)
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
                self._last_scale_sec_by_function[function_node] = int(timestamp_sec)
            elif to_remove > 0:
                plans.append(ScaleDownPlan(function_node=function_node, count=to_remove))
                self._scale_down_requested += to_remove
                self._last_scale_sec_by_function[function_node] = int(timestamp_sec)
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
        return {
            "type": self.name,
            "mode": "model_driven_mm_c_reconcile",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "latency_target_ms": self._latency_target_ms,
                "load_window_sec": self._load_window_sec,
                "speed_ewma_alpha": self._speed_ewma_alpha,
                "min_speed_req_per_sec": self._min_speed_req_per_sec,
                "min_samples": self._min_samples,
                "default_exec_ms": self._default_exec_ms,
                "desired_scale": self._desired_scale,
                "min_desired_when_active": self._min_desired_when_active,
                "topk_ratio": self._topk_ratio,
                "low_load_boost": self._low_load_boost,
                "high_load_dampen": self._high_load_dampen,
                "low_avg_load_threshold": self._low_avg_load_threshold,
                "high_avg_load_threshold": self._high_avg_load_threshold,
                "inflight_credit": self._inflight_credit,
                "scale_cooldown_sec": self._scale_cooldown_sec,
                "max_desired_cap": self._max_desired_cap,
                "max_create_per_tick": self._max_create_per_tick,
                "low_load_max_create_per_tick": self._low_load_max_create_per_tick,
                "low_load_max_create_threshold": self._low_load_max_create_threshold,
                "min_replicas": self._min_replicas,
                "max_replicas": self._max_replicas,
            },
            "tracked_functions": len(set(self._arrivals_by_function.keys()) | set(self._speed_by_function.keys())),
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

    def _trim_arrivals(self, *, function_node: str, now_ms: int) -> None:
        arrivals = self._arrivals_by_function.get(function_node)
        if arrivals is None:
            return
        threshold = now_ms - self._load_window_ms
        while arrivals and arrivals[0] < threshold:
            arrivals.popleft()
        if not arrivals:
            self._arrivals_by_function.pop(function_node, None)

    def _effective_speed(self, *, function_node: str) -> float:
        if self._samples_by_function.get(function_node, 0) < self._min_samples:
            return self._default_speed_req_per_sec
        observed = self._speed_by_function.get(function_node, self._default_speed_req_per_sec)
        return max(self._min_speed_req_per_sec, observed)

    def _effective_service_mean_ms(self, function_node: str) -> float:
        values = self._exec_samples_ms_by_function.get(function_node)
        if not values:
            return self._default_exec_ms
        return float(sum(values) / len(values))

    def _effective_service_p99_ms(self, *, function_node: str) -> float:
        values = self._exec_samples_ms_by_function.get(function_node)
        if not values:
            return self._default_exec_ms
        ordered = sorted(values)
        idx = int(math.ceil(0.99 * len(ordered)) - 1)
        idx = max(0, min(idx, len(ordered) - 1))
        return float(max(1, int(ordered[idx])))

    def _solve_required_containers(
        self,
        *,
        lambda_rate: float,
        mu_rate: float,
        wait_target_ms: float,
    ) -> int:
        if lambda_rate <= self._eps:
            return 0
        mu = max(self._eps, float(mu_rate))
        wait_t_sec = max(1e-6, float(wait_target_ms) / 1000.0)
        lower = max(1, int(math.ceil(lambda_rate / mu)))
        lower = max(lower, self._min_replicas)
        upper = self._max_replicas if self._max_replicas > 0 else max(lower, lower + 128)

        for c in range(lower, upper + 1):
            p_ok = self._prob_wait_within_t(lambda_rate=lambda_rate, mu_rate=mu, c=c, wait_t_sec=wait_t_sec)
            if p_ok >= 0.99:
                return c
        return upper

    def _prob_wait_within_t(
        self,
        *,
        lambda_rate: float,
        mu_rate: float,
        c: int,
        wait_t_sec: float,
    ) -> float:
        if c <= 0:
            return 0.0
        r = float(lambda_rate) / float(mu_rate)
        rho = float(lambda_rate) / float(c * mu_rate)
        if rho >= 1.0:
            return 0.0

        p0 = self._mmc_p0(r=r, c=c, rho=rho)
        if p0 <= 0.0:
            return 0.0

        limit_n = int(math.floor(wait_t_sec * float(c) * float(mu_rate) + float(c) - 1.0))
        if limit_n < 0:
            return 0.0

        # Sum 0..min(limit, c-1)
        up_to = min(limit_n, c - 1)
        cumulative = 0.0
        for n in range(0, up_to + 1):
            cumulative += ((r**n) / float(math.factorial(n))) * p0

        if limit_n < c:
            return float(max(0.0, min(1.0, cumulative)))

        # Tail from c..limit is geometric with ratio rho.
        p_c = ((r**c) / float(math.factorial(c))) * p0
        k = limit_n - c
        tail = p_c * ((1.0 - (rho ** (k + 1))) / max(self._eps, (1.0 - rho)))
        return float(max(0.0, min(1.0, cumulative + tail)))

    @staticmethod
    def _mmc_p0(*, r: float, c: int, rho: float) -> float:
        acc = 0.0
        for n in range(0, c):
            acc += (r**n) / float(math.factorial(n))
        tail = (r**c) / (float(math.factorial(c)) * max(1e-12, (1.0 - rho)))
        denom = acc + tail
        if denom <= 0.0:
            return 0.0
        return 1.0 / denom

    def _decrease_inflight(self, function_node: str) -> None:
        current = int(self._inflight_by_function.get(function_node, 0))
        if current <= 1:
            self._inflight_by_function.pop(function_node, None)
        else:
            self._inflight_by_function[function_node] = current - 1

    @property
    def _eps(self) -> float:
        return 1e-9
