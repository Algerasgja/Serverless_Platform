from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from simulator.strategies.autoscaler.base import AutoscalerStrategy, PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


class HptdAutoscaler(AutoscalerStrategy):
    """Hawkes Process Temperature-Driven autoscaler (paper-aligned core)."""

    name = "hptd_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        time_granularity_sec: int,
        wcall_t: int,
        whistory_t: int,
        wchange_t: int,
        alpha: float,
        beta: float,
        mu_floor: float,
        std_floor: float,
        temp_floor: float,
        scale_max_step: int,
        min_replicas: int,
        max_replicas: int,
    ) -> None:
        del templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._t_sec = max(1, int(time_granularity_sec))
        self._wcall_sec = max(1, int(wcall_t)) * self._t_sec
        self._whistory_sec = max(2, int(whistory_t)) * self._t_sec
        self._wchange_sec = max(1, int(wchange_t)) * self._t_sec
        if self._wchange_sec >= self._whistory_sec:
            self._wchange_sec = max(1, self._whistory_sec // 2)
        # Keep enough request history for both Hawkes excitation and mu estimation.
        self._event_keep_sec = max(self._wcall_sec, self._whistory_sec)

        self._alpha = float(max(0.0, alpha))
        self._beta = float(max(1e-9, beta))
        self._mu_floor = float(max(1e-9, mu_floor))
        self._std_floor = float(max(0.0, std_floor))
        self._temp_floor = float(max(1e-9, temp_floor))
        self._scale_max_step = max(1, int(scale_max_step))
        self._min_replicas = max(0, int(min_replicas))
        self._max_replicas = max(0, int(max_replicas))
        if self._max_replicas > 0 and self._max_replicas < self._min_replicas:
            self._max_replicas = self._min_replicas

        self._call_times_by_function: dict[str, deque[int]] = defaultdict(deque)
        self._temp_history_by_function: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._current_temp_by_function: dict[str, float] = {}
        self._inflight_by_function: dict[str, int] = defaultdict(int)
        self._request_active_function: dict[str, str] = {}
        self._request_active_um: dict[str, str] = {}
        self._desired_by_function: dict[str, int] = {}

        self._trigger_count_by_function: dict[str, int] = defaultdict(int)
        self._scale_events_by_function: dict[str, int] = defaultdict(int)
        self._trigger_count_total = 0
        self._scale_events_total = 0

        self._reconcile_rounds = 0
        self._desired_peak_total = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._capacity_blocked_creations = 0
        self._prewarm_created_by_function: dict[str, int] = defaultdict(int)
        self._prewarm_consumed_by_function: dict[str, int] = defaultdict(int)
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
        del true_future_path
        previous = self._request_active_function.get(request_id)
        if previous and previous != function_node:
            self._decrease_inflight(previous)
        self._request_active_function[request_id] = function_node
        self._request_active_um[request_id] = um
        self._inflight_by_function[function_node] += 1
        events = self._call_times_by_function[function_node]
        events.append(int(timestamp_ms))
        self._trim_call_times(function_node=function_node, now_ms=int(timestamp_ms))

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
        del um, timestamp_ms, execution_ms, cold_start_ms, transfer_ms, prefix
        self._decrease_inflight(function_node)
        active_fn = self._request_active_function.get(request_id)
        if active_fn == function_node:
            self._request_active_function.pop(request_id, None)
            self._request_active_um.pop(request_id, None)

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del status, timestamp_ms
        active_fn = self._request_active_function.pop(request_id, None)
        self._request_active_um.pop(request_id, None)
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
        now_ms = int(timestamp_ms)
        self._update_temperatures(
            now_sec=int(timestamp_sec),
            now_ms=now_ms,
            ready_pool_by_function=ready_pool_by_function,
        )
        if timestamp_sec % self._sync_period_sec != 0:
            return []

        self._reconcile_rounds += 1
        desired: dict[str, int] = {}

        tracked_functions = (
            set(self._current_temp_by_function.keys())
            | set(self._inflight_by_function.keys())
            | set(ready_pool_by_function.keys())
        )
        for function_node in sorted(tracked_functions):
            history = [temp for _, temp in self._temp_history_by_function.get(function_node, ())]
            ready_now = max(0, int(ready_pool_by_function.get(function_node, 0)))
            inflight_now = max(0, int(self._inflight_by_function.get(function_node, 0)))
            ccurrent = ready_now + inflight_now
            desired_count = ccurrent
            if len(history) >= 2:
                change_n = min(len(history), self._wchange_sec)
                change_vals = history[-change_n:]
                baseline_vals = history[:-change_n] if len(history) > change_n else history
                avg_change = sum(change_vals) / len(change_vals)
                avg_history = sum(baseline_vals) / len(baseline_vals)
                tthreshold = max(self._std_floor, _stddev(baseline_vals))
                delta_t = avg_change - avg_history
                if delta_t > tthreshold:
                    self._trigger_count_total += 1
                    self._trigger_count_by_function[function_node] += 1
                    denom = max(self._temp_floor, abs(avg_history))
                    delta_ratio = (delta_t - tthreshold) / denom
                    ceff = max(1, ccurrent)
                    delta_c = max(0, int(math.ceil(delta_ratio * ceff)))
                    delta_c = min(delta_c, self._scale_max_step)
                    if delta_c > 0:
                        self._scale_events_total += 1
                        self._scale_events_by_function[function_node] += 1
                    desired_count = ccurrent + delta_c

            desired_count = max(self._min_replicas, desired_count)
            if self._max_replicas > 0:
                desired_count = min(self._max_replicas, desired_count)
            if desired_count > 0:
                desired[function_node] = desired_count

        plans: list[PrewarmPlan | ScaleDownPlan] = []
        for function_node in sorted(tracked_functions):
            desired_count = int(desired.get(function_node, 0))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))
            inflight = max(0, int(self._inflight_by_function.get(function_node, 0)))
            ccurrent = ready + inflight
            to_create = max(0, desired_count - ccurrent)
            to_remove = min(idle, max(0, ccurrent - desired_count))
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
            elif to_remove > 0:
                plans.append(ScaleDownPlan(function_node=function_node, count=to_remove))
                self._scale_down_requested += to_remove

        self._desired_by_function = desired
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))
        return plans

    def on_prewarm_create_result(
        self,
        *,
        function_node: str,
        success: bool,
        timestamp_ms: int,
        reason: str,
    ) -> None:
        del timestamp_ms, reason
        if success:
            self._prewarm_created += 1
            self._prewarm_created_by_function[function_node] += 1
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
        del request_id, container_id, timestamp_ms
        self._prewarm_consumed += 1
        self._prewarm_consumed_by_function[function_node] += 1

    def summary(self) -> dict[str, Any]:
        avg_temp = None
        if self._current_temp_by_function:
            avg_temp = sum(self._current_temp_by_function.values()) / len(self._current_temp_by_function)
        return {
            "type": self.name,
            "mode": "hawkes_temperature_delta_threshold",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "time_granularity_sec": self._t_sec,
                "wcall_t": int(self._wcall_sec / self._t_sec),
                "whistory_t": int(self._whistory_sec / self._t_sec),
                "wchange_t": int(self._wchange_sec / self._t_sec),
                "alpha": self._alpha,
                "beta": self._beta,
                "mu_floor": self._mu_floor,
                "std_floor": self._std_floor,
                "temp_floor": self._temp_floor,
                "scale_max_step": self._scale_max_step,
                "min_replicas": self._min_replicas,
                "max_replicas": self._max_replicas,
            },
            "tracked_functions": len(self._current_temp_by_function),
            "avg_temperature": avg_temp,
            "trigger_count_total": self._trigger_count_total,
            "scale_events_total": self._scale_events_total,
            "trigger_count_by_function": dict(sorted(self._trigger_count_by_function.items())),
            "scale_events_by_function": dict(sorted(self._scale_events_by_function.items())),
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

    def _update_temperatures(
        self,
        *,
        now_sec: int,
        now_ms: int,
        ready_pool_by_function: dict[str, int],
    ) -> None:
        tracked_functions = (
            set(self._call_times_by_function.keys())
            | set(self._inflight_by_function.keys())
            | set(ready_pool_by_function.keys())
            | set(self._temp_history_by_function.keys())
        )
        for function_node in tracked_functions:
            self._trim_call_times(function_node=function_node, now_ms=now_ms)
            self._trim_temp_history(function_node=function_node, now_sec=now_sec)

            call_times = self._call_times_by_function.get(function_node, ())
            history_call_count = sum(
                1 for ts in call_times if ts >= now_ms - self._whistory_sec * 1000
            )
            mu = max(self._mu_floor, float(history_call_count) / float(self._whistory_sec))

            self_excitation = 0.0
            for ts in call_times:
                if ts < now_ms - self._wcall_sec * 1000:
                    continue
                if ts >= now_ms:
                    continue
                dt_sec = max(0.0, (now_ms - ts) / 1000.0)
                self_excitation += self._alpha * math.exp(-self._beta * dt_sec)

            intensity = max(self._temp_floor, mu + self_excitation)
            temp = math.log(max(self._temp_floor, intensity))
            self._current_temp_by_function[function_node] = temp

            temp_history = self._temp_history_by_function[function_node]
            if temp_history and temp_history[-1][0] == now_sec:
                temp_history[-1] = (now_sec, temp)
            else:
                temp_history.append((now_sec, temp))
            self._trim_temp_history(function_node=function_node, now_sec=now_sec)

    def _trim_call_times(self, *, function_node: str, now_ms: int) -> None:
        events = self._call_times_by_function.get(function_node)
        if events is None:
            return
        threshold = now_ms - self._event_keep_sec * 1000
        while events and events[0] < threshold:
            events.popleft()
        if not events:
            self._call_times_by_function.pop(function_node, None)

    def _trim_temp_history(self, *, function_node: str, now_sec: int) -> None:
        hist = self._temp_history_by_function.get(function_node)
        if hist is None:
            return
        threshold = now_sec - self._whistory_sec
        while hist and hist[0][0] < threshold:
            hist.popleft()
        if not hist:
            self._temp_history_by_function.pop(function_node, None)
            self._current_temp_by_function.pop(function_node, None)

    def _decrease_inflight(self, function_node: str) -> None:
        current = int(self._inflight_by_function.get(function_node, 0))
        if current <= 1:
            self._inflight_by_function.pop(function_node, None)
        else:
            self._inflight_by_function[function_node] = current - 1


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = sum(values) / len(values)
    var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))
