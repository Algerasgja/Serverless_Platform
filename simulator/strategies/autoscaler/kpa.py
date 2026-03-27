from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from simulator.strategies.autoscaler.base import AutoscalerStrategy, PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


class KpaAutoscaler(AutoscalerStrategy):
    """Knative-style concurrency autoscaler (panic/stable with platform-compatible semantics)."""

    name = "kpa_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        target_concurrency: float,
        stable_window_sec: int,
        panic_window_sec: int,
        panic_threshold: float,
        min_replicas: int,
        max_replicas: int,
        target_utilization: float = 0.7,
        use_target_utilization: bool = True,
        panic_min_hold_sec: int = 6,
        panic_exit_streak_sec: int = 60,
        max_scale_up_step: int = 0,
    ) -> None:
        del templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._target_concurrency = max(0.1, float(target_concurrency))
        self._target_utilization = min(1.0, max(0.05, float(target_utilization)))
        self._use_target_utilization = bool(use_target_utilization)
        self._stable_window_sec = max(1, int(stable_window_sec))
        self._panic_window_sec = max(1, min(int(panic_window_sec), self._stable_window_sec))
        self._panic_threshold = max(1.0, float(panic_threshold))
        self._min_replicas = max(0, int(min_replicas))
        self._max_replicas = max(0, int(max_replicas))
        if self._max_replicas > 0 and self._max_replicas < self._min_replicas:
            self._max_replicas = self._min_replicas
        self._panic_min_hold_sec = max(0, int(panic_min_hold_sec))
        self._panic_exit_streak_sec = max(1, int(panic_exit_streak_sec))
        self._max_scale_up_step = max(0, int(max_scale_up_step))

        self._inflight_by_function: dict[str, int] = defaultdict(int)
        self._history_by_function: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        self._desired_by_function: dict[str, int] = {}
        self._mode_by_function: dict[str, str] = {}

        self._panic_active_by_function: dict[str, bool] = defaultdict(bool)
        self._panic_enter_sec_by_function: dict[str, int] = {}
        self._panic_peak_desired_by_function: dict[str, int] = defaultdict(int)
        self._panic_below_streak_sec_by_function: dict[str, int] = defaultdict(int)
        self._panic_enter_events = 0
        self._panic_exit_events = 0

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
        del request_id, um, timestamp_ms, true_future_path
        self._inflight_by_function[function_node] += 1

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
        del request_id, um, timestamp_ms, execution_ms, cold_start_ms, transfer_ms, prefix
        current = self._inflight_by_function.get(function_node, 0)
        if current <= 1:
            self._inflight_by_function.pop(function_node, None)
        else:
            self._inflight_by_function[function_node] = current - 1

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del request_id, status, timestamp_ms
        return None

    def on_tick(
        self,
        *,
        timestamp_sec: int,
        timestamp_ms: int,
        ready_pool_by_function: dict[str, int],
        idle_pool_by_function: dict[str, int] | None = None,
    ) -> list[PrewarmPlan | ScaleDownPlan]:
        del timestamp_ms
        if idle_pool_by_function is None:
            idle_pool_by_function = ready_pool_by_function
        self._record_second_sample(timestamp_sec)
        if timestamp_sec % self._sync_period_sec != 0:
            return []

        self._reconcile_rounds += 1
        desired = self._compute_desired_by_function(timestamp_sec=timestamp_sec)
        self._desired_by_function = desired
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))

        plans: list[PrewarmPlan | ScaleDownPlan] = []
        all_functions = set(desired.keys()) | set(self._inflight_by_function.keys()) | set(ready_pool_by_function.keys())
        for function_node in sorted(all_functions):
            desired_count = int(desired.get(function_node, 0))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))
            inflight = max(0, int(self._inflight_by_function.get(function_node, 0)))
            current_capacity = ready + inflight
            to_create = max(0, desired_count - current_capacity)
            to_remove = min(idle, max(0, current_capacity - desired_count))
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
            elif to_remove > 0:
                plans.append(ScaleDownPlan(function_node=function_node, count=to_remove))
                self._scale_down_requested += to_remove
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
        mode_counts = {"stable": 0, "panic": 0}
        for mode in self._mode_by_function.values():
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        return {
            "type": self.name,
            "mode": "knative_panic_stable_reconcile",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "target_concurrency": self._target_concurrency,
                "target_utilization": self._target_utilization,
                "use_target_utilization": self._use_target_utilization,
                "effective_target_concurrency": self._effective_target_concurrency(),
                "stable_window_sec": self._stable_window_sec,
                "panic_window_sec": self._panic_window_sec,
                "panic_threshold": self._panic_threshold,
                "panic_min_hold_sec": self._panic_min_hold_sec,
                "panic_exit_streak_sec": self._panic_exit_streak_sec,
                "max_scale_up_step": self._max_scale_up_step,
                "min_replicas": self._min_replicas,
                "max_replicas": self._max_replicas,
            },
            "mode_counts": mode_counts,
            "panic_enter_events": self._panic_enter_events,
            "panic_exit_events": self._panic_exit_events,
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

    def _effective_target_concurrency(self) -> float:
        if not self._use_target_utilization:
            return self._target_concurrency
        return max(0.1, self._target_concurrency * self._target_utilization)

    def _record_second_sample(self, timestamp_sec: int) -> None:
        all_functions = (
            set(self._history_by_function.keys())
            | set(self._inflight_by_function.keys())
            | set(self._desired_by_function.keys())
            | set(self._panic_active_by_function.keys())
        )
        retention = max(self._stable_window_sec, self._panic_window_sec, self._panic_exit_streak_sec) * 4
        min_keep_sec = timestamp_sec - retention

        for function_node in all_functions:
            history = self._history_by_function[function_node]
            history.append((timestamp_sec, max(0, int(self._inflight_by_function.get(function_node, 0)))))
            while history and history[0][0] < min_keep_sec:
                history.popleft()

    def _compute_desired_by_function(self, *, timestamp_sec: int) -> dict[str, int]:
        desired: dict[str, int] = {}
        all_functions = (
            set(self._history_by_function.keys())
            | set(self._inflight_by_function.keys())
            | set(self._desired_by_function.keys())
            | set(self._panic_active_by_function.keys())
        )
        eff_target = self._effective_target_concurrency()

        for function_node in all_functions:
            stable_avg = self._window_avg(
                function_node=function_node,
                now_sec=timestamp_sec,
                window_sec=self._stable_window_sec,
            )
            panic_avg = self._window_avg(
                function_node=function_node,
                now_sec=timestamp_sec,
                window_sec=self._panic_window_sec,
            )
            if math.isnan(stable_avg):
                stable_avg = 0.0
            if math.isnan(panic_avg):
                panic_avg = 0.0

            stable_desired = math.ceil(stable_avg / eff_target)
            panic_desired = math.ceil(panic_avg / eff_target)

            panic_trigger = panic_avg > (eff_target * self._panic_threshold)
            panic_active = bool(self._panic_active_by_function.get(function_node, False))

            if panic_trigger:
                if not panic_active:
                    panic_active = True
                    self._panic_enter_events += 1
                    self._panic_enter_sec_by_function[function_node] = int(timestamp_sec)
                    self._panic_below_streak_sec_by_function[function_node] = 0
                self._panic_peak_desired_by_function[function_node] = max(
                    int(self._panic_peak_desired_by_function.get(function_node, 0)),
                    int(panic_desired),
                    int(stable_desired),
                    int(self._min_replicas),
                )
                self._panic_below_streak_sec_by_function[function_node] = 0
            elif panic_active:
                self._panic_below_streak_sec_by_function[function_node] = (
                    int(self._panic_below_streak_sec_by_function.get(function_node, 0)) + self._sync_period_sec
                )
                enter_sec = int(self._panic_enter_sec_by_function.get(function_node, timestamp_sec))
                hold_elapsed = (timestamp_sec - enter_sec) >= self._panic_min_hold_sec
                below_elapsed = (
                    int(self._panic_below_streak_sec_by_function.get(function_node, 0))
                    >= self._panic_exit_streak_sec
                )
                if hold_elapsed and below_elapsed:
                    panic_active = False
                    self._panic_exit_events += 1
                    self._panic_enter_sec_by_function.pop(function_node, None)
                    self._panic_peak_desired_by_function.pop(function_node, None)
                    self._panic_below_streak_sec_by_function.pop(function_node, None)

            if panic_active:
                peak = int(self._panic_peak_desired_by_function.get(function_node, 0))
                desired_count = max(self._min_replicas, stable_desired, peak)
                self._mode_by_function[function_node] = "panic"
                self._panic_active_by_function[function_node] = True
            else:
                desired_count = max(self._min_replicas, stable_desired)
                self._mode_by_function[function_node] = "stable"
                self._panic_active_by_function[function_node] = False

            if self._max_replicas > 0:
                desired_count = min(desired_count, self._max_replicas)

            if self._max_scale_up_step > 0:
                prev_desired = int(self._desired_by_function.get(function_node, 0))
                desired_count = min(desired_count, prev_desired + self._max_scale_up_step)

            if desired_count > 0:
                desired[function_node] = desired_count

        return desired

    def _window_avg(self, *, function_node: str, now_sec: int, window_sec: int) -> float:
        history = self._history_by_function.get(function_node)
        if not history:
            return math.nan
        start = now_sec - window_sec + 1
        values = [count for sec, count in history if sec >= start]
        if not values:
            return math.nan
        return float(sum(values) / len(values))
