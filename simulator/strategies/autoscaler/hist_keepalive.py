from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


@dataclass
class WindowDecision:
    mode: str
    prewarm_window_sec: int
    keepalive_window_sec: int
    sample_count: int
    oob_ratio: float
    cv: float


class HistogramKeepalivePrewarmAutoscaler:
    """Hybrid histogram keepalive/prewarm policy (paper-inspired).

    Design mapping to the paper:
    - Range-limited histogram over per-function idle times (ITs), 1-minute bins.
    - Histogram mode: prewarm from IT head percentile, keepalive from IT tail percentile.
    - Conservative fallback when histogram is not representative.
    - Forecast fallback when OOB IT ratio is high (lightweight EWMA forecast).
    """

    name = "hist_keepalive_prewarm_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        keepalive_idle_sec: int,
        keepalive_min_replicas: int,
        prewarm_buffer: int,
        min_replicas: int,
        max_replicas: int,
        # Hybrid histogram knobs.
        bin_minutes: int = 1,
        range_minutes: int = 240,
        head_percentile: float = 0.05,
        tail_percentile: float = 0.99,
        margin_ratio: float = 0.10,
        min_samples: int = 20,
        cv_threshold: float = 0.25,
        oob_ratio_threshold: float = 0.30,
        history_retention_sec: int = 21600,
        # Forecast fallback knobs.
        forecast_margin_ratio: float = 0.15,
        forecast_alpha: float = 0.35,
        forecast_min_samples: int = 8,
    ) -> None:
        del templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._keepalive_idle_sec = max(1, int(keepalive_idle_sec))
        self._keepalive_min_replicas = max(0, int(keepalive_min_replicas))
        self._prewarm_buffer = max(0, int(prewarm_buffer))
        self._min_replicas = max(0, int(min_replicas))
        self._max_replicas = max(0, int(max_replicas))
        if self._max_replicas > 0 and self._max_replicas < self._min_replicas:
            self._max_replicas = self._min_replicas

        self._bin_sec = max(60, int(bin_minutes) * 60)
        self._range_sec = max(self._bin_sec, int(range_minutes) * 60)
        self._range_bins = max(1, int(math.ceil(self._range_sec / self._bin_sec)))
        self._head_percentile = min(0.50, max(0.0, float(head_percentile)))
        self._tail_percentile = min(0.999, max(0.5, float(tail_percentile)))
        if self._tail_percentile < self._head_percentile:
            self._tail_percentile = self._head_percentile
        self._margin_ratio = min(0.50, max(0.0, float(margin_ratio)))
        self._min_samples = max(2, int(min_samples))
        self._cv_threshold = max(0.0, float(cv_threshold))
        self._oob_ratio_threshold = min(1.0, max(0.0, float(oob_ratio_threshold)))
        self._history_retention_sec = max(self._range_sec, int(history_retention_sec))

        self._forecast_margin_ratio = min(0.50, max(0.0, float(forecast_margin_ratio)))
        self._forecast_alpha = min(1.0, max(0.01, float(forecast_alpha)))
        self._forecast_min_samples = max(2, int(forecast_min_samples))

        self._inflight_by_function: dict[str, int] = defaultdict(int)
        self._request_active_function: dict[str, str] = {}

        # Idle-start timestamp for currently idle functions.
        self._idle_start_sec_by_function: dict[str, int] = {}
        # Recent observed IT samples: (arrival_sec, idle_sec, is_oob).
        self._idle_samples_by_function: dict[str, deque[tuple[int, int, bool]]] = defaultdict(deque)
        self._idle_ewma_sec_by_function: dict[str, float] = {}

        self._desired_by_function: dict[str, int] = {}
        self._last_windows_by_function: dict[str, WindowDecision] = {}

        self._reconcile_rounds = 0
        self._desired_peak_total = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._capacity_blocked_creations = 0
        self._scale_down_requested = 0
        self._mode_counts: dict[str, int] = defaultdict(int)

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
        now_sec = max(0, int(timestamp_ms // 1000))

        previous = self._request_active_function.get(request_id)
        if previous and previous != function_node:
            self._finish_step(previous, now_sec)

        self._request_active_function[request_id] = function_node

        # New arrival for this function: close current idle interval if any.
        if self._inflight_by_function.get(function_node, 0) <= 0:
            idle_start = self._idle_start_sec_by_function.pop(function_node, None)
            if idle_start is not None and now_sec > idle_start:
                idle_sec = now_sec - idle_start
                self._observe_idle_interval(
                    function_node=function_node,
                    arrival_sec=now_sec,
                    idle_sec=idle_sec,
                )

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
        del um, execution_ms, cold_start_ms, transfer_ms, prefix
        now_sec = max(0, int(timestamp_ms // 1000))
        active_fn = self._request_active_function.get(request_id)
        if active_fn == function_node:
            self._request_active_function.pop(request_id, None)
        self._finish_step(function_node, now_sec)

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del status
        now_sec = max(0, int(timestamp_ms // 1000))
        active_fn = self._request_active_function.pop(request_id, None)
        if active_fn:
            self._finish_step(active_fn, now_sec)

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
        if timestamp_sec % self._sync_period_sec != 0:
            return []

        self._reconcile_rounds += 1
        tracked_functions = (
            set(self._inflight_by_function.keys())
            | set(self._idle_start_sec_by_function.keys())
            | set(self._idle_samples_by_function.keys())
            | set(ready_pool_by_function.keys())
        )

        desired: dict[str, int] = {}
        windows: dict[str, WindowDecision] = {}
        for function_node in sorted(tracked_functions):
            decision = self._compute_windows(function_node=function_node, now_sec=timestamp_sec)
            windows[function_node] = decision
            self._mode_counts[decision.mode] += 1
            inflight = max(0, int(self._inflight_by_function.get(function_node, 0)))
            desired_count = self._compute_desired(
                function_node=function_node,
                now_sec=timestamp_sec,
                inflight=inflight,
                prewarm_window_sec=decision.prewarm_window_sec,
                keepalive_window_sec=decision.keepalive_window_sec,
            )
            if desired_count > 0:
                desired[function_node] = desired_count

        plans: list[PrewarmPlan | ScaleDownPlan] = []
        plan_functions = set(tracked_functions) | set(desired.keys())
        for function_node in sorted(plan_functions):
            desired_count = max(0, int(desired.get(function_node, 0)))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))
            inflight = max(0, int(self._inflight_by_function.get(function_node, 0)))
            current_total = ready + inflight
            to_create = max(0, desired_count - current_total)
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
                continue
            to_remove = min(idle, max(0, current_total - desired_count))
            if to_remove > 0:
                plans.append(ScaleDownPlan(function_node=function_node, count=to_remove))
                self._scale_down_requested += to_remove

        self._desired_by_function = desired
        self._last_windows_by_function = windows
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
            "mode": "hybrid_histogram_keepalive_prewarm",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "bin_minutes": int(self._bin_sec / 60),
                "range_minutes": int(self._range_sec / 60),
                "head_percentile": self._head_percentile,
                "tail_percentile": self._tail_percentile,
                "margin_ratio": self._margin_ratio,
                "min_samples": self._min_samples,
                "cv_threshold": self._cv_threshold,
                "oob_ratio_threshold": self._oob_ratio_threshold,
                "history_retention_sec": self._history_retention_sec,
                "forecast_margin_ratio": self._forecast_margin_ratio,
                "forecast_alpha": self._forecast_alpha,
                "forecast_min_samples": self._forecast_min_samples,
                "keepalive_min_replicas": self._keepalive_min_replicas,
                "keepalive_idle_sec_compat": self._keepalive_idle_sec,
                "prewarm_buffer": self._prewarm_buffer,
                "min_replicas": self._min_replicas,
                "max_replicas": self._max_replicas,
            },
            "reconcile_rounds": self._reconcile_rounds,
            "desired_peak_total": self._desired_peak_total,
            "final_desired_by_function": dict(sorted(self._desired_by_function.items())),
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "capacity_blocked_creations": self._capacity_blocked_creations,
            "scale_down_requested": self._scale_down_requested,
            "mode_counts": dict(sorted(self._mode_counts.items())),
            "tracked_functions": len(self._last_windows_by_function),
        }

    def _finish_step(self, function_node: str, now_sec: int) -> None:
        current = self._inflight_by_function.get(function_node, 0)
        if current <= 1:
            self._inflight_by_function.pop(function_node, None)
            # Function becomes idle now.
            self._idle_start_sec_by_function.setdefault(function_node, now_sec)
            return
        self._inflight_by_function[function_node] = current - 1

    def _observe_idle_interval(self, *, function_node: str, arrival_sec: int, idle_sec: int) -> None:
        sample_q = self._idle_samples_by_function[function_node]
        is_oob = idle_sec > self._range_sec
        sample_q.append((arrival_sec, int(idle_sec), bool(is_oob)))

        # EWMA for forecast fallback.
        if function_node not in self._idle_ewma_sec_by_function:
            self._idle_ewma_sec_by_function[function_node] = float(idle_sec)
        else:
            prev = self._idle_ewma_sec_by_function[function_node]
            self._idle_ewma_sec_by_function[function_node] = (
                (1.0 - self._forecast_alpha) * prev + self._forecast_alpha * float(idle_sec)
            )

        min_keep_sec = arrival_sec - self._history_retention_sec
        while sample_q and sample_q[0][0] < min_keep_sec:
            sample_q.popleft()

    def _compute_windows(self, *, function_node: str, now_sec: int) -> WindowDecision:
        self._trim_samples(function_node=function_node, now_sec=now_sec)
        samples = list(self._idle_samples_by_function.get(function_node, ()))
        sample_count = len(samples)
        if sample_count <= 0:
            return WindowDecision(
                mode="conservative",
                prewarm_window_sec=0,
                keepalive_window_sec=self._range_sec,
                sample_count=0,
                oob_ratio=0.0,
                cv=0.0,
            )

        oob_count = sum(1 for _, _, is_oob in samples if is_oob)
        oob_ratio = float(oob_count) / float(sample_count)
        in_bounds = [idle for _, idle, is_oob in samples if not is_oob]

        if oob_ratio > self._oob_ratio_threshold and sample_count >= self._forecast_min_samples:
            prewarm_sec, keepalive_sec = self._forecast_windows(function_node=function_node)
            return WindowDecision(
                mode="forecast",
                prewarm_window_sec=prewarm_sec,
                keepalive_window_sec=keepalive_sec,
                sample_count=sample_count,
                oob_ratio=oob_ratio,
                cv=0.0,
            )

        if len(in_bounds) < self._min_samples:
            return WindowDecision(
                mode="conservative",
                prewarm_window_sec=0,
                keepalive_window_sec=self._range_sec,
                sample_count=sample_count,
                oob_ratio=oob_ratio,
                cv=0.0,
            )

        counts = self._hist_counts(in_bounds)
        cv = self._hist_cv(counts)
        if cv < self._cv_threshold:
            return WindowDecision(
                mode="conservative",
                prewarm_window_sec=0,
                keepalive_window_sec=self._range_sec,
                sample_count=sample_count,
                oob_ratio=oob_ratio,
                cv=cv,
            )

        head_sec = self._hist_head_sec(counts)
        tail_sec = self._hist_tail_sec(counts)
        # Margin: move prewarm earlier, extend keepalive end.
        prewarm_start_sec = max(0, int(math.floor(head_sec * (1.0 - self._margin_ratio))))
        keepalive_end_sec = int(math.ceil(tail_sec * (1.0 + self._margin_ratio)))
        keepalive_sec = max(0, keepalive_end_sec - prewarm_start_sec)
        return WindowDecision(
            mode="histogram",
            prewarm_window_sec=prewarm_start_sec,
            keepalive_window_sec=keepalive_sec,
            sample_count=sample_count,
            oob_ratio=oob_ratio,
            cv=cv,
        )

    def _compute_desired(
        self,
        *,
        function_node: str,
        now_sec: int,
        inflight: int,
        prewarm_window_sec: int,
        keepalive_window_sec: int,
    ) -> int:
        target = float(max(0, inflight))
        if inflight <= 0:
            idle_start = self._idle_start_sec_by_function.get(function_node)
            if idle_start is not None:
                elapsed = max(0, now_sec - idle_start)
                active_start = max(0, int(prewarm_window_sec))
                active_end = active_start + max(0, int(keepalive_window_sec))
                if active_start <= elapsed <= active_end:
                    target = max(
                        target,
                        float(self._keepalive_min_replicas + self._prewarm_buffer),
                    )

        desired = int(math.ceil(target))
        desired = max(desired, self._min_replicas)
        if self._max_replicas > 0:
            desired = min(desired, self._max_replicas)
        if desired <= 0:
            return 0
        return desired

    def _trim_samples(self, *, function_node: str, now_sec: int) -> None:
        samples = self._idle_samples_by_function.get(function_node)
        if not samples:
            return
        min_keep_sec = now_sec - self._history_retention_sec
        while samples and samples[0][0] < min_keep_sec:
            samples.popleft()

    def _hist_counts(self, in_bounds: list[int]) -> list[int]:
        counts = [0 for _ in range(self._range_bins)]
        for idle_sec in in_bounds:
            idx = int(idle_sec // self._bin_sec)
            if idx < 0:
                idx = 0
            if idx >= self._range_bins:
                idx = self._range_bins - 1
            counts[idx] += 1
        return counts

    def _hist_head_sec(self, counts: list[int]) -> int:
        idx = self._percentile_bin_index(counts, self._head_percentile)
        # Round to lower edge (head).
        return max(0, idx * self._bin_sec)

    def _hist_tail_sec(self, counts: list[int]) -> int:
        idx = self._percentile_bin_index(counts, self._tail_percentile)
        # Round to upper edge (tail).
        return min(self._range_sec, (idx + 1) * self._bin_sec)

    @staticmethod
    def _percentile_bin_index(counts: list[int], p: float) -> int:
        total = sum(counts)
        if total <= 0:
            return 0
        target = p * float(total)
        cumulative = 0.0
        for idx, cnt in enumerate(counts):
            cumulative += float(cnt)
            if cumulative >= target:
                return idx
        return max(0, len(counts) - 1)

    @staticmethod
    def _hist_cv(counts: list[int]) -> float:
        if not counts:
            return 0.0
        vals = [float(v) for v in counts]
        mean_v = sum(vals) / len(vals)
        if mean_v <= 1e-12:
            return 0.0
        var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        std = math.sqrt(max(0.0, var))
        return std / mean_v

    def _forecast_windows(self, *, function_node: str) -> tuple[int, int]:
        pred = float(self._idle_ewma_sec_by_function.get(function_node, self._range_sec))
        pred = max(float(self._bin_sec), pred)
        prewarm = int(max(0.0, pred * (1.0 - self._forecast_margin_ratio)))
        keepalive = int(max(float(self._bin_sec), pred * (2.0 * self._forecast_margin_ratio)))
        return prewarm, keepalive
