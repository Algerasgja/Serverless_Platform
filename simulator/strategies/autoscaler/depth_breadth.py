from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


@dataclass
class FunctionExecStats:
    mean_ms: float = 0.0
    samples: int = 0


@dataclass
class EdgeTransferStats:
    mean_ms: float = 0.0
    samples: int = 0


@dataclass
class RequestState:
    um: str
    current_node: str


class DepthBreadthAutoscaler:
    """DBW: ConScale-style decision, topology full-cover candidates."""

    name = "depth_breadth_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        sched_eta_exec: float,
        sched_min_sec: int,
        sched_max_sec: int,
        horizon_alpha: float,
        default_exec_ms: float,
        default_trans_ms: float,
        horizon_boost: float = 1.0,
        desired_scale: float = 1.0,
        desired_ewma_alpha: float = 0.35,
        guard_buffer_min: int = 1,
        down_margin: int = 1,
        min_idle_age_sec: int = 5,
        down_cooldown_sec: int = 6,
        max_down_ratio: float = 0.15,
        osc_window_sec: int = 30,
        osc_trigger_count: int = 3,
    ) -> None:
        self._templates = templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._sched_eta_exec = max(1e-6, float(sched_eta_exec))
        self._sched_min_sec = max(1, int(sched_min_sec))
        self._sched_max_sec = max(self._sched_min_sec, int(sched_max_sec))
        self._horizon_alpha = max(1.0, float(horizon_alpha))
        self._horizon_boost = max(1.0, float(horizon_boost))
        self._desired_scale = max(1.0, float(desired_scale))
        self._default_exec_ms = max(1.0, float(default_exec_ms))
        self._default_trans_ms = max(0.0, float(default_trans_ms))

        # Compatibility knobs: preserved in config/summary, not used by decision path.
        self._desired_ewma_alpha = float(desired_ewma_alpha)
        self._guard_buffer_min = int(guard_buffer_min)
        self._down_margin = int(down_margin)
        self._min_idle_age_sec = int(min_idle_age_sec)
        self._down_cooldown_sec = int(down_cooldown_sec)
        self._max_down_ratio = float(max_down_ratio)
        self._osc_window_sec = int(osc_window_sec)
        self._osc_trigger_count = int(osc_trigger_count)

        self._function_stats: dict[str, FunctionExecStats] = {}
        self._edge_stats: dict[tuple[str, str], EdgeTransferStats] = {}
        self._request_states: dict[str, RequestState] = {}
        self._inflight_by_function: dict[str, int] = defaultdict(int)
        self._request_open_steps: set[str] = set()

        self._next_plan_sec = 0
        self._last_sched_sec = self._sync_period_sec
        self._window_horizon_ms = int(
            math.ceil(self._horizon_alpha * self._last_sched_sec * 1000.0 * self._horizon_boost)
        )

        self._reconcile_rounds = 0
        self._planning_rounds = 0
        self._candidate_in_window_total = 0
        self._desired_raw_peak_total = 0
        self._desired_peak_total = 0
        self._desired_raw_by_function: dict[str, int] = {}

        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._prewarm_created_by_function: dict[str, int] = defaultdict(int)
        self._prewarm_consumed_by_function: dict[str, int] = defaultdict(int)
        self._capacity_blocked_creations = 0
        self._scale_up_requested = 0
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
        del timestamp_ms, true_future_path
        self._request_states[request_id] = RequestState(um=um, current_node=function_node)
        self._inflight_by_function[function_node] += 1
        self._request_open_steps.add(request_id)

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
        del request_id, um, timestamp_ms, prefix
        if transfer_ms is None:
            return
        key = (src_node, dst_node)
        metric = self._edge_stats.get(key)
        value = float(max(0, int(transfer_ms)))
        if metric is None:
            self._edge_stats[key] = EdgeTransferStats(mean_ms=value, samples=1)
            return
        metric.samples += 1
        metric.mean_ms += (value - metric.mean_ms) / metric.samples

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
        metric = self._function_stats.get(function_node)
        value = float(max(1, int(execution_ms)))
        if metric is None:
            self._function_stats[function_node] = FunctionExecStats(mean_ms=value, samples=1)
        else:
            metric.samples += 1
            metric.mean_ms += (value - metric.mean_ms) / metric.samples
        self._decrease_inflight(function_node=function_node)
        self._request_open_steps.discard(request_id)

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del status, timestamp_ms
        state = self._request_states.get(request_id)
        if state is not None and request_id in self._request_open_steps:
            self._decrease_inflight(function_node=state.current_node)
            self._request_open_steps.discard(request_id)
        self._request_states.pop(request_id, None)

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
        if timestamp_sec < self._next_plan_sec:
            return []
        if timestamp_sec % self._sync_period_sec != 0:
            return []

        exec_scale_ms = self._execution_scale_ms()
        sched_sec = self._compute_sched_sec(exec_scale_ms)
        self._last_sched_sec = sched_sec
        self._window_horizon_ms = int(
            math.ceil(self._horizon_alpha * sched_sec * 1000.0 * self._horizon_boost)
        )
        self._next_plan_sec = timestamp_sec + sched_sec
        self._planning_rounds += 1
        self._reconcile_rounds += 1

        context_counts: dict[tuple[str, str], int] = defaultdict(int)
        for state in self._request_states.values():
            context_counts[(state.um, state.current_node)] += 1

        desired_by_function: dict[str, int] = defaultdict(int)
        candidate_in_window = 0
        for (um, current_node), n_active in context_counts.items():
            reachable = self._reachable_nodes(
                um=um,
                current_node=current_node,
                horizon_ms=self._window_horizon_ms,
            )
            if not reachable:
                continue
            candidate_in_window += len(reachable)
            for function_node in reachable:
                desired_by_function[function_node] += n_active

        scaled_desired_by_function = {
            function_node: int(math.ceil(float(desired_count) * self._desired_scale))
            for function_node, desired_count in desired_by_function.items()
            if desired_count > 0
        }
        self._candidate_in_window_total += candidate_in_window
        self._desired_raw_by_function = dict(scaled_desired_by_function)
        self._desired_raw_peak_total = max(self._desired_raw_peak_total, sum(scaled_desired_by_function.values()))
        self._desired_peak_total = max(self._desired_peak_total, sum(scaled_desired_by_function.values()))

        plans: list[PrewarmPlan | ScaleDownPlan] = []
        all_functions = sorted(set(scaled_desired_by_function) | set(ready_pool_by_function) | set(idle_pool_by_function))
        for function_node in all_functions:
            desired = max(0, int(scaled_desired_by_function.get(function_node, 0)))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))

            to_create = max(0, desired - ready)
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
                self._scale_up_requested += to_create

            # No strategy-side scale-down. Reclamation is handled by platform idle TTL.
            del idle

        return plans

    def on_prewarm_create_result(
        self,
        *,
        function_node: str,
        success: bool,
        timestamp_ms: int,
        reason: str,
    ) -> None:
        del reason, timestamp_ms
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
        return {
            "type": self.name,
            "mode": "dbw_full_cover_window_only",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "sched_eta_exec": self._sched_eta_exec,
                "sched_min_sec": self._sched_min_sec,
                "sched_max_sec": self._sched_max_sec,
                "horizon_alpha": self._horizon_alpha,
                "horizon_boost": self._horizon_boost,
                "desired_scale": self._desired_scale,
                # compatibility-only knobs
                "desired_ewma_alpha": self._desired_ewma_alpha,
                "guard_buffer_min": self._guard_buffer_min,
                "down_margin": self._down_margin,
                "min_idle_age_sec": self._min_idle_age_sec,
                "down_cooldown_sec": self._down_cooldown_sec,
                "max_down_ratio": self._max_down_ratio,
                "osc_window_sec": self._osc_window_sec,
                "osc_trigger_count": self._osc_trigger_count,
                "default_exec_ms": self._default_exec_ms,
                "default_trans_ms": self._default_trans_ms,
            },
            "planning_rounds": self._planning_rounds,
            "reconcile_rounds": self._reconcile_rounds,
            "last_sched_sec": self._last_sched_sec,
            "window_horizon_ms": self._window_horizon_ms,
            "current_horizon_ms": self._window_horizon_ms,
            "candidate_in_window_total": self._candidate_in_window_total,
            "desired_raw_peak_total": self._desired_raw_peak_total,
            "desired_smooth_peak_total": 0,
            "desired_peak_total": self._desired_peak_total,
            "final_desired_raw_by_function": dict(sorted(self._desired_raw_by_function.items())),
            "final_desired_by_function": dict(sorted(self._desired_raw_by_function.items())),
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "scale_up_requested": self._scale_up_requested,
            "scale_down_requested": self._scale_down_requested,
            # compatibility statistics kept as neutral values
            "scale_down_suppressed_cooldown": 0,
            "scale_down_suppressed_idle_age": 0,
            "scale_down_suppressed_hysteresis": 0,
            "scale_down_suppressed_health": 0,
            "scale_down_budget_limited": 0,
            "oscillation_events": 0,
            "avg_removed_idle_age_ms": 0.0,
            "capacity_blocked_creations": self._capacity_blocked_creations,
        }

    def _reachable_nodes(
        self,
        *,
        um: str,
        current_node: str,
        horizon_ms: int,
    ) -> set[str]:
        template = self._templates.get(um)
        if template is None:
            return set()
        transitions = template.transitions
        if current_node not in transitions:
            return set()

        result: set[str] = set()
        best_arrival: dict[str, float] = {}
        queue: deque[tuple[str, float]] = deque()
        queue.append((current_node, 0.0))

        while queue:
            node, elapsed = queue.popleft()
            outgoing = transitions.get(node, {})
            if not outgoing:
                continue
            for child in outgoing.keys():
                step = self._mu_exec_ms(node) + self._mu_trans_ms(node, child)
                arrival = elapsed + step
                if arrival > float(horizon_ms):
                    continue
                prev_best = best_arrival.get(child)
                if prev_best is not None and arrival >= prev_best:
                    continue
                best_arrival[child] = arrival
                if child != current_node:
                    result.add(child)
                queue.append((child, arrival))
        return result

    def _mu_exec_ms(self, function_node: str) -> float:
        stats = self._function_stats.get(function_node)
        if stats is None or stats.samples <= 0:
            return self._default_exec_ms
        return max(1.0, stats.mean_ms)

    def _mu_trans_ms(self, src_node: str, dst_node: str) -> float:
        stats = self._edge_stats.get((src_node, dst_node))
        if stats is None or stats.samples <= 0:
            return self._default_trans_ms
        return max(0.0, stats.mean_ms)

    def _execution_scale_ms(self) -> float:
        values = [stats.mean_ms for stats in self._function_stats.values() if stats.samples > 0]
        if not values:
            return self._default_exec_ms
        values.sort()
        idx = int(round(0.75 * (len(values) - 1)))
        return max(1.0, values[idx])

    def _compute_sched_sec(self, exec_scale_ms: float) -> int:
        raw = self._sched_eta_exec * (max(1.0, exec_scale_ms) / 1000.0)
        sec = int(math.ceil(raw))
        return max(self._sched_min_sec, min(self._sched_max_sec, sec))

    def _decrease_inflight(self, *, function_node: str) -> None:
        current = int(self._inflight_by_function.get(function_node, 0))
        if current <= 1:
            self._inflight_by_function.pop(function_node, None)
        else:
            self._inflight_by_function[function_node] = current - 1
