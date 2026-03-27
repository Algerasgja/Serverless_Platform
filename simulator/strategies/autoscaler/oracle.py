from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan, ScaleDownPlan
from simulator.types import DagTemplate


@dataclass
class OracleRequestState:
    um: str
    active_pairs: set[str] = field(default_factory=set)


class OracleFutureAutoscaler:
    """Ideal upper-bound autoscaler using true future path within a fixed window."""

    name = "oracle_future_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        window_steps: int,
    ) -> None:
        del templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._window_steps = max(1, int(window_steps))

        self._request_states: dict[str, OracleRequestState] = {}
        self._desired_by_function: dict[str, int] = {}
        self._active_pairs_total = 0

        self._window_refresh_total = 0
        self._plan_pairs_added = 0
        self._plan_pairs_active_peak = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._reconcile_rounds = 0
        self._desired_peak_total = 0
        self._scale_down_requested = 0
        self._capacity_blocked_creations = 0

    def on_step_start(
        self,
        *,
        request_id: str,
        um: str,
        function_node: str,
        timestamp_ms: int,
        true_future_path: tuple[str, ...] | None = None,
    ) -> None:
        del function_node, timestamp_ms
        state = self._request_states.get(request_id)
        if state is None:
            state = OracleRequestState(um=um)
            self._request_states[request_id] = state
        else:
            state.um = um

        self._clear_request_pairs(request_id)
        future = tuple(true_future_path or ())
        for function_id in future[: self._window_steps]:
            self._add_pair(request_id, function_id)
        self._window_refresh_total += 1

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
        del request_id, um, function_node, timestamp_ms, execution_ms, cold_start_ms, transfer_ms, prefix
        return None

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del status, timestamp_ms
        self._clear_request_pairs(request_id)
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
        if timestamp_sec % self._sync_period_sec != 0:
            return []
        self._reconcile_rounds += 1

        desired = {k: v for k, v in self._desired_by_function.items() if v > 0}
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))
        plans: list[PrewarmPlan | ScaleDownPlan] = []
        all_functions = sorted(set(desired) | set(ready_pool_by_function))
        for function_node in all_functions:
            desired_count = max(0, int(desired.get(function_node, 0)))
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            idle = max(0, int(idle_pool_by_function.get(function_node, 0)))
            to_create = max(0, int(desired_count) - ready)
            if to_create > 0:
                plans.append(PrewarmPlan(function_node=function_node, count=to_create))
                self._prewarm_create_attempted += to_create
                continue
            to_remove = min(idle, max(0, ready - desired_count))
            if to_remove > 0:
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
        return {
            "type": self.name,
            "mode": "true_future_window_oracle",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "window_steps": self._window_steps,
            },
            "window_refresh_total": self._window_refresh_total,
            "plan_pairs_added": self._plan_pairs_added,
            "plan_pairs_active_peak": self._plan_pairs_active_peak,
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "reconcile_rounds": self._reconcile_rounds,
            "desired_peak_total": self._desired_peak_total,
            "scale_down_requested": self._scale_down_requested,
            "capacity_blocked_creations": self._capacity_blocked_creations,
            "final_desired_by_function": dict(sorted((k, v) for k, v in self._desired_by_function.items() if v > 0)),
        }

    def _add_pair(self, request_id: str, function_node: str) -> None:
        state = self._request_states.get(request_id)
        if state is None:
            return
        if function_node in state.active_pairs:
            return
        state.active_pairs.add(function_node)
        self._desired_by_function[function_node] = self._desired_by_function.get(function_node, 0) + 1
        self._plan_pairs_added += 1
        self._active_pairs_total += 1
        self._plan_pairs_active_peak = max(self._plan_pairs_active_peak, self._active_pairs_total)

    def _remove_pair(self, request_id: str, function_node: str) -> None:
        state = self._request_states.get(request_id)
        if state is None or function_node not in state.active_pairs:
            return
        state.active_pairs.remove(function_node)
        current = self._desired_by_function.get(function_node, 0)
        if current <= 1:
            self._desired_by_function.pop(function_node, None)
        else:
            self._desired_by_function[function_node] = current - 1
        self._active_pairs_total = max(0, self._active_pairs_total - 1)

    def _clear_request_pairs(self, request_id: str) -> None:
        state = self._request_states.get(request_id)
        if state is None:
            return
        for function_node in list(state.active_pairs):
            self._remove_pair(request_id, function_node)
