from __future__ import annotations

from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan


class NoOpAutoscaler:
    def __init__(self, *, name: str = "noop_v1") -> None:
        self.name = str(name)

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
        return None

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
        return None

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        return None

    def on_tick(
        self,
        *,
        timestamp_sec: int,
        timestamp_ms: int,
        ready_pool_by_function: dict[str, int],
        idle_pool_by_function: dict[str, int] | None = None,
    ) -> list[PrewarmPlan]:
        del timestamp_sec, timestamp_ms, ready_pool_by_function, idle_pool_by_function
        return []

    def on_prewarm_create_result(
        self,
        *,
        function_node: str,
        success: bool,
        timestamp_ms: int,
        reason: str,
    ) -> None:
        return None

    def on_prewarm_ready(
        self,
        *,
        function_node: str,
        container_id: str,
        timestamp_ms: int,
    ) -> None:
        return None

    def on_prewarm_consumed(
        self,
        *,
        function_node: str,
        request_id: str,
        container_id: str,
        timestamp_ms: int,
    ) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "enabled": False,
            "mode": "no_autoscale",
        }
