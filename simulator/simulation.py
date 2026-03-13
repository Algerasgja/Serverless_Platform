from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any

from simulator.config import SimulationConfig
from simulator.dag.engine import ConditionalDagEngine
from simulator.persistence import RunArtifactsWriter
from simulator.runtime.pool import (
    CONTAINER_COLD_STARTING,
    CONTAINER_IDLE,
    CONTAINER_RUNNING,
    FunctionContainer,
    PhysicalCluster,
)
from simulator.runtime.resource_model import build_bandwidth_matrix, build_function_profiles
from simulator.runtime.workload import generate_arrivals
from simulator.strategies.factory import build_scheduler
from simulator.strategies.scheduler import CandidateScore
from simulator.types import DagCorpus, FunctionProfile, RequestContext


@dataclass(order=True)
class Event:
    timestamp_ms: int
    seq: int
    kind: str
    payload: dict[str, Any]


class SimulationRunner:
    def __init__(self, config: SimulationConfig, corpus: DagCorpus) -> None:
        self._config = config
        self._corpus = corpus
        self._rng = random.Random(config.experiment.random_seed)
        self._scheduler = build_scheduler(config.scheduler, self._rng)
        self._dag_engine = ConditionalDagEngine(
            corpus.templates,
            path_rule=config.dag_policy.path_rule,
            context_regime=config.dag_policy.context_regime,
            mode_count=config.dag_policy.mode_count,
            mode_prior_concentration=config.dag_policy.mode_prior_concentration,
            mode_strength=config.dag_policy.mode_strength,
            prefix_strength=config.dag_policy.prefix_strength,
            prefix_decay=config.dag_policy.prefix_decay,
            prefix_window=config.dag_policy.prefix_window,
            temperature=config.dag_policy.temperature,
            coupling_seed_offset=config.dag_policy.coupling_seed_offset,
            drifting_interval_sec=config.dag_policy.drifting_interval_sec,
            drifting_strength=config.dag_policy.drifting_strength,
            drifting_concentration=config.dag_policy.drifting_concentration,
            drifting_floor=config.dag_policy.drifting_floor,
            eps=config.dag_policy.eps,
            base_seed=config.experiment.random_seed,
            session_gap_sec=config.dag_policy.session_gap_sec,
            session_continue_prob=config.dag_policy.session_continue_prob,
            context_alpha=config.dag_policy.context_alpha,
            rng=self._rng,
        )
        self._event_queue: list[Event] = []
        self._event_seq = 0
        self._request_counter = 0
        self._requests: dict[str, RequestContext] = {}
        self._frame_tick_ms = max(1, int(config.runtime.frame_tick_ms))

        self._cluster = PhysicalCluster(
            node_count=config.physical_nodes.count,
            max_containers_per_node=config.physical_nodes.max_containers_per_node,
            cpu_total_mcpu_per_node=config.physical_nodes.cpu_total_mcpu_per_node,
            mem_total_mb_per_node=config.physical_nodes.mem_total_mb_per_node,
            rng=self._rng,
        )
        function_nodes = self._collect_function_nodes()
        self._function_profiles, function_profile_source = build_function_profiles(
            function_ids=function_nodes,
            runtime_cfg=config.runtime,
            physical_cfg=config.physical_nodes,
            base_seed=config.experiment.random_seed,
        )
        self._bandwidth_matrix, bandwidth_source = build_bandwidth_matrix(
            node_ids=sorted(self._cluster.nodes.keys()),
            physical_cfg=config.physical_nodes,
            base_seed=config.experiment.random_seed,
        )
        self._resource_metadata = {
            "model": "cpu_mem_bandwidth_frame_tick",
            "function_profile_mode": config.runtime.function_profile_mode,
            "function_profile_source": function_profile_source,
            "bandwidth_mode": config.physical_nodes.bandwidth_mode,
            "bandwidth_source": bandwidth_source,
            "frame_tick_ms": self._frame_tick_ms,
            "compute_mb_per_sec_per_1000mcpu": config.runtime.compute_mb_per_sec_per_1000mcpu,
        }

        self._last_housekeeping_sec = -1
        self._last_event_ms = 0
        self._run_writer = RunArtifactsWriter(
            output_root=config.output.runs_dir,
            config=config,
            scheduler_name=self._scheduler.name,
            autoscaler_name=config.autoscaler.type,
            dataset_metadata=corpus.metadata,
        )
        self._workload_replay_profile: dict[str, Any] = {}

    def run(self) -> str:
        duration_sec = self._config.experiment.duration_seconds
        self._schedule_arrivals(duration_sec)

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self._run_housekeeping_until(event.timestamp_ms)
            self._last_event_ms = max(self._last_event_ms, event.timestamp_ms)
            if event.kind == "arrival":
                self._handle_arrival(event)
            elif event.kind == "step_ready":
                self._handle_step_ready(event)
            elif event.kind == "step_running":
                self._handle_step_running(event)
            elif event.kind == "step_complete":
                self._handle_step_complete(event)

        self._run_housekeeping_until(max(self._last_event_ms, duration_sec * 1000))
        self._finalize_requests()
        summary = self._build_summary()
        self._run_writer.finalize(
            summary,
            resource_metadata=self._resource_metadata,
            dag_selection=self._corpus.metadata.get("dag_selection"),
            workload_replay_profile=self._workload_replay_profile,
            path_model=self._dag_engine.path_model_summary(),
        )
        return str(self._run_writer.run_dir)

    def _collect_function_nodes(self) -> set[str]:
        nodes: set[str] = set()
        for template in self._corpus.templates.values():
            for src, nexts in template.transitions.items():
                if src != "__start__":
                    nodes.add(src)
                for dst in nexts:
                    if dst != "__start__":
                        nodes.add(dst)
        return nodes

    def _schedule_arrivals(self, duration_sec: int) -> None:
        arrivals = generate_arrivals(
            workload_cfg=self._config.workload,
            corpus=self._corpus,
            duration_seconds=duration_sec,
            rng=self._rng,
            base_seed=self._config.experiment.random_seed,
            profile_sink=self._workload_replay_profile,
        )
        for at_ms, um in arrivals:
            self._push_event(at_ms, "arrival", {"um": um})

    def _handle_arrival(self, event: Event) -> None:
        um = event.payload["um"]
        now_sec = event.timestamp_ms // 1000
        session_id = self._dag_engine.assign_session(um, now_sec)
        path = self._dag_engine.generate_path(um, session_id, now_sec=now_sec)
        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        self._requests[request_id] = RequestContext(
            request_id=request_id,
            um=um,
            session_id=session_id,
            arrival_ms=event.timestamp_ms,
            path=path,
        )
        if path:
            self._push_event(event.timestamp_ms, "step_ready", {"request_id": request_id})
        else:
            req = self._requests[request_id]
            req.completed = True
            req.completed_ms = event.timestamp_ms

    def _handle_step_ready(self, event: Event) -> None:
        request_id = event.payload["request_id"]
        req = self._requests[request_id]
        if req.completed or req.timed_out or req.failed_reason:
            return
        if event.timestamp_ms - req.arrival_ms > self._config.runtime.request_timeout_ms:
            req.timed_out = True
            req.failed_reason = "timeout"
            req.completed_ms = event.timestamp_ms
            return
        if req.current_index >= len(req.path):
            req.completed = True
            req.completed_ms = event.timestamp_ms
            return

        function_node = req.path[req.current_index]
        profile = self._profile_for(function_node)
        warm_container = self._cluster.acquire_warm_container(
            function_node=function_node,
            now_ms=event.timestamp_ms,
        )
        if warm_container is not None:
            self._schedule_step_execution(
                timestamp_ms=event.timestamp_ms,
                req=req,
                function_node=function_node,
                container=warm_container,
                profile=profile,
                decision_type="warm_reuse",
                cold_start_ms=0,
                reason="warm_container_available",
                state_before=CONTAINER_IDLE,
                state_after=CONTAINER_RUNNING,
            )
            return

        if not self._has_global_container_capacity():
            self._mark_capacity_exhausted(
                req=req,
                timestamp_ms=event.timestamp_ms,
                function_node=function_node,
                reason="global_container_cap",
            )
            return

        host_id = self._select_least_loaded_host_for_cold_start(profile)
        if host_id is None:
            self._mark_capacity_exhausted(
                req=req,
                timestamp_ms=event.timestamp_ms,
                function_node=function_node,
                reason="host_resource_exhausted",
            )
            return

        container = self._cluster.create_cold_container_on_host(
            host_id=host_id,
            function_node=function_node,
            cpu_request_mcpu=profile.cpu_request_mcpu,
            memory_mb=profile.memory_mb,
            now_ms=event.timestamp_ms,
        )
        if container is None:
            self._mark_capacity_exhausted(
                req=req,
                timestamp_ms=event.timestamp_ms,
                function_node=function_node,
                reason="host_resource_exhausted",
            )
            return

        cold_start_ms = self._quantize_duration(profile.cold_start_ms)
        self._schedule_step_execution(
            timestamp_ms=event.timestamp_ms,
            req=req,
            function_node=function_node,
            container=container,
            profile=profile,
            decision_type="cold_start",
            cold_start_ms=cold_start_ms,
            reason="no_warm_container",
            state_before="",
            state_after=CONTAINER_COLD_STARTING,
        )

    def _schedule_step_execution(
        self,
        *,
        timestamp_ms: int,
        req: RequestContext,
        function_node: str,
        container: FunctionContainer,
        profile: FunctionProfile,
        decision_type: str,
        cold_start_ms: int,
        reason: str,
        state_before: str,
        state_after: str,
    ) -> None:
        transfer_ms, transfer_data_mb, bandwidth_mbps = self._compute_transfer(
            req=req,
            current_host_id=container.host_id,
        )
        req.data_transfer_latency_ms += transfer_ms
        req.cold_start_latency_ms += cold_start_ms
        run_at_ms = timestamp_ms + transfer_ms + cold_start_ms

        self._push_event(
            run_at_ms,
            "step_running",
            {
                "request_id": req.request_id,
                "function_node": function_node,
                "container_id": container.container_id,
                "host_id": container.host_id,
                "decision_timestamp_ms": timestamp_ms,
                "decision_type": decision_type,
                "reason": reason,
                "cold_start_ms": cold_start_ms,
                "transfer_ms": transfer_ms,
                "transfer_data_mb": transfer_data_mb,
                "bandwidth_mbps": bandwidth_mbps,
                "state_before": state_before,
                "state_after": state_after,
                "cpu_request_mcpu": profile.cpu_request_mcpu,
                "memory_mb": profile.memory_mb,
            },
        )

    def _handle_step_running(self, event: Event) -> None:
        request_id = event.payload["request_id"]
        container_id = event.payload["container_id"]
        function_node = event.payload["function_node"]
        req = self._requests[request_id]
        container = self._cluster.get_container(container_id)
        if container is None:
            return

        if req.completed or req.timed_out or req.failed_reason:
            self._cluster.release_container(container_id, event.timestamp_ms)
            return
        if event.timestamp_ms - req.arrival_ms > self._config.runtime.request_timeout_ms:
            req.timed_out = True
            req.failed_reason = "timeout"
            req.completed_ms = event.timestamp_ms
            self._cluster.release_container(container_id, event.timestamp_ms)
            return

        self._cluster.set_container_running(container_id)
        profile = self._profile_for(function_node)
        allocated_cpu_mcpu = self._cluster.allocated_cpu_for_container(container_id)
        execution_ms = self._compute_execution_ms(
            compute_data_mb=profile.compute_data_mb,
            allocated_cpu_mcpu=allocated_cpu_mcpu,
        )
        req.execution_latency_ms += execution_ms

        self._run_writer.log_scheduler_decision(
            timestamp_ms=int(event.payload["decision_timestamp_ms"]),
            request_id=req.request_id,
            um=req.um,
            function_node=function_node,
            decision_type=str(event.payload["decision_type"]),
            physical_node=container.host_id,
            container_id=container.container_id,
            cold_start_ms=int(event.payload["cold_start_ms"]),
            transfer_ms=int(event.payload["transfer_ms"]),
            execution_ms=execution_ms,
            container_state_before=str(event.payload["state_before"]),
            container_state_after=CONTAINER_RUNNING,
            cpu_request_mcpu=int(event.payload["cpu_request_mcpu"]),
            memory_mb=int(event.payload["memory_mb"]),
            transfer_data_mb=float(event.payload["transfer_data_mb"]),
            bandwidth_mbps=(
                None
                if event.payload["bandwidth_mbps"] is None
                else float(event.payload["bandwidth_mbps"])
            ),
            allocated_cpu_mcpu=allocated_cpu_mcpu,
            reason=str(event.payload["reason"]),
        )
        self._push_event(
            event.timestamp_ms + execution_ms,
            "step_complete",
            {
                "request_id": req.request_id,
                "function_node": function_node,
                "container_id": container.container_id,
                "host_id": container.host_id,
            },
        )

    def _handle_step_complete(self, event: Event) -> None:
        request_id = event.payload["request_id"]
        function_node = event.payload["function_node"]
        container_id = event.payload["container_id"]
        host_id = event.payload["host_id"]
        self._cluster.release_container(container_id, event.timestamp_ms)

        req = self._requests[request_id]
        if req.completed or req.timed_out or req.failed_reason:
            return

        if event.timestamp_ms - req.arrival_ms > self._config.runtime.request_timeout_ms:
            req.timed_out = True
            req.failed_reason = "timeout"
            req.completed_ms = event.timestamp_ms
            return

        req.prev_host_id = host_id
        req.prev_function_node = function_node
        req.current_index += 1
        if req.current_index >= len(req.path):
            req.completed_ms = event.timestamp_ms
            req.completed = True
        else:
            self._push_event(event.timestamp_ms, "step_ready", {"request_id": request_id})

    def _profile_for(self, function_node: str) -> FunctionProfile:
        profile = self._function_profiles.get(function_node)
        if profile is not None:
            return profile
        # Safety fallback for unexpected nodes.
        return FunctionProfile(
            function_id=function_node,
            cold_start_ms=self._config.runtime.function_cold_start_ms_min,
            memory_mb=self._config.runtime.function_memory_mb_min,
            output_data_mb=self._config.runtime.function_output_data_mb_min,
            compute_data_mb=self._config.runtime.function_compute_data_mb_min,
            cpu_request_mcpu=self._config.runtime.function_cpu_request_mcpu_min,
        )

    def _has_global_container_capacity(self) -> bool:
        return self._cluster.total_active_containers < max(0, self._config.capacity.max_total_instances)

    def _select_least_loaded_host_for_cold_start(self, profile: FunctionProfile) -> str | None:
        host_candidates = self._cluster.host_candidates_with_slot(
            cpu_request_mcpu=profile.cpu_request_mcpu,
            memory_mb=profile.memory_mb,
        )
        candidates = [
            CandidateScore(
                instance_id=host.node_id,
                inflight=host.busy_containers,
                max_concurrency=host.max_containers,
                score=host.utilization,
            )
            for host in host_candidates
        ]
        return self._scheduler.select_instance(candidates)

    def _compute_transfer(
        self,
        *,
        req: RequestContext,
        current_host_id: str,
    ) -> tuple[int, float, float | None]:
        if req.current_index <= 0 or req.prev_host_id is None or req.prev_function_node is None:
            return 0, 0.0, None
        prev_profile = self._profile_for(req.prev_function_node)
        data_mb = max(0.0, prev_profile.output_data_mb)
        if data_mb <= 0:
            return 0, 0.0, None
        if req.prev_host_id == current_host_id:
            return 0, data_mb, None

        bandwidth_mbps = self._bandwidth_matrix.get(req.prev_host_id, {}).get(current_host_id)
        if bandwidth_mbps is None or bandwidth_mbps <= 0:
            raise ValueError(f"missing bandwidth edge ({req.prev_host_id}, {current_host_id})")
        transfer_ms = math.ceil((data_mb * 8.0 / bandwidth_mbps) * 1000.0)
        return self._quantize_duration(transfer_ms), data_mb, bandwidth_mbps

    def _compute_execution_ms(
        self,
        *,
        compute_data_mb: float,
        allocated_cpu_mcpu: float,
    ) -> int:
        throughput = (
            self._config.runtime.compute_mb_per_sec_per_1000mcpu
            * (max(1.0, allocated_cpu_mcpu) / 1000.0)
        )
        if throughput <= 0:
            throughput = 0.001
        raw_ms = math.ceil((max(0.001, compute_data_mb) / throughput) * 1000.0)
        return self._quantize_duration(raw_ms)

    def _quantize_duration(self, duration_ms: int | float) -> int:
        duration = float(duration_ms)
        if duration <= 0:
            return 0
        ticks = math.ceil(duration / self._frame_tick_ms)
        return max(self._frame_tick_ms, ticks * self._frame_tick_ms)

    def _mark_capacity_exhausted(
        self,
        *,
        req: RequestContext,
        timestamp_ms: int,
        function_node: str,
        reason: str,
    ) -> None:
        req.failed_reason = "capacity_exhausted"
        req.completed_ms = timestamp_ms
        profile = self._profile_for(function_node)
        self._run_writer.log_scheduler_decision(
            timestamp_ms=timestamp_ms,
            request_id=req.request_id,
            um=req.um,
            function_node=function_node,
            decision_type="capacity_exhausted",
            physical_node=None,
            container_id=None,
            cold_start_ms=0,
            transfer_ms=0,
            execution_ms=0,
            container_state_before=None,
            container_state_after=None,
            cpu_request_mcpu=profile.cpu_request_mcpu,
            memory_mb=profile.memory_mb,
            transfer_data_mb=0.0,
            bandwidth_mbps=None,
            allocated_cpu_mcpu=None,
            reason=reason,
        )

    def _run_housekeeping_until(self, timestamp_ms: int) -> None:
        target_sec = max(0, timestamp_ms // 1000)
        for sec in range(self._last_housekeeping_sec + 1, target_sec + 1):
            now_ms = sec * 1000
            self._cluster.reap_idle_containers(
                now_ms=now_ms,
                idle_ttl_sec=self._config.physical_nodes.idle_ttl_sec,
            )
            self._log_node_metrics(sec)
            self._last_housekeeping_sec = sec

    def _log_node_metrics(self, timestamp_sec: int) -> None:
        for row in self._cluster.snapshot_node_metrics():
            self._run_writer.log_node_metric(
                timestamp_sec=timestamp_sec,
                node=str(row["node"]),
                active_containers=int(row["active_containers"]),
                busy_containers=int(row["busy_containers"]),
                max_containers=int(row["max_containers"]),
                replicas=int(row["active_containers"]),
                inflight=int(row["busy_containers"]),
                queue_len=0,
                utilization=float(row["utilization"]),
                draining_replicas=0,
                cpu_total_mcpu=int(row["cpu_total_mcpu"]),
                cpu_reserved_mcpu=int(row["cpu_reserved_mcpu"]),
                cpu_utilization=float(row["cpu_utilization"]),
                mem_total_mb=int(row["mem_total_mb"]),
                mem_reserved_mb=int(row["mem_reserved_mb"]),
                mem_utilization=float(row["mem_utilization"]),
                cold_starting_containers=int(row["cold_starting_containers"]),
                running_containers=int(row["running_containers"]),
                idle_containers=int(row["idle_containers"]),
            )

    def _finalize_requests(self) -> None:
        for req in self._requests.values():
            latency = None
            if req.completed_ms is not None:
                latency = req.completed_ms - req.arrival_ms
            if req.completed_ms is None and not req.timed_out and not req.failed_reason:
                req.timed_out = True
                req.failed_reason = "timeout"
            self._run_writer.log_request_path(
                request_id=req.request_id,
                session_id=req.session_id,
                um=req.um,
                arrival_ms=req.arrival_ms,
                path=req.path,
                completed=req.completed,
                timed_out=req.timed_out,
                failed_reason=req.failed_reason,
                total_latency_ms=latency,
                cold_start_latency_ms=req.cold_start_latency_ms,
                data_transfer_latency_ms=req.data_transfer_latency_ms,
                execution_latency_ms=req.execution_latency_ms,
                queue_wait_latency_ms=req.queue_wait_latency_ms,
            )

    def _build_summary(self) -> dict[str, Any]:
        total = len(self._requests)
        completed = 0
        timed_out = 0
        capacity_exhausted = 0
        latencies: list[int] = []
        cold_components: list[int] = []
        transfer_components: list[int] = []
        execution_components: list[int] = []
        queue_components: list[int] = []

        for req in self._requests.values():
            if req.completed:
                completed += 1
                if req.completed_ms is not None:
                    latencies.append(req.completed_ms - req.arrival_ms)
                cold_components.append(req.cold_start_latency_ms)
                transfer_components.append(req.data_transfer_latency_ms)
                execution_components.append(req.execution_latency_ms)
                queue_components.append(req.queue_wait_latency_ms)
            if req.timed_out:
                timed_out += 1
            if req.failed_reason == "capacity_exhausted":
                capacity_exhausted += 1

        return {
            "total_requests": total,
            "completed_requests": completed,
            "timed_out_requests": timed_out,
            "capacity_exhausted_requests": capacity_exhausted,
            "failed_requests": total - completed,
            "success_rate": round((completed / total), 6) if total else 0.0,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p99_latency_ms": _percentile(latencies, 99),
            "max_latency_ms": max(latencies) if latencies else None,
            "min_latency_ms": min(latencies) if latencies else None,
            "avg_cold_start_latency_ms": _avg(cold_components),
            "avg_data_transfer_latency_ms": _avg(transfer_components),
            "avg_execution_latency_ms": _avg(execution_components),
            "avg_queue_wait_latency_ms": _avg(queue_components),
        }

    def _push_event(self, timestamp_ms: int, kind: str, payload: dict[str, Any]) -> None:
        self._event_seq += 1
        heapq.heappush(
            self._event_queue,
            Event(timestamp_ms=timestamp_ms, seq=self._event_seq, kind=kind, payload=payload),
        )


def _percentile(samples: list[int], pct: int) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    ratio = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * ratio


def _avg(samples: list[int]) -> float | None:
    if not samples:
        return None
    return sum(samples) / len(samples)
