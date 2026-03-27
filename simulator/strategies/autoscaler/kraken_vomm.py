from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan
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
    prefix: list[str] = field(default_factory=list)


@dataclass
class UmGraphStats:
    transitions: dict[str, dict[str, float]]
    topo: list[str]
    topo_index: dict[str, int]
    connectivity: dict[str, float]
    commonality: dict[str, float]


class KrakenVomMAutoscaler:
    """Kraken-style autoscaler with VOMM probability estimation.

    This implementation keeps the proactive+reactive spirit of Kraken while
    adapting reactive logic to the simulator's no-queue execution semantics.
    """

    name = "kraken_vomm_v1"

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
        misallocation_ratio: float = 0.0,
        desired_scale: float = 1.0,
        max_prewarm_per_tick: int = 0,
        min_desired_units: float = 0.0,
        uniform_mix: float = 0.0,
        mid_pressure_threshold: float = 55.0,
        high_pressure_threshold: float = 75.0,
        mid_pressure_scale: float = 0.85,
        high_pressure_scale: float = 0.7,
        reconcile_stride: int = 1,
        active_mid_threshold: int = 2,
        active_high_threshold: int = 3,
        active_mid_scale: float = 0.75,
        active_high_scale: float = 0.55,
        vomm_order_max: int = 2,
        vomm_context_min_count: int = 3,
        vomm_prior_weight: float = 1.0,
        load_ewma_alpha: float = 0.3,
        batch_size_default: int = 2,
        commonality_cap: float = 1.0,
        connectivity_cap: float = 1.0,
        reactive_window_sec: int = 30,
        reactive_cold_ratio_threshold: float = 0.35,
        reactive_scale_factor: float = 1.0,
    ) -> None:
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._sched_eta_exec = max(1e-6, float(sched_eta_exec))
        self._sched_min_sec = max(1, int(sched_min_sec))
        self._sched_max_sec = max(self._sched_min_sec, int(sched_max_sec))
        self._horizon_alpha = max(1.0, float(horizon_alpha))
        self._default_exec_ms = max(1.0, float(default_exec_ms))
        self._default_trans_ms = max(0.0, float(default_trans_ms))
        self._desired_scale = max(0.0, float(desired_scale))
        self._max_prewarm_per_tick = max(0, int(max_prewarm_per_tick))
        self._min_desired_units = max(0.0, float(min_desired_units))
        self._uniform_mix = min(1.0, max(0.0, float(uniform_mix)))
        self._reconcile_stride = max(1, int(reconcile_stride))
        self._active_mid_threshold = max(0, int(active_mid_threshold))
        self._active_high_threshold = max(self._active_mid_threshold, int(active_high_threshold))
        self._active_mid_scale = min(1.0, max(0.0, float(active_mid_scale)))
        self._active_high_scale = min(self._active_mid_scale, max(0.0, float(active_high_scale)))
        self._vomm_order_max = max(1, int(vomm_order_max))
        self._vomm_context_min_count = max(1, int(vomm_context_min_count))
        self._vomm_prior_weight = max(0.0, float(vomm_prior_weight))
        self._load_ewma_alpha = min(1.0, max(0.0, float(load_ewma_alpha)))
        self._batch_size_default = max(1, int(batch_size_default))
        self._commonality_cap = max(0.0, float(commonality_cap))
        self._connectivity_cap = max(0.0, float(connectivity_cap))
        self._reactive_window_sec = max(1, int(reactive_window_sec))
        self._reactive_cold_ratio_threshold = min(1.0, max(0.0, float(reactive_cold_ratio_threshold)))
        self._reactive_scale_factor = max(0.0, float(reactive_scale_factor))
        self._eps = 1e-9

        # Kept for backward compatibility with existing config/factory wiring.
        del misallocation_ratio
        del mid_pressure_threshold
        del high_pressure_threshold
        del mid_pressure_scale
        del high_pressure_scale

        self._graphs: dict[str, UmGraphStats] = {}
        self._prepare_graph_stats(templates)

        self._function_exec_stats: dict[str, FunctionExecStats] = {}
        self._edge_transfer_stats: dict[tuple[str, str], EdgeTransferStats] = {}
        self._request_states: dict[str, RequestState] = {}
        self._desired_by_function: dict[str, int] = {}
        self._load_ewma_by_um: dict[str, float] = {}

        # VOMM counts: um -> context -> next -> count
        self._vomm_counts: dict[str, dict[tuple[str, ...], dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # Reactive stats per function.
        self._step_events_by_function: dict[str, deque[tuple[int, int]]] = defaultdict(deque)

        self._next_plan_sec = 0
        self._last_sched_sec = self._sync_period_sec
        self._current_horizon_ms = int(math.ceil(self._horizon_alpha * self._last_sched_sec * 1000.0))
        self._last_active_current_requests = 0

        self._planning_rounds = 0
        self._reconcile_rounds = 0
        self._desired_peak_total = 0
        self._reachable_pairs_added = 0
        self._reactive_scale_events = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
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
        del timestamp_ms, true_future_path
        state = self._request_states.get(request_id)
        if state is None:
            self._request_states[request_id] = RequestState(um=um, prefix=[function_node])
            return
        if state.um != um:
            state.um = um
            state.prefix = [function_node]
            return
        if not state.prefix or state.prefix[-1] != function_node:
            state.prefix.append(function_node)
            if len(state.prefix) > (self._vomm_order_max + 1):
                state.prefix = state.prefix[-(self._vomm_order_max + 1) :]

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
        del request_id, timestamp_ms
        if transfer_ms is not None:
            key = (src_node, dst_node)
            metric = self._edge_transfer_stats.get(key)
            value = float(max(0, int(transfer_ms)))
            if metric is None:
                self._edge_transfer_stats[key] = EdgeTransferStats(mean_ms=value, samples=1)
            else:
                metric.samples += 1
                metric.mean_ms += (value - metric.mean_ms) / metric.samples

        history = tuple(prefix or ())
        if not history or history[-1] != src_node:
            history = tuple(list(history) + [src_node]) if history else (src_node,)
        contexts = [()]
        max_ctx = min(self._vomm_order_max, len(history))
        for k in range(1, max_ctx + 1):
            contexts.append(history[-k:])
        for ctx in contexts:
            per_ctx = self._vomm_counts[um][ctx]
            per_ctx[dst_node] = float(per_ctx.get(dst_node, 0.0)) + 1.0

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
        del request_id, um, transfer_ms, prefix
        metric = self._function_exec_stats.get(function_node)
        value = float(max(1, int(execution_ms)))
        if metric is None:
            self._function_exec_stats[function_node] = FunctionExecStats(mean_ms=value, samples=1)
        else:
            metric.samples += 1
            metric.mean_ms += (value - metric.mean_ms) / metric.samples

        sec = max(0, int(timestamp_ms) // 1000)
        cold = 1 if int(cold_start_ms) > 0 else 0
        self._step_events_by_function[function_node].append((sec, cold))

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        del status, timestamp_ms
        self._request_states.pop(request_id, None)

    def on_tick(
        self,
        *,
        timestamp_sec: int,
        timestamp_ms: int,
        ready_pool_by_function: dict[str, int],
        idle_pool_by_function: dict[str, int] | None = None,
    ) -> list[PrewarmPlan]:
        del timestamp_ms, idle_pool_by_function
        if timestamp_sec < self._next_plan_sec:
            return []
        if timestamp_sec % self._sync_period_sec != 0:
            return []
        if self._reconcile_stride > 1:
            slot = int(timestamp_sec // self._sync_period_sec)
            if slot % self._reconcile_stride != 0:
                return []

        exec_scale_ms = self._execution_scale_ms()
        sched_sec = self._compute_sched_sec(exec_scale_ms)
        self._last_sched_sec = sched_sec
        self._current_horizon_ms = int(math.ceil(self._horizon_alpha * sched_sec * 1000.0))
        self._next_plan_sec = timestamp_sec + sched_sec
        self._planning_rounds += 1
        self._reconcile_rounds += 1

        grouped: dict[tuple[str, tuple[str, ...], str], int] = defaultdict(int)
        active_by_um: dict[str, int] = defaultdict(int)
        active_by_function: dict[str, int] = defaultdict(int)
        for state in self._request_states.values():
            if not state.prefix:
                continue
            current_node = state.prefix[-1]
            context = tuple(state.prefix[-self._vomm_order_max :])
            grouped[(state.um, context, current_node)] += 1
            active_by_um[state.um] += 1
            active_by_function[current_node] += 1
        self._last_active_current_requests = sum(grouped.values())

        self._update_load_ewma(active_by_um)
        self._prune_reactive_events(timestamp_sec)

        desired_float: dict[str, float] = defaultdict(float)
        round_pairs = 0
        for (um, context, current_node), req_count in grouped.items():
            probs, arrivals = self._predict_reach_probabilities(
                um=um,
                context=context,
                current_node=current_node,
            )
            if not probs:
                continue
            active_um = max(1, int(active_by_um.get(um, 0)))
            predicted_um_load = float(self._load_ewma_by_um.get(um, float(active_um)))
            group_ratio = float(req_count) / float(active_um)
            group_predicted_load = max(float(req_count), predicted_um_load * group_ratio)
            batches = int(math.ceil(group_predicted_load / float(self._batch_size_default)))
            if batches <= 0:
                continue

            graph = self._graphs.get(um)
            if graph is None:
                continue
            for function_node, prob in probs.items():
                if function_node == current_node:
                    continue
                arrival = float(arrivals.get(function_node, math.inf))
                if arrival > float(self._current_horizon_ms):
                    continue
                if prob <= self._eps:
                    continue
                base = int(math.ceil(float(batches) * float(prob)))
                if base <= 0:
                    continue
                comm = min(self._commonality_cap, max(0.0, float(graph.commonality.get(function_node, 0.0))))
                conn = min(self._connectivity_cap, max(0.0, float(graph.connectivity.get(function_node, 0.0))))
                extra = int(math.ceil((comm + conn) * float(base)))
                desired_units = base + extra
                if desired_units <= 0:
                    continue
                desired_float[function_node] += float(desired_units)
                round_pairs += req_count
        self._reachable_pairs_added += round_pairs

        desired: dict[str, int] = {}
        for function_node, raw in desired_float.items():
            scaled = float(raw) * self._desired_scale
            if scaled < self._min_desired_units:
                continue
            desired[function_node] = int(math.ceil(scaled))

        # Reactive scaler (no-queue adaptation): if recent cold ratio is high,
        # proactively add containers for currently active overloaded functions.
        for function_node, active_count in active_by_function.items():
            if active_count <= 0:
                continue
            cold_ratio = self._recent_cold_ratio(function_node=function_node)
            if cold_ratio <= self._reactive_cold_ratio_threshold:
                continue
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            delayed = max(0, int(active_count) - (ready * self._batch_size_default))
            if delayed <= 0:
                continue
            extra = int(
                math.ceil((float(delayed) / float(self._batch_size_default)) * self._reactive_scale_factor)
            )
            if extra <= 0:
                continue
            desired[function_node] = max(desired.get(function_node, 0), ready + extra)
            self._reactive_scale_events += 1

        self._desired_by_function = desired
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))

        plans: list[PrewarmPlan] = []
        remaining_budget = self._max_prewarm_per_tick
        for function_node, desired_count in sorted(desired.items(), key=lambda item: (-item[1], item[0])):
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            to_create = max(0, int(desired_count) - ready)
            if to_create <= 0:
                continue
            if self._max_prewarm_per_tick > 0:
                if remaining_budget <= 0:
                    break
                to_create = min(to_create, remaining_budget)
                remaining_budget -= to_create
            if to_create <= 0:
                continue
            plans.append(PrewarmPlan(function_node=function_node, count=to_create))
            self._prewarm_create_attempted += to_create
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
            "mode": "kraken_vomm_pws_rs",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "sched_eta_exec": self._sched_eta_exec,
                "sched_min_sec": self._sched_min_sec,
                "sched_max_sec": self._sched_max_sec,
                "horizon_alpha": self._horizon_alpha,
                "desired_scale": self._desired_scale,
                "max_prewarm_per_tick": self._max_prewarm_per_tick,
                "min_desired_units": self._min_desired_units,
                "uniform_mix": self._uniform_mix,
                "vomm_order_max": self._vomm_order_max,
                "vomm_context_min_count": self._vomm_context_min_count,
                "vomm_prior_weight": self._vomm_prior_weight,
                "load_ewma_alpha": self._load_ewma_alpha,
                "batch_size_default": self._batch_size_default,
                "commonality_cap": self._commonality_cap,
                "connectivity_cap": self._connectivity_cap,
                "reactive_window_sec": self._reactive_window_sec,
                "reactive_cold_ratio_threshold": self._reactive_cold_ratio_threshold,
                "reactive_scale_factor": self._reactive_scale_factor,
            },
            "planning_rounds": self._planning_rounds,
            "reconcile_rounds": self._reconcile_rounds,
            "last_sched_sec": self._last_sched_sec,
            "current_horizon_ms": self._current_horizon_ms,
            "active_current_requests": self._last_active_current_requests,
            "reachable_pairs_added": self._reachable_pairs_added,
            "reactive_scale_events": self._reactive_scale_events,
            "desired_peak_total": self._desired_peak_total,
            "final_desired_by_function": dict(sorted(self._desired_by_function.items())),
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "capacity_blocked_creations": self._capacity_blocked_creations,
        }

    def _update_load_ewma(self, active_by_um: dict[str, int]) -> None:
        for um, active in active_by_um.items():
            observed = float(max(0, int(active)))
            prev = self._load_ewma_by_um.get(um, observed)
            self._load_ewma_by_um[um] = (
                self._load_ewma_alpha * observed + (1.0 - self._load_ewma_alpha) * prev
            )

    def _prune_reactive_events(self, now_sec: int) -> None:
        min_keep = now_sec - self._reactive_window_sec
        for events in self._step_events_by_function.values():
            while events and events[0][0] < min_keep:
                events.popleft()

    def _recent_cold_ratio(self, *, function_node: str) -> float:
        events = self._step_events_by_function.get(function_node)
        if not events:
            return 0.0
        total = len(events)
        if total <= 0:
            return 0.0
        cold = sum(c for _, c in events)
        return float(cold) / float(total)

    @staticmethod
    def _normalize_outgoing(outgoing: dict[str, float]) -> dict[str, float]:
        if not outgoing:
            return {}
        total = sum(max(0.0, float(v)) for v in outgoing.values())
        if total <= 0.0:
            uniform = 1.0 / max(1, len(outgoing))
            return {str(k): uniform for k in outgoing}
        return {str(k): max(0.0, float(v)) / total for k, v in outgoing.items()}

    def _prepare_graph_stats(self, templates: dict[str, DagTemplate]) -> None:
        for um, template in templates.items():
            raw_transitions: dict[str, dict[str, float]] = {}
            nodes: set[str] = {"__start__"}
            for src, outgoing in template.transitions.items():
                nodes.add(src)
                norm = self._normalize_outgoing(outgoing)
                raw_transitions[src] = norm
                for dst in norm:
                    nodes.add(dst)
            for node in nodes:
                raw_transitions.setdefault(node, {})

            topo = self._topological_sort(nodes, raw_transitions)
            topo_index = {node: idx for idx, node in enumerate(topo)}
            connectivity = self._compute_connectivity(raw_transitions)
            commonality = self._compute_commonality(raw_transitions, topo)
            self._graphs[um] = UmGraphStats(
                transitions=raw_transitions,
                topo=topo,
                topo_index=topo_index,
                connectivity=connectivity,
                commonality=commonality,
            )

    @staticmethod
    def _topological_sort(nodes: set[str], transitions: dict[str, dict[str, float]]) -> list[str]:
        indegree: dict[str, int] = {node: 0 for node in nodes}
        for outgoing in transitions.values():
            for dst in outgoing:
                indegree[dst] = indegree.get(dst, 0) + 1
        queue = sorted([node for node, deg in indegree.items() if deg == 0])
        out: list[str] = []
        idx = 0
        while idx < len(queue):
            node = queue[idx]
            idx += 1
            out.append(node)
            for child in transitions.get(node, {}):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(out) != len(indegree):
            return sorted(indegree.keys())
        return out

    def _compute_connectivity(self, transitions: dict[str, dict[str, float]]) -> dict[str, float]:
        function_nodes = [node for node in transitions.keys() if node != "__start__"]
        total_functions = max(1, len(function_nodes))
        memo: dict[str, set[str]] = {}

        def descendants(node: str) -> set[str]:
            cached = memo.get(node)
            if cached is not None:
                return cached
            result: set[str] = set()
            for child in transitions.get(node, {}):
                if child == "__start__":
                    continue
                result.add(child)
                result.update(descendants(child))
            memo[node] = result
            return result

        connectivity: dict[str, float] = {}
        for node in function_nodes:
            connectivity[node] = float(len(descendants(node))) / float(total_functions)
        return connectivity

    def _compute_commonality(
        self,
        transitions: dict[str, dict[str, float]],
        topo: list[str],
    ) -> dict[str, float]:
        path_from_start: dict[str, int] = {node: 0 for node in topo}
        if "__start__" in path_from_start:
            path_from_start["__start__"] = 1
        for node in topo:
            start_paths = path_from_start.get(node, 0)
            if start_paths <= 0:
                continue
            for child in transitions.get(node, {}):
                path_from_start[child] = path_from_start.get(child, 0) + start_paths

        path_to_leaf: dict[str, int] = {node: 0 for node in topo}
        for node in reversed(topo):
            outgoing = transitions.get(node, {})
            if not outgoing:
                path_to_leaf[node] = 1
            else:
                total = 0
                for child in outgoing:
                    total += path_to_leaf.get(child, 0)
                path_to_leaf[node] = total

        total_paths = max(1, path_to_leaf.get("__start__", 0))
        commonality: dict[str, float] = {}
        for node in topo:
            if node == "__start__":
                continue
            containing = path_from_start.get(node, 0) * path_to_leaf.get(node, 0)
            commonality[node] = float(containing) / float(total_paths)
        return commonality

    def _predict_reach_probabilities(
        self,
        *,
        um: str,
        context: tuple[str, ...],
        current_node: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        graph = self._graphs.get(um)
        if graph is None:
            return {}, {}
        if current_node not in graph.topo_index:
            return {}, {}

        frontier: list[tuple[tuple[str, ...], str, float, float, int]] = [
            (context, current_node, 1.0, 0.0, 0)
        ]
        max_depth = max(1, len(graph.topo))
        prob_map: dict[str, float] = defaultdict(float)
        arrival_map: dict[str, float] = {}

        while frontier:
            next_frontier: list[tuple[tuple[str, ...], str, float, float, int]] = []
            for ctx, node, mass, arrival, depth in frontier:
                if depth >= max_depth:
                    continue
                outgoing = graph.transitions.get(node, {})
                if not outgoing:
                    continue
                children = sorted(outgoing.keys())
                dist = self._vomm_next_distribution(
                    um=um,
                    context=ctx,
                    src_node=node,
                    candidates=children,
                )
                if not dist:
                    continue
                exec_ms = self._mu_exec_ms(node)
                for child in children:
                    step_prob = float(dist.get(child, 0.0))
                    if step_prob <= self._eps:
                        continue
                    child_mass = float(mass) * step_prob
                    if child_mass <= self._eps:
                        continue
                    child_arrival = float(arrival) + exec_ms + self._mu_trans_ms(node, child)
                    if child_arrival > float(self._current_horizon_ms):
                        continue
                    prob_map[child] += child_mass
                    prev_arrival = arrival_map.get(child)
                    if prev_arrival is None or child_arrival < prev_arrival:
                        arrival_map[child] = child_arrival
                    child_ctx = tuple((list(ctx) + [child])[-self._vomm_order_max :])
                    next_frontier.append((child_ctx, child, child_mass, child_arrival, depth + 1))
            frontier = next_frontier

        prob_map.pop(current_node, None)
        arrival_map.pop(current_node, None)
        return dict(prob_map), arrival_map

    def _vomm_next_distribution(
        self,
        *,
        um: str,
        context: tuple[str, ...],
        src_node: str,
        candidates: list[str],
    ) -> dict[str, float]:
        if not candidates:
            return {}
        graph = self._graphs.get(um)
        if graph is None:
            return {}
        priors = graph.transitions.get(src_node, {})
        if not priors:
            return {}

        uniform = 1.0 / float(len(candidates))
        prior_dist = {
            node: ((1.0 - self._uniform_mix) * float(priors.get(node, 0.0))) + (self._uniform_mix * uniform)
            for node in candidates
        }
        prior_total = sum(max(0.0, v) for v in prior_dist.values())
        if prior_total <= 0:
            return {node: uniform for node in candidates}
        prior_dist = {node: max(0.0, prior_dist[node]) / prior_total for node in candidates}

        per_um = self._vomm_counts.get(um, {})
        for k in range(min(self._vomm_order_max, len(context)), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            counts = per_um.get(ctx)
            if not counts:
                continue
            total = sum(float(counts.get(node, 0.0)) for node in candidates)
            if total < float(self._vomm_context_min_count):
                continue
            denom = total + self._vomm_prior_weight
            if denom <= 0:
                continue
            blended = {
                node: (float(counts.get(node, 0.0)) + (self._vomm_prior_weight * prior_dist[node])) / denom
                for node in candidates
            }
            return _normalize_prob_map(blended, candidates)
        return prior_dist

    def _mu_exec_ms(self, function_node: str) -> float:
        stats = self._function_exec_stats.get(function_node)
        if stats is None or stats.samples <= 0:
            return self._default_exec_ms
        return max(1.0, stats.mean_ms)

    def _mu_trans_ms(self, src_node: str, dst_node: str) -> float:
        stats = self._edge_transfer_stats.get((src_node, dst_node))
        if stats is None or stats.samples <= 0:
            return self._default_trans_ms
        return max(0.0, stats.mean_ms)

    def _execution_scale_ms(self) -> float:
        values = [stats.mean_ms for stats in self._function_exec_stats.values() if stats.samples > 0]
        if not values:
            return self._default_exec_ms
        values.sort()
        idx = int(round(0.75 * (len(values) - 1)))
        return max(1.0, values[idx])

    def _compute_sched_sec(self, exec_scale_ms: float) -> int:
        raw = self._sched_eta_exec * (max(1.0, exec_scale_ms) / 1000.0)
        sec = int(math.ceil(raw))
        return max(self._sched_min_sec, min(self._sched_max_sec, sec))


def _normalize_prob_map(raw: dict[str, float], candidates: list[str]) -> dict[str, float]:
    total = sum(max(0.0, raw.get(c, 0.0)) for c in candidates)
    if total <= 0:
        uniform = 1.0 / float(max(1, len(candidates)))
        return {c: uniform for c in candidates}
    return {c: max(0.0, raw.get(c, 0.0)) / total for c in candidates}
