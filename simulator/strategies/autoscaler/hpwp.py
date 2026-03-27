from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan
from simulator.types import DagTemplate


@dataclass
class FunctionStats:
    mu_exec_ms: float = 0.0
    mu_cold_ms: float = 0.0
    samples: int = 0


@dataclass
class TransitionMetrics:
    weighted_count: float = 0.0
    avg_trans_latency_ms: float = 0.0


@dataclass
class ActiveRequestState:
    um: str
    prefix: list[str] = field(default_factory=list)
    last_seen_ms: int = 0


@dataclass
class HitEvent:
    sec: int
    hit: int
    branch_node: str


@dataclass
class MassFrontierState:
    prefix: tuple[str, ...]
    current_node: str
    prob: float
    elapsed_ms: float
    hop: int


@dataclass
class ReachProfile:
    prob: float
    arrival_ms: float


class HpwpAutoscaler:
    name = "hpwp_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        sync_period_sec: int,
        lmax_low: int,
        lmax_high: int,
        beta_hi: float,
        beta_lo: float,
        alpha_exp: float,
        alpha_stable: float,
        sched_eta_exec: float,
        sched_min_sec: int,
        sched_max_sec: int,
        horizon_alpha: float,
        rho_mass: float,
        tau_p: float,
        urgency_epsilon_ms: float,
        phase_window_k: int,
        phase_n_min: int,
        phase_var_threshold: float,
        drift_short_k: int,
        drift_long_k: int,
        drift_delta_mr: float,
        drift_tau_mr: float,
        drift_branch_tau: float,
        forget_gamma: float,
        default_exec_ms: float,
        default_cold_ms: float,
        default_trans_ms: float,
        seed_offset: int,
    ) -> None:
        self._templates = templates
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._lmax_low = max(1, int(lmax_low))
        self._lmax_high = max(self._lmax_low, int(lmax_high))
        self._beta_hi = max(1e-6, float(beta_hi))
        self._beta_lo = max(1e-6, float(beta_lo))
        self._alpha_exp = min(1.0, max(0.0, float(alpha_exp)))
        self._alpha_stable = min(1.0, max(0.0, float(alpha_stable)))
        self._sched_eta_exec = max(1e-6, float(sched_eta_exec))
        self._sched_min_sec = max(1, int(sched_min_sec))
        self._sched_max_sec = max(self._sched_min_sec, int(sched_max_sec))
        self._horizon_alpha = max(1.0, float(horizon_alpha))
        self._rho_mass = min(1.0, max(0.05, float(rho_mass)))
        self._tau_p = min(1.0, max(0.0, float(tau_p)))
        self._urgency_epsilon_ms = max(0.0, float(urgency_epsilon_ms))
        self._phase_window_k = max(1, int(phase_window_k))
        self._phase_n_min = max(1, int(phase_n_min))
        self._phase_var_threshold = max(0.0, float(phase_var_threshold))
        self._drift_short_k = max(1, int(drift_short_k))
        self._drift_long_k = max(self._drift_short_k, int(drift_long_k))
        self._drift_delta_mr = max(0.0, float(drift_delta_mr))
        self._drift_tau_mr = max(0.0, float(drift_tau_mr))
        self._drift_branch_tau = max(0.0, float(drift_branch_tau))
        self._forget_gamma = min(1.0, max(1e-6, float(forget_gamma)))
        self._default_exec_ms = max(1.0, float(default_exec_ms))
        self._default_cold_ms = max(1.0, float(default_cold_ms))
        self._default_trans_ms = max(0.0, float(default_trans_ms))
        self._seed_offset = int(seed_offset)
        self._eps = 1e-9
        self._max_prediction_hops = 128
        self._prior_weight = 1.0
        self._path_prob_floor = 1e-6

        self._dag_outgoing: dict[str, dict[str, dict[str, float]]] = {}
        self._branch_nodes_by_um: dict[str, set[str]] = {}
        self._transition_stats: dict[str, dict[tuple[str, ...], dict[str, TransitionMetrics]]] = {}
        self._bootstrap_transition_priors()

        self._request_states: dict[str, ActiveRequestState] = {}
        self._function_stats: dict[str, FunctionStats] = {}
        self._hit_events: deque[HitEvent] = deque()
        self._last_drift_sec = -10**9

        self._phase = "exploration"
        self._current_lmax = self._lmax_low
        self._current_beta = self._beta_hi
        self._current_alpha = self._alpha_exp
        self._phase_switches = 0
        self._set_phase("exploration", force=True)

        self._next_plan_sec = 0
        self._last_sched_sec = self._sync_period_sec
        self._planning_rounds = 0
        self._reconcile_rounds = 0
        self._desired_by_function: dict[str, int] = {}

        self._predictions_total = 0
        self._hit_total = 0
        self._hit_correct = 0
        self._drift_events = 0
        self._desired_peak_total = 0
        self._active_prefix_buckets_peak = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._capacity_blocked_creations = 0

    def _bootstrap_transition_priors(self) -> None:
        for um, template in self._templates.items():
            per_um: dict[tuple[str, ...], dict[str, TransitionMetrics]] = defaultdict(dict)
            outgoing_map: dict[str, dict[str, float]] = {}
            branch_nodes: set[str] = set()
            for src, outgoing in template.transitions.items():
                if not outgoing:
                    continue
                cleaned = {dst: float(prob) for dst, prob in outgoing.items()}
                outgoing_map[src] = cleaned
                if len(cleaned) > 1:
                    branch_nodes.add(src)
                c0 = per_um[()]
                c1 = per_um[(src,)]
                for dst, prob in cleaned.items():
                    prior = max(self._eps, prob) * self._prior_weight
                    c0_metric = c0.setdefault(dst, TransitionMetrics())
                    c1_metric = c1.setdefault(dst, TransitionMetrics())
                    c0_metric.weighted_count += prior
                    c1_metric.weighted_count += prior
                    if c0_metric.avg_trans_latency_ms <= 0:
                        c0_metric.avg_trans_latency_ms = self._default_trans_ms
                    if c1_metric.avg_trans_latency_ms <= 0:
                        c1_metric.avg_trans_latency_ms = self._default_trans_ms
            self._transition_stats[um] = dict(per_um)
            self._dag_outgoing[um] = outgoing_map
            self._branch_nodes_by_um[um] = branch_nodes

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
        state = self._request_states.get(request_id)
        if state is None:
            state = ActiveRequestState(um=um)
            self._request_states[request_id] = state
        if state.um != um:
            state.um = um
            state.prefix = []
        if not state.prefix or state.prefix[-1] != function_node:
            state.prefix.append(function_node)
        state.last_seen_ms = timestamp_ms

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
        state = self._request_states.get(request_id or "")
        if state is None and not prefix:
            return
        if state is not None and state.um != um:
            return

        prefix_tuple = tuple(prefix) if prefix else tuple(state.prefix if state else ())
        if not prefix_tuple or prefix_tuple[-1] != src_node:
            return

        outgoing = self._dag_outgoing.get(um, {}).get(src_node, {})
        if not outgoing:
            return
        candidates = sorted(outgoing.keys())
        if dst_node not in outgoing:
            candidates = sorted(set(candidates + [dst_node]))
            outgoing = dict(outgoing)
            outgoing[dst_node] = self._eps

        if len(candidates) > 1:
            predicted = self._predict_next(
                um=um,
                prefix=prefix_tuple,
                candidates=candidates,
                dag_probs=outgoing,
            )
            hit = int(predicted == dst_node)
            self._record_hit(timestamp_ms=timestamp_ms, hit=hit, branch_node=src_node)

        contexts = self._contexts_for_prefix(prefix_tuple)
        observed_transfer = float(max(0, int(transfer_ms or 0)))
        for context in contexts:
            metrics_map = self._ensure_metrics(
                um=um,
                context=context,
                src_node=src_node,
                candidates=candidates,
            )
            for candidate in candidates:
                metric = metrics_map[candidate]
                obs = 1.0 if candidate == dst_node else 0.0
                metric.weighted_count = (
                    self._current_alpha * obs + (1.0 - self._current_alpha) * metric.weighted_count
                )
                if candidate == dst_node:
                    if metric.avg_trans_latency_ms <= 0:
                        metric.avg_trans_latency_ms = observed_transfer
                    else:
                        metric.avg_trans_latency_ms = (
                            self._current_alpha * observed_transfer
                            + (1.0 - self._current_alpha) * metric.avg_trans_latency_ms
                        )

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
        del request_id, um, timestamp_ms, transfer_ms, prefix
        stats = self._function_stats.setdefault(function_node, FunctionStats())
        exec_value = float(max(1, int(execution_ms)))
        cold_value = float(max(0, int(cold_start_ms)))
        if stats.samples == 0:
            stats.mu_exec_ms = exec_value
            stats.mu_cold_ms = max(self._default_cold_ms, cold_value)
        else:
            stats.mu_exec_ms = self._current_alpha * exec_value + (1.0 - self._current_alpha) * stats.mu_exec_ms
            stats.mu_cold_ms = self._current_alpha * cold_value + (1.0 - self._current_alpha) * stats.mu_cold_ms
        stats.samples += 1

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
        self._prune_hit_events(now_sec=timestamp_sec)
        self._update_phase(now_sec=timestamp_sec)
        self._maybe_handle_drift(now_sec=timestamp_sec)

        if timestamp_sec < self._next_plan_sec:
            return []
        if timestamp_sec % self._sync_period_sec != 0:
            return []

        exec_scale_ms = self._execution_scale_ms()
        sched_sec = self._compute_sched_sec(exec_scale_ms)
        self._last_sched_sec = sched_sec
        self._next_plan_sec = timestamp_sec + sched_sec
        self._planning_rounds += 1

        horizon_ms = max(1, int(math.ceil(self._horizon_alpha * sched_sec * 1000.0)))
        desired = self._compute_desired(horizon_ms=horizon_ms)
        self._desired_by_function = desired
        self._desired_peak_total = max(self._desired_peak_total, sum(desired.values()))
        self._reconcile_rounds += 1

        plans: list[PrewarmPlan] = []
        for function_node, desired_count in sorted(desired.items()):
            ready_pool = max(0, int(ready_pool_by_function.get(function_node, 0)))
            to_create = max(0, int(desired_count) - ready_pool)
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
        del function_node, timestamp_ms
        if success:
            self._prewarm_created += 1
            return None
        del reason
        self._capacity_blocked_creations += 1
        return None

    def on_prewarm_ready(
        self,
        *,
        function_node: str,
        container_id: str,
        timestamp_ms: int,
    ) -> None:
        del function_node, container_id, timestamp_ms
        self._prewarm_ready += 1
        return None

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
        return None

    def summary(self) -> dict[str, Any]:
        hit_rate = (self._hit_correct / self._hit_total) if self._hit_total else None
        return {
            "type": self.name,
            "mode": "active_reconcile",
            "parameters": {
                "sync_period_sec": self._sync_period_sec,
                "lmax_low": self._lmax_low,
                "lmax_high": self._lmax_high,
                "beta_hi": self._beta_hi,
                "beta_lo": self._beta_lo,
                "alpha_exp": self._alpha_exp,
                "alpha_stable": self._alpha_stable,
                "sched_eta_exec": self._sched_eta_exec,
                "sched_min_sec": self._sched_min_sec,
                "sched_max_sec": self._sched_max_sec,
                "horizon_alpha": self._horizon_alpha,
                "urgency_epsilon_ms": self._urgency_epsilon_ms,
                "tau_p": self._tau_p,
                "rho_mass": self._rho_mass,
                "phase_window_k": self._phase_window_k,
                "phase_n_min": self._phase_n_min,
                "phase_var_threshold": self._phase_var_threshold,
                "drift_short_k": self._drift_short_k,
                "drift_long_k": self._drift_long_k,
                "drift_delta_mr": self._drift_delta_mr,
                "drift_tau_mr": self._drift_tau_mr,
                "drift_branch_tau": self._drift_branch_tau,
                "forget_gamma": self._forget_gamma,
                "seed_offset": self._seed_offset,
            },
            "phase": self._phase,
            "phase_switches": self._phase_switches,
            "drift_events": self._drift_events,
            "predictions_total": self._predictions_total,
            "transition_events": self._hit_total,
            "hit1_rate": hit_rate,
            "planning_rounds": self._planning_rounds,
            "reconcile_rounds": self._reconcile_rounds,
            "last_sched_sec": self._last_sched_sec,
            "active_prefix_buckets_peak": self._active_prefix_buckets_peak,
            "desired_peak_total": self._desired_peak_total,
            "final_desired_by_function": dict(sorted(self._desired_by_function.items())),
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "capacity_blocked_creations": self._capacity_blocked_creations,
        }

    def _compute_desired(self, *, horizon_ms: int) -> dict[str, int]:
        prefix_buckets: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
        for state in self._request_states.values():
            if not state.prefix:
                continue
            prefix_buckets[(state.um, tuple(state.prefix))] += 1
        self._active_prefix_buckets_peak = max(self._active_prefix_buckets_peak, len(prefix_buckets))

        desired: dict[str, int] = defaultdict(int)
        for (um, prefix), active_count in prefix_buckets.items():
            reach_profile = self._predict_reach_profile(
                um=um,
                prefix=prefix,
                horizon_ms=horizon_ms,
            )
            if not reach_profile:
                continue
            for fn, profile in reach_profile.items():
                bounded_prob = max(0.0, min(1.0, float(profile.prob)))
                if bounded_prob < self._tau_p:
                    continue
                if profile.arrival_ms > float(horizon_ms):
                    continue
                contribution = int(math.ceil(active_count * bounded_prob))
                if contribution > 0:
                    desired[fn] += contribution
        return dict(desired)

    def _predict_reach_profile(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        horizon_ms: int,
    ) -> dict[str, ReachProfile]:
        if not prefix:
            return {}

        root_node = prefix[-1]
        frontier: list[MassFrontierState] = [
            MassFrontierState(
                prefix=prefix,
                current_node=root_node,
                prob=1.0,
                elapsed_ms=0.0,
                hop=0,
            )
        ]
        reach_prob: dict[str, float] = defaultdict(float)
        min_arrival_ms: dict[str, float] = {}

        for _ in range(self._max_prediction_hops):
            candidates: list[MassFrontierState] = []
            for state in frontier:
                outgoing = self._dag_outgoing.get(um, {}).get(state.current_node, {})
                if not outgoing:
                    continue
                child_nodes = sorted(outgoing.keys())
                dist = self._predict_distribution(
                    um=um,
                    prefix=state.prefix,
                    candidates=child_nodes,
                )
                for child in child_nodes:
                    branch_prob = float(state.prob) * float(dist.get(child, 0.0))
                    if branch_prob <= self._path_prob_floor:
                        continue
                    step_ms = self._mu_trans_ms(
                        um=um,
                        prefix=state.prefix,
                        dst_node=child,
                    )
                    # Keep existing timing semantics: first predicted hop does not include
                    # current-node execution; deeper hops include predecessor execution.
                    if state.hop > 0:
                        step_ms += self._mu_exec_ms(state.current_node)
                    arrival_ms = float(state.elapsed_ms + step_ms)
                    if arrival_ms > float(horizon_ms):
                        continue
                    candidates.append(
                        MassFrontierState(
                            prefix=tuple(list(state.prefix) + [child]),
                            current_node=child,
                            prob=branch_prob,
                            elapsed_ms=arrival_ms,
                            hop=state.hop + 1,
                        )
                    )
                    self._predictions_total += 1

            if not candidates:
                break

            candidates.sort(
                key=lambda item: (
                    -float(item.prob),
                    float(item.elapsed_ms),
                    item.current_node,
                )
            )
            covered_mass = 0.0
            next_frontier: list[MassFrontierState] = []
            for state in candidates:
                if next_frontier and covered_mass >= self._rho_mass:
                    break
                next_frontier.append(state)
                covered_mass += float(state.prob)
                reach_prob[state.current_node] += float(state.prob)
                previous = min_arrival_ms.get(state.current_node)
                if previous is None or state.elapsed_ms < previous:
                    min_arrival_ms[state.current_node] = float(state.elapsed_ms)

            if not next_frontier:
                break
            frontier = next_frontier

        profile: dict[str, ReachProfile] = {}
        for node, prob in reach_prob.items():
            if node not in min_arrival_ms:
                continue
            profile[node] = ReachProfile(
                prob=max(0.0, min(1.0, float(prob))),
                arrival_ms=max(0.0, float(min_arrival_ms[node])),
            )
        return profile

    def _predict_next(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
        dag_probs: dict[str, float],
    ) -> str:
        dist = self._predict_distribution(um=um, prefix=prefix, candidates=candidates)
        ordered = sorted(
            candidates,
            key=lambda node: (
                -dist.get(node, 0.0),
                -float(dag_probs.get(node, 0.0)),
                node,
            ),
        )
        return ordered[0]

    def _predict_distribution(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
    ) -> dict[str, float]:
        if not candidates:
            return {}
        src = prefix[-1] if prefix else "__start__"
        base = self._base_distribution(um=um, src_node=src, candidates=candidates)
        contexts = self._contexts_for_prefix(prefix)
        for context in contexts[1:]:
            metrics_map = self._transition_stats.get(um, {}).get(context, {})
            total = sum(max(0.0, metrics_map.get(c, TransitionMetrics()).weighted_count) for c in candidates)
            denom = total + self._current_beta
            if denom <= 0:
                continue
            blended: dict[str, float] = {}
            for c in candidates:
                count = max(0.0, metrics_map.get(c, TransitionMetrics()).weighted_count)
                blended[c] = (count + self._current_beta * base[c]) / denom
            base = _normalize_prob_map(blended, candidates)
        return base

    def _base_distribution(
        self,
        *,
        um: str,
        src_node: str,
        candidates: list[str],
    ) -> dict[str, float]:
        global_map = self._transition_stats.get(um, {}).get((), {})
        total = sum(max(0.0, global_map.get(c, TransitionMetrics()).weighted_count) for c in candidates)
        if total > 0:
            return {
                c: max(0.0, global_map.get(c, TransitionMetrics()).weighted_count) / total
                for c in candidates
            }
        dag_probs = self._dag_outgoing.get(um, {}).get(src_node, {})
        if dag_probs:
            raw = {c: max(self._eps, float(dag_probs.get(c, 0.0))) for c in candidates}
            return _normalize_prob_map(raw, candidates)
        uniform = 1.0 / len(candidates)
        return {c: uniform for c in candidates}

    def _contexts_for_prefix(self, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        contexts: list[tuple[str, ...]] = [()]
        lmax = min(self._current_lmax, len(prefix))
        for l in range(1, lmax + 1):
            contexts.append(prefix[-l:])
        return contexts

    def _ensure_metrics(
        self,
        *,
        um: str,
        context: tuple[str, ...],
        src_node: str,
        candidates: list[str],
    ) -> dict[str, TransitionMetrics]:
        per_um = self._transition_stats.setdefault(um, {})
        context_map = per_um.setdefault(context, {})
        dag_probs = self._dag_outgoing.get(um, {}).get(src_node, {})
        for candidate in candidates:
            if candidate in context_map:
                continue
            prior = max(self._eps, float(dag_probs.get(candidate, self._eps))) * self._prior_weight
            context_map[candidate] = TransitionMetrics(
                weighted_count=prior,
                avg_trans_latency_ms=self._default_trans_ms,
            )
        return context_map

    def _mu_exec_ms(self, function_node: str) -> float:
        stats = self._function_stats.get(function_node)
        if stats is None or stats.samples <= 0:
            return self._default_exec_ms
        return max(1.0, stats.mu_exec_ms)

    def _mu_cold_ms(self, function_node: str) -> float:
        stats = self._function_stats.get(function_node)
        if stats is None or stats.samples <= 0:
            return self._default_cold_ms
        return max(0.0, stats.mu_cold_ms)

    def _mu_trans_ms(self, *, um: str, prefix: tuple[str, ...], dst_node: str) -> float:
        for context in reversed(self._contexts_for_prefix(prefix)):
            metrics_map = self._transition_stats.get(um, {}).get(context, {})
            metric = metrics_map.get(dst_node)
            if metric is not None and metric.avg_trans_latency_ms >= 0:
                return max(0.0, metric.avg_trans_latency_ms)
        return self._default_trans_ms

    def _record_hit(self, *, timestamp_ms: int, hit: int, branch_node: str) -> None:
        sec = max(0, timestamp_ms // 1000)
        self._hit_events.append(HitEvent(sec=sec, hit=int(hit), branch_node=branch_node))
        self._hit_total += 1
        self._hit_correct += int(hit)

    def _prune_hit_events(self, *, now_sec: int) -> None:
        keep_after = now_sec - max(self._phase_window_k, self._drift_long_k) - 2
        while self._hit_events and self._hit_events[0].sec < keep_after:
            self._hit_events.popleft()

    def _update_phase(self, *, now_sec: int) -> None:
        window = [e for e in self._hit_events if e.sec >= now_sec - self._phase_window_k + 1]
        if len(window) < self._phase_n_min:
            self._set_phase("exploration")
            return
        by_sec: dict[int, list[int]] = defaultdict(list)
        for e in window:
            by_sec[e.sec].append(e.hit)
        rates = [sum(vals) / len(vals) for vals in by_sec.values() if vals]
        if not rates:
            self._set_phase("exploration")
            return
        avg = sum(rates) / len(rates)
        var = sum((r - avg) ** 2 for r in rates) / len(rates)
        if var > self._phase_var_threshold:
            self._set_phase("exploration")
            return
        self._set_phase("stable")

    def _set_phase(self, phase: str, *, force: bool = False) -> None:
        if not force and self._phase == phase:
            return
        if (not force) and self._phase != phase:
            self._phase_switches += 1
        self._phase = phase
        if phase == "stable":
            self._current_alpha = self._alpha_stable
            self._current_beta = self._beta_lo
            self._current_lmax = self._lmax_high
        else:
            self._current_alpha = self._alpha_exp
            self._current_beta = self._beta_hi
            self._current_lmax = self._lmax_low

    def _maybe_handle_drift(self, *, now_sec: int) -> None:
        short_events = [e for e in self._hit_events if e.sec >= now_sec - self._drift_short_k + 1]
        long_events = [e for e in self._hit_events if e.sec >= now_sec - self._drift_long_k + 1]
        if not short_events or not long_events:
            return
        mr_short = 1.0 - (sum(e.hit for e in short_events) / len(short_events))
        mr_long = 1.0 - (sum(e.hit for e in long_events) / len(long_events))
        triggered = (mr_short - mr_long > self._drift_delta_mr) or (mr_short > self._drift_tau_mr)
        if not triggered:
            return
        if now_sec - self._last_drift_sec < self._drift_short_k:
            return
        self._last_drift_sec = now_sec
        self._drift_events += 1

        branch_stats: dict[str, list[int]] = defaultdict(list)
        for e in short_events:
            branch_stats[e.branch_node].append(e.hit)
        drift_nodes = {
            node
            for node, values in branch_stats.items()
            if values and (1.0 - (sum(values) / len(values))) > self._drift_branch_tau
        }
        if not drift_nodes:
            all_branch_nodes = set()
            for nodes in self._branch_nodes_by_um.values():
                all_branch_nodes.update(nodes)
            drift_nodes = all_branch_nodes
        self._apply_forgetting(drift_nodes=drift_nodes)
        self._set_phase("exploration")

    def _apply_forgetting(self, *, drift_nodes: set[str]) -> None:
        for per_um in self._transition_stats.values():
            for context, metrics_map in per_um.items():
                if context:
                    if drift_nodes and context[-1] not in drift_nodes:
                        continue
                    gamma_l = self._forget_gamma ** (len(context) / max(1, self._current_lmax))
                else:
                    if drift_nodes:
                        continue
                    gamma_l = self._forget_gamma
                for metric in metrics_map.values():
                    metric.weighted_count *= gamma_l

    def _execution_scale_ms(self) -> float:
        values = [stats.mu_exec_ms for stats in self._function_stats.values() if stats.samples > 0]
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
        uni = 1.0 / max(1, len(candidates))
        return {c: uni for c in candidates}
    return {c: max(0.0, raw.get(c, 0.0)) / total for c in candidates}
