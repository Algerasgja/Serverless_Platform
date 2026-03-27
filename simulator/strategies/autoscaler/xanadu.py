from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from simulator.strategies.autoscaler.base import PrewarmPlan
from simulator.types import DagTemplate


@dataclass
class RequestPlanState:
    um: str
    enabled: bool = True
    expected_nodes: set[str] = field(default_factory=set)
    active_pairs: set[str] = field(default_factory=set)


@dataclass
class RuntimeProfile:
    exec_ms: float = 40.0
    cold_ms: float = 1200.0
    samples: int = 0


class XanaduAutoscaler:
    name = "xanadu_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        depth: int,
        ewma_alpha: float,
        sync_period_sec: int,
        allow_reentry: bool = False,
        strategy_name: str = "xanadu_v1",
        expected_topk_uncertain: int = 1,
        uncertainty_margin: float = 0.0,
        uncertainty_top1_max: float = 0.0,
        backup_on_uncertain: bool = False,
        backup_limit_per_request: int = 0,
        depth_boost_on_stable: int = 0,
        stable_hit_rate_threshold: float = 0.75,
        stable_hit_window: int = 80,
        aggressiveness: float = 1.0,
        profile_ewma_alpha: float = 0.2,
        jit_safety_ms: float = 0.0,
        jit_min_lookahead_ms: float = 0.0,
        default_exec_ms: float = 40.0,
        default_cold_ms: float = 1200.0,
        default_trans_ms: float = 8.0,
    ) -> None:
        self.name = str(strategy_name)
        self._templates = templates
        self._depth = max(0, int(depth))
        self._alpha = min(1.0, max(0.0, float(ewma_alpha)))
        self._sync_period_sec = max(1, int(sync_period_sec))
        self._allow_reentry = bool(allow_reentry)
        self._expected_topk_uncertain = max(1, int(expected_topk_uncertain))
        self._uncertainty_margin = max(0.0, float(uncertainty_margin))
        self._uncertainty_top1_max = max(0.0, min(1.0, float(uncertainty_top1_max)))
        self._backup_on_uncertain = bool(backup_on_uncertain)
        self._backup_limit_per_request = max(0, int(backup_limit_per_request))
        self._depth_boost_on_stable = max(0, int(depth_boost_on_stable))
        self._stable_hit_rate_threshold = max(0.0, min(1.0, float(stable_hit_rate_threshold)))
        self._stable_hit_window = max(1, int(stable_hit_window))
        self._aggressiveness = max(0.0, min(1.0, float(aggressiveness)))
        self._profile_alpha = max(0.0, min(1.0, float(profile_ewma_alpha)))
        self._jit_safety_ms = max(0.0, float(jit_safety_ms))
        self._jit_min_lookahead_ms = max(0.0, float(jit_min_lookahead_ms))
        self._default_exec_ms = max(1.0, float(default_exec_ms))
        self._default_cold_ms = max(1.0, float(default_cold_ms))
        self._default_trans_ms = max(0.0, float(default_trans_ms))

        self._edge_priors: dict[str, dict[str, dict[str, float]]] = {}
        self._edge_scores: dict[str, dict[str, dict[str, float]]] = {}
        self._runtime_profile: dict[str, RuntimeProfile] = {}
        self._trans_profile_ms: dict[tuple[str, str, str], float] = {}
        self._initialize_edge_scores()
        self._recent_hits: deque[int] = deque(maxlen=self._stable_hit_window)

        self._request_states: dict[str, RequestPlanState] = {}
        self._desired_by_function: dict[str, int] = {}
        self._active_pairs_total = 0

        self._predictions_total = 0
        self._offpath_disabled_requests = 0
        self._plan_pairs_added = 0
        self._plan_pairs_active_peak = 0
        self._uncertain_steps = 0
        self._backup_pairs_added = 0
        self._stable_depth_boost_steps = 0
        self._jit_gated_nodes = 0
        self._jit_selected_nodes = 0
        self._prediction_miss_reentries = 0
        self._prewarm_create_attempted = 0
        self._prewarm_created = 0
        self._prewarm_ready = 0
        self._prewarm_consumed = 0
        self._reconcile_rounds = 0
        self._desired_peak_total = 0
        self._capacity_blocked_creations = 0

    def _initialize_edge_scores(self) -> None:
        for um, template in self._templates.items():
            priors_um: dict[str, dict[str, float]] = {}
            scores_um: dict[str, dict[str, float]] = {}
            for src, outgoing in template.transitions.items():
                if not outgoing:
                    continue
                priors = {dst: float(prob) for dst, prob in outgoing.items()}
                priors_um[src] = priors
                scores_um[src] = dict(priors)
            self._edge_priors[um] = priors_um
            self._edge_scores[um] = scores_um

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
            state = RequestPlanState(
                um=um,
                enabled=True,
                expected_nodes=set(),
            )
            self._request_states[request_id] = state

        if state.expected_nodes and function_node not in state.expected_nodes:
            self._offpath_disabled_requests += 1
            state.enabled = False
            self._clear_request_pairs(request_id)
            if not self._allow_reentry:
                return
            state.enabled = True
            state.expected_nodes = set()
            self._prediction_miss_reentries += 1

        self._remove_pair(request_id, function_node)
        if not state.enabled:
            return

        predicted, expected_next, backups = self._predict_chain(
            um=state.um,
            src_node=function_node,
            depth=self._effective_depth(),
        )
        self._predictions_total += len(predicted)
        for target_node in predicted:
            self._add_pair(request_id, target_node)
        for backup_node in backups:
            self._add_pair(request_id, backup_node)
            self._backup_pairs_added += 1
        state.expected_nodes = expected_next

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
        per_um = self._edge_scores.get(um, {})
        src_scores = per_um.get(src_node)
        if not src_scores:
            return
        if len(src_scores) > 1:
            predicted = self._predict_next(um=um, src_node=src_node)
            if predicted is not None:
                self._recent_hits.append(1 if predicted == dst_node else 0)
        keep = max(0.0, 1.0 - self._alpha)
        for candidate_dst in list(src_scores.keys()):
            old = src_scores[candidate_dst]
            if candidate_dst == dst_node:
                src_scores[candidate_dst] = keep * old + self._alpha
            else:
                src_scores[candidate_dst] = keep * old

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
        del request_id, timestamp_ms
        profile = self._runtime_profile.setdefault(
            function_node,
            RuntimeProfile(
                exec_ms=self._default_exec_ms,
                cold_ms=self._default_cold_ms,
                samples=0,
            ),
        )
        observed_exec = float(max(1, int(execution_ms)))
        if profile.samples <= 0:
            profile.exec_ms = observed_exec
        else:
            profile.exec_ms = (1.0 - self._profile_alpha) * profile.exec_ms + self._profile_alpha * observed_exec

        observed_cold = float(max(0, int(cold_start_ms)))
        if observed_cold > 0:
            if profile.samples <= 0:
                profile.cold_ms = observed_cold
            else:
                profile.cold_ms = (1.0 - self._profile_alpha) * profile.cold_ms + self._profile_alpha * observed_cold
        profile.samples += 1

        if len(prefix) >= 2:
            src = str(prefix[-2])
            dst = str(prefix[-1])
            edge = (um, src, dst)
            observed_trans = float(max(0, int(transfer_ms)))
            old = self._trans_profile_ms.get(edge, self._default_trans_ms)
            self._trans_profile_ms[edge] = (1.0 - self._profile_alpha) * old + self._profile_alpha * observed_trans
        return None

    def on_request_finish(
        self,
        *,
        request_id: str,
        status: str,
        timestamp_ms: int,
    ) -> None:
        self._clear_request_pairs(request_id)
        self._request_states.pop(request_id, None)

    def on_tick(
        self,
        *,
        timestamp_sec: int,
        timestamp_ms: int,
        ready_pool_by_function: dict[str, int],
        idle_pool_by_function: dict[str, int] | None = None,
    ) -> list[PrewarmPlan]:
        del idle_pool_by_function
        if timestamp_sec % self._sync_period_sec != 0:
            return []
        self._reconcile_rounds += 1

        desired = {k: v for k, v in self._desired_by_function.items() if v > 0}
        desired_total = sum(desired.values())
        self._desired_peak_total = max(self._desired_peak_total, desired_total)

        plans: list[PrewarmPlan] = []
        for function_node, desired_count in sorted(desired.items()):
            ready = max(0, int(ready_pool_by_function.get(function_node, 0)))
            to_create = max(0, int(desired_count) - ready)
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
        self._prewarm_ready += 1

    def on_prewarm_consumed(
        self,
        *,
        function_node: str,
        request_id: str,
        container_id: str,
        timestamp_ms: int,
    ) -> None:
        self._prewarm_consumed += 1

    def summary(self) -> dict[str, Any]:
        final_desired = dict(sorted((k, v) for k, v in self._desired_by_function.items() if v > 0))
        return {
            "type": self.name,
            "parameters": {
                "depth": self._depth,
                "alpha": self._alpha,
                "sync_period_sec": self._sync_period_sec,
                "allow_reentry": self._allow_reentry,
                "expected_topk_uncertain": self._expected_topk_uncertain,
                "uncertainty_margin": self._uncertainty_margin,
                "uncertainty_top1_max": self._uncertainty_top1_max,
                "backup_on_uncertain": self._backup_on_uncertain,
                "backup_limit_per_request": self._backup_limit_per_request,
                "depth_boost_on_stable": self._depth_boost_on_stable,
                "stable_hit_rate_threshold": self._stable_hit_rate_threshold,
                "stable_hit_window": self._stable_hit_window,
                "aggressiveness": self._aggressiveness,
                "profile_ewma_alpha": self._profile_alpha,
                "jit_safety_ms": self._jit_safety_ms,
                "jit_min_lookahead_ms": self._jit_min_lookahead_ms,
            },
            "predictions_total": self._predictions_total,
            "offpath_disabled_requests": self._offpath_disabled_requests,
            "prediction_miss_reentries": self._prediction_miss_reentries,
            "plan_pairs_added": self._plan_pairs_added,
            "plan_pairs_active_peak": self._plan_pairs_active_peak,
            "uncertain_steps": self._uncertain_steps,
            "backup_pairs_added": self._backup_pairs_added,
            "stable_depth_boost_steps": self._stable_depth_boost_steps,
            "jit_gated_nodes": self._jit_gated_nodes,
            "jit_selected_nodes": self._jit_selected_nodes,
            "prewarm_create_attempted": self._prewarm_create_attempted,
            "prewarm_created": self._prewarm_created,
            "prewarm_ready": self._prewarm_ready,
            "prewarm_consumed": self._prewarm_consumed,
            "reconcile_rounds": self._reconcile_rounds,
            "desired_peak_total": self._desired_peak_total,
            "capacity_blocked_creations": self._capacity_blocked_creations,
            "final_desired_by_function": final_desired,
        }

    def _predict_chain(
        self,
        *,
        um: str,
        src_node: str,
        depth: int,
    ) -> tuple[list[str], set[str], list[str]]:
        if depth <= 0:
            return [], set(), []
        mlp_nodes: list[str] = []
        backups: list[str] = []
        expected_next: set[str] = set()
        current = src_node
        for _ in range(depth):
            ranked = self._rank_candidates(um=um, src_node=current)
            if not ranked:
                break
            nxt = ranked[0]
            if nxt is None:
                break
            mlp_nodes.append(nxt)
            if len(mlp_nodes) == 1:
                topk = self._expected_nodes_for_next_hop(
                    um=um,
                    src_node=current,
                    ranked=ranked,
                )
                expected_next.update(topk)
                if self._backup_on_uncertain and len(topk) > 1 and self._backup_limit_per_request > 0:
                    for node in topk:
                        if node == nxt:
                            continue
                        backups.append(node)
                        if len(backups) >= self._backup_limit_per_request:
                            break
            current = nxt

        if not mlp_nodes:
            return [], expected_next, backups
        plan_depth = self._aggressive_depth(len(mlp_nodes))
        selected = self._jit_select_nodes(um=um, src_node=src_node, mlp_nodes=mlp_nodes[:plan_depth])
        return selected, expected_next, backups

    def _predict_next(self, *, um: str, src_node: str) -> str | None:
        ranked = self._rank_candidates(um=um, src_node=src_node)
        return ranked[0] if ranked else None

    def _rank_candidates(self, *, um: str, src_node: str) -> list[str]:
        priors = self._edge_priors.get(um, {}).get(src_node, {})
        if not priors:
            return []
        scores = self._edge_scores.get(um, {}).get(src_node, {})
        candidates = sorted(
            priors.keys(),
            key=lambda dst: (
                -float(scores.get(dst, 0.0)),
                -float(priors.get(dst, 0.0)),
                dst,
            ),
        )
        return candidates

    def _expected_nodes_for_next_hop(
        self,
        *,
        um: str,
        src_node: str,
        ranked: list[str],
    ) -> set[str]:
        if not ranked:
            return set()
        if len(ranked) == 1:
            return {ranked[0]}

        top1 = ranked[0]
        top2 = ranked[1]
        scores = self._edge_scores.get(um, {}).get(src_node, {})
        priors = self._edge_priors.get(um, {}).get(src_node, {})
        s1 = float(scores.get(top1, priors.get(top1, 0.0)))
        s2 = float(scores.get(top2, priors.get(top2, 0.0)))
        gap = s1 - s2
        uncertain = (s1 <= self._uncertainty_top1_max) or (gap <= self._uncertainty_margin)
        if uncertain and self._expected_topk_uncertain > 1:
            self._uncertain_steps += 1
            return set(ranked[: self._expected_topk_uncertain])
        return {top1}

    def _effective_depth(self) -> int:
        if self._depth_boost_on_stable <= 0:
            return self._depth
        if len(self._recent_hits) < self._stable_hit_window:
            return self._depth
        hit_rate = sum(self._recent_hits) / len(self._recent_hits)
        if hit_rate >= self._stable_hit_rate_threshold:
            self._stable_depth_boost_steps += 1
            return self._depth + self._depth_boost_on_stable
        return self._depth

    def _aggressive_depth(self, predicted_len: int) -> int:
        if predicted_len <= 0:
            return 0
        if self._aggressiveness <= 0:
            return 0
        return max(1, min(predicted_len, int(round(predicted_len * self._aggressiveness))))

    def _jit_select_nodes(self, *, um: str, src_node: str, mlp_nodes: list[str]) -> list[str]:
        selected: list[str] = []
        if not mlp_nodes:
            return selected
        elapsed_ms = 0.0
        prev_node = src_node
        for idx, node in enumerate(mlp_nodes):
            if idx == 0:
                elapsed_ms += self._mu_exec_ms(src_node)
                elapsed_ms += self._mu_trans_ms(um=um, src_node=src_node, dst_node=node)
            else:
                elapsed_ms += self._mu_exec_ms(prev_node)
                elapsed_ms += self._mu_trans_ms(um=um, src_node=prev_node, dst_node=node)
            prev_node = node

            startup_budget_ms = self._mu_cold_ms(node) + self._jit_safety_ms
            if elapsed_ms <= startup_budget_ms and elapsed_ms >= self._jit_min_lookahead_ms:
                selected.append(node)
                self._jit_selected_nodes += 1
            else:
                self._jit_gated_nodes += 1
        return selected

    def _mu_exec_ms(self, function_node: str) -> float:
        profile = self._runtime_profile.get(function_node)
        if profile is None or profile.samples <= 0:
            return self._default_exec_ms
        return max(1.0, float(profile.exec_ms))

    def _mu_cold_ms(self, function_node: str) -> float:
        profile = self._runtime_profile.get(function_node)
        if profile is None or profile.samples <= 0:
            return self._default_cold_ms
        return max(1.0, float(profile.cold_ms))

    def _mu_trans_ms(self, *, um: str, src_node: str, dst_node: str) -> float:
        edge = (um, src_node, dst_node)
        return max(0.0, float(self._trans_profile_ms.get(edge, self._default_trans_ms)))

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
        if state is None:
            return
        if function_node not in state.active_pairs:
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


class XanaduOptimizedAutoscaler(XanaduAutoscaler):
    name = "xanadu_opt_v1"

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        depth: int,
        ewma_alpha: float,
        sync_period_sec: int,
    ) -> None:
        super().__init__(
            templates=templates,
            depth=depth,
            ewma_alpha=ewma_alpha,
            sync_period_sec=sync_period_sec,
            allow_reentry=True,
            strategy_name=self.name,
            expected_topk_uncertain=1,
            uncertainty_margin=0.0,
            uncertainty_top1_max=0.0,
            backup_on_uncertain=False,
            backup_limit_per_request=0,
            depth_boost_on_stable=0,
            stable_hit_rate_threshold=0.75,
            stable_hit_window=80,
            aggressiveness=0.75,
            profile_ewma_alpha=0.2,
            jit_safety_ms=50.0,
            jit_min_lookahead_ms=0.0,
        )
