from __future__ import annotations

import argparse
import copy
import math
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiment_common import ensure_results_dir, repo_root, resolve_config_path, write_csv  # noqa: E402
from analysis.experiments.compare_experiments import SCENARIO_FACTORS, SCENARIO_ORDER, normalize_scenarios  # noqa: E402
from simulator.config import SimulationConfig, load_config  # noqa: E402
from simulator.dag.dataset import AlibabaDatasetAdapter  # noqa: E402
from simulator.simulation import SimulationRunner  # noqa: E402
from simulator.types import DagTemplate  # noqa: E402


METHOD_NO_CONTEXT = "conscale_no_context"
METHOD_XANADU = "xanadu"
METHOD_KRAKEN = "kraken"
METHOD_CONSCALE = "conscale"

METHOD_ORDER = [
    METHOD_NO_CONTEXT,
    METHOD_XANADU,
    METHOD_KRAKEN,
    METHOD_CONSCALE,
]

METHOD_LABELS = {
    METHOD_NO_CONTEXT: "ConScale w/o Context",
    METHOD_XANADU: "Xanadu",
    METHOD_KRAKEN: "Kraken",
    METHOD_CONSCALE: "ConScale",
}

SCENARIO_LABELS = {
    "low": "Low",
    "mid": "Medium",
    "high": "High",
}


@dataclass
class TransitionMetric:
    weighted_count: float = 0.0


@dataclass
class HitEvent:
    sec: int
    hit: int
    branch_node: str


@dataclass
class MethodScore:
    method: str
    hits: int = 0
    events: int = 0

    @property
    def hit1(self) -> float:
        return (float(self.hits) / float(self.events)) if self.events else math.nan


@dataclass(frozen=True)
class ConScalePredictionParams:
    lmax_low: int
    lmax_high: int
    beta_hi: float
    beta_lo: float
    alpha_exp: float
    alpha_stable: float
    drift_enabled: bool


class NextHopPredictor:
    method: str

    def predict(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
    ) -> str | None:
        raise NotImplementedError

    def update(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        src_node: str,
        dst_node: str,
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
        hit: int | None = None,
    ) -> None:
        raise NotImplementedError


class XanaduNextHopPredictor(NextHopPredictor):
    method = METHOD_XANADU

    def __init__(self, *, templates: dict[str, DagTemplate], ewma_alpha: float) -> None:
        self._alpha = min(1.0, max(0.0, float(ewma_alpha)))
        self._edge_priors: dict[str, dict[str, dict[str, float]]] = {}
        self._edge_scores: dict[str, dict[str, dict[str, float]]] = {}
        for um, template in templates.items():
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

    def predict(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
    ) -> str | None:
        del candidates, dag_probs, timestamp_ms
        src_node = prefix[-1] if prefix else ""
        priors = self._edge_priors.get(um, {}).get(src_node, {})
        if not priors:
            return None
        scores = self._edge_scores.get(um, {}).get(src_node, {})
        ranked = sorted(
            priors.keys(),
            key=lambda dst: (
                -float(scores.get(dst, 0.0)),
                -float(priors.get(dst, 0.0)),
                dst,
            ),
        )
        return ranked[0] if ranked else None

    def update(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        src_node: str,
        dst_node: str,
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
        hit: int | None = None,
    ) -> None:
        del prefix, candidates, dag_probs, timestamp_ms, hit
        src_scores = self._edge_scores.get(um, {}).get(src_node)
        if not src_scores:
            return
        keep = max(0.0, 1.0 - self._alpha)
        for candidate_dst in list(src_scores.keys()):
            old = float(src_scores[candidate_dst])
            if candidate_dst == dst_node:
                src_scores[candidate_dst] = keep * old + self._alpha
            else:
                src_scores[candidate_dst] = keep * old


class SourceLocalNextHopPredictor(NextHopPredictor):
    method = METHOD_NO_CONTEXT

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        alpha: float,
    ) -> None:
        self._alpha = min(1.0, max(0.0, float(alpha)))
        self._edge_priors: dict[str, dict[str, dict[str, float]]] = {}
        self._edge_scores: dict[str, dict[str, dict[str, float]]] = {}
        for um, template in templates.items():
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

    def predict(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
    ) -> str | None:
        del timestamp_ms
        if not prefix:
            return None
        src_node = prefix[-1]
        scores = self._edge_scores.get(um, {}).get(src_node, {})
        if not scores:
            return None
        return sorted(
            candidates,
            key=lambda dst: (
                -float(scores.get(dst, 0.0)),
                -float(dag_probs.get(dst, 0.0)),
                dst,
            ),
        )[0]

    def update(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        src_node: str,
        dst_node: str,
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
        hit: int | None = None,
    ) -> None:
        del prefix, dag_probs, timestamp_ms, hit
        src_scores = self._edge_scores.setdefault(um, {}).setdefault(src_node, {})
        keep = max(0.0, 1.0 - self._alpha)
        for candidate_dst in candidates:
            old = float(src_scores.get(candidate_dst, 0.0))
            if candidate_dst == dst_node:
                src_scores[candidate_dst] = keep * old + self._alpha
            else:
                src_scores[candidate_dst] = keep * old


class KrakenNextHopPredictor(NextHopPredictor):
    method = METHOD_KRAKEN

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        uniform_mix: float,
        vomm_order_max: int = 2,
        vomm_context_min_count: int = 3,
        vomm_prior_weight: float = 1.0,
    ) -> None:
        self._transitions = {
            um: {src: {dst: float(prob) for dst, prob in outgoing.items()} for src, outgoing in template.transitions.items()}
            for um, template in templates.items()
        }
        self._uniform_mix = min(1.0, max(0.0, float(uniform_mix)))
        self._vomm_order_max = max(1, int(vomm_order_max))
        self._vomm_context_min_count = max(1, int(vomm_context_min_count))
        self._vomm_prior_weight = max(0.0, float(vomm_prior_weight))
        self._vomm_counts: dict[str, dict[tuple[str, ...], dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def predict(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
    ) -> str | None:
        del timestamp_ms
        if not prefix:
            return None
        dist = self._vomm_next_distribution(
            um=um,
            context=prefix,
            src_node=prefix[-1],
            candidates=candidates,
        )
        if not dist:
            return None
        return _top1_from_distribution(dist, candidates, dag_probs)

    def update(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        src_node: str,
        dst_node: str,
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
        hit: int | None = None,
    ) -> None:
        del candidates, dag_probs, timestamp_ms, hit
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
        priors = self._transitions.get(um, {}).get(src_node, {})
        if not priors:
            return {}
        uniform = 1.0 / float(len(candidates))
        prior_dist = {
            node: ((1.0 - self._uniform_mix) * float(priors.get(node, 0.0))) + (self._uniform_mix * uniform)
            for node in candidates
        }
        prior_dist = _normalize_prob_map(prior_dist, candidates)

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


class ConScaleNextHopPredictor(NextHopPredictor):
    method = METHOD_CONSCALE

    def __init__(
        self,
        *,
        templates: dict[str, DagTemplate],
        lmax_low: int,
        lmax_high: int,
        beta_hi: float,
        beta_lo: float,
        alpha_exp: float,
        alpha_stable: float,
        phase_window_k: int,
        phase_n_min: int,
        phase_var_threshold: float,
        drift_short_k: int,
        drift_long_k: int,
        drift_delta_mr: float,
        drift_tau_mr: float,
        drift_branch_tau: float,
        forget_gamma: float,
        drift_enabled: bool = True,
        no_context: bool = False,
    ) -> None:
        self.method = METHOD_NO_CONTEXT if no_context else METHOD_CONSCALE
        self._lmax_low = 1 if no_context else max(1, int(lmax_low))
        self._lmax_high = 1 if no_context else max(self._lmax_low, int(lmax_high))
        self._beta_hi = max(1e-6, float(beta_hi))
        self._beta_lo = max(1e-6, float(beta_lo))
        self._alpha_exp = min(1.0, max(0.0, float(alpha_exp)))
        self._alpha_stable = min(1.0, max(0.0, float(alpha_stable)))
        self._phase_window_k = max(1, int(phase_window_k))
        self._phase_n_min = max(1, int(phase_n_min))
        self._phase_var_threshold = max(0.0, float(phase_var_threshold))
        self._drift_short_k = max(1, int(drift_short_k))
        self._drift_long_k = max(self._drift_short_k, int(drift_long_k))
        self._drift_delta_mr = max(0.0, float(drift_delta_mr))
        self._drift_tau_mr = max(0.0, float(drift_tau_mr))
        self._drift_branch_tau = max(0.0, float(drift_branch_tau))
        self._forget_gamma = min(1.0, max(1e-6, float(forget_gamma)))
        self._drift_enabled = bool(drift_enabled)
        self._eps = 1e-9
        self._prior_weight = 1.0

        self._dag_outgoing = {
            um: {src: {dst: float(prob) for dst, prob in outgoing.items()} for src, outgoing in template.transitions.items()}
            for um, template in templates.items()
        }
        self._branch_nodes_by_um: dict[str, set[str]] = {}
        self._transition_stats: dict[str, dict[tuple[str, ...], dict[str, TransitionMetric]]] = {}
        self._bootstrap_transition_priors(templates)

        self._hit_events: deque[HitEvent] = deque()
        self._last_drift_sec = -10**9
        self._phase = "exploration"
        self._current_lmax = self._lmax_low
        self._current_beta = self._beta_hi
        self._current_alpha = self._alpha_exp

    def _bootstrap_transition_priors(self, templates: dict[str, DagTemplate]) -> None:
        for um, template in templates.items():
            per_um: dict[tuple[str, ...], dict[str, TransitionMetric]] = defaultdict(dict)
            branch_nodes: set[str] = set()
            for src, outgoing in template.transitions.items():
                if not outgoing:
                    continue
                if len(outgoing) > 1:
                    branch_nodes.add(src)
                c0 = per_um[()]
                c1 = per_um[(src,)]
                for dst, prob in outgoing.items():
                    prior = max(self._eps, float(prob)) * self._prior_weight
                    c0.setdefault(dst, TransitionMetric()).weighted_count += prior
                    c1.setdefault(dst, TransitionMetric()).weighted_count += prior
            self._transition_stats[um] = dict(per_um)
            self._branch_nodes_by_um[um] = branch_nodes

    def predict(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
    ) -> str | None:
        if not prefix:
            return None
        self._advance(timestamp_ms=timestamp_ms)
        dist = self._predict_distribution(um=um, prefix=prefix, candidates=candidates)
        if not dist:
            return None
        return _top1_from_distribution(dist, candidates, dag_probs)

    def update(
        self,
        *,
        um: str,
        prefix: tuple[str, ...],
        src_node: str,
        dst_node: str,
        candidates: list[str],
        dag_probs: dict[str, float],
        timestamp_ms: int,
        hit: int | None = None,
    ) -> None:
        self._advance(timestamp_ms=timestamp_ms)
        if hit is not None:
            self._record_hit(timestamp_ms=timestamp_ms, hit=hit, branch_node=src_node)
        contexts = self._contexts_for_prefix(prefix)
        for context in contexts:
            metrics_map = self._ensure_metrics(
                um=um,
                context=context,
                src_node=src_node,
                candidates=candidates,
                dag_probs=dag_probs,
            )
            for candidate in candidates:
                metric = metrics_map[candidate]
                obs = 1.0 if candidate == dst_node else 0.0
                metric.weighted_count = (
                    self._current_alpha * obs + (1.0 - self._current_alpha) * metric.weighted_count
                )

    def _advance(self, *, timestamp_ms: int) -> None:
        now_sec = max(0, int(timestamp_ms) // 1000)
        self._prune_hit_events(now_sec=now_sec)
        self._update_phase(now_sec=now_sec)
        self._maybe_handle_drift(now_sec=now_sec)

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
        for context in contexts[2:]:
            metrics_map = self._transition_stats.get(um, {}).get(context, {})
            total = sum(max(0.0, metrics_map.get(c, TransitionMetric()).weighted_count) for c in candidates)
            denom = total + self._current_beta
            if denom <= 0:
                continue
            blended = {
                c: (max(0.0, metrics_map.get(c, TransitionMetric()).weighted_count) + self._current_beta * base[c])
                / denom
                for c in candidates
            }
            base = _normalize_prob_map(blended, candidates)
        return base

    def _base_distribution(self, *, um: str, src_node: str, candidates: list[str]) -> dict[str, float]:
        source_map = self._transition_stats.get(um, {}).get((src_node,), {})
        total = sum(max(0.0, source_map.get(c, TransitionMetric()).weighted_count) for c in candidates)
        if total > 0:
            return {
                c: max(0.0, source_map.get(c, TransitionMetric()).weighted_count) / total
                for c in candidates
            }
        dag_probs = self._dag_outgoing.get(um, {}).get(src_node, {})
        if dag_probs:
            return _normalize_prob_map({c: max(self._eps, float(dag_probs.get(c, 0.0))) for c in candidates}, candidates)
        uniform = 1.0 / len(candidates)
        return {c: uniform for c in candidates}

    def _contexts_for_prefix(self, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        contexts: list[tuple[str, ...]] = [()]
        lmax = min(self._current_lmax, len(prefix))
        for length in range(1, lmax + 1):
            contexts.append(prefix[-length:])
        return contexts

    def _ensure_metrics(
        self,
        *,
        um: str,
        context: tuple[str, ...],
        src_node: str,
        candidates: list[str],
        dag_probs: dict[str, float],
    ) -> dict[str, TransitionMetric]:
        per_um = self._transition_stats.setdefault(um, {})
        context_map = per_um.setdefault(context, {})
        for candidate in candidates:
            if candidate in context_map:
                continue
            prior = max(self._eps, float(dag_probs.get(candidate, self._eps))) * self._prior_weight
            context_map[candidate] = TransitionMetric(weighted_count=prior)
        return context_map

    def _record_hit(self, *, timestamp_ms: int, hit: int, branch_node: str) -> None:
        sec = max(0, int(timestamp_ms) // 1000)
        self._hit_events.append(HitEvent(sec=sec, hit=int(hit), branch_node=branch_node))

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
        for event in window:
            by_sec[event.sec].append(event.hit)
        rates = [sum(vals) / len(vals) for vals in by_sec.values() if vals]
        if not rates:
            self._set_phase("exploration")
            return
        avg = sum(rates) / len(rates)
        var = sum((rate - avg) ** 2 for rate in rates) / len(rates)
        self._set_phase("exploration" if var > self._phase_var_threshold else "stable")

    def _set_phase(self, phase: str) -> None:
        if self._phase == phase:
            return
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
        if not self._drift_enabled:
            return
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

        branch_stats: dict[str, list[int]] = defaultdict(list)
        for event in short_events:
            branch_stats[event.branch_node].append(event.hit)
        drift_nodes = {
            node
            for node, values in branch_stats.items()
            if values and (1.0 - (sum(values) / len(values))) > self._drift_branch_tau
        }
        if not drift_nodes:
            drift_nodes = {node for nodes in self._branch_nodes_by_um.values() for node in nodes}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic-DAG branching next-hop Hit@1 for context-aware predictors."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=["low", "mid", "high"],
        default=None,
        help="Scenario subset. Omitted means low/mid/high all.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Repeated-run seeds.")
    parser.add_argument(
        "--results-dir",
        default="results/prediction_accuracy/latest",
        help="Output directory.",
    )
    parser.add_argument(
        "--conscale-lmax-low",
        type=int,
        default=2,
        help="Prediction-only ConScale exploration context length.",
    )
    parser.add_argument(
        "--conscale-lmax-high",
        type=int,
        default=4,
        help="Prediction-only ConScale stable context length.",
    )
    parser.add_argument(
        "--conscale-beta-hi",
        type=float,
        default=0.1,
        help="Prediction-only ConScale exploration smoothing weight.",
    )
    parser.add_argument(
        "--conscale-beta-lo",
        type=float,
        default=0.1,
        help="Prediction-only ConScale stable smoothing weight.",
    )
    parser.add_argument(
        "--conscale-alpha-exp",
        type=float,
        default=0.12,
        help="Prediction-only ConScale exploration EWMA alpha.",
    )
    parser.add_argument(
        "--conscale-alpha-stable",
        type=float,
        default=0.12,
        help="Prediction-only ConScale stable EWMA alpha.",
    )
    parser.add_argument(
        "--conscale-drift-mode",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Enable or disable ConScale drift forgetting inside the prediction probe.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Write CSV files only.")
    return parser.parse_args()


def build_conscale_prediction_params(args: argparse.Namespace) -> ConScalePredictionParams:
    return ConScalePredictionParams(
        lmax_low=max(1, int(args.conscale_lmax_low)),
        lmax_high=max(max(1, int(args.conscale_lmax_low)), int(args.conscale_lmax_high)),
        beta_hi=max(1e-6, float(args.conscale_beta_hi)),
        beta_lo=max(1e-6, float(args.conscale_beta_lo)),
        alpha_exp=min(1.0, max(0.0, float(args.conscale_alpha_exp))),
        alpha_stable=min(1.0, max(0.0, float(args.conscale_alpha_stable))),
        drift_enabled=(str(args.conscale_drift_mode).lower() == "enabled"),
    )


def build_predictors(
    *,
    config: SimulationConfig,
    templates: dict[str, DagTemplate],
    conscale_params: ConScalePredictionParams,
) -> list[NextHopPredictor]:
    autoscaler = config.autoscaler
    return [
        SourceLocalNextHopPredictor(
            templates=templates,
            alpha=conscale_params.alpha_exp,
        ),
        XanaduNextHopPredictor(
            templates=templates,
            ewma_alpha=autoscaler.xanadu_ewma_alpha,
        ),
        KrakenNextHopPredictor(
            templates=templates,
            uniform_mix=autoscaler.kraken_uniform_mix,
        ),
        ConScaleNextHopPredictor(
            templates=templates,
            lmax_low=conscale_params.lmax_low,
            lmax_high=conscale_params.lmax_high,
            beta_hi=conscale_params.beta_hi,
            beta_lo=conscale_params.beta_lo,
            alpha_exp=conscale_params.alpha_exp,
            alpha_stable=conscale_params.alpha_stable,
            phase_window_k=autoscaler.hpwp_phase_window_k,
            phase_n_min=autoscaler.hpwp_phase_n_min,
            phase_var_threshold=autoscaler.hpwp_phase_var_threshold,
            drift_short_k=autoscaler.hpwp_drift_short_k,
            drift_long_k=autoscaler.hpwp_drift_long_k,
            drift_delta_mr=autoscaler.hpwp_drift_delta_mr,
            drift_tau_mr=autoscaler.hpwp_drift_tau_mr,
            drift_branch_tau=autoscaler.hpwp_drift_branch_tau,
            forget_gamma=autoscaler.hpwp_forget_gamma,
            drift_enabled=conscale_params.drift_enabled,
            no_context=False,
        ),
    ]


class PredictionAccuracyProbe:
    def __init__(
        self,
        *,
        config: SimulationConfig,
        templates: dict[str, DagTemplate],
        conscale_params: ConScalePredictionParams,
    ) -> None:
        self._templates = templates
        self._predictors = build_predictors(
            config=config,
            templates=templates,
            conscale_params=conscale_params,
        )
        self._score_by_method = {
            predictor.method: MethodScore(method=predictor.method) for predictor in self._predictors
        }
        self._prefix_score: dict[tuple[str, str], MethodScore] = {}
        self._branch_score: dict[tuple[str, str, str], MethodScore] = {}
        self.transition_events = 0
        self.branch_events = 0

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
        del request_id, transfer_ms
        template = self._templates.get(um)
        if template is None:
            return
        outgoing = template.transitions.get(src_node, {})
        if not outgoing:
            return

        self.transition_events += 1
        candidates = sorted(outgoing.keys())
        if dst_node not in outgoing:
            candidates = sorted(set(candidates + [dst_node]))
            dag_probs = dict(outgoing)
            dag_probs[dst_node] = 1e-9
        else:
            dag_probs = dict(outgoing)

        prefix_tuple = tuple(prefix or ())
        if not prefix_tuple or prefix_tuple[-1] != src_node:
            prefix_tuple = tuple(list(prefix_tuple) + [src_node]) if prefix_tuple else (src_node,)

        branch_event = len(candidates) > 1
        if branch_event:
            self.branch_events += 1
        prefix_bucket = _prefix_bucket(len(prefix_tuple))

        for predictor in self._predictors:
            hit: int | None = None
            if branch_event:
                predicted = predictor.predict(
                    um=um,
                    prefix=prefix_tuple,
                    candidates=candidates,
                    dag_probs=dag_probs,
                    timestamp_ms=timestamp_ms,
                )
                hit = int(predicted == dst_node)
                score = self._score_by_method[predictor.method]
                score.events += 1
                score.hits += hit
                prefix_score = self._prefix_score.setdefault(
                    (predictor.method, prefix_bucket),
                    MethodScore(method=predictor.method),
                )
                prefix_score.events += 1
                prefix_score.hits += hit
                branch_score = self._branch_score.setdefault(
                    (predictor.method, um, src_node),
                    MethodScore(method=predictor.method),
                )
                branch_score.events += 1
                branch_score.hits += hit
            predictor.update(
                um=um,
                prefix=prefix_tuple,
                src_node=src_node,
                dst_node=dst_node,
                candidates=candidates,
                dag_probs=dag_probs,
                timestamp_ms=timestamp_ms,
                hit=hit,
            )

    def scores(self) -> list[MethodScore]:
        return [self._score_by_method[method] for method in METHOD_ORDER]

    def prefix_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (method, prefix_len_bucket), score in self._prefix_score.items():
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "prefix_len_bucket": prefix_len_bucket,
                    "events": score.events,
                    "hits": score.hits,
                    "hit1": score.hit1,
                    "hit1_pct": score.hit1 * 100.0 if not math.isnan(score.hit1) else math.nan,
                }
            )
        rows.sort(key=lambda row: (_method_index(str(row["method"])), _prefix_bucket_index(str(row["prefix_len_bucket"]))))
        return rows

    def branch_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (method, um, branch_node), score in self._branch_score.items():
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "um": um,
                    "branch_node": branch_node,
                    "events": score.events,
                    "hits": score.hits,
                    "hit1": score.hit1,
                    "hit1_pct": score.hit1 * 100.0 if not math.isnan(score.hit1) else math.nan,
                }
            )
        rows.sort(
            key=lambda row: (
                _method_index(str(row["method"])),
                str(row["um"]),
                str(row["branch_node"]),
            )
        )
        return rows


def run_prediction_scenario(
    *,
    base_config: SimulationConfig,
    scenario: str,
    rate_factor: float,
    seed: int,
    run_output_root: Path,
    conscale_params: ConScalePredictionParams,
) -> tuple[list[MethodScore], int, int, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    config = copy.deepcopy(base_config)
    config.experiment.random_seed = int(seed)
    config.experiment.name = f"{base_config.experiment.name}_prediction_hit1_{scenario}_seed{seed}"
    config.workload.rate_multiplier = float(base_config.workload.rate_multiplier) * float(rate_factor)
    config.output.runs_dir = str(run_output_root)
    corpus = AlibabaDatasetAdapter(config.dataset, config.workload).load_corpus()
    probe = PredictionAccuracyProbe(
        config=config,
        templates=corpus.templates,
        conscale_params=conscale_params,
    )
    runner = SimulationRunner(config, corpus, transition_observers=[probe])
    run_dir = Path(runner.run())
    return probe.scores(), probe.transition_events, probe.branch_events, run_dir, probe.prefix_rows(), probe.branch_rows()


def aggregate_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["scenario"]), str(row["method"]))].append(row)

    out: list[dict[str, Any]] = []
    for (scenario, method), rows in grouped.items():
        hit1_values = [float(row["hit1"]) for row in rows if not math.isnan(float(row["hit1"]))]
        transition_values = [float(row["transition_events"]) for row in rows]
        branch_values = [float(row["branching_events"]) for row in rows]
        event_values = [float(row["events"]) for row in rows]
        hit_values = [float(row["hits"]) for row in rows]
        hit1_mean, hit1_std = _mean_std(hit1_values)
        transitions_mean, transitions_std = _mean_std(transition_values)
        branches_mean, branches_std = _mean_std(branch_values)
        events_mean, events_std = _mean_std(event_values)
        hits_mean, hits_std = _mean_std(hit_values)
        out.append(
            {
                "scenario": scenario,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seeds": len(rows),
                "transition_events_mean": transitions_mean,
                "transition_events_std": transitions_std,
                "branching_events_mean": branches_mean,
                "branching_events_std": branches_std,
                "events_mean": events_mean,
                "events_std": events_std,
                "hits_mean": hits_mean,
                "hits_std": hits_std,
                "hit1_mean": hit1_mean,
                "hit1_std": hit1_std,
                "hit1_pct_mean": hit1_mean * 100.0 if not math.isnan(hit1_mean) else math.nan,
                "hit1_pct_std": hit1_std * 100.0 if not math.isnan(hit1_std) else math.nan,
            }
        )
    out.sort(key=lambda row: (_scenario_index(str(row["scenario"])), _method_index(str(row["method"]))))
    return out


def aggregate_diagnostic_rows(rows: list[dict[str, Any]], *, group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_values, group_rows in grouped.items():
        item = {key: value for key, value in zip(group_keys, key_values)}
        method = str(item.get("method", ""))
        hit1_values = [float(row["hit1"]) for row in group_rows if not math.isnan(float(row["hit1"]))]
        event_values = [float(row["events"]) for row in group_rows]
        hit_values = [float(row["hits"]) for row in group_rows]
        hit1_mean, hit1_std = _mean_std(hit1_values)
        events_mean, events_std = _mean_std(event_values)
        hits_mean, hits_std = _mean_std(hit_values)
        item.update(
            {
                "method_label": METHOD_LABELS.get(method, method),
                "seeds": len(group_rows),
                "events_mean": events_mean,
                "events_std": events_std,
                "hits_mean": hits_mean,
                "hits_std": hits_std,
                "hit1_mean": hit1_mean,
                "hit1_std": hit1_std,
                "hit1_pct_mean": hit1_mean * 100.0 if not math.isnan(hit1_mean) else math.nan,
                "hit1_pct_std": hit1_std * 100.0 if not math.isnan(hit1_std) else math.nan,
            }
        )
        out.append(item)

    out.sort(key=_diagnostic_sort_key)
    return out


def plot_hit1(aggregate_rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install with .[analysis]") from exc

    row_map = {(str(row["scenario"]), str(row["method"])): row for row in aggregate_rows}
    scenarios = [scenario for scenario in SCENARIO_ORDER if any((scenario, method) in row_map for method in METHOD_ORDER)]
    if not scenarios:
        raise RuntimeError("no aggregate rows available for plotting")

    x = list(range(len(scenarios)))
    width = 0.18
    offsets = {
        METHOD_NO_CONTEXT: -1.5 * width,
        METHOD_XANADU: -0.5 * width,
        METHOD_KRAKEN: 0.5 * width,
        METHOD_CONSCALE: 1.5 * width,
    }
    colors = {
        METHOD_NO_CONTEXT: "#8A8F98",
        METHOD_XANADU: "#287271",
        METHOD_KRAKEN: "#2A9D8C",
        METHOD_CONSCALE: "#264653",
    }

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for method in METHOD_ORDER:
        vals: list[float] = []
        errs: list[float] = []
        for scenario in scenarios:
            row = row_map.get((scenario, method))
            vals.append(float(row["hit1_pct_mean"]) if row else math.nan)
            errs.append(float(row["hit1_pct_std"]) if row else 0.0)
        positions = [pos + offsets[method] for pos in x]
        ax.bar(
            positions,
            vals,
            width=width,
            yerr=errs,
            capsize=3,
            label=METHOD_LABELS[method],
            color=colors[method],
            edgecolor="#2F3E46",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s.title()) for s in scenarios], fontsize=13)
    ax.set_ylabel("Next-hop Hit@1 (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, float):
                item[key] = "" if math.isnan(value) else f"{value:.8f}"
        formatted.append(item)
    return formatted


def _top1_from_distribution(dist: dict[str, float], candidates: list[str], dag_probs: dict[str, float]) -> str | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda node: (
            -float(dist.get(node, 0.0)),
            -float(dag_probs.get(node, 0.0)),
            node,
        ),
    )
    return ordered[0]


def _normalize_prob_map(raw: dict[str, float], candidates: list[str]) -> dict[str, float]:
    total = sum(max(0.0, float(raw.get(candidate, 0.0))) for candidate in candidates)
    if total <= 0:
        uniform = 1.0 / float(max(1, len(candidates)))
        return {candidate: uniform for candidate in candidates}
    return {candidate: max(0.0, float(raw.get(candidate, 0.0))) / total for candidate in candidates}


def _mean_std(values: list[float]) -> tuple[float, float]:
    cleaned = [float(value) for value in values if not math.isnan(float(value))]
    if not cleaned:
        return math.nan, math.nan
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    return float(statistics.fmean(cleaned)), float(statistics.stdev(cleaned))


def _scenario_index(scenario: str) -> int:
    return SCENARIO_ORDER.index(scenario) if scenario in SCENARIO_ORDER else 999


def _method_index(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 999


def _prefix_bucket(prefix_len: int) -> str:
    if prefix_len <= 1:
        return "1"
    if prefix_len == 2:
        return "2"
    if prefix_len == 3:
        return "3"
    return "4+"


def _prefix_bucket_index(bucket: str) -> int:
    order = {"1": 1, "2": 2, "3": 3, "4+": 4}
    return order.get(bucket, 999)


def _diagnostic_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _scenario_index(str(row.get("scenario", ""))),
        _method_index(str(row.get("method", ""))),
        _prefix_bucket_index(str(row.get("prefix_len_bucket", ""))),
        str(row.get("um", "")),
        str(row.get("branch_node", "")),
    )


def main() -> int:
    args = parse_args()
    root = repo_root()
    config_path = resolve_config_path(args.config, root=root)
    base_config = load_config(config_path)
    conscale_params = build_conscale_prediction_params(args)
    selected_scenarios = normalize_scenarios(args.scenarios)
    selected_factors = [(name, factor) for name, factor in SCENARIO_FACTORS if name in selected_scenarios]
    results_dir = ensure_results_dir(root=root, relative=str(args.results_dir))
    run_output_root = results_dir / "runs"
    run_output_root.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    prefix_diag_rows: list[dict[str, Any]] = []
    branch_diag_rows: list[dict[str, Any]] = []
    for scenario, factor in selected_factors:
        for seed in [int(item) for item in args.seeds]:
            scores, transition_events, branch_events, run_dir, prefix_rows, branch_rows = run_prediction_scenario(
                base_config=base_config,
                scenario=scenario,
                rate_factor=float(factor),
                seed=seed,
                run_output_root=run_output_root,
                conscale_params=conscale_params,
            )
            for score in scores:
                seed_rows.append(
                    {
                        "scenario": scenario,
                        "rate_multiplier": float(base_config.workload.rate_multiplier) * float(factor),
                        "seed": seed,
                        "method": score.method,
                        "method_label": METHOD_LABELS.get(score.method, score.method),
                        "transition_events": transition_events,
                        "branching_events": branch_events,
                        "events": score.events,
                        "hits": score.hits,
                        "hit1": score.hit1,
                        "hit1_pct": score.hit1 * 100.0 if not math.isnan(score.hit1) else math.nan,
                        "run_dir": str(run_dir),
                    }
                )
            for row in prefix_rows:
                prefix_diag_rows.append(
                    {
                        "scenario": scenario,
                        "rate_multiplier": float(base_config.workload.rate_multiplier) * float(factor),
                        "seed": seed,
                        **row,
                    }
                )
            for row in branch_rows:
                branch_diag_rows.append(
                    {
                        "scenario": scenario,
                        "rate_multiplier": float(base_config.workload.rate_multiplier) * float(factor),
                        "seed": seed,
                        **row,
                    }
                )
            print(
                "[prediction-accuracy] "
                f"scenario={scenario} seed={seed} "
                f"transitions={transition_events} branches={branch_events} run={run_dir}"
            )

    aggregate_rows = aggregate_seed_rows(seed_rows)
    by_seed_path = results_dir / "prediction_hit1_by_seed.csv"
    aggregate_path = results_dir / "prediction_hit1.csv"
    write_csv(by_seed_path, _format_rows(seed_rows))
    write_csv(aggregate_path, _format_rows(aggregate_rows))
    print(f"[prediction-accuracy] saved seed metrics: {by_seed_path}")
    print(f"[prediction-accuracy] saved aggregate metrics: {aggregate_path}")

    diagnostics_dir = results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    prefix_by_seed_path = diagnostics_dir / "hit1_by_prefix_len_by_seed.csv"
    prefix_aggregate_path = diagnostics_dir / "hit1_by_prefix_len.csv"
    branch_by_seed_path = diagnostics_dir / "hit1_by_branch_node_by_seed.csv"
    branch_aggregate_path = diagnostics_dir / "hit1_by_branch_node.csv"
    write_csv(prefix_by_seed_path, _format_rows(prefix_diag_rows))
    write_csv(
        prefix_aggregate_path,
        _format_rows(
            aggregate_diagnostic_rows(
                prefix_diag_rows,
                group_keys=["scenario", "method", "prefix_len_bucket"],
            )
        ),
    )
    write_csv(branch_by_seed_path, _format_rows(branch_diag_rows))
    write_csv(
        branch_aggregate_path,
        _format_rows(
            aggregate_diagnostic_rows(
                branch_diag_rows,
                group_keys=["scenario", "method", "um", "branch_node"],
            )
        ),
    )
    print(f"[prediction-accuracy] saved diagnostics: {diagnostics_dir}")

    if not args.no_plot:
        fig_path = results_dir / "figures" / "next_hop_hit1.png"
        plot_hit1(aggregate_rows, fig_path)
        print(f"[prediction-accuracy] saved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
