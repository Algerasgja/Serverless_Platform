from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

from simulator.types import DagTemplate

START_NODE = "__start__"


@dataclass
class SessionContext:
    session_id: str
    um: str
    last_seen_sec: int
    edge_counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class DagPathModel:
    um: str
    seed: int
    split_nodes: list[str]
    split_index: dict[str, int]
    branches_by_split: list[list[str]]
    base_probs_by_split: list[list[float]]
    theta: list[list[list[float]]]
    coupling: dict[tuple[int, int], list[list[float]]]
    pi_fixed: list[float]
    pi_cur: list[float]
    pi_target: list[float]
    pi_initial: list[float]
    last_update_sec: int
    next_refresh_sec: int


class ConditionalDagEngine:
    """Generate exactly one execution path per request for a selected UM."""

    def __init__(
        self,
        templates: dict[str, DagTemplate],
        *,
        session_gap_sec: int,
        session_continue_prob: float,
        context_alpha: float,
        rng: random.Random,
        path_rule: str = "critical_path",
        context_regime: str = "fixed",
        mode_count: int = 3,
        mode_prior_concentration: float = 2.0,
        mode_strength: float = 1.0,
        prefix_strength: float = 1.2,
        prefix_decay: float = 0.85,
        prefix_window: int = 3,
        temperature: float = 1.0,
        coupling_seed_offset: int = 707,
        drifting_interval_sec: int = 30,
        drifting_strength: float = 0.08,
        drifting_concentration: float = 200.0,
        drifting_floor: float = 1e-3,
        eps: float = 1e-9,
        base_seed: int = 0,
    ) -> None:
        if not templates:
            raise ValueError("at least one template is required")
        self._templates = templates
        self._session_gap_sec = max(1, session_gap_sec)
        self._session_continue_prob = min(1.0, max(0.0, session_continue_prob))
        self._context_alpha = max(0.0, context_alpha)
        self._rng = rng
        self._session_counter = 0
        self._sessions: dict[str, SessionContext] = {}
        self._sessions_by_um: dict[str, set[str]] = defaultdict(set)

        self._path_rule = str(path_rule or "critical_path").lower()
        self._context_regime = str(context_regime or "fixed").lower()
        if self._context_regime not in {"fixed", "drifting"}:
            self._context_regime = "fixed"
        self._mode_count = max(1, int(mode_count))
        self._mode_prior_concentration = max(1e-6, float(mode_prior_concentration))
        self._mode_strength = float(mode_strength)
        self._prefix_strength = float(prefix_strength)
        self._prefix_decay = max(0.0, float(prefix_decay))
        self._prefix_window = max(0, int(prefix_window))
        self._temperature = max(1e-6, float(temperature))
        self._coupling_seed_offset = int(coupling_seed_offset)
        self._drifting_interval_sec = max(1, int(drifting_interval_sec))
        self._drifting_strength = min(1.0, max(0.0, float(drifting_strength)))
        self._drifting_concentration = max(0.0, float(drifting_concentration))
        self._drifting_floor = max(0.0, float(drifting_floor))
        self._eps = max(1e-12, float(eps))
        self._base_seed = int(base_seed)

        self._path_models: dict[str, DagPathModel] = {}
        if self._path_rule == "mode_prefix_coupled_v1":
            self._path_models = self._build_path_models()

    @property
    def ums(self) -> list[str]:
        return list(self._templates.keys())

    def assign_session(self, um: str, now_sec: int) -> str:
        self._evict_stale_sessions(um, now_sec)
        candidates = list(self._sessions_by_um[um])
        if candidates and self._rng.random() < self._session_continue_prob:
            chosen = self._rng.choice(candidates)
            self._sessions[chosen].last_seen_sec = now_sec
            return chosen
        self._session_counter += 1
        sid = f"s-{um}-{self._session_counter}"
        self._sessions[sid] = SessionContext(session_id=sid, um=um, last_seen_sec=now_sec)
        self._sessions_by_um[um].add(sid)
        return sid

    def generate_path(self, um: str, session_id: str, now_sec: int | None = None) -> list[str]:
        if self._path_rule == "mode_prefix_coupled_v1":
            return self._generate_mode_prefix_path(um, now_sec=0 if now_sec is None else now_sec)
        return self._generate_critical_path(um, session_id)

    def path_model_summary(self) -> dict[str, object]:
        if self._path_rule != "mode_prefix_coupled_v1":
            return {
                "path_rule": self._path_rule,
                "context_regime": self._context_regime,
                "enabled": False,
            }
        per_um: dict[str, object] = {}
        for um in sorted(self._path_models.keys()):
            model = self._path_models[um]
            item: dict[str, object] = {
                "seed": model.seed,
                "split_count": len(model.split_nodes),
                "mode_count": self._mode_count,
                "pi_fixed": [_round6(x) for x in model.pi_fixed],
            }
            if self._context_regime == "drifting":
                item["pi_initial"] = [_round6(x) for x in model.pi_initial]
                item["pi_current"] = [_round6(x) for x in model.pi_cur]
            per_um[um] = item
        return {
            "path_rule": self._path_rule,
            "context_regime": self._context_regime,
            "enabled": True,
            "parameters": {
                "mode_count": self._mode_count,
                "mode_prior_concentration": self._mode_prior_concentration,
                "mode_strength": self._mode_strength,
                "prefix_strength": self._prefix_strength,
                "prefix_decay": self._prefix_decay,
                "prefix_window": self._prefix_window,
                "temperature": self._temperature,
                "coupling_seed_offset": self._coupling_seed_offset,
                "drifting_interval_sec": self._drifting_interval_sec,
                "drifting_strength": self._drifting_strength,
                "drifting_concentration": self._drifting_concentration,
                "drifting_floor": self._drifting_floor,
                "eps": self._eps,
            },
            "per_um": per_um,
        }

    def _generate_critical_path(self, um: str, session_id: str) -> list[str]:
        template = self._templates[um]
        session = self._sessions.get(session_id)
        if session is None:
            session = SessionContext(session_id=session_id, um=um, last_seen_sec=0)
            self._sessions[session_id] = session
            self._sessions_by_um[um].add(session_id)

        node = START_NODE
        path: list[str] = []
        traversed_edges: list[tuple[str, str]] = []

        for _ in range(128):
            outgoing = template.transitions.get(node, {})
            if not outgoing:
                if node != START_NODE:
                    path.append(node)
                break
            if node != START_NODE:
                path.append(node)
            next_node = self._select_next_node_critical(node, outgoing, session)
            traversed_edges.append((node, next_node))
            node = next_node
        else:
            if node != START_NODE:
                path.append(node)

        for edge in traversed_edges:
            session.edge_counts[edge] = session.edge_counts.get(edge, 0) + 1
        return path

    def _generate_mode_prefix_path(self, um: str, now_sec: int) -> list[str]:
        template = self._templates[um]
        model = self._path_models[um]
        mode_idx = self._sample_mode(model, now_sec)
        node = START_NODE
        path: list[str] = []
        prefix_history: list[tuple[int, int]] = []

        for _ in range(128):
            outgoing = template.transitions.get(node, {})
            if not outgoing:
                if node != START_NODE:
                    path.append(node)
                break
            if node != START_NODE:
                path.append(node)
            if len(outgoing) == 1:
                node = next(iter(outgoing.keys()))
                continue

            split_idx = model.split_index.get(node)
            if split_idx is None:
                node = self._sample_outgoing(outgoing)
                continue
            branches = model.branches_by_split[split_idx]
            base_probs = model.base_probs_by_split[split_idx]

            scores: list[float] = []
            for branch_idx, _ in enumerate(branches):
                prefix_term = self._prefix_term(
                    model=model,
                    prefix_history=prefix_history,
                    current_split_idx=split_idx,
                    current_branch_idx=branch_idx,
                )
                score = (
                    math.log(max(base_probs[branch_idx], self._eps))
                    + self._mode_strength * model.theta[mode_idx][split_idx][branch_idx]
                    + self._prefix_strength * prefix_term
                )
                scores.append(score)

            probs = _softmax(scores, self._temperature)
            chosen_idx = _sample_categorical(probs, self._rng)
            prefix_history.append((split_idx, chosen_idx))
            node = branches[chosen_idx]
        else:
            if node != START_NODE:
                path.append(node)
        return path

    def _prefix_term(
        self,
        *,
        model: DagPathModel,
        prefix_history: list[tuple[int, int]],
        current_split_idx: int,
        current_branch_idx: int,
    ) -> float:
        if self._prefix_window <= 0 or not prefix_history:
            return 0.0
        term = 0.0
        for src_split_idx, src_branch_idx in prefix_history[-self._prefix_window :]:
            gap = current_split_idx - src_split_idx
            if gap <= 0:
                continue
            matrix = model.coupling.get((src_split_idx, current_split_idx))
            if matrix is None:
                continue
            if src_branch_idx >= len(matrix):
                continue
            row = matrix[src_branch_idx]
            if current_branch_idx >= len(row):
                continue
            term += (self._prefix_decay**gap) * row[current_branch_idx]
        return term

    def _sample_mode(self, model: DagPathModel, now_sec: int) -> int:
        if self._context_regime == "drifting":
            self._update_drifting_state(model, now_sec)
            probs = model.pi_cur
        else:
            probs = model.pi_fixed
        return _sample_categorical(probs, self._rng)

    def _update_drifting_state(self, model: DagPathModel, now_sec: int) -> None:
        while now_sec >= model.next_refresh_sec:
            alphas = [
                max(self._eps, self._drifting_concentration * p + self._drifting_floor)
                for p in model.pi_cur
            ]
            model.pi_target = _sample_dirichlet(alphas, self._rng)
            model.next_refresh_sec += self._drifting_interval_sec
        dt = max(1, now_sec - model.last_update_sec)
        blend = 1.0 - ((1.0 - self._drifting_strength) ** dt)
        if blend > 0:
            model.pi_cur = _normalize_probs(
                [(1.0 - blend) * cur + blend * tgt for cur, tgt in zip(model.pi_cur, model.pi_target)]
            )
        model.last_update_sec = now_sec

    def _build_path_models(self) -> dict[str, DagPathModel]:
        models: dict[str, DagPathModel] = {}
        for dag_idx, um in enumerate(sorted(self._templates.keys())):
            template = self._templates[um]
            dag_seed = self._base_seed + self._coupling_seed_offset + dag_idx
            rng = random.Random(dag_seed)

            split_nodes = self._ordered_split_nodes(template)
            split_index = {node: idx for idx, node in enumerate(split_nodes)}
            branches_by_split: list[list[str]] = []
            base_probs_by_split: list[list[float]] = []

            for node in split_nodes:
                outgoing = template.transitions.get(node, {})
                branches = sorted(outgoing.keys())
                base_probs = _normalize_probs([max(self._eps, float(outgoing[b])) for b in branches])
                branches_by_split.append(branches)
                base_probs_by_split.append(base_probs)

            theta: list[list[list[float]]] = []
            for _ in range(self._mode_count):
                mode_theta: list[list[float]] = []
                for branches in branches_by_split:
                    mode_theta.append([rng.gauss(0.0, 1.0) for _ in branches])
                theta.append(mode_theta)

            coupling: dict[tuple[int, int], list[list[float]]] = {}
            for i, src_branches in enumerate(branches_by_split):
                for j in range(i + 1, len(branches_by_split)):
                    dst_branches = branches_by_split[j]
                    mat: list[list[float]] = []
                    for _ in src_branches:
                        mat.append([rng.gauss(0.0, 1.0) for _ in dst_branches])
                    coupling[(i, j)] = mat

            pi_fixed = _sample_dirichlet(
                [self._mode_prior_concentration for _ in range(self._mode_count)],
                rng,
            )
            pi_cur = list(pi_fixed)
            models[um] = DagPathModel(
                um=um,
                seed=dag_seed,
                split_nodes=split_nodes,
                split_index=split_index,
                branches_by_split=branches_by_split,
                base_probs_by_split=base_probs_by_split,
                theta=theta,
                coupling=coupling,
                pi_fixed=list(pi_fixed),
                pi_cur=pi_cur,
                pi_target=list(pi_fixed),
                pi_initial=list(pi_fixed),
                last_update_sec=0,
                next_refresh_sec=self._drifting_interval_sec,
            )
        return models

    def _ordered_split_nodes(self, template: DagTemplate) -> list[str]:
        transitions = template.transitions
        topo_nodes = _topological_order(transitions)
        split_nodes: list[str] = []
        start_out = transitions.get(START_NODE, {})
        if len(start_out) > 1:
            split_nodes.append(START_NODE)
        for node in topo_nodes:
            if len(transitions.get(node, {})) > 1:
                split_nodes.append(node)
        return split_nodes

    def _sample_outgoing(self, outgoing: dict[str, float]) -> str:
        branches = sorted(outgoing.keys())
        probs = _normalize_probs([max(self._eps, float(outgoing[b])) for b in branches])
        idx = _sample_categorical(probs, self._rng)
        return branches[idx]

    def _evict_stale_sessions(self, um: str, now_sec: int) -> None:
        stale: list[str] = []
        for sid in self._sessions_by_um[um]:
            state = self._sessions.get(sid)
            if state is None:
                stale.append(sid)
                continue
            if now_sec - state.last_seen_sec > self._session_gap_sec:
                stale.append(sid)
        for sid in stale:
            self._sessions_by_um[um].discard(sid)
            self._sessions.pop(sid, None)

    def _select_next_node_critical(
        self,
        src: str,
        outgoing: dict[str, float],
        session: SessionContext,
    ) -> str:
        weighted: list[tuple[str, float]] = []
        total = 0.0
        for dst, base_prob in outgoing.items():
            prior_count = session.edge_counts.get((src, dst), 0)
            score = base_prob * (1.0 + self._context_alpha * prior_count)
            weighted.append((dst, score))
            total += score
        if total <= 0:
            return self._rng.choice(list(outgoing.keys()))

        marker = self._rng.random() * total
        cumulative = 0.0
        for dst, score in weighted:
            cumulative += score
            if marker <= cumulative:
                return dst
        return weighted[-1][0]


def _topological_order(transitions: dict[str, dict[str, float]]) -> list[str]:
    nodes: set[str] = set()
    indegree: dict[str, int] = {}
    adj: dict[str, set[str]] = defaultdict(set)

    for src, nexts in transitions.items():
        if src != START_NODE:
            nodes.add(src)
        for dst in nexts:
            if dst == START_NODE:
                continue
            nodes.add(dst)
            if src != START_NODE:
                adj[src].add(dst)

    for node in nodes:
        indegree[node] = 0
    for src, dsts in adj.items():
        for dst in dsts:
            indegree[dst] = indegree.get(dst, 0) + 1
            indegree.setdefault(src, 0)

    q = deque(sorted([node for node in nodes if indegree.get(node, 0) == 0]))
    out: list[str] = []
    while q:
        node = q.popleft()
        out.append(node)
        for nxt in sorted(adj.get(node, set())):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if len(out) < len(nodes):
        seen = set(out)
        out.extend(sorted(node for node in nodes if node not in seen))
    return out


def _normalize_probs(weights: list[float]) -> list[float]:
    cleaned = [max(0.0, float(w)) for w in weights]
    total = sum(cleaned)
    if total <= 0:
        if not cleaned:
            return []
        uni = 1.0 / len(cleaned)
        return [uni for _ in cleaned]
    return [w / total for w in cleaned]


def _sample_dirichlet(alphas: list[float], rng: random.Random) -> list[float]:
    if not alphas:
        return []
    samples = [rng.gammavariate(max(1e-9, a), 1.0) for a in alphas]
    return _normalize_probs(samples)


def _sample_categorical(probs: list[float], rng: random.Random) -> int:
    if not probs:
        return 0
    marker = rng.random() * sum(probs)
    cumulative = 0.0
    for idx, p in enumerate(probs):
        cumulative += p
        if marker <= cumulative:
            return idx
    return len(probs) - 1


def _softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    t = max(1e-6, temperature)
    max_score = max(scores)
    exps = [math.exp((s - max_score) / t) for s in scores]
    return _normalize_probs(exps)


def _round6(value: float) -> float:
    return round(float(value), 6)
