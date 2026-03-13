from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from collections import deque
from pathlib import Path

from simulator.config import DatasetConfig, WorkloadConfig
from simulator.types import DagCorpus, DagTemplate

START_NODE = "__start__"
TASK_ID_RE = re.compile(r"(\d+)")


def _pick_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    return None


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


class AlibabaDatasetAdapter:
    """Build DAG templates from filtered task DAGs (cluster trace style)."""

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        workload_cfg: WorkloadConfig,
    ) -> None:
        self._dataset_cfg = dataset_cfg
        self._workload_cfg = workload_cfg
        self._processed_dir = Path(dataset_cfg.processed_dir)

    def load_corpus(self) -> DagCorpus:
        dag_tasks_path = Path(self._dataset_cfg.dag_tasks_file)
        if dag_tasks_path.exists():
            corpus = self._build_from_filtered_tasks(dag_tasks_path)
            self._persist_processed_snapshot(corpus)
            return corpus

        if self._workload_cfg.mode == "replay":
            raise FileNotFoundError(
                f"workload.mode=replay requires DAG tasks file: {dag_tasks_path}"
            )

        corpus = self._fallback_corpus(warning="dag_tasks_file_not_found")
        self._persist_processed_snapshot(corpus)
        return corpus

    def _build_from_filtered_tasks(self, path: Path) -> DagCorpus:
        total_rows = 0
        invalid_rows = 0
        job_nodes: dict[str, set[str]] = defaultdict(set)
        job_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("filtered tasks file has no header")
            task_col = _pick_column(reader.fieldnames, ["task_name", "task"])
            job_col = _pick_column(reader.fieldnames, ["job_name", "job"])
            if not task_col or not job_col:
                raise ValueError("filtered tasks columns are not recognized")

            for row in reader:
                total_rows += 1
                job_name = (row.get(job_col) or "").strip()
                parsed = self._parse_task_name(row.get(task_col))
                if not job_name or parsed is None:
                    invalid_rows += 1
                    continue

                child, deps = parsed
                job_nodes[job_name].add(child)
                for dep in deps:
                    job_nodes[job_name].add(dep)
                    if dep != child:
                        job_edges[job_name].add((dep, child))

        structure_acc: dict[str, dict[str, object]] = {}
        valid_jobs = 0
        for job_name, nodes in job_nodes.items():
            if not nodes:
                continue
            valid_jobs += 1
            edges = job_edges.get(job_name, set())
            signature_data = self._canonicalize_structure(nodes, edges)
            signature = signature_data["signature"]
            if signature not in structure_acc:
                structure_acc[signature] = signature_data
                structure_acc[signature]["support_count"] = 0
            structure_acc[signature]["support_count"] = int(structure_acc[signature]["support_count"]) + 1

        ranked_structures = sorted(
            structure_acc.values(),
            key=lambda x: (
                -int(x["node_count"]),
                -int(x["edge_count"]),
                -int(x["support_count"]),
                str(x["signature"]),
            ),
        )
        eligible_structures = [
            s for s in ranked_structures if self._is_structure_eligible(s)
        ]
        top_k = max(1, int(self._dataset_cfg.dag_top_k))
        selection_mode = str(self._dataset_cfg.dag_selection_mode).strip().lower()
        if selection_mode in {"random", "random_unique"}:
            rng = random.Random(int(self._dataset_cfg.dag_selection_seed))
            if top_k >= len(eligible_structures):
                selected = list(eligible_structures)
            else:
                sample_indexes = sorted(rng.sample(range(len(eligible_structures)), top_k))
                selected = [eligible_structures[i] for i in sample_indexes]
        elif selection_mode in {"topk_unique", "ranked_topk_unique"}:
            selected = eligible_structures[:top_k]
        else:
            raise ValueError(
                f"unsupported dataset.dag_selection_mode={self._dataset_cfg.dag_selection_mode}"
            )

        if not selected:
            if self._workload_cfg.mode == "replay":
                raise ValueError(
                    "no eligible DAG structures after parsing filtered_tasks.csv; "
                    "please relax dataset complexity constraints"
                )
            return self._fallback_corpus(warning="no_valid_dag_structure")

        templates: dict[str, DagTemplate] = {}
        um_weights_raw: dict[str, float] = {}
        selected_meta: list[dict[str, object]] = []

        for idx, structure in enumerate(selected, start=1):
            um = f"dag_{idx:04d}"
            node_ids = list(structure["nodes"])
            edges = list(structure["edges"])
            roots = list(structure["roots"])
            transitions = self._build_template_transitions(um, node_ids, edges, roots)
            node_latency_ms = {self._node_name(um, node_id): 50 for node_id in node_ids}
            templates[um] = DagTemplate(
                um=um,
                transitions=transitions,
                node_latency_ms=node_latency_ms,
            )
            support_count = int(structure["support_count"])
            um_weights_raw[um] = float(support_count)
            selected_meta.append(
                {
                    "um": um,
                    "node_count": int(structure["node_count"]),
                    "edge_count": int(structure["edge_count"]),
                    "longest_path_len": int(structure["longest_path_len"]),
                    "split_count": int(structure["split_count"]),
                    "support_count": support_count,
                    "signature": str(structure["signature"]),
                }
            )

        um_weights = self._normalize_weights(um_weights_raw)
        dag_selection_policy = "unique_structure_random" if selection_mode in {"random", "random_unique"} else "unique_structure_topk"
        metadata = {
            "dataset_source": self._dataset_cfg.source,
            "mode": f"filtered_tasks_{selection_mode}",
            "dag_tasks_file": str(path),
            "dag_top_k": top_k,
            "total_rows": total_rows,
            "valid_rows": total_rows - invalid_rows,
            "invalid_rows": invalid_rows,
            "total_jobs": len(job_nodes),
            "valid_jobs": valid_jobs,
            "unique_structures": len(ranked_structures),
            "eligible_structures": len(eligible_structures),
            "selected_structures": len(selected_meta),
            "dag_selection": {
                "policy": dag_selection_policy,
                "sort": "node_count_desc,edge_count_desc,support_count_desc,signature_asc",
                "top_k": top_k,
                "seed": int(self._dataset_cfg.dag_selection_seed),
                "constraints": {
                    "dag_max_nodes": self._dataset_cfg.dag_max_nodes,
                    "dag_max_edges": self._dataset_cfg.dag_max_edges,
                    "dag_max_longest_path": self._dataset_cfg.dag_max_longest_path,
                    "dag_max_splits": self._dataset_cfg.dag_max_splits,
                },
                "selected": selected_meta,
            },
        }
        replay = [self._workload_cfg.baseline_rps * 60.0] * max(1, self._dataset_cfg.time_window_minutes)
        return DagCorpus(
            templates=templates,
            um_weights=um_weights,
            replay_total_qps_per_minute=replay,
            metadata=metadata,
        )

    def _parse_task_name(self, raw: str | None) -> tuple[str, list[str]] | None:
        if raw is None:
            return None
        task_name = raw.strip()
        if not task_name:
            return None

        parts = task_name.split("_")
        head = parts[0].strip()
        match = TASK_ID_RE.search(head)
        if not match:
            return None

        child = str(int(match.group(1)))
        deps: list[str] = []
        for token in parts[1:]:
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                deps.append(str(int(token)))
        return child, deps

    def _canonicalize_structure(
        self,
        nodes: set[str],
        edges: set[tuple[str, str]],
    ) -> dict[str, object]:
        sorted_nodes = sorted(nodes, key=lambda n: _safe_int(n))
        idx = {node: i for i, node in enumerate(sorted_nodes)}
        canonical_nodes = tuple(range(len(sorted_nodes)))
        canonical_edges = sorted(
            {
                (idx[src], idx[dst])
                for src, dst in edges
                if src in idx and dst in idx and src != dst
            }
        )

        indegree = {node: 0 for node in canonical_nodes}
        for _, dst in canonical_edges:
            indegree[dst] = indegree.get(dst, 0) + 1
        roots = sorted(node for node in canonical_nodes if indegree.get(node, 0) == 0)
        if not roots and canonical_nodes:
            roots = [canonical_nodes[0]]
        longest_path_len, split_count = self._compute_structure_metrics(
            nodes=canonical_nodes,
            edges=canonical_edges,
            roots=roots,
        )

        signature = (
            f"n={len(canonical_nodes)}|"
            f"e={';'.join(f'{src}>{dst}' for src, dst in canonical_edges)}"
        )
        return {
            "signature": signature,
            "nodes": canonical_nodes,
            "edges": tuple(canonical_edges),
            "roots": tuple(roots),
            "node_count": len(canonical_nodes),
            "edge_count": len(canonical_edges),
            "longest_path_len": longest_path_len,
            "split_count": split_count,
        }

    def _compute_structure_metrics(
        self,
        *,
        nodes: tuple[int, ...],
        edges: list[tuple[int, int]],
        roots: list[int],
    ) -> tuple[int, int]:
        adj: dict[int, list[int]] = defaultdict(list)
        indegree: dict[int, int] = {node: 0 for node in nodes}
        outdegree: dict[int, int] = {node: 0 for node in nodes}
        for src, dst in edges:
            adj[src].append(dst)
            indegree[dst] = indegree.get(dst, 0) + 1
            outdegree[src] = outdegree.get(src, 0) + 1

        q = deque(sorted(roots))
        indegree_work = dict(indegree)
        distance: dict[int, int] = {root: 1 for root in roots}
        while q:
            src = q.popleft()
            for dst in adj.get(src, []):
                distance[dst] = max(distance.get(dst, 1), distance.get(src, 1) + 1)
                indegree_work[dst] = indegree_work.get(dst, 0) - 1
                if indegree_work[dst] == 0:
                    q.append(dst)

        longest_path_len = max(distance.values()) if distance else (1 if nodes else 0)
        split_count = sum(1 for node in nodes if outdegree.get(node, 0) > 1)
        return int(longest_path_len), int(split_count)

    def _is_structure_eligible(self, structure: dict[str, object]) -> bool:
        max_nodes = self._dataset_cfg.dag_max_nodes
        max_edges = self._dataset_cfg.dag_max_edges
        max_longest_path = self._dataset_cfg.dag_max_longest_path
        max_splits = self._dataset_cfg.dag_max_splits

        if max_nodes is not None and int(max_nodes) > 0:
            if int(structure["node_count"]) > int(max_nodes):
                return False
        if max_edges is not None and int(max_edges) > 0:
            if int(structure["edge_count"]) > int(max_edges):
                return False
        if max_longest_path is not None and int(max_longest_path) > 0:
            if int(structure.get("longest_path_len", 0)) > int(max_longest_path):
                return False
        if max_splits is not None and int(max_splits) > 0:
            if int(structure.get("split_count", 0)) > int(max_splits):
                return False
        return True

    def _build_template_transitions(
        self,
        um: str,
        node_ids: list[int],
        edges: list[tuple[int, int]],
        roots: list[int],
    ) -> dict[str, dict[str, float]]:
        transitions: dict[str, dict[str, float]] = {}
        if not node_ids:
            return transitions

        if roots:
            root_targets = [self._node_name(um, node_id) for node_id in roots]
        else:
            root_targets = [self._node_name(um, node_ids[0])]
        root_prob = 1.0 / len(root_targets)
        transitions[START_NODE] = {target: root_prob for target in root_targets}

        children: dict[int, list[int]] = defaultdict(list)
        for src, dst in edges:
            children[src].append(dst)
        for src, dsts in children.items():
            next_nodes = sorted(set(dsts))
            prob = 1.0 / len(next_nodes)
            transitions[self._node_name(um, src)] = {
                self._node_name(um, dst): prob for dst in next_nodes
            }
        return transitions

    def _normalize_weights(self, raw: dict[str, float]) -> dict[str, float]:
        cleaned = {k: v for k, v in raw.items() if v > 0}
        total = sum(cleaned.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in cleaned.items()}

    def _node_name(self, um: str, node_id: int) -> str:
        return f"{um}_fn_{node_id + 1:03d}"

    def _fallback_corpus(self, warning: str | None = None) -> DagCorpus:
        templates = {
            "search": DagTemplate(
                um="search",
                transitions={
                    START_NODE: {"gateway": 1.0},
                    "gateway": {"auth": 0.55, "cache": 0.45},
                    "auth": {"ranker": 1.0},
                    "cache": {"ranker": 1.0},
                    "ranker": {"inventory": 0.7, "ads": 0.3},
                    "inventory": {"response": 1.0},
                    "ads": {"response": 1.0},
                },
                node_latency_ms={
                    "gateway": 20,
                    "auth": 35,
                    "cache": 15,
                    "ranker": 60,
                    "inventory": 40,
                    "ads": 50,
                    "response": 10,
                },
            ),
            "order": DagTemplate(
                um="order",
                transitions={
                    START_NODE: {"gateway": 1.0},
                    "gateway": {"auth": 1.0},
                    "auth": {"pricing": 0.4, "inventory": 0.6},
                    "pricing": {"payment": 1.0},
                    "inventory": {"payment": 1.0},
                    "payment": {"response": 1.0},
                },
                node_latency_ms={
                    "gateway": 25,
                    "auth": 40,
                    "pricing": 45,
                    "inventory": 55,
                    "payment": 70,
                    "response": 12,
                },
            ),
            "feed": DagTemplate(
                um="feed",
                transitions={
                    START_NODE: {"gateway": 1.0},
                    "gateway": {"profile": 0.5, "trend": 0.5},
                    "profile": {"composer": 1.0},
                    "trend": {"composer": 1.0},
                    "composer": {"response": 1.0},
                },
                node_latency_ms={
                    "gateway": 18,
                    "profile": 30,
                    "trend": 40,
                    "composer": 45,
                    "response": 8,
                },
            ),
        }
        replay = [self._workload_cfg.baseline_rps * 60.0] * max(1, self._dataset_cfg.time_window_minutes)
        metadata = {
            "dataset_source": self._dataset_cfg.source,
            "mode": "fallback",
            "warning": warning or "raw_dataset_not_found",
        }
        return DagCorpus(
            templates=templates,
            um_weights={"search": 0.5, "order": 0.3, "feed": 0.2},
            replay_total_qps_per_minute=replay,
            metadata=metadata,
        )

    def _persist_processed_snapshot(self, corpus: DagCorpus) -> None:
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._processed_dir / "dag_corpus_snapshot.json"
        payload = {
            "metadata": corpus.metadata,
            "ums": sorted(corpus.templates.keys()),
            "um_weights": corpus.um_weights,
            "replay_minutes": len(corpus.replay_total_qps_per_minute),
        }
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
