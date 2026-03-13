from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DagTemplate:
    um: str
    transitions: dict[str, dict[str, float]]
    node_latency_ms: dict[str, int]


@dataclass
class DagCorpus:
    templates: dict[str, DagTemplate]
    um_weights: dict[str, float]
    replay_total_qps_per_minute: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    function_id: str
    cold_start_ms: int
    memory_mb: int
    output_data_mb: float
    compute_data_mb: float
    cpu_request_mcpu: int


@dataclass
class RequestContext:
    request_id: str
    um: str
    session_id: str
    arrival_ms: int
    path: list[str]
    current_index: int = 0
    completed: bool = False
    completed_ms: int | None = None
    timed_out: bool = False
    failed_reason: str | None = None
    prev_host_id: str | None = None
    prev_function_node: str | None = None
    cold_start_latency_ms: int = 0
    data_transfer_latency_ms: int = 0
    execution_latency_ms: int = 0
    queue_wait_latency_ms: int = 0
