from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings

import yaml


@dataclass
class ExperimentConfig:
    name: str
    duration_seconds: int
    random_seed: int


@dataclass
class DatasetConfig:
    source: str
    processed_dir: str
    time_window_minutes: int
    callgraph_file: str = "data/raw/unused_callgraph.csv"
    qps_file: str = "data/raw/unused_qps.csv"
    sample_um_count: int = 8
    checksum_required: bool = False
    dag_tasks_file: str = "data/raw/filtered_tasks.csv"
    dag_top_k: int = 20
    dag_selection_mode: str = "random_unique"
    dag_selection_seed: int = 42
    dag_max_nodes: int | None = None
    dag_max_edges: int | None = None
    dag_max_longest_path: int | None = None
    dag_max_splits: int | None = None


@dataclass
class DagPolicyConfig:
    granularity: str = "um"
    path_rule: str = "mode_prefix_coupled_v1"
    markov_order: int = 1
    session_gap_sec: int = 30
    session_continue_prob: float = 0.7
    context_alpha: float = 0.35
    context_regime: str = "fixed"
    mode_count: int = 3
    mode_prior_concentration: float = 2.0
    mode_strength: float = 1.0
    prefix_strength: float = 1.2
    prefix_decay: float = 0.85
    prefix_window: int = 3
    temperature: float = 1.0
    coupling_seed_offset: int = 707
    drifting_interval_sec: int = 30
    drifting_strength: float = 0.08
    drifting_concentration: float = 200.0
    drifting_floor: float = 1e-3
    eps: float = 1e-9


@dataclass
class WorkloadConfig:
    mode: str
    baseline_rps: float = 6.0
    burst_rps: float = 35.0
    burst_start_sec: int = 90
    burst_duration_sec: int = 90
    rate_multiplier: float = 0.0001
    invokes_cdf_file: str = "data/real-world-emulation/CDFs/invokesCDF.csv"
    cvs_cdf_file: str = "data/real-world-emulation/CDFs/CVs.csv"
    realworld_seed_offset: int = 303
    min_iat_ms: float = 1.0


@dataclass
class RuntimeConfig:
    node_base_latency_ms: int = 50
    node_latency_jitter_ms: int = 20
    cold_start_ms_min: int = 1200
    cold_start_ms_max: int = 3200
    request_timeout_ms: int = 5000
    max_concurrency_per_instance: int = 1
    data_transfer_ms_min: int = 5
    data_transfer_ms_max: int = 15
    frame_tick_ms: int = 1
    function_profile_mode: str = "cpu_intensive_random"
    function_profiles_file: str = "data/raw/function_profiles/function_profiles.csv"
    function_profile_seed_offset: int = 202
    compute_mb_per_sec_per_1000mcpu: float = 180.0
    function_cpu_request_mcpu_min: int = 600
    function_cpu_request_mcpu_max: int = 2000
    function_memory_mb_min: int = 256
    function_memory_mb_max: int = 2048
    function_output_data_mb_min: float = 10.0
    function_output_data_mb_max: float = 30.0
    function_compute_data_mb_min: float = 40.0
    function_compute_data_mb_max: float = 220.0
    function_cold_start_ms_min: int = 1200
    function_cold_start_ms_max: int = 3200
    cold_start_ratio_floor: float = 1.2


@dataclass
class PhysicalNodesConfig:
    count: int = 20
    max_containers_per_node: int = 10
    idle_ttl_sec: int = 60
    same_node_transfer_ms_min: int = 1
    same_node_transfer_ms_max: int = 3
    cross_node_transfer_ms_min: int = 8
    cross_node_transfer_ms_max: int = 20
    cpu_total_mcpu_per_node: int = 32000
    mem_total_mb_per_node: int = 131072
    bandwidth_mode: str = "random_uniform"
    bandwidth_mbps_min: float = 1000.0
    bandwidth_mbps_max: float = 25000.0
    bandwidth_seed_offset: int = 101
    bandwidth_matrix_file: str = "data/raw/network/node_bandwidth_matrix.csv"


@dataclass
class SchedulerConfig:
    type: str


@dataclass
class AutoscalerConfig:
    type: str
    target_utilization: float
    sync_period_sec: int
    scale_down_stabilization_sec: int
    min_replicas: int
    max_replicas_per_node: int


@dataclass
class CapacityConfig:
    max_total_instances: int


@dataclass
class OutputConfig:
    runs_dir: str


@dataclass
class SimulationConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    dag_policy: DagPolicyConfig
    workload: WorkloadConfig
    runtime: RuntimeConfig
    physical_nodes: PhysicalNodesConfig
    scheduler: SchedulerConfig
    autoscaler: AutoscalerConfig
    capacity: CapacityConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_config(raw: dict[str, Any]) -> SimulationConfig:
    autoscaler_raw = raw.get(
        "autoscaler",
        {
            "type": "hpa_v1",
            "target_utilization": 0.7,
            "sync_period_sec": 15,
            "scale_down_stabilization_sec": 60,
            "min_replicas": 0,
            "max_replicas_per_node": 20,
        },
    )
    if "autoscaler" in raw:
        warnings.warn(
            "autoscaler config is retained for compatibility but ignored in no-queue physical-node mode",
            RuntimeWarning,
            stacklevel=2,
        )
    physical_nodes_raw = raw.get(
        "physical_nodes",
        {
            "count": 20,
            "max_containers_per_node": 10,
            "idle_ttl_sec": 60,
            "same_node_transfer_ms_min": 1,
            "same_node_transfer_ms_max": 3,
            "cross_node_transfer_ms_min": 8,
            "cross_node_transfer_ms_max": 20,
        },
    )
    runtime_raw = dict(raw["runtime"])
    if runtime_raw.get("max_concurrency_per_instance", 1) != 1:
        warnings.warn(
            "max_concurrency_per_instance is ignored in no-queue physical-node mode; effective behavior is single-task containers",
            RuntimeWarning,
            stacklevel=2,
        )
    if "function_cold_start_ms_min" not in runtime_raw:
        runtime_raw["function_cold_start_ms_min"] = runtime_raw.get("cold_start_ms_min", 1200)
    if "function_cold_start_ms_max" not in runtime_raw:
        runtime_raw["function_cold_start_ms_max"] = runtime_raw.get("cold_start_ms_max", 3200)
    if "frame_tick_ms" in runtime_raw and int(runtime_raw["frame_tick_ms"]) <= 0:
        warnings.warn(
            "frame_tick_ms must be > 0; forcing to 1",
            RuntimeWarning,
            stacklevel=2,
        )
        runtime_raw["frame_tick_ms"] = 1
    dag_policy_input = dict(raw["dag_policy"])
    dag_policy_raw = dict(dag_policy_input)
    if "context_regime" not in dag_policy_raw:
        dag_policy_raw["context_regime"] = "fixed"
    if (
        str(dag_policy_raw.get("path_rule", "")).lower() == "mode_prefix_coupled_v1"
        and (
            ("session_continue_prob" in dag_policy_input)
            or ("context_alpha" in dag_policy_input)
        )
    ):
        warnings.warn(
            "dag_policy.session_continue_prob/context_alpha are ignored when path_rule=mode_prefix_coupled_v1",
            RuntimeWarning,
            stacklevel=2,
        )
    regime = str(dag_policy_raw.get("context_regime", "fixed")).lower()
    if regime not in {"fixed", "drifting"}:
        warnings.warn(
            f"unsupported dag_policy.context_regime={dag_policy_raw.get('context_regime')}, forcing to 'fixed'",
            RuntimeWarning,
            stacklevel=2,
        )
        dag_policy_raw["context_regime"] = "fixed"
    dataset_input = dict(raw["dataset"])
    dataset_raw = dict(dataset_input)
    dataset_raw.setdefault("callgraph_file", "data/raw/unused_callgraph.csv")
    dataset_raw.setdefault("qps_file", "data/raw/unused_qps.csv")
    dataset_raw.setdefault("sample_um_count", 8)
    dataset_raw.setdefault("checksum_required", False)
    dataset_raw.setdefault("dag_tasks_file", "data/raw/filtered_tasks.csv")
    dataset_raw.setdefault("dag_top_k", 20)
    dataset_raw.setdefault("dag_selection_mode", "random_unique")
    dataset_raw.setdefault("dag_selection_seed", 42)
    dataset_raw.setdefault("dag_max_nodes", None)
    dataset_raw.setdefault("dag_max_edges", None)
    dataset_raw.setdefault("dag_max_longest_path", None)
    dataset_raw.setdefault("dag_max_splits", None)
    workload_raw = dict(raw["workload"])
    workload_raw.setdefault("invokes_cdf_file", "data/real-world-emulation/CDFs/invokesCDF.csv")
    workload_raw.setdefault("cvs_cdf_file", "data/real-world-emulation/CDFs/CVs.csv")
    workload_raw.setdefault("realworld_seed_offset", 303)
    workload_raw.setdefault("min_iat_ms", 1.0)
    if (
        str(workload_raw.get("mode", "")).lower() == "replay"
        and (("callgraph_file" in dataset_input) or ("qps_file" in dataset_input))
    ):
        warnings.warn(
            "workload.mode=replay now uses filtered_tasks + Real-World CDFs; callgraph_file/qps_file are ignored for replay generation",
            RuntimeWarning,
            stacklevel=2,
        )
    return SimulationConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        dataset=DatasetConfig(**dataset_raw),
        dag_policy=DagPolicyConfig(**dag_policy_raw),
        workload=WorkloadConfig(**workload_raw),
        runtime=RuntimeConfig(**runtime_raw),
        physical_nodes=PhysicalNodesConfig(**physical_nodes_raw),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        autoscaler=AutoscalerConfig(**autoscaler_raw),
        capacity=CapacityConfig(**raw["capacity"]),
        output=OutputConfig(**raw["output"]),
    )


def load_config(path: str | Path) -> SimulationConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _build_config(raw)


def dump_config(config: SimulationConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
