from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

from simulator.config import PhysicalNodesConfig, RuntimeConfig
from simulator.types import FunctionProfile


def build_function_profiles(
    *,
    function_ids: set[str],
    runtime_cfg: RuntimeConfig,
    physical_cfg: PhysicalNodesConfig,
    base_seed: int,
) -> tuple[dict[str, FunctionProfile], str]:
    profiles: dict[str, FunctionProfile] = {}
    source = "random"
    profile_path = Path(runtime_cfg.function_profiles_file)
    if profile_path.exists():
        profiles.update(_load_function_profiles_from_file(profile_path, runtime_cfg))
        source = f"file:{profile_path}"

    for function_id in sorted(function_ids):
        if function_id in profiles:
            continue
        profiles[function_id] = _generate_cpu_intensive_profile(
            function_id=function_id,
            runtime_cfg=runtime_cfg,
            physical_cfg=physical_cfg,
            base_seed=base_seed,
        )

    return profiles, source


def build_bandwidth_matrix(
    *,
    node_ids: list[str],
    physical_cfg: PhysicalNodesConfig,
    base_seed: int,
) -> tuple[dict[str, dict[str, float]], str]:
    mode = physical_cfg.bandwidth_mode.strip().lower()
    if mode == "from_file":
        matrix = _load_bandwidth_from_file(
            path=Path(physical_cfg.bandwidth_matrix_file),
            node_ids=node_ids,
        )
        return matrix, f"file:{physical_cfg.bandwidth_matrix_file}"
    if mode != "random_uniform":
        raise ValueError(f"unsupported bandwidth_mode: {physical_cfg.bandwidth_mode}")

    low = min(physical_cfg.bandwidth_mbps_min, physical_cfg.bandwidth_mbps_max)
    high = max(physical_cfg.bandwidth_mbps_min, physical_cfg.bandwidth_mbps_max)
    if high <= 0:
        raise ValueError("bandwidth_mbps_max must be > 0")
    if low <= 0:
        low = 1.0

    rng = random.Random(base_seed + int(physical_cfg.bandwidth_seed_offset))
    matrix: dict[str, dict[str, float]] = {node_id: {} for node_id in node_ids}
    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            matrix[src][dst] = rng.uniform(low, high)
    return matrix, f"random_uniform(seed={base_seed + int(physical_cfg.bandwidth_seed_offset)})"


def _load_function_profiles_from_file(
    path: Path,
    runtime_cfg: RuntimeConfig,
) -> dict[str, FunctionProfile]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "function_id",
            "cold_start_ms",
            "memory_mb",
            "output_data_mb",
            "compute_data_mb",
        }
        headers = set(reader.fieldnames or [])
        missing = sorted(required - headers)
        if missing:
            raise ValueError(f"function profile file missing required columns: {missing}")

        profiles: dict[str, FunctionProfile] = {}
        for row in reader:
            function_id = (row.get("function_id") or "").strip()
            if not function_id:
                raise ValueError("function profile contains empty function_id")
            if function_id in profiles:
                raise ValueError(f"duplicate function_id in function profile file: {function_id}")

            cold_start_ms = int(float(row["cold_start_ms"]))
            memory_mb = int(float(row["memory_mb"]))
            output_data_mb = float(row["output_data_mb"])
            compute_data_mb = float(row["compute_data_mb"])
            if cold_start_ms <= 0 or memory_mb <= 0 or output_data_mb <= 0 or compute_data_mb <= 0:
                raise ValueError(f"function profile has non-positive values for {function_id}")

            cpu_raw = row.get("cpu_request_mcpu")
            if cpu_raw is None or str(cpu_raw).strip() == "":
                cpu_request_mcpu = _derive_cpu_request_mcpu(
                    compute_data_mb=compute_data_mb,
                    runtime_cfg=runtime_cfg,
                )
            else:
                cpu_request_mcpu = int(float(cpu_raw))
            if cpu_request_mcpu <= 0:
                raise ValueError(f"cpu_request_mcpu must be > 0 for {function_id}")

            profiles[function_id] = FunctionProfile(
                function_id=function_id,
                cold_start_ms=cold_start_ms,
                memory_mb=memory_mb,
                output_data_mb=output_data_mb,
                compute_data_mb=compute_data_mb,
                cpu_request_mcpu=cpu_request_mcpu,
            )
    return profiles


def _load_bandwidth_from_file(
    *,
    path: Path,
    node_ids: list[str],
) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"bandwidth matrix file not found: {path}")

    node_set = set(node_ids)
    matrix: dict[str, dict[str, float]] = {node_id: {} for node_id in node_ids}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src_node_id", "dst_node_id", "bandwidth_mbps"}
        headers = set(reader.fieldnames or [])
        missing = sorted(required - headers)
        if missing:
            raise ValueError(f"bandwidth matrix file missing required columns: {missing}")

        for row in reader:
            src = (row.get("src_node_id") or "").strip()
            dst = (row.get("dst_node_id") or "").strip()
            if not src or not dst:
                raise ValueError("bandwidth matrix contains empty src_node_id/dst_node_id")
            if src == dst:
                continue
            if src not in node_set or dst not in node_set:
                raise ValueError(
                    f"bandwidth matrix contains unknown node id pair: ({src}, {dst}); expected {sorted(node_ids)}"
                )

            bw = float(row["bandwidth_mbps"])
            if bw <= 0:
                raise ValueError(f"bandwidth_mbps must be > 0 for ({src}, {dst})")
            if dst in matrix[src]:
                raise ValueError(f"duplicate bandwidth entry for ({src}, {dst})")
            matrix[src][dst] = bw

    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            if dst not in matrix[src]:
                raise ValueError(f"bandwidth matrix missing edge ({src}, {dst})")
    return matrix


def _generate_cpu_intensive_profile(
    *,
    function_id: str,
    runtime_cfg: RuntimeConfig,
    physical_cfg: PhysicalNodesConfig,
    base_seed: int,
) -> FunctionProfile:
    rng = random.Random(
        base_seed + int(runtime_cfg.function_profile_seed_offset) + _stable_int_hash(function_id)
    )

    cpu_min = min(runtime_cfg.function_cpu_request_mcpu_min, runtime_cfg.function_cpu_request_mcpu_max)
    cpu_max = max(runtime_cfg.function_cpu_request_mcpu_min, runtime_cfg.function_cpu_request_mcpu_max)
    mem_min = min(runtime_cfg.function_memory_mb_min, runtime_cfg.function_memory_mb_max)
    mem_max = max(runtime_cfg.function_memory_mb_min, runtime_cfg.function_memory_mb_max)
    out_min = min(runtime_cfg.function_output_data_mb_min, runtime_cfg.function_output_data_mb_max)
    out_max = max(runtime_cfg.function_output_data_mb_min, runtime_cfg.function_output_data_mb_max)
    comp_min = min(runtime_cfg.function_compute_data_mb_min, runtime_cfg.function_compute_data_mb_max)
    comp_max = max(runtime_cfg.function_compute_data_mb_min, runtime_cfg.function_compute_data_mb_max)
    cold_min = min(runtime_cfg.function_cold_start_ms_min, runtime_cfg.function_cold_start_ms_max)
    cold_max = max(runtime_cfg.function_cold_start_ms_min, runtime_cfg.function_cold_start_ms_max)

    cpu_request_mcpu = rng.randint(max(1, int(cpu_min)), max(1, int(cpu_max)))
    memory_mb = rng.randint(max(1, int(mem_min)), max(1, int(mem_max)))
    output_data_mb = rng.uniform(max(0.001, float(out_min)), max(0.001, float(out_max)))
    compute_data_mb = rng.uniform(max(0.001, float(comp_min)), max(0.001, float(comp_max)))

    cold_start_ms = rng.randint(max(1, int(cold_min)), max(1, int(cold_max)))
    throughput = max(
        0.001,
        runtime_cfg.compute_mb_per_sec_per_1000mcpu * (cpu_request_mcpu / 1000.0),
    )
    estimated_execution_ms = math.ceil((compute_data_mb / throughput) * 1000.0)
    min_bandwidth = max(1.0, min(physical_cfg.bandwidth_mbps_min, physical_cfg.bandwidth_mbps_max))
    estimated_transfer_ms = math.ceil((output_data_mb * 8.0 / min_bandwidth) * 1000.0)
    floor_ratio = max(0.0, runtime_cfg.cold_start_ratio_floor)
    floor_cold_ms = math.ceil(floor_ratio * (estimated_execution_ms + estimated_transfer_ms))
    if cold_start_ms < floor_cold_ms:
        cold_start_ms = min(int(cold_max), max(int(cold_min), floor_cold_ms))

    return FunctionProfile(
        function_id=function_id,
        cold_start_ms=max(1, int(cold_start_ms)),
        memory_mb=max(1, int(memory_mb)),
        output_data_mb=max(0.001, float(output_data_mb)),
        compute_data_mb=max(0.001, float(compute_data_mb)),
        cpu_request_mcpu=max(1, int(cpu_request_mcpu)),
    )


def _derive_cpu_request_mcpu(
    *,
    compute_data_mb: float,
    runtime_cfg: RuntimeConfig,
) -> int:
    low = min(runtime_cfg.function_cpu_request_mcpu_min, runtime_cfg.function_cpu_request_mcpu_max)
    high = max(runtime_cfg.function_cpu_request_mcpu_min, runtime_cfg.function_cpu_request_mcpu_max)
    comp_low = min(runtime_cfg.function_compute_data_mb_min, runtime_cfg.function_compute_data_mb_max)
    comp_high = max(runtime_cfg.function_compute_data_mb_min, runtime_cfg.function_compute_data_mb_max)
    if comp_high - comp_low <= 0.000001:
        return max(1, int(low))
    ratio = min(1.0, max(0.0, (compute_data_mb - comp_low) / (comp_high - comp_low)))
    return max(1, int(round(low + ratio * (high - low))))


def _stable_int_hash(raw: str) -> int:
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
