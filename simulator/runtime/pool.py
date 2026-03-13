from __future__ import annotations

import random
from dataclasses import dataclass


CONTAINER_COLD_STARTING = "COLD_STARTING"
CONTAINER_RUNNING = "RUNNING"
CONTAINER_IDLE = "IDLE"


@dataclass
class FunctionContainer:
    container_id: str
    host_id: str
    function_node: str
    cpu_request_mcpu: int
    memory_mb: int
    state: str = CONTAINER_IDLE
    executing: bool = False
    warm: bool = False
    last_idle_ms: int | None = None

    @property
    def busy(self) -> bool:
        return self.state != CONTAINER_IDLE


@dataclass
class PhysicalNode:
    node_id: str
    max_containers: int
    cpu_total_mcpu: int
    mem_total_mb: int

    def __post_init__(self) -> None:
        self.containers: dict[str, FunctionContainer] = {}
        self.cpu_reserved_mcpu: int = 0
        self.mem_reserved_mb: int = 0

    @property
    def active_containers(self) -> int:
        return len(self.containers)

    @property
    def busy_containers(self) -> int:
        return sum(1 for c in self.containers.values() if c.busy)

    @property
    def running_containers(self) -> int:
        return sum(1 for c in self.containers.values() if c.state == CONTAINER_RUNNING)

    @property
    def cold_starting_containers(self) -> int:
        return sum(1 for c in self.containers.values() if c.state == CONTAINER_COLD_STARTING)

    @property
    def idle_containers(self) -> int:
        return sum(1 for c in self.containers.values() if c.state == CONTAINER_IDLE)

    @property
    def utilization(self) -> float:
        if self.max_containers <= 0:
            return 1.0
        return self.busy_containers / self.max_containers

    @property
    def cpu_utilization(self) -> float:
        if self.cpu_total_mcpu <= 0:
            return 1.0
        return min(1.0, self.cpu_reserved_mcpu / self.cpu_total_mcpu)

    @property
    def mem_utilization(self) -> float:
        if self.mem_total_mb <= 0:
            return 1.0
        return min(1.0, self.mem_reserved_mb / self.mem_total_mb)

    @property
    def has_slot(self) -> bool:
        return self.active_containers < self.max_containers

    def has_resources_for(self, *, cpu_request_mcpu: int, memory_mb: int) -> bool:
        return (
            (self.cpu_reserved_mcpu + cpu_request_mcpu <= self.cpu_total_mcpu)
            and (self.mem_reserved_mb + memory_mb <= self.mem_total_mb)
        )

    def idle_containers_for(self, function_node: str) -> list[FunctionContainer]:
        return [
            c
            for c in self.containers.values()
            if (c.state == CONTAINER_IDLE) and c.function_node == function_node
        ]


class PhysicalCluster:
    """Physical-node runtime model with single-task containers and resource constraints."""

    def __init__(
        self,
        *,
        node_count: int,
        max_containers_per_node: int,
        cpu_total_mcpu_per_node: int,
        mem_total_mb_per_node: int,
        rng: random.Random,
    ) -> None:
        if node_count <= 0:
            raise ValueError("node_count must be > 0")
        if max_containers_per_node <= 0:
            raise ValueError("max_containers_per_node must be > 0")
        if cpu_total_mcpu_per_node <= 0:
            raise ValueError("cpu_total_mcpu_per_node must be > 0")
        if mem_total_mb_per_node <= 0:
            raise ValueError("mem_total_mb_per_node must be > 0")

        self._rng = rng
        self.nodes: dict[str, PhysicalNode] = {
            f"host-{i + 1}": PhysicalNode(
                node_id=f"host-{i + 1}",
                max_containers=max_containers_per_node,
                cpu_total_mcpu=cpu_total_mcpu_per_node,
                mem_total_mb=mem_total_mb_per_node,
            )
            for i in range(node_count)
        }
        self._containers: dict[str, FunctionContainer] = {}
        self._container_counter = 0

    @property
    def total_active_containers(self) -> int:
        return sum(node.active_containers for node in self.nodes.values())

    @property
    def total_busy_containers(self) -> int:
        return sum(node.busy_containers for node in self.nodes.values())

    def get_container(self, container_id: str) -> FunctionContainer | None:
        return self._containers.get(container_id)

    def acquire_warm_container(
        self,
        *,
        function_node: str,
        now_ms: int,
    ) -> FunctionContainer | None:
        candidates: list[tuple[float, int, FunctionContainer]] = []
        for node in self.nodes.values():
            for container in node.idle_containers_for(function_node):
                candidates.append((node.utilization, node.active_containers, container))
        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2].container_id))
        min_key = (candidates[0][0], candidates[0][1])
        tied = [c[2] for c in candidates if (c[0], c[1]) == min_key]
        chosen = self._rng.choice(tied)
        chosen.state = CONTAINER_RUNNING
        chosen.executing = False
        chosen.last_idle_ms = None
        return chosen

    def create_cold_container_on_host(
        self,
        *,
        host_id: str,
        function_node: str,
        cpu_request_mcpu: int,
        memory_mb: int,
        now_ms: int,
    ) -> FunctionContainer | None:
        node = self.nodes.get(host_id)
        if node is None or (not node.has_slot):
            return None
        if not node.has_resources_for(cpu_request_mcpu=cpu_request_mcpu, memory_mb=memory_mb):
            return None

        self._container_counter += 1
        container_id = f"ctr-{self._container_counter}"
        container = FunctionContainer(
            container_id=container_id,
            host_id=node.node_id,
            function_node=function_node,
            cpu_request_mcpu=cpu_request_mcpu,
            memory_mb=memory_mb,
            state=CONTAINER_COLD_STARTING,
            executing=False,
            warm=False,
            last_idle_ms=None,
        )
        node.containers[container_id] = container
        node.cpu_reserved_mcpu += cpu_request_mcpu
        node.mem_reserved_mb += memory_mb
        self._containers[container_id] = container
        return container

    def set_container_running(self, container_id: str) -> None:
        container = self._containers.get(container_id)
        if container is None:
            return
        container.state = CONTAINER_RUNNING
        container.executing = True

    def release_container(self, container_id: str, now_ms: int) -> None:
        container = self._containers.get(container_id)
        if container is None:
            return
        container.state = CONTAINER_IDLE
        container.executing = False
        container.warm = True
        container.last_idle_ms = now_ms

    def allocated_cpu_for_container(self, container_id: str) -> float:
        container = self._containers.get(container_id)
        if container is None:
            return 0.0
        node = self.nodes.get(container.host_id)
        if node is None:
            return 0.0

        running = [
            c
            for c in node.containers.values()
            if c.state == CONTAINER_RUNNING and c.executing
        ]
        if not running:
            return float(container.cpu_request_mcpu)

        total_requested = sum(max(1, c.cpu_request_mcpu) for c in running)
        if total_requested <= 0:
            return float(container.cpu_request_mcpu)
        fair_share = node.cpu_total_mcpu * (container.cpu_request_mcpu / total_requested)
        return max(1.0, min(float(container.cpu_request_mcpu), float(fair_share)))

    def reap_idle_containers(self, *, now_ms: int, idle_ttl_sec: int) -> int:
        ttl_ms = max(0, idle_ttl_sec) * 1000
        if ttl_ms <= 0:
            return 0
        removed = 0
        for node in self.nodes.values():
            removable: list[str] = []
            for container in node.containers.values():
                if container.state != CONTAINER_IDLE:
                    continue
                if container.last_idle_ms is None:
                    continue
                if now_ms - container.last_idle_ms >= ttl_ms:
                    removable.append(container.container_id)

            for container_id in removable:
                container = node.containers.pop(container_id, None)
                if container is None:
                    continue
                self._containers.pop(container_id, None)
                node.cpu_reserved_mcpu = max(0, node.cpu_reserved_mcpu - container.cpu_request_mcpu)
                node.mem_reserved_mb = max(0, node.mem_reserved_mb - container.memory_mb)
                removed += 1
        return removed

    def snapshot_node_metrics(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for node in sorted(self.nodes.values(), key=lambda n: n.node_id):
            rows.append(
                {
                    "node": node.node_id,
                    "active_containers": node.active_containers,
                    "busy_containers": node.busy_containers,
                    "max_containers": node.max_containers,
                    "utilization": node.utilization,
                    "cpu_total_mcpu": node.cpu_total_mcpu,
                    "cpu_reserved_mcpu": node.cpu_reserved_mcpu,
                    "cpu_utilization": node.cpu_utilization,
                    "mem_total_mb": node.mem_total_mb,
                    "mem_reserved_mb": node.mem_reserved_mb,
                    "mem_utilization": node.mem_utilization,
                    "cold_starting_containers": node.cold_starting_containers,
                    "running_containers": node.running_containers,
                    "idle_containers": node.idle_containers,
                }
            )
        return rows

    def host_candidates_with_slot(
        self,
        *,
        cpu_request_mcpu: int,
        memory_mb: int,
    ) -> list[PhysicalNode]:
        return [
            node
            for node in self.nodes.values()
            if node.has_slot and node.has_resources_for(cpu_request_mcpu=cpu_request_mcpu, memory_mb=memory_mb)
        ]
