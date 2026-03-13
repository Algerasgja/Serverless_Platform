from __future__ import annotations

import random

from simulator.config import SchedulerConfig
from simulator.strategies.scheduler import LeastLoadScheduler
from simulator.strategies.scheduler.base import SchedulerStrategy


def build_scheduler(config: SchedulerConfig, rng: random.Random) -> SchedulerStrategy:
    if config.type == "least_load":
        return LeastLoadScheduler(rng)
    raise ValueError(f"unsupported scheduler type: {config.type}")
