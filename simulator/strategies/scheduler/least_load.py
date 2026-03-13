from __future__ import annotations

import random
from typing import Sequence

from simulator.strategies.scheduler.base import CandidateScore, SchedulerStrategy


class LeastLoadScheduler(SchedulerStrategy):
    name = "least_load"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select_instance(self, candidates: Sequence[CandidateScore]) -> str | None:
        if not candidates:
            return None
        min_score = min(c.score for c in candidates)
        lowest = [c for c in candidates if c.score == min_score]
        return self._rng.choice(lowest).instance_id

