from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CandidateScore:
    instance_id: str
    inflight: int
    max_concurrency: int
    score: float


class SchedulerStrategy(Protocol):
    name: str

    def select_instance(
        self,
        candidates: Sequence[CandidateScore],
    ) -> str | None:
        """Return the selected instance id or None when no candidate is available."""

