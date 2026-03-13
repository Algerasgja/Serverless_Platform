import random
import unittest

from simulator.strategies.scheduler import CandidateScore, LeastLoadScheduler


class LeastLoadSchedulerTests(unittest.TestCase):
    def test_selects_lowest_score_candidate(self) -> None:
        scheduler = LeastLoadScheduler(random.Random(42))
        candidates = [
            CandidateScore("a", inflight=7, max_concurrency=10, score=0.7),
            CandidateScore("b", inflight=1, max_concurrency=10, score=0.1),
            CandidateScore("c", inflight=4, max_concurrency=10, score=0.4),
        ]
        self.assertEqual("b", scheduler.select_instance(candidates))

    def test_tie_breaking_is_randomized(self) -> None:
        scheduler = LeastLoadScheduler(random.Random(7))
        candidates = [
            CandidateScore("a", inflight=2, max_concurrency=10, score=0.2),
            CandidateScore("b", inflight=2, max_concurrency=10, score=0.2),
        ]
        seen = set()
        for _ in range(20):
            seen.add(scheduler.select_instance(candidates))
        self.assertEqual({"a", "b"}, seen)


if __name__ == "__main__":
    unittest.main()

