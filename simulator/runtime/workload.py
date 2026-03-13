from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from simulator.config import WorkloadConfig
from simulator.types import DagCorpus

SECONDS_OF_A_DAY = 24 * 3600.0


def poisson_sample(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    if lam > 50:
        # Fast approximation in high-rate regions.
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


def weighted_choice(weights: dict[str, float], rng: random.Random) -> str:
    total = sum(weights.values())
    if total <= 0:
        return next(iter(weights))
    marker = rng.random() * total
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += weight
        if marker <= cumulative:
            return key
    return next(reversed(weights))


def generate_arrivals(
    *,
    workload_cfg: WorkloadConfig,
    corpus: DagCorpus,
    duration_seconds: int,
    rng: random.Random,
    base_seed: int | None = None,
    profile_sink: dict[str, Any] | None = None,
) -> list[tuple[int, str]]:
    if workload_cfg.mode == "replay":
        arrivals, replay_profile = _generate_replay_arrivals(
            workload_cfg=workload_cfg,
            corpus=corpus,
            duration_seconds=duration_seconds,
            base_seed=0 if base_seed is None else int(base_seed),
        )
        if profile_sink is not None:
            profile_sink.clear()
            profile_sink.update(replay_profile)
        return arrivals

    arrivals: list[tuple[int, str]] = []
    um_weights = corpus.um_weights or {um: 1.0 for um in corpus.templates}
    for sec in range(max(1, duration_seconds)):
        qps = _qps_for_second(sec, workload_cfg, corpus)
        qps *= max(0.0, workload_cfg.rate_multiplier)
        count = poisson_sample(qps, rng)
        for _ in range(count):
            um = weighted_choice(um_weights, rng)
            offset_ms = rng.randint(0, 999)
            arrivals.append((sec * 1000 + offset_ms, um))
    arrivals.sort(key=lambda x: x[0])
    if profile_sink is not None:
        profile_sink.clear()
        profile_sink.update({"mode": "generative"})
    return arrivals


def _qps_for_second(sec: int, workload_cfg: WorkloadConfig, corpus: DagCorpus) -> float:
    # replay mode no longer uses minute qps series.
    if workload_cfg.mode == "replay":
        return 0.0
    qps = workload_cfg.baseline_rps
    burst_start = workload_cfg.burst_start_sec
    burst_end = burst_start + workload_cfg.burst_duration_sec
    if burst_start <= sec < burst_end:
        qps = workload_cfg.burst_rps
    return qps


def _generate_replay_arrivals(
    *,
    workload_cfg: WorkloadConfig,
    corpus: DagCorpus,
    duration_seconds: int,
    base_seed: int,
) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    invokes_cdf = _load_cdf(Path(workload_cfg.invokes_cdf_file))
    cvs_cdf = _load_cdf(Path(workload_cfg.cvs_cdf_file))

    rate_multiplier = float(workload_cfg.rate_multiplier)
    duration_ms = max(1, int(duration_seconds * 1000))
    min_iat_ms = max(0.001, float(workload_cfg.min_iat_ms))
    if rate_multiplier <= 0:
        return [], {
            "mode": "replay",
            "invokes_cdf_file": workload_cfg.invokes_cdf_file,
            "cvs_cdf_file": workload_cfg.cvs_cdf_file,
            "base_seed": base_seed,
            "realworld_seed_offset": workload_cfg.realworld_seed_offset,
            "rate_multiplier": rate_multiplier,
            "min_iat_ms": min_iat_ms,
            "reason": "non_positive_rate_multiplier",
            "per_um": {},
        }

    arrivals: list[tuple[int, str]] = []
    per_um: dict[str, dict[str, float | int]] = {}
    ums = sorted(corpus.templates.keys())
    for dag_idx, um in enumerate(ums):
        dag_seed = base_seed + int(workload_cfg.realworld_seed_offset) + dag_idx
        dag_rng = random.Random(dag_seed)

        avg_iat_sec = _sample_avg_iat_sec(invokes_cdf, dag_rng)
        cv = _sample_cv(cvs_cdf, dag_rng)
        base_iat_ms = max(min_iat_ms, avg_iat_sec * 1000.0)
        effective_iat_ms = max(min_iat_ms, base_iat_ms / rate_multiplier)
        mu, sigma = _lognormal_params_from_mean_cv(effective_iat_ms, cv)

        t_ms = dag_rng.uniform(0.0, effective_iat_ms)
        generated = 0
        while t_ms < duration_ms:
            arrivals.append((int(t_ms), um))
            generated += 1
            next_iat = dag_rng.lognormvariate(mu, sigma) if sigma > 0 else effective_iat_ms
            if next_iat < min_iat_ms:
                next_iat = min_iat_ms
            t_ms += next_iat

        per_um[um] = {
            "dag_index": dag_idx,
            "seed": dag_seed,
            "avg_iat_ms": base_iat_ms,
            "cv": cv,
            "effective_iat_ms": effective_iat_ms,
            "generated_requests": generated,
        }

    arrivals.sort(key=lambda x: x[0])
    profile = {
        "mode": "replay",
        "invokes_cdf_file": workload_cfg.invokes_cdf_file,
        "cvs_cdf_file": workload_cfg.cvs_cdf_file,
        "base_seed": base_seed,
        "realworld_seed_offset": workload_cfg.realworld_seed_offset,
        "rate_multiplier": rate_multiplier,
        "min_iat_ms": min_iat_ms,
        "dag_count": len(ums),
        "generated_requests": len(arrivals),
        "per_um": per_um,
    }
    return arrivals, profile


def _load_cdf(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"replay mode requires CDF file: {path}")

    entries: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                value = float(parts[0])
                cdf = float(parts[1])
            except ValueError:
                continue
            entries.append((value, cdf))

    if not entries:
        raise ValueError(f"CDF file has no valid rows: {path}")
    entries.sort(key=lambda x: x[1])
    return entries


def _sample_by_cdf(entries: list[tuple[float, float]], rng: random.Random) -> float:
    marker = rng.random()
    for value, cdf in entries:
        if marker <= cdf:
            return value
    return entries[-1][0]


def _compress_iat(iat: float) -> float:
    # Align with ServerlessBench Real-World-App-Emulation script behavior.
    offset = 0.0001
    lower_threshold = 1.0
    upper_threshold = 100.0
    max_value = 10.0

    if iat < lower_threshold:
        adjusted_iat = math.log1p(iat) / math.log(10.0)
    elif iat < upper_threshold:
        adjusted_iat = lower_threshold + (
            (max_value - lower_threshold) * (iat - lower_threshold) / (upper_threshold - lower_threshold)
        )
    else:
        adjusted_iat = max_value - math.exp(-math.log(iat - lower_threshold + 1.0))
    return max(adjusted_iat, offset)


def _sample_avg_iat_sec(invokes_cdf: list[tuple[float, float]], rng: random.Random) -> float:
    invoke_time = _sample_by_cdf(invokes_cdf, rng)
    raw_iat = invoke_time / SECONDS_OF_A_DAY
    return _compress_iat(raw_iat)


def _sample_cv(cvs_cdf: list[tuple[float, float]], rng: random.Random) -> float:
    return max(0.0, _sample_by_cdf(cvs_cdf, rng))


def _lognormal_params_from_mean_cv(mean: float, cv: float) -> tuple[float, float]:
    safe_mean = max(0.001, mean)
    safe_cv = max(0.0, cv)
    sigma2 = math.log(safe_cv * safe_cv + 1.0)
    sigma = math.sqrt(max(0.0, sigma2))
    mu = math.log(safe_mean) - (sigma2 / 2.0)
    return mu, sigma
