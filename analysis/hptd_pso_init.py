from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PsoConfig:
    alpha_min: float = 0.1
    alpha_max: float = 0.5
    beta_min: float = 1.0
    beta_max: float = 2.0
    particles: int = 50
    max_iters: int = 100
    improve_threshold: float = 1e-3
    patience: int = 10
    seed: int = 20260318
    inertia: float = 0.7
    c1: float = 1.4
    c2: float = 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline PSO initializer for HPTD Hawkes parameters.")
    parser.add_argument("--output", default="data/processed/hptd_pso_params.json", help="Output JSON path.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path to update autoscaler.hptd_alpha/beta.",
    )
    parser.add_argument("--seed", type=int, default=20260318, help="Random seed.")
    return parser.parse_args()


def _objective(alpha: float, beta: float, freq: int, length: int = 100) -> float:
    # For a stable constant-frequency sequence, temperature should fluctuate minimally.
    mu = float(freq)
    event_times: list[float] = []
    temps: list[float] = []
    for t in range(length):
        cur = float(t)
        lam = mu
        for past in event_times:
            if past >= cur:
                continue
            lam += alpha * math.exp(-beta * (cur - past))
        lam = max(1e-9, lam)
        temps.append(math.log(lam))
        for _ in range(freq):
            event_times.append(cur)

    avg_t = sum(temps) / len(temps)
    return sum((x - avg_t) ** 2 for x in temps)


def _optimize_for_freq(freq: int, cfg: PsoConfig) -> tuple[float, float, float]:
    rng = random.Random(cfg.seed + freq)

    particles: list[list[float]] = []
    velocities: list[list[float]] = []
    pbest: list[list[float]] = []
    pbest_score: list[float] = []

    gbest = [cfg.alpha_min, cfg.beta_min]
    gbest_score = float("inf")

    for _ in range(cfg.particles):
        p = [
            rng.uniform(cfg.alpha_min, cfg.alpha_max),
            rng.uniform(cfg.beta_min, cfg.beta_max),
        ]
        v = [0.0, 0.0]
        s = _objective(p[0], p[1], freq)
        particles.append(p)
        velocities.append(v)
        pbest.append(p.copy())
        pbest_score.append(s)
        if s < gbest_score:
            gbest = p.copy()
            gbest_score = s

    no_improve_rounds = 0
    for _ in range(cfg.max_iters):
        improved = False
        for i in range(cfg.particles):
            r1 = rng.random()
            r2 = rng.random()
            velocities[i][0] = (
                cfg.inertia * velocities[i][0]
                + cfg.c1 * r1 * (pbest[i][0] - particles[i][0])
                + cfg.c2 * r2 * (gbest[0] - particles[i][0])
            )
            velocities[i][1] = (
                cfg.inertia * velocities[i][1]
                + cfg.c1 * r1 * (pbest[i][1] - particles[i][1])
                + cfg.c2 * r2 * (gbest[1] - particles[i][1])
            )
            particles[i][0] = min(cfg.alpha_max, max(cfg.alpha_min, particles[i][0] + velocities[i][0]))
            particles[i][1] = min(cfg.beta_max, max(cfg.beta_min, particles[i][1] + velocities[i][1]))

            score = _objective(particles[i][0], particles[i][1], freq)
            if score < pbest_score[i]:
                pbest[i] = particles[i].copy()
                pbest_score[i] = score
            if score + cfg.improve_threshold < gbest_score:
                gbest = particles[i].copy()
                gbest_score = score
                improved = True

        if improved:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= cfg.patience:
                break

    return gbest[0], gbest[1], gbest_score


def _write_params(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _update_config_yaml(path: Path, *, alpha: float, beta: float) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    autoscaler = raw.setdefault("autoscaler", {})
    autoscaler["hptd_alpha"] = float(alpha)
    autoscaler["hptd_beta"] = float(beta)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = PsoConfig(seed=int(args.seed))
    freqs = [5, 10, 15]

    per_freq = {}
    alpha_vals: list[float] = []
    beta_vals: list[float] = []
    for freq in freqs:
        alpha, beta, score = _optimize_for_freq(freq=freq, cfg=cfg)
        per_freq[str(freq)] = {"alpha": alpha, "beta": beta, "objective": score}
        alpha_vals.append(alpha)
        beta_vals.append(beta)

    alpha_avg = sum(alpha_vals) / len(alpha_vals)
    beta_avg = sum(beta_vals) / len(beta_vals)

    output_payload = {
        "method": "pso_once_offline",
        "search_space": {
            "alpha": [cfg.alpha_min, cfg.alpha_max],
            "beta": [cfg.beta_min, cfg.beta_max],
        },
        "pso": {
            "particles": cfg.particles,
            "max_iters": cfg.max_iters,
            "improve_threshold": cfg.improve_threshold,
            "patience": cfg.patience,
            "seed": cfg.seed,
        },
        "frequencies": freqs,
        "per_frequency_best": per_freq,
        "averaged_best": {
            "hptd_alpha": alpha_avg,
            "hptd_beta": beta_avg,
        },
    }

    out_path = Path(args.output)
    _write_params(out_path, output_payload)
    print(f"[hptd-pso] wrote: {out_path}")
    print(
        "[hptd-pso] averaged_best "
        f"alpha={alpha_avg:.6f} beta={beta_avg:.6f}"
    )

    if args.config:
        cfg_path = Path(args.config)
        _update_config_yaml(cfg_path, alpha=alpha_avg, beta=beta_avg)
        print(f"[hptd-pso] updated config: {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
